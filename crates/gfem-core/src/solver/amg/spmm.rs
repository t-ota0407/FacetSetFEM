//! 決定的なスパース行列積 `C = A B`（SPEC §4.5）。
//!
//! AMG の粗レベル行列は Galerkin 積 `A_c = R A P` で作る。積の各成分は
//! 複数の項の和なので、**足す順序を固定しないと結果が実行ごとに変わる**
//! （CLAUDE.md 設計原則 1）。ここでは
//!
//! 1. 出力の行 `i` は、`A` の行 `i` を**列番号の昇順**にたどり、
//!    各項について `B` の対応する行を**列番号の昇順**にたどって足し込む
//! 2. 出力の行内は列番号の昇順に並べる
//!
//! と決めてある。行同士は独立なので、**行を rayon で並列に配っても
//! 加算順序は 1 つも変わらない**。逐次と並列がビット単位で一致することは
//! テストで確認している。
//!
//! 実装は密な作業配列（sparse accumulator）による古典的な 2 パス:
//! 記号パス（非ゼロ数を数える）→ 数値パス（値を入れる）。

use crate::assemble::csr::CsrMatrix;

/// 行を rayon に配りはじめる行数。これより小さいとタスク投入のほうが高くつく。
const PARALLEL_MIN_ROWS: usize = 4_096;

/// 1 行ぶんの作業領域（sparse accumulator）。
///
/// `mark` は「この列を今の行で既に触ったか」を世代番号で覚える。
/// 行ごとに配列全体をクリアすると O(列数) かかるが、世代番号なら O(1) で済む。
struct Accumulator {
    value: Vec<f64>,
    mark: Vec<u64>,
    generation: u64,
    touched: Vec<u32>,
}

impl Accumulator {
    fn new(n_cols: usize) -> Self {
        Self {
            value: vec![0.0; n_cols],
            mark: vec![0; n_cols],
            generation: 0,
            touched: Vec::new(),
        }
    }

    /// 新しい行を始める。
    fn start_row(&mut self) {
        self.generation += 1;
        self.touched.clear();
    }

    /// `value[col] += v`。初めて触る列なら記録する。
    #[inline]
    fn add(&mut self, col: u32, v: f64) {
        let c = col as usize;
        if self.mark[c] != self.generation {
            self.mark[c] = self.generation;
            self.value[c] = v;
            self.touched.push(col);
        } else {
            self.value[c] += v;
        }
    }

    /// 触った列を昇順に並べる。
    fn sorted(&mut self) -> &[u32] {
        self.touched.sort_unstable();
        &self.touched
    }
}

/// `C = A B`。`A` は m×k、`B` は k×n。
///
/// 行内の列番号は昇順、加算順序はモジュール冒頭の規則に従う。
pub fn spmm(a: &CsrMatrix, b: &CsrMatrix) -> CsrMatrix {
    debug_assert_eq!(a.n_cols(), b.n_rows());
    let (m, n) = (a.n_rows(), b.n_cols());

    // --- 記号パス: 行ごとの非ゼロ数 -------------------------------------
    let mut counts = vec![0usize; m];
    apply_rows(m, n, &mut counts, |acc, i, out| {
        acc.start_row();
        for_each_product(a, b, i, |col, _| acc.add(col, 0.0));
        *out = acc.touched.len();
    });

    let mut row_ptr = vec![0usize; m + 1];
    for i in 0..m {
        row_ptr[i + 1] = row_ptr[i] + counts[i];
    }
    let nnz = row_ptr[m];

    // --- 数値パス -------------------------------------------------------
    let mut col_idx = vec![0u32; nnz];
    let mut values = vec![0.0f64; nnz];
    {
        let mut rows: Vec<(&mut [u32], &mut [f64])> = Vec::with_capacity(m);
        let (mut cols_rest, mut vals_rest) = (&mut col_idx[..], &mut values[..]);
        for i in 0..m {
            let take = row_ptr[i + 1] - row_ptr[i];
            let (ch, ct) = cols_rest.split_at_mut(take);
            let (vh, vt) = vals_rest.split_at_mut(take);
            rows.push((ch, vh));
            cols_rest = ct;
            vals_rest = vt;
        }
        apply_rows(m, n, &mut rows, |acc, i, out| {
            acc.start_row();
            for_each_product(a, b, i, |col, v| acc.add(col, v));
            let (cols, vals) = out;
            for (slot, &c) in cols.iter_mut().zip(acc.sorted()) {
                *slot = c;
            }
            for (slot, &c) in vals.iter_mut().zip(cols.iter()) {
                *slot = acc.value[c as usize];
            }
        });
    }

    let mut c = CsrMatrix::from_structure_rect(m, n, row_ptr, col_idx);
    c.values_mut().copy_from_slice(&values);
    c
}

/// 行 `i` の全ての積の項 `(列, 値)` を、モジュール冒頭で決めた順序で渡す。
#[inline]
fn for_each_product(a: &CsrMatrix, b: &CsrMatrix, i: usize, mut f: impl FnMut(u32, f64)) {
    let (a_cols, a_vals) = a.row(i);
    for (&j, &aij) in a_cols.iter().zip(a_vals) {
        let (b_cols, b_vals) = b.row(j as usize);
        for (&c, &bjc) in b_cols.iter().zip(b_vals) {
            f(c, aij * bjc);
        }
    }
}

/// 行ごとの処理を、逐次か rayon の並列で走らせる。
///
/// **どちらで走らせても結果はビット単位で同じ**——行同士は独立で、
/// 行内の加算順序は担当スレッドに依存しないため。
fn apply_rows<T: Send>(
    m: usize,
    n_cols: usize,
    out: &mut [T],
    body: impl Fn(&mut Accumulator, usize, &mut T) + Sync + Send,
) {
    if m < PARALLEL_MIN_ROWS {
        let mut acc = Accumulator::new(n_cols);
        for (i, slot) in out.iter_mut().enumerate() {
            body(&mut acc, i, slot);
        }
        return;
    }
    use rayon::prelude::*;
    // 作業配列は列数ぶんの大きさがあるので、いくつ作られるかを見積もれないと
    // メモリが読めない。行をスレッド数ちょうどの塊に切り、塊 1 つにつき 1 個だけ作る
    let threads = rayon::current_num_threads().max(1);
    let block = m.div_ceil(threads).max(1);
    out.par_chunks_mut(block)
        .enumerate()
        .for_each(|(b, chunk)| {
            let mut acc = Accumulator::new(n_cols);
            let base = b * block;
            for (k, slot) in chunk.iter_mut().enumerate() {
                body(&mut acc, base + k, slot);
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 密行列から CSR を作る（明示的なゼロは落とす）。
    fn from_dense(m: usize, n: usize, d: &[f64]) -> CsrMatrix {
        let mut row_ptr = vec![0usize];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        for i in 0..m {
            for j in 0..n {
                if d[i * n + j] != 0.0 {
                    col_idx.push(j as u32);
                    values.push(d[i * n + j]);
                }
            }
            row_ptr.push(col_idx.len());
        }
        let mut c = CsrMatrix::from_structure_rect(m, n, row_ptr, col_idx);
        c.values_mut().copy_from_slice(&values);
        c
    }

    fn to_dense(a: &CsrMatrix) -> Vec<f64> {
        let mut d = vec![0.0; a.n_rows() * a.n_cols()];
        for i in 0..a.n_rows() {
            let (cols, vals) = a.row(i);
            for (&c, &v) in cols.iter().zip(vals) {
                d[i * a.n_cols() + c as usize] = v;
            }
        }
        d
    }

    #[test]
    fn matches_a_dense_product() {
        // 2×3 と 3×2
        let a = from_dense(2, 3, &[1.0, 0.0, 2.0, 0.0, 3.0, 4.0]);
        let b = from_dense(3, 2, &[5.0, 0.0, 0.0, 6.0, 7.0, 8.0]);
        let c = spmm(&a, &b);
        assert_eq!(c.n_rows(), 2);
        assert_eq!(c.n_cols(), 2);
        // [1 0 2; 0 3 4] * [5 0; 0 6; 7 8] = [19 16; 28 50]
        assert_eq!(to_dense(&c), vec![19.0, 16.0, 28.0, 50.0]);
    }

    #[test]
    fn column_indices_are_ascending() {
        let a = from_dense(3, 3, &[1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let c = spmm(&a, &a);
        for i in 0..c.n_rows() {
            let (cols, _) = c.row(i);
            assert!(cols.windows(2).all(|w| w[0] < w[1]), "行 {i}: {cols:?}");
        }
    }

    #[test]
    fn parallel_and_serial_agree_bitwise() {
        // CLAUDE.md 設計原則 1: 行を並列に配っても 1 ビットも変わらない。
        // PARALLEL_MIN_ROWS をまたぐ大きさで、同じ行列を「小さく切って」比べる
        let n = PARALLEL_MIN_ROWS * 2;
        let a = banded(n, 5);
        let big = spmm(&a, &a);

        // 逐次経路と並列経路を直接突き合わせる（行の担当が変わっても値は同じはず）
        let row_value = |acc: &mut Accumulator, i: usize| -> Vec<f64> {
            acc.start_row();
            for_each_product(&a, &a, i, |c, v| acc.add(c, v));
            let cols = acc.sorted().to_vec();
            cols.iter().map(|&c| acc.value[c as usize]).collect()
        };
        let mut serial = vec![Vec::new(); n];
        {
            let mut acc = Accumulator::new(n);
            for (i, slot) in serial.iter_mut().enumerate() {
                *slot = row_value(&mut acc, i);
            }
        }
        let mut parallel = vec![Vec::new(); n];
        apply_rows(n, n, &mut parallel, |acc, i, slot| {
            *slot = row_value(acc, i)
        });
        assert_eq!(serial, parallel);

        // 積そのものも安定していること
        for _ in 0..3 {
            let again = spmm(&a, &a);
            assert_eq!(again.values(), big.values());
            assert_eq!(again.col_idx(), big.col_idx());
        }
    }

    /// 幅 `w` の帯行列。
    fn banded(n: usize, w: usize) -> CsrMatrix {
        let mut row_ptr = vec![0usize];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            let lo = i.saturating_sub(w / 2);
            let hi = (i + w / 2 + 1).min(n);
            for j in lo..hi {
                col_idx.push(j as u32);
                values.push(if i == j {
                    2.0
                } else {
                    -1.0 / (1 + i + j) as f64
                });
            }
            row_ptr.push(col_idx.len());
        }
        let mut m = CsrMatrix::from_structure(n, row_ptr, col_idx);
        m.values_mut().copy_from_slice(&values);
        m
    }

    #[test]
    fn empty_rows_are_handled() {
        let a = from_dense(3, 3, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let c = spmm(&a, &a);
        assert_eq!(c.row(1).0.len(), 0);
        assert_eq!(
            to_dense(&c),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        );
    }
}
