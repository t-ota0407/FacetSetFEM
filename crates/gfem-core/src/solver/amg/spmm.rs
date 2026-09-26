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
use crate::solver::amg::rowwise;

/// 1 行ぶんの作業領域（sparse accumulator）。
///
/// 列ごとに「この列を今の行で既に触ったか」を世代番号で覚える。行ごとに配列全体を
/// クリアすると O(列数) かかるが、世代番号なら O(1) で済む。世代番号と値は 1 つの配列に
/// 並べて持つ（別々の配列だと 1 回の足し込みで離れた 2 か所を触ることになる）。
struct Accumulator {
    /// 列ごとの `(世代番号, 値)`
    slots: Vec<(u64, f64)>,
    generation: u64,
    touched: Vec<u32>,
}

impl Accumulator {
    fn new(n_cols: usize) -> Self {
        Self {
            slots: vec![(0, 0.0); n_cols],
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
        let slot = &mut self.slots[col as usize];
        if slot.0 != self.generation {
            *slot = (self.generation, v);
            self.touched.push(col);
        } else {
            slot.1 += v;
        }
    }

    /// 列 `col` の値（今の行で触った列に限る）。
    #[inline]
    fn value(&self, col: u32) -> f64 {
        self.slots[col as usize].1
    }

    /// 触った列を昇順に並べる。
    fn sorted(&mut self) -> &[u32] {
        self.touched.sort_unstable();
        &self.touched
    }
}

/// `C = A B`。`A` は m×k、`B` は k×n。
///
/// 行内の列番号は昇順、加算順序はモジュール冒頭の規則に従う。行は [`rowwise`] の塊ごとに
/// 並列に処理し、作業配列はタスクごとに 1 つ作って塊の間で使い回す。
pub fn spmm(a: &CsrMatrix, b: &CsrMatrix) -> CsrMatrix {
    debug_assert_eq!(a.n_cols(), b.n_rows());
    let (m, n) = (a.n_rows(), b.n_cols());

    // --- 記号パス: 行ごとの非ゼロ数 -------------------------------------
    let mut counts = vec![0usize; m];
    rowwise::for_each_row_init(
        &mut counts,
        || Accumulator::new(n),
        |acc, i, out| {
            acc.start_row();
            for_each_product(a, b, i, |col, _| acc.add(col, 0.0));
            *out = acc.touched.len();
        },
    );
    let row_ptr = rowwise::row_pointers(&counts);
    let nnz = row_ptr[m];

    // --- 数値パス -------------------------------------------------------
    // ゼロで確保するので、ページは書き込むときに（並列に）初めて触る
    let mut col_idx = vec![0u32; nnz];
    let mut values = vec![0.0f64; nnz];
    rowwise::fill_rows_init(
        &row_ptr,
        &mut col_idx,
        &mut values,
        || Accumulator::new(n),
        |acc, i, cols, vals| {
            acc.start_row();
            for_each_product(a, b, i, |col, v| acc.add(col, v));
            for (slot, &c) in cols.iter_mut().zip(acc.sorted()) {
                *slot = c;
            }
            for (slot, &c) in vals.iter_mut().zip(cols.iter()) {
                *slot = acc.value(c);
            }
        },
    );
    CsrMatrix::from_parts(m, n, row_ptr, col_idx, values)
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
        // 並列に切り替わる大きさ（rowwise::PARALLEL_MIN_ROWS）を超える行列で、
        // 1 行ずつ逐次に作ったものと CSR 全体を突き合わせる
        let n = rowwise::PARALLEL_MIN_ROWS * 3 + 17;
        let a = banded(n, 5);
        let big = spmm(&a, &a);

        let mut row_ptr = vec![0usize];
        let mut col_idx: Vec<u32> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        let mut acc = Accumulator::new(n);
        for i in 0..n {
            acc.start_row();
            for_each_product(&a, &a, i, |c, v| acc.add(c, v));
            let cols = acc.sorted().to_vec();
            values.extend(cols.iter().map(|&c| acc.value(c)));
            col_idx.extend(cols);
            row_ptr.push(col_idx.len());
        }
        assert_eq!(big.row_ptr(), &row_ptr[..]);
        assert_eq!(big.col_idx(), &col_idx[..]);
        assert_eq!(big.values(), &values[..]);

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
