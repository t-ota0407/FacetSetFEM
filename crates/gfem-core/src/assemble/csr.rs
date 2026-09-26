//! CSR スパース行列（SPEC §4.2）。
//!
//! 行列フリーではなく CSR を採用する理由は SPEC §4.2 のとおり:
//! Tet4 では 1 節点に平均約 6 要素が接するため要素剛性を保存すると CSR の 2.5 倍のメモリを食い、
//! かつ **CSR は加算順序が構築時に固定される**ので CPU 並列でも GPU でもビット単位で同じ結果になる。
//!
//! # 再現性の不変条件（CLAUDE.md 設計原則 1）
//!
//! - 各行の `col_idx` は**昇順**。SpMV は行内を列番号順に足す
//! - 行内の総和は逐次。行は高々 20 項程度なので、この順序を全バックエンドで揃えれば
//!   （CPU 逐次 / CPU 並列 / GPU の 1 行 1 スレッド）ビット単位で一致する
//! - アセンブリは**要素の通し番号順**に加算する。行ごとに並列化しても、
//!   その行に触る要素を通し番号順にたどれば同じ加算順序になる

use crate::error::{GfemError, Result};

/// CSR 行列。剛性行列は対称正半定値の正方行列だが、AMG の補間演算子
/// （SPEC §4.5）は矩形なので、行数と列数を別に持つ。
#[derive(Debug, Clone, Default)]
pub struct CsrMatrix {
    n: usize,
    /// 列数。正方行列では `n` と等しい
    n_cols: usize,
    /// 長さ `n + 1`
    row_ptr: Vec<usize>,
    /// 行ごとに昇順
    col_idx: Vec<u32>,
    values: Vec<f64>,
}

impl CsrMatrix {
    /// 構造（`row_ptr` / `col_idx`）から、値を 0 にした正方行列を作る。
    pub fn from_structure(n: usize, row_ptr: Vec<usize>, col_idx: Vec<u32>) -> Self {
        Self::from_structure_rect(n, n, row_ptr, col_idx)
    }

    /// 矩形版。`n_cols` は補間演算子 P（細 × 粗）などで行数と異なる。
    pub fn from_structure_rect(
        n_rows: usize,
        n_cols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<u32>,
    ) -> Self {
        debug_assert_eq!(row_ptr.len(), n_rows + 1);
        debug_assert_eq!(row_ptr[n_rows], col_idx.len());
        let values = vec![0.0; col_idx.len()];
        Self {
            n: n_rows,
            n_cols,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// 行ポインタ・列番号・値から作る（値を複製しない）。行内の列番号は昇順であること。
    pub fn from_parts(
        n_rows: usize,
        n_cols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<u32>,
        values: Vec<f64>,
    ) -> Self {
        debug_assert_eq!(row_ptr.len(), n_rows + 1);
        debug_assert_eq!(row_ptr[n_rows], col_idx.len());
        debug_assert_eq!(col_idx.len(), values.len());
        Self {
            n: n_rows,
            n_cols,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// 行数（＝節点数）。
    pub fn n_rows(&self) -> usize {
        self.n
    }

    /// 列数。正方行列では [`n_rows`](Self::n_rows) と等しい。
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// 非ゼロ要素数。
    pub fn nnz(&self) -> usize {
        self.col_idx.len()
    }

    /// 行の開始位置配列。
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// 列番号配列。
    pub fn col_idx(&self) -> &[u32] {
        &self.col_idx
    }

    /// 値配列。
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// 値配列（可変）。
    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// 構造と値を同時に借りる。
    ///
    /// 並列アセンブリで「構造を読みながら値を書く」ために要る。
    /// 別々のフィールドなので同時に借りても安全。
    pub fn parts_mut(&mut self) -> (&[usize], &[u32], &mut [f64]) {
        (&self.row_ptr, &self.col_idx, &mut self.values)
    }

    /// 行 `i` の列番号と値。
    pub fn row(&self, i: usize) -> (&[u32], &[f64]) {
        let (s, e) = (self.row_ptr[i], self.row_ptr[i + 1]);
        (&self.col_idx[s..e], &self.values[s..e])
    }

    /// 成分 `(i, j)` の値配列内での位置。構造に無ければ `None`。
    pub fn position(&self, i: usize, j: u32) -> Option<usize> {
        let (s, e) = (self.row_ptr[i], self.row_ptr[i + 1]);
        self.col_idx[s..e].binary_search(&j).ok().map(|k| s + k)
    }

    /// 対角成分。
    pub fn diagonal(&self) -> Vec<f64> {
        use rayon::prelude::*;
        // 行ごとに独立なので並列でよい（AMG の各レベルで何度も呼ぶ。1000 万行で逐次 0.3 秒）
        (0..self.n)
            .into_par_iter()
            .with_min_len(4_096)
            .map(|i| {
                self.position(i, i as u32)
                    .map(|k| self.values[k])
                    .unwrap_or(0.0)
            })
            .collect()
    }

    /// `y = A x`。
    ///
    /// 行内は列番号の昇順に逐次加算する。この順序が全バックエンド共通の約束
    /// （モジュール冒頭の不変条件を参照）。
    pub fn spmv(&self, x: &[f64], y: &mut [f64]) {
        debug_assert_eq!(x.len(), self.n_cols);
        debug_assert_eq!(y.len(), self.n);
        for (i, yi) in y.iter_mut().enumerate() {
            let (s, e) = (self.row_ptr[i], self.row_ptr[i + 1]);
            let mut acc = 0.0;
            for k in s..e {
                acc += self.values[k] * x[self.col_idx[k] as usize];
            }
            *yi = acc;
        }
    }

    /// 転置 `Aᵀ`（SPEC §4.5。AMG の制限演算子 R = Pᵀ を明示的に持つために使う）。
    ///
    /// AMG では制限 `R x` が要る。`Pᵀ` を「列方向に足し込む」形で計算すると
    /// GPU では書き込みが衝突して atomicAdd が要り、加算順序が実行ごとに変わってしまう
    /// （CLAUDE.md 設計原則 1 に反する）。だから**転置を CSR として明示的に持ち**、
    /// 制限も普通の行ごと SpMV で行う。メモリは増えるが再現性が守れる。
    ///
    /// 計数ソートなので O(nnz)。各行の列番号は昇順に並ぶ。
    pub fn transpose(&self) -> CsrMatrix {
        let mut counts = vec![0usize; self.n_cols + 1];
        for &c in &self.col_idx {
            counts[c as usize + 1] += 1;
        }
        for i in 0..self.n_cols {
            counts[i + 1] += counts[i];
        }
        let row_ptr = counts.clone();
        let mut col_idx = vec![0u32; self.col_idx.len()];
        let mut values = vec![0.0f64; self.values.len()];
        // 元の行を昇順にたどるので、転置後の各行内も行番号（＝元の列）の昇順に並ぶ
        let mut cursor = counts;
        for i in 0..self.n {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                let c = self.col_idx[k] as usize;
                let pos = cursor[c];
                cursor[c] += 1;
                col_idx[pos] = i as u32;
                values[pos] = self.values[k];
            }
        }
        CsrMatrix {
            n: self.n_cols,
            n_cols: self.n,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// 行列が対称かどうか（テスト用。O(nnz log n)）。
    pub fn is_symmetric(&self, tolerance: f64) -> bool {
        for i in 0..self.n {
            let (cols, vals) = self.row(i);
            for (&j, &v) in cols.iter().zip(vals) {
                match self.position(j as usize, i as u32) {
                    Some(k) if (self.values[k] - v).abs() <= tolerance => {}
                    _ => return false,
                }
            }
        }
        true
    }
}

/// 行列のグラフの連結成分を数える（並列な union-find、ほぼ O(nnz)）。
///
/// SPEC §2.2 は「K の零空間は定数ベクトル span{1}」を前提に射影 CG を組む。
/// これが成り立つのは**メッシュが連結なとき**だけで、非連結なら零空間の次元が
/// 成分数だけ増え、射影 CG は正しく収束しない。だから解く前に検査する。
///
/// 戻り値は `(成分数, 節点ごとの成分番号)`。成分番号は「その成分に属する最小の節点番号」
/// の順に 0, 1, 2, ... と振るので、入力が同じなら常に同じ番号になる。
///
/// 以前は逐次の探索で、1000 万節点で 0.9 秒かかっていた。いまは非ゼロ（辺）ごとに並列に
/// 根どうしをつなぐ。**大きい根を必ず小さい根の下につなぐ**（親は常に自分より小さい）ので、
/// つなぐ順序がどうであれ、各成分の根はその成分の最小の節点になる。成分の分け方は
/// グラフだけで決まり、番号も根の順に振るので、結果は処理の順序に依らない。
pub fn connected_components(matrix: &CsrMatrix) -> (usize, Vec<u32>) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    let n = matrix.n_rows();
    let parent: Vec<AtomicU32> = (0..n as u32).into_par_iter().map(AtomicU32::new).collect();

    // 根を探す。途中で親を祖父に付け替えて経路を縮める（祖父は親より小さいので、
    // 「親は自分より小さい」は崩れない）
    let find = |mut x: u32| -> u32 {
        loop {
            let p = parent[x as usize].load(Ordering::Acquire);
            if p == x {
                return x;
            }
            let g = parent[p as usize].load(Ordering::Acquire);
            if g != p {
                let _ =
                    parent[x as usize].compare_exchange(p, g, Ordering::AcqRel, Ordering::Acquire);
            }
            x = g;
        }
    };

    (0..n).into_par_iter().with_min_len(1_024).for_each(|i| {
        let (cols, _) = matrix.row(i);
        for &j in cols {
            if j as usize == i {
                continue;
            }
            let (mut a, mut b) = (i as u32, j);
            loop {
                a = find(a);
                b = find(b);
                if a == b {
                    break;
                }
                let (high, low) = if a > b { (a, b) } else { (b, a) };
                // high がまだ根なら low の下につなぐ。誰かに先を越されたら探し直す
                if parent[high as usize]
                    .compare_exchange(high, low, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    break;
                }
            }
        }
    });

    let root: Vec<u32> = (0..n as u32).into_par_iter().map(find).collect();
    // 成分番号は根（＝成分の最小の節点）の順
    let mut id = vec![0u32; n];
    let mut count = 0u32;
    for (i, &r) in root.iter().enumerate() {
        if r as usize == i {
            id[i] = count;
            count += 1;
        }
    }
    let component: Vec<u32> = root.par_iter().map(|&r| id[r as usize]).collect();
    (count as usize, component)
}

/// メッシュが連結でなければ、成分の大きさを添えてエラーにする。
pub fn require_connected(matrix: &CsrMatrix) -> Result<()> {
    let (count, component) = connected_components(matrix);
    if count <= 1 {
        return Ok(());
    }
    let mut sizes = vec![0usize; count];
    for &c in &component {
        sizes[c as usize] += 1;
    }
    let mut listed: Vec<String> = sizes
        .iter()
        .enumerate()
        .map(|(i, s)| format!("成分{}: {s} 節点", i + 1))
        .collect();
    listed.truncate(10);
    Err(GfemError::model(format!(
        "メッシュが {count} 個の連結成分に分かれています（{}）",
        listed.join(", ")
    ))
    .with_hint(
        "全 Neumann 境界の電流場では、連結成分ごとに電位の基準が独立に決まってしまい解が\
         一意になりません（SPEC §2.2）。要素の抜けや節点の重複でモデルが分断されていないか\
         確認してください",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3×3 の三重対角行列。
    fn tridiagonal() -> CsrMatrix {
        let row_ptr = vec![0, 2, 5, 7];
        let col_idx = vec![0, 1, 0, 1, 2, 1, 2];
        let mut m = CsrMatrix::from_structure(3, row_ptr, col_idx);
        m.values_mut()
            .copy_from_slice(&[2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0]);
        m
    }

    /// 以前の逐次の探索（最小の未訪問節点から順に辿り、見つけた順に番号を振る）。
    fn serial_components(m: &CsrMatrix) -> (usize, Vec<u32>) {
        let n = m.n_rows();
        let mut component = vec![u32::MAX; n];
        let mut count = 0u32;
        for start in 0..n {
            if component[start] != u32::MAX {
                continue;
            }
            component[start] = count;
            let mut stack = vec![start];
            while let Some(i) = stack.pop() {
                for &j in m.row(i).0 {
                    if component[j as usize] == u32::MAX {
                        component[j as usize] = count;
                        stack.push(j as usize);
                    }
                }
            }
            count += 1;
        }
        (count as usize, component)
    }

    #[test]
    fn parallel_components_match_the_serial_search() {
        // 節点番号が入り組んだ（互いに挟み合う）連結成分をいくつも持つ大きなグラフ。
        // 並列に根をつないでも、成分の分け方と番号（最小の節点の順）が逐次と同じ
        let n = 60_000usize;
        let component_of = |i: usize| (i * 7919 % 13) as u32; // 13 個の成分に散らばる
        let mut members: Vec<Vec<u32>> = vec![Vec::new(); 13];
        for i in 0..n {
            members[component_of(i) as usize].push(i as u32);
        }
        // 各成分を、少し先の仲間と結ぶ鎖と飛び石にする（対称な構造）
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); n];
        for list in &members {
            for (k, &i) in list.iter().enumerate() {
                for step in [1usize, 37] {
                    if let Some(&j) = list.get(k + step) {
                        rows[i as usize].push(j);
                        rows[j as usize].push(i);
                    }
                }
            }
        }
        let mut row_ptr = vec![0usize];
        let mut col_idx = Vec::new();
        for (i, mut row) in rows.into_iter().enumerate() {
            row.push(i as u32);
            row.sort_unstable();
            row.dedup();
            col_idx.extend(row);
            row_ptr.push(col_idx.len());
        }
        let m = CsrMatrix::from_structure(n, row_ptr, col_idx);
        let expected = serial_components(&m);
        assert_eq!(expected.0, 13);
        for _ in 0..3 {
            assert_eq!(connected_components(&m), expected);
        }
        // 孤立した節点だけの行列（どの節点も自分だけの成分）
        let lonely = CsrMatrix::from_structure(5, (0..=5).collect(), (0..5).collect());
        assert_eq!(connected_components(&lonely), (5, vec![0, 1, 2, 3, 4]));
    }

    #[test]
    fn spmv_matches_dense() {
        let m = tridiagonal();
        let x = [1.0, 2.0, 3.0];
        let mut y = [0.0; 3];
        m.spmv(&x, &mut y);
        assert_eq!(y, [0.0, 0.0, 4.0]);
    }

    #[test]
    fn diagonal_and_position() {
        let m = tridiagonal();
        assert_eq!(m.diagonal(), vec![2.0, 2.0, 2.0]);
        assert_eq!(m.position(1, 2), Some(4));
        assert_eq!(m.position(0, 2), None);
        assert!(m.is_symmetric(0.0));
    }

    #[test]
    fn detects_a_single_component() {
        let (count, comp) = connected_components(&tridiagonal());
        assert_eq!(count, 1);
        assert_eq!(comp, vec![0, 0, 0]);
        assert!(require_connected(&tridiagonal()).is_ok());
    }

    #[test]
    fn detects_disconnected_graph() {
        // 節点 0-1 と 2-3 が別々に繋がっている
        let row_ptr = vec![0, 2, 4, 6, 8];
        let col_idx = vec![0, 1, 0, 1, 2, 3, 2, 3];
        let m = CsrMatrix::from_structure(4, row_ptr, col_idx);
        let (count, comp) = connected_components(&m);
        assert_eq!(count, 2);
        assert_eq!(comp, vec![0, 0, 1, 1]);
        let e = require_connected(&m).unwrap_err();
        assert!(e.to_string().contains("2 個の連結成分"), "{e}");
        assert!(e.to_string().contains("SPEC §2.2"), "{e}");
    }
}
