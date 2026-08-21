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

/// 対称正半定値の CSR 行列。
#[derive(Debug, Clone, Default)]
pub struct CsrMatrix {
    n: usize,
    /// 長さ `n + 1`
    row_ptr: Vec<usize>,
    /// 行ごとに昇順
    col_idx: Vec<u32>,
    values: Vec<f64>,
}

impl CsrMatrix {
    /// 構造（`row_ptr` / `col_idx`）から、値を 0 にした行列を作る。
    pub fn from_structure(n: usize, row_ptr: Vec<usize>, col_idx: Vec<u32>) -> Self {
        debug_assert_eq!(row_ptr.len(), n + 1);
        debug_assert_eq!(row_ptr[n], col_idx.len());
        let values = vec![0.0; col_idx.len()];
        Self {
            n,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// 行数（＝節点数）。
    pub fn n_rows(&self) -> usize {
        self.n
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
        (0..self.n)
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
        debug_assert_eq!(x.len(), self.n);
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

/// 行列のグラフの連結成分を数える（幅優先探索、O(nnz)）。
///
/// SPEC §2.2 は「K の零空間は定数ベクトル span{1}」を前提に射影 CG を組む。
/// これが成り立つのは**メッシュが連結なとき**だけで、非連結なら零空間の次元が
/// 成分数だけ増え、射影 CG は正しく収束しない。だから解く前に検査する。
///
/// 戻り値は `(成分数, 節点ごとの成分番号)`。成分番号は「その成分に属する最小の節点番号」
/// の順に 0, 1, 2, ... と振るので、入力が同じなら常に同じ番号になる。
pub fn connected_components(matrix: &CsrMatrix) -> (usize, Vec<u32>) {
    const UNVISITED: u32 = u32::MAX;
    let n = matrix.n_rows();
    let mut component = vec![UNVISITED; n];
    let mut count = 0u32;
    let mut stack: Vec<u32> = Vec::new();
    for start in 0..n {
        if component[start] != UNVISITED {
            continue;
        }
        component[start] = count;
        stack.push(start as u32);
        while let Some(node) = stack.pop() {
            let (cols, _) = matrix.row(node as usize);
            for &c in cols {
                if component[c as usize] == UNVISITED {
                    component[c as usize] = count;
                    stack.push(c);
                }
            }
        }
        count += 1;
    }
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
