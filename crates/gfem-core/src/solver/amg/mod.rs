//! 代数マルチグリッド前処理（平滑化凝集、SPEC §4.5 / §8 M6）。
//!
//! Jacobi 前処理の CG は、メッシュを細かくするほど反復回数が増える
//! （条件数 κ ∝ (σmax/σmin)(L/h)² に対して反復数 ∝ √κ）。実測でも 15 万節点 517 回に対し
//! 1000 万節点で 1423 回だった。AMG はこの依存を断ち切るための前処理で、
//! 誤差のうち平滑化で消えない「なだらかな成分」を粗い格子に移して直接解く。
//!
//! # なぜ平滑化凝集（smoothed aggregation）か
//!
//! 古典 AMG（Ruge-Stüben）の C/F 分割より、この問題には凝集型が合っている。
//!
//! - 全 Neumann の電流場では**近零空間がちょうど定数ベクトル**（SPEC §2.2）。
//!   凝集型は「近零空間ベクトルを与えると、それを厳密に補間する P を作る」
//!   のが定義そのもので、この問題に誂え向き
//! - 粗化と補間が**疎行列の積だけ**で書けるので、GPU でも同じ経路に載る
//! - C/F 分割は独立集合の選び方に乱数を使う実装が多く、再現性を保つのが難しい
//!
//! # 再現性（CLAUDE.md 設計原則 1）
//!
//! **階層の構築は完全にホスト側（CPU）で行い、行列だけを入力にする。**
//! バックエンドは一切関与しない。だから CPU 逐次・CPU 並列・GPU のどれで
//! 解いても**同じ階層**ができる。V サイクルの実行は既存の
//! [`LinearBackend`](crate::solver::LinearBackend) の演算しか使わないので、
//! こちらも自動的にビット単位で一致する。
//!
//! 構築の中で順序が問題になるのは 2 箇所だけで、どちらも潰してある。
//!
//! - 凝集（[`aggregate`]）: 節点番号の昇順に逐次で貪欲。並列化しない
//! - 疎行列積（[`spmm`]）: 行ごとに独立で、行内の加算順序は固定。並列化しても変わらない
//!
//! # 構成
//!
//! ```text
//! setup(A)  →  Hierarchy { levels: [Level; L], coarse: CoarseSolver }
//!                Level { a, inv_diag, omega, transfer: Some(P, R) }
//!                                                        ↑ 最粗レベルだけ None
//! ```

pub mod aggregate;
pub mod coarse;
pub mod cycle;
pub mod prolongation;
pub mod rowwise;
pub mod spmm;

use crate::assemble::csr::CsrMatrix;
use crate::error::Result;
use crate::solver::cg::Gauge;
use coarse::CoarseSolver;
use std::borrow::Cow;

/// AMG の設定（SPEC §4.5）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmgOptions {
    /// 最細レベルの強連結の閾値 θ。平滑化凝集の標準値は 0.08 前後
    pub strength_threshold: f64,
    /// レベルを 1 つ粗くするごとに θ に掛ける係数（Vaněk の元論文と同じ 1/2）。
    ///
    /// 粗レベルの行列は 1 行あたりの項が増え、その分 1 項ずつは小さくなる。
    /// θ を据え置くとほとんどの連結が「弱い」と判定され、凝集が単独節点だらけになり
    /// **粗化が進まないまま作用素だけが密になる**（実測で 1 レベルあたり 1.9 倍しか
    /// 粗くならず、非ゼロが 1 行 234 個まで膨らんだ）。θ を半分ずつ緩めるとこれが直る。
    pub strength_decay: f64,
    /// レベル数の上限
    pub max_levels: usize,
    /// これ以下の大きさになったら粗化をやめて直接解く
    pub coarse_size: usize,
    /// 下りの平滑化回数
    pub presmooth: usize,
    /// 上りの平滑化回数。V サイクルを対称に保つため `presmooth` と同じにすること
    pub postsmooth: usize,
    /// 粗化がこの割合より進まなくなったら打ち切る（無限ループ防止）
    pub max_coarsening_ratio: f64,
}

/// 最細レベルの強連結の閾値 θ の既定値（1 次要素で調整した値）。
pub const DEFAULT_STRENGTH_THRESHOLD: f64 = 0.08;

/// 2 次要素（Tet10）の行列で使う θ（SPEC §4.5「2 次要素の行列」）。
///
/// 2 次要素の剛性行列は非対角に正の成分を持ち、1 行の項も倍ほど多い。θ = 0.08 のままだと
/// 粗化が弱く、反復回数が同じ節点数の 1 次要素の 3 倍前後になる。0.15 にすると 3 割ほど減る
/// （3 層スラブ、2.7 万〜69 万節点で 78 → 56、87 → 60、97 → 63 回）。
pub const QUADRATIC_STRENGTH_THRESHOLD: f64 = 0.15;

impl Default for AmgOptions {
    fn default() -> Self {
        Self {
            strength_threshold: DEFAULT_STRENGTH_THRESHOLD,
            strength_decay: 0.5,
            max_levels: 25,
            coarse_size: 400,
            presmooth: 1,
            postsmooth: 1,
            max_coarsening_ratio: 0.85,
        }
    }
}

/// レベル間の移送。`R` は `P` の転置を CSR として明示的に持つ。
///
/// 転置をその場で使う（列方向に足し込む）実装にすると GPU で書き込みが衝突し、
/// atomicAdd の順序が実行ごとに変わってしまう。メモリを使ってでも再現性を採る
/// （[`CsrMatrix::transpose`] の説明を参照）。
#[derive(Debug, Clone)]
pub struct Transfer {
    /// 補間 P（細 × 粗）
    pub p: CsrMatrix,
    /// 制限 R = Pᵀ（粗 × 細）
    pub r: CsrMatrix,
}

/// 階層の 1 レベル。
#[derive(Debug, Clone)]
pub struct Level<'a> {
    /// このレベルの行列。最細レベルは [`setup`] に渡された行列を借りる（複製しない。
    /// 1000 万節点で約 1.9 GB）。粗いレベルは階層が自分で持つ
    pub a: Cow<'a, CsrMatrix>,
    /// 対角の逆数（平滑化に使う）
    pub inv_diag: Vec<f64>,
    /// 減衰 Jacobi の緩和係数 ω = 4/(3ρ)
    pub omega: f64,
    /// 近零空間ベクトル（最細レベルでは 1）
    pub nullspace: Vec<f64>,
    /// 1 つ粗いレベルへの移送。最粗レベルでは `None`
    pub transfer: Option<Transfer>,
}

/// AMG の階層。最細レベルの行列を借りているので、その行列より長くは生きられない。
#[derive(Debug, Clone)]
pub struct Hierarchy<'a> {
    /// `levels[0]` が最細
    pub levels: Vec<Level<'a>>,
    /// 最粗レベルの直接解法
    pub coarse: CoarseSolver,
    /// 平滑化回数（[`AmgOptions`] から写す）
    pub presmooth: usize,
    /// 最細レベルの強連結の閾値 θ（記録用。[`AmgOptions`] から写す）
    pub strength_threshold: f64,
    /// 上りの平滑化回数
    pub postsmooth: usize,
}

/// 記録・表示用の階層の要約（SPEC §5.5 のプロビナンス）。
#[derive(Debug, Clone, PartialEq)]
pub struct AmgInfo {
    /// レベルごとの `(節点数, 非ゼロ数)`
    pub levels: Vec<(usize, usize)>,
    /// 作用素複雑度 Σ nnz(A_l) / nnz(A_0)。V サイクル 1 回の相対コストの目安
    pub operator_complexity: f64,
    /// 格子複雑度 Σ n_l / n_0
    pub grid_complexity: f64,
    /// 最粗レベルを密分解で解けたか
    pub direct_coarse_solve: bool,
    /// 最細レベルの強連結の閾値 θ
    pub strength_threshold: f64,
}

impl Hierarchy<'_> {
    /// 記録用の要約。
    pub fn info(&self) -> AmgInfo {
        let levels: Vec<(usize, usize)> = self
            .levels
            .iter()
            .map(|l| (l.a.n_rows(), l.a.nnz()))
            .collect();
        let (n0, nnz0) = levels[0];
        let total_nnz: usize = levels.iter().map(|&(_, nnz)| nnz).sum();
        let total_n: usize = levels.iter().map(|&(n, _)| n).sum();
        AmgInfo {
            levels,
            operator_complexity: total_nnz as f64 / nnz0.max(1) as f64,
            grid_complexity: total_n as f64 / n0.max(1) as f64,
            direct_coarse_solve: self.coarse.is_direct(),
            strength_threshold: self.strength_threshold,
        }
    }

    /// レベル数。
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }
}

/// 行列 `a` から階層を作る（SPEC §4.5）。
///
/// 近零空間は定数ベクトル span{1}（全 Neumann の電流場。SPEC §2.2）を仮定する。
pub fn setup<'a>(a: &'a CsrMatrix, options: &AmgOptions) -> Result<Hierarchy<'a>> {
    setup_with(a, options, Gauge::MeanZero)
}

/// [`setup`] の、解く系が特異かどうかを指定する版（SPEC §4.5 の「CEM の電極があるとき」）。
///
/// 近零空間のベクトルはどちらでも定数 1 を使う。完全電極モデル（SPEC §2.5）の
/// 接触インピーダンスの項は電極の下にしか効かないので、定数はなお平滑化で消えない成分である。
/// 違うのは最粗レベルの解き方だけ（[`CoarseSolver::factor`]）。
pub fn setup_with<'a>(
    a: &'a CsrMatrix,
    options: &AmgOptions,
    gauge: Gauge,
) -> Result<Hierarchy<'a>> {
    let mut levels: Vec<Level<'a>> = Vec::new();
    // 最細レベルは借りる。CG が同じ行列を持っているので複製しない
    let mut current: Cow<'a, CsrMatrix> = Cow::Borrowed(a);
    let mut nullspace = vec![1.0f64; a.n_rows()];

    loop {
        let n = current.n_rows();
        let rho = prolongation::spectral_bound(&current)?;
        // 減衰 Jacobi。ω < 2/ρ なら平滑化作用素が「A に対して収束的」になり、
        // V サイクル全体が対称正定値になる（CG の前処理として使うための条件）
        let omega = 4.0 / (3.0 * rho);
        let inv_diag: Vec<f64> = current.diagonal().iter().map(|d| 1.0 / d).collect();

        let last = n <= options.coarse_size || levels.len() + 1 >= options.max_levels;
        if last {
            levels.push(Level {
                a: current,
                inv_diag,
                omega,
                nullspace,
                transfer: None,
            });
            break;
        }

        // --- 粗化 -------------------------------------------------------
        let theta = options.strength_threshold * options.strength_decay.powi(levels.len() as i32);
        let strength = aggregate::strength_graph(&current, theta);
        let agg = aggregate::aggregate(&strength, n);
        if agg.count as f64 > n as f64 * options.max_coarsening_ratio || agg.count == 0 {
            // 粗化が進まない（強連結がほとんど無い等）。ここを最粗レベルにする
            levels.push(Level {
                a: current,
                inv_diag,
                omega,
                nullspace,
                transfer: None,
            });
            break;
        }

        let af = prolongation::filtered(&current, &strength, &nullspace)?;
        let omega_p = 4.0 / (3.0 * prolongation::spectral_bound(&af)?);
        // A_f はこのあと使わないので、複製せずにその場で平滑化行列へ書き換える
        let smoother = prolongation::into_jacobi_smoother(af, omega_p);
        let (p0, coarse_null) = prolongation::tentative(&agg, &nullspace)?;
        let p = spmm::spmm(&smoother, &p0);
        let r = p.transpose();

        // Galerkin 粗行列 A_c = R (A P)
        let ap = spmm::spmm(&current, &p);
        let coarse_a = spmm::spmm(&r, &ap);

        levels.push(Level {
            a: current,
            inv_diag,
            omega,
            nullspace,
            transfer: Some(Transfer { p, r }),
        });
        current = Cow::Owned(coarse_a);
        nullspace = coarse_null;
    }

    let last = levels.last().expect("レベルが 1 つも無い");
    let coarse = CoarseSolver::factor(&last.a, &last.nullspace, gauge);
    Ok(Hierarchy {
        levels,
        coarse,
        presmooth: options.presmooth,
        postsmooth: options.postsmooth,
        strength_threshold: options.strength_threshold,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3 次元の 7 点ラプラシアン（全 Neumann）。導電率のコントラストを付けられる。
    ///
    /// `sigma(i, j, k)` が要素の導電率を返す。
    pub(crate) fn laplacian_3d(
        nx: usize,
        ny: usize,
        nz: usize,
        sigma: impl Fn(usize, usize, usize) -> f64,
    ) -> CsrMatrix {
        let n = nx * ny * nz;
        let index = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
        let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
        // 隣接する格子点の間に「抵抗」を置く。両端の対角に足すので行和がゼロになる
        let connect = |a: usize, b: usize, w: f64, rows: &mut Vec<Vec<(u32, f64)>>| {
            rows[a].push((b as u32, -w));
            rows[b].push((a as u32, -w));
            rows[a].push((a as u32, w));
            rows[b].push((b as u32, w));
        };
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let s = sigma(i, j, k);
                    if i + 1 < nx {
                        let w = 0.5 * (s + sigma(i + 1, j, k));
                        connect(index(i, j, k), index(i + 1, j, k), w, &mut rows);
                    }
                    if j + 1 < ny {
                        let w = 0.5 * (s + sigma(i, j + 1, k));
                        connect(index(i, j, k), index(i, j + 1, k), w, &mut rows);
                    }
                    if k + 1 < nz {
                        let w = 0.5 * (s + sigma(i, j, k + 1));
                        connect(index(i, j, k), index(i, j, k + 1), w, &mut rows);
                    }
                }
            }
        }
        let mut row_ptr = vec![0usize];
        let mut col_idx: Vec<u32> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        for row in rows.iter_mut() {
            row.sort_by_key(|&(c, _)| c);
            let mut k = 0;
            while k < row.len() {
                let c = row[k].0;
                let mut acc = 0.0;
                while k < row.len() && row[k].0 == c {
                    acc += row[k].1;
                    k += 1;
                }
                col_idx.push(c);
                values.push(acc);
            }
            row_ptr.push(col_idx.len());
        }
        let mut m = CsrMatrix::from_structure(n, row_ptr, col_idx);
        m.values_mut().copy_from_slice(&values);
        m
    }

    #[test]
    fn the_hierarchy_coarsens_and_terminates() {
        let a = laplacian_3d(20, 20, 20, |_, _, _| 1.0);
        let h = setup(&a, &AmgOptions::default()).unwrap();
        assert!(h.n_levels() >= 2, "レベル数 {}", h.n_levels());
        // レベルごとに単調に小さくなること
        for w in h.levels.windows(2) {
            assert!(
                w[1].a.n_rows() < w[0].a.n_rows(),
                "{} → {}",
                w[0].a.n_rows(),
                w[1].a.n_rows()
            );
        }
        assert!(h.levels.last().unwrap().transfer.is_none());
        assert!(h.coarse.is_direct(), "最粗レベルを密分解できていない");
        let info = h.info();
        assert!(
            info.operator_complexity < 3.0,
            "作用素複雑度 {} が大きすぎる",
            info.operator_complexity
        );
    }

    #[test]
    fn the_nullspace_survives_every_level() {
        // AMG が効くかどうかの本質。各レベルで A_l b_l = 0 が保たれること
        let a = laplacian_3d(16, 16, 16, |_, _, k| if k < 8 { 1.0 } else { 0.05 });
        let h = setup(&a, &AmgOptions::default()).unwrap();
        for (l, level) in h.levels.iter().enumerate() {
            let n = level.a.n_rows();
            let mut y = vec![0.0; n];
            level.a.spmv(&level.nullspace, &mut y);
            let worst = y.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let scale = level.a.values().iter().fold(0.0f64, |m, v| m.max(v.abs()))
                * level.nullspace.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            assert!(
                worst < 1e-10 * scale.max(1.0),
                "レベル {l}: ‖A b‖∞ = {worst}（スケール {scale}）"
            );
        }
    }

    #[test]
    fn the_transfer_operators_are_transposes_of_each_other() {
        let a = laplacian_3d(12, 12, 12, |_, _, _| 1.0);
        let h = setup(&a, &AmgOptions::default()).unwrap();
        for level in &h.levels {
            let Some(t) = &level.transfer else { continue };
            assert_eq!(t.r.n_rows(), t.p.n_cols());
            assert_eq!(t.r.n_cols(), t.p.n_rows());
            // ⟨P x, y⟩ = ⟨x, R y⟩ を数値で確かめる
            let (nf, nc) = (t.p.n_rows(), t.p.n_cols());
            let x: Vec<f64> = (0..nc).map(|i| (i as f64 * 0.7).sin()).collect();
            let y: Vec<f64> = (0..nf).map(|i| (i as f64 * 0.3).cos()).collect();
            let mut px = vec![0.0; nf];
            t.p.spmv(&x, &mut px);
            let mut ry = vec![0.0; nc];
            t.r.spmv(&y, &mut ry);
            let lhs: f64 = px.iter().zip(&y).map(|(a, b)| a * b).sum();
            let rhs: f64 = x.iter().zip(&ry).map(|(a, b)| a * b).sum();
            assert!(
                (lhs - rhs).abs() < 1e-10 * lhs.abs().max(1.0),
                "{lhs} vs {rhs}"
            );
        }
    }

    #[test]
    fn coarse_matrices_stay_symmetric() {
        let a = laplacian_3d(14, 14, 14, |i, _, _| if i < 7 { 1.0 } else { 0.02 });
        let h = setup(&a, &AmgOptions::default()).unwrap();
        for (l, level) in h.levels.iter().enumerate() {
            let scale = level.a.values().iter().fold(0.0f64, |m, v| m.max(v.abs()));
            assert!(
                level.a.is_symmetric(1e-10 * scale.max(1.0)),
                "レベル {l} の行列が対称でない"
            );
        }
    }

    #[test]
    fn setup_is_reproducible() {
        // CLAUDE.md 設計原則 1: 同じ行列から常に同じ階層
        let a = laplacian_3d(15, 15, 15, |_, j, _| 1.0 + j as f64);
        let first = setup(&a, &AmgOptions::default()).unwrap();
        for _ in 0..3 {
            let again = setup(&a, &AmgOptions::default()).unwrap();
            assert_eq!(first.n_levels(), again.n_levels());
            for (x, y) in first.levels.iter().zip(&again.levels) {
                assert_eq!(x.a.values(), y.a.values());
                assert_eq!(x.a.col_idx(), y.a.col_idx());
                assert_eq!(x.nullspace, y.nullspace);
                assert_eq!(x.omega, y.omega);
            }
        }
    }
}
