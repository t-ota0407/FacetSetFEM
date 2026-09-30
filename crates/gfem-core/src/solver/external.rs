//! 外部クレートのバックエンドを差し込むための入口（SPEC §4.4 / §7）。
//!
//! `gfem-core` は CUDA に一切依存しない（CLAUDE.md 設計原則 5）。そのため
//! [`crate::analysis::solve`] が自分で CUDA バックエンドを選ぶことはできない。
//! 代わりに、**呼び出し側が組み立てたソルバを渡せる**ようにしてある。
//!
//! ```text
//! gfem-cuda  →  CudaSolver（LinearSolver を実装）
//!                     ↓ 渡す
//! gfem-py    →  gfem_core::analysis::solve_with(mesh, case, progress, Some(&solver))
//! ```
//!
//! [`LinearBackend`] は関連型を持つのでトレイトオブジェクトにできないが、
//! この trait はホスト側のスライスだけを受け渡すのでオブジェクト安全になる。
//! 実装は [`BackendSolver`] でまとめて用意できる。

use crate::assemble::csr::CsrMatrix;
use crate::case::SolverOptions;
use crate::error::Result;
use crate::solver::backend::LinearBackend;
use crate::solver::cg::{self, CgReport, Gauge, ProgressFn};

/// 線形システムを解くもの。オブジェクト安全な、バックエンドの外向きの顔。
pub trait LinearSolver {
    /// 記録用の名前（`cuda` など。SPEC §5.5 のプロビナンス）。
    fn name(&self) -> &'static str;

    /// `K φ = f` を解く。契約は [`cg::solve`] と同じ。
    fn solve(
        &self,
        matrix: &CsrMatrix,
        rhs: &[f64],
        phi: &mut [f64],
        options: &SolverOptions,
        progress: Option<ProgressFn<'_>>,
    ) -> Result<CgReport>;

    /// 同じ行列で右辺を替えて順に解く。契約は [`cg::solve_many`] と同じ
    /// （完全電極モデルの重ね合わせ。SPEC §2.5 / §4.3）。
    fn solve_many(
        &self,
        matrix: &CsrMatrix,
        rhs: &[&[f64]],
        solutions: &mut [&mut [f64]],
        options: &SolverOptions,
        gauge: Gauge,
        progress: Option<ProgressFn<'_>>,
    ) -> Result<Vec<CgReport>>;
}

/// 任意の [`LinearBackend`] を [`LinearSolver`] にする包み。
///
/// CG 本体は 1 箇所（[`cg::solve`]）のままで、バックエンドだけが差し替わる。
#[derive(Debug, Clone, Copy, Default)]
pub struct BackendSolver<B>(pub B);

impl<B: LinearBackend> LinearSolver for BackendSolver<B> {
    fn name(&self) -> &'static str {
        self.0.name()
    }

    fn solve(
        &self,
        matrix: &CsrMatrix,
        rhs: &[f64],
        phi: &mut [f64],
        options: &SolverOptions,
        progress: Option<ProgressFn<'_>>,
    ) -> Result<CgReport> {
        cg::solve(&self.0, matrix, rhs, phi, options, progress)
    }

    fn solve_many(
        &self,
        matrix: &CsrMatrix,
        rhs: &[&[f64]],
        solutions: &mut [&mut [f64]],
        options: &SolverOptions,
        gauge: Gauge,
        progress: Option<ProgressFn<'_>>,
    ) -> Result<Vec<CgReport>> {
        cg::solve_many(&self.0, matrix, rhs, solutions, options, gauge, progress)
    }
}
