//! 前処理の切り替え（SPEC §4.3 / §4.5）。
//!
//! CG 本体（[`super::cg`]）は `z = M⁻¹ r` を 1 行で呼びたいだけなので、
//! 前処理ごとの違いをここに閉じ込める。前処理はバックエンド側の状態
//! （GPU なら VRAM 上の対角ベクトルや階層）を持つので、
//! [`LinearBackend::Vector`] を含む型になる。

use crate::assemble::csr::CsrMatrix;
use crate::case::{Preconditioner, SolverOptions};
use crate::error::{GfemError, Result};
use crate::solver::amg::cycle::DeviceHierarchy;
use crate::solver::amg::{self, AmgInfo, Hierarchy};
use crate::solver::backend::LinearBackend;

/// バックエンドに載せた前処理。
pub enum DevicePreconditioner<'h, B: LinearBackend> {
    /// 何もしない（`z = r`）
    Identity,
    /// 対角スケーリング。持っているのは対角の逆数
    Jacobi(B::Vector),
    /// 代数マルチグリッド。V サイクルを 1 回まわす
    Amg(Box<DeviceHierarchy<'h, B>>),
}

impl<'h, B: LinearBackend> DevicePreconditioner<'h, B> {
    /// `z = M⁻¹ r`。
    pub fn apply(&mut self, backend: &B, r: &B::Vector, z: &mut B::Vector) -> Result<()> {
        match self {
            DevicePreconditioner::Identity => backend.copy_from(r, z),
            DevicePreconditioner::Jacobi(inv_diag) => backend.multiply(inv_diag, r, z),
            DevicePreconditioner::Amg(h) => h.apply(backend, r, z),
        }
    }

    /// 記録用の名前。
    pub fn label(&self) -> &'static str {
        match self {
            DevicePreconditioner::Identity => Preconditioner::None.label(),
            DevicePreconditioner::Jacobi(_) => Preconditioner::Jacobi.label(),
            DevicePreconditioner::Amg(_) => Preconditioner::Amg.label(),
        }
    }

    /// AMG の階層の要約（AMG 以外では `None`）。
    pub fn amg_info(&self) -> Option<AmgInfo> {
        match self {
            DevicePreconditioner::Amg(h) => Some(h.host().info()),
            _ => None,
        }
    }
}

/// AMG の階層はホスト側で組み立てる。
///
/// バックエンドを一切使わないので、**CPU 逐次・CPU 並列・GPU のどれで解いても
/// まったく同じ階層になる**（CLAUDE.md 設計原則 1）。`preconditioner: amg`
/// 以外では `None` を返す。
pub fn build_hierarchy(matrix: &CsrMatrix, options: &SolverOptions) -> Result<Option<Hierarchy>> {
    match options.preconditioner {
        Preconditioner::Amg => Ok(Some(amg::setup(matrix, &options.amg)?)),
        _ => Ok(None),
    }
}

/// [`build_hierarchy`] の結果をバックエンドに載せる。
pub fn build<'h, B: LinearBackend>(
    backend: &B,
    matrix: &CsrMatrix,
    options: &SolverOptions,
    hierarchy: Option<&'h Hierarchy>,
) -> Result<DevicePreconditioner<'h, B>> {
    match options.preconditioner {
        Preconditioner::None => Ok(DevicePreconditioner::Identity),
        Preconditioner::Jacobi => {
            let inv_diag = inverse_diagonal(matrix)?;
            Ok(DevicePreconditioner::Jacobi(backend.upload(&inv_diag)?))
        }
        Preconditioner::Amg => {
            let h = hierarchy.ok_or_else(|| {
                GfemError::internal("AMG の階層が組み立てられていません".to_string())
            })?;
            Ok(DevicePreconditioner::Amg(Box::new(DeviceHierarchy::new(
                backend, h,
            )?)))
        }
    }
}

/// 対角の逆数。対角が正でなければ、どの行かを添えてエラーにする（設計原則 6）。
fn inverse_diagonal(matrix: &CsrMatrix) -> Result<Vec<f64>> {
    let diag = matrix.diagonal();
    if let Some((i, d)) = diag
        .iter()
        .enumerate()
        .find(|&(_, &d)| d <= 0.0 || d.is_nan())
    {
        return Err(GfemError::internal(format!(
            "剛性行列の対角成分が正ではありません: 行 {i} で {d}"
        ))
        .with_hint("退化要素か、導電率がゼロ以下の領域が残っている可能性があります"));
    }
    Ok(diag.iter().map(|d| 1.0 / d).collect())
}
