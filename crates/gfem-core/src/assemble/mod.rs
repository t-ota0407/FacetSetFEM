//! 線形システムの組み立て（SPEC §4.1 / §4.2）。
//!
//! - [`csr`] — CSR スパース行列と、そのグラフの連結性検査
//! - [`sparsity`] — 非ゼロ構造の構築
//! - [`stiffness`] — 剛性行列 K のアセンブリ
//! - [`load`] — 電極荷重ベクトル f の構築

pub mod csr;
pub mod load;
pub mod sparsity;
pub mod stiffness;

use crate::case::Case;
use crate::error::Result;
use crate::mesh::Mesh;

pub use csr::{CsrMatrix, connected_components, require_connected};
pub use load::{ElectrodeGeometry, assemble_load};
pub use sparsity::build_sparsity;
pub use stiffness::assemble_stiffness;

/// 解くべき線形システム `K φ = f`。
#[derive(Debug, Clone)]
pub struct LinearSystem {
    /// 剛性行列
    pub matrix: CsrMatrix,
    /// 荷重ベクトル
    pub load: Vec<f64>,
    /// 電極ごとの幾何情報
    pub electrodes: Vec<ElectrodeGeometry>,
}

/// メッシュとケースから線形システムを組み立てる。
///
/// ケースの整合性は事前に [`Case::validate`] で検査しておくこと。
/// ここでは連結性（射影 CG の前提。SPEC §2.2）だけ改めて確認する。
pub fn assemble(mesh: &Mesh, case: &Case) -> Result<LinearSystem> {
    let mut matrix = build_sparsity(mesh)?;
    require_connected(&matrix)?;
    assemble_stiffness(mesh, &case.conductivity, &mut matrix)?;
    let (load, electrodes) = assemble_load(mesh, case)?;
    Ok(LinearSystem {
        matrix,
        load,
        electrodes,
    })
}
