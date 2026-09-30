//! 線形システムの組み立て（SPEC §4.1 / §4.2）。
//!
//! - [`adjacency`] — 節点 → 要素の表と、それを使った並列アセンブリ
//! - [`csr`] — CSR スパース行列と、そのグラフの連結性検査
//! - [`sparsity`] — 非ゼロ構造の構築
//! - [`stiffness`] — 剛性行列 K のアセンブリ
//! - [`load`] — 電極荷重ベクトル f の構築

pub mod adjacency;
pub mod csr;
pub mod load;
pub mod sparsity;
pub mod stiffness;

use crate::case::{Case, FaceLookup};
use crate::error::Result;
use crate::mesh::Mesh;
use crate::solver::cg::Gauge;

pub use adjacency::{NodeElements, assemble_stiffness_parallel, build_node_elements};
pub use csr::{CsrMatrix, connected_components, require_connected};
pub use load::{ContactFace, ElectrodeGeometry, assemble_load};
pub use sparsity::{build_sparsity, build_sparsity_with};
pub use stiffness::assemble_stiffness;

/// 解くべき線形システム `K φ = f`。
///
/// 完全電極モデルの電極があれば、行列は `K + Σ M_l/z_l`、荷重は一様電流密度の電極の分
/// f_u だけになり、電極の電位を入れた右辺は重ね合わせで作る（SPEC §2.5 / [`crate::contact`]）。
#[derive(Debug, Clone)]
pub struct LinearSystem {
    /// 剛性行列（CEM の電極があれば接触インピーダンスの項を含む）
    pub matrix: CsrMatrix,
    /// 荷重ベクトル（一様電流密度の電極の分）
    pub load: Vec<f64>,
    /// 電極ごとの幾何情報
    pub electrodes: Vec<ElectrodeGeometry>,
    /// 行列が特異（全 Neumann）か正則（CEM の電極がある）か
    pub gauge: Gauge,
}

/// メッシュとケースから線形システムを組み立てる。
///
/// ケースの整合性は事前に [`Case::validate`] で検査しておくこと。
/// ここでは連結性（射影 CG の前提。SPEC §2.2）だけ改めて確認する。
///
/// `parallel` が true なら行ごとの並列アセンブリを使う。
/// **どちらでも同じ行列**になる（[`adjacency`] の説明を参照）。
pub fn assemble(
    mesh: &Mesh,
    case: &Case,
    lookup: &FaceLookup,
    parallel: bool,
) -> Result<LinearSystem> {
    // 節点 → 要素の表は、非ゼロ構造と並列アセンブリの両方で使う（作るのは 1 回）
    let adjacency = build_node_elements(mesh);
    let mut matrix = build_sparsity_with(mesh, &adjacency)?;
    require_connected(&matrix)?;
    if parallel {
        assemble_stiffness_parallel(mesh, &case.conductivity, &adjacency, &mut matrix)?;
    } else {
        assemble_stiffness(mesh, &case.conductivity, &mut matrix)?;
    }
    let (load, electrodes) = assemble_load(mesh, case, lookup)?;
    // 完全電極モデル（SPEC §2.5）。剛性の後に、電極の並び → 面の通し番号の順で足す
    let gauge = if case.has_complete_electrodes() {
        crate::contact::add_contact_terms(&mut matrix, &electrodes)?;
        Gauge::NotNeeded
    } else {
        Gauge::MeanZero
    };
    Ok(LinearSystem {
        matrix,
        load,
        electrodes,
        gauge,
    })
}
