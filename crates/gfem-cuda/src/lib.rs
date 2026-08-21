//! GenericFEM の CUDA バックエンド（SPEC §4.4 / §8 M5）。
//!
//! `gfem-core` は CUDA に一切依存しない（CLAUDE.md 設計原則 5）。このクレートが
//! [`gfem_core::LinearBackend`] を実装し、[`gfem_core::LinearSolver`] として
//! 差し込めるようにする。**CG のアルゴリズム本体は `gfem-core` の 1 箇所のまま**で、
//! ここが持つのはベクトル演算の中身だけ。
//!
//! ```no_run
//! # use gfem_core::{Case, Mesh};
//! # fn demo(mesh: &Mesh, case: &Case) -> gfem_core::Result<()> {
//! // 使えるなら GPU、駄目なら CPU に落として、その理由を受け取る
//! let (solver, note) = gfem_cuda::solver_or_fallback(mesh.nodes.len(), None);
//! if let Some(note) = note {
//!     eprintln!("{note}");
//! }
//! let analysis = gfem_core::solve_with(mesh, case, None, solver.as_deref())?;
//! # let _ = analysis;
//! # Ok(())
//! # }
//! ```
//!
//! # 結果は CPU と 1 ビットも変わらない
//!
//! これは速度と同じくらい重要な要件（CLAUDE.md 設計原則 1、SPEC §8 M5）。
//! 仕組みは 2 段構えになっている。
//!
//! - **カーネル側**（`kernels.cu`）: 積和を FMA にまとめさせない。
//!   総和は長さ 512 の連続ブロックをスレッド 1 本が先頭から順に足す
//! - **ホスト側**（[`backend`]）: ブロック和の列を、CPU の
//!   `solver::reduce::sum` と**同じ木**でたたむ。最後の 512 要素以下は
//!   ホストへ戻して CPU 側の関数をそのまま呼ぶ
//!
//! CG のスカラー演算（α・β・残差の割り算）は元からホスト側の `f64` なので、
//! バックエンドによって変わりようがない。
//!
//! 検証は `tests/bitwise.rs`。

pub mod backend;
pub mod device;
mod error;

pub use backend::{CudaBackend, CudaCsr, CudaVector};
pub use device::{DeviceInfo, MIN_COMPUTE_CAPABILITY, devices};

use gfem_core::solver::{BackendSolver, LinearSolver};

/// CG が VRAM に置く作業ベクトルの本数。
///
/// 解 x・右辺 f・残差 r・前処理後 z・探索方向 p・Kp・前処理の対角の 7 本
/// （`gfem_core::solver::cg` を参照）。リダクションの一時領域は
/// 節点数の 1/512 以下なので、この見積りには含めない。
const WORK_VECTORS: usize = 7;

/// 節点数 `n_nodes`・非ゼロ数 `nnz` の問題に要る VRAM の見積り [byte]。
///
/// CSR（行ポインタ 8B・列番号 4B・成分 8B）と作業ベクトル。
pub fn required_memory(n_nodes: usize, nnz: usize) -> usize {
    (n_nodes + 1) * 8 + nnz * 12 + n_nodes * WORK_VECTORS * 8
}

/// 空き VRAM に対して確保しすぎないための余裕（実測の断片化ぶん）。
const HEADROOM: f64 = 0.90;

/// GPU で解けるかを判定し、駄目なら理由を返す。
///
/// `nnz` が分かっていれば渡す。分からない場合は節点あたり 15 非ゼロ（Tet4 の実測値）
/// で見積もる。
pub fn can_use_gpu(n_nodes: usize, nnz: Option<usize>) -> std::result::Result<DeviceInfo, String> {
    let nnz = nnz.unwrap_or(n_nodes * 15);
    let needed = required_memory(n_nodes, nnz);

    let all = devices();
    if all.is_empty() {
        return Err("CUDA が使える GPU が見つかりませんでした".to_string());
    }
    let usable: Vec<&DeviceInfo> = all.iter().filter(|d| d.supports_f64()).collect();
    let Some(&best) = usable
        .iter()
        .max_by_key(|d| d.free_memory)
        .map(|d| &**d)
        .as_ref()
    else {
        let (major, minor) = MIN_COMPUTE_CAPABILITY;
        return Err(format!(
            "見つかった GPU はどれも Compute Capability {major}.{minor} 未満で、\
             倍精度演算に使えません"
        ));
    };
    let budget = (best.free_memory as f64 * HEADROOM) as usize;
    if needed > budget {
        return Err(format!(
            "VRAM が足りません: {:.1} GB 必要ですが、GPU {} の空きは {:.1} GB です",
            needed as f64 / 1e9,
            best.ordinal,
            best.free_memory as f64 / 1e9
        ));
    }
    Ok(best.clone())
}

/// GPU で解けるならソルバを、駄目なら `None` と理由を返す（SPEC §4.4 のフォールバック）。
///
/// 「黙って CPU に落ちる」ことは避け、必ず理由を返す。呼び出し側は
/// それを結果の警告なりログなりに載せること。
#[allow(clippy::type_complexity)]
pub fn solver_or_fallback(
    n_nodes: usize,
    nnz: Option<usize>,
) -> (Option<Box<dyn LinearSolver>>, Option<String>) {
    let info = match can_use_gpu(n_nodes, nnz) {
        Ok(info) => info,
        Err(reason) => {
            return (None, Some(format!("{reason}。CPU で解きます")));
        }
    };
    match CudaBackend::new(info.ordinal) {
        Ok(backend) => (Some(Box::new(BackendSolver(backend))), None),
        Err(e) => (
            None,
            Some(format!(
                "GPU を初期化できませんでした（{e}）。CPU で解きます"
            )),
        ),
    }
}
