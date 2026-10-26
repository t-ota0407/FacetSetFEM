//! GPU の初期化を使い回すこと（SPEC §4.4）。
//!
//! 空き VRAM を見るテストがあるので、ほかの GPU テストと同じプロセスに入れない
//! （同時に走ると空き VRAM がぶれる）。GPU が無い環境では黙って通す（設計原則 5）。

#[path = "../../facetsetfem-core/tests/common/mod.rs"]
mod common;

use common::{BOTTOM_PID, TOP_PID};
use facetsetfem_core::case::{Backend, Case, Electrode};
use facetsetfem_core::solve_with;
use facetsetfem_core::solver::BackendSolver;
use facetsetfem_cuda::{CudaBackend, Device};

fn gpu() -> Option<CudaBackend> {
    match CudaBackend::new(0) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("GPU が使えないので飛ばします: {e}");
            None
        }
    }
}

#[test]
fn the_device_is_initialised_once_per_process() {
    let Some(a) = gpu() else { return };
    let Some(b) = gpu() else { return };
    // コンテキストとコンパイル済みのカーネルを共有している（2 回目は初期化しない）
    assert!(std::ptr::eq(a.device(), b.device()));
    assert!(std::ptr::eq(a.device(), &*Device::shared(0).unwrap()));
}

#[test]
fn vram_is_given_back_after_each_solve() {
    let Some(gpu) = gpu() else { return };
    let device = Device::shared(0).unwrap();

    // 22 万節点。CSR と作業ベクトルで 50 MB ほど使う
    let mesh = common::uniform_slab(0.060, 0.040, 0.020, 120, 80, 22);
    let mut case = Case::new();
    case.set_conductivity(1, 0.35).unwrap();
    case.add_electrode(Electrode::anode("top", 2e-3).with_shell_pids([TOP_PID]));
    case.add_electrode(Electrode::cathode("bottom", 2e-3).with_shell_pids([BOTTOM_PID]));
    case.solver.backend = Backend::Cuda;
    case.solver.tolerance = 1e-8;

    let before = device.free_memory().unwrap();
    let analysis = solve_with(&mesh, &case, None, Some(&BackendSolver(gpu))).unwrap();
    assert!(analysis.solver.converged);
    let after = device.free_memory().unwrap();

    // デバイスを使い回すので、プールに残ったぶんは後始末で返さないと掴んだままになる
    let kept = before.saturating_sub(after);
    assert!(
        kept < 16 << 20,
        "解き終わっても VRAM を {:.0} MB 掴んだまま",
        kept as f64 / 1e6
    );
}
