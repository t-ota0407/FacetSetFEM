//! AMG 前処理の検証（SPEC §4.5 / §8 M6）。
//!
//! 見ているのは 4 つ。
//!
//! 1. **V サイクルが対称正定値**であること。CG の前処理はこれを満たさないと
//!    直交性が崩れて収束が保証されない。実装のどこかを崩すと真っ先に壊れる性質
//! 2. **答えが変わらない**こと。前処理は収束の速さを変えるだけで、
//!    解いている方程式は同じ
//! 3. **反復回数がメッシュの細かさにほとんど依存しない**こと。これが AMG を
//!    入れる唯一の理由（Jacobi は h→0 で反復回数が増える）
//! 4. **CPU 逐次と CPU 並列がビット単位で一致する**こと（CLAUDE.md 設計原則 1）

mod common;

use common::{BOTTOM_PID, Layer, TOP_PID, slab};
use gfem_core::assemble::csr::CsrMatrix;
use gfem_core::case::{
    Backend, Case, Electrode, Preconditioner, ReferencePotential, SolverOptions,
};
use gfem_core::solver::amg::cycle::DeviceHierarchy;
use gfem_core::solver::amg::{self, AmgOptions};
use gfem_core::solver::backend::{CpuParallel, CpuSerial, LinearBackend};
use gfem_core::{Analysis, Mesh, solve};

const CURRENT: f64 = 2.0e-3;
const LX: f64 = 0.060;
const LY: f64 = 0.040;

/// 3 層スラブ（皮膚 0.10 / 脂肪 0.04 / 筋肉 0.35 S/m）。
fn mesh(nx: usize, ny: usize, refine: usize) -> Mesh {
    slab(
        LX,
        LY,
        nx,
        ny,
        &[
            Layer {
                pid: 1,
                thickness: 0.002,
                divisions: refine,
            },
            Layer {
                pid: 2,
                thickness: 0.004,
                divisions: refine,
            },
            Layer {
                pid: 3,
                thickness: 0.014,
                divisions: 2 * refine,
            },
        ],
    )
}

fn case_with(preconditioner: Preconditioner, backend: Backend) -> Case {
    let mut case = Case::new();
    for (pid, sigma) in [(1u32, 0.10), (2, 0.04), (3, 0.35)] {
        case.set_conductivity(pid, sigma).unwrap();
    }
    case.add_electrode(Electrode::anode("top", CURRENT).with_shell_pids([TOP_PID]));
    case.add_electrode(Electrode::cathode("bottom", CURRENT).with_shell_pids([BOTTOM_PID]));
    case.solver = SolverOptions {
        tolerance: 1.0e-10,
        max_iterations: 50_000,
        preconditioner,
        backend,
        ..Default::default()
    };
    case.reference = ReferencePotential::CathodeMean;
    case
}

/// 剛性行列だけが欲しいときの近道。
fn stiffness(mesh: &Mesh, case: &Case) -> CsrMatrix {
    let surface = gfem_core::mesh::extract_surface(mesh).unwrap();
    let lookup = gfem_core::case::FaceLookup::new(mesh, &surface);
    gfem_core::assemble::assemble(mesh, case, &lookup, false)
        .unwrap()
        .matrix
}

#[test]
fn the_v_cycle_is_a_symmetric_positive_definite_operator() {
    // CG の前処理として使うための必要条件。ここが崩れると収束が保証されない
    let mesh = mesh(10, 7, 2);
    let case = case_with(Preconditioner::Amg, Backend::CpuSerial);
    let matrix = stiffness(&mesh, &case);
    let n = matrix.n_rows();

    let hierarchy = amg::setup(&matrix, &AmgOptions::default()).unwrap();
    let backend = CpuSerial;
    let mut device = DeviceHierarchy::new(&backend, &hierarchy).unwrap();

    let mut apply = |v: &[f64]| -> Vec<f64> {
        let input = backend.upload(v).unwrap();
        let mut out = backend.zeros(n).unwrap();
        device.apply(&backend, &input, &mut out).unwrap();
        out
    };

    for seed in 0..4 {
        let r: Vec<f64> = (0..n)
            .map(|i| ((i * 7 + seed * 13) as f64 * 0.37).sin())
            .collect();
        let s: Vec<f64> = (0..n)
            .map(|i| ((i * 3 + seed * 11) as f64 * 0.21).cos())
            .collect();
        let mr = apply(&r);
        let ms = apply(&s);

        // 対称性 ⟨M r, s⟩ = ⟨r, M s⟩
        let lhs: f64 = mr.iter().zip(&s).map(|(a, b)| a * b).sum();
        let rhs: f64 = r.iter().zip(&ms).map(|(a, b)| a * b).sum();
        let scale = lhs.abs().max(rhs.abs()).max(1e-300);
        assert!(
            (lhs - rhs).abs() < 1e-9 * scale,
            "対称でない（seed {seed}）: {lhs} vs {rhs}"
        );

        // 正定値性 ⟨M r, r⟩ > 0
        let quad: f64 = mr.iter().zip(&r).map(|(a, b)| a * b).sum();
        assert!(quad > 0.0, "正定値でない（seed {seed}）: {quad}");
    }
}

#[test]
fn the_hierarchy_does_not_depend_on_the_backend() {
    // 階層はホスト側だけで作るので、バックエンドに依存しようがない。
    // それでも「依存させない」ことを明示的に固定しておく
    let mesh = mesh(10, 7, 2);
    let case = case_with(Preconditioner::Amg, Backend::CpuSerial);
    let matrix = stiffness(&mesh, &case);
    let a = amg::setup(&matrix, &AmgOptions::default()).unwrap();
    let b = amg::setup(&matrix, &AmgOptions::default()).unwrap();
    assert_eq!(a.n_levels(), b.n_levels());
    for (x, y) in a.levels.iter().zip(&b.levels) {
        assert_eq!(x.a.values(), y.a.values());
        assert_eq!(x.omega, y.omega);
        match (&x.transfer, &y.transfer) {
            (Some(p), Some(q)) => {
                assert_eq!(p.p.values(), q.p.values());
                assert_eq!(p.r.values(), q.r.values());
            }
            (None, None) => {}
            _ => panic!("最粗レベルの位置が違う"),
        }
    }
}

#[test]
fn amg_and_jacobi_reach_the_same_answer() {
    let mesh = mesh(14, 10, 2);
    let jacobi = solve(
        &mesh,
        &case_with(Preconditioner::Jacobi, Backend::CpuSerial),
        None,
    )
    .unwrap();
    let amg = solve(
        &mesh,
        &case_with(Preconditioner::Amg, Backend::CpuSerial),
        None,
    )
    .unwrap();

    assert!(amg.solver.converged, "{:?}", amg.solver.failure_reason());
    assert_eq!(amg.solver.preconditioner, "amg");
    assert!(amg.solver.amg.is_some());

    // 前処理を変えても解いている方程式は同じ。収束判定 1e-10 の範囲で一致する
    let scale = jacobi
        .potential
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()))
        .max(1e-300);
    let worst = jacobi
        .potential
        .iter()
        .zip(&amg.potential)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(worst < 1e-7 * scale, "電位の差 {worst}（振幅 {scale}）");

    let r_jacobi = jacobi.conservation.impedance;
    let r_amg = amg.conservation.impedance;
    assert!(
        (r_jacobi - r_amg).abs() < 1e-7 * r_jacobi.abs(),
        "等価抵抗 {r_jacobi} vs {r_amg}"
    );
    assert!(
        amg.conservation.is_clean(),
        "保存則: {:?}",
        amg.conservation
    );
}

#[test]
fn amg_needs_far_fewer_iterations_than_jacobi() {
    // AMG を入れる唯一の理由。ここが逆転していたら入れる意味がない
    let mesh = mesh(18, 12, 3);
    let jacobi = solve(
        &mesh,
        &case_with(Preconditioner::Jacobi, Backend::CpuSerial),
        None,
    )
    .unwrap();
    let amg = solve(
        &mesh,
        &case_with(Preconditioner::Amg, Backend::CpuSerial),
        None,
    )
    .unwrap();
    println!(
        "節点 {}: Jacobi {} 反復 / AMG {} 反復（レベル {:?}）",
        mesh.nodes.len(),
        jacobi.solver.iterations,
        amg.solver.iterations,
        amg.solver.amg.as_ref().map(|i| i.levels.clone())
    );
    assert!(
        amg.solver.iterations * 4 < jacobi.solver.iterations,
        "AMG {} 反復 / Jacobi {} 反復——4 倍も減っていない",
        amg.solver.iterations,
        jacobi.solver.iterations
    );
}

#[test]
fn the_iteration_count_barely_grows_with_the_mesh() {
    // Jacobi は反復回数が √κ ∝ N^(1/3) で増える。AMG はここが頭打ちになるはず
    let mut counts = Vec::new();
    // 最粗レベルの大きさ（既定 400 節点）より十分大きいところから始める。
    // それ未満だと階層が 1 段しかできず、最初から直接解くことになって比較にならない
    for refine in [2usize, 3, 4] {
        // 3 方向とも細かくする（z だけ割っても規模が伸びない）
        let mesh = mesh(6 * refine, 4 * refine, refine);
        let analysis = solve(
            &mesh,
            &case_with(Preconditioner::Amg, Backend::CpuSerial),
            None,
        )
        .unwrap();
        assert!(
            analysis.solver.converged,
            "{:?}",
            analysis.solver.failure_reason()
        );
        println!(
            "節点 {:>7}: {} 反復（レベル数 {}）",
            mesh.nodes.len(),
            analysis.solver.iterations,
            analysis.solver.amg.as_ref().unwrap().levels.len()
        );
        counts.push((mesh.nodes.len(), analysis.solver.iterations));
    }
    let (n_small, it_small) = counts[0];
    let (n_large, it_large) = counts[counts.len() - 1];
    assert!(n_large > 6 * n_small, "規模差が足りない");
    // 節点数が 6 倍以上になっても反復回数は 1.5 倍未満に収まるはず。
    // Jacobi なら N^(1/3) で増えるので同じ差で 1.8 倍になる
    assert!(
        2 * it_large < 3 * it_small,
        "{n_small} 節点 {it_small} 反復 → {n_large} 節点 {it_large} 反復。増えすぎ"
    );
}

#[test]
fn amg_handles_a_large_conductivity_contrast() {
    // 骨 0.02 と筋肉 0.35 で 17 倍。SPEC §8 M6 が挙げているケース
    let mesh = mesh(14, 10, 2);
    let mut case = case_with(Preconditioner::Amg, Backend::CpuSerial);
    for (pid, sigma) in [(1u32, 0.35), (2, 0.02), (3, 0.35)] {
        case.set_conductivity(pid, sigma).unwrap();
    }
    let amg = solve(&mesh, &case, None).unwrap();
    assert!(amg.solver.converged, "{:?}", amg.solver.failure_reason());

    let mut jacobi_case = case.clone();
    jacobi_case.solver.preconditioner = Preconditioner::Jacobi;
    let jacobi = solve(&mesh, &jacobi_case, None).unwrap();
    println!(
        "コントラスト 17 倍: Jacobi {} 反復 / AMG {} 反復",
        jacobi.solver.iterations, amg.solver.iterations
    );
    assert!(amg.solver.iterations < jacobi.solver.iterations);

    let scale = jacobi
        .potential
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()))
        .max(1e-300);
    let worst = jacobi
        .potential
        .iter()
        .zip(&amg.potential)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(worst < 1e-7 * scale, "電位の差 {worst}");
}

#[test]
fn serial_and_parallel_agree_bitwise_with_amg() {
    // CLAUDE.md 設計原則 1。AMG の V サイクルもバックエンドの演算だけでできているので、
    // 逐次と並列で 1 ビットも変わってはいけない
    let mesh = mesh(16, 11, 2);
    let serial = solve(
        &mesh,
        &case_with(Preconditioner::Amg, Backend::CpuSerial),
        None,
    )
    .unwrap();
    let parallel = solve(
        &mesh,
        &case_with(Preconditioner::Amg, Backend::CpuParallel),
        None,
    )
    .unwrap();
    assert_identical(&serial, &parallel);
    assert_eq!(serial.solver.amg, parallel.solver.amg, "AMG の階層");
}

#[test]
fn repeated_amg_runs_are_identical() {
    let mesh = mesh(12, 8, 2);
    let case = case_with(Preconditioner::Amg, Backend::CpuSerial);
    let first = solve(&mesh, &case, None).unwrap();
    for _ in 0..3 {
        let again = solve(&mesh, &case, None).unwrap();
        assert_identical(&first, &again);
    }
}

fn assert_identical(a: &Analysis, b: &Analysis) {
    assert_eq!(a.potential, b.potential, "電位が一致しません");
    assert_eq!(a.fields.electric_field, b.fields.electric_field, "E");
    assert_eq!(a.fields.current_density, b.fields.current_density, "J");
    assert_eq!(a.solver.iterations, b.solver.iterations, "反復回数");
    assert_eq!(a.solver.history, b.solver.history, "収束履歴");
    assert_eq!(
        a.conservation.impedance, b.conservation.impedance,
        "等価抵抗"
    );
}

/// バックエンドの trait を使っていることの型チェック（未使用警告よけも兼ねる）。
#[test]
fn the_parallel_backend_can_host_the_hierarchy_too() {
    let mesh = mesh(8, 6, 1);
    let case = case_with(Preconditioner::Amg, Backend::CpuParallel);
    let matrix = stiffness(&mesh, &case);
    let hierarchy = amg::setup(&matrix, &AmgOptions::default()).unwrap();
    let backend = CpuParallel;
    let device = DeviceHierarchy::new(&backend, &hierarchy).unwrap();
    assert_eq!(device.host().n_levels(), hierarchy.n_levels());
}

#[test]
#[ignore = "計測用。--ignored で実行する"]
fn measure_amg_against_jacobi() {
    // AMG は「1 反復を重くする代わりに反復回数を減らす」取引なので、
    // どこから元が取れるかは実測でしか決まらない
    use std::time::Instant;
    println!(
        "{:>9} {:>7} {:>9} {:>7} {:>9} {:>8} {:>7} {:>6}",
        "節点", "J 反復", "J 時間[s]", "A 反復", "A 時間[s]", "階層[s]", "倍率", "レベル"
    );
    for (nx, ny, refine) in [
        (20, 14, 1),
        (30, 20, 2),
        (50, 34, 2),
        (80, 54, 3),
        (140, 94, 4),
    ] {
        let m = mesh(nx, ny, refine);

        let t = Instant::now();
        let jacobi = solve(
            &m,
            &case_with(Preconditioner::Jacobi, Backend::CpuParallel),
            None,
        )
        .unwrap();
        let jacobi_time = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let amg = solve(
            &m,
            &case_with(Preconditioner::Amg, Backend::CpuParallel),
            None,
        )
        .unwrap();
        let amg_time = t.elapsed().as_secs_f64();

        let info = amg.solver.amg.as_ref().unwrap();
        println!(
            "{:>9} {:>7} {:>9.3} {:>7} {:>9.3} {:>8.3} {:>7.2} {:>6}",
            m.nodes.len(),
            jacobi.solver.iterations,
            jacobi.timings.solve,
            amg.solver.iterations,
            amg.timings.solve,
            amg.solver.setup_seconds,
            jacobi.timings.solve / amg.timings.solve.max(1e-12),
            info.levels.len(),
        );
        let _ = (jacobi_time, amg_time);

        // どの規模でも同じ答えにたどり着くこと
        let scale = jacobi.potential.iter().fold(0.0f64, |x, v| x.max(v.abs()));
        let worst = jacobi
            .potential
            .iter()
            .zip(&amg.potential)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(worst < 1e-7 * scale, "電位の差 {worst}");
    }
}
