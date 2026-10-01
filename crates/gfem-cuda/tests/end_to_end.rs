//! GPU で実際に問題を解き、CPU と 1 ビットも変わらないことを確かめる（SPEC §8 M5）。
//!
//! `bitwise.rs` が演算 1 つずつを見るのに対し、こちらは
//! **アセンブリから CG・ポスト処理まで通した結果**を突き合わせる。
//! 反復のたびに丸めの差は増幅されるので、数百反復して一致するなら本物。
//!
//! GPU が無い環境では黙って通す（設計原則 5）。

// gfem-core の結合テストと同じメッシュ生成を使う（別クレートなので path で読み込む）
#[path = "../../gfem-core/tests/common/mod.rs"]
mod common;

use common::{BOTTOM_PID, Layer, TOP_PID};
use gfem_core::case::{
    Backend, Case, Electrode, Preconditioner, ReferencePotential, SolverOptions,
};
use gfem_core::solver::{BackendSolver, CpuParallel, CpuSerial};
use gfem_core::{Analysis, Mesh, solve_with};
use gfem_cuda::CudaBackend;
use std::time::Instant;

const CURRENT: f64 = 2.0e-3;
const LX: f64 = 0.060;
const LY: f64 = 0.040;
/// 層構成（PID, 厚さ [m], σ [S/m], 分割数）
const LAYERS: [(u32, f64, f64, usize); 3] = [
    (1, 0.002, 0.10, 2),
    (2, 0.004, 0.04, 3),
    (3, 0.014, 0.35, 6),
];

fn gpu() -> Option<CudaBackend> {
    match CudaBackend::new(0) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("GPU が使えないので飛ばします: {e}");
            None
        }
    }
}

macro_rules! gpu_or_skip {
    () => {
        match gpu() {
            Some(b) => b,
            None => return,
        }
    };
}

/// nx × ny × Σ分割 の格子を Kuhn 分割した 3 層スラブ（z の分割数は `refine` 倍）。
fn slab(nx: usize, ny: usize, refine: usize) -> Mesh {
    let layers: Vec<Layer> = LAYERS
        .iter()
        .map(|&(pid, thickness, _sigma, divisions)| Layer {
            pid,
            thickness,
            divisions: divisions * refine,
        })
        .collect();
    common::slab(LX, LY, nx, ny, &layers)
}

fn case_with(backend: Backend) -> Case {
    let mut case = Case::new();
    for (pid, _t, sigma, _d) in LAYERS {
        case.set_conductivity(pid, sigma).unwrap();
    }
    case.add_electrode(Electrode::anode("top", CURRENT).with_shell_pids([TOP_PID]));
    case.add_electrode(Electrode::cathode("bottom", CURRENT).with_shell_pids([BOTTOM_PID]));
    case.solver = SolverOptions {
        tolerance: 1.0e-12,
        max_iterations: 100_000,
        preconditioner: Preconditioner::Jacobi,
        backend,
        ..Default::default()
    };
    case.reference = ReferencePotential::CathodeMean;
    case
}

fn assert_identical(a: &Analysis, b: &Analysis) {
    assert_eq!(a.solver.iterations, b.solver.iterations, "反復回数");
    assert_eq!(a.solver.history, b.solver.history, "収束履歴");
    assert_eq!(a.potential, b.potential, "電位");
    assert_eq!(a.fields.electric_field, b.fields.electric_field, "E");
    assert_eq!(a.fields.current_density, b.fields.current_density, "J");
    assert_eq!(
        a.conservation.impedance, b.conservation.impedance,
        "等価抵抗"
    );
    assert_eq!(
        a.conservation.power_from_field, b.conservation.power_from_field,
        "消費電力"
    );
}

#[test]
fn gpu_and_cpu_solve_bitwise_identically() {
    let gpu = gpu_or_skip!();
    let mesh = slab(20, 14, 1);
    println!(
        "節点 {}, 四面体 {}",
        mesh.nodes.len(),
        mesh.n_solid_elements()
    );

    let case = case_with(Backend::CpuSerial);
    let serial = solve_with(&mesh, &case, None, None).unwrap();
    let parallel = solve_with(
        &mesh,
        &case,
        None,
        Some(&BackendSolver(CpuParallel) as &dyn gfem_core::LinearSolver),
    )
    .unwrap();
    let cuda = solve_with(&mesh, &case, None, Some(&BackendSolver(gpu))).unwrap();

    println!("{} 反復で収束", cuda.solver.iterations);
    assert!(
        cuda.solver.iterations > 100,
        "反復が少なすぎて検証にならない"
    );
    assert_eq!(cuda.solver.backend, "cuda");

    // CLAUDE.md 設計原則 1: 3 つのバックエンドが 1 ビットも違わないこと
    assert_identical(&serial, &parallel);
    assert_identical(&serial, &cuda);
}

#[test]
fn gpu_and_cpu_agree_bitwise_with_complete_electrodes() {
    // 完全電極モデル（SPEC §2.5）: 射影なしの CG・前処理の使い回し・重ね合わせも
    // バックエンドの演算とホストの逐次計算だけなので、1 ビットも変わってはいけない。
    // 陰極を 2 枚にして、右辺を 2 本（前処理を使い回す経路）にする
    let solver = BackendSolver(gpu_or_skip!());
    let xs = common::linear_grid(0.0, LX, 20);
    let ys = common::linear_grid(0.0, LY, 14);
    let zs = common::linear_grid(0.0, 0.02, 11);
    let mesh = common::box_mesh(
        &xs,
        &ys,
        &zs,
        &[3; 11],
        &[
            common::Patch {
                pid: TOP_PID,
                x: [0.02, 0.04],
                y: [0.01, 0.03],
                top: true,
            },
            common::Patch {
                pid: 401,
                x: [0.0, 0.025],
                y: [0.0, LY],
                top: false,
            },
            common::Patch {
                pid: 402,
                x: [0.035, LX],
                y: [0.0, LY],
                top: false,
            },
        ],
    );
    for preconditioner in [Preconditioner::Jacobi, Preconditioner::Amg] {
        let mut case = Case::new();
        case.set_conductivity(3, 0.35).unwrap();
        case.add_electrode(
            Electrode::anode("A", CURRENT)
                .with_shell_pids([TOP_PID])
                .with_contact_impedance(1.0e-3),
        );
        case.add_electrode(
            Electrode::cathode("C1", 0.5 * CURRENT)
                .with_shell_pids([401])
                .with_contact_impedance(2.0e-3),
        );
        case.add_electrode(
            Electrode::cathode("C2", 0.5 * CURRENT)
                .with_shell_pids([402])
                .with_contact_impedance(2.0e-3),
        );
        case.solver = SolverOptions {
            tolerance: 1.0e-12,
            max_iterations: 100_000,
            preconditioner,
            backend: Backend::CpuSerial,
            ..Default::default()
        };
        let serial = solve_with(&mesh, &case, None, None).unwrap();
        let cuda = solve_with(&mesh, &case, None, Some(&solver)).unwrap();
        assert_eq!(cuda.solver.backend, "cuda");
        assert_eq!(serial.solves.len(), 2);
        assert_eq!(
            serial.solves, cuda.solves,
            "{preconditioner:?}: 右辺ごとの反復"
        );
        assert_identical(&serial, &cuda);
        let us: Vec<_> = serial
            .electrodes
            .iter()
            .map(|e| e.terminal_potential)
            .collect();
        let uc: Vec<_> = cuda
            .electrodes
            .iter()
            .map(|e| e.terminal_potential)
            .collect();
        assert_eq!(us, uc, "{preconditioner:?}: 板の電位");
    }
}

#[test]
fn gpu_and_cpu_agree_bitwise_with_amg_too() {
    // AMG の V サイクルはバックエンドの演算しか使わないので、
    // CPU 逐次・CPU 並列・GPU で 1 ビットも変わってはいけない（CLAUDE.md 設計原則 1）。
    // 補間・制限は矩形の SpMV、最粗レベルはホストでの密分解を経由する——
    // どちらも M5 の経路には無かったので、ここで初めて通る道になる
    let gpu = gpu_or_skip!();
    let mesh = slab(20, 14, 1);
    let mut case = case_with(Backend::CpuSerial);
    case.solver.preconditioner = Preconditioner::Amg;

    let serial = solve_with(&mesh, &case, None, None).unwrap();
    let parallel = solve_with(
        &mesh,
        &case,
        None,
        Some(&BackendSolver(CpuParallel) as &dyn gfem_core::LinearSolver),
    )
    .unwrap();
    let cuda = solve_with(&mesh, &case, None, Some(&BackendSolver(gpu))).unwrap();

    let info = cuda.solver.amg.as_ref().expect("AMG の情報が無い");
    println!(
        "節点 {}: AMG {} 反復（レベル {:?}, 作用素複雑度 {:.2}）",
        mesh.nodes.len(),
        cuda.solver.iterations,
        info.levels,
        info.operator_complexity
    );
    assert!(info.levels.len() >= 2, "階層が 1 段しかない");
    assert!(info.direct_coarse_solve, "最粗レベルを密分解できていない");
    assert_eq!(cuda.solver.preconditioner, "amg");

    assert_identical(&serial, &parallel);
    assert_identical(&serial, &cuda);
    assert_eq!(serial.solver.amg, cuda.solver.amg, "階層の要約");

    // Jacobi と同じ答えに、はるかに少ない反復でたどり着いていること
    let jacobi = solve_with(&mesh, &case_with(Backend::CpuSerial), None, None).unwrap();
    println!(
        "Jacobi {} 反復 / AMG {} 反復",
        jacobi.solver.iterations, cuda.solver.iterations
    );
    assert!(cuda.solver.iterations * 4 < jacobi.solver.iterations);
    let scale = jacobi.potential.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let worst = jacobi
        .potential
        .iter()
        .zip(&cuda.potential)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(worst < 1e-8 * scale, "電位の差 {worst}");
}

#[test]
fn gpu_result_matches_the_analytic_solution() {
    // 一次元問題なので層の直列抵抗が厳密解になる
    let gpu = gpu_or_skip!();
    let mesh = slab(10, 7, 1);
    let analysis = solve_with(
        &mesh,
        &case_with(Backend::CpuSerial),
        None,
        Some(&BackendSolver(gpu)),
    )
    .unwrap();

    assert!(analysis.solver.converged);
    assert!(
        analysis.conservation.is_clean(),
        "{:?}",
        analysis.conservation.warnings
    );
    let area = LX * LY;
    let expected: f64 = LAYERS
        .iter()
        .map(|&(_p, t, sigma, _d)| t / (sigma * area))
        .sum();
    let error = (analysis.impedance() - expected).abs() / expected;
    println!(
        "R = {} Ω（期待 {expected} Ω、相対誤差 {error:.2e}）",
        analysis.impedance()
    );
    assert!(error < 1e-9, "R の相対誤差 {error}");
}

#[test]
fn gpu_and_cpu_agree_bitwise_on_second_order_elements() {
    // 2 次四面体（SPEC §8 M12）。求解の入口は後処理が揃うまで止めてあるので、組み立てと
    // CG を直接つなぐ。一様電流密度（射影 CG）と CEM（重ね合わせ）、Jacobi と AMG
    use gfem_core::assemble;
    use gfem_core::case::FaceLookup;
    use gfem_core::contact::solve_superposition;
    use gfem_core::mesh::{elevate, topology::extract_surface};
    use gfem_core::solver::{Gauge, LinearSolver};

    let gpu = gpu_or_skip!();
    let gpu = BackendSolver(gpu);
    let mesh = elevate(&slab(8, 6, 1)).unwrap();
    let surface = extract_surface(&mesh).unwrap();
    let lookup = FaceLookup::new(&mesh, &surface);
    let solve = |case: &Case, solver: &dyn LinearSolver| -> (Vec<f64>, Vec<usize>) {
        let system = assemble(&mesh, case, &lookup, false).unwrap();
        let options = case.solver;
        match system.gauge {
            Gauge::MeanZero => {
                let mut phi = vec![0.0; mesh.nodes.len()];
                let r = solver
                    .solve(&system.matrix, &system.load, &mut phi, &options, None)
                    .unwrap();
                assert!(r.converged);
                (phi, vec![r.iterations])
            }
            Gauge::NotNeeded => {
                let s = solve_superposition(
                    solver,
                    &system.matrix,
                    &system.load,
                    &system.electrodes,
                    &options,
                    None,
                )
                .unwrap();
                (
                    s.potential,
                    s.reports.iter().map(|r| r.iterations).collect(),
                )
            }
        }
    };
    for preconditioner in [Preconditioner::Jacobi, Preconditioner::Amg] {
        for complete in [false, true] {
            let mut case = case_with(Backend::CpuSerial);
            case.solver.preconditioner = preconditioner;
            if complete {
                for e in &mut case.electrodes {
                    *e = e.clone().with_contact_impedance(1.0e-4);
                }
            }
            let (cpu, cpu_its) = solve(&case, &BackendSolver(CpuSerial));
            let (cuda, cuda_its) = solve(&case, &gpu);
            println!("{preconditioner:?} CEM={complete}: 反復 {cpu_its:?}");
            assert_eq!(cpu_its, cuda_its, "{preconditioner:?} CEM={complete}");
            assert!(
                cpu.iter()
                    .zip(&cuda)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{preconditioner:?} CEM={complete}: 電位がビット単位で一致しない"
            );
        }
    }
}

#[test]
fn repeated_gpu_runs_are_identical() {
    let gpu = gpu_or_skip!();
    let mesh = slab(12, 8, 1);
    let case = case_with(Backend::CpuSerial);
    let solver = BackendSolver(gpu);
    let reference = solve_with(&mesh, &case, None, Some(&solver)).unwrap();
    for _ in 0..3 {
        assert_identical(
            &reference,
            &solve_with(&mesh, &case, None, Some(&solver)).unwrap(),
        );
    }
}

#[test]
#[ignore = "重い（1000 万節点・VRAM 数 GB・主記憶数十 GB）。--ignored で実行する"]
fn ten_million_nodes_fit_on_the_gpu() {
    // M5 の目標「1000 万節点が回る」の確認（SPEC §8）
    let gpu = gpu_or_skip!();
    let info = gpu.device().info().clone();
    println!(
        "GPU: {}（空き VRAM {:.1} GB）",
        info.name,
        info.free_memory as f64 / 1e9
    );

    let build = Instant::now();
    let mesh = slab(410, 273, 8);
    let n = mesh.nodes.len();
    println!(
        "節点 {n}, 四面体 {}（生成 {:.1} 秒）",
        mesh.n_solid_elements(),
        build.elapsed().as_secs_f64()
    );
    assert!(n >= 10_000_000, "節点数が足りません: {n}");

    // VRAM の見積りが実際に収まる範囲かを、確保する前に確かめる
    let needed = gfem_cuda::required_memory(
        n,
        mesh.n_solid_elements() * 8,
        gfem_core::case::Preconditioner::Jacobi,
    );
    println!("VRAM 見積り {:.1} GB", needed as f64 / 1e9);
    assert!(
        gfem_cuda::can_use_gpu(n, None, gfem_core::case::Preconditioner::Jacobi).is_ok(),
        "この規模を GPU に載せられないと判定されました"
    );

    let mut case = case_with(Backend::Cuda);
    // 1000 万節点では反復回数が増えるので、判定を M2 の検証より少し緩める
    case.solver.tolerance = 1.0e-9;

    let t = Instant::now();
    let analysis = solve_with(&mesh, &case, None, Some(&BackendSolver(gpu))).unwrap();
    let elapsed = t.elapsed().as_secs_f64();
    println!(
        "GPU {elapsed:.1} 秒（準備 {:.1} / アセンブリ {:.1} / CG {:.1} / ポスト {:.1}）, {} 反復",
        analysis.timings.prepare,
        analysis.timings.assemble,
        analysis.timings.solve,
        analysis.timings.post,
        analysis.solver.iterations
    );

    assert!(
        analysis.solver.converged,
        "{:?}",
        analysis.solver.failure_reason()
    );
    assert!(
        analysis.conservation.is_clean(),
        "{:?}",
        analysis.conservation.warnings
    );

    // 一次元問題の厳密解と一致すること
    let area = LX * LY;
    let expected: f64 = LAYERS
        .iter()
        .map(|&(_p, t, sigma, _d)| t / (sigma * area))
        .sum();
    let error = (analysis.impedance() - expected).abs() / expected;
    println!(
        "R = {} Ω（期待 {expected} Ω、相対誤差 {error:.2e}）",
        analysis.impedance()
    );
    assert!(error < 1e-6, "R の相対誤差 {error}");
}

#[test]
fn second_order_analyses_agree_bitwise_from_assembly_to_post_processing() {
    // 2 次四面体（SPEC §8 M12）を解析として後処理まで通し、CPU 逐次・CPU 並列・GPU で
    // E・J・電力・要素内の最大まで 1 ビットも変わらないこと（設計原則 1）
    let gpu = gpu_or_skip!();
    let gpu = BackendSolver(gpu);
    let mesh = gfem_core::mesh::elevate(&slab(8, 6, 1)).unwrap();
    for preconditioner in [Preconditioner::Jacobi, Preconditioner::Amg] {
        let mut case = case_with(Backend::CpuSerial);
        case.solver.preconditioner = preconditioner;
        let serial = solve_with(&mesh, &case, None, None).unwrap();
        let parallel = solve_with(
            &mesh,
            &case,
            None,
            Some(&BackendSolver(CpuParallel) as &dyn gfem_core::LinearSolver),
        )
        .unwrap();
        let cuda = solve_with(&mesh, &case, None, Some(&gpu)).unwrap();
        assert!(
            serial.conservation.is_clean(),
            "{:?}",
            serial.conservation.warnings
        );
        assert_identical(&serial, &parallel);
        assert_identical(&serial, &cuda);
        assert_eq!(serial.fields.j_peak, cuda.fields.j_peak, "要素内の最大");
        assert_eq!(
            serial.fields.power_density, cuda.fields.power_density,
            "電力密度"
        );
        println!(
            "{preconditioner:?}: 節点 {}、{} 反復",
            mesh.nodes.len(),
            cuda.solver.iterations
        );
    }
}

#[test]
#[ignore = "重い（2 次で 1000 万節点・VRAM 十数 GB・主記憶数十 GB）。--ignored で実行する"]
fn ten_million_second_order_nodes_on_the_gpu() {
    // 2 次四面体（SPEC §8 M12）の規模の確認。1 次の 134 万節点を 2 次に上げて約 1000 万節点
    let gpu = gpu_or_skip!();
    let t = Instant::now();
    let linear = slab(210, 140, 4);
    let mesh = gfem_core::mesh::elevate(&linear).unwrap();
    let n = mesh.nodes.len();
    println!(
        "1 次 {} 節点 → 2 次 {n} 節点、四面体 {}（生成と変換 {:.1} 秒）",
        linear.nodes.len(),
        mesh.n_solid_elements(),
        t.elapsed().as_secs_f64()
    );
    assert!(n >= 10_000_000, "節点数が足りません: {n}");
    drop(linear);
    assert!(
        gfem_cuda::can_use_gpu(n, Some(mesh.estimated_nnz()), Preconditioner::Amg).is_ok(),
        "この規模を GPU に載せられないと判定されました"
    );

    let mut case = case_with(Backend::Cuda);
    case.solver.tolerance = 1.0e-9;
    case.solver.preconditioner = Preconditioner::Amg;
    let t = Instant::now();
    let analysis = solve_with(&mesh, &case, None, Some(&BackendSolver(gpu))).unwrap();
    let elapsed = t.elapsed().as_secs_f64();
    let amg = analysis.solver.amg.as_ref().unwrap();
    println!(
        "GPU（AMG）{elapsed:.1} 秒（準備 {:.1} / アセンブリ {:.1} / CG {:.1} / ポスト {:.1}）、{} 反復、作用素複雑度 {:.2}、θ {}",
        analysis.timings.prepare,
        analysis.timings.assemble,
        analysis.timings.solve,
        analysis.timings.post,
        analysis.solver.iterations,
        amg.operator_complexity,
        amg.strength_threshold
    );
    assert!(
        analysis.solver.converged,
        "{:?}",
        analysis.solver.failure_reason()
    );
    assert!(
        analysis.conservation.is_clean(),
        "{:?}",
        analysis.conservation.warnings
    );

    // 一次元問題の厳密解（層の直列抵抗）と一致すること
    let area = LX * LY;
    let expected: f64 = LAYERS
        .iter()
        .map(|&(_p, t, sigma, _d)| t / (sigma * area))
        .sum();
    let error = (analysis.impedance() - expected).abs() / expected;
    println!(
        "R = {} Ω（期待 {expected} Ω、相対誤差 {error:.2e}）",
        analysis.impedance()
    );
    assert!(error < 1e-6, "R の相対誤差 {error}");
}

#[test]
#[ignore = "計測用。--ignored で実行する"]
fn measure_gpu_speedup() {
    let gpu = gpu_or_skip!();
    println!("GPU: {}", gpu.device().info().name);
    println!(
        "{:>10} {:>11} {:>6} {:>10} {:>10} {:>10} {:>7}",
        "節点", "四面体", "反復", "逐次[s]", "並列[s]", "GPU[s]", "倍率"
    );
    let solver = BackendSolver(gpu);
    for (nx, ny, refine) in [
        (20, 14, 1),
        (30, 20, 2),
        (50, 34, 2),
        (80, 54, 3),
        (114, 76, 3),
        (140, 94, 4),
    ] {
        let mesh = slab(nx, ny, refine);
        let case = case_with(Backend::CpuSerial);

        let t = Instant::now();
        let serial = solve_with(&mesh, &case, None, Some(&BackendSolver(CpuSerial))).unwrap();
        let serial_time = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let parallel = solve_with(&mesh, &case, None, Some(&BackendSolver(CpuParallel))).unwrap();
        let parallel_time = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let cuda = solve_with(&mesh, &case, None, Some(&solver)).unwrap();
        let cuda_time = t.elapsed().as_secs_f64();

        assert_identical(&serial, &parallel);
        assert_identical(&serial, &cuda);

        println!(
            "{:>10} {:>11} {:>6} {:>10.3} {:>10.3} {:>10.3} {:>7.2}",
            mesh.nodes.len(),
            mesh.n_solid_elements(),
            cuda.solver.iterations,
            serial_time,
            parallel_time,
            cuda_time,
            serial_time / cuda_time.max(1e-12)
        );
        println!(
            "           CG だけ: 逐次 {:.3} / 並列 {:.3} / GPU {:.3} 秒（{:.1} 倍）",
            serial.timings.solve,
            parallel.timings.solve,
            cuda.timings.solve,
            serial.timings.solve / cuda.timings.solve.max(1e-12)
        );
    }
}

#[test]
#[ignore = "計測用。--ignored で実行する"]
fn measure_amg_on_the_gpu() {
    // AMG の V サイクルは SpMV を何度も回すので、GPU の帯域がそのまま効く。
    // ただし階層の構築はホスト側の逐次寄りの処理で、そこが新しい律速になりうる
    let gpu = gpu_or_skip!();
    println!("GPU: {}", gpu.device().info().name);
    println!(
        "{:>10} {:>7} {:>10} {:>7} {:>10} {:>9} {:>7}",
        "節点", "J 反復", "J CG[s]", "A 反復", "A CG[s]", "階層[s]", "倍率"
    );
    let solver = BackendSolver(gpu);
    for (nx, ny, refine) in [(50, 34, 2), (80, 54, 3), (140, 94, 4), (200, 134, 5)] {
        let mesh = slab(nx, ny, refine);

        let jacobi = solve_with(&mesh, &case_with(Backend::Cuda), None, Some(&solver)).unwrap();

        let mut case = case_with(Backend::Cuda);
        case.solver.preconditioner = Preconditioner::Amg;
        let amg = solve_with(&mesh, &case, None, Some(&solver)).unwrap();

        let jacobi_cg = jacobi.timings.solve;
        let amg_cg = amg.timings.solve;
        println!(
            "{:>10} {:>7} {:>10.3} {:>7} {:>10.3} {:>9.3} {:>7.2}",
            mesh.nodes.len(),
            jacobi.solver.iterations,
            jacobi_cg,
            amg.solver.iterations,
            amg_cg,
            amg.solver.setup_seconds,
            jacobi_cg / amg_cg.max(1e-12)
        );
        if let Some(info) = &amg.solver.amg {
            println!(
                "           レベル {:?}, 作用素複雑度 {:.2}",
                info.levels, info.operator_complexity
            );
        }
    }
}

#[test]
#[ignore = "重い（1000 万節点・VRAM 十数 GB）。--ignored で実行する"]
fn ten_million_nodes_with_amg() {
    // M5 の 1000 万節点は Jacobi で 1423 反復・CG 17.5 秒だった。
    // AMG がその規模でも反復回数を抑えられるか、VRAM に収まるかを見る
    use std::time::Instant;
    let gpu = gpu_or_skip!();
    println!("GPU: {}", gpu.device().info().name);

    let build = Instant::now();
    let mesh = slab(410, 273, 8);
    let n = mesh.nodes.len();
    println!(
        "節点 {n}, 四面体 {}（生成 {:.1} 秒）",
        mesh.n_solid_elements(),
        build.elapsed().as_secs_f64()
    );
    assert!(n >= 10_000_000, "節点数が足りません: {n}");

    let mut case = case_with(Backend::Cuda);
    case.solver.tolerance = 1.0e-9;
    case.solver.preconditioner = Preconditioner::Amg;

    let t = Instant::now();
    let analysis = solve_with(&mesh, &case, None, Some(&BackendSolver(gpu))).unwrap();
    let elapsed = t.elapsed().as_secs_f64();
    let info = analysis.solver.amg.as_ref().unwrap();
    println!(
        "GPU + AMG {elapsed:.1} 秒（準備 {:.1} / アセンブリ {:.1} / CG {:.1}（うち階層 {:.1}） / ポスト {:.1}）, {} 反復",
        analysis.timings.prepare,
        analysis.timings.assemble,
        analysis.timings.solve,
        analysis.solver.setup_seconds,
        analysis.timings.post,
        analysis.solver.iterations
    );
    println!(
        "レベル {:?}, 作用素複雑度 {:.2}",
        info.levels, info.operator_complexity
    );

    assert!(
        analysis.solver.converged,
        "{:?}",
        analysis.solver.failure_reason()
    );
    assert!(
        analysis.conservation.is_clean(),
        "{:?}",
        analysis.conservation.warnings
    );
    // M5 の Jacobi は 1423 反復だった。桁で減っていること
    assert!(
        analysis.solver.iterations < 200,
        "{} 反復。AMG の効き目が出ていない",
        analysis.solver.iterations
    );

    let area = LX * LY;
    let expected: f64 = LAYERS
        .iter()
        .map(|&(_p, t, sigma, _d)| t / (sigma * area))
        .sum();
    let error = (analysis.impedance() - expected).abs() / expected;
    println!(
        "R = {} Ω（期待 {expected} Ω、相対誤差 {error:.2e}）",
        analysis.impedance()
    );
    assert!(error < 1e-6, "R の相対誤差 {error}");
}

#[test]
#[ignore = "重い（1000 万節点・VRAM 十数 GB）。--ignored で実行する"]
fn ten_million_nodes_with_complete_electrodes() {
    // 完全電極モデル（SPEC §2.5）で同じ 1000 万節点を解く。電極 2 枚とも CEM なら右辺は 1 本で、
    // 一様電流密度（ten_million_nodes_with_amg）と同じ手間に収まるはず。
    // 上下面が丸ごと電極なので 1 次元の直列抵抗 R = Σ t/(σA) + 2z/A と比べられる
    use std::time::Instant;
    let gpu = gpu_or_skip!();
    let mesh = slab(410, 273, 8);
    let n = mesh.nodes.len();
    assert!(n >= 10_000_000, "節点数が足りません: {n}");

    let z = 0.01; // Ω·m²
    let mut case = Case::new();
    for (pid, _t, sigma, _d) in LAYERS {
        case.set_conductivity(pid, sigma).unwrap();
    }
    case.add_electrode(
        Electrode::anode("top", CURRENT)
            .with_shell_pids([TOP_PID])
            .with_contact_impedance(z),
    );
    case.add_electrode(
        Electrode::cathode("bottom", CURRENT)
            .with_shell_pids([BOTTOM_PID])
            .with_contact_impedance(z),
    );
    case.solver = SolverOptions {
        tolerance: 1.0e-9,
        max_iterations: 100_000,
        preconditioner: Preconditioner::Amg,
        backend: Backend::Cuda,
        ..Default::default()
    };

    let t = Instant::now();
    let analysis = solve_with(&mesh, &case, None, Some(&BackendSolver(gpu))).unwrap();
    let elapsed = t.elapsed().as_secs_f64();
    println!(
        "GPU + AMG + CEM {elapsed:.1} 秒（準備 {:.1} / アセンブリ {:.1} / CG {:.1}（うち階層 {:.1}） / ポスト {:.1}）, {} 反復",
        analysis.timings.prepare,
        analysis.timings.assemble,
        analysis.timings.solve,
        analysis.solver.setup_seconds,
        analysis.timings.post,
        analysis.solver.iterations
    );
    assert_eq!(analysis.solves.len(), 1);
    assert!(
        analysis.conservation.is_clean(),
        "{:?}",
        analysis.conservation.warnings
    );
    let area = LX * LY;
    let body: f64 = LAYERS.iter().map(|&(_, t, s, _)| t / (s * area)).sum();
    let expected = body + 2.0 * z / area;
    let rel = (analysis.impedance() - expected).abs() / expected;
    println!(
        "R = {:.6} Ω（期待 {expected:.6} Ω、相対誤差 {rel:.2e}）",
        analysis.impedance()
    );
    assert!(rel < 1e-6, "R の相対誤差 {rel:.2e}");
}
