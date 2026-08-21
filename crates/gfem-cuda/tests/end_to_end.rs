//! GPU で実際に問題を解き、CPU と 1 ビットも変わらないことを確かめる（SPEC §8 M5）。
//!
//! `bitwise.rs` が演算 1 つずつを見るのに対し、こちらは
//! **アセンブリから CG・ポスト処理まで通した結果**を突き合わせる。
//! 反復のたびに丸めの差は増幅されるので、数百反復して一致するなら本物。
//!
//! GPU が無い環境では黙って通す（設計原則 5）。

use gfem_core::case::{
    Backend, Case, Electrode, Preconditioner, ReferencePotential, SolverOptions,
};
use gfem_core::io::nastran::read_nastran_from;
use gfem_core::solver::{BackendSolver, CpuParallel, CpuSerial};
use gfem_core::units::LengthUnit;
use gfem_core::{Analysis, Mesh, solve_with};
use gfem_cuda::CudaBackend;
use std::path::Path;
use std::time::Instant;

const CURRENT: f64 = 2.0e-3;
const LX: f64 = 0.060;
const LY: f64 = 0.040;
const TOP_PID: u32 = 101;
const BOTTOM_PID: u32 = 201;
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

/// nx × ny × Σ分割 の格子を Kuhn 分割した 3 層スラブ。
fn slab(nx: usize, ny: usize, refine: usize) -> Mesh {
    let mut z = vec![0.0f64];
    for (_pid, thickness, _sigma, divisions) in LAYERS {
        let base = *z.last().unwrap();
        let divisions = divisions * refine;
        for i in 1..=divisions {
            z.push(base + thickness * i as f64 / divisions as f64);
        }
    }
    let nz = z.len() - 1;
    let node = |i: usize, j: usize, k: usize| 1 + i + (nx + 1) * (j + (ny + 1) * k);

    let mut deck = String::new();
    for (pid, _t, _s, _d) in LAYERS {
        deck.push_str(&format!("PSOLID,{pid},{pid}\n"));
    }
    deck.push_str(&format!("PSHELL,{TOP_PID},{TOP_PID},0.1\n"));
    deck.push_str(&format!("PSHELL,{BOTTOM_PID},{BOTTOM_PID},0.1\n"));
    for (k, &zk) in z.iter().enumerate() {
        for j in 0..=ny {
            for i in 0..=nx {
                let x = LX * i as f64 / nx as f64;
                let y = LY * j as f64 / ny as f64;
                deck.push_str(&format!("GRID,{},,{x},{y},{zk}\n", node(i, j, k)));
            }
        }
    }

    // z 方向の段 k がどの層に属するか
    let layer_of = |k: usize| {
        let mut rest = k;
        for (pid, _t, _s, d) in LAYERS {
            let d = d * refine;
            if rest < d {
                return pid;
            }
            rest -= d;
        }
        unreachable!()
    };

    let mut eid = 1;
    for k in 0..nz {
        let pid = layer_of(k);
        for j in 0..ny {
            for i in 0..nx {
                let c = [
                    node(i, j, k),
                    node(i + 1, j, k),
                    node(i, j + 1, k),
                    node(i + 1, j + 1, k),
                    node(i, j, k + 1),
                    node(i + 1, j, k + 1),
                    node(i, j + 1, k + 1),
                    node(i + 1, j + 1, k + 1),
                ];
                for t in [
                    [0, 1, 3, 7],
                    [0, 1, 5, 7],
                    [0, 2, 3, 7],
                    [0, 2, 6, 7],
                    [0, 4, 5, 7],
                    [0, 4, 6, 7],
                ] {
                    deck.push_str(&format!(
                        "CTETRA,{eid},{pid},{},{},{},{}\n",
                        c[t[0]], c[t[1]], c[t[2]], c[t[3]]
                    ));
                    eid += 1;
                }
            }
        }
    }
    for (k, pid) in [(nz, TOP_PID), (0, BOTTOM_PID)] {
        for j in 0..ny {
            for i in 0..nx {
                let q = [
                    node(i, j, k),
                    node(i + 1, j, k),
                    node(i + 1, j + 1, k),
                    node(i, j + 1, k),
                ];
                deck.push_str(&format!("CTRIA3,{eid},{pid},{},{},{}\n", q[0], q[1], q[2]));
                eid += 1;
                deck.push_str(&format!("CTRIA3,{eid},{pid},{},{},{}\n", q[0], q[2], q[3]));
                eid += 1;
            }
        }
    }
    read_nastran_from(deck.as_bytes(), Path::new("<test>"), LengthUnit::Meter)
        .unwrap()
        .0
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
    let needed = gfem_cuda::required_memory(n, mesh.n_solid_elements() * 8);
    println!("VRAM 見積り {:.1} GB", needed as f64 / 1e9);
    assert!(
        gfem_cuda::can_use_gpu(n, None).is_ok(),
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
