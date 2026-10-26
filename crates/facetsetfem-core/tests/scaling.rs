//! 規模を変えたときの所要時間と、大規模での決定性の確認（SPEC §8 M4）。
//!
//! M4 の目標は「100 万節点クラスが快適に回る」こと。小さなテストでは
//! 並列化の効きも、大規模でのビット単位一致も確かめられないので、
//! 重い計測を分けてここに置く。
//!
//! ```text
//! cargo test --release -p facetsetfem-core --test scaling -- --ignored --nocapture
//! ```
//!
//! # 実測（20 コア / RTX A6000 のワークステーション、release ビルド）
//!
//! | 節点 | 四面体 | 反復 | 逐次 [s] | 並列 [s] | 倍率 |
//! |---|---|---|---|---|---|
//! | 1,056 | 4,620 | 123 | 0.007 | 0.029 | 0.23 |
//! | 3,780 | 18,480 | 141 | 0.027 | 0.064 | 0.42 |
//! | 14,973 | 79,200 | 232 | 0.148 | 0.136 | 1.09 |
//! | 41,055 | 224,400 | 280 | 0.491 | 0.309 | 1.59 |
//! | 151,470 | 855,360 | 421 | 2.959 | 0.918 | 3.22 |
//! | 602,775 | 3,474,240 | 633 | 18.170 | 4.326 | 4.20 |
//!
//! 60 万節点の内訳（並列）はアセンブリ 0.56 秒・CG 2.19 秒・ポスト処理 0.24 秒で、
//! **CG 単体では 7.1 倍**速くなっている。全体の倍率がそれより低いのは、
//! アセンブリが行ごとの担当ぶん要素剛性を重複計算していることと
//! （`assemble::adjacency` の説明を参照）、ポスト処理がまだ逐次なため。
//! いずれも全体の 2 割弱なので、まずは CG の速度を優先している。
//!
//! `Backend::Auto` のしきい値（2 万節点）はこの表の損益分岐点から決めた。

mod common;

use common::{BOTTOM_PID, Layer, TOP_PID, slab};
use facetsetfem_core::case::{
    Backend, Case, Electrode, Preconditioner, ReferencePotential, SolverOptions,
};
use facetsetfem_core::{Mesh, solve};
use std::time::Instant;

const CURRENT: f64 = 2.0e-3;

fn mesh(nx: usize, ny: usize, refine: usize) -> Mesh {
    slab(
        0.060,
        0.040,
        nx,
        ny,
        &[
            Layer {
                pid: 1,
                thickness: 0.002,
                divisions: 2 * refine,
            },
            Layer {
                pid: 2,
                thickness: 0.004,
                divisions: 3 * refine,
            },
            Layer {
                pid: 3,
                thickness: 0.014,
                divisions: 6 * refine,
            },
        ],
    )
}

fn case_with(backend: Backend) -> Case {
    let mut case = Case::new();
    for (pid, sigma) in [(1u32, 0.10), (2, 0.04), (3, 0.35)] {
        case.set_conductivity(pid, sigma).unwrap();
    }
    case.add_electrode(Electrode::anode("top", CURRENT).with_shell_pids([TOP_PID]));
    case.add_electrode(Electrode::cathode("bottom", CURRENT).with_shell_pids([BOTTOM_PID]));
    case.solver = SolverOptions {
        tolerance: 1.0e-10,
        max_iterations: 100_000,
        preconditioner: Preconditioner::Jacobi,
        backend,
        ..Default::default()
    };
    case.reference = ReferencePotential::CathodeMean;
    case
}

#[test]
#[ignore = "計測用。--ignored で実行する"]
fn measure_parallel_speedup() {
    println!(
        "{:>10} {:>11} {:>6} {:>10} {:>10} {:>7}",
        "節点", "四面体", "反復", "逐次[s]", "並列[s]", "倍率"
    );
    for (nx, ny, refine) in [
        (10, 7, 1),
        (20, 14, 1),
        (30, 20, 2),
        (50, 34, 2),
        (80, 54, 3),
    ] {
        let m = mesh(nx, ny, refine);

        let t = Instant::now();
        let serial = solve(&m, &case_with(Backend::CpuSerial), None).unwrap();
        let serial_time = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let parallel = solve(&m, &case_with(Backend::CpuParallel), None).unwrap();
        let parallel_time = t.elapsed().as_secs_f64();

        // 規模が大きくても 1 ビットも変わらないこと（CLAUDE.md 設計原則 1）
        assert_eq!(serial.potential, parallel.potential, "電位が一致しません");
        assert_eq!(serial.solver.history, parallel.solver.history);

        println!(
            "{:>10} {:>11} {:>6} {:>10.3} {:>10.3} {:>7.2}",
            m.nodes.len(),
            m.n_solid_elements(),
            serial.solver.iterations,
            serial_time,
            parallel_time,
            serial_time / parallel_time.max(1e-12)
        );
    }
}

#[test]
#[ignore = "重い（60 万節点）。--ignored で実行する"]
fn large_mesh_stays_deterministic_and_fast() {
    // M4 の目標「100 万節点クラスが快適に回る」の確認
    let m = mesh(140, 94, 4);
    println!("節点 {}, 四面体 {}", m.nodes.len(), m.n_solid_elements());
    assert!(m.nodes.len() > 500_000);

    let case = case_with(Backend::CpuParallel);
    let t = Instant::now();
    let a = solve(&m, &case, None).unwrap();
    let elapsed = t.elapsed().as_secs_f64();
    println!(
        "並列 {elapsed:.2} 秒（assemble {:.2} / solve {:.2} / post {:.2}）, {} 反復",
        a.timings.assemble, a.timings.solve, a.timings.post, a.solver.iterations
    );

    assert!(a.solver.converged, "{:?}", a.solver.failure_reason());
    assert!(a.conservation.is_clean(), "{:?}", a.conservation.warnings);

    // 2 回まわしてビット単位で一致すること
    let b = solve(&m, &case, None).unwrap();
    assert_eq!(a.potential, b.potential);
    assert_eq!(a.fields.electric_field, b.fields.electric_field);

    // 一様場の解析解と一致すること（層状スラブなので厳密）
    let area = 0.060 * 0.040;
    let expected_r: f64 = [(0.002, 0.10), (0.004, 0.04), (0.014, 0.35)]
        .iter()
        .map(|&(t, s)| t / (s * area))
        .sum();
    let error = (a.impedance() - expected_r).abs() / expected_r;
    println!(
        "R = {} Ω（期待 {expected_r} Ω、相対誤差 {error:.2e}）",
        a.impedance()
    );
    assert!(error < 1e-7, "R の相対誤差 {error}");
}
