//! CPU 並列と CPU 逐次がビット単位で一致することの検証（SPEC §8 M4 / §10）。
//!
//! CLAUDE.md 設計原則 1 は「同一入力から CPU 逐次・CPU 並列・GPU のいずれでも
//! ビット単位で同じ結果」を要求している。並列化は速度のためだけに行い、
//! 答えを 1 ビットも変えてはいけない。
//!
//! 一致を保証している仕組みは 2 つ:
//!
//! - **総和**は固定ブロック分割の階層リダクション（`solver::reduce`）。
//!   どのスレッドがどのブロックを担当しても木の形が変わらない
//! - **アセンブリ**は行ごとに担当し、その行に触る要素を通し番号の昇順にたどる。
//!   逐次の要素ループと同じ加算順序になる

mod common;

use common::{BOTTOM_PID, Layer, TOP_PID, slab};
use facetsetfem_core::case::{
    Backend, Case, Electrode, Preconditioner, ReferencePotential, SolverOptions,
};
use facetsetfem_core::{Analysis, Mesh, solve};
use std::time::Instant;

const CURRENT: f64 = 2.0e-3;
const LX: f64 = 0.060;
const LY: f64 = 0.040;

fn mesh(nx: usize, ny: usize) -> Mesh {
    slab(
        LX,
        LY,
        nx,
        ny,
        &[
            Layer {
                pid: 1,
                thickness: 0.002,
                divisions: 2,
            },
            Layer {
                pid: 2,
                thickness: 0.004,
                divisions: 3,
            },
            Layer {
                pid: 3,
                thickness: 0.014,
                divisions: 6,
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
        tolerance: 1.0e-12,
        max_iterations: 50_000,
        preconditioner: Preconditioner::Jacobi,
        backend,
        ..Default::default()
    };
    case.reference = ReferencePotential::CathodeMean;
    case
}

fn assert_identical(a: &Analysis, b: &Analysis) {
    assert_eq!(a.potential, b.potential, "電位が一致しません");
    assert_eq!(a.fields.electric_field, b.fields.electric_field, "E");
    assert_eq!(a.fields.current_density, b.fields.current_density, "J");
    assert_eq!(a.fields.volume, b.fields.volume, "要素体積");
    assert_eq!(a.solver.iterations, b.solver.iterations, "反復回数");
    assert_eq!(a.solver.history, b.solver.history, "収束履歴");
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
fn serial_and_parallel_give_bitwise_identical_results() {
    let mesh = mesh(20, 14);
    println!(
        "節点 {}, 四面体 {}",
        mesh.nodes.len(),
        mesh.n_solid_elements()
    );

    let start = Instant::now();
    let serial = solve(&mesh, &case_with(Backend::CpuSerial), None).unwrap();
    let serial_time = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let parallel = solve(&mesh, &case_with(Backend::CpuParallel), None).unwrap();
    let parallel_time = start.elapsed().as_secs_f64();

    assert_eq!(serial.solver.backend, "cpu_serial");
    assert_eq!(parallel.solver.backend, "cpu_parallel");
    assert_identical(&serial, &parallel);

    println!(
        "逐次 {serial_time:.3} 秒 / 並列 {parallel_time:.3} 秒（{:.2} 倍）",
        serial_time / parallel_time.max(1e-12)
    );
}

#[test]
fn repeated_parallel_runs_are_identical() {
    // スレッドの割り当ては実行ごとに変わりうるので、複数回まわして確かめる
    let mesh = mesh(12, 8);
    let case = case_with(Backend::CpuParallel);
    let reference = solve(&mesh, &case, None).unwrap();
    for _ in 0..3 {
        assert_identical(&reference, &solve(&mesh, &case, None).unwrap());
    }
}

#[test]
fn auto_backend_switches_at_the_measured_break_even() {
    // 実測では 1.5 万節点あたりで並列が逐次に追いつく（tests/scaling.rs）。
    // しきい値はそこより少し上に置いてあり、下では逐次を選ぶ
    assert_eq!(Backend::Auto.resolve(1_000), Backend::CpuSerial);
    assert_eq!(Backend::Auto.resolve(19_999), Backend::CpuSerial);
    assert_eq!(Backend::Auto.resolve(20_000), Backend::CpuParallel);
    assert_eq!(Backend::Auto.resolve(10_000_000), Backend::CpuParallel);
    // 明示指定は規模に関わらずそのまま
    assert_eq!(Backend::CpuParallel.resolve(1), Backend::CpuParallel);
    assert_eq!(Backend::CpuSerial.resolve(10_000_000), Backend::CpuSerial);
}

#[test]
fn auto_backend_gives_the_same_answer_as_an_explicit_one() {
    // どちらを選んでも結果は同じなので、この判断が答えを変えることはない
    let small = mesh(3, 2);
    assert!(small.nodes.len() < 20_000);
    let a = solve(&small, &case_with(Backend::Auto), None).unwrap();
    assert_eq!(a.solver.backend, "cpu_serial");
    assert_identical(
        &a,
        &solve(&small, &case_with(Backend::CpuSerial), None).unwrap(),
    );
    assert_identical(
        &a,
        &solve(&small, &case_with(Backend::CpuParallel), None).unwrap(),
    );
}

#[test]
fn backend_names_round_trip() {
    for (text, expected) in [
        ("auto", Backend::Auto),
        ("cpu", Backend::CpuSerial),
        ("cpu_serial", Backend::CpuSerial),
        ("cpu_parallel", Backend::CpuParallel),
        ("  CPU_PARALLEL ", Backend::CpuParallel),
        ("cuda", Backend::Cuda),
        ("gpu", Backend::Cuda),
    ] {
        assert_eq!(Backend::parse(text).unwrap(), expected, "{text}");
    }
    assert_eq!(Backend::CpuParallel.label(), "cpu_parallel");
    assert_eq!(Backend::Cuda.label(), "cuda");

    let e = Backend::parse("opencl").unwrap_err();
    assert!(e.to_string().contains("opencl"), "{e}");
    assert!(e.to_string().contains("cuda"), "{e}");
}

#[test]
fn region_stats_and_summary_agree_between_backends() {
    let mesh = mesh(10, 7);
    let serial_case = case_with(Backend::CpuSerial);
    let parallel_case = case_with(Backend::CpuParallel);
    let serial = solve(&mesh, &serial_case, None).unwrap();
    let parallel = solve(&mesh, &parallel_case, None).unwrap();

    let a = serial.region_stats(&mesh, &serial_case).unwrap();
    let b = parallel.region_stats(&mesh, &parallel_case).unwrap();
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.volume, y.volume);
        assert_eq!(x.power, y.power);
        assert_eq!(x.current_density.max, y.current_density.max);
        assert_eq!(x.current_density.p99, y.current_density.p99);
        assert_eq!(x.hotspot.element_id, y.hotspot.element_id);
    }
}
