//! メッシュ収束性試験（SPEC §8 M2）。
//!
//! 1 次四面体の理論値は
//!
//! ```text
//! ‖φ_h − φ‖_L2 = O(h²)      ‖E_h − E‖_L2 = O(h)
//! ```
//!
//! 平板の一様場は 1 次要素で厳密に解けてしまい（誤差が丸めレベル）次数が測れないので、
//! **球殻**を使う。球殻なら φ ∝ 1/r という滑らかな非線形解になり、
//! かつ厳密解が閉じた式で書ける。
//!
//! 誤差には半径方向の離散化と、球面を多面体で近似したことによる幾何誤差の
//! 両方が乗る。細分化のときに半径方向と角度方向を同じ比率で細かくすれば、
//! どちらも O(h²) なので合計も理論次数に乗る。

mod common;

use common::{INNER_PID, OUTER_PID, uniform_shell};
use gfem_core::case::{
    Backend, Case, Electrode, Preconditioner, ReferencePotential, SolverOptions,
};
use gfem_core::post::{convergence_order, l2_error_field, l2_error_potential, mean_element_size};
use gfem_core::{Mesh, solve};
use std::f64::consts::PI;

const CURRENT: f64 = 2.0e-3;
const SIGMA: f64 = 0.35;
const INNER: f64 = 0.010;
const OUTER: f64 = 0.050;

/// 細分化列。半径方向と角度方向を同じ比率で細かくする。
const LEVELS: [(usize, usize, usize); 3] = [(3, 4, 8), (6, 8, 16), (12, 16, 32)];

fn case() -> Case {
    let mut case = Case::new();
    case.set_conductivity(1, SIGMA).unwrap();
    case.add_electrode(Electrode::anode("inner", CURRENT).with_shell_pids([INNER_PID]));
    case.add_electrode(Electrode::cathode("outer", CURRENT).with_shell_pids([OUTER_PID]));
    case.solver = SolverOptions {
        tolerance: 1.0e-12,
        max_iterations: 100_000,
        preconditioner: Preconditioner::Jacobi,
        backend: Backend::CpuSerial,
    };
    case.reference = ReferencePotential::CathodeMean;
    case
}

fn exact_potential(solid_angle: f64, r: f64) -> f64 {
    CURRENT / (solid_angle * SIGMA) * (1.0 / r - 1.0 / OUTER)
}

fn exact_field(solid_angle: f64, x: [f64; 3]) -> [f64; 3] {
    let r = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
    let magnitude = CURRENT / (solid_angle * SIGMA * r * r);
    [
        magnitude * x[0] / r,
        magnitude * x[1] / r,
        magnitude * x[2] / r,
    ]
}

fn exact_resistance(solid_angle: f64) -> f64 {
    exact_potential(solid_angle, INNER) / CURRENT
}

struct Measurement {
    h: f64,
    phi_error: f64,
    field_error: f64,
    resistance_error: f64,
    nodes: usize,
    iterations: usize,
}

fn measure(hemisphere: bool) -> Vec<Measurement> {
    let solid_angle = if hemisphere { 2.0 * PI } else { 4.0 * PI };
    LEVELS
        .iter()
        .map(|&(n_radial, n_theta, n_phi)| {
            let mesh: Mesh = uniform_shell(INNER, OUTER, n_radial, n_theta, n_phi, hemisphere);
            let a = solve(&mesh, &case(), None).unwrap();
            assert!(a.solver.converged, "{:?}", a.solver.failure_reason());
            assert!(
                a.conservation.is_clean(),
                "保存則チェックに警告: {:?}",
                a.conservation.warnings
            );

            let phi = l2_error_potential(&mesh, &a.potential, |x| {
                exact_potential(
                    solid_angle,
                    (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt(),
                )
            })
            .unwrap();
            let field = l2_error_field(&mesh, &a.fields.electric_field, |x| {
                exact_field(solid_angle, x)
            })
            .unwrap();
            let expected_r = exact_resistance(solid_angle);

            Measurement {
                h: mean_element_size(&mesh).unwrap(),
                phi_error: phi.relative,
                field_error: field.relative,
                resistance_error: (a.impedance() - expected_r).abs() / expected_r,
                nodes: mesh.nodes.len(),
                iterations: a.solver.iterations,
            }
        })
        .collect()
}

fn report(name: &str, data: &[Measurement]) {
    println!("\n{name}");
    println!(
        "{:>8} {:>10} {:>7} {:>12} {:>12} {:>12}",
        "節点数", "h [m]", "反復", "φ 誤差", "E 誤差", "R 誤差"
    );
    for m in data {
        println!(
            "{:>8} {:>10.3e} {:>7} {:>12.3e} {:>12.3e} {:>12.3e}",
            m.nodes, m.h, m.iterations, m.phi_error, m.field_error, m.resistance_error
        );
    }
}

fn orders(data: &[Measurement]) -> (f64, f64, f64) {
    let h: Vec<f64> = data.iter().map(|m| m.h).collect();
    (
        convergence_order(&h, &data.iter().map(|m| m.phi_error).collect::<Vec<_>>()).unwrap(),
        convergence_order(&h, &data.iter().map(|m| m.field_error).collect::<Vec<_>>()).unwrap(),
        convergence_order(
            &h,
            &data.iter().map(|m| m.resistance_error).collect::<Vec<_>>(),
        )
        .unwrap(),
    )
}

#[test]
fn concentric_shell_converges_at_the_theoretical_order() {
    let data = measure(false);
    report("同心球殻（立体角 4π）", &data);
    let (phi, field, resistance) = orders(&data);
    println!("収束次数: φ {phi:.2}, E {field:.2}, R {resistance:.2}");

    // 理論値は φ が 2 次、E が 1 次（SPEC §8 M2）
    assert!(
        (1.7..=2.4).contains(&phi),
        "φ の収束次数 {phi} が 2 次から外れています"
    );
    assert!(
        (0.8..=1.5).contains(&field),
        "E の収束次数 {field} が 1 次から外れています"
    );
    // 等価抵抗は積分量なので φ と同じ 2 次
    assert!(resistance > 1.5, "R の収束次数 {resistance} が低すぎます");

    // 最も細かいメッシュで実用的な精度に達していること
    let finest = data.last().unwrap();
    assert!(finest.phi_error < 1e-2, "φ の相対誤差 {}", finest.phi_error);
    // E は要素内一定なので、内側電極近傍（E が最大の場所）の誤差が支配的になる。
    // 実用精度は「次数が 1 次で乗っていること」で担保する
    assert!(
        finest.field_error < 0.2,
        "E の相対誤差 {}",
        finest.field_error
    );
    assert!(
        finest.resistance_error < 2e-2,
        "R の相対誤差 {}",
        finest.resistance_error
    );
}

#[test]
fn hemispherical_shell_converges_at_the_theoretical_order() {
    // 半無限媒質上の電極（SPEC §8 M2 の 4）。赤道面の絶縁境界が正しく効いていれば
    // 全球殻と同じ次数で収束する
    let data = measure(true);
    report("半球殻（立体角 2π、赤道面は絶縁）", &data);
    let (phi, field, resistance) = orders(&data);
    println!("収束次数: φ {phi:.2}, E {field:.2}, R {resistance:.2}");

    assert!(
        (1.7..=2.4).contains(&phi),
        "φ の収束次数 {phi} が 2 次から外れています"
    );
    assert!(
        (0.8..=1.5).contains(&field),
        "E の収束次数 {field} が 1 次から外れています"
    );
    assert!(resistance > 1.5, "R の収束次数 {resistance} が低すぎます");
}

#[test]
fn errors_decrease_monotonically() {
    for data in [measure(false), measure(true)] {
        for pair in data.windows(2) {
            assert!(
                pair[1].phi_error < pair[0].phi_error,
                "φ の誤差が細分化で減っていません"
            );
            assert!(
                pair[1].field_error < pair[0].field_error,
                "E の誤差が細分化で減っていません"
            );
        }
    }
}
