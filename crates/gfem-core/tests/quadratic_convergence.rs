//! 2 次四面体のメッシュ収束性試験（SPEC §8 M12 段階 6）。
//!
//! 2 次四面体の理論値は
//!
//! ```text
//! ‖φ_h − φ‖_L2 = O(h³)      ‖E_h − E‖_L2 = O(h²)
//! ```
//!
//! で、1 次（`convergence.rs`）より 1 つずつ高い。問題は同じ球殻（φ ∝ 1/r）。
//!
//! # 球面に中点を載せる
//!
//! 1 次の多面体メッシュを 2 次に上げただけ（辺が真っ直ぐ）では、球面を多面体で近似した
//! 幾何の誤差が O(h²) のまま残り、2 次要素の次数が出ない。球面上の辺の中点を球面に
//! 載せ直す（辺が曲がった要素にする）と、境界が 2 次曲面で近似されて理論次数に乗る。
//! 曲がった要素の経路（積分点ごとのヤコビアン・6 節点の曲面の面積分）の検証にもなる。

mod common;

use common::{INNER_PID, OUTER_PID, uniform_shell};
use gfem_core::case::{
    Backend, Case, Electrode, Preconditioner, ReferencePotential, SolverOptions,
};
use gfem_core::element::tet10::EDGES;
use gfem_core::mesh::elevate;
use gfem_core::post::{
    convergence_order, l2_error_electric_field, l2_error_potential, mean_element_size,
};
use gfem_core::{Mesh, Nodes, solve};
use std::f64::consts::PI;

const CURRENT: f64 = 2.0e-3;
const SIGMA: f64 = 0.35;
const INNER: f64 = 0.010;
const OUTER: f64 = 0.050;

/// 細分化列（はじめの 3 段は 1 次の `convergence.rs` と同じ）。半径方向と角度方向を同じ比率で
/// 細かくする。ふだんのテストは 2 段（2 次で 6 千節点。デバッグビルドで数秒）、
/// 重いテスト（`#[ignore]`）は 4 段（2 次で 39 万節点）まで。
const LEVELS: [(usize, usize, usize); 4] = [(3, 4, 8), (6, 8, 16), (12, 16, 32), (24, 32, 64)];

fn case() -> Case {
    let mut case = Case::new();
    case.set_conductivity(1, SIGMA).unwrap();
    case.add_electrode(Electrode::anode("inner", CURRENT).with_shell_pids([INNER_PID]));
    case.add_electrode(Electrode::cathode("outer", CURRENT).with_shell_pids([OUTER_PID]));
    case.solver = SolverOptions {
        tolerance: 1.0e-12,
        max_iterations: 100_000,
        preconditioner: Preconditioner::Amg,
        backend: Backend::CpuParallel,
        ..Default::default()
    };
    case.reference = ReferencePotential::CathodeMean;
    case
}

fn radius(x: [f64; 3]) -> f64 {
    (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt()
}

fn exact_potential(solid_angle: f64, r: f64) -> f64 {
    CURRENT / (solid_angle * SIGMA) * (1.0 / r - 1.0 / OUTER)
}

fn exact_field(solid_angle: f64, x: [f64; 3]) -> [f64; 3] {
    let r = radius(x);
    let magnitude = CURRENT / (solid_angle * SIGMA * r * r);
    [
        magnitude * x[0] / r,
        magnitude * x[1] / r,
        magnitude * x[2] / r,
    ]
}

/// 2 次に上げたメッシュの、内外の球面に載る辺の中点を球面へ動かす（辺が曲がった要素）。
///
/// 両端の頂点が同じ球面（半径 `INNER` か `OUTER`）にある辺だけを動かす。中の辺は
/// 真っ直ぐのまま（境界の形だけが幾何の誤差を決める）。半球殻の赤道面上の辺は、
/// 両端が z = 0 なので半径方向に動かしても z = 0 のまま。
fn curve_onto_spheres(mesh: Mesh) -> Mesh {
    let mut coords = mesh.nodes.coords().to_vec();
    let on_sphere = |x: [f64; 3]| {
        let r = radius(x);
        [INNER, OUTER]
            .into_iter()
            .find(|&s| (r - s).abs() < 1e-9 * OUTER)
    };
    for e in mesh.iter_solids() {
        for (k, [i, j]) in EDGES.into_iter().enumerate() {
            let (a, b) = (mesh.nodes.coord(e.conn[i]), mesh.nodes.coord(e.conn[j]));
            let (Some(ra), Some(rb)) = (on_sphere(a), on_sphere(b)) else {
                continue;
            };
            if ra != rb {
                continue;
            }
            let m = e.conn[4 + k] as usize;
            let x = [coords[3 * m], coords[3 * m + 1], coords[3 * m + 2]];
            let scale = ra / radius(x);
            for d in 0..3 {
                coords[3 * m + d] = x[d] * scale;
            }
        }
    }
    Mesh {
        nodes: Nodes::new(mesh.nodes.ids().to_vec(), coords),
        ..mesh
    }
}

struct Measurement {
    h: f64,
    phi_error: f64,
    field_error: f64,
    resistance_error: f64,
    nodes: usize,
    iterations: usize,
}

/// 要素の作り方。
#[derive(Clone, Copy, PartialEq)]
enum Elements {
    /// 1 次のまま（比較用）
    Linear,
    /// 2 次に上げただけ（辺が真っ直ぐ）
    Straight,
    /// 2 次に上げて、球面上の中点を球面に載せたもの
    Curved,
}

fn measure(
    hemisphere: bool,
    elements: Elements,
    levels: &[(usize, usize, usize)],
) -> Vec<Measurement> {
    let solid_angle = if hemisphere { 2.0 * PI } else { 4.0 * PI };
    levels
        .iter()
        .map(|&(n_radial, n_theta, n_phi)| {
            let linear = uniform_shell(INNER, OUTER, n_radial, n_theta, n_phi, hemisphere);
            let mesh = match elements {
                Elements::Linear => linear,
                Elements::Straight => elevate(&linear).unwrap(),
                Elements::Curved => curve_onto_spheres(elevate(&linear).unwrap()),
            };
            let a = solve(&mesh, &case(), None).unwrap();
            assert!(a.solver.converged, "{:?}", a.solver.failure_reason());
            assert!(
                a.conservation.is_clean(),
                "保存則チェックに警告: {:?}",
                a.conservation.warnings
            );
            let phi = l2_error_potential(&mesh, &a.potential, |x| {
                exact_potential(solid_angle, radius(x))
            })
            .unwrap();
            let field = l2_error_electric_field(&mesh, &a.potential, &a.fields, |x| {
                exact_field(solid_angle, x)
            })
            .unwrap();
            let expected_r = exact_potential(solid_angle, INNER) / CURRENT;
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

/// 表を出し、**最後の 2 段**から測った収束次数 `(φ, E, R)` を返す。
///
/// 細分化の初めは漸近域に入っていないので（φ の次数は粗い側から 2.5 → 2.7 → 2.8 と上がる）、
/// 全段の最小二乗ではなく細かい側の 2 段で見る。
fn report(name: &str, data: &[Measurement]) -> (f64, f64, f64) {
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
    let last = &data[data.len() - 2..];
    let h: Vec<f64> = last.iter().map(|m| m.h).collect();
    let order = |f: fn(&Measurement) -> f64| {
        convergence_order(&h, &last.iter().map(f).collect::<Vec<_>>()).unwrap()
    };
    let orders = (
        order(|m| m.phi_error),
        order(|m| m.field_error),
        order(|m| m.resistance_error),
    );
    println!(
        "収束次数: φ {:.2}, E {:.2}, R {:.2}",
        orders.0, orders.1, orders.2
    );
    orders
}

#[test]
fn curved_second_order_shells_converge_faster_than_first_order() {
    // 2 段（2 次で 6 千節点まで）。漸近域の手前なので次数の下限だけを見る（細かくすると
    // φ 2.5 → 2.8、E 1.7 → 1.9 と理論値に近づく。重いテストで確かめる）。
    // 1 次（`convergence.rs`）は φ が 2 次・E が 1 次・R が 2 次
    for hemisphere in [false, true] {
        let name = if hemisphere {
            "2 次・球面に中点を載せた半球殻"
        } else {
            "2 次・球面に中点を載せた同心球殻"
        };
        let data = measure(hemisphere, Elements::Curved, &LEVELS[..2]);
        let (phi, field, resistance) = report(name, &data);
        assert!(
            phi > 2.2,
            "{name}: φ の収束次数 {phi} が低すぎます（理論 3）"
        );
        assert!(
            field > 1.4,
            "{name}: E の収束次数 {field} が低すぎます（理論 2）"
        );
        assert!(
            resistance > 2.8,
            "{name}: R の収束次数 {resistance} が低すぎます"
        );
        for pair in data.windows(2) {
            assert!(pair[1].phi_error < pair[0].phi_error);
            assert!(pair[1].field_error < pair[0].field_error);
        }
    }
}

#[test]
fn straight_second_order_shells_are_limited_by_the_faceted_geometry() {
    // 辺が真っ直ぐなままでは、球面を多面体で近似した誤差（O(h²)）が残る。積分量の R で
    // いちばんはっきり出る（中点を球面に載せると 3 次を超えるのに、真っ直ぐだと 2 次で頭打ち）
    let straight = measure(false, Elements::Straight, &LEVELS[..2]);
    let curved = measure(false, Elements::Curved, &LEVELS[..2]);
    let (_, _, r_straight) = report("2 次・辺が真っ直ぐな同心球殻", &straight);
    let (_, _, r_curved) = report("2 次・球面に中点を載せた同心球殻", &curved);
    assert!(
        r_straight < 2.3,
        "辺が真っ直ぐなのに R が {r_straight} 次で収束している"
    );
    assert!(r_curved > r_straight + 1.0, "{r_curved} vs {r_straight}");
    let (s, c) = (straight.last().unwrap(), curved.last().unwrap());
    assert!(
        c.phi_error < s.phi_error,
        "{} vs {}",
        c.phi_error,
        s.phi_error
    );
    assert!(
        c.resistance_error < 0.2 * s.resistance_error,
        "{} vs {}",
        c.resistance_error,
        s.resistance_error
    );
}

#[test]
#[ignore = "重い（2 次で 39 万節点まで 4 段を 2 回解く）。cargo test --release -- --include-ignored で回す"]
fn curved_second_order_shells_reach_the_theoretical_order() {
    for hemisphere in [false, true] {
        let name = if hemisphere {
            "2 次・球面に中点を載せた半球殻（4 段）"
        } else {
            "2 次・球面に中点を載せた同心球殻（4 段）"
        };
        let data = measure(hemisphere, Elements::Curved, &LEVELS);
        let (phi, field, resistance) = report(name, &data);
        assert!(
            (2.7..=3.3).contains(&phi),
            "{name}: φ の収束次数 {phi} が 3 次から外れています"
        );
        assert!(
            (1.8..=2.3).contains(&field),
            "{name}: E の収束次数 {field} が 2 次から外れています"
        );
        assert!(
            resistance > 3.0,
            "{name}: R の収束次数 {resistance} が低すぎます"
        );
    }

    // 同じくらいの節点数の 1 次要素と比べる（1 次は 1 段細かいメッシュ）
    let linear = measure(false, Elements::Linear, &LEVELS[1..]);
    let curved = measure(false, Elements::Curved, &LEVELS[..3]);
    report("1 次・同心球殻（比較用）", &linear);
    for (p1, p2) in linear.iter().zip(&curved) {
        println!(
            "節点 {} の 1 次 / {} の 2 次: φ 誤差 {:.2e} / {:.2e}、E 誤差 {:.2e} / {:.2e}、R 誤差 {:.2e} / {:.2e}",
            p1.nodes,
            p2.nodes,
            p1.phi_error,
            p2.phi_error,
            p1.field_error,
            p2.field_error,
            p1.resistance_error,
            p2.resistance_error
        );
        // 節点数が同じ程度なら、2 次のほうが φ・E・R のどれも精度が高い
        assert!(p2.phi_error < p1.phi_error);
        assert!(p2.field_error < p1.field_error);
        assert!(p2.resistance_error < p1.resistance_error);
    }
}
