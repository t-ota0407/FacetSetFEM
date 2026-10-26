//! 二層半無限媒質上の電極（SPEC §8 M2 の 5）。
//!
//! 地電気探査の古典解と照合する。表面に置いた点電極から流した電流の、
//! 二層構造（上層 ρ1・厚さ h、下層 ρ2・半無限）における表面電位は鏡像法の級数で
//!
//! ```text
//! φ(r) = (I ρ1 / 2π) [ 1/r + 2 Σ_{n≥1} kⁿ / √(r² + (2nh)²) ],   k = (ρ2 − ρ1)/(ρ2 + ρ1)
//! ```
//!
//! と書ける。ここでは 4 電極の **Wenner 配置**（C1 P1 P2 C2 を等間隔 s で並べる）を使い、
//! 見かけ比抵抗 `ρ_a = 2πs ΔV/I` を測る。二層理論では
//!
//! ```text
//! ρ_a/ρ1 = 1 + 4 Σ_{n≥1} kⁿ [ 1/√(1+(2nh/s)²) − 1/√(4+(2nh/s)²) ]
//! ```
//!
//! # 打ち切り誤差の消し方
//!
//! 解析解は「無限に広い半無限媒質」「点電極」を仮定しているが、計算は有限の箱と
//! 有限の大きさの電極で行う。そこで**同じメッシュで一様媒質の場合も解き、比を取る**。
//! 一様半無限媒質では ρ_a = ρ1 が厳密に成り立つので、
//!
//! ```text
//! ρ_a(二層) / ρ_a(一様)  ←→  理論値 ρ_a/ρ1
//! ```
//!
//! の比較になり、箱の打ち切り・電極の大きさ・メッシュの粗さといった系統誤差が
//! ほぼ相殺される。層構造そのものの効き方だけを見ることができる。
//!
//! # 実行方法
//!
//! メッシュが大きく既定のテスト実行には重いので `#[ignore]` を付けてある。
//!
//! ```text
//! cargo test --release -p facetsetfem-core --test two_layer -- --ignored --nocapture
//! ```

mod common;

use common::{Patch, box_mesh, graded_grid, nearest_node, symmetric_graded_grid};
use facetsetfem_core::case::{
    Backend, Case, Electrode, Preconditioner, ReferencePotential, SolverOptions,
};
use facetsetfem_core::mesh::PropId;
use facetsetfem_core::{Mesh, solve};
use std::f64::consts::PI;

/// 電極間隔 [m]（Wenner 配置の等間隔）。
const SPACING: f64 = 0.10;
/// 上層の厚さ [m]。
const LAYER_DEPTH: f64 = 0.05;
/// 総電流 [A]（対称性で半分の領域だけ解くので、投入するのはこの半分）。
const CURRENT: f64 = 2.0e-3;
/// 細かい格子の刻み [m]。
const FINE_STEP: f64 = 0.0125;
/// 細かい格子を張る範囲 [m]。
const FINE_TO: f64 = 0.25;
/// 領域の外寸 [m]。
const OUTER: f64 = 1.0;
/// 遠方へ伸ばすときの等比。
const RATIO: f64 = 1.6;

const UPPER_PID: PropId = 1;
const LOWER_PID: PropId = 2;
const ANODE_PID: PropId = 101;
const CATHODE_PID: PropId = 201;
/// 遠方境界（下面）に置く帰路電極。
const RETURN_PID: PropId = 301;

/// 二層理論の見かけ比抵抗比（Wenner 配置）。
fn wenner_ratio(k: f64, depth_over_spacing: f64, terms: usize) -> f64 {
    let mut sum = 0.0;
    let mut power = 1.0;
    for n in 1..=terms {
        power *= k;
        let t = 2.0 * n as f64 * depth_over_spacing;
        sum += power * (1.0 / (1.0 + t * t).sqrt() - 1.0 / (4.0 + t * t).sqrt());
    }
    1.0 + 4.0 * sum
}

/// y = 0 を対称面にした半分の領域のメッシュ。
///
/// y = 0 の面には電極を貼らないので絶縁境界になり、これは対称面と同じ条件になる。
/// したがって全電流の半分を流せば、得られる電位は全領域を解いた場合と一致する。
fn half_space_mesh() -> Mesh {
    let xs = symmetric_graded_grid(FINE_TO, FINE_STEP, OUTER, RATIO);
    let ys = graded_grid(FINE_TO, FINE_STEP, OUTER, RATIO);
    // z は地表 0 から下向き。上層の界面 −h が格子線に乗るように刻む
    let down = graded_grid(3.0 * LAYER_DEPTH, FINE_STEP, OUTER, RATIO);
    let mut zs: Vec<f64> = down.iter().skip(1).rev().map(|v| -v).collect();
    zs.push(0.0);

    let cell_pid: Vec<PropId> = zs
        .windows(2)
        .map(|w| {
            if w[0] >= -LAYER_DEPTH - 1e-9 {
                UPPER_PID
            } else {
                LOWER_PID
            }
        })
        .collect();

    let half = FINE_STEP;
    let patch = |pid: PropId, centre: f64| Patch {
        pid,
        x: [centre - half, centre + half],
        y: [0.0, FINE_STEP],
        top: true,
    };
    box_mesh(
        &xs,
        &ys,
        &zs,
        &cell_pid,
        &[
            patch(ANODE_PID, -1.5 * SPACING),
            patch(CATHODE_PID, 1.5 * SPACING),
            Patch::whole(RETURN_PID, false),
        ],
    )
}

/// 見かけ比抵抗 `ρ_a = 2πs ΔV/I` を求める。
fn apparent_resistivity(mesh: &Mesh, sigma_upper: f64, sigma_lower: f64) -> f64 {
    let mut case = Case::new();
    case.set_conductivity(UPPER_PID, sigma_upper).unwrap();
    case.set_conductivity(LOWER_PID, sigma_lower).unwrap();
    // 対称面で切っているので、投入するのは全電流の半分
    case.add_electrode(Electrode::anode("C1", 0.5 * CURRENT).with_shell_pids([ANODE_PID]));
    case.add_electrode(Electrode::cathode("C2", 0.5 * CURRENT).with_shell_pids([CATHODE_PID]));
    case.solver = SolverOptions {
        tolerance: 1.0e-11,
        max_iterations: 200_000,
        preconditioner: Preconditioner::Jacobi,
        backend: Backend::CpuSerial,
        ..Default::default()
    };
    case.reference = ReferencePotential::ZeroMean;

    let a = solve(mesh, &case, None).unwrap();
    assert!(a.solver.converged, "{:?}", a.solver.failure_reason());
    assert!(
        a.conservation.is_clean(),
        "保存則チェックに警告: {:?}",
        a.conservation.warnings
    );

    let p1 = nearest_node(mesh, [-0.5 * SPACING, 0.0, 0.0]);
    let p2 = nearest_node(mesh, [0.5 * SPACING, 0.0, 0.0]);
    let delta_v = a.potential[p1 as usize] - a.potential[p2 as usize];
    2.0 * PI * SPACING * delta_v / CURRENT
}

#[test]
fn wenner_series_matches_known_values() {
    // 一様媒質（k = 0）なら ρ_a = ρ1
    assert!((wenner_ratio(0.0, 0.5, 500) - 1.0).abs() < 1e-15);
    // 下層が完全絶縁（k = +1）なら ρ_a は ρ1 より大きくなる
    assert!(wenner_ratio(1.0, 0.5, 5000) > 1.0);
    // 下層が良導体（k = −1）なら小さくなる
    assert!(wenner_ratio(-1.0, 0.5, 5000) < 1.0);
    // 層が非常に深ければ上層だけを見る
    assert!((wenner_ratio(-0.6, 50.0, 2000) - 1.0).abs() < 1e-3);
}

#[test]
#[ignore = "メッシュが大きいので既定の実行からは外す。--ignored で実行する"]
fn two_layer_half_space_matches_the_image_series() {
    let sigma1 = 0.10; // ρ1 = 10 Ω·m
    let sigma2 = 0.40; // ρ2 = 2.5 Ω·m（下層が良導体）
    let k = (1.0 / sigma2 - 1.0 / sigma1) / (1.0 / sigma2 + 1.0 / sigma1);

    let mesh = half_space_mesh();
    println!(
        "メッシュ: 節点 {}, 四面体 {}",
        mesh.nodes.len(),
        mesh.n_solid_elements()
    );

    let uniform = apparent_resistivity(&mesh, sigma1, sigma1);
    let layered = apparent_resistivity(&mesh, sigma1, sigma2);
    let ratio = layered / uniform;
    let expected = wenner_ratio(k, LAYER_DEPTH / SPACING, 5000);

    println!("k = {k:.4}, h/s = {:.2}", LAYER_DEPTH / SPACING);
    println!("ρ_a(一様)  = {uniform:.4} Ω·m（理論 {:.4}）", 1.0 / sigma1);
    println!("ρ_a(二層)  = {layered:.4} Ω·m");
    println!(
        "比 {ratio:.4} / 理論 {expected:.4} / 相対誤差 {:.3e}",
        (ratio - expected).abs() / expected
    );

    // 一様媒質の見かけ比抵抗は理論上ちょうど ρ1 になる。
    // これは「半無限媒質上の点電極 φ = I/(2πσr)」（SPEC §8 M2 の 4）を
    // 半球殻ではなく本来の半空間の形で確かめていることにもなる
    assert!(
        (uniform * sigma1 - 1.0).abs() < 0.01,
        "一様媒質の ρ_a = {uniform} Ω·m（理論 {} Ω·m）",
        1.0 / sigma1
    );
    // 二層構造の効きは大きい（理論比は 0.5 前後）ので、この判定は
    // 「層が正しく効いているか」を十分に切り分けられる
    assert!(
        (ratio - expected).abs() / expected < 0.02,
        "見かけ比抵抗比 {ratio}（理論 {expected}）"
    );
}

#[test]
#[ignore = "メッシュが大きいので既定の実行からは外す。--ignored で実行する"]
fn resistive_lower_layer_raises_the_apparent_resistivity() {
    // 下層が絶縁側（k > 0）のときは見かけ比抵抗が上がる
    let sigma1 = 0.10;
    let sigma2 = 0.02;
    let k = (1.0 / sigma2 - 1.0 / sigma1) / (1.0 / sigma2 + 1.0 / sigma1);

    let mesh = half_space_mesh();
    let uniform = apparent_resistivity(&mesh, sigma1, sigma1);
    let layered = apparent_resistivity(&mesh, sigma1, sigma2);
    let ratio = layered / uniform;
    let expected = wenner_ratio(k, LAYER_DEPTH / SPACING, 5000);

    println!("k = {k:.4}, 比 {ratio:.4} / 理論 {expected:.4}");
    assert!(
        ratio > 1.0,
        "下層が絶縁側なのに見かけ比抵抗が上がっていません"
    );
    assert!(
        (ratio - expected).abs() / expected < 0.02,
        "見かけ比抵抗比 {ratio}（理論 {expected}）"
    );
}
