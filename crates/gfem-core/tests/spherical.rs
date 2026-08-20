//! 球対称の解析解ベンチマーク（SPEC §8 M2 の 3・4）。
//!
//! - **同心球殻**: 内側電極から外側電極へ電流 I。`J_r = I/(4πr²)`
//! - **半球殻**: 赤道面が絶縁境界になるので「半無限媒質上の電極」の模型になる。
//!   `J_r = I/(2πr²)`、`φ = I/(2πσ)(1/r − 1/R)`
//!
//! どちらも層ごとに σ を変えた多層版で「層構造が正しく効くこと」も見る。
//!
//! メッシュは球面を多面体で近似しているので、解析解との差には
//! 半径方向の離散化 O(h_r²) と角度方向の多面体近似 O(h_ang²) の両方が乗る。
//! ここでは代表的な分割で許容誤差の範囲に入ることだけを確かめ、
//! **次数そのものは `convergence.rs` で測る**。
//!
//! # 許容誤差の考え方
//!
//! Tet4 では E が要素内で一定なので、`E ∝ 1/r²` のように急変する場では
//! 1 要素ぶんの差がそのまま誤差になる。内側電極のすぐ外側（E が最大の場所）では
//! 相対誤差が `2h/r` 程度まで開き、代表的な分割では十数 % に達する。
//! これは実装の不備ではなく 1 次要素の性質なので、
//!
//! - **最大値**の判定は緩め（内側 1 層に律速される）
//! - **体積重み平均**の判定は厳しめ（場全体としての正しさを見る）
//!
//! という二段構えにしてある。

mod common;

use common::{INNER_PID, OUTER_PID, ShellLayer, centroid, spherical_shell, uniform_shell};
use gfem_core::case::{Case, Electrode, Preconditioner, ReferencePotential, SolverOptions};
use gfem_core::solve;
use std::f64::consts::PI;

const CURRENT: f64 = 2.0e-3; // A
const INNER: f64 = 0.010; // m
const OUTER: f64 = 0.050; // m

/// 全球殻の立体角。
const FULL: f64 = 4.0 * PI;
/// 半球殻の立体角（赤道面が絶縁境界）。
const HEMI: f64 = 2.0 * PI;

fn case_for(sigmas: &[(u32, f64)]) -> Case {
    let mut case = Case::new();
    for &(pid, s) in sigmas {
        case.set_conductivity(pid, s).unwrap();
    }
    case.add_electrode(Electrode::anode("inner", CURRENT).with_shell_pids([INNER_PID]));
    case.add_electrode(Electrode::cathode("outer", CURRENT).with_shell_pids([OUTER_PID]));
    case.solver = SolverOptions {
        tolerance: 1.0e-12,
        max_iterations: 50_000,
        preconditioner: Preconditioner::Jacobi,
    };
    // 解析解は φ(R) = 0 と揃えてあるので、基準も外側電極に合わせる
    case.reference = ReferencePotential::CathodeMean;
    case
}

/// 層構造 `(σ, 外半径)` に対する `φ(r)`（φ(最外半径) = 0）。
///
/// `φ(r) − φ(R) = ∫_r^R I/(Ω σ(s) s²) ds` を層ごとに区分積分する。
fn exact_potential(layers: &[(f64, f64)], solid_angle: f64, inner: f64, r: f64) -> f64 {
    let mut phi = 0.0;
    let mut lower = inner;
    for &(sigma, outer) in layers {
        let start = lower.max(r);
        if outer > start {
            phi += CURRENT / (solid_angle * sigma) * (1.0 / start - 1.0 / outer);
        }
        lower = outer;
    }
    phi
}

/// `r` を含む層の σ。
fn sigma_at(layers: &[(f64, f64)], inner: f64, r: f64) -> f64 {
    let mut lower = inner;
    for &(sigma, outer) in layers {
        let inside = r <= outer || outer == layers[layers.len() - 1].1;
        if inside && r >= lower - 1e-12 {
            return sigma;
        }
        lower = outer;
    }
    layers[layers.len() - 1].0
}

/// 解析解の電界 `E = I/(Ω σ r²) r̂`。
fn exact_field(layers: &[(f64, f64)], solid_angle: f64, inner: f64, x: [f64; 3]) -> [f64; 3] {
    let r = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
    let sigma = sigma_at(layers, inner, r);
    let magnitude = CURRENT / (solid_angle * sigma * r * r);
    [
        magnitude * x[0] / r,
        magnitude * x[1] / r,
        magnitude * x[2] / r,
    ]
}

/// 電極間の等価抵抗。
fn exact_resistance(layers: &[(f64, f64)], solid_angle: f64, inner: f64) -> f64 {
    exact_potential(layers, solid_angle, inner, inner) / CURRENT
}

fn radius(x: [f64; 3]) -> f64 {
    (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt()
}

/// 要素ごとの相対誤差の `(最大, 体積重み平均)`。
fn relative_errors<F>(values: &[f64], volumes: &[f64], expected: F) -> (f64, f64)
where
    F: Fn(usize) -> f64,
{
    let mut worst: f64 = 0.0;
    let mut weighted = 0.0;
    let mut total = 0.0;
    for (e, &value) in values.iter().enumerate() {
        let truth = expected(e);
        let error = (value - truth).abs() / truth.abs();
        worst = worst.max(error);
        weighted += volumes[e] * error;
        total += volumes[e];
    }
    (worst, weighted / total)
}

#[test]
fn concentric_shell_reproduces_the_inverse_square_current_density() {
    // SPEC §8 M2 の 3: J_r = I/(4πr²)
    let sigma = 0.35;
    let mesh = uniform_shell(INNER, OUTER, 12, 16, 32, false);
    let a = solve(&mesh, &case_for(&[(1, sigma)]), None).unwrap();
    assert!(a.solver.converged, "{:?}", a.solver.failure_reason());

    let (worst, mean) = relative_errors(&a.fields.j_magnitude, &a.fields.volume, |e| {
        let r = radius(centroid(&mesh, e));
        CURRENT / (FULL * r * r)
    });
    println!("|J| の相対誤差: 最大 {worst:.3e}, 体積重み平均 {mean:.3e}");
    // 最大は内側 1 層（E が最大で、要素内一定近似が最も苦しい場所）に律速される
    assert!(worst < 0.25, "|J| の最大相対誤差 {worst}");
    assert!(mean < 0.05, "|J| の体積重み平均誤差 {mean}");

    // 等価抵抗 R = (1/(4πσ))(1/a − 1/R)
    let expected_r = exact_resistance(&[(sigma, OUTER)], FULL, INNER);
    let error = (a.impedance() - expected_r).abs() / expected_r;
    println!(
        "R = {} Ω / 期待 {expected_r} Ω / 相対誤差 {error:.3e}",
        a.impedance()
    );
    assert!(error < 0.02, "R の相対誤差 {error}");
}

#[test]
fn concentric_shell_potential_matches_the_analytic_profile() {
    let sigma = 0.35;
    let mesh = uniform_shell(INNER, OUTER, 12, 16, 32, false);
    let a = solve(&mesh, &case_for(&[(1, sigma)]), None).unwrap();
    let layers = [(sigma, OUTER)];

    // メッシュは半径ごとに節点がまとまって並んでいる（球殻生成器の作り）ので、
    // 層ごとの平均電位を取れば動径方向の分布がそのまま見える
    let n_layers = 13; // n_radial = 12
    let per_layer = mesh.nodes.len() / n_layers;
    let scale = exact_potential(&layers, FULL, INNER, INNER);

    let mut worst_error: f64 = 0.0;
    let mut worst_spread: f64 = 0.0;
    let mut layer_means = Vec::with_capacity(n_layers);
    for k in 0..n_layers {
        let slice = &a.potential[k * per_layer..(k + 1) * per_layer];
        let r = radius(mesh.nodes.coord((k * per_layer) as u32));
        let expected = exact_potential(&layers, FULL, INNER, r);
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        layer_means.push(mean);
        worst_error = worst_error.max((mean - expected).abs() / scale);
        let lo = slice.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        worst_spread = worst_spread.max((hi - lo) / scale);
    }
    println!("φ: 層平均の最大誤差 {worst_error:.3e}, 同一半径での揺らぎ {worst_spread:.3e}");

    // 動径方向の分布が解析解に沿っていること
    assert!(
        worst_error < 0.02,
        "層平均 φ の最大誤差 {worst_error}（相対）"
    );
    // 単調減少（内側電極が高電位）
    assert!(
        layer_means.windows(2).all(|w| w[1] < w[0]),
        "φ が半径方向に単調減少していません: {layer_means:?}"
    );
    // 同一半径での揺らぎは、球面を多面体で近似したことによる幾何誤差。
    // 角度分割を細かくすれば O(h_ang²) で消える（convergence.rs で確認）
    assert!(
        worst_spread < 0.08,
        "同一半径での φ の揺らぎ {worst_spread}"
    );
}

#[test]
fn hemispherical_shell_models_a_point_electrode_on_a_half_space() {
    // SPEC §8 M2 の 4: φ = I/(2πσ)(1/r − 1/R)
    // 赤道面には電極を貼らないので絶縁境界になり、半無限媒質の対称面と同じ働きをする
    let sigma = 0.35;
    let mesh = uniform_shell(INNER, OUTER, 12, 12, 32, true);
    let a = solve(&mesh, &case_for(&[(1, sigma)]), None).unwrap();
    assert!(a.solver.converged, "{:?}", a.solver.failure_reason());

    let expected_r = exact_resistance(&[(sigma, OUTER)], HEMI, INNER);
    let error = (a.impedance() - expected_r).abs() / expected_r;
    println!(
        "R = {} Ω / 期待 {expected_r} Ω / 相対誤差 {error:.3e}",
        a.impedance()
    );
    assert!(error < 0.03, "R の相対誤差 {error}");

    // 電流密度は全球殻の 2 倍になる（立体角が半分だから）
    let (worst, mean) = relative_errors(&a.fields.j_magnitude, &a.fields.volume, |e| {
        let r = radius(centroid(&mesh, e));
        CURRENT / (HEMI * r * r)
    });
    println!("|J| の相対誤差: 最大 {worst:.3e}, 体積重み平均 {mean:.3e}");
    assert!(worst < 0.3, "|J| の最大相対誤差 {worst}");
    assert!(mean < 0.05, "|J| の体積重み平均誤差 {mean}");

    // 電位が最も高いのは内側電極の側
    assert!(a.conservation.voltage > 0.0);
}

#[test]
fn layered_shell_adds_the_resistances_in_series() {
    // SPEC §8 M2 の 3 の「層ごとに σ を変えた多層版」
    let layers = [
        ShellLayer {
            pid: 1,
            outer_radius: 0.020,
            divisions: 4,
        },
        ShellLayer {
            pid: 2,
            outer_radius: 0.030,
            divisions: 4,
        },
        ShellLayer {
            pid: 3,
            outer_radius: OUTER,
            divisions: 6,
        },
    ];
    let sigmas = [(1u32, 0.10), (2, 0.04), (3, 0.35)];
    let exact_layers = [(0.10, 0.020), (0.04, 0.030), (0.35, OUTER)];

    let mesh = spherical_shell(INNER, &layers, 16, 32, false);
    let a = solve(&mesh, &case_for(&sigmas), None).unwrap();
    assert!(a.solver.converged, "{:?}", a.solver.failure_reason());

    let expected_r = exact_resistance(&exact_layers, FULL, INNER);
    let error = (a.impedance() - expected_r).abs() / expected_r;
    println!(
        "R = {} Ω / 期待 {expected_r} Ω / 相対誤差 {error:.3e}",
        a.impedance()
    );
    assert!(error < 0.02, "R の相対誤差 {error}");

    // |J| は σ に依らず I/(4πr²)、|E| は σ に反比例して層ごとに階段状になる
    let block = mesh.block(gfem_core::ElementKind::Tet4).unwrap();
    let sigma_of = |e: usize| {
        sigmas
            .iter()
            .find(|&&(pid, _)| pid == block.pids()[e])
            .unwrap()
            .1
    };
    let (worst_j, mean_j) = relative_errors(&a.fields.j_magnitude, &a.fields.volume, |e| {
        let r = radius(centroid(&mesh, e));
        CURRENT / (FULL * r * r)
    });
    let (worst_e, mean_e) = relative_errors(&a.fields.e_magnitude, &a.fields.volume, |e| {
        let r = radius(centroid(&mesh, e));
        CURRENT / (FULL * r * r * sigma_of(e))
    });
    println!(
        "|J|: 最大 {worst_j:.3e} 平均 {mean_j:.3e} / |E|: 最大 {worst_e:.3e} 平均 {mean_e:.3e}"
    );
    assert!(worst_j < 0.25, "|J| の最大相対誤差 {worst_j}");
    assert!(mean_j < 0.05, "|J| の体積重み平均誤差 {mean_j}");
    // σ は要素ごとに厳密なので、|E| の相対誤差は |J| と一致する
    assert!(
        (worst_e - worst_j).abs() < 1e-12,
        "|E| と |J| の誤差が食い違います"
    );
    assert!(mean_e < 0.05, "|E| の体積重み平均誤差 {mean_e}");
}

#[test]
fn field_direction_is_radial() {
    let mesh = uniform_shell(INNER, OUTER, 6, 8, 16, false);
    let a = solve(&mesh, &case_for(&[(1, 0.35)]), None).unwrap();
    let mut worst: f64 = 0.0;
    for e in 0..a.fields.n_elements {
        let c = centroid(&mesh, e);
        let r = radius(c);
        let j = [
            a.fields.current_density[3 * e],
            a.fields.current_density[3 * e + 1],
            a.fields.current_density[3 * e + 2],
        ];
        let magnitude = a.fields.j_magnitude[e];
        // 単位動径ベクトルとの内積が 1 に近いこと（外向き）
        let cosine = (j[0] * c[0] + j[1] * c[1] + j[2] * c[2]) / (magnitude * r);
        worst = worst.max(1.0 - cosine);
    }
    println!("動径方向からのずれ（1 − cosθ）の最大: {worst:.3e}");
    // 多面体近似のぶんだけ動径からずれる。角度分割を細かくすれば減る
    assert!(worst < 0.05, "動径方向からのずれ {worst}");
}

#[test]
fn conservation_checks_pass_on_curved_meshes() {
    for hemisphere in [false, true] {
        let mesh = uniform_shell(INNER, OUTER, 6, 8, 16, hemisphere);
        let a = solve(&mesh, &case_for(&[(1, 0.35)]), None).unwrap();
        assert!(
            a.conservation.is_clean(),
            "hemisphere={hemisphere}: {:?}",
            a.conservation.warnings
        );
        assert!(a.conservation.current_balance.abs() < 1e-12);
        assert!(a.conservation.energy_relative_error < 1e-9);
        assert!(a.conservation.discrete_residual < 1e-11);
    }
}

#[test]
fn analytic_helpers_are_self_consistent() {
    // 層をまたいだ積分が単一層の式に一致すること
    let single = exact_potential(&[(0.35, OUTER)], FULL, INNER, INNER);
    let split = exact_potential(&[(0.35, 0.030), (0.35, OUTER)], FULL, INNER, INNER);
    assert!((single - split).abs() < 1e-15);
    // φ(R) = 0
    assert!(exact_potential(&[(0.35, OUTER)], FULL, INNER, OUTER).abs() < 1e-18);
    // E は φ の勾配と整合（差分で確認）
    let r = 0.02;
    let h = 1e-7;
    let d = (exact_potential(&[(0.35, OUTER)], FULL, INNER, r + h)
        - exact_potential(&[(0.35, OUTER)], FULL, INNER, r - h))
        / (2.0 * h);
    let e = exact_field(&[(0.35, OUTER)], FULL, INNER, [r, 0.0, 0.0]);
    assert!((e[0] + d).abs() < 1e-6 * e[0].abs(), "{} vs {}", e[0], -d);
}
