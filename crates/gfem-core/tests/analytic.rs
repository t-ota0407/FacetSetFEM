//! 解析解ベンチマーク（SPEC §8 M2 の 1・2 を先取り）。
//!
//! M1 で組んだアセンブリ・射影 CG・電極荷重・ポスト処理が、
//! 厳密解の分かる問題で正しい答えを出すことを確かめる。
//!
//! 1 次要素は線形場を**厳密に**再現するため、一様場・層状媒質のいずれも
//! 離散化誤差はゼロで、残るのはソルバの収束誤差と丸めだけになる。

mod common;

use common::{BOTTOM_PID, Layer, TOP_PID, uniform_slab};
use gfem_core::case::{
    Backend, Case, Electrode, Preconditioner, ReferencePotential, SolverOptions,
};
use gfem_core::{Analysis, solve};

const CURRENT: f64 = 2.0e-3; // A
const LX: f64 = 0.060;
const LY: f64 = 0.040;
const LZ: f64 = 0.020;

fn tight_solver() -> SolverOptions {
    SolverOptions {
        tolerance: 1.0e-13,
        max_iterations: 20_000,
        preconditioner: Preconditioner::Jacobi,
        backend: Backend::CpuSerial,
    }
}

/// 上面を陽極、下面を陰極にしたケースを作る。
fn top_bottom_case(sigmas: &[(u32, f64)]) -> Case {
    let mut case = Case::new();
    for &(pid, s) in sigmas {
        case.set_conductivity(pid, s).unwrap();
    }
    case.add_electrode(Electrode::anode("top", CURRENT).with_shell_pids([TOP_PID]));
    case.add_electrode(Electrode::cathode("bottom", CURRENT).with_shell_pids([BOTTOM_PID]));
    case.solver = tight_solver();
    case
}

fn relative(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs()
}

#[test]
fn uniform_field_in_a_homogeneous_slab() {
    // SPEC §8 M2 の 1: 直方体、上下面に一様電流、σ 一定
    // → φ は線形、E = I/(σA) 一定。1 次要素なら厳密に再現されるべき
    let sigma = 0.35;
    let mesh = uniform_slab(LX, LY, LZ, 6, 4, 4);
    let case = top_bottom_case(&[(1, sigma)]);
    let a: Analysis = solve(&mesh, &case, None).unwrap();

    assert!(a.solver.converged, "{:?}", a.solver.failure_reason());

    let area = LX * LY;
    let expected_e = CURRENT / (sigma * area);
    let expected_j = CURRENT / area;

    for e in 0..a.fields.n_elements {
        assert!(
            relative(a.fields.e_magnitude[e], expected_e) < 1e-8,
            "要素 {e}: |E| = {} （期待 {expected_e}）",
            a.fields.e_magnitude[e]
        );
        assert!(relative(a.fields.j_magnitude[e], expected_j) < 1e-8);
        // 電流は下向き（陽極 → 陰極）。x, y 成分はゼロ
        assert!(a.fields.current_density[3 * e].abs() < 1e-10 * expected_j);
        assert!(a.fields.current_density[3 * e + 1].abs() < 1e-10 * expected_j);
        assert!(a.fields.current_density[3 * e + 2] < 0.0);
    }

    // 電位は z の線形関数
    let (lo, hi) = a.potential_range();
    let expected_drop = CURRENT * LZ / (sigma * area);
    assert!(relative(hi - lo, expected_drop) < 1e-8, "ΔV = {}", hi - lo);

    // 等価抵抗 R = L / (σA)（SPEC §5.3）
    let expected_r = LZ / (sigma * area);
    assert!(
        relative(a.impedance(), expected_r) < 1e-8,
        "R = {} Ω（期待 {expected_r} Ω）",
        a.impedance()
    );
}

#[test]
fn potential_is_linear_in_z() {
    let sigma = 0.35;
    let nz = 4;
    let mesh = uniform_slab(LX, LY, LZ, 3, 2, nz);
    let mut case = top_bottom_case(&[(1, sigma)]);
    // 陰極面をゼロ電位にすると φ = (I/(σA))·z になるはず
    case.reference = ReferencePotential::CathodeMean;
    let a = solve(&mesh, &case, None).unwrap();

    let slope = CURRENT / (sigma * LX * LY);
    for i in 0..mesh.nodes.len() {
        let z = mesh.nodes.coord(i as u32)[2];
        let expected = slope * z;
        assert!(
            (a.potential[i] - expected).abs() < 1e-9 * slope * LZ,
            "節点 {i} (z={z}): φ = {}（期待 {expected}）",
            a.potential[i]
        );
    }
    // 陽極の電位が高い（SPEC §2.1 の符号の約束）
    assert!(a.conservation.voltage > 0.0);
}

#[test]
fn layered_medium_splits_the_field_inversely_to_sigma() {
    // SPEC §8 M2 の 2: 直列に積んだ層。各層の E は σ に反比例して階段状になる
    let layers = [
        Layer {
            pid: 1,
            thickness: 0.002,
            divisions: 2,
        },
        Layer {
            pid: 2,
            thickness: 0.004,
            divisions: 2,
        },
        Layer {
            pid: 3,
            thickness: 0.014,
            divisions: 4,
        },
    ];
    let sigmas = [(1u32, 0.10), (2, 0.04), (3, 0.35)];
    let mesh = common::slab(LX, LY, 4, 3, &layers);
    let case = top_bottom_case(&sigmas);
    let a = solve(&mesh, &case, None).unwrap();
    assert!(a.solver.converged, "{:?}", a.solver.failure_reason());

    let area = LX * LY;
    let block = mesh.solids[0].pids();
    for (e, &pid) in block.iter().enumerate() {
        let sigma = sigmas.iter().find(|&&(p, _)| p == pid).unwrap().1;
        let expected_e = CURRENT / (sigma * area);
        assert!(
            relative(a.fields.e_magnitude[e], expected_e) < 1e-7,
            "PID {pid} の要素 {e}: |E| = {}（期待 {expected_e}）",
            a.fields.e_magnitude[e]
        );
        // |J| はどの層でも一定（電流連続）
        assert!(relative(a.fields.j_magnitude[e], CURRENT / area) < 1e-7);
    }

    // 直列抵抗 R = Σ t_i/(σ_i A)
    let expected_r: f64 = layers
        .iter()
        .zip(&sigmas)
        .map(|(l, &(_, s))| l.thickness / (s * area))
        .sum();
    assert!(
        relative(a.impedance(), expected_r) < 1e-7,
        "R = {} Ω（期待 {expected_r} Ω）",
        a.impedance()
    );
}

#[test]
fn conservation_checks_pass() {
    // SPEC §5.4: 電流保存・エネルギー収支・離散残差
    let mesh = uniform_slab(LX, LY, LZ, 5, 4, 3);
    let case = top_bottom_case(&[(1, 0.35)]);
    let a = solve(&mesh, &case, None).unwrap();

    assert!(
        a.conservation.is_clean(),
        "保存則チェックに警告: {:?}",
        a.conservation.warnings
    );
    assert!(a.conservation.current_balance.abs() < 1e-12);
    assert!(a.conservation.energy_relative_error < 1e-9);
    assert!(a.conservation.discrete_residual < 1e-12);

    // 消費電力 = I²R
    let expected_power = CURRENT * CURRENT * a.impedance();
    assert!(relative(a.conservation.power_from_field, expected_power) < 1e-8);
    assert!(a.warnings.is_empty(), "{:?}", a.warnings);
}

#[test]
fn solution_is_independent_of_the_mesh_density() {
    // 一様場は 1 次要素で厳密なので、粗いメッシュでも細かいメッシュでも同じ値になる
    let sigma = 0.35;
    let case = top_bottom_case(&[(1, sigma)]);
    let coarse = solve(&uniform_slab(LX, LY, LZ, 2, 2, 2), &case, None).unwrap();
    let fine = solve(&uniform_slab(LX, LY, LZ, 8, 6, 6), &case, None).unwrap();

    let expected = CURRENT / (sigma * LX * LY);
    assert!(relative(coarse.fields.e_magnitude[0], expected) < 1e-8);
    assert!(relative(fine.fields.e_magnitude[0], expected) < 1e-8);
    assert!(relative(coarse.impedance(), fine.impedance()) < 1e-8);
}

#[test]
fn repeated_runs_are_bitwise_identical() {
    // CLAUDE.md 設計原則 1: 同一入力からビット単位で同じ結果
    let mesh = uniform_slab(LX, LY, LZ, 4, 3, 3);
    let case = top_bottom_case(&[(1, 0.35)]);
    let a = solve(&mesh, &case, None).unwrap();
    let b = solve(&mesh, &case, None).unwrap();
    assert_eq!(a.potential, b.potential);
    assert_eq!(a.fields.electric_field, b.fields.electric_field);
    assert_eq!(a.solver.iterations, b.solver.iterations);
    assert_eq!(a.solver.history, b.solver.history);
}

#[test]
fn doubling_the_current_doubles_the_field() {
    // 線形問題であることの確認
    let mesh = uniform_slab(LX, LY, LZ, 3, 3, 3);
    let base = solve(&mesh, &top_bottom_case(&[(1, 0.35)]), None).unwrap();

    let mut doubled = top_bottom_case(&[(1, 0.35)]);
    for e in &mut doubled.electrodes {
        e.current *= 2.0;
    }
    let scaled = solve(&mesh, &doubled, None).unwrap();

    for e in 0..base.fields.n_elements {
        assert!(
            relative(
                scaled.fields.e_magnitude[e],
                2.0 * base.fields.e_magnitude[e]
            ) < 1e-9
        );
    }
    // 抵抗は電流に依らない
    assert!(relative(scaled.impedance(), base.impedance()) < 1e-9);
}

#[test]
fn swapping_the_electrodes_flips_the_sign() {
    let mesh = uniform_slab(LX, LY, LZ, 3, 2, 2);
    let forward = solve(&mesh, &top_bottom_case(&[(1, 0.35)]), None).unwrap();

    let mut reversed = Case::new();
    reversed.set_conductivity(1, 0.35).unwrap();
    reversed.add_electrode(Electrode::cathode("top", CURRENT).with_shell_pids([TOP_PID]));
    reversed.add_electrode(Electrode::anode("bottom", CURRENT).with_shell_pids([BOTTOM_PID]));
    reversed.solver = tight_solver();
    let backward = solve(&mesh, &reversed, None).unwrap();

    for i in 0..mesh.nodes.len() {
        assert!((forward.potential[i] + backward.potential[i]).abs() < 1e-12);
    }
    // 抵抗は向きに依らず正
    assert!(backward.impedance() > 0.0);
    assert!(relative(backward.impedance(), forward.impedance()) < 1e-9);
}
