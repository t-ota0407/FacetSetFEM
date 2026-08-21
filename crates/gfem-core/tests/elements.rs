//! 要素種を変えても同じ答えが出ることの検証（SPEC §8 M4）。
//!
//! 1 次要素（Tet4 / Hex8 / Penta6）はいずれも**線形場を厳密に再現する**ので、
//! 一様場・層状媒質の解析解に対しては要素種によらず同じ値になるはず。
//! 離散化誤差がゼロの問題を使うことで、形状関数・積分則・アセンブリの誤りを
//! そのまま検出できる。

mod common;

use common::{BOTTOM_PID, Layer, TOP_PID, slab_of};
use gfem_core::case::{
    Backend, Case, Electrode, Preconditioner, ReferencePotential, SolverOptions,
};
use gfem_core::{Analysis, ElementKind, Mesh, solve};

const CURRENT: f64 = 2.0e-3;
const LX: f64 = 0.060;
const LY: f64 = 0.040;
const AREA: f64 = LX * LY;

/// (PID, 厚さ [m], σ [S/m])
const LAYERS: [(u32, f64, f64); 3] = [(1, 0.002, 0.10), (2, 0.004, 0.04), (3, 0.014, 0.35)];

const SOLID_KINDS: [ElementKind; 3] = [ElementKind::Tet4, ElementKind::Hex8, ElementKind::Penta6];

fn layered_mesh(kind: ElementKind) -> Mesh {
    slab_of(
        kind,
        LX,
        LY,
        4,
        3,
        &[
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
        ],
    )
}

fn layered_case() -> Case {
    let mut case = Case::new();
    for (pid, _t, sigma) in LAYERS {
        case.set_conductivity(pid, sigma).unwrap();
    }
    case.add_electrode(Electrode::anode("top", CURRENT).with_shell_pids([TOP_PID]));
    case.add_electrode(Electrode::cathode("bottom", CURRENT).with_shell_pids([BOTTOM_PID]));
    case.solver = SolverOptions {
        tolerance: 1.0e-13,
        max_iterations: 50_000,
        preconditioner: Preconditioner::Jacobi,
        backend: Backend::CpuSerial,
        ..Default::default()
    };
    case.reference = ReferencePotential::CathodeMean;
    case
}

fn expected_resistance() -> f64 {
    LAYERS.iter().map(|&(_, t, s)| t / (s * AREA)).sum()
}

fn relative(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs()
}

#[test]
fn every_element_type_reproduces_the_layered_solution() {
    for kind in SOLID_KINDS {
        let mesh = layered_mesh(kind);
        let a: Analysis = solve(&mesh, &layered_case(), None).unwrap();
        assert!(
            a.solver.converged,
            "{}: {:?}",
            kind.card_name(),
            a.solver.failure_reason()
        );
        assert!(
            a.conservation.is_clean(),
            "{}: {:?}",
            kind.card_name(),
            a.conservation.warnings
        );

        // 各層で |E| = I/(σA)、|J| はどこでも I/A
        let block = &mesh.solids[0];
        for e in 0..a.fields.n_elements {
            let pid = block.pids()[e];
            let sigma = LAYERS.iter().find(|&&(p, _, _)| p == pid).unwrap().2;
            let expected_e = CURRENT / (sigma * AREA);
            assert!(
                relative(a.fields.e_magnitude[e], expected_e) < 1e-8,
                "{} PID {pid}: |E| = {}（期待 {expected_e}）",
                kind.card_name(),
                a.fields.e_magnitude[e]
            );
            assert!(relative(a.fields.j_magnitude[e], CURRENT / AREA) < 1e-8);
            // 電流は下向き
            assert!(a.fields.current_density[3 * e + 2] < 0.0);
        }

        // 直列抵抗
        assert!(
            relative(a.impedance(), expected_resistance()) < 1e-8,
            "{}: R = {}（期待 {}）",
            kind.card_name(),
            a.impedance(),
            expected_resistance()
        );
    }
}

#[test]
fn element_types_agree_with_each_other() {
    let reference = solve(&layered_mesh(ElementKind::Tet4), &layered_case(), None).unwrap();
    for kind in [ElementKind::Hex8, ElementKind::Penta6] {
        let a = solve(&layered_mesh(kind), &layered_case(), None).unwrap();
        // 同じ格子から作っているので節点も同じ並び。電位が一致する
        assert_eq!(a.potential.len(), reference.potential.len());
        let scale = reference
            .potential
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()));
        for (i, (x, y)) in a.potential.iter().zip(&reference.potential).enumerate() {
            assert!(
                (x - y).abs() < 1e-9 * scale,
                "{} 節点 {i}: {x} vs {y}",
                kind.card_name()
            );
        }
        assert!(relative(a.impedance(), reference.impedance()) < 1e-9);
        assert!(
            relative(
                a.conservation.power_from_field,
                reference.conservation.power_from_field
            ) < 1e-9
        );
    }
}

#[test]
fn element_volumes_fill_the_model() {
    for kind in SOLID_KINDS {
        let mesh = layered_mesh(kind);
        let a = solve(&mesh, &layered_case(), None).unwrap();
        let total: f64 = a.fields.volume.iter().sum();
        let expected = AREA * LAYERS.iter().map(|&(_, t, _)| t).sum::<f64>();
        assert!(
            relative(total, expected) < 1e-12,
            "{}: 総体積 {total}（期待 {expected}）",
            kind.card_name()
        );
    }
}

#[test]
fn region_stats_agree_across_element_types() {
    let case = layered_case();
    let mut reference: Vec<f64> = Vec::new();
    for kind in SOLID_KINDS {
        let mesh = layered_mesh(kind);
        let a = solve(&mesh, &case, None).unwrap();
        let stats = a.region_stats(&mesh, &case).unwrap();
        assert_eq!(stats.len(), 3);
        let powers: Vec<f64> = stats.iter().map(|s| s.power).collect();
        if reference.is_empty() {
            reference = powers;
        } else {
            for (a, b) in powers.iter().zip(&reference) {
                assert!(
                    relative(*a, *b) < 1e-9,
                    "{}: 消費電力が食い違います {a} vs {b}",
                    kind.card_name()
                );
            }
        }
    }
}

#[test]
fn quad_electrode_faces_carry_the_same_current() {
    // Hex8 メッシュの電極面は CQUAD4 になる。面積按分（SPEC §2.4）が
    // 三角形の場合と同じ総電流を与えることを確かめる
    let mesh = layered_mesh(ElementKind::Hex8);
    assert_eq!(mesh.shells[0].kind(), ElementKind::Quad4);
    let a = solve(&mesh, &layered_case(), None).unwrap();
    let anode = &a.electrodes[0];
    assert!((anode.area - AREA).abs() < 1e-15 * AREA);
    let injected: f64 = anode
        .node_weights
        .iter()
        .map(|&(_, w)| w * anode.current_density)
        .sum();
    assert!((injected - CURRENT).abs() < 1e-15);
}

#[test]
fn results_are_bitwise_repeatable_for_every_element_type() {
    for kind in SOLID_KINDS {
        let mesh = layered_mesh(kind);
        let a = solve(&mesh, &layered_case(), None).unwrap();
        let b = solve(&mesh, &layered_case(), None).unwrap();
        assert_eq!(a.potential, b.potential, "{}", kind.card_name());
        assert_eq!(a.fields.electric_field, b.fields.electric_field);
        assert_eq!(a.solver.history, b.solver.history);
    }
}
