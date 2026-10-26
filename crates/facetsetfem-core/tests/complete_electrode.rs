//! 完全電極モデル（CEM。SPEC §2.5 / §8 M11）の検証。
//!
//! 1. **厳密解**: 上下面を丸ごと電極にした直方体は 1 次元の直列抵抗
//!    `R = L/(σA) + z_a/A + z_c/A` になり、1 次要素で厳密に再現される
//! 2. **極限**: z → ∞ で一様電流密度（SPEC §2.4）に近づき、z が小さいと電流が縁に集まる
//! 3. **重ね合わせ**: 3 電極・一様電流密度との混在でも、電極ごとの電流が指定どおりになり、
//!    基準の電極の選び方（電極の並び）に答えが依らない
//! 4. **保存則**: 電流・エネルギー（接触面の消費を含む）・離散残差（SPEC §5.4）
//! 5. **再現性**: CPU 逐次と CPU 並列でビット単位で一致（設計原則 1）
//! 6. **半無限媒質上の正方形電極**: 等電位の正方形板（容量から）と一様電流密度の正方形の抵抗の差
//!    （`#[ignore]`。重いので検証一式のときだけ）

mod common;

use common::{BOTTOM_PID, Patch, TOP_PID, box_mesh, linear_grid, uniform_slab};
use facetsetfem_core::analysis::solve_with_backend;
use facetsetfem_core::case::{
    Backend, Case, Electrode, ElectrodeModel, FaceRef, Preconditioner, ReferencePotential,
    SolverOptions,
};
use facetsetfem_core::solver::{CpuParallel, CpuSerial};
use facetsetfem_core::{Analysis, Gauge, Mesh, solve};

const CURRENT: f64 = 2.0e-3; // A
const SIGMA: f64 = 0.35; // S/m
const LX: f64 = 0.020;
const LY: f64 = 0.010;
const LZ: f64 = 0.030;

/// 上面の中央だけに貼る小さな陽極の PID（下面は全面が陰極）。
const PATCH_PID: u32 = 301;

fn solver(preconditioner: Preconditioner, backend: Backend) -> SolverOptions {
    SolverOptions {
        tolerance: 1.0e-12,
        max_iterations: 20_000,
        preconditioner,
        backend,
        ..Default::default()
    }
}

fn relative(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs()
}

/// 上下面が丸ごと電極の直方体。`anode_z` / `cathode_z` が `None` なら一様電流密度。
fn bar_case(anode_z: Option<f64>, cathode_z: Option<f64>) -> Case {
    let mut case = Case::new();
    case.set_conductivity(1, SIGMA).unwrap();
    let mut anode = Electrode::anode("top", CURRENT).with_shell_pids([TOP_PID]);
    if let Some(z) = anode_z {
        anode = anode.with_contact_impedance(z);
    }
    let mut cathode = Electrode::cathode("bottom", CURRENT).with_shell_pids([BOTTOM_PID]);
    if let Some(z) = cathode_z {
        cathode = cathode.with_contact_impedance(z);
    }
    case.add_electrode(anode);
    case.add_electrode(cathode);
    case.solver = solver(Preconditioner::Jacobi, Backend::CpuSerial);
    case
}

#[test]
fn a_bar_with_full_face_electrodes_is_a_series_resistance() {
    // 1 次元の問題なので φ は z の線形関数、接触電流密度はどこでも I/A。
    // 1 次要素で厳密に表せるので、残るのはソルバの収束誤差と丸めだけ
    let mesh = uniform_slab(LX, LY, LZ, 4, 2, 6);
    let area = LX * LY;
    let (za, zc) = (0.012, 0.004); // Ω·m²
    let a = solve(&mesh, &bar_case(Some(za), Some(zc)), None).unwrap();
    assert_eq!(a.gauge, Gauge::NotNeeded);
    assert_eq!(a.solves.len(), 1, "電極 2 枚とも CEM なら 1 回で解ける");

    let expected = LZ / (SIGMA * area) + za / area + zc / area;
    assert!(
        relative(a.impedance(), expected) < 1e-9,
        "R = {} Ω（期待 {expected} Ω）",
        a.impedance()
    );
    let j = CURRENT / area;
    for e in 0..a.fields.n_elements {
        assert!(relative(a.fields.j_magnitude[e], j) < 1e-8, "要素 {e}");
    }
    for state in a.electrode_states() {
        assert!(relative(state.peak_current_density, j) < 1e-8);
        assert!(relative(state.current.abs(), CURRENT) < 1e-9);
    }
    // 板と皮膚の電位差は I z / A（陽極は板が高い）
    let states = a.electrode_states();
    assert!(
        relative(
            states[0].potential - states[0].mean_surface_potential,
            CURRENT * za / area
        ) < 1e-8
    );
    assert!(
        relative(
            states[1].mean_surface_potential - states[1].potential,
            CURRENT * zc / area
        ) < 1e-8
    );
    // 接触面の消費は I² z / A
    let contact = CURRENT * CURRENT * (za + zc) / area;
    assert!(relative(a.conservation.contact_power, contact) < 1e-8);
    assert!(a.conservation.is_clean(), "{:?}", a.conservation.warnings);
}

#[test]
fn quadrilateral_faces_and_body_surface_faces_give_the_same_series_resistance() {
    // Hex8 の電極面は CQUAD4（面の質量行列は 2×2 ガウス積分。SPEC §2.5）。
    // シェルを使わず体表面の面で指定した電極（外表面の面から下の要素を引く経路）も同じになる
    let area = LX * LY;
    let (za, zc) = (0.012, 0.004);
    let expected = LZ / (SIGMA * area) + za / area + zc / area;

    let hex = common::slab_of(
        facetsetfem_core::ElementKind::Hex8,
        LX,
        LY,
        4,
        2,
        &[common::Layer {
            pid: 1,
            thickness: LZ,
            divisions: 6,
        }],
    );
    let a = solve(&hex, &bar_case(Some(za), Some(zc)), None).unwrap();
    assert!(
        relative(a.impedance(), expected) < 1e-9,
        "Hex8: R = {}",
        a.impedance()
    );
    assert!(a.conservation.is_clean(), "{:?}", a.conservation.warnings);

    let mesh = uniform_slab(LX, LY, LZ, 4, 2, 6);
    let mut case = Case::new();
    case.set_conductivity(1, SIGMA).unwrap();
    case.add_electrode(
        Electrode::anode("top", CURRENT)
            .with_added_faces(body_surface_faces(&mesh, LZ))
            .with_contact_impedance(za),
    );
    case.add_electrode(
        Electrode::cathode("bottom", CURRENT)
            .with_added_faces(body_surface_faces(&mesh, 0.0))
            .with_contact_impedance(zc),
    );
    case.solver = solver(Preconditioner::Jacobi, Backend::CpuSerial);
    let b = solve(&mesh, &case, None).unwrap();
    assert!(
        relative(b.impedance(), expected) < 1e-9,
        "パッチ: R = {}",
        b.impedance()
    );
    // 電極の下の組織の σ（縁の分解能の目安 ℓ = zσ に使う）が、外表面の面から引けている
    for e in &b.electrodes {
        assert!(
            relative(e.underlying_sigma, SIGMA) < 1e-15,
            "{}",
            e.underlying_sigma
        );
    }
}

/// 外表面の面のうち、全節点が z = `z` の平面にあるもの（体表面から手で足した面として指定する）。
fn body_surface_faces(mesh: &Mesh, z: f64) -> Vec<FaceRef> {
    let surface = facetsetfem_core::mesh::topology::extract_surface(mesh).unwrap();
    surface
        .faces()
        .iter()
        .filter(|f| {
            f.nodes()
                .iter()
                .all(|&n| (mesh.nodes.coord(n)[2] - z).abs() < 1e-12)
        })
        .map(|f| FaceRef::new(mesh.solid(f.element as usize).id, f.local_face))
        .collect()
}

#[test]
fn one_complete_electrode_with_one_uniform_electrode() {
    // 陰極は一様電流密度のまま。その電位は面積重み平均電位なので、接触抵抗は陽極の分だけ
    let mesh = uniform_slab(LX, LY, LZ, 4, 2, 6);
    let area = LX * LY;
    let za = 0.02;
    let a = solve(&mesh, &bar_case(Some(za), None), None).unwrap();
    assert_eq!(
        a.solves.len(),
        1,
        "CEM 1 枚 + 一様 1 枚は 1 回（一様電流密度の荷重だけ）"
    );
    let expected = LZ / (SIGMA * area) + za / area;
    assert!(
        relative(a.impedance(), expected) < 1e-9,
        "R = {}",
        a.impedance()
    );
    assert!(a.electrodes[1].terminal_potential.is_none());
    assert!(a.conservation.is_clean(), "{:?}", a.conservation.warnings);
}

#[test]
fn uniform_electrodes_alone_keep_the_singular_system() {
    // CEM を選ばなければ、それまでと同じ射影 CG（SPEC §2.5 の「既定は一様電流密度のまま」）
    let mesh = uniform_slab(LX, LY, LZ, 4, 2, 6);
    let a = solve(&mesh, &bar_case(None, None), None).unwrap();
    assert_eq!(a.gauge, Gauge::MeanZero);
    assert_eq!(a.solves.len(), 1);
    assert_eq!(a.conservation.contact_power, 0.0);
    let expected = LZ / (SIGMA * LX * LY);
    assert!(relative(a.impedance(), expected) < 1e-9);
}

/// 上面の中央に小さな陽極、下面の全面に陰極を置いた直方体。
fn patch_mesh() -> Mesh {
    let xs = linear_grid(0.0, LX, 20);
    let ys = linear_grid(0.0, LY, 10);
    let zs = linear_grid(0.0, LZ, 12);
    box_mesh(
        &xs,
        &ys,
        &zs,
        &[1; 12],
        &[
            Patch {
                pid: PATCH_PID,
                x: [0.006, 0.014],
                y: [0.003, 0.007],
                top: true,
            },
            Patch::whole(BOTTOM_PID, false),
        ],
    )
}

fn patch_case(model: ElectrodeModel) -> Case {
    let mut case = Case::new();
    case.set_conductivity(1, SIGMA).unwrap();
    let mut anode = Electrode::anode("E1", CURRENT).with_shell_pids([PATCH_PID]);
    let mut cathode = Electrode::cathode("E2", CURRENT).with_shell_pids([BOTTOM_PID]);
    anode.model = model;
    cathode.model = model;
    case.add_electrode(anode);
    case.add_electrode(cathode);
    case.solver = solver(Preconditioner::Jacobi, Backend::CpuSerial);
    case
}

fn complete(z: f64) -> ElectrodeModel {
    ElectrodeModel::Complete {
        contact_impedance: z,
    }
}

#[test]
fn a_large_contact_impedance_approaches_the_uniform_current_density() {
    // z → ∞ で接触電流密度の中で φ のばらつきが効かなくなり、一様電流密度に近づく（SPEC §2.5）。
    // 差は 1/z で縮む
    let mesh = patch_mesh();
    let uniform = solve(
        &mesh,
        &patch_case(ElectrodeModel::UniformCurrentDensity),
        None,
    )
    .unwrap();
    let differences: Vec<(f64, f64)> = [1.0, 1.0e2]
        .iter()
        .map(|&z| {
            let a = solve(&mesh, &patch_case(complete(z)), None).unwrap();
            let states = a.electrode_states();
            // 板と皮膚の平均電位の差は、電流の定義 I = (U A − ∫φ dS)/z からちょうど I z / A
            let area_a = a.electrodes[0].area;
            let contact = states[0].potential - states[0].mean_surface_potential;
            assert!(relative(contact, CURRENT * z / area_a) < 1e-9, "z = {z}");
            // 体の中の抵抗（皮膚の平均電位の差 / I）は一様電流密度の R に近づく。
            // R から z/A を引くと、z が大きいとき桁が落ちるのでこちらで比べる
            let r_body =
                (states[0].mean_surface_potential - states[1].mean_surface_potential) / CURRENT;
            let dr = relative(r_body, uniform.impedance());
            let dj = a
                .fields
                .j_magnitude
                .iter()
                .zip(&uniform.fields.j_magnitude)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f64, f64::max)
                / uniform
                    .fields
                    .j_magnitude
                    .iter()
                    .copied()
                    .fold(0.0, f64::max);
            (dr, dj)
        })
        .collect();
    let (dr_small, dj_small) = differences[0];
    let (dr_large, dj_large) = differences[1];
    assert!(
        dj_large < 1e-5,
        "z = 100 でも |J| が一様電流密度と {dj_large:.2e} ずれる"
    );
    assert!(
        dr_large < 1e-6,
        "z = 100 でも R が一様電流密度と {dr_large:.2e} ずれる"
    );
    // z を 100 倍にすると、ずれはおよそ 1/100 になる（z をさらに大きくすると CG の収束誤差に埋もれる）
    assert!(
        dj_large < dj_small / 30.0,
        "{dj_small:.2e} → {dj_large:.2e}"
    );
    assert!(
        dr_large < dr_small / 30.0,
        "{dr_small:.2e} → {dr_large:.2e}"
    );
}

#[test]
fn a_small_contact_impedance_gathers_current_at_the_edge() {
    let mesh = patch_mesh();
    let uniform = solve(
        &mesh,
        &patch_case(ElectrodeModel::UniformCurrentDensity),
        None,
    )
    .unwrap();
    let small = solve(&mesh, &patch_case(complete(1.0e-5)), None).unwrap();
    let mean = CURRENT / uniform.electrodes[0].area;
    let u = uniform.electrode_states()[0];
    let s = small.electrode_states()[0];
    assert!(
        relative(u.peak_current_density, mean) < 1e-12,
        "一様電流密度のピークは I/A"
    );
    assert!(
        s.peak_current_density > 2.0 * mean,
        "縁に集まっていない: ピーク {} / 平均 {mean}",
        s.peak_current_density
    );
    // 縁の節点で最大になる（中央の節点より大きい）
    let j = small.electrode_current_density(&mesh);
    let centre = common::nearest_node(&mesh, [0.010, 0.005, LZ]) as usize;
    let corner = common::nearest_node(&mesh, [0.006, 0.003, LZ]) as usize;
    assert!(
        j[corner] > 2.0 * j[centre],
        "角 {} / 中央 {}",
        j[corner],
        j[centre]
    );
    // 等電位に近いので、同じ電流なら体の中の抵抗は一様電流密度より小さい（SPEC §2.5）
    let r_body = small.impedance()
        - 1.0e-5 * (1.0 / small.electrodes[0].area + 1.0 / small.electrodes[1].area);
    assert!(
        r_body < uniform.impedance(),
        "{r_body} vs {}",
        uniform.impedance()
    );
    // この粗いメッシュでは ℓ = zσ が要素より小さいので、分解能の警告が出る
    assert!(
        small.warnings.iter().any(|w| w.contains("ℓ = zσ")),
        "{:?}",
        small.warnings
    );
}

#[test]
fn conservation_holds_with_complete_electrodes() {
    let mesh = patch_mesh();
    let a = solve(&mesh, &patch_case(complete(1.0e-3)), None).unwrap();
    let c = &a.conservation;
    assert!(
        c.electrode_current_error < 1e-12,
        "{}",
        c.electrode_current_error
    );
    assert!(c.current_balance.abs() < 1e-12, "{}", c.current_balance);
    assert!(
        c.energy_relative_error < 1e-8,
        "{}",
        c.energy_relative_error
    );
    assert!(c.discrete_residual < 1e-10, "{}", c.discrete_residual);
    assert!(c.contact_power > 0.0);
    // 電極から入る電力 = 体内の消費 + 接触面の消費
    assert!(relative(c.power_from_load, c.power_from_field + c.contact_power) < 1e-8);
}

#[test]
fn a_small_contact_impedance_does_not_raise_false_conservation_warnings() {
    // 接触抵抗が体の抵抗よりずっと小さいと、基準の電極の電流（＝電流の総和）のずれは
    // CG の残差 −1ᵀ(Aφ − f) が拡大されて 1e-9 を超える。以前はこれを「電流が合わない」と
    // 誤って警告していた（既定の収束判定 1e-10 で 3.6e-9）。収束判定から見込まれる大きさと比べる
    let mesh = patch_mesh();
    let mut case = patch_case(complete(1.0e-5));
    case.solver.tolerance = 1.0e-10;
    let a = solve(&mesh, &case, None).unwrap();
    let c = &a.conservation;
    assert!(c.is_clean(), "{:?}", c.warnings);
    assert!(
        c.electrode_current_error < 1e-12,
        "{}",
        c.electrode_current_error
    );
    // 離散残差は表示の基準電位に左右されない（割る量は CG が抑えた右辺の大きさ）
    case.reference = ReferencePotential::CathodeMean;
    let b = solve(&mesh, &case, None).unwrap();
    assert!(
        relative(b.conservation.discrete_residual, c.discrete_residual) < 1e-3,
        "{} vs {}",
        b.conservation.discrete_residual,
        c.discrete_residual
    );
}

/// 上面に陽極を 1 枚、下面に陰極を 2 枚（一方は一様電流密度でもよい）。
fn three_electrode_mesh() -> Mesh {
    let xs = linear_grid(0.0, LX, 20);
    let ys = linear_grid(0.0, LY, 10);
    let zs = linear_grid(0.0, LZ, 12);
    box_mesh(
        &xs,
        &ys,
        &zs,
        &[1; 12],
        &[
            Patch {
                pid: PATCH_PID,
                x: [0.006, 0.014],
                y: [0.003, 0.007],
                top: true,
            },
            Patch {
                pid: 401,
                x: [0.0, 0.008],
                y: [0.0, LY],
                top: false,
            },
            Patch {
                pid: 402,
                x: [0.012, LX],
                y: [0.0, LY],
                top: false,
            },
        ],
    )
}

fn three_electrode_case(order: [usize; 3], third_uniform: bool) -> Case {
    let mut electrodes = [
        Electrode::anode("A", 3.0e-3)
            .with_shell_pids([PATCH_PID])
            .with_contact_impedance(2.0e-3),
        Electrode::cathode("C1", 1.0e-3)
            .with_shell_pids([401])
            .with_contact_impedance(5.0e-3),
        Electrode::cathode("C2", 2.0e-3).with_shell_pids([402]),
    ];
    if !third_uniform {
        electrodes[2] = electrodes[2].clone().with_contact_impedance(1.0e-3);
    }
    let mut case = Case::new();
    case.set_conductivity(1, SIGMA).unwrap();
    for i in order {
        case.add_electrode(electrodes[i].clone());
    }
    case.solver = solver(Preconditioner::Jacobi, Backend::CpuSerial);
    case.solver.tolerance = 1.0e-13;
    case
}

#[test]
fn three_electrodes_are_superposed_correctly() {
    let mesh = three_electrode_mesh();
    for third_uniform in [false, true] {
        let a = solve(&mesh, &three_electrode_case([0, 1, 2], third_uniform), None).unwrap();
        // 3 枚とも CEM なら 2 回、CEM 2 枚 + 一様 1 枚でも 2 回（SPEC §2.5 の「解く回数」）
        assert_eq!(a.solves.len(), 2, "third_uniform = {third_uniform}");
        let expected = [3.0e-3, -1.0e-3, -2.0e-3];
        for (state, want) in a.electrode_states().iter().zip(expected) {
            assert!(
                relative(state.current, want) < 1e-10,
                "{} vs {want}",
                state.current
            );
        }
        assert!(a.conservation.is_clean(), "{:?}", a.conservation.warnings);

        // 電極の並びを変えると基準の電極が変わるが、E・J と電極間の電位差は同じ
        let b = solve(&mesh, &three_electrode_case([2, 0, 1], third_uniform), None).unwrap();
        let scale = a.fields.j_magnitude.iter().copied().fold(0.0, f64::max);
        let worst = a
            .fields
            .j_magnitude
            .iter()
            .zip(&b.fields.j_magnitude)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f64, f64::max);
        assert!(worst < 1e-8 * scale, "並びで |J| が {worst:.2e} 変わった");
        let pa = a.electrode_states();
        let pb = b.electrode_states();
        // b の並びは [C2, A, C1]
        let da = pa[0].potential - pa[1].potential;
        let db = pb[1].potential - pb[2].potential;
        assert!(relative(db, da) < 1e-9, "{da} vs {db}");
    }
}

#[test]
fn serial_and_parallel_backends_agree_bit_for_bit() {
    // 設計原則 1。重ね合わせ・コンダクタンス行列・足し合わせも含めて一致すること
    let mesh = three_electrode_mesh();
    for preconditioner in [Preconditioner::Jacobi, Preconditioner::Amg] {
        let mut case = three_electrode_case([0, 1, 2], true);
        case.solver.preconditioner = preconditioner;
        let serial = solve_with_backend(CpuSerial, &mesh, &case, None).unwrap();
        let parallel = solve_with_backend(CpuParallel, &mesh, &case, None).unwrap();
        assert_bitwise(&serial, &parallel);
    }
}

fn assert_bitwise(a: &Analysis, b: &Analysis) {
    assert_eq!(a.potential, b.potential);
    assert_eq!(a.fields.current_density, b.fields.current_density);
    assert_eq!(a.solves, b.solves);
    let ua: Vec<_> = a.electrodes.iter().map(|e| e.terminal_potential).collect();
    let ub: Vec<_> = b.electrodes.iter().map(|e| e.terminal_potential).collect();
    assert_eq!(ua, ub);
    assert_eq!(a.impedance().to_bits(), b.impedance().to_bits());
}

#[test]
fn amg_and_jacobi_agree_within_the_tolerance() {
    let mesh = patch_mesh();
    let mut case = patch_case(complete(1.0e-3));
    let jacobi = solve(&mesh, &case, None).unwrap();
    case.solver.preconditioner = Preconditioner::Amg;
    let amg = solve(&mesh, &case, None).unwrap();
    assert!(amg.solver.amg.is_some());
    assert!(amg.solver.iterations < jacobi.solver.iterations);
    assert!(relative(amg.impedance(), jacobi.impedance()) < 1e-9);
    assert!(
        amg.conservation.is_clean(),
        "{:?}",
        amg.conservation.warnings
    );
}

#[test]
fn the_cathode_plate_can_be_the_reference_potential() {
    let mesh = patch_mesh();
    let mut case = patch_case(complete(1.0e-3));
    case.reference = ReferencePotential::CathodeMean;
    let a = solve(&mesh, &case, None).unwrap();
    assert_eq!(a.electrodes[1].terminal_potential, Some(0.0));
    // 平均ゼロの基準なら φ の平均がゼロ
    case.reference = ReferencePotential::ZeroMean;
    let b = solve(&mesh, &case, None).unwrap();
    let mean: f64 = b.potential.iter().sum::<f64>() / b.potential.len() as f64;
    let (lo, hi) = b.potential_range();
    assert!(mean.abs() < 1e-12 * (hi - lo), "平均 {mean}");
    // 基準を変えても電位差・電流・保存則は変わらない
    assert!(relative(a.impedance(), b.impedance()) < 1e-12);
    assert!(b.conservation.is_clean(), "{:?}", b.conservation.warnings);
}

#[test]
fn invalid_contact_impedances_are_rejected() {
    let mesh = uniform_slab(LX, LY, LZ, 2, 2, 2);
    for z in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let err = solve(&mesh, &bar_case(Some(z), Some(0.01)), None).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("接触インピーダンス"), "{text}");
        assert!(text.contains("top"), "どの電極か書いていない: {text}");
    }
}

#[test]
fn the_summary_records_the_electrode_model() {
    let mesh = three_electrode_mesh();
    let a = solve(&mesh, &three_electrode_case([0, 1, 2], true), None).unwrap();
    let case = three_electrode_case([0, 1, 2], true);
    let provenance = facetsetfem_core::Provenance::default();
    let json = a
        .summary_json(&mesh, &case, &provenance)
        .unwrap()
        .to_pretty_string();
    assert!(
        json.contains("\"distribution\": \"complete_electrode\""),
        "{json}"
    );
    assert!(json.contains("\"contact_impedance_ohm_m2\""));
    assert!(json.contains("\"electrode_potential_V\""));
    assert!(json.contains("\"gauge\": \"none\""));
    assert!(json.contains("\"n_solves\": 2"));
    assert!(json.contains("\"solves\""));
}

/// 半無限媒質上の正方形電極（一辺 s）。等電位の正方形板は容量から `R = 1/(2π C σ s)`
/// （C = 0.3667874。単位正方形の板の容量を 4πε₀ 単位で表した値。Read 1997 / Hwang & Mascagni 2004。
/// 2026-09 に境界要素法で 0.36685 と確かめた）、一様電流密度は面積重み平均電位で
/// `R = (4 ln(1+√2) − 4(√2−1)/3) / (2πσ s)`（単位正方形の 2 点間距離の逆数の平均）。
///
/// 有限の箱で打ち切った分と遠くの陰極までの抵抗は両者で同じなので（箱を 8 倍にしても差は 4 桁
/// 変わらないことを確かめた）、**差** `≈ 0.03928/(σs)` を比べる。接触インピーダンスが小さい CEM
/// （zσ/s = 10⁻⁴）を等電位の代わりにし、接触抵抗の分 z/A を引く。円板（Newman の解）ではなく
/// 正方形にしたのは、格子に沿うので電極の形がメッシュで崩れないため（円板は三角形の階段状になり、
/// その誤差がメッシュを細かくしても遅くしか減らない。試した時点で外挿しても 11 % ずれた）。
///
/// 等電位の縁の特異性のため、格子は縁へ向かって等比で細かくする。それでも R_等電位 は下から
/// ゆっくり近づく（有限要素法の R は真の値以下）ので、h を半分ずつにした 3 つから外挿する（Aitken）。
#[test]
#[ignore = "重い（最大 110 万四面体を 6 回解く）。cargo test --release -- --include-ignored で回す"]
fn a_square_on_a_half_space_matches_the_plate_capacitance() {
    let side = 0.02; // 正方形の一辺 [m]
    let capacitance = 0.366_787_4;
    let mean_inverse_distance = 4.0 * (1.0 + 2f64.sqrt()).ln() - 4.0 * (2f64.sqrt() - 1.0) / 3.0;
    let pi2 = 2.0 * std::f64::consts::PI;
    let expected = mean_inverse_distance / pi2 - 1.0 / (pi2 * capacitance);
    let mut results = Vec::new();
    for divisions in [6.0, 12.0, 24.0] {
        let (difference, r_uniform, r_eq) = square_difference(side, side / divisions);
        println!(
            "h = s/{divisions}: R_一様 = {r_uniform:.4} Ω、R_等電位 = {r_eq:.4} Ω、             差 × σs = {difference:.5}（理論 {expected:.5}）"
        );
        results.push(difference);
    }
    // メッシュを細かくするほど理論値に近づくこと
    for w in results.windows(2) {
        assert!(
            (w[1] - expected).abs() < (w[0] - expected).abs(),
            "メッシュを細かくしても理論値に近づかない: {results:?}"
        );
    }
    let (d1, d2) = (results[1] - results[0], results[2] - results[1]);
    let order = (d1 / d2).log2();
    let extrapolated = results[2] + d2 / (d1 / d2 - 1.0);
    println!("推定した収束次数 {order:.2}、外挿 {extrapolated:.5}（理論 {expected:.5}）");
    assert!(
        relative(extrapolated, expected) < 0.01,
        "外挿した差 × σs = {extrapolated:.5}（期待 {expected:.5}）"
    );
    // 外挿しなくても、いちばん細かいメッシュで 5 % 以内
    assert!(relative(results[2], expected) < 0.05, "{results:?}");
}

/// 正方形電極の R（一様電流密度）と R（ほぼ等電位の CEM）の差 × σs と、両方の R。
///
/// 格子は電極の縁（x, y = ±s/2）と表面（z = 0）に向かって等比で細かくし（最小 `h_cap / 8`）、
/// 電極の近く（|x| < 0.75 s・深さ 0.75 s まで）では間隔を `h_cap` 以下に抑える。
/// 縁の特異性を局所的に解像しないと、一様に細かくしても収束が遅すぎて外挿も効かない。
fn square_difference(side: f64, h_cap: f64) -> (f64, f64, f64) {
    let half = 0.5 * side;
    let (h_min, ratio, outer) = (h_cap / 8.0, 1.3, 20.0 * half);
    let positive = edge_graded(Some(half), h_min, h_cap, 1.5 * half, ratio, outer);
    let mut xs: Vec<f64> = positive.iter().skip(1).rev().map(|v| -v).collect();
    xs.extend(&positive);
    let depth = edge_graded(None, h_min, h_cap, 1.5 * half, ratio, outer);
    let zs: Vec<f64> = depth.iter().rev().map(|d| -d).collect();
    let mesh = box_mesh(
        &xs,
        &xs,
        &zs,
        &vec![1; zs.len() - 1],
        &[
            Patch {
                pid: PATCH_PID,
                x: [-half, half],
                y: [-half, half],
                top: true,
            },
            Patch::whole(BOTTOM_PID, false),
        ],
    );
    let run = |model: ElectrodeModel| {
        let mut case = Case::new();
        case.set_conductivity(1, SIGMA).unwrap();
        let mut anode = Electrode::anode("square", CURRENT).with_shell_pids([PATCH_PID]);
        anode.model = model;
        case.add_electrode(anode);
        case.add_electrode(Electrode::cathode("far", CURRENT).with_shell_pids([BOTTOM_PID]));
        case.solver = solver(Preconditioner::Amg, Backend::CpuParallel);
        case.solver.tolerance = 1.0e-10;
        solve(&mesh, &case, None).unwrap()
    };
    let uniform = run(ElectrodeModel::UniformCurrentDensity);
    let z = 1.0e-4 * side / SIGMA; // zσ/s = 1e-4: ほぼ等電位
    let equipotential = run(complete(z));
    let area = uniform.electrodes[0].area;
    assert!(relative(area, side * side) < 1e-12, "正方形の面積 {area}");
    let r_eq = equipotential.impedance() - z / area;
    let difference = (uniform.impedance() - r_eq) * SIGMA * side;
    (difference, uniform.impedance(), r_eq)
}

/// 0 から `outer` までの昇順の格子。`edge` があればそこ、無ければ 0 で間隔が `h_min` になり、
/// 離れるほど `ratio` 倍ずつ広がる。`cap_to` より内側では間隔を `h_cap` 以下に抑える。
fn edge_graded(
    edge: Option<f64>,
    h_min: f64,
    h_cap: f64,
    cap_to: f64,
    ratio: f64,
    outer: f64,
) -> Vec<f64> {
    let centre = edge.unwrap_or(0.0);
    let mut points = vec![centre];
    // 内側（0 へ向かう）
    if centre > 0.0 {
        let (mut x, mut step) = (centre, h_min);
        loop {
            x -= step;
            if x <= 0.5 * step {
                points.push(0.0);
                break;
            }
            points.push(x);
            step = (step * ratio).min(h_cap);
        }
    }
    // 外側（outer へ向かう）
    let (mut x, mut step) = (centre, h_min);
    while x < outer - 1e-12 {
        x = (x + step).min(outer);
        points.push(x);
        step *= ratio;
        if x < cap_to {
            step = step.min(h_cap);
        }
    }
    points.sort_by(|a, b| a.partial_cmp(b).unwrap());
    points
}
