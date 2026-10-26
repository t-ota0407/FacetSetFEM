//! 2 次四面体（Tet10。SPEC §2.3 / §8 M12）の組み立て・求解・後処理の検証。
//!
//! 1〜4 は組み立て（`assemble`）と CG を直接つないで確かめ、5〜7 は解析（`analysis::solve`）を
//! 後処理まで通して確かめる。
//!
//! 1. **厳密解**: 多層スラブの上下面を丸ごと電極にすると、φ は層ごとに z の 1 次式になる。
//!    層の境目も節点に乗るので、2 次要素でも**全ての節点（辺の中点を含む）で厳密**に出る
//! 2. **再現性**: 逐次・並列の組み立て × CPU 逐次・CPU 並列の CG で、電位がビット単位で一致
//! 3. **AMG**: 2 次要素の行列でも収束し、反復回数が 1 次要素と同じ桁に収まる
//! 4. **GPU のメモリの見積り**: 組み立てる前の非ゼロ数の見積りが、実際の非ゼロ数を下回らない
//! 5. **解析を通す**: 直列抵抗の厳密解・保存則・VTU・領域統計
//! 6. **エネルギー収支**: 小さな電極で電界が要素内で大きく変わっても、消費電力を積分点で
//!    数えるので保存則の検査が合う（中心の 1 点で数えると合わない）
//! 7. **表示の分割**: 等値面・断面・クリップが 8 個の四面体に分けた上で作れる

mod common;

use common::{BOTTOM_PID, Layer, Patch, TOP_PID, box_mesh, linear_grid, slab};
use facetsetfem_core::analysis::{solve as analyse, write_analysis_vtu};
use facetsetfem_core::assemble::build_sparsity;
use facetsetfem_core::case::{Backend, Case, Electrode, FaceLookup, Preconditioner, SolverOptions};
use facetsetfem_core::io::vtu::VtuOptions;
use facetsetfem_core::mesh::elevate;
use facetsetfem_core::mesh::topology::extract_surface;
use facetsetfem_core::mesh::{Plane, TetView, clip, iso_surface, slice};
use facetsetfem_core::solver::{BackendSolver, CpuParallel, CpuSerial, LinearSolver};
use facetsetfem_core::{Mesh, assemble};

const CURRENT: f64 = 2.0e-3; // A
const LX: f64 = 0.02;
const LY: f64 = 0.015;

/// 3 層（皮膚・脂肪・筋）のスラブ。
const LAYERS: [Layer; 3] = [
    Layer {
        pid: 3,
        thickness: 0.010,
        divisions: 3,
    },
    Layer {
        pid: 2,
        thickness: 0.004,
        divisions: 2,
    },
    Layer {
        pid: 1,
        thickness: 0.002,
        divisions: 2,
    },
];

/// PID ごとの σ [S/m]。
fn sigma(pid: u32) -> f64 {
    match pid {
        1 => 0.10,
        2 => 0.04,
        _ => 0.35,
    }
}

fn case_for(anode_pid: u32, preconditioner: Preconditioner) -> Case {
    let mut case = Case::new();
    for pid in 1..=3 {
        case.set_conductivity(pid, sigma(pid)).unwrap();
    }
    case.add_electrode(Electrode::anode("A", CURRENT).with_shell_pids([anode_pid]));
    case.add_electrode(Electrode::cathode("C", CURRENT).with_shell_pids([BOTTOM_PID]));
    case.solver = SolverOptions {
        tolerance: 1.0e-12,
        max_iterations: 20_000,
        preconditioner,
        backend: Backend::CpuSerial,
        ..Default::default()
    };
    case
}

/// 組み立てて解く。戻り値は (電位, 反復回数)。
fn solve(
    mesh: &Mesh,
    case: &Case,
    parallel_assembly: bool,
    solver: &dyn LinearSolver,
) -> (Vec<f64>, usize) {
    let surface = extract_surface(mesh).unwrap();
    let lookup = FaceLookup::new(mesh, &surface);
    case.validate_with(&lookup).unwrap();
    let system = assemble(mesh, case, &lookup, parallel_assembly).unwrap();
    let mut phi = vec![0.0; mesh.nodes.len()];
    let report = solver
        .solve(&system.matrix, &system.load, &mut phi, &case.solver, None)
        .unwrap();
    assert!(report.converged, "{report:?}");
    (phi, report.iterations)
}

/// 上下面を丸ごと電極にした多層スラブの厳密解（下面で 0）。
///
/// 電流は −z 向きに一様に I/A 流れるので、各層で dφ/dz = I / (σ A)。
fn exact_potential(z: f64) -> f64 {
    let density = CURRENT / (LX * LY);
    let mut phi = 0.0;
    let mut bottom = 0.0;
    for layer in &LAYERS {
        let top = bottom + layer.thickness;
        let dz = (z.min(top) - bottom).max(0.0);
        phi += density / sigma(layer.pid) * dz;
        bottom = top;
    }
    phi
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// 上面の中央に小さな陽極（PID 301）、下面全体が陰極の 3 層スラブ（`nx × ny` の格子）。
fn patch_slab(nx: usize, ny: usize) -> Mesh {
    let zs = {
        let mut zs = vec![0.0];
        for layer in &LAYERS {
            let z0 = *zs.last().unwrap();
            for k in 1..=layer.divisions {
                zs.push(z0 + layer.thickness * k as f64 / layer.divisions as f64);
            }
        }
        zs
    };
    let cell_pid: Vec<u32> = LAYERS
        .iter()
        .flat_map(|l| std::iter::repeat_n(l.pid, l.divisions))
        .collect();
    let patch = Patch {
        pid: 301,
        x: [0.006, 0.014],
        y: [0.005, 0.010],
        top: true,
    };
    box_mesh(
        &linear_grid(0.0, LX, nx),
        &linear_grid(0.0, LY, ny),
        &zs,
        &cell_pid,
        &[patch, Patch::whole(BOTTOM_PID, false)],
    )
}

#[test]
fn a_raised_slab_reproduces_the_exact_potential_at_every_node() {
    let linear = slab(LX, LY, 3, 2, &LAYERS);
    let mesh = elevate(&linear).unwrap();
    let case = case_for(TOP_PID, Preconditioner::Jacobi);
    let (phi, _) = solve(&mesh, &case, false, &BackendSolver(CpuSerial));

    // 射影 CG は節点平均ゼロの代表元を返すので、厳密解も平均を揃えて比べる
    let exact: Vec<f64> = (0..mesh.nodes.len())
        .map(|i| exact_potential(mesh.nodes.coord(i as u32)[2]))
        .collect();
    let (shift_num, shift_exact) = (mean(&phi), mean(&exact));
    let range = exact.iter().cloned().fold(0.0f64, f64::max);
    let n_corners = linear.nodes.len();
    let mut worst = 0.0f64;
    for i in 0..mesh.nodes.len() {
        let error = ((phi[i] - shift_num) - (exact[i] - shift_exact)).abs() / range;
        worst = worst.max(error);
        assert!(
            error < 1e-9,
            "節点 {i}（{}）: φ = {}, 厳密 = {}",
            if i < n_corners {
                "頂点"
            } else {
                "辺の中点"
            },
            phi[i] - shift_num,
            exact[i] - shift_exact
        );
    }
    eprintln!("2 次要素の多層スラブ: 最大相対誤差 {worst:.2e}");
}

#[test]
fn second_order_results_are_bitwise_reproducible() {
    // 逐次・並列の組み立て × CPU 逐次・並列の CG。前処理は Jacobi と AMG
    let mesh = elevate(&slab(LX, LY, 4, 3, &LAYERS)).unwrap();
    for preconditioner in [Preconditioner::Jacobi, Preconditioner::Amg] {
        let case = case_for(TOP_PID, preconditioner);
        let (reference, iterations) = solve(&mesh, &case, false, &BackendSolver(CpuSerial));
        for (parallel, solver) in [
            (true, &BackendSolver(CpuSerial) as &dyn LinearSolver),
            (false, &BackendSolver(CpuParallel)),
            (true, &BackendSolver(CpuParallel)),
        ] {
            let (phi, its) = solve(&mesh, &case, parallel, solver);
            assert_eq!(its, iterations, "{preconditioner:?}");
            assert!(
                phi.iter()
                    .zip(&reference)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{preconditioner:?} 並列の組み立て {parallel}, {}: 電位がビット単位で一致しない",
                solver.name()
            );
        }
    }
}

#[test]
fn amg_converges_on_second_order_matrices() {
    // 上面の中央に小さな陽極、下面全体が陰極の 3 層スラブ。1 次のまま解いたものと、
    // 同じメッシュを 2 次に上げたものとで、Jacobi と AMG の反復回数を比べる
    let linear = patch_slab(10, 6);
    let quadratic = elevate(&linear).unwrap();

    let mut counts = Vec::new();
    for preconditioner in [Preconditioner::Jacobi, Preconditioner::Amg] {
        let case = case_for(301, preconditioner);
        let (_, p1) = solve(&linear, &case, true, &BackendSolver(CpuParallel));
        let (_, p2) = solve(&quadratic, &case, true, &BackendSolver(CpuParallel));
        counts.push((preconditioner, p1, p2));
    }
    eprintln!(
        "反復回数（1 次 {} 節点 / 2 次 {} 節点）: {counts:?}",
        linear.nodes.len(),
        quadratic.nodes.len()
    );
    let (_, amg_p1, amg_p2) = counts[1];
    // AMG は 2 次でも Jacobi よりずっと少なく、1 次と同じ桁で収まる
    assert!(amg_p2 < counts[0].2 / 3, "{counts:?}");
    assert!(amg_p2 <= 4 * amg_p1, "{counts:?}");
}

#[test]
fn the_nnz_estimate_covers_the_assembled_matrix() {
    // GPU に載るかは組み立てる前に見積もる（SPEC §4.4）。見積りが実際を下回ると、
    // VRAM が足りないのに GPU を選んで解いている途中で落ちうる
    let linear = slab(LX, LY, 6, 5, &LAYERS);
    let quadratic = elevate(&linear).unwrap();
    for mesh in [&linear, &quadratic] {
        let nnz = build_sparsity(mesh).unwrap().nnz();
        let estimate = mesh.estimated_nnz();
        assert!(nnz <= estimate, "非ゼロ {nnz} > 見積り {estimate}");
        assert!(
            2 * nnz > estimate,
            "見積りが大きすぎる: 非ゼロ {nnz}, 見積り {estimate}"
        );
    }
}

#[test]
fn the_analysis_runs_end_to_end_on_second_order_elements() {
    // 上下面を丸ごと電極にした 3 層スラブ。電極間抵抗は層の直列抵抗、|J| はどこでも I/A
    let mesh = elevate(&slab(LX, LY, 3, 2, &LAYERS)).unwrap();
    let case = case_for(TOP_PID, Preconditioner::Jacobi);
    let a = analyse(&mesh, &case, None).unwrap();
    assert!(a.conservation.is_clean(), "{:?}", a.conservation.warnings);
    let area = LX * LY;
    let expected: f64 = LAYERS
        .iter()
        .map(|l| l.thickness / (sigma(l.pid) * area))
        .sum();
    let error = (a.impedance() - expected).abs() / expected;
    assert!(error < 1e-9, "R = {}（期待 {expected}）", a.impedance());
    let j = CURRENT / area;
    for e in 0..a.fields.n_elements {
        assert!((a.fields.j_magnitude[e] - j).abs() < 1e-8 * j, "要素 {e}");
        assert!((a.fields.j_peak[e] - j).abs() < 1e-8 * j, "要素 {e} の最大");
    }
    // 領域統計と VTU も通る
    let stats = a.region_stats(&mesh, &case).unwrap();
    assert_eq!(stats.len(), 3);
    let dir = std::env::temp_dir().join(format!("facetsetfem-quadratic-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tet10.vtu");
    write_analysis_vtu(&path, &mesh, &a, &VtuOptions::default()).unwrap();
    assert!(std::fs::metadata(&path).unwrap().len() > 0);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_energy_balance_holds_where_the_field_varies_inside_elements() {
    // 小さな電極の縁では E が要素内で大きく変わる。消費電力を中心の 1 点で数えると
    // ∫σ|E|²dΩ と fᵀφ が合わず、保存則の検査（許容 1e-6）が誤った警告を出す
    let linear = patch_slab(10, 6);
    let quadratic = elevate(&linear).unwrap();
    for preconditioner in [Preconditioner::Jacobi, Preconditioner::Amg] {
        let case = case_for(301, preconditioner);
        let a = analyse(&quadratic, &case, None).unwrap();
        assert!(
            a.conservation.is_clean(),
            "{preconditioner:?}: {:?}",
            a.conservation.warnings
        );
        assert!(
            a.conservation.energy_relative_error < 1e-8,
            "{:?}",
            a.conservation
        );

        // 中心の 1 点で数えた電力は、積分した電力と 1e-6 を超えてずれる（だから積分が要る）
        let one_point: f64 = (0..a.fields.n_elements)
            .map(|e| {
                let field = &a.fields.electric_field[3 * e..3 * e + 3];
                let current = &a.fields.current_density[3 * e..3 * e + 3];
                let density: f64 = (0..3).map(|d| field[d] * current[d]).sum();
                density * a.fields.volume[e]
            })
            .sum();
        let integrated = a.conservation.power_from_field;
        assert!(
            (one_point - integrated).abs() / integrated > 1e-6,
            "1 点の電力 {one_point} と積分 {integrated} がほとんど同じ（検査の前提が崩れている）"
        );

        // 最大値とホットスポットは要素内の最大（頂点の値）で、中心の値の最大以上
        let stats = a.region_stats(&quadratic, &case).unwrap();
        for s in &stats {
            let elements: Vec<usize> = quadratic
                .iter_solids()
                .filter(|e| e.pid == s.pid)
                .map(|e| e.index)
                .collect();
            let centre_max = elements
                .iter()
                .map(|&e| a.fields.j_magnitude[e])
                .fold(0.0f64, f64::max);
            assert!(s.current_density.max >= centre_max, "PID {}", s.pid);
            assert_eq!(
                s.hotspot.j_magnitude, s.current_density.max,
                "PID {}",
                s.pid
            );
        }
    }
}

#[test]
fn isosurfaces_and_sections_are_built_on_the_eight_tetrahedra() {
    let mesh = elevate(&patch_slab(10, 6)).unwrap();
    let a = analyse(&mesh, &case_for(301, Preconditioner::Jacobi), None).unwrap();
    let tets = TetView::build(&mesh).unwrap();
    assert_eq!(tets.len(), 8 * mesh.n_solid_elements());
    let (lo, hi) = a.potential_range();
    let iso = iso_surface(&mesh, &tets, &a.potential, 0.5 * (lo + hi), &[]).unwrap();
    assert!(!iso.triangles.is_empty());
    let plane = Plane::new([0.5 * LX, 0.5 * LY, 0.008], [0.3, 1.0, 0.2]).unwrap();
    assert!(!slice(&mesh, &tets, &plane).triangles.is_empty());
    let surface = extract_surface(&mesh).unwrap();
    let clipped = clip(&mesh, &tets, &surface, &plane);
    assert!(!clipped.triangles.is_empty());
    // 外表面の残す側は 4 分割した三角形から、切り口は 8 個の四面体から作る。元の要素は
    // どちらも分割前の要素の通し番号
    let n = mesh.n_solid_elements() as u32;
    assert!(clipped.source_elements.iter().all(|&e| e < n));
}
