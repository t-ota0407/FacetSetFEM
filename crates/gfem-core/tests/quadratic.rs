//! 2 次四面体（Tet10。SPEC §2.3 / §8 M12）の組み立てと求解の検証。
//!
//! 求解の入口（`analysis::solve`）は後処理が 2 次要素に対応するまで止めてあるので、
//! ここでは組み立て（`assemble`）と CG を直接つないで確かめる。
//!
//! 1. **厳密解**: 多層スラブの上下面を丸ごと電極にすると、φ は層ごとに z の 1 次式になる。
//!    層の境目も節点に乗るので、2 次要素でも**全ての節点（辺の中点を含む）で厳密**に出る
//! 2. **再現性**: 逐次・並列の組み立て × CPU 逐次・CPU 並列の CG で、電位がビット単位で一致
//! 3. **AMG**: 2 次要素の行列でも収束し、反復回数が 1 次要素と同じ桁に収まる。
//!    2 次要素では θ を 0.15 にする（`SolverOptions::for_mesh`。SPEC §4.5）
//! 4. **GPU のメモリの見積り**: 組み立てる前の非ゼロ数の見積りが、実際の非ゼロ数を下回らない

mod common;

use common::{BOTTOM_PID, Layer, Patch, TOP_PID, box_mesh, linear_grid, slab};
use gfem_core::assemble::build_sparsity;
use gfem_core::case::{Backend, Case, Electrode, FaceLookup, Preconditioner, SolverOptions};
use gfem_core::mesh::elevate;
use gfem_core::mesh::topology::extract_surface;
use gfem_core::solver::{
    BackendSolver, CpuParallel, CpuSerial, DEFAULT_STRENGTH_THRESHOLD, LinearSolver,
    QUADRATIC_STRENGTH_THRESHOLD,
};
use gfem_core::{Mesh, assemble};

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

/// 組み立てて解く。戻り値は (電位, 反復回数)。設定は解析と同じく `for_mesh` を通す。
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
    let options = case.solver.for_mesh(mesh);
    let report = solver
        .solve(&system.matrix, &system.load, &mut phi, &options, None)
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
    let linear = box_mesh(
        &linear_grid(0.0, LX, 10),
        &linear_grid(0.0, LY, 6),
        &zs,
        &cell_pid,
        &[patch, Patch::whole(BOTTOM_PID, false)],
    );
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
fn the_amg_threshold_follows_the_element_order() {
    let linear = slab(LX, LY, 2, 2, &LAYERS);
    let quadratic = elevate(&linear).unwrap();
    let options = case_for(TOP_PID, Preconditioner::Amg).solver;
    assert_eq!(options.amg.strength_threshold, DEFAULT_STRENGTH_THRESHOLD);
    assert_eq!(
        options.for_mesh(&linear).amg.strength_threshold,
        DEFAULT_STRENGTH_THRESHOLD
    );
    assert_eq!(
        options.for_mesh(&quadratic).amg.strength_threshold,
        QUADRATIC_STRENGTH_THRESHOLD
    );
    // 既定値でない θ を選んでいれば、2 次要素でもそれを使う
    let mut chosen = options;
    chosen.amg.strength_threshold = 0.3;
    assert_eq!(chosen.for_mesh(&quadratic).amg.strength_threshold, 0.3);
    // 1 次要素では θ 以外も含めて何も変えない
    assert_eq!(options.for_mesh(&linear), options);
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
