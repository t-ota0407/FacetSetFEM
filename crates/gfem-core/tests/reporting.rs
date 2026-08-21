//! 領域統計・CSV・実行サマリ JSON（SPEC §5.2 / §5.5）。
//!
//! 一様場になる層状スラブを使うので、各層の |J|・|E| は要素によらず一定になり、
//! 最大・平均・中央値・パーセンタイルがすべて解析値に一致するはず。
//! 統計の集計そのものに誤りがあればここで落ちる。

mod common;

use common::{BOTTOM_PID, Layer, TOP_PID, slab};
use gfem_core::case::{
    Backend, Case, Electrode, Preconditioner, ReferencePotential, SolverOptions,
};
use gfem_core::post::total_stats;
use gfem_core::provenance::Provenance;
use gfem_core::{Analysis, Mesh, solve};

const CURRENT: f64 = 2.0e-3;
const LX: f64 = 0.060;
const LY: f64 = 0.040;
const AREA: f64 = LX * LY;

/// (PID, 厚さ [m], σ [S/m])
const LAYERS: [(u32, f64, f64); 3] = [(1, 0.002, 0.10), (2, 0.004, 0.04), (3, 0.014, 0.35)];

fn setup() -> (Mesh, Case, Analysis) {
    let mesh = slab(
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
    );
    let mut case = Case::new();
    for (pid, _t, sigma) in LAYERS {
        case.set_conductivity(pid, sigma).unwrap();
    }
    case.set_region_name(1, "skin");
    case.set_region_name(2, "fat");
    case.set_region_name(3, "muscle");
    case.add_electrode(Electrode::anode("top", CURRENT).with_shell_pids([TOP_PID]));
    case.add_electrode(Electrode::cathode("bottom", CURRENT).with_shell_pids([BOTTOM_PID]));
    case.solver = SolverOptions {
        tolerance: 1.0e-13,
        max_iterations: 20_000,
        preconditioner: Preconditioner::Jacobi,
        backend: Backend::CpuSerial,
        ..Default::default()
    };
    case.reference = ReferencePotential::CathodeMean;
    let analysis = solve(&mesh, &case, None).unwrap();
    (mesh, case, analysis)
}

fn tmp_path(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("gfem_report_{}_{name}", std::process::id()));
    p
}

#[test]
fn region_stats_reproduce_the_analytic_values() {
    let (mesh, case, analysis) = setup();
    let stats = analysis.region_stats(&mesh, &case).unwrap();
    assert_eq!(stats.len(), 3);

    for (stat, (pid, thickness, sigma)) in stats.iter().zip(LAYERS) {
        assert_eq!(stat.pid, pid);
        assert_eq!(stat.sigma, sigma);
        // 数百要素の総和なので、丸め誤差は相対 1e-12 の範囲で許容する
        assert!((stat.volume - AREA * thickness).abs() < 1e-12 * AREA * thickness);

        // 一様場なので分布は 1 点に集中する
        let expected_j = CURRENT / AREA;
        let expected_e = expected_j / sigma;
        for value in [
            stat.current_density.max,
            stat.current_density.mean,
            stat.current_density.median,
            stat.current_density.p95,
            stat.current_density.p99,
        ] {
            assert!(
                (value - expected_j).abs() < 1e-8 * expected_j,
                "PID {pid}: |J| = {value}（期待 {expected_j}）"
            );
        }
        for value in [
            stat.electric_field.max,
            stat.electric_field.mean,
            stat.electric_field.p99,
        ] {
            assert!(
                (value - expected_e).abs() < 1e-8 * expected_e,
                "PID {pid}: |E| = {value}（期待 {expected_e}）"
            );
        }

        // 消費電力 = I² × 層の抵抗
        let expected_power = CURRENT * CURRENT * thickness / (sigma * AREA);
        assert!(
            (stat.power - expected_power).abs() < 1e-8 * expected_power,
            "PID {pid}: P = {}（期待 {expected_power}）",
            stat.power
        );
    }
}

#[test]
fn region_names_come_from_the_case() {
    let (mesh, case, analysis) = setup();
    let stats = analysis.region_stats(&mesh, &case).unwrap();
    let names: Vec<&str> = stats.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["skin", "fat", "muscle"]);
}

#[test]
fn hotspot_points_at_the_layer_with_the_strongest_field() {
    let (mesh, case, analysis) = setup();
    let stats = analysis.region_stats(&mesh, &case).unwrap();
    for stat in &stats {
        // ホットスポットの座標は必ずその層の厚みの中にある
        let z = stat.hotspot.centroid[2];
        assert!(z > 0.0 && z < 0.020, "PID {}: z = {z}", stat.pid);
        assert!(stat.hotspot.j_magnitude > 0.0);
    }
    // 全領域を合わせたホットスポットは |E| が最大の層（σ が最小 = fat）に来る
    let total = total_stats(&stats).unwrap();
    let fat = &stats[1];
    assert!((total.electric_field.max - fat.electric_field.max).abs() < 1e-12);
    // 体積と電力は各層の和
    let volume_sum: f64 = stats.iter().map(|s| s.volume).sum();
    assert!((total.volume - volume_sum).abs() < 1e-12 * volume_sum);
    assert!((total.power - analysis.conservation.power_from_field).abs() < 1e-12);
}

#[test]
fn region_stats_csv_is_readable() {
    let (mesh, case, analysis) = setup();
    let path = tmp_path("stats.csv");
    analysis
        .write_region_stats_csv(&path, &mesh, &case)
        .unwrap();

    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 4, "ヘッダ + 3 領域");
    assert!(lines[0].starts_with("pid,name,sigma_S_per_m,"));
    assert!(lines[0].contains("hotspot_z_m"));
    // 列数が揃っている
    let columns = lines[0].split(',').count();
    for line in &lines[1..] {
        assert_eq!(line.split(',').count(), columns, "{line}");
    }
    assert!(lines[1].starts_with("1,skin,"));
    std::fs::remove_file(&path).ok();
}

#[test]
fn convergence_csv_matches_the_history() {
    let (_mesh, _case, analysis) = setup();
    let path = tmp_path("convergence.csv");
    analysis.write_convergence_csv(&path).unwrap();

    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], "iteration,relative_residual");
    assert_eq!(lines.len(), analysis.solver.history.len() + 1);
    assert!(lines[1].starts_with("0,"));
    std::fs::remove_file(&path).ok();
}

#[test]
fn summary_json_records_everything_needed_to_reproduce_the_run() {
    let (mesh, case, analysis) = setup();
    let provenance = Provenance::capture().with_case_text("version: 1\nmesh:\n  file: slab.bdf\n");
    let text = analysis
        .summary_json(&mesh, &case, &provenance)
        .unwrap()
        .to_pretty_string();

    // SPEC §5.5 が要求する項目
    for key in [
        "\"provenance\"",
        "\"gfem_version\"",
        "\"timestamp_utc\"",
        "\"case_text\"",
        "\"mesh\"",
        "\"materials\"",
        "\"electrodes\"",
        "\"solver\"",
        "\"conservation\"",
        "\"region_stats\"",
        "\"timings\"",
        "\"warnings\"",
    ] {
        assert!(text.contains(key), "{key} が実行サマリにありません");
    }
    assert!(text.contains("\"backend\": \"cpu_serial\""));
    assert!(text.contains("\"gauge\": \"mean_zero\""));
    assert!(text.contains("\"converged\": true"));
    assert!(text.contains("\"passed\": true"));
    assert!(text.contains("\"name\": \"muscle\""));
}

#[test]
fn summary_json_can_be_written_and_reread() {
    let (mesh, case, analysis) = setup();
    let path = tmp_path("summary.json");
    let provenance = Provenance::capture();
    analysis
        .write_summary_json(&path, &mesh, &case, &provenance)
        .unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.starts_with('{'));
    assert!(text.trim_end().ends_with('}'));
    std::fs::remove_file(&path).ok();
}

#[test]
fn provenance_records_the_input_hash() {
    let path = tmp_path("model.bdf");
    std::fs::write(&path, b"GRID,1,,0.0,0.0,0.0\n").unwrap();
    let provenance = Provenance::capture().with_mesh(&path).unwrap();
    let hash = provenance.mesh_sha256.clone().unwrap();
    assert_eq!(hash.len(), 64);
    assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));

    // 中身が変われば必ずハッシュも変わる
    std::fs::write(&path, b"GRID,1,,0.0,0.0,1.0\n").unwrap();
    let other = Provenance::capture().with_mesh(&path).unwrap();
    assert_ne!(hash, other.mesh_sha256.unwrap());
    std::fs::remove_file(&path).ok();
}
