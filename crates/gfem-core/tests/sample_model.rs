//! `tests/models/slab_3layer.bdf`（scripts/make_sample_mesh.py が生成）を
//! 実際に読み込む結合テスト。M0 の到達点そのものを固定する。

use gfem_core::{ElementKind, LengthUnit, VtuOptions, read_nastran, write_mesh_vtu};
use std::path::PathBuf;

fn sample_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/models/slab_3layer.bdf")
        .canonicalize()
        .expect("サンプルモデルが見つかりません。scripts/make_sample_mesh.py を実行してください")
}

#[test]
fn reads_the_three_layer_slab() {
    let (mesh, report) = read_nastran(sample_path(), LengthUnit::Millimeter).unwrap();

    assert_eq!(mesh.nodes.len(), 936);
    assert_eq!(mesh.n_solid_elements(), 4032);
    assert_eq!(mesh.n_shell_elements(), 48);
    assert_eq!(mesh.solid_pids(), vec![1, 2, 3]);
    assert_eq!(mesh.shell_pids(), vec![101, 201]);
    assert!(report.warnings.is_empty(), "{:?}", report.warnings);
    assert!(report.unsupported.is_empty(), "{:?}", report.unsupported);

    // MAT1 は「認識するが導電率としては使わない」カード（SPEC §3.1）
    assert!(
        report
            .card_counts
            .iter()
            .any(|(name, n)| name == "MAT1" && *n == 1)
    );
}

#[test]
fn coordinates_are_converted_to_metres() {
    let (mesh, _) = read_nastran(sample_path(), LengthUnit::Millimeter).unwrap();
    let (lo, hi) = mesh.bounding_box().unwrap();
    assert_eq!(lo, [0.0, 0.0, 0.0]);
    for (got, want) in hi.iter().zip([0.060, 0.040, 0.020]) {
        assert!((got - want).abs() < 1e-15, "{got} != {want}");
    }
}

#[test]
fn tetrahedra_fill_the_box_exactly() {
    let (mesh, _) = read_nastran(sample_path(), LengthUnit::Millimeter).unwrap();
    let volumes = mesh.tet4_signed_volumes();
    assert_eq!(volumes.len(), 4032);
    // 節点順序は生成時に正となるよう揃えてある
    assert!(volumes.iter().all(|&v| v > 0.0));
    // 60 x 40 x 20 mm = 4.8e-5 m³ を四面体が過不足なく埋めている
    // 4032 個の総和なので、丸め誤差は相対 1e-12 の範囲で許容する
    let total: f64 = volumes.iter().sum();
    assert!(
        (total - 4.8e-5).abs() < 4.8e-5 * 1e-12,
        "総体積 {total} が直方体の体積と一致しません"
    );
}

#[test]
fn layer_volumes_match_the_layer_thicknesses() {
    let (mesh, _) = read_nastran(sample_path(), LengthUnit::Millimeter).unwrap();
    let block = mesh.block(ElementKind::Tet4).unwrap();
    let volumes = mesh.tet4_signed_volumes();
    let area = 0.060 * 0.040;
    // (PID, 厚さ [m])
    for (pid, thickness) in [(1u32, 0.002), (2, 0.004), (3, 0.014)] {
        let v: f64 = volumes
            .iter()
            .zip(block.pids())
            .filter(|&(_, &p)| p == pid)
            .map(|(v, _)| v)
            .sum();
        assert!(
            (v - area * thickness).abs() < area * thickness * 1e-12,
            "PID {pid} の体積 {v} が想定 {} と一致しません",
            area * thickness
        );
    }
}

#[test]
fn writes_a_vtu_that_starts_with_a_valid_header() {
    let (mesh, _) = read_nastran(sample_path(), LengthUnit::Millimeter).unwrap();
    let mut out = std::env::temp_dir();
    out.push(format!("gfem_sample_{}.vtu", std::process::id()));
    write_mesh_vtu(&out, &mesh, &VtuOptions::default()).unwrap();

    let bytes = std::fs::read(&out).unwrap();
    let head = String::from_utf8_lossy(&bytes[..800]).to_string();
    assert!(head.contains("type=\"UnstructuredGrid\""), "{head}");
    assert!(head.contains("NumberOfPoints=\"936\""), "{head}");
    assert!(head.contains("NumberOfCells=\"4080\""), "{head}");
    assert!(head.contains("vtkZLibDataCompressor"), "{head}");
    std::fs::remove_file(&out).ok();
}
