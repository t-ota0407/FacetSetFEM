"""M0（骨組み）の受け入れテスト。

SPEC §8 の M0 が満たすべきこと:

1. Nastran バルクデータ（GRID / CTETRA / PSOLID ほか）を読める
2. 小フィールド形式・自由形式・大フィールド形式で同じ結果になる
3. 単位が SI（m）に正規化される（SPEC §3.2）
4. VTU として書き出せて、VTK / ParaView が読める（SPEC §5.5）
5. Python から呼べる（SPEC §6.5）
"""

from __future__ import annotations

import numpy as np
import pytest

import gfem


# --- 読み込み ---------------------------------------------------------------


def test_reads_the_sample_model(sample_bdf):
    mesh = gfem.read_nastran(sample_bdf, length_unit="mm")
    assert mesh.n_nodes == 936
    assert mesh.n_solid_elements == 4032
    assert mesh.n_shell_elements == 48
    assert mesh.n_elements == 4080
    assert mesh.element_counts() == [("CTETRA", 4032), ("CTRIA3", 48)]


def test_regions_and_electrode_faces_are_listed(sample_bdf):
    mesh = gfem.read_nastran(sample_bdf, length_unit="mm")
    # PSOLID の PID = 領域（σ を割り当てる単位）
    assert mesh.solid_pids() == [1, 2, 3]
    # PSHELL の PID = 電極候補面
    assert mesh.shell_pids() == [101, 201]


def test_report_has_no_warnings_for_a_clean_model(sample_bdf):
    report = gfem.read_nastran(sample_bdf, length_unit="mm").report
    assert report.warnings == []
    assert report.unsupported == []
    counts = dict(report.card_counts)
    assert counts["GRID"] == 936
    assert counts["CTETRA"] == 4032
    # MAT1 は認識するが導電率としては使わない（SPEC §3.1）
    assert counts["MAT1"] == 1


# --- 単位 -------------------------------------------------------------------


def test_coordinates_are_normalised_to_si(sample_bdf):
    mesh = gfem.read_nastran(sample_bdf, length_unit="mm")
    (lo, hi) = mesh.bounding_box()
    assert lo == (0.0, 0.0, 0.0)
    assert hi == pytest.approx((0.060, 0.040, 0.020), abs=1e-15)


def test_length_unit_scales_the_model(sample_bdf):
    """同じファイルでも単位を変えれば寸法が変わる（＝単位は必ず効いている）。"""
    in_mm = gfem.read_nastran(sample_bdf, length_unit="mm").bounding_box()[1]
    in_m = gfem.read_nastran(sample_bdf, length_unit="m").bounding_box()[1]
    assert in_m == pytest.approx(tuple(v * 1000.0 for v in in_mm), rel=1e-15)


def test_bounding_box_is_shown_in_both_units(sample_bdf):
    """単位の取り違えに気づかせる表示（SPEC §3.2 の必須要件）。"""
    mesh = gfem.read_nastran(sample_bdf, length_unit="mm")
    text = gfem.describe_bounding_box(mesh)
    assert "60 mm x 40 mm x 20 mm" in text
    assert "0.06 m x 0.04 m x 0.02 m" in text


def test_unknown_unit_is_rejected(sample_bdf):
    with pytest.raises(gfem.GfemError) as excinfo:
        gfem.read_nastran(sample_bdf, length_unit="furlong")
    assert "mm / cm / m / inch" in str(excinfo.value)


# --- 書式 -------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["small", "free", "large"])
def test_all_bulk_data_formats_give_the_same_mesh(make_sample_mesh, tmp_path, fmt):
    out = tmp_path / f"slab_{fmt}.bdf"
    make_sample_mesh.main(["-o", str(out), "--format", fmt])
    mesh = gfem.read_nastran(out, length_unit="mm")
    assert mesh.n_nodes == 936
    assert mesh.n_solid_elements == 4032
    assert mesh.n_shell_elements == 48
    assert mesh.bounding_box()[1] == pytest.approx((0.060, 0.040, 0.020), abs=1e-15)
    assert mesh.report.warnings == []


# --- エラーメッセージ -------------------------------------------------------


def test_error_says_what_where_and_how_to_fix(tmp_path):
    """CLAUDE.md 設計原則 6: 何が / どこで / どうすれば直るか。"""
    bad = tmp_path / "broken.bdf"
    bad.write_text(
        "GRID,1,,0.0,0.0,0.0\nCTETRA,10,1,1,2,3,4\n", encoding="ascii", newline="\n"
    )
    with pytest.raises(gfem.GfemError) as excinfo:
        gfem.read_nastran(bad, length_unit="mm")
    message = str(excinfo.value)
    assert "節点 ID 2" in message  # 何が
    assert "broken.bdf:2" in message  # どこで
    assert "EID=10" in message  # どの要素で
    assert "対処:" in message  # どうすれば直るか


def test_missing_file_is_reported_clearly(tmp_path):
    with pytest.raises(gfem.GfemError) as excinfo:
        gfem.read_nastran(tmp_path / "does_not_exist.bdf", length_unit="mm")
    assert "開けません" in str(excinfo.value)


# --- VTU 出力 ---------------------------------------------------------------


@pytest.mark.parametrize("compress", [True, False])
def test_vtu_is_readable_by_vtk(sample_bdf, tmp_path, compress):
    pv = pytest.importorskip("pyvista")
    mesh = gfem.read_nastran(sample_bdf, length_unit="mm")
    out = tmp_path / "out.vtu"
    mesh.write_vtu(out, compress=compress)

    grid = pv.read(out)
    assert grid.n_points == 936
    assert grid.n_cells == 4080

    # VTK のセルタイプ: 四面体 = 10, 三角形 = 5
    kinds, counts = np.unique(grid.celltypes, return_counts=True)
    assert dict(zip(kinds.tolist(), counts.tolist())) == {10: 4032, 5: 48}

    # 座標は m（倍精度で書いている）
    assert grid.points.dtype == np.float64
    assert grid.bounds.x_max == pytest.approx(0.060, abs=1e-15)
    assert grid.bounds.z_max == pytest.approx(0.020, abs=1e-15)

    pids, pid_counts = np.unique(grid.cell_data["PID"], return_counts=True)
    assert pids.tolist() == [1, 2, 3, 101, 201]
    assert pid_counts.tolist() == [576, 1152, 2304, 24, 24]
    assert grid.point_data["NodeID"].min() == 1
    assert grid.point_data["NodeID"].max() == 936


def test_compression_actually_shrinks_the_file(sample_bdf, tmp_path):
    mesh = gfem.read_nastran(sample_bdf, length_unit="mm")
    raw, zipped = tmp_path / "raw.vtu", tmp_path / "zip.vtu"
    mesh.write_vtu(raw, compress=False)
    mesh.write_vtu(zipped, compress=True)
    assert zipped.stat().st_size < raw.stat().st_size


# --- Python API -------------------------------------------------------------


def test_module_surface(sample_bdf):
    assert gfem.__version__ == "0.1.0"
    assert gfem.length_units() == ["mm", "cm", "m", "inch"]
    mesh = gfem.read_nastran(sample_bdf, length_unit="mm")
    assert mesh.length_unit == "mm"
    assert mesh.source.endswith("slab_3layer.bdf")
    assert "nodes=936" in repr(mesh)
