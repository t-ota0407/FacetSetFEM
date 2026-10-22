"""電極を含めたメッシュの書き出し（Nastran）と、節点・要素ごとの結果の CSV（SPEC §3.1 / §5.5）。

ほかのソフトウェア（COMSOL など）に GenericFEM で選んだ電極を持っていく・結果を突き合わせる経路。
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

import gfem
from gfem.gui.state import Session

SIGMA = {1: 0.10, 2: 0.04, 3: 0.35}
CURRENT = 2.0e-3


def _case(mesh: gfem.Mesh, anode: dict, cathode: dict) -> gfem.Case:
    case = gfem.Case(mesh)
    case.set_conductivity(SIGMA)
    case.add_electrode("E1", role="anode", current=CURRENT, **anode)
    case.add_electrode("E2", role="cathode", current=CURRENT, **cathode)
    case.set_solver(tolerance=1e-12)
    return case


@pytest.fixture(scope="module")
def mesh(sample_bdf) -> gfem.Mesh:
    return gfem.read_nastran(sample_bdf, length_unit="mm")


def _read_csv(path) -> tuple[list[str], np.ndarray]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[0], np.array([[float(v) for v in row] for row in rows[1:]])


# --- Nastran -----------------------------------------------------------------


def test_electrodes_are_written_as_new_shell_pids(mesh, tmp_path):
    case = _case(mesh, {"shell_pids": [101]}, {"shell_pids": [201]})
    path = tmp_path / "with_electrodes.bdf"
    pids = dict(case.write_nastran(path))
    # メッシュの PID（領域 1〜3、電極候補面 101・201）と重ならない番号を順に振る
    assert pids == {"E1": 202, "E2": 203}

    again = gfem.read_nastran(path, length_unit="mm")
    assert again.n_nodes == mesh.n_nodes
    assert again.solid_pids() == mesh.solid_pids()
    assert again.shell_pids() == [101, 201, 202, 203]
    assert again.report.unsupported == []
    np.testing.assert_allclose(again.points(), mesh.points(), rtol=1e-15, atol=0)

    # 書き足した PID だけで電極を指定して解くと、元のメッシュと同じ答えになる
    original = case.solve()
    exported = _case(again, {"shell_pids": [202]}, {"shell_pids": [203]}).solve()
    assert exported.impedance == pytest.approx(original.impedance, rel=1e-12)


def test_surface_patches_and_chosen_pids_can_be_exported(mesh, tmp_path):
    lo, hi = mesh.bounding_box()
    centre = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2]]
    patch = {"center": centre, "radius": 0.005}
    case = _case(mesh, {"patches": [patch]}, {"shell_pids": [201]})
    path = tmp_path / "patch.bdf"
    pids = dict(case.write_nastran(path, pids={"E1": 901}))
    assert pids["E1"] == 901 and pids["E2"] not in (901, 1, 2, 3, 101, 201)
    again = gfem.read_nastran(path, length_unit="mm")
    # 体表面から選んだ面が、そのまま 901 の面になる
    original = case.solve()
    exported = _case(again, {"shell_pids": [901]}, {"shell_pids": [pids["E2"]]}).solve()
    assert exported.impedance == pytest.approx(original.impedance, rel=1e-12)


def test_a_pid_already_in_the_mesh_is_rejected(mesh, tmp_path):
    case = _case(mesh, {"shell_pids": [101]}, {"shell_pids": [201]})
    with pytest.raises(gfem.GfemError, match="すでに使われています"):
        case.write_nastran(tmp_path / "dup.bdf", pids={"E1": 3})


def test_coordinates_can_be_written_in_another_unit(mesh, tmp_path):
    case = _case(mesh, {"shell_pids": [101]}, {"shell_pids": [201]})
    path = tmp_path / "metres.bdf"
    case.write_nastran(path, length_unit="m")
    assert "$ Length unit: m" in path.read_text(encoding="ascii")
    again = gfem.read_nastran(path, length_unit="m")
    np.testing.assert_allclose(again.points(), mesh.points(), rtol=1e-15, atol=0)


def test_a_second_order_mesh_is_written_with_ctria6_electrodes(mesh, tmp_path):
    quadratic = mesh.elevate()
    case = _case(quadratic, {"shell_pids": [101]}, {"shell_pids": [201]})
    path = tmp_path / "tet10.bdf"
    pids = dict(case.write_nastran(path))
    again = gfem.read_nastran(path, length_unit="mm")
    assert again.element_order == 2
    counts = dict(again.element_counts())
    assert counts["CTRIA6"] > 0
    original = case.solve()
    exported = _case(again, {"shell_pids": [pids["E1"]]}, {"shell_pids": [pids["E2"]]}).solve()
    assert exported.impedance == pytest.approx(original.impedance, rel=1e-12)


# --- 結果の CSV ----------------------------------------------------------------


def test_nodal_and_element_results_are_written_exactly(mesh, tmp_path):
    result = _case(mesh, {"shell_pids": [101]}, {"shell_pids": [201]}).solve()

    result.write_nodes_csv(tmp_path / "nodes.csv")
    header, nodes = _read_csv(tmp_path / "nodes.csv")
    assert header == ["node_id", "x_m", "y_m", "z_m", "phi_V"]
    assert len(nodes) == mesh.n_nodes
    np.testing.assert_array_equal(nodes[:, 1:4], mesh.points())
    # 読み直すと同じ値に戻る表記で書いている（丸めない）
    np.testing.assert_array_equal(nodes[:, 4], result.potential)

    result.write_elements_csv(tmp_path / "elements.csv")
    header, elements = _read_csv(tmp_path / "elements.csv")
    assert header[:5] == ["element_id", "pid", "x_m", "y_m", "z_m"]
    assert len(elements) == mesh.n_solid_elements
    column = {name: k for k, name in enumerate(header)}
    np.testing.assert_array_equal(elements[:, column["J_A_per_m2"]], result.j_magnitude)
    np.testing.assert_array_equal(elements[:, column["Ex_V_per_m"]], result.electric_field[:, 0])
    np.testing.assert_array_equal(elements[:, column["volume_m3"]], result.element_volume)
    np.testing.assert_array_equal(elements[:, 1], mesh.solid_pid_array())


def test_element_centroids_are_shared_by_first_and_second_order(mesh, tmp_path):
    # 重心は頂点の平均なので、1 次と 2 次の表を同じ点で突き合わせられる
    first = _case(mesh, {"shell_pids": [101]}, {"shell_pids": [201]}).solve()
    second = _case(mesh.elevate(), {"shell_pids": [101]}, {"shell_pids": [201]}).solve()
    first.write_elements_csv(tmp_path / "p1.csv")
    second.write_elements_csv(tmp_path / "p2.csv")
    _, a = _read_csv(tmp_path / "p1.csv")
    _, b = _read_csv(tmp_path / "p2.csv")
    np.testing.assert_array_equal(a[:, :5], b[:, :5])


# --- GUI ---------------------------------------------------------------------


def test_the_session_exports_the_mesh_with_its_electrodes(sample_bdf, tmp_path):
    s = Session()
    s.open_mesh(sample_bdf, "mm")
    with pytest.raises(gfem.GfemError, match="電極がありません"):
        s.export_nastran(tmp_path / "none.bdf")
    s.add_electrode("E1", role="anode", current=CURRENT, shell_pids=[101])
    s.add_electrode("E2", role="cathode", current=CURRENT, shell_pids=[201])
    # 導電率を決めていなくても（解いていなくても）書き出せる
    pids = s.export_nastran(tmp_path / "gui.bdf")
    assert [name for name, _ in pids] == ["E1", "E2"]
    assert (tmp_path / "gui.bdf").exists()


def test_the_file_menu_offers_the_exports(main_window):
    from PySide6.QtGui import QAction

    texts = [a.text() for a in main_window.findChildren(QAction)]
    assert "電極を含めたメッシュを Nastran に書き出す..." in texts
    assert "節点ごとの結果を CSV に書き出す..." in texts
    assert "要素ごとの結果を CSV に書き出す..." in texts
