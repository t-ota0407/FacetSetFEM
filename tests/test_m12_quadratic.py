"""2 次四面体（SPEC §2.3 / §8 M12）の Python API・ケースファイル・GUI。

物理の検証（厳密解・保存則・再現性・AMG・GPU）は Rust の `crates/gfem-core/tests/quadratic.rs`
にある。ここは Python から使う経路を見る。
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

import gfem
from gfem.gui.state import Session

SIGMA = {1: 0.10, 2: 0.04, 3: 0.35}
CURRENT = 2.0e-3


@pytest.fixture(scope="module")
def linear(sample_bdf) -> gfem.Mesh:
    return gfem.read_nastran(sample_bdf, length_unit="mm")


@pytest.fixture(scope="module")
def quadratic(linear) -> gfem.Mesh:
    return linear.elevate()


@pytest.fixture
def slab_study(tmp_path, sample_bdf, repo_root) -> gfem.Study:
    shutil.copy(sample_bdf, tmp_path / "slab_3layer.bdf")
    shutil.copy(repo_root / "tests" / "models" / "slab_3layer.yaml", tmp_path / "case.yaml")
    return gfem.Study.load(tmp_path / "case.yaml")


# --- メッシュ ----------------------------------------------------------------


def test_a_mesh_can_be_raised_to_second_order(linear, quadratic):
    assert linear.element_order == 1
    assert quadratic.element_order == 2
    assert quadratic.n_elements == linear.n_elements
    assert quadratic.n_nodes > 5 * linear.n_nodes
    counts = dict(quadratic.element_counts())
    assert counts["CTETRA(10)"] == dict(linear.element_counts())["CTETRA"]
    # 領域と電極候補面はそのまま
    assert quadratic.solid_pids() == linear.solid_pids()
    assert quadratic.shell_pids() == linear.shell_pids()
    # 頂点は元の節点、足した節点は辺の中点
    mask = quadratic.corner_node_mask()
    assert mask.dtype == bool and mask.shape == (quadratic.n_nodes,)
    assert mask[: linear.n_nodes].all() and not mask[linear.n_nodes :].any()
    assert linear.corner_node_mask().all()


def test_a_second_order_mesh_cannot_be_raised_again(quadratic):
    with pytest.raises(gfem.GfemError, match="すでに"):
        quadratic.elevate()


def test_the_body_surface_is_split_into_four_per_face(linear, quadratic):
    # 2 次要素の面は辺の中点で 4 枚に割って表示する（SPEC §5.1）
    assert len(quadratic.surface_triangles()) == 4 * len(linear.surface_triangles())
    owners = quadratic.surface_triangle_elements()
    assert len(owners) == len(quadratic.surface_triangles())


# --- ケースファイル ----------------------------------------------------------


def test_the_element_order_round_trips_through_the_case_file(slab_study, tmp_path):
    assert slab_study.element_order == 1
    # 既定（1 次）のときは書かない。以前からのケースファイルと同じ形に保つ
    assert "element_order" not in slab_study.to_dict()["mesh"]

    slab_study.element_order = 2
    assert slab_study.to_dict()["mesh"]["element_order"] == 2
    slab_study.save(tmp_path / "quadratic.yaml")
    again = gfem.Study.load(tmp_path / "quadratic.yaml")
    assert again.element_order == 2
    assert again.read_mesh().element_order == 2


@pytest.mark.parametrize("value", [3, 0, "2", True])
def test_an_unknown_element_order_is_a_helpful_error(slab_study, value):
    data = slab_study.to_dict()
    data["mesh"]["element_order"] = value
    with pytest.raises(gfem.GfemError) as info:
        gfem.Study.from_dict(data)
    text = str(info.value)
    assert "要素の次数" in text and "element_order" in text


def test_a_study_solves_with_second_order_elements(slab_study):
    first = slab_study.solve()
    slab_study.element_order = 2
    second = slab_study.solve()
    assert second.residual < 1e-9
    assert second.conservation["warnings"] == []
    assert len(second.potential) > 5 * len(first.potential)
    # 同じメッシュの 2 次要素の解の空間は 1 次要素の空間を含む。電流を与える問題では
    # 空間が広いほど電極間抵抗は大きく出て、下から真の値に近づく（ポテンシャルエネルギー
    # −I²R/2 の最小化）。だから 2 次 ≥ 1 次が必ず成り立つ。差はこの粗いサンプルで 1 割弱
    assert second.impedance > first.impedance
    assert second.impedance == pytest.approx(first.impedance, rel=0.15)
    # 要素内の最大は中心の値以上。1 次要素では要素値と同じ
    assert np.all(second.j_peak >= second.j_magnitude * (1 - 1e-12))
    assert np.array_equal(first.j_peak, first.j_magnitude)


# --- GUI のセッション --------------------------------------------------------


def _setup(session: Session) -> None:
    for pid, sigma in SIGMA.items():
        session.set_conductivity(pid, sigma)
    session.add_electrode("E1", role="anode", current=CURRENT, shell_pids=[101])
    session.add_electrode("E2", role="cathode", current=CURRENT, shell_pids=[201])


def test_the_session_swaps_the_mesh_with_the_element_order(sample_bdf):
    s = Session()
    s.open_mesh(sample_bdf, "mm")
    _setup(s)
    n_linear = s.mesh.n_nodes
    serial = s.mesh_serial
    s.solve()
    assert s.solution is not None

    s.study.element_order = 2
    assert s.element_order_out_of_sync()
    assert s.sync_element_order()
    assert s.mesh.element_order == 2 and s.mesh.n_nodes > n_linear
    # 節点の数が変わるので結果は捨てる。形は同じなので視点は保つ（mesh_serial は変えない）
    assert s.solution is None
    assert s.mesh_serial == serial
    assert not s.sync_element_order()  # もう揃っている
    assert "2（ファイルの 1 次のメッシュを 2 次に上げたもの）" in s.mesh_summary()

    solution = s.solve()
    assert solution.residual < 1e-9
    assert len(solution.potential) == s.mesh.n_nodes

    # 1 次に戻すと、読み込んだままのメッシュに戻る
    s.study.element_order = 1
    assert s.sync_element_order()
    assert s.mesh.element_order == 1 and s.mesh.n_nodes == n_linear


def test_undo_brings_the_mesh_back_with_the_element_order(sample_bdf):
    s = Session()
    s.open_mesh(sample_bdf, "mm")
    s.checkpoint("要素の次数を変更")
    s.study.element_order = 2
    s.sync_element_order()
    assert s.mesh.element_order == 2
    s.undo()
    assert s.element_order_out_of_sync()
    s.sync_element_order()
    assert s.mesh.element_order == 1


def test_a_second_order_case_opens_with_the_raised_mesh(slab_study, tmp_path):
    slab_study.element_order = 2
    slab_study.save(tmp_path / "quadratic.yaml")
    s = Session()
    s.open_case(tmp_path / "quadratic.yaml")
    assert s.mesh.element_order == 2
    assert s.file_element_order() == 1
    assert not s.element_order_out_of_sync()


def test_display_helpers_work_on_second_order_meshes(sample_bdf):
    s = Session()
    s.open_mesh(sample_bdf, "mm")
    _setup(s)
    s.study.element_order = 2
    s.sync_element_order()
    s.solve()

    surface = s.surface_polydata()
    assert surface.n_cells == len(s.mesh.surface_triangles())
    # 稜線は元の要素の辺（頂点と中点を結ぶ線）だけ。分割の線（中点どうし）は描かない
    lines = s.element_edges(surface)
    assert lines is not None and lines.n_cells > 0
    mask = s.mesh.corner_node_mask()
    segments = lines.lines.reshape(-1, 3)[:, 1:]
    assert np.all(mask[segments[:, 0]] != mask[segments[:, 1]])
    # 外表面の三角形 1 枚あたり、元の要素の辺は 3 本の半分ずつ（中点で 2 本に割れる）
    assert lines.n_cells == 3 * surface.n_cells // 4 * 2 // 2

    # 表示中の領域に色の範囲を合わせる（2 次要素の VTK の型番号 24 を読む）
    full = s.field_range("phi [V]")
    some = s.field_range("phi [V]", [1])
    assert some is not None and full is not None
    assert full[0] <= some[0] <= some[1] <= full[1]

    # 1 次要素のメッシュでは稜線を足さない（VTK の show_edges に任せる）
    s.study.element_order = 1
    s.sync_element_order()
    assert s.element_edges(s.surface_polydata()) is None


def test_picked_electrode_faces_are_split_like_the_surface(sample_bdf):
    s = Session()
    s.open_mesh(sample_bdf, "mm")
    lo, hi = s.model_box()
    centre = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2]]
    linear = s.pick_patch(centre, radius=0.004)
    s.study.element_order = 2
    s.sync_element_order()
    quadratic = s.pick_patch(centre, radius=0.004)
    # 選ばれる面は同じ（面番号・面積）。強調表示の三角形は 4 倍
    assert quadratic["n_faces"] == linear["n_faces"]
    assert quadratic["area"] == pytest.approx(linear["area"], rel=1e-12)
    assert len(quadratic["triangles"]) == 4 * len(linear["triangles"])
    assert np.array_equal(quadratic["face_refs"], linear["face_refs"])


# --- GUI ---------------------------------------------------------------------


def test_the_solver_page_switches_the_element_order(main_window, sample_bdf, qt_app):
    w = main_window
    w.open_mesh(sample_bdf, "mm")
    page = w.page_solver
    page.reload()
    assert page.element_order.isEnabled()
    assert page.element_order.currentIndex() == 0
    n_linear = w.session.mesh.n_nodes

    page.element_order.setCurrentIndex(1)
    qt_app.processEvents()
    assert w.session.study.element_order == 2
    assert w.session.mesh.element_order == 2
    assert w.session.mesh.n_nodes > n_linear
    assert "要素の次数" in w.action_undo.text()

    # 元に戻すとメッシュも 1 次に戻り、ページの表示も戻る
    w.undo()
    qt_app.processEvents()
    assert w.session.mesh.element_order == 1
    assert w.session.mesh.n_nodes == n_linear
    page.reload()
    assert page.element_order.currentIndex() == 0
