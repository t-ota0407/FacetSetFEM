"""領域・電極ごとの見せ方（表示 / 半透明 / 非表示。SPEC §6.3「パーツごとの表示」）。

ツリーの目のアイコンで切り替える。見方の好みなので解析条件には入れず、Undo の対象にも、
古い結果の知らせの対象にもならない。
"""

from __future__ import annotations

import numpy as np
import pytest

from facetsetfem.gui.state import HIDDEN, TRANSPARENT, VISIBLE, Session

FIELD = "|J| [A/m2]"


@pytest.fixture(scope="module")
def solved(repo_root) -> Session:
    s = Session()
    s.open_case(repo_root / "tests" / "models" / "slab_3layer.yaml")
    s.solve()
    return s


@pytest.fixture
def session(solved) -> Session:
    solved.set_region_view(solved.mesh.solid_pids(), VISIBLE)
    solved.set_electrode_view([e.name for e in solved.study.electrodes], VISIBLE)
    return solved


def test_region_boundaries_cover_every_region_once_per_side(session):
    """全領域の境界面は、領域ごとの境界面を全部合わせたものと同じ面の数。"""
    poly = session.region_boundaries_polydata()
    per_region = sum(session.region_boundary(p).n_cells for p in session.mesh.solid_pids())
    assert poly.n_cells == per_region
    assert poly.n_cells > session.surface_polydata().n_cells  # 領域の境目も入る


def test_part_surfaces_follow_the_views(session):
    session.set_region_view([1], HIDDEN)
    session.set_region_view([2], TRANSPARENT)
    parts = session.part_surfaces(FIELD)
    assert set(np.unique(parts[VISIBLE].cell_data["PID"])) == {3}
    assert set(np.unique(parts[TRANSPARENT].cell_data["PID"])) == {2}
    # 面はそれぞれの領域の要素の値で塗る
    owners = parts[VISIBLE].cell_data["__owner"]
    assert np.allclose(parts[VISIBLE].cell_data[FIELD], session.result_array(FIELD)[owners])
    assert session.regions_customised()
    assert session.shown_region_pids() == [2, 3]


def test_the_views_are_not_part_of_the_analysis(session):
    """見せ方を変えても、解析条件（保存・Undo・古い結果の知らせ）は変わらない。"""
    before = session.study.to_yaml()
    session.set_region_view([1, 2], HIDDEN)
    session.set_electrode_view(["E1"], TRANSPARENT)
    assert session.study.to_yaml() == before
    assert session.changes_since_solve() == []
    assert not session.can_undo()


def test_the_colour_range_can_follow_the_shown_regions(session):
    whole = session.field_range(FIELD)
    muscle = session.field_range(FIELD, [3])
    j = session.result_array(FIELD)[session.mesh.solid_pid_array() == 3]
    assert muscle == pytest.approx((j.min(), j.max()))
    assert muscle[1] < whole[1], "皮膚の最大値が外れる"
    # 節点値は、その領域の要素が使う節点の値
    lo, hi = session.field_range("phi [V]", [3])
    assert session.field_range("phi [V]")[0] <= lo <= hi


def test_renaming_an_electrode_keeps_its_view(session):
    session.set_electrode_view(["E2"], HIDDEN)
    session.rename_electrode("E2", "GND")
    assert session.electrode_view("GND") == HIDDEN
    session.rename_electrode("GND", "E2")


# --- GUI ---------------------------------------------------------------------


@pytest.fixture
def window(main_window, repo_root, qt_app):
    w = main_window
    w.open_case(repo_root / "tests" / "models" / "slab_3layer.yaml")
    w.start_solve()
    assert w.solver.wait(60_000)
    qt_app.processEvents()
    yield w
    w.page_result.fit_range.setChecked(False)


def _item(window, kind, key):
    return next(i for i in window._tree_items() if window._item_key(i) == (kind, key))


def _click_eye(window, item):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    header = window.tree.header()
    x = header.sectionPosition(1) + header.sectionSize(1) // 2
    y = window.tree.visualItemRect(item).center().y()
    QTest.mouseClick(window.tree.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(x, y))


def test_the_eye_cycles_the_view_without_changing_the_selection(window):
    window.select_item("result", None)
    skin = _item(window, "region", 1)
    _click_eye(window, skin)
    assert window.session.region_view(1) == TRANSPARENT
    assert [window._item_key(i) for i in window.tree.selectedItems()] == [("result", None)]
    _click_eye(window, _item(window, "region", 1))  # ツリーを作り直していてもよいよう引き直す
    assert window.session.region_view(1) == HIDDEN
    _click_eye(window, _item(window, "region", 1))
    assert window.session.region_view(1) == VISIBLE


def test_hidden_regions_are_not_drawn(window):
    window.set_part_view([_item(window, "region", 1), _item(window, "region", 2)], HIDDEN)
    drawn = set()
    for actor in window.plotter.renderer.actors.values():
        data = actor.GetMapper().GetInput() if hasattr(actor, "GetMapper") else None
        if data is not None and data.GetCellData().GetArray("PID") is not None:
            from vtkmodules.util.numpy_support import vtk_to_numpy

            drawn |= set(vtk_to_numpy(data.GetCellData().GetArray("PID")).tolist())
    assert drawn == {3}
    assert window.page_result.fit_range.isEnabled()
    window.page_result.fit_range.setChecked(True)
    bar = window.plotter.scalar_bars[FIELD]
    assert bar.GetLookupTable().GetRange()[1] == pytest.approx(window.session.field_range(FIELD, [3])[1])
    window.set_part_view([_item(window, "region", p) for p in (1, 2)], VISIBLE)
    assert not window.page_result.fit_range.isEnabled()


def test_show_only_hides_the_others(window):
    muscle = _item(window, "region", 3)
    siblings = [i for i in window._tree_items() if i.data(0, 256) == "region"]
    window._show_only([muscle], siblings)
    assert [window.session.region_view(p) for p in (1, 2, 3)] == [HIDDEN, HIDDEN, VISIBLE]
    window.set_part_view(siblings, VISIBLE)


def test_hidden_electrodes_lose_their_outline(window):
    from facetsetfem.gui.main_window import ELECTRODE_OUTLINE_COLORS
    from pyvista import Color

    anode = Color(ELECTRODE_OUTLINE_COLORS["anode"]).float_rgb

    def anode_outlines():
        return [
            a for a in window.plotter.renderer.actors.values()
            if hasattr(a, "GetProperty") and hasattr(a.GetProperty(), "GetColor")
            and tuple(a.GetProperty().GetColor()) == pytest.approx(anode)
        ]

    window.select_item("result", None)
    assert anode_outlines()
    window.set_part_view([_item(window, "electrode", "E1")], HIDDEN)
    assert not anode_outlines()
    window.set_part_view([_item(window, "electrode", "E1")], VISIBLE)
