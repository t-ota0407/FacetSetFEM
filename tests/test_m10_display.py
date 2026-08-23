"""3D ビューの描画（SPEC §6.3）。

**PID で色分けしない。** PID は領域を識別する番号で大小に意味が無く、連続的な
カラーマップに載せると「51 は 62 より小さい」という無い情報を描いてしまう。
形は陰影と稜線で見せ、色は物理量のためにとっておく。
"""

from __future__ import annotations

import pytest

from gfem.gui.state import DEFAULT_RESULT_FIELD, SHAPE_FIELD, Session

SIGMA = {1: 0.10, 2: 0.04, 3: 0.35}


@pytest.fixture
def session(sample_bdf) -> Session:
    s = Session()
    s.open_mesh(sample_bdf, "mm")
    for pid, sigma in SIGMA.items():
        s.set_conductivity(pid, sigma)
    return s


def test_the_first_field_is_the_neutral_one(session):
    """既定は色分けなし。PID での塗り分けは選べるが既定ではない。"""
    fields = [f.key for f in session.available_fields()]
    assert fields[0] == SHAPE_FIELD
    assert "PID" in fields  # 選べはする（領域の確認に使うことがある）


def test_the_display_surface_is_light_and_has_normals(session):
    """表示に使うのは外表面で、法線を持っていること。

    法線が無いと（環境によっては）陰影が付かず、どんな形も**のっぺりした
    影絵**になる。体積メッシュをそのまま渡すのも遅すぎる（SPEC §6.3）。
    """
    surface = session.display_surface(SHAPE_FIELD)
    grid = session.grid()
    assert surface.n_cells < grid.n_cells, "外表面のほうが軽いはず"
    assert surface.n_cells == 944
    assert "Normals" in surface.point_data
    # 節点番号との対応が崩れていない（点は分裂させない）
    assert surface.n_points == session.mesh.n_nodes


def test_the_result_is_coloured_by_value(session):
    """結果があれば、値で色分けできること。"""
    session.add_electrode("E1", role="anode", current=2.0e-3, shell_pids=[101])
    session.add_electrode("E2", role="cathode", current=2.0e-3, shell_pids=[201])
    session.solve()

    surface = session.display_surface(DEFAULT_RESULT_FIELD)
    assert DEFAULT_RESULT_FIELD in surface.cell_data
    # 面には「その面を持つ要素の値」が乗る
    values = surface.cell_data[DEFAULT_RESULT_FIELD]
    assert len(values) == surface.n_cells
    assert values.min() >= 0.0
    # 面に乗るのは体表面の要素の値なので、全体の最大とは限らない
    assert values.max() <= session.solution.j_magnitude.max()

    # 色の範囲は**モデル全体**で決める。外表面だけ・断面だけで決めると、
    # 同じ色が場所によって違う値を指すことになる
    lo, hi = session.field_range(DEFAULT_RESULT_FIELD)
    assert lo == pytest.approx(session.solution.j_magnitude.min())
    assert hi == pytest.approx(session.solution.j_magnitude.max())
    assert session.field_range(SHAPE_FIELD) is None

    # 節点値も貼れる
    nodal = session.display_surface("phi [V]")
    assert "phi [V]" in nodal.point_data


def test_a_region_on_the_surface_is_highlighted_from_the_surface(session):
    """体表面に出ている領域は、外表面から切り出す（速いし表示と揃う）。"""
    skin = session.region_polydata(3)
    assert skin.n_cells > 0
    assert skin.n_cells < session.display_surface(SHAPE_FIELD).n_cells


# --- GUI ---------------------------------------------------------------------


@pytest.fixture
def window(main_window, sample_bdf):
    w = main_window
    w.open_mesh(sample_bdf, "mm")
    for pid, sigma in SIGMA.items():
        w.session.set_conductivity(pid, sigma)
    w.session.add_electrode("E1", role="anode", current=2.0e-3, shell_pids=[101])
    w.session.add_electrode("E2", role="cathode", current=2.0e-3, shell_pids=[201])
    w.refresh_tree()
    w.page_result.mode.setCurrentIndex(w.page_result.mode.findData("volume"))
    return w


def test_the_view_keeps_its_lights(window):
    """描き直しても光源が残ること。

    `plotter.clear()` は**光源も消す**。置き直さないと以降ずっと陰影が付かず、
    どんな形も影絵になる（気づきにくいので回帰として残す）。
    """
    window.refresh_view()
    lights = window.plotter.renderer.GetLights()
    assert lights.GetNumberOfItems() > 0, "描き直しで光源が消えている"
    window.refresh_view()
    assert window.plotter.renderer.GetLights().GetNumberOfItems() > 0


def test_the_default_view_has_no_colour_bar(window):
    """色分けなしのときはカラーバーを出さない。"""
    assert window.page_result.field.currentData() == SHAPE_FIELD
    window.refresh_view()
    assert not window.plotter.scalar_bars.keys()


def test_solving_switches_the_view_to_current_density(window, qt_app):
    """解いたら電流密度の表示に切り替わること。"""
    assert window.page_result.field.currentData() == SHAPE_FIELD
    window.start_solve()
    assert window.solver.wait(60_000)
    qt_app.processEvents()
    assert window.page_result.field.currentData() == DEFAULT_RESULT_FIELD
    assert DEFAULT_RESULT_FIELD in window.plotter.scalar_bars.keys()


def test_edges_are_on_for_small_models_only(window, monkeypatch):
    """細かいメッシュで稜線を全部描くと、潰れて形がかえって見えなくなる。"""
    from gfem.gui import main_window as mw

    assert window.page_result.edges.isChecked(), "小さなモデルでは稜線を出す"

    # 大きなモデルのつもりで開き直すと、既定で消える
    monkeypatch.setattr(mw, "EDGE_LIMIT", 10)
    window.page_result._mesh_key = None
    window.page_result.reload()
    assert not window.page_result.edges.isChecked()
