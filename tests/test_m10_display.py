"""3D ビューの描画（SPEC §6.3）。

**PID で色分けしない。** PID は領域を識別する番号で大小に意味が無く、連続的な
カラーマップに載せると「51 は 62 より小さい」という無い情報を描いてしまう。
形は陰影と稜線で見せ、色は物理量のためにとっておく。
"""

from __future__ import annotations

import pytest

from facetsetfem.gui.state import DEFAULT_RESULT_FIELD, SHAPE_FIELD, Session

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


def test_the_shape_button_restores_the_view_but_keeps_the_result(window, qt_app, monkeypatch):
    """「形状表示に戻す」は見た目だけを戻し、結果は残すこと。描き直しは 1 回だけ。"""
    page = window.page_result
    window.start_solve()
    assert window.solver.wait(60_000)
    qt_app.processEvents()
    page.mode.setCurrentIndex(page.mode.findData("slice"))
    page.streamlines.setChecked(True)
    assert DEFAULT_RESULT_FIELD in window.plotter.scalar_bars.keys()

    draws = []
    original = window.refresh_view
    monkeypatch.setattr(window, "refresh_view", lambda: (draws.append(1), original()))
    page.shape_button.click()

    assert page.field.currentData() == SHAPE_FIELD
    assert page.mode.currentData() == "volume"
    assert not page.streamlines.isChecked()
    assert len(draws) == 1, "3 つの設定を変えても描き直しは 1 回"
    assert not window.plotter.scalar_bars.keys()
    assert window.session.solution is not None, "結果は消さない"
    # 選び直せば、解き直さずにそのまま見られる
    page.field.setCurrentIndex(page.field.findData(DEFAULT_RESULT_FIELD))
    assert DEFAULT_RESULT_FIELD in window.plotter.scalar_bars.keys()


def _solve(window, qt_app):
    window.start_solve()
    assert window.solver.wait(60_000)
    qt_app.processEvents()


def test_the_same_button_goes_back_to_the_result(window, qt_app, monkeypatch):
    """ボタン 1 つで形状表示と結果の表示を行き来できること。解き直さない。"""
    from facetsetfem.gui.pages import SHOW_RESULT_LABEL, SHOW_SHAPE_LABEL
    from facetsetfem.i18n import tr

    page = window.page_result
    _solve(window, qt_app)
    page.field.setCurrentIndex(page.field.findData("|E| [V/m]"))
    page.mode.setCurrentIndex(page.mode.findData("slice"))
    page.streamlines.setChecked(True)
    solution = window.session.solution

    page.shape_button.click()
    assert page.shape_button.text() == tr(SHOW_RESULT_LABEL)
    assert page.shape_button.isEnabled()

    draws = []
    original = window.refresh_view
    monkeypatch.setattr(window, "refresh_view", lambda: (draws.append(1), original()))
    solves = []
    monkeypatch.setattr(window, "start_solve", lambda: solves.append(1))
    page.shape_button.click()

    # 形状表示の前の見方（表示量・表示モード・流線）がそのまま戻る
    assert page.field.currentData() == "|E| [V/m]"
    assert page.mode.currentData() == "slice"
    assert page.streamlines.isChecked()
    assert "|E| [V/m]" in window.plotter.scalar_bars.keys()
    assert len(draws) == 1, "3 つの設定を戻しても描き直しは 1 回"
    assert not solves and window.session.solution is solution, "解き直していない"
    assert page.shape_button.text() == tr(SHOW_SHAPE_LABEL)


def test_choosing_another_view_by_hand_forgets_the_saved_one(window, qt_app):
    """形状表示の間に自分で表示量を選び直したら、覚えていた見方は捨てる。

    もう別の見方に移っているのに「結果の表示に戻す」が残ると、押したときに
    いま選んだ見方が上書きされてしまう。稜線の切り替えは見方に含めない。
    """
    from facetsetfem.gui.pages import SHOW_RESULT_LABEL, SHOW_SHAPE_LABEL
    from facetsetfem.i18n import tr

    page = window.page_result
    _solve(window, qt_app)
    page.shape_button.click()
    edges = page.edges.isChecked()
    page.edges.setChecked(not edges)
    page.edges.setChecked(edges)  # MainWindow は共有なので元に戻しておく
    assert page.shape_button.text() == tr(SHOW_RESULT_LABEL), "稜線の切り替えでは忘れない"

    page.field.setCurrentIndex(page.field.findData("phi [V]"))
    assert page.shape_button.text() == tr(SHOW_SHAPE_LABEL)
    page.shape_button.click()
    page.shape_button.click()
    assert page.field.currentData() == "phi [V]", "選び直した見方に戻る"


def test_a_new_solve_forgets_the_view_of_the_old_result(window, qt_app):
    """形状表示のまま解き直したら、新しい結果の |J| を出し、古い見方は忘れる。"""
    from facetsetfem.gui.pages import SHOW_SHAPE_LABEL
    from facetsetfem.i18n import tr

    page = window.page_result
    _solve(window, qt_app)
    page.field.setCurrentIndex(page.field.findData("phi [V]"))
    page.shape_button.click()
    _solve(window, qt_app)
    assert page.field.currentData() == DEFAULT_RESULT_FIELD
    assert page.shape_button.text() == tr(SHOW_SHAPE_LABEL)


def test_the_button_is_disabled_when_there_is_nothing_to_change(window):
    """解く前の見た目のまま（戻す先も無い）なら、押しても何も変わらないので押せない。"""
    page = window.page_result
    assert page.field.currentData() == SHAPE_FIELD
    page.mode.setCurrentIndex(page.mode.findData("volume"))
    page.streamlines.setChecked(False)
    assert not page.shape_button.isEnabled()
    page.mode.setCurrentIndex(page.mode.findData("clip"))
    assert page.shape_button.isEnabled()


def test_edges_are_on_for_small_models_only(window, monkeypatch):
    """細かいメッシュで稜線を全部描くと、潰れて形がかえって見えなくなる。"""
    from facetsetfem.gui import pages

    assert window.page_result.edges.isChecked(), "小さなモデルでは稜線を出す"

    # 大きなモデルのつもりで開き直すと、既定で消える
    monkeypatch.setattr(pages, "EDGE_LIMIT", 10)
    window.page_result._mesh_key = None
    window.page_result.reload()
    assert not window.page_result.edges.isChecked()
