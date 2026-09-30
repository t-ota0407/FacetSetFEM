"""実機で操作して見つけた GUI の不具合の回帰テスト（SPEC §8 の「2026-09」）。

アプリを本物のウィンドウで起動し、操作を流し込んでスクリーンショットで確かめたときに
見つけたもの。どれも「動いてはいるが、見た目や流れがおかしい」種類で、単体のテストでは
拾えていなかった。
"""

from __future__ import annotations

import pytest

from gfem.gui.state import Session

SIGMA = {1: 0.10, 2: 0.04, 3: 0.35}
CURRENT = 2.0e-3
FIELD = "|J| [A/m2]"


def _setup(session: Session) -> None:
    for pid, sigma in SIGMA.items():
        session.set_conductivity(pid, sigma)
    session.add_electrode("E1", role="anode", current=CURRENT, shell_pids=[101])
    session.add_electrode("E2", role="cathode", current=CURRENT, shell_pids=[201])


# --- Qt を使わない部分 ------------------------------------------------------


def test_electrode_area_is_shown_before_any_conductivity_is_set(sample_bdf):
    """電極の面積は σ に依らない。σ 未設定でも出す（シェル要素の無いモデルの手順）。

    以前は σ の検査に引っかかり、面積の代わりに「導電率が未設定」と出ていた。
    """
    s = Session()
    s.open_mesh(sample_bdf, "mm")
    s.add_electrode("E1", role="anode", current=CURRENT, shell_pids=[101])
    info = s.electrode_info("E1")[0]
    assert info["area"] == pytest.approx(300e-6, rel=1e-12)
    assert info["current_density"] == pytest.approx(CURRENT / 300e-6, rel=1e-12)
    assert s.electrode_info()[0]["area"] == info["area"]
    # 覗いている間だけ外した σ と電極は、元どおりに戻っている
    assert [m.pid for m in s.study.materials] == [1, 2, 3]
    assert [e.name for e in s.study.electrodes] == ["E1"]


def test_unsaved_changes_are_found_by_content(sample_bdf, tmp_path):
    """閉じるときの確認に使う。記録ではなく中身を比べるので、Undo で戻せば変更なし。"""
    s = Session()
    s.open_mesh(sample_bdf, "mm")
    assert not s.is_modified()
    s.checkpoint("σ")
    s.set_conductivity(1, 0.1)
    assert s.is_modified()
    s.undo()
    assert not s.is_modified()
    s.set_conductivity(1, 0.1)
    s.save_case(tmp_path / "case.yaml")
    assert not s.is_modified()
    s.open_case(tmp_path / "case.yaml")
    assert not s.is_modified()


def test_the_status_line_keeps_the_substance_of_a_headed_message():
    """「エラー:」だけの見出し行は、2 行目の中身と続けてステータスバーに出す。"""
    from gfem.gui.main_window import status_line

    assert status_line("エラー:\n[設定エラー] 電極 E1 はすでにあります\n  対処: ...") == (
        "エラー: [設定エラー] 電極 E1 はすでにあります"
    )
    assert status_line("求解完了: 94 反復") == "求解完了: 94 反復"
    assert status_line("") == ""


@pytest.mark.parametrize("text", ["abc", "0.1, x", "0:1:y", "nan"])
def test_unreadable_sweep_values_are_explained_in_japanese(text):
    """以前は `could not convert string to float: 'abc'` がそのまま出ていた。"""
    from gfem.gui.sweep_dialog import parse_values

    with pytest.raises(ValueError) as excinfo:
        parse_values(text)
    message = str(excinfo.value)
    assert "could not convert" not in message
    assert "「" in message and "カンマか空白で区切る" in message


def test_a_cancelled_solve_is_not_reported_as_a_failure(sample_bdf, qt_app):
    """中断は失敗ではないので、別のシグナル（`cancelled`）で知らせる。"""
    from gfem.gui.solver_thread import SolverThread

    s = Session()
    s.open_mesh(sample_bdf, "mm")
    _setup(s)
    case, study = s.prepare_solve()
    thread = SolverThread(case, study)
    events: list[str] = []
    thread.failed.connect(lambda message: events.append(f"failed: {message}"))
    thread.cancelled.connect(lambda: events.append("cancelled"))
    thread.succeeded.connect(lambda _result: events.append("succeeded"))
    thread.cancel()  # 最初の反復で止まる
    thread.start()
    assert thread.wait(60_000)
    qt_app.processEvents()
    assert events == ["cancelled"]


# --- GUI ---------------------------------------------------------------------


@pytest.fixture
def window(main_window, sample_bdf, qt_app):
    w = main_window
    w.open_mesh(sample_bdf, "mm")
    _setup(w.session)
    w.refresh_tree()
    yield w
    # ウィンドウはテスト全体で 1 つなので、表示の設定を既定に戻しておく
    page = w.page_result
    page._loading = True
    page.mode.setCurrentIndex(0)
    page.streamlines.setChecked(False)
    page._loading = False
    w.page_electrode.unbind()


def _solve(window, qt_app) -> None:
    window.start_solve()
    assert window.solver.wait(60_000)
    qt_app.processEvents()


def test_the_isosurface_is_coloured_on_the_whole_model_range(window, qt_app):
    """等値面は表示量の見出しとモデル全体の値域で塗る。しきい値の実際の値も出す。

    以前は内部名（`j_magnitude`）で貼っていて色の範囲を引けず、範囲が等値面の値
    （しきい値 1 点）に潰れて、カラーバーの目盛りが全部同じ数字になっていた。
    """
    _solve(window, qt_app)
    page = window.page_result
    page.field.setCurrentIndex(page.field.findData(FIELD))
    page.mode.setCurrentIndex(page.mode.findData("isosurface"))
    assert FIELD in window.plotter.scalar_bars.keys()
    lo, hi = window.plotter.scalar_bars[FIELD].GetLookupTable().GetRange()
    assert (lo, hi) == pytest.approx(window.session.field_range(FIELD))
    assert hi > lo
    assert page.iso_value.text().startswith("いまのしきい値")
    page.mode.setCurrentIndex(page.mode.findData("volume"))
    assert page.iso_value.text() == "", "等値面をやめたら消す"


def test_streamlines_can_be_seen_through_the_surface(window, qt_app):
    """流線は体の中を通るので、重ねている間は外表面を透かす。"""
    from gfem.gui.main_window import STREAMLINE_SURFACE_OPACITY

    _solve(window, qt_app)
    window.page_result.streamlines.setChecked(True)
    opacities = sorted(
        actor.GetProperty().GetOpacity()
        for actor in window.plotter.renderer.actors.values()
        if hasattr(actor, "GetProperty")
    )
    assert STREAMLINE_SURFACE_OPACITY in opacities, "外表面が透けていない"
    window.page_result.streamlines.setChecked(False)
    for actor in window.plotter.renderer.actors.values():
        if hasattr(actor, "GetProperty"):
            assert actor.GetProperty().GetOpacity() == 1.0


def test_redrawing_keeps_the_users_zoom(window, sample_bdf):
    """ズームは描き直しで失わない。新しいメッシュを開いたときだけ全体に合わせ直す。"""
    def state():
        camera = window.plotter.camera
        return (tuple(camera.position), tuple(camera.focal_point), camera.view_angle)

    window.plotter.camera.Dolly(3.0)  # マウスホイールの拡大と同じ（視点を近づける）
    zoomed = state()
    window.page_result.cmap.setCurrentText("plasma")
    window.refresh_view(highlight_pid=2)
    assert state() == zoomed
    window.open_mesh(sample_bdf, "mm")
    assert state() != zoomed


def test_removing_an_electrode_leaves_its_page(window, qt_app):
    """消した電極のページ（名前・設定）を右のパネルに出したままにしない。"""
    window.session.add_electrode("E3", role="cathode", current=CURRENT, shell_pids=[201])
    window.refresh_tree()
    item = next(i for i in window._tree_items() if i.text(0).startswith("E3"))
    window.tree.setCurrentItem(item)
    assert window.pages.currentWidget() is window.page_electrode
    assert window.page_electrode.name == "E3"
    window.remove_selected_electrode()
    assert window.pages.currentWidget() is window.page_mesh
    assert window.page_electrode.name is None


def test_undo_reloads_the_open_page(window):
    """Undo で戻した値を、開いているページの欄にも戻す（以前は古い値のまま）。"""
    item = next(i for i in window._tree_items() if i.text(0).startswith("E1"))
    window.tree.setCurrentItem(item)
    page = window.page_electrode
    page.current_spin.setValue(5.0)
    assert window.session.electrode("E1").current == pytest.approx(5.0e-3)
    window.undo()
    assert window.session.electrode("E1").current == pytest.approx(CURRENT)
    assert page.current_spin.value() == pytest.approx(CURRENT * 1e3)


def test_the_mesh_page_is_filled_after_opening(window):
    """読み込んだ直後にメッシュのページを出し、中身も入れる（以前は空だった）。"""
    assert window.pages.currentWidget() is window.page_mesh
    assert "節点数" in window.page_mesh.text.toPlainText()


def test_pages_scroll_instead_of_stretching_the_window(window):
    """ページが縦に収まらなくても、ウィンドウの最小の高さを引っぱらない。

    以前は電極ページで矩形を選んでパッチを置くと、ウィンドウが勝手に縦に伸びた。
    """
    from PySide6.QtWidgets import QScrollArea

    for page in (window.page_mesh, window.page_region, window.page_electrode,
                 window.page_solver, window.page_result):
        assert isinstance(page, QScrollArea)
    item = next(i for i in window._tree_items() if i.text(0).startswith("E1"))
    window.tree.setCurrentItem(item)
    window.page_electrode.shape_combo.setCurrentIndex(1)  # 矩形は欄が増える
    window.page_result.show_staleness(["σ（PID 2）"])
    assert window.pages.minimumSizeHint().height() < 400


def test_closing_asks_about_unsaved_changes(window, monkeypatch):
    """保存していない変更があれば、閉じる前に確かめる（やめたら閉じない）。"""
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import QMessageBox

    asked: list[str] = []

    def answer(button):
        def question(_parent, title, _text, *_args):
            asked.append(title)
            return button
        return question

    window.session.set_conductivity(1, 0.2)
    assert window.session.is_modified()
    monkeypatch.setattr(QMessageBox, "question", answer(QMessageBox.StandardButton.Cancel))
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()
    assert asked == ["保存していない変更"]

    monkeypatch.setattr(QMessageBox, "question", answer(QMessageBox.StandardButton.Discard))
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()


# --- 見やすさ・使い勝手（2026-09 の見直しの続き）-----------------------------


def test_the_scalar_bar_title_drops_the_bars_and_splits_the_unit():
    """VTK の既定のフォントでは `|` が描かれない。量と単位は 2 行に分けて細い帯に収める。"""
    from gfem.gui.main_window import scalar_bar_title

    assert scalar_bar_title("|J| [A/m2]") == "J\n[A/m2]"
    assert scalar_bar_title("power density [W/m3]") == "power density\n[W/m3]"
    assert scalar_bar_title("PID") == "PID"
    # 長い名前は語の切れ目で折り返す（帯の外に切れないように。M11）
    assert scalar_bar_title("electrode current density [A/m2]") == (
        "electrode\ncurrent\ndensity\n[A/m2]"
    )


def test_the_scalar_bar_gets_its_own_strip_beside_the_model(window, qt_app):
    """カラーバーはモデルと別の描画領域（右端の帯）に置き、モデルと重ねない。"""
    _solve(window, qt_app)
    main = window.plotter.renderer
    strip = window._scalar_bar_renderer
    assert main.GetViewport()[2] < 1.0
    assert strip.GetViewport()[0] == pytest.approx(main.GetViewport()[2])
    assert strip.GetActors2D().GetNumberOfItems() >= 1  # バーと見出し
    window.page_result.show_shape()  # 色分けなし → バーは無く、全幅に戻る
    assert main.GetViewport()[2] == 1.0


def test_coloured_surfaces_are_not_darkened_too_much_by_shading(window, qt_app):
    """値で塗っている間は環境光を足し、光の当たらない面でも色が読めるようにする。"""
    from gfem.gui.main_window import SCALAR_LIGHTING

    _solve(window, qt_app)
    ambients = [
        actor.GetProperty().GetAmbient()
        for actor in window.plotter.renderer.actors.values()
        if hasattr(actor, "GetMapper")
        and hasattr(actor.GetProperty(), "GetAmbient")
        and actor.GetMapper().GetScalarVisibility()
    ]
    assert SCALAR_LIGHTING["ambient"] in ambients


def test_choosing_a_slice_faces_it_and_shows_the_body_outline(window, qt_app):
    """断面を選ぶと正面から見る向きにし、体の外形を薄く重ねる。"""
    import numpy as np

    from gfem.gui.main_window import SECTION_CONTEXT_OPACITY

    _solve(window, qt_app)
    page = window.page_result
    page.slice_normal.setCurrentText("y")
    page.mode.setCurrentIndex(page.mode.findData("slice"))
    direction = np.asarray(window.plotter.renderer.GetActiveCamera().GetDirectionOfProjection())
    assert abs(direction @ np.array([0.0, 1.0, 0.0])) == pytest.approx(1.0, abs=1e-9)
    opacities = [
        actor.GetProperty().GetOpacity()
        for actor in window.plotter.renderer.actors.values()
        if hasattr(actor, "GetProperty")
    ]
    assert SECTION_CONTEXT_OPACITY in opacities


def test_other_electrodes_are_outlined_while_one_is_edited(window):
    """選んでいない電極も、縁を役割の色で描いて位置を見せる。"""
    from gfem.gui.main_window import ELECTRODE_OUTLINE_COLORS
    from pyvista import Color

    outline = window.session.electrode_outline("E1")
    assert outline.n_cells > 0
    window.select_item("electrode", "E2")
    anode = Color(ELECTRODE_OUTLINE_COLORS["anode"]).float_rgb
    colours = [
        tuple(actor.GetProperty().GetColor())
        for actor in window.plotter.renderer.actors.values()
        if hasattr(actor, "GetProperty")
    ]
    assert any(c == pytest.approx(anode) for c in colours), "E1（陽極）の縁が無い"


def test_a_new_electrode_is_selected_right_away(window):
    window.add_electrode()
    assert window.pages.currentWidget() is window.page_electrode
    assert window.page_electrode.name == "E3"
    assert window.tree.currentItem().text(0).startswith("E3")


def test_a_bad_tolerance_is_explained_where_it_was_typed(window):
    from gfem.gui.pages import tolerance_problem

    assert tolerance_problem("1e-10") is None
    assert "範囲外" in tolerance_problem("5")
    assert "範囲外" in tolerance_problem("-1")
    assert "数値として読めません" in tolerance_problem("abc")
    window.select_item("solver", None)
    page = window.page_solver
    page.tolerance.setText("5")
    page.tolerance.editingFinished.emit()
    assert window.session.study.solver.tolerance == 1e-10, "範囲外の値は入れない"
    assert page.tolerance.text() == "1e-10"
    assert not page.tolerance_error.isHidden()


def test_every_problem_is_listed_at_once(sample_bdf):
    """整合性チェックは不備を全部まとめて出す（以前は 1 件ずつ）。"""
    s = Session()
    s.open_mesh(sample_bdf, "mm")
    s.add_electrode("E1", role="anode", current=0.0, shell_pids=[101])
    s.add_electrode("E2", role="anode", current=CURRENT, shell_pids=[101])
    problems, _warnings = s.problems()
    assert len(problems) == 4, problems
    assert "導電率が未設定" in problems[0]
    assert any("cathode が定義されていません" in p for p in problems)
    assert any("総電流が正の有限値ではありません" in p for p in problems)
    assert any("両方に含まれています" in p for p in problems)


def test_an_unset_conductivity_reads_as_unset(window):
    page = window.page_region
    window.session.set_conductivity(2, 0.0)
    page.bind(2)
    assert page.sigma_spin.text() == "未設定"
    page.bind(1)
    assert page.sigma_spin.text() == "0.1 S/m", "末尾の 0 は落とす"


def test_removing_a_patch_needs_a_selection(window):
    page = window.page_electrode
    window.select_item("electrode", "E1")
    assert not page.remove_patch_button.isEnabled()
    window.session.add_patch("E1", (0.030, 0.020, 0.020), 0.005, pid=1)
    page.bind("E1")
    assert not page.remove_patch_button.isEnabled()
    page.patch_list.setCurrentRow(0)
    assert page.remove_patch_button.isEnabled()


def test_only_the_controls_of_the_current_mode_can_be_used(window, qt_app):
    page = window.page_result
    assert not page.slice_box.isEnabled() and not page.iso_box.isEnabled()
    assert not page.stream_box.isEnabled(), "解く前は流線を引けない"
    _solve(window, qt_app)
    assert page.stream_box.isEnabled()
    page.mode.setCurrentIndex(page.mode.findData("clip"))
    assert page.slice_box.isEnabled() and not page.iso_box.isEnabled()
    page.mode.setCurrentIndex(page.mode.findData("isosurface"))
    assert page.iso_box.isEnabled() and not page.slice_box.isEnabled()


def test_statistics_carry_units_and_a_zero_reference_reads_as_zero(sample_bdf):
    from gfem.gui.state import REGION_STATS_COLUMNS

    assert "|J| max [A/m²]" in REGION_STATS_COLUMNS
    assert "|E| 99% [V/m]" in REGION_STATS_COLUMNS
    s = Session()
    s.open_mesh(sample_bdf, "mm")
    _setup(s)
    s.study.output.reference_potential = "cathode_mean"
    s.solve()
    summary = s.result_summary()
    assert "E2: 0 V" in summary, summary  # 以前は -5.3e-17 V と出ていた
    assert "電位の基準: 陰極の平均 = 0 V" in summary


def test_the_sample_model_has_skin_under_the_electrodes(sample_bdf):
    """サンプルモデルの電極は皮膚（PID 1）の上に貼ってある（以前は筋肉の上だった）。"""
    s = Session()
    s.open_mesh(sample_bdf, "mm")
    _setup(s)
    faces = s.electrode_polydata("E1")
    top = faces.points[:, 2].max()
    assert top == pytest.approx(0.020)
    pick = s.pick_patch((0.030, 0.020, 0.020), 0.001)
    assert pick["n_faces"] > 0
    assert s.pick_patch((0.030, 0.020, 0.020), 0.001, pid=1)["n_faces"] == pick["n_faces"]


# --- コミット後の見直しで見つけたもの -----------------------------------------


def test_overlays_on_the_surface_are_drawn_in_front(window, qt_app):
    """電極の面は体表面と同じ高さなので、奥行きの比較で手前に出す（z-fighting を避ける）。"""
    window.select_item("electrode", "E1")
    def offset(actor):
        from vtkmodules.vtkCommonCore import reference

        factor, units = reference(0.0), reference(0.0)
        actor.GetMapper().GetRelativeCoincidentTopologyPolygonOffsetParameters(factor, units)
        return float(units)

    overlays = [
        actor
        for actor in window.plotter.renderer.actors.values()
        if hasattr(actor.GetMapper(), "GetResolveCoincidentTopology")
        and actor.GetMapper().GetResolveCoincidentTopology() != 0
        and offset(actor) < 0
    ]
    assert overlays, "電極の面が手前に出されていない"


def test_redrawing_keeps_the_selected_electrode_highlighted(window, qt_app):
    """結果の取り込みや表示の設定の変更で描き直しても、選んでいる電極の強調は残る。"""
    from pyvista import Color

    def has_red_face():
        red = Color("red").float_rgb
        return any(
            tuple(a.GetProperty().GetColor()) == pytest.approx(red)
            and a.GetMapper().GetInput().GetNumberOfPolys() > 0
            for a in window.plotter.renderer.actors.values()
            if hasattr(a, "GetMapper") and hasattr(a.GetProperty(), "GetAmbient")
        )

    window.select_item("electrode", "E2")
    assert has_red_face()
    _solve(window, qt_app)
    assert has_red_face(), "求解の後で強調が消えた"
    window.page_result.cmap.setCurrentText("plasma")
    assert has_red_face(), "カラーマップを変えたら強調が消えた"


def test_leaving_a_section_restores_the_previous_view(window, qt_app):
    _solve(window, qt_app)
    camera = window.plotter.renderer.GetActiveCamera()
    before = window._camera_state(camera)
    page = window.page_result
    page.mode.setCurrentIndex(page.mode.findData("slice"))
    assert window._camera_state(camera) != before
    page.mode.setCurrentIndex(page.mode.findData("volume"))
    after = window._camera_state(camera)
    for a, b in zip(after, before):
        assert a == pytest.approx(b)


def test_the_undo_menu_stays_narrow_with_a_long_label(window):
    """操作名が長くても（領域をまとめて選んだときなど）、メニューは横に伸びない。全文はツールチップに。"""
    from gfem.gui.main_window import HISTORY_LABEL_LIMIT

    label = "領域 " + ", ".join(str(p) for p in range(1, 63)) + " の設定を変更"
    window.checkpoint(label)
    window.session.set_conductivity(1, 0.2)
    window._refresh_history_actions()
    text = window.action_undo.text()
    assert text == "元に戻す: " + label[: HISTORY_LABEL_LIMIT - 1] + "…"
    assert window.action_undo.toolTip() == f"元に戻す: {label}"
    window.undo()  # 記録の名前は切っていないので、そのまま戻せる
    assert window.action_redo.toolTip() == f"やり直す: {label}"
    assert window.action_redo.text() == "やり直す: " + label[: HISTORY_LABEL_LIMIT - 1] + "…"
