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
    lo, hi = window.plotter.mapper.scalar_range
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
