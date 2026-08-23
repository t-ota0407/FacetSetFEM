"""処理中のローディング表示（SPEC §6.4）の回帰テスト。

見た目そのものは自動で確かめられないので、**壊れると使えなくなる骨組み**を見る。

- 重い処理をワーカースレッドに出しても、呼び出し側からは同期関数のままであること
  （戻り値が返り、例外がそのまま飛んでくること）。ここが崩れると `open_mesh` の
  エラー処理が素通りする
- 短い処理では出ないこと（クリックのたびに一瞬光るのを防ぐ）
- 入れ子にしても、いちばん外側が終わるまで消えないこと
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6.QtWidgets")

from gfem.gui.busy import REVEAL_DELAY_MS, BusyOverlay, busy_scope, run_in_thread


@pytest.fixture
def overlay(qt_app):
    from PySide6.QtWidgets import QWidget

    host = QWidget()
    yield BusyOverlay(host)
    host.close()


def test_a_worker_result_comes_back_as_a_plain_return_value(overlay):
    assert run_in_thread(overlay, "計算中", lambda: 6 * 7) == 42
    assert not overlay.active


def test_a_worker_exception_is_raised_in_the_caller(overlay):
    """呼び出し側の `try` / `except` がそのまま効くこと。"""

    def boom():
        raise ValueError("読めません")

    with pytest.raises(ValueError, match="読めません"):
        run_in_thread(overlay, "読み込み中", boom)
    # 失敗しても表示は必ず片付く
    assert not overlay.active


def test_the_work_really_runs_off_the_main_thread(overlay):
    import threading

    main = threading.get_ident()
    assert run_in_thread(overlay, "計算中", threading.get_ident) != main


def test_nothing_is_shown_for_short_work(overlay):
    with busy_scope(overlay, "すぐ終わる処理"):
        pass
    assert not overlay.isVisible()


def test_the_scope_survives_an_exception(overlay):
    with pytest.raises(RuntimeError):
        with busy_scope(overlay, "途中で失敗する処理"):
            raise RuntimeError("失敗")
    assert not overlay.active


def test_nesting_keeps_the_overlay_up_until_the_outermost_ends(overlay):
    with busy_scope(overlay, "外側") as outer:
        assert overlay.message == "外側"
        with busy_scope(overlay, "内側"):
            assert overlay.message == "内側"
        # 内側が終わっても外側の文言に戻るだけで、表示は続く
        assert overlay.active
        assert overlay.message == "外側"
        outer.step("次の段階")
        assert overlay.message == "次の段階"
    assert not overlay.active


def test_the_reveal_delay_is_short_enough() -> None:
    """出るまでの待ちは、人が「固まった」と感じる前であること。"""
    assert 0 < REVEAL_DELAY_MS <= 300


def test_when_to_wait_before_showing(main_window, sample_bdf):
    """待つのは「速いと分かっている」ときだけ（SPEC §6.4）。

    大きなモデルでは領域を選び直すだけでも 1 秒近くかかり、その間は画面を更新する
    隙が無い。猶予を置くと処理の最後に一瞬だけ光って終わってしまう。
    """
    from gfem.gui.main_window import BIG_MODEL_NODES

    window = main_window
    window.open_mesh(sample_bdf, "mm")
    assert window.session.mesh.n_nodes < BIG_MODEL_NODES
    assert window._reveal_delay() is None, "小さなモデルではちらつき防止の猶予を置く"

    class FakeMesh:
        n_nodes = BIG_MODEL_NODES
        n_elements = 6 * BIG_MODEL_NODES

    small = window.session.mesh
    window.session.mesh = FakeMesh()
    try:
        assert window._reveal_delay() == 0, "大きなモデルでは待たずに出す"
        # 補足行にはモデルの規模を出す（待ち時間の目安になる）
        assert window._detail(None) == "100,000 節点 / 600,000 要素"
    finally:
        window.session.mesh = small


def test_a_scope_can_ask_for_the_normal_delay(main_window, sample_bdf):
    """段階の区切りが細かく、ふつうは一瞬で終わる処理は待ちを効かせられること。

    大きなモデルでは既定で「待たずに出す」が、電極をクリックで置く経路は
    1 回 0.1 秒程度なので、そのまま出すと**クリックのたびに一瞬光る**。
    """
    from gfem.gui.busy import REVEAL_DELAY_MS

    window = main_window
    window.open_mesh(sample_bdf, "mm")
    with window.busy("すぐ終わる処理", wait_ms=REVEAL_DELAY_MS):
        assert window.overlay.active
        assert not window.overlay.isVisible()
    assert not window.overlay.active


def test_the_overlay_appears_for_long_work(qt_app, overlay):
    """親が画面に出ている状態で長い処理を回すと、実際に現れて回る。"""
    parent = overlay.parent()
    parent.resize(400, 300)
    parent.show()
    qt_app.processEvents()
    if not parent.isVisible():  # オフスクリーン環境
        pytest.skip("ウィンドウを表示できない環境です")

    seen: list[bool] = []
    with busy_scope(overlay, "長い処理"):
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            qt_app.processEvents()
            seen.append(overlay.isVisible())
    assert any(seen), "長い処理でローディングが出ていない"
    assert not overlay.isVisible()
