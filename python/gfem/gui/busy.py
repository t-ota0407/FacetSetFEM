"""処理中のローディング表示（SPEC §6.4）。

依存を増やさない方針（`widgets.py` と同じ）なので、スピナーは QPainter で描く。

使い方は 2 通りある。**どちらを使うかは「その処理が GIL を手放すか」で決まる**。

1. Rust 側が GIL を手放す処理（読み込み・VTU 書き出し）はワーカースレッドへ出す。
   イベントループが回り続けるのでスピナーが滑らかに回る::

       window.run_busy("メッシュを読み込んでいます", lambda: session.open_mesh(path, unit))

   呼び出し側から見ると**同期関数のまま**（結果を返し、例外はそのまま送出される）
   なので、既存の `try` / `except` を書き換えずに済む

2. VTK と Qt は他スレッドから触れないので、描画まわりはメインスレッドに残す。
   こちらは区切りごとに `step()` を呼び、そこで表示を進める::

       with window.busy("表示を作成しています") as scope:
           grid = session.grid()
           scope.step("描画しています")
           plotter.render()

**短い処理では出さない。** 既定で 200 ms を超えてから現れるので、小さなモデルを
クリックして回っている間はローディングが一度も見えない（ちらつきを避ける）。
"""

from __future__ import annotations

import time
from contextlib import contextmanager

from PySide6.QtCore import QEvent, QEventLoop, QPoint, QRectF, Qt, QThread, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

__all__ = ["BusyOverlay", "BusyScope", "busy_scope", "run_in_thread"]

#: これより短い処理ではローディングを出さない [ms]
REVEAL_DELAY_MS = 200

#: スピナーの更新間隔 [ms] と 1 回転にかける時間 [ms]
FRAME_MS = 16
PERIOD_MS = 1100

#: 経過時間を出し始めるまで [s]。長い処理で「生きている」ことを示すため
ELAPSED_AFTER_S = 3.0

# 配色。ConvergencePlot と同じ青を差し色にして GUI 全体で色を揃える
_SCRIM = QColor(247, 248, 250, 208)
_CARD = QColor(255, 255, 255)
_BORDER = QColor(0, 0, 0, 20)
_TRACK = QColor(228, 232, 238)
_ACCENT = QColor(30, 90, 190)
_TITLE = QColor(32, 38, 46)
_DETAIL = QColor(122, 130, 142)


class BusyOverlay(QWidget):
    """ウィンドウを覆う、スピナー付きの半透明レイヤ。

    - 親ウィンドウの移動・リサイズに追従する（イベントフィルタ）
    - マウス操作を飲み込むので、処理中に下のウィジェットを触れない
    - `begin()` / `end()` は入れ子にできる。内側は文言だけ差し替え、
      いちばん外側が終わったときに消える

    **子ウィジェットではなく、親に貼りついた枠なしウィンドウとして作る。**
    PyVista の 3D ビューは VTK にウィンドウハンドルを渡すネイティブウィンドウで、
    ふつうの子ウィジェットを重ねてもその下に隠れてしまうため（Qt の兄弟同士では
    ネイティブウィンドウが必ず手前に来る）。ウィンドウなら半透明合成も効く。
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setVisible(False)
        parent.installEventFilter(self)

        #: (文言, 補足) の入れ子スタック
        self._stack: list[tuple[str, str]] = []
        self._angle = 0.0
        self._started = 0.0
        self._card_w = 0.0

        self._spin = QTimer(self)
        self._spin.setInterval(FRAME_MS)
        self._spin.timeout.connect(self._advance)

        self._reveal = QTimer(self)
        self._reveal.setSingleShot(True)
        self._reveal.timeout.connect(self._show_now)

    # --- 開始と終了 ---------------------------------------------------------

    def begin(self, message: str, detail: str = "", delay_ms: int | None = None) -> None:
        """処理の開始を伝える。既定では `REVEAL_DELAY_MS` 後に現れる。

        `delay_ms=0` を渡すと即座に出す。**遅らせてよいのは「速いと分かっている」
        ときだけ**で、重い処理の最中は画面を更新する隙が無く、タイマーが発火
        できないまま処理が終わってしまう（最後に一瞬だけ光る）。
        """
        self._stack.append((message, detail))
        if len(self._stack) == 1:
            self._started = time.monotonic()
            self._angle = 0.0
            self._card_w = 0.0
            self._reveal.start(REVEAL_DELAY_MS if delay_ms is None else delay_ms)
        elif self.isVisible():
            self.update()

    def end(self) -> None:
        """処理の終了を伝える。いちばん外側なら消える。"""
        if self._stack:
            self._stack.pop()
        if self._stack:
            if self.isVisible():
                self.update()
            return
        self._reveal.stop()
        self._spin.stop()
        self.hide()

    def set_message(self, message: str, detail: str | None = None) -> None:
        """いま表示している文言を差し替える。"""
        if not self._stack:
            return
        previous = self._stack[-1]
        self._stack[-1] = (message, previous[1] if detail is None else detail)
        if self.isVisible():
            self.update()

    def pump(self) -> None:
        """メインスレッドを塞ぐ処理の合間に、画面だけ描き直す。

        ユーザ入力は意図的に処理しない。処理中のクリックで別の重い処理が
        走り出す（再入する）のを防ぐため。
        """
        if self._stack:
            QApplication.processEvents(
                QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
            )

    @property
    def active(self) -> bool:
        """いま何らかの処理を待っているか（表示前の 200 ms も含む）。"""
        return bool(self._stack)

    @property
    def message(self) -> str:
        return self._stack[-1][0] if self._stack else ""

    # --- Qt -----------------------------------------------------------------

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt の規約
        if watched is self.parent():
            kind = event.type()
            if kind in (QEvent.Type.Resize, QEvent.Type.Move) and self.isVisible():
                self._follow_parent()
            elif kind in (QEvent.Type.Hide, QEvent.Type.Close):
                self.hide()
        return False

    def _follow_parent(self) -> None:
        """親ウィンドウにぴったり重ねる（別ウィンドウなので座標は画面基準）。"""
        parent = self.parent()
        self.setGeometry(
            parent.mapToGlobal(QPoint(0, 0)).x(),
            parent.mapToGlobal(QPoint(0, 0)).y(),
            parent.width(),
            parent.height(),
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt の規約
        event.accept()  # 下のウィジェットには通さない

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt の規約
        event.accept()

    def _show_now(self) -> None:
        # 親が画面に出ていないとき（テストや起動直後）は何も出さない
        if not self._stack or not self.parent().isVisible():
            return
        self._follow_parent()
        self.show()
        self.raise_()
        self._spin.start()
        self.repaint()

    def _advance(self) -> None:
        self._angle = (self._angle + 360.0 * FRAME_MS / PERIOD_MS) % 360.0
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt の規約
        del event
        if not self._stack:
            return
        message, detail = self._stack[-1]
        elapsed = time.monotonic() - self._started
        if elapsed >= ELAPSED_AFTER_S:
            suffix = f"経過 {elapsed:.0f} 秒"
            detail = f"{detail}　{suffix}" if detail else suffix

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), _SCRIM)

        title_font = QFont(self.font())
        title_font.setBold(True)
        title_font.setPointSizeF(title_font.pointSizeF() + 0.5)
        detail_font = QFont(self.font())
        detail_font.setPointSizeF(max(7.0, detail_font.pointSizeF() - 0.5))
        title_metrics = QFontMetrics(title_font)
        detail_metrics = QFontMetrics(detail_font)

        pad, gap, spinner = 26.0, 16.0, 30.0
        text_w = float(title_metrics.horizontalAdvance(message))
        if detail:
            text_w = max(text_w, float(detail_metrics.horizontalAdvance(detail)))
        text_w += 4.0  # 端がぎりぎりだと 1 px 足りずに「...」に化ける
        # 文言が変わっても幅は縮めない。経過秒が出た瞬間などにカードが
        # 伸び縮みして目障りになるのを防ぐ（表示のたびに `begin` で戻す）
        self._card_w = max(self._card_w, pad * 2 + spinner + gap + text_w)
        card_w = max(240.0, min(self.width() - 32.0, self._card_w))
        card_h = 86.0
        card = QRectF(
            (self.width() - card_w) / 2.0,
            (self.height() - card_h) / 2.0,
            card_w,
            card_h,
        )

        # 影。角丸を外側へ薄く重ねるだけで、グラフィックスエフェクトは使わない
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(15, 23, 42, 5))
        for i in range(6, 0, -1):
            painter.drawRoundedRect(card.adjusted(-i, -i + 1, i, i + 2), 14 + i, 14 + i)

        painter.setBrush(_CARD)
        painter.setPen(QPen(_BORDER, 1))
        painter.drawRoundedRect(card, 14, 14)

        # スピナー（下地の輪 + 回る円弧）
        ring = QRectF(card.left() + pad, card.center().y() - spinner / 2, spinner, spinner)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(_TRACK, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(ring, 0, 360 * 16)
        painter.setPen(QPen(_ACCENT, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(ring, int(-self._angle * 16), -110 * 16)

        # 文言
        left = ring.right() + gap
        available = max(1.0, card.right() - pad - left)
        centre = card.center().y()
        painter.setFont(title_font)
        painter.setPen(_TITLE)
        title = title_metrics.elidedText(
            message, Qt.TextElideMode.ElideRight, int(available)
        )
        flags = int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if detail:
            painter.drawText(QRectF(left, centre - 23, available, 22), flags, title)
            painter.setFont(detail_font)
            painter.setPen(_DETAIL)
            painter.drawText(
                QRectF(left, centre + 1, available, 20),
                flags,
                detail_metrics.elidedText(
                    detail, Qt.TextElideMode.ElideRight, int(available)
                ),
            )
        else:
            painter.drawText(QRectF(left, centre - 11, available, 22), flags, title)


class BusyScope:
    """`with window.busy(...)` が返すもの。処理の区切りで `step()` を呼ぶ。"""

    def __init__(self, overlay: BusyOverlay) -> None:
        self._overlay = overlay

    def step(self, message: str, detail: str | None = None) -> None:
        """次の段階に進んだことを表示する（ここで画面が描き直される）。"""
        self._overlay.set_message(message, detail)
        self._overlay.pump()


@contextmanager
def busy_scope(
    overlay: BusyOverlay, message: str, detail: str = "", delay_ms: int | None = None
):
    """メインスレッドを塞ぐ処理を囲む。"""
    overlay.begin(message, detail, delay_ms)
    overlay.pump()
    try:
        yield BusyScope(overlay)
    finally:
        overlay.end()


class _Task(QThread):
    """任意の呼び出しをワーカースレッドで 1 回だけ実行する。"""

    def __init__(self, work) -> None:
        super().__init__()
        self._work = work
        self.result = None
        self.error: BaseException | None = None

    def run(self) -> None:  # noqa: D102 - QThread の規約
        try:
            self.result = self._work()
        except BaseException as exc:  # noqa: BLE001 - 呼び出し元へ投げ直す
            self.error = exc


def run_in_thread(
    overlay: BusyOverlay,
    message: str,
    work,
    detail: str = "",
    delay_ms: int | None = None,
):
    """`work()` をワーカースレッドで走らせ、終わるまでローディングを出して待つ。

    **呼び出し側から見ると同期関数のまま**（戻り値を返し、例外はそのまま送出する）。
    待つ間もイベントループを回すのでスピナーが止まらない。

    回すのは**ユーザ入力以外のイベントだけ**。読み込みの最中に Ctrl+O が効いて
    2 つ目の読み込みが始まる、といった再入を防ぐため（クリックはローディングが
    受け止めるが、ショートカットキーはウィンドウに届いてしまう）。
    """
    task = _Task(work)
    loop = QEventLoop()
    task.finished.connect(loop.quit)

    # 保険。finished を取りこぼしてもここで抜ける（イベントループに閉じ込めない）
    watchdog = QTimer()
    watchdog.setInterval(50)
    watchdog.timeout.connect(lambda: loop.quit() if task.isFinished() else None)

    overlay.begin(message, detail, delay_ms)
    try:
        task.start()
        watchdog.start()
        while not task.isFinished():
            loop.exec(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
    finally:
        watchdog.stop()
        task.wait()
        overlay.end()

    if task.error is not None:
        raise task.error
    return task.result
