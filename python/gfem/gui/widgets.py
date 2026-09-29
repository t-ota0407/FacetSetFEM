"""GUI の部品（SPEC §6.1）。

依存を増やさないため、収束曲線は QPainter で自前に描く。
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QTreeWidget, QWidget

__all__ = [
    "ConvergencePlot",
    "MixedValueSpinBox",
    "PartTree",
    "UnitSelector",
    "part_view_icon",
]

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QLabel,
    QVBoxLayout,
)

from .. import describe_extent, length_units, metres_per_unit
from ..i18n import N_, tr


class ConvergencePlot(QWidget):
    """相対残差の収束曲線（縦軸は常用対数）。SPEC §4.3 / §6.2 の手順 5。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._history: list[float] = []
        self._tolerance: float | None = None
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def clear(self) -> None:
        self._history = []
        self.update()

    def set_tolerance(self, tolerance: float | None) -> None:
        self._tolerance = tolerance
        self.update()

    def append(self, residual: float) -> None:
        self._history.append(residual)
        self.update()

    def set_history(self, history) -> None:
        self._history = [float(v) for v in history]
        self.update()

    @property
    def history(self) -> list[float]:
        return list(self._history)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt の規約
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(250, 250, 250))

        metrics = QFontMetrics(self.font())
        left = metrics.horizontalAdvance("1e-00") + 8
        bottom = metrics.height() + 6
        plot_w = max(1, self.width() - left - 10)
        plot_h = max(1, self.height() - bottom - 10)

        values = [v for v in self._history if v > 0.0]
        if len(values) < 2:
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                tr("Convergence curve (appears when you solve)"),
            )
            return

        top = math.log10(max(values))
        low = math.log10(min(values))
        if self._tolerance:
            low = min(low, math.log10(self._tolerance))
        if top - low < 1.0:
            low = top - 1.0

        def to_point(index: int, residual: float) -> QPointF:
            x = left + plot_w * index / max(1, len(self._history) - 1)
            t = (math.log10(max(residual, 1e-300)) - low) / (top - low)
            y = 10 + plot_h * (1.0 - min(max(t, 0.0), 1.0))
            return QPointF(x, y)

        # 目盛り（10 の冪ごと）
        painter.setPen(QPen(QColor(220, 220, 220), 1))
        for exponent in range(math.floor(low), math.ceil(top) + 1):
            y = to_point(0, 10.0**exponent).y()
            painter.drawLine(QPointF(left, y), QPointF(left + plot_w, y))
            painter.setPen(QColor(140, 140, 140))
            painter.drawText(QPointF(2, y + metrics.ascent() / 2), f"1e{exponent}")
            painter.setPen(QPen(QColor(220, 220, 220), 1))

        # 収束判定のしきい値
        if self._tolerance:
            painter.setPen(QPen(QColor(200, 80, 80), 1, Qt.PenStyle.DashLine))
            y = to_point(0, self._tolerance).y()
            painter.drawLine(QPointF(left, y), QPointF(left + plot_w, y))

        # 残差
        painter.setPen(QPen(QColor(30, 90, 190), 2))
        points = [to_point(i, v) for i, v in enumerate(self._history) if v > 0.0]
        painter.drawPolyline(points)

        painter.setPen(QColor(60, 60, 60))
        painter.drawText(
            QPointF(left, self.height() - 4),
            tr(
                "Iteration {n}   relative residual {r:.3e}",
                n=len(self._history) - 1,
                r=self._history[-1],
            ),
        )


class MixedValueSpinBox(QDoubleSpinBox):
    """「値がそろっていない」を表せる数値欄（複数の領域をまとめて編集するとき）。

    `set_mixed()` の間は下限の 1 つ下を番兵にして `MIXED_TEXT` を出す（Qt の
    specialValueText の仕組み）。数値を打つか上矢印を押すとふつうの値に戻り、
    `valueChanged` で入った値を選んだ全部に入れてもらう。

    - **打っている途中で範囲を変えない。** 範囲を変えると Qt が欄を書き直し、打ちかけの
      文字が消える。番兵の下限は、編集が終わったとき（`editingFinished`）に戻す
    - **番兵の値を外へ漏らさない。** 上矢印は番兵からではなく下限から数える
      （番兵から数えると下限を割った値が出る）。下矢印では何もしない
    - 番兵の値そのもので `valueChanged` は出さない。受け手は `value() < floor` を無視すればよい

    ほかに、下限ちょうどの値に別の文字を出せる（`set_floor_text`。σ の「未設定」）。
    数値は末尾の 0 を落として出す（小数 6 桁の欄で 0.04 が「0.040000」と出ていた）。
    """

    MIXED_TEXT = N_("(values differ)")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._floor = 0.0
        self._mixed = False
        #: 下限ちょうどの値のときに出す文字（空なら数値のまま）
        self._floor_text = ""
        # 受け手より先に呼ばれるよう、最初につなぐ
        self.valueChanged.connect(self._on_value_changed)
        self.editingFinished.connect(self._restore_range)

    @property
    def floor(self) -> float:
        """ふつうの値の下限（番兵はこれより下）。"""
        return self._floor

    def setRange(self, lo: float, hi: float) -> None:  # noqa: N802 - Qt の名前に合わせる
        self._floor = lo
        super().setRange(lo - 1.0 if self._mixed else lo, hi)

    def set_floor_text(self, text: str) -> None:
        """下限ちょうどの値のときに、数値の代わりに `text` を出す（σ = 0 の「未設定」）。"""
        self._floor_text = text
        if not self._mixed:
            self.setSpecialValueText(text)

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt の名前に合わせる
        """末尾の 0 を落とす（0.040000 → 0.04）。桁数の上限は `decimals()` のまま。"""
        text = f"{value:.{self.decimals()}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"

    @property
    def is_mixed(self) -> bool:
        return self._mixed

    def set_mixed(self) -> None:
        """値がそろっていないことを出す（`valueChanged` は出さない）。"""
        self.blockSignals(True)
        self._mixed = True
        super().setRange(self._floor - 1.0, self.maximum())
        self.setSpecialValueText(tr(self.MIXED_TEXT))
        self.setValue(self._floor - 1.0)
        self.blockSignals(False)

    def set_value(self, value: float) -> None:
        """ふつうの値を出す（`valueChanged` は出さない）。"""
        self.blockSignals(True)
        self._mixed = False
        self._restore_range()
        self.setValue(value)
        self.blockSignals(False)

    def _on_value_changed(self, value: float) -> None:
        if self._mixed and value >= self._floor:
            self._mixed = False  # 数値が打たれた。範囲は打ち終わってから戻す

    def _restore_range(self) -> None:
        if self._mixed or self.minimum() >= self._floor:
            return
        self.setSpecialValueText(self._floor_text)
        super().setRange(self._floor, self.maximum())

    def stepBy(self, steps: int) -> None:  # noqa: N802 - Qt の名前に合わせる
        if self._mixed:
            if steps <= 0:
                return  # 下限より下は無い。そろっていないまま
            self.blockSignals(True)
            self._mixed = False
            self._restore_range()
            self.setValue(self._floor)
            self.blockSignals(False)
        super().stepBy(steps)


class UnitSelector(QDialog):
    """長さ単位を選ぶダイアログ（SPEC §3.2 / §6.2 の手順 1）。

    Nastran ファイルには単位情報が無い。単位の取り違えは結果を桁違いに狂わせ、
    しかも一見もっともらしい絵が出てしまうので、**選んだ単位と m の両方で
    バウンディングボックス寸法を表示する**ことを必須要件にしている。
    """

    def __init__(self, raw_extent: tuple[float, float, float], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Choose the length unit"))
        self._raw = raw_extent

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(tr("Nastran files carry no unit information. Choose the unit of the coordinates."))
        )
        self.combo = QComboBox()
        self.combo.addItems(length_units())
        self.combo.setCurrentText("mm")
        layout.addWidget(self.combo)

        self.preview = QLabel()
        self.preview.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.preview)
        layout.addWidget(
            QLabel(tr("If the size is off by orders of magnitude, check the unit."))
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.combo.currentTextChanged.connect(self._update_preview)
        self._update_preview(self.combo.currentText())

    @property
    def length_unit(self) -> str:
        return self.combo.currentText()

    def _update_preview(self, unit: str) -> None:
        # ファイル中の生の寸法を、選んだ単位で読んだときの m に直して見せる
        scale = metres_per_unit(unit)
        self.preview.setText(describe_extent([v * scale for v in self._raw], unit))


class PartTree(QTreeWidget):
    """モデルツリー。2 列目に領域・電極の見せ方（表示 / 半透明 / 非表示）の目のアイコンを出す。

    **目の列をクリックしても選択を変えない。** 見せ方を変えるたびに、右のページが
    その項目に切り替わったり、複数選択が外れたりすると使いにくい。押したら
    `viewClicked(item)` を出すだけにする（見せ方の切り替えはメインウィンドウが行う）。
    """

    #: 目の列（列番号）
    VIEW_COLUMN = 1

    viewClicked = Signal(object)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt の規約
        index = self.indexAt(event.position().toPoint())
        item = self.itemFromIndex(index) if index.isValid() else None
        if (
            item is not None
            and index.column() == self.VIEW_COLUMN
            and event.button() == Qt.MouseButton.LeftButton
            and not item.icon(self.VIEW_COLUMN).isNull()
        ):
            self.viewClicked.emit(item)
            event.accept()
            return
        super().mousePressEvent(event)


_VIEW_ICONS: dict[str, QIcon] = {}


def part_view_icon(state: str) -> QIcon:
    """見せ方のアイコン（目）。表示は濃く、半透明は薄く、非表示は斜線入り。"""
    if state in _VIEW_ICONS:
        return _VIEW_ICONS[state]
    size = 32
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    colour = {
        "visible": QColor(40, 44, 52),
        "transparent": QColor(40, 44, 52, 110),
        "hidden": QColor(150, 150, 150),
    }[state]
    painter.setPen(QPen(colour, 2.6))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(QRectF(3, 9, 26, 14))
    if state != "hidden":
        painter.setBrush(colour)
        painter.drawEllipse(QPointF(16, 16), 4.5, 4.5)
    else:
        painter.drawLine(QPointF(6, 27), QPointF(26, 5))
    painter.end()
    _VIEW_ICONS[state] = QIcon(pixmap)
    return _VIEW_ICONS[state]
