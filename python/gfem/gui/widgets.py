"""GUI の部品（SPEC §6.1）。

依存を増やさないため、収束曲線は QPainter で自前に描く。
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

__all__ = ["ConvergencePlot", "UnitSelector"]

from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QLabel, QVBoxLayout

from .. import length_units

#: 1 単位が何メートルか
_METRES_PER_UNIT = {"mm": 1.0e-3, "cm": 1.0e-2, "m": 1.0, "inch": 0.0254}


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
                self.rect(), Qt.AlignmentFlag.AlignCenter, "収束曲線（solve を実行すると表示されます）"
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
            f"反復 {len(self._history) - 1}　相対残差 {self._history[-1]:.3e}",
        )


class UnitSelector(QDialog):
    """長さ単位を選ぶダイアログ（SPEC §3.2 / §6.2 の手順 1）。

    Nastran ファイルには単位情報が無い。単位の取り違えは結果を桁違いに狂わせ、
    しかも一見もっともらしい絵が出てしまうので、**選んだ単位と m の両方で
    バウンディングボックス寸法を表示する**ことを必須要件にしている。
    """

    def __init__(self, raw_extent: tuple[float, float, float], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("長さ単位の選択")
        self._raw = raw_extent

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("Nastran ファイルには単位情報がありません。座標値の単位を選んでください。")
        )
        self.combo = QComboBox()
        self.combo.addItems(length_units())
        self.combo.setCurrentText("mm")
        layout.addWidget(self.combo)

        self.preview = QLabel()
        self.preview.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.preview)
        layout.addWidget(
            QLabel("寸法が想定と桁違いなら、単位の選択を見直してください。")
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
        self.preview.setText(format_extent(self._raw, unit))


def format_extent(raw_extent, unit: str) -> str:
    """ファイル中の生の寸法を、選んだ単位と m の両方で表した文字列にする。"""
    scale = _METRES_PER_UNIT[unit]
    in_unit = " × ".join(f"{v:.6g} {unit}" for v in raw_extent)
    in_metre = " × ".join(f"{v * scale:.6g} m" for v in raw_extent)
    return f"{in_unit}  =  {in_metre}"
