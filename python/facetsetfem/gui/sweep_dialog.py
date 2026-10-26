"""パラメータスタディのダイアログ（SPEC §6.2 / §8 M7）。

振るパラメータと値を決めて一括で解き、結果を表で見比べる。
計算そのものは [`facetsetfem.sweep`](facetsetfem.sweep) がやる——**GUI にロジックは置かない**
（SPEC §7）ので、ここは入力を集めて結果を並べるだけ。

長くかかるのでワーカースレッドで走らせ、途中経過を進捗バーに出す。
"""

from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..i18n import N_, tr
from ..study import Study
from ..sweep import SweepTable, sweep

__all__ = ["SweepDialog", "parameter_choices"]


def parameter_choices(study: Study) -> list[tuple[str, str]]:
    """このケースで振れるパラメータの `(表示名, 指定子)` 一覧。

    電流は **mA** で入れる（電極のページと同じ単位）。以前はここだけ A で、
    電極のページの感覚で「2」と打つと 2 A（1000 倍）になった。`facetsetfem.sweep` には
    `display_scale` で A に直して渡す。
    """
    out: list[tuple[str, str]] = [(tr("Stimulation current (all electrodes) [mA]"), "current")]
    for m in study.materials:
        label = f"σ: PID {m.pid}" + (tr(" ({name})", name=m.name) if m.name else "")
        out.append((f"{label} [S/m]", f"sigma.{m.pid}"))
    for e in study.electrodes:
        out.append((tr("Current of electrode {name} [mA]", name=e.name), f"electrode.{e.name}.current"))
    # 接触インピーダンスを入れた電極は完全電極モデルになる（SPEC §2.5）。GUI では Ω·cm²
    out.append(
        (tr("Contact impedance (all electrodes, complete electrode model) [Ω·cm²]"), "contact_impedance")
    )
    for e in study.electrodes:
        out.append(
            (
                tr("Contact impedance of electrode {name} [Ω·cm²]", name=e.name),
                f"electrode.{e.name}.contact_impedance",
            )
        )
    out.append((tr("Convergence tolerance (relative residual)"), "solver.tolerance"))
    return out


def display_scale(key: str) -> float:
    """入力欄・表の値に掛けると SI になる係数（電流は mA で扱うので 1e-3）。"""
    if key == "current" or (key.startswith("electrode.") and key.endswith(".current")):
        return 1e-3
    # 接触インピーダンスは Ω·cm² で扱う（電極のページと同じ。1 Ω·cm² = 1e-4 Ω·m²）
    if key.endswith("contact_impedance"):
        return 1e-4
    return 1.0


#: 入力欄の例（振るものごと。表示の単位で書く）
VALUE_EXAMPLES = {
    "current": N_("1, 2, 4  or  1:4:1"),
    "sigma": N_("0.02, 0.04, 0.08  or  0.02:0.08:0.02"),
    "contact_impedance": "10, 100, 1000",
    "solver.tolerance": "1e-8, 1e-10, 1e-12",
}

#: 結果の列の見出し（表示用）。CSV の列名は ASCII のまま（SPEC §5.5）
RESULT_HEADINGS = {
    "impedance_ohm": N_("Resistance R [Ω]"),
    "voltage_V": N_("Voltage between electrodes [V]"),
    "power_W": N_("Power [W]"),
    "j_max_A_per_m2": N_("|J| max [A/m²]"),
    "e_max_V_per_m": N_("|E| max [V/m]"),
    "contact_j_max_A_per_m2": N_("Current density into the skin max [A/m²]"),
    "phi_min_V": N_("φ min [V]"),
    "phi_max_V": N_("φ max [V]"),
    "iterations": N_("Iterations"),
    "residual": N_("Relative residual"),
    "backend": N_("Backend"),
    "preconditioner": N_("Preconditioner"),
    "energy_relative_error": N_("Energy balance relative error"),
    "conservation_warnings": N_("Conservation warnings"),
}


def parse_values(text: str) -> list[float]:
    """`0.02, 0.04, 0.06` や `1e-3 2e-3` を数値の並びにする。

    `0.02:0.06:0.01`（始め:終わり:刻み）も受け付ける。
    """
    body = text.replace(",", " ").strip()
    if not body:
        raise ValueError(tr("No values entered"))
    if ":" in body:
        parts = body.split(":")
        if len(parts) != 3:
            raise ValueError(tr("Write a range as start:stop:step"))
        start, stop, step = (_number(p) for p in parts)
        if step == 0.0:
            raise ValueError(tr("The step is 0"))
        count = int(round((stop - start) / step)) + 1
        if count <= 0 or count > 1000:
            raise ValueError(tr("That makes {count} cases (use 1 to 1000)", count=count))
        return [start + i * step for i in range(count)]
    return [_number(p) for p in body.split()]


#: 値を読めなかったときの対処（どの書き方を受け付けるか）
VALUES_HINT = N_(
    "Separate values with commas or spaces (0.02, 0.04), or write start:stop:step (0.02:0.06:0.01)"
)


def _number(text: str) -> float:
    """数値 1 つを読む。読めなければ、どの値が読めないかと書き方を添えて `ValueError`。

    以前は `float()` の例外をそのまま出していて、ダイアログに
    `could not convert string to float: 'abc'` と英語の内部の言葉が出ていた。
    """
    try:
        value = float(text)
    except ValueError:
        raise ValueError(
            tr("Cannot read \"{text}\" as a number.", text=text.strip()) + "\n" + tr(VALUES_HINT)
        ) from None
    if not math.isfinite(value):
        raise ValueError(
            tr("\"{text}\" is not a finite number.", text=text.strip()) + "\n" + tr(VALUES_HINT)
        )
    return value


class SweepThread(QThread):
    """`facetsetfem.sweep` を別スレッドで走らせる。"""

    #: (終わったケース数, 全ケース数)
    progressed = Signal(int, int)
    #: 成功したときの `SweepTable`
    succeeded = Signal(object)
    #: 失敗したときのメッセージ
    failed = Signal(str)

    def __init__(self, study: Study, parameters: dict, mesh, parent=None) -> None:
        super().__init__(parent)
        self._study = study
        self._parameters = parameters
        self._mesh = mesh

    def run(self) -> None:  # noqa: D102 - QThread の規約
        try:
            table = sweep(
                self._study,
                self._parameters,
                mesh=self._mesh,
                keep_going=True,
                progress=lambda i, n, _values: self.progressed.emit(i, n),
            )
        except Exception as exc:  # noqa: BLE001 - GUI に出すため全部拾う
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(table)


class SweepDialog(QDialog):
    """パラメータを振って一括で解き、結果を表で比べる。"""

    def __init__(self, window) -> None:
        super().__init__(window)
        self.window = window
        self.table: SweepTable | None = None
        self._thread: SweepThread | None = None

        self.setWindowTitle(tr("Parameter study"))
        self.resize(900, 600)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.parameter = QComboBox()
        study = window.session.study
        for label, key in parameter_choices(study):
            self.parameter.addItem(label, key)
        self.values = QLineEdit()
        self.parameter.currentIndexChanged.connect(self._show_example)
        self._show_example()
        form.addRow(tr("Parameter"), self.parameter)
        form.addRow(tr("Values"), self.values)
        layout.addLayout(form)
        layout.addWidget(
            QLabel(
                tr(
                    "The original case is not modified. The mesh is read once and reused for every case.\n"
                    "Cases that fail are skipped; the reason is shown below."
                )
            )
        )

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.result_table = QTableWidget()
        self.result_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.result_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        layout.addWidget(self.result_table, 1)

        self.message = QLabel("")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)

        buttons = QHBoxLayout()
        self.run_button = QPushButton(tr("Run"))
        self.run_button.clicked.connect(self.start)
        self.save_button = QPushButton(tr("Save as CSV..."))
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_csv)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.save_button)
        buttons.addStretch(1)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        # 標準のボタンは Qt の翻訳が無いと英語（Close）になる
        box.button(QDialogButtonBox.StandardButton.Close).setText(tr("Close"))
        box.rejected.connect(self.reject)
        buttons.addWidget(box)
        layout.addLayout(buttons)

    def _show_example(self, *_args) -> None:
        """入力欄の例を、振るものに合わせて出す（表示の単位で）。"""
        key = self.parameter.currentData() or ""
        if key.startswith("sigma."):
            head = "sigma"
        elif key.endswith("contact_impedance"):
            head = "contact_impedance"
        elif display_scale(key) != 1.0:
            head = "current"
        else:
            head = key
        self.values.setPlaceholderText(
            tr(VALUE_EXAMPLES.get(head, N_("0.02, 0.04  or  0.02:0.06:0.01")))
        )

    # --- 実行 ---------------------------------------------------------------

    def start(self) -> None:
        session = self.window.session
        if session.study is None or session.mesh is None:
            return
        try:
            values = parse_values(self.values.text())
        except ValueError as exc:
            QMessageBox.warning(self, tr("Cannot read the values"), str(exc))
            return

        key = self.parameter.currentData()
        values = [v * display_scale(key) for v in values]  # 表示の単位 → SI
        self.run_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(values))
        self.progress.setValue(0)
        self.message.setText(tr("Running {n} cases...", n=len(values)))

        self._thread = SweepThread(session.study, {key: values}, session.mesh, self)
        self._thread.progressed.connect(self._on_progress)
        self._thread.succeeded.connect(self._on_done)
        self._thread.failed.connect(self._on_failed)
        self._thread.finished.connect(lambda: self.run_button.setEnabled(True))
        self._thread.start()

    def _on_progress(self, index: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(index)

    def _on_failed(self, message: str) -> None:
        self.progress.setVisible(False)
        self.message.setText(message)
        self.window.log(tr("The parameter study failed: {message}", message=message))

    def _on_done(self, table: SweepTable) -> None:
        self.progress.setVisible(False)
        self.table = table
        self._fill(table)
        self.save_button.setEnabled(bool(table.rows))
        note = tr("{n} cases finished.", n=len(table))
        if table.failures:
            first = table.failures[0][1].splitlines()[0]
            note += tr(" {n} cases could not be solved: {first}", n=len(table.failures), first=first)
        self.message.setText(note)
        self.window.log(note)

    def _fill(self, table: SweepTable) -> None:
        """結果を表に並べる。見出しは日本語と単位、振った値は入力と同じ単位で出す。

        以前は CSV の列名（`impedance_ohm` など）をそのまま見出しにしていた。
        """
        columns = table.columns()
        labels = {key: label for label, key in parameter_choices(self.window.session.study)}
        self.result_table.setColumnCount(len(columns))
        self.result_table.setHorizontalHeaderLabels(
            [labels.get(k) or tr(RESULT_HEADINGS.get(k, k)) for k in columns]
        )
        self.result_table.setRowCount(len(table.rows))
        for r, row in enumerate(table.rows):
            for c, key in enumerate(columns):
                value = row.get(key, "")
                if key in table.parameters and isinstance(value, float):
                    value /= display_scale(key)  # SI → 表示の単位
                text = f"{value:.6g}" if isinstance(value, float) else str(value)
                item = QTableWidgetItem(text)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self.result_table.setItem(r, c, item)
        self.result_table.resizeColumnsToContents()

    def save_csv(self) -> None:
        if self.table is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, tr("Save the parameter study results"), "sweep.csv", "CSV (*.csv)"
        )
        if not path:
            return
        self.table.write_csv(Path(path))
        self.window.log(tr("Wrote the parameter study results: {path}", path=path))
