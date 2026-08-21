"""パラメータスタディのダイアログ（SPEC §6.2 / §8 M7）。

振るパラメータと値を決めて一括で解き、結果を表で見比べる。
計算そのものは [`gfem.sweep`](gfem.sweep) がやる——**GUI にロジックは置かない**
（SPEC §7）ので、ここは入力を集めて結果を並べるだけ。

長くかかるのでワーカースレッドで走らせ、途中経過を進捗バーに出す。
"""

from __future__ import annotations

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

from ..study import Study
from ..sweep import SweepTable, sweep

__all__ = ["SweepDialog", "parameter_choices"]


def parameter_choices(study: Study) -> list[tuple[str, str]]:
    """このケースで振れるパラメータの `(表示名, 指定子)` 一覧。"""
    out: list[tuple[str, str]] = [("刺激電流（全電極） [A]", "current")]
    for m in study.materials:
        label = f"σ: PID {m.pid}" + (f"（{m.name}）" if m.name else "")
        out.append((f"{label} [S/m]", f"sigma.{m.pid}"))
    for e in study.electrodes:
        out.append((f"電極 {e.name} の電流 [A]", f"electrode.{e.name}.current"))
    out.append(("収束判定（相対残差）", "solver.tolerance"))
    return out


def parse_values(text: str) -> list[float]:
    """`0.02, 0.04, 0.06` や `1e-3 2e-3` を数値の並びにする。

    `0.02:0.06:0.01`（始め:終わり:刻み）も受け付ける。
    """
    body = text.replace(",", " ").strip()
    if not body:
        raise ValueError("値が入力されていません")
    if ":" in body:
        parts = body.split(":")
        if len(parts) != 3:
            raise ValueError("範囲は 始め:終わり:刻み の形で書いてください")
        start, stop, step = (float(p) for p in parts)
        if step == 0.0:
            raise ValueError("刻みが 0 です")
        count = int(round((stop - start) / step)) + 1
        if count <= 0 or count > 1000:
            raise ValueError(f"ケース数が {count} 個になります（1〜1000 にしてください）")
        return [start + i * step for i in range(count)]
    return [float(p) for p in body.split()]


class SweepThread(QThread):
    """`gfem.sweep` を別スレッドで走らせる。"""

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

        self.setWindowTitle("パラメータスタディ")
        self.resize(900, 600)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.parameter = QComboBox()
        study = window.session.study
        for label, key in parameter_choices(study):
            self.parameter.addItem(label, key)
        self.values = QLineEdit()
        self.values.setPlaceholderText("0.02, 0.04, 0.06  または  0.02:0.06:0.01")
        form.addRow("振るもの", self.parameter)
        form.addRow("値", self.values)
        layout.addLayout(form)
        layout.addWidget(
            QLabel(
                "元のケースは書き換えません。メッシュは 1 度だけ読んで全ケースで使い回します。\n"
                "解けなかったケースは飛ばして続け、下に理由を出します。"
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
        self.run_button = QPushButton("実行")
        self.run_button.clicked.connect(self.start)
        self.save_button = QPushButton("CSV に保存...")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_csv)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.save_button)
        buttons.addStretch(1)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        buttons.addWidget(box)
        layout.addLayout(buttons)

    # --- 実行 ---------------------------------------------------------------

    def start(self) -> None:
        session = self.window.session
        if session.study is None or session.mesh is None:
            return
        try:
            values = parse_values(self.values.text())
        except ValueError as exc:
            QMessageBox.warning(self, "値を読めません", str(exc))
            return

        key = self.parameter.currentData()
        self.run_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(values))
        self.progress.setValue(0)
        self.message.setText(f"{len(values)} ケースを実行中...")

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
        self.window.log(f"パラメータスタディに失敗しました: {message}")

    def _on_done(self, table: SweepTable) -> None:
        self.progress.setVisible(False)
        self.table = table
        self._fill(table)
        self.save_button.setEnabled(bool(table.rows))
        note = f"{len(table)} ケースが完了しました。"
        if table.failures:
            first = table.failures[0][1].splitlines()[0]
            note += f" 解けなかったケースが {len(table.failures)} 件あります: {first}"
        self.message.setText(note)
        self.window.log(note)

    def _fill(self, table: SweepTable) -> None:
        columns = table.columns()
        self.result_table.setColumnCount(len(columns))
        self.result_table.setHorizontalHeaderLabels(columns)
        self.result_table.setRowCount(len(table.rows))
        for r, row in enumerate(table.rows):
            for c, key in enumerate(columns):
                value = row.get(key, "")
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
            self, "パラメータスタディの結果を保存", "sweep.csv", "CSV (*.csv)"
        )
        if not path:
            return
        self.table.write_csv(Path(path))
        self.window.log(f"パラメータスタディの結果を書き出しました: {path}")
