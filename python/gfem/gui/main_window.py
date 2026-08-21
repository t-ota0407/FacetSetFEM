"""メインウィンドウ（SPEC §6.1 / §6.2）。

```
┌──────────────────────────────────────────────────────────┐
│ メニュー / ツールバー                                      │
├──────────────┬───────────────────────────┬───────────────┤
│ モデルツリー  │      3D ビュー (PyVista)   │ プロパティ     │
├──────────────┴───────────────────────────┴───────────────┤
│ ログ / 収束曲線 / 保存則チェック結果                        │
└──────────────────────────────────────────────────────────┘
```

**GUI にロジックは置かない**（SPEC §7）。状態は [`Session`] が持ち、
ここはその表示と編集だけを担当する。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import GfemError
from .solver_thread import SolverThread
from .state import (
    DISPLAY_FIELDS,
    NODAL_VALUE_WARNING,
    REGION_STATS_COLUMNS,
    TISSUE_HINTS,
    Session,
    field_by_key,
)
from .widgets import ConvergencePlot, UnitSelector

__all__ = ["MainWindow"]

#: ツリーの項目種別
ROLE_KIND = Qt.ItemDataRole.UserRole
ROLE_KEY = Qt.ItemDataRole.UserRole + 1

COLORMAPS = ["viridis", "plasma", "coolwarm", "turbo", "gray"]

#: 3D ビューの表示モード（表示名, 内部名）
VIEW_MODES = [
    ("全体", "volume"),
    ("断面スライス", "slice"),
    ("クリップ", "clip"),
    ("等値面", "isosurface"),
]


class MainWindow(QMainWindow):
    """GenericFEM の GUI。"""

    def __init__(self, session: Session | None = None) -> None:
        super().__init__()
        self.session = session or Session()
        self.solver: SolverThread | None = None
        self._actor = None

        self.setWindowTitle("GenericFEM")
        self.resize(1400, 900)
        self._build_ui()
        self._build_menus()
        self.refresh_all()

    # --- UI の構築 ---------------------------------------------------------

    def _build_ui(self) -> None:
        from pyvistaqt import QtInteractor

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["モデル"])
        self.tree.setMinimumWidth(240)
        self.tree.currentItemChanged.connect(self._on_tree_selection)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("white")

        self.pages = QStackedWidget()
        self.page_mesh = MeshPage(self)
        self.page_region = RegionPage(self)
        self.page_electrode = ElectrodePage(self)
        self.page_solver = SolverPage(self)
        self.page_result = ResultPage(self)
        for page in (
            self.page_mesh,
            self.page_region,
            self.page_electrode,
            self.page_solver,
            self.page_result,
        ):
            self.pages.addWidget(page)
        self.pages.setMinimumWidth(320)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.tree)
        splitter.addWidget(self.plotter.interactor)
        splitter.addWidget(self.pages)
        splitter.setStretchFactor(1, 1)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.convergence = ConvergencePlot()
        self.checks_view = QPlainTextEdit()
        self.checks_view.setReadOnly(True)

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(len(REGION_STATS_COLUMNS))
        self.stats_table.setHorizontalHeaderLabels(REGION_STATS_COLUMNS)
        self.stats_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self.bottom = QTabWidget()
        self.bottom.addTab(self.log_view, "ログ")
        self.bottom.addTab(self.convergence, "収束曲線")
        self.bottom.addTab(self.checks_view, "保存則チェック")
        self.bottom.addTab(self.stats_table, "領域統計")
        self.bottom.setMaximumHeight(240)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(splitter, stretch=1)
        layout.addWidget(self.bottom)
        self.setCentralWidget(central)
        self.statusBar().showMessage("モデルを開いてください")

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("ファイル(&F)")
        self._add_action(file_menu, "モデルを開く...", self.open_mesh_dialog, "Ctrl+O")
        self._add_action(file_menu, "ケースを開く...", self.open_case_dialog)
        file_menu.addSeparator()
        self._add_action(file_menu, "ケースを保存...", self.save_case_dialog, "Ctrl+S")
        self._add_action(file_menu, "結果を VTU に書き出す...", self.export_vtu_dialog)
        self._add_action(
            file_menu, "領域統計を CSV に書き出す...", self.export_region_stats_dialog
        )
        self._add_action(
            file_menu, "収束履歴を CSV に書き出す...", self.export_convergence_dialog
        )
        self._add_action(
            file_menu, "実行サマリを JSON に書き出す...", self.export_summary_dialog
        )
        self._add_action(file_menu, "スクリーンショット...", self.screenshot_dialog)
        file_menu.addSeparator()
        self._add_action(file_menu, "終了", self.close)

        run_menu = self.menuBar().addMenu("実行(&R)")
        self._add_action(run_menu, "整合性チェック", self.check_consistency, "Ctrl+K")
        self._add_action(run_menu, "solve", self.start_solve, "F5")
        self.action_cancel = self._add_action(run_menu, "中断", self.cancel_solve)
        self.action_cancel.setEnabled(False)

        electrode_menu = self.menuBar().addMenu("電極(&E)")
        self._add_action(electrode_menu, "電極を追加", self.add_electrode)
        self._add_action(electrode_menu, "選択中の電極を削除", self.remove_selected_electrode)

    def _add_action(self, menu, text, slot, shortcut: str | None = None) -> QAction:
        action = QAction(text, self)
        action.triggered.connect(slot)
        if shortcut:
            action.setShortcut(shortcut)
        menu.addAction(action)
        return action

    # --- ログ ---------------------------------------------------------------

    def log(self, message: str) -> None:
        self.log_view.appendPlainText(message)
        self.statusBar().showMessage(message.splitlines()[0][:200])

    def show_error(self, exc: Exception) -> None:
        text = str(exc)
        self.log(f"エラー:\n{text}")
        QMessageBox.critical(self, "エラー", text)

    # --- ファイル操作 -------------------------------------------------------

    def open_mesh_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "モデルを開く", "", "Nastran バルクデータ (*.bdf *.nas);;すべて (*)"
        )
        if not path:
            return
        try:
            # SPEC §3.2: 単位の選択前に寸法を見せたいので、一度そのままの値で読む
            probe = self.session.__class__()
            probe.open_mesh(path, "m")
            box = probe.mesh.bounding_box()
            extent = tuple(hi - lo for lo, hi in zip(*box))
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc)
            return

        dialog = UnitSelector(extent, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        self.open_mesh(path, dialog.length_unit)

    def open_mesh(self, path: str | Path, length_unit: str) -> None:
        """メッシュを読み込む（テストからも呼べるようダイアログを挟まない）。"""
        try:
            self.session.open_mesh(path, length_unit)
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc)
            return
        self.convergence.clear()
        self.checks_view.clear()
        self.log(f"読み込み完了: {path}")
        self.log(self.session.mesh_summary())
        self.refresh_all()

    def open_case_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "ケースを開く", "", "ケースファイル (*.yaml *.yml);;すべて (*)"
        )
        if path:
            self.open_case(path)

    def open_case(self, path: str | Path) -> None:
        try:
            self.session.open_case(path)
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc)
            return
        self.convergence.clear()
        self.checks_view.clear()
        self.log(f"ケースを読み込みました: {path}")
        self.log(self.session.mesh_summary())
        self.refresh_all()

    def save_case_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "ケースを保存", "case.yaml", "ケースファイル (*.yaml)"
        )
        if path:
            self.save_case(path)

    def save_case(self, path: str | Path) -> None:
        try:
            self.session.save_case(path)
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc)
            return
        self.log(f"ケースを保存しました: {path}")

    def export_vtu_dialog(self) -> None:
        if self.session.solution is None:
            QMessageBox.information(self, "VTU 書き出し", "先に solve を実行してください。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "VTU に書き出す", "result.vtu", "VTU (*.vtu)")
        if path:
            self.export_vtu(path)

    def export_vtu(self, path: str | Path) -> None:
        try:
            self.session.solution.write_vtu(path)
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc)
            return
        self.log(f"VTU を書き出しました: {path}")

    def export_region_stats_dialog(self) -> None:
        self._export_dialog(
            "領域統計を書き出す",
            "region_stats.csv",
            "CSV (*.csv)",
            self.session.write_region_stats_csv,
        )

    def export_convergence_dialog(self) -> None:
        self._export_dialog(
            "収束履歴を書き出す",
            "convergence.csv",
            "CSV (*.csv)",
            self.session.write_convergence_csv,
        )

    def export_summary_dialog(self) -> None:
        self._export_dialog(
            "実行サマリを書き出す",
            "summary.json",
            "JSON (*.json)",
            self.session.write_summary_json,
        )

    def _export_dialog(self, title: str, default: str, filters: str, writer) -> None:
        if self.session.solution is None:
            QMessageBox.information(self, title, "先に solve を実行してください。")
            return
        path, _ = QFileDialog.getSaveFileName(self, title, default, filters)
        if not path:
            return
        try:
            writer(path)
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc)
            return
        self.log(f"書き出しました: {path}")

    def screenshot_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "スクリーンショット", "view.png", "PNG (*.png)")
        if path:
            self.screenshot(path)

    def screenshot(self, path: str | Path) -> None:
        self.plotter.screenshot(str(path))
        self.log(f"スクリーンショットを保存しました: {path}")

    # --- 電極 ---------------------------------------------------------------

    def add_electrode(self) -> None:
        if self.session.study is None:
            return
        existing = {e.name for e in self.session.study.electrodes}
        index = len(existing) + 1
        while f"E{index}" in existing:
            index += 1
        role = "anode" if not existing else "cathode"
        self.session.add_electrode(f"E{index}", role=role)
        self.refresh_tree()
        self.log(f"電極 E{index} を追加しました（{role}）")

    def remove_selected_electrode(self) -> None:
        item = self.tree.currentItem()
        if item is None or item.data(0, ROLE_KIND) != "electrode":
            return
        name = item.data(0, ROLE_KEY)
        self.session.remove_electrode(name)
        self.refresh_tree()
        self.log(f"電極 {name} を削除しました")

    # --- 実行 ---------------------------------------------------------------

    def check_consistency(self) -> list[str] | None:
        """整合性チェック（SPEC §6.2 の手順 4）。"""
        try:
            warnings = self.session.validate()
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc)
            return None
        if warnings:
            self.log("整合性チェック: 警告 " + str(len(warnings)) + " 件")
            for w in warnings:
                self.log(f"  [警告] {w}")
        else:
            self.log("整合性チェック: 問題ありません")
        for info in self.session.electrode_info():
            self.log(
                f"  電極 {info['name']} ({info['role']}): "
                f"{info['n_faces']} 面, 面積 {info['area'] * 1e6:.1f} mm², "
                f"電流密度 {info['current_density']:.4g} A/m²"
            )
        return warnings

    def start_solve(self) -> None:
        if self.solver is not None and self.solver.isRunning():
            return
        try:
            case = self.session.build_case()
            case.validate()
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc)
            return

        self.convergence.clear()
        self.convergence.set_tolerance(self.session.study.solver.tolerance)
        self.bottom.setCurrentWidget(self.convergence)
        self.log("solve を開始しました")
        self.action_cancel.setEnabled(True)

        self.solver = SolverThread(case, self)
        self.solver.progressed.connect(self._on_progress)
        self.solver.succeeded.connect(self._on_solved)
        self.solver.failed.connect(self._on_solve_failed)
        self.solver.finished.connect(lambda: self.action_cancel.setEnabled(False))
        self.solver.start()

    def cancel_solve(self) -> None:
        if self.solver is not None:
            self.solver.cancel()
            self.log("中断を要求しました")

    def _on_progress(self, iteration: int, residual: float) -> None:
        self.statusBar().showMessage(f"反復 {iteration}　相対残差 {residual:.3e}")
        self.convergence.append(residual)

    def _on_solved(self, solution) -> None:
        self.session.set_solution(solution)
        self.convergence.set_history(solution.history)
        self.checks_view.setPlainText(self.session.result_summary())
        self.refresh_stats_table()
        self.bottom.setCurrentWidget(self.checks_view)
        self.log(
            f"solve 完了: {solution.iterations} 反復, "
            f"相対残差 {solution.residual:.3e}, R = {solution.impedance:.4g} Ω"
        )
        for w in solution.warnings:
            self.log(f"  [警告] {w}")
        self.refresh_tree()
        self.page_result.reload()
        self.refresh_view()

    def _on_solve_failed(self, message: str) -> None:
        self.log(f"solve 失敗:\n{message}")
        QMessageBox.warning(self, "solve", message)

    # --- 表示 ---------------------------------------------------------------

    def refresh_all(self) -> None:
        self.refresh_tree()
        self.page_result.reload()
        self.refresh_stats_table()
        self.refresh_view()

    def refresh_stats_table(self) -> None:
        """領域統計の表を作り直す（SPEC §5.2）。"""
        try:
            rows = self.session.region_stats_rows()
        except Exception:  # noqa: BLE001 - まだ解いていない場合など
            rows = []
        self.stats_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                self.stats_table.setItem(r, c, QTableWidgetItem(value))
        self.stats_table.resizeColumnsToContents()

    def refresh_tree(self) -> None:
        self.tree.clear()
        study = self.session.study
        if study is None:
            return

        mesh_item = QTreeWidgetItem([f"Mesh（単位: {study.length_unit}）"])
        mesh_item.setData(0, ROLE_KIND, "mesh")
        self.tree.addTopLevelItem(mesh_item)

        regions = QTreeWidgetItem(["領域 (PID)"])
        regions.setData(0, ROLE_KIND, "regions")
        for m in study.materials:
            label = f"{m.pid}"
            if m.name:
                label += f" {m.name}"
            label += f"　σ = {m.sigma:g} S/m" if m.sigma > 0 else "　σ 未設定"
            child = QTreeWidgetItem([label])
            child.setData(0, ROLE_KIND, "region")
            child.setData(0, ROLE_KEY, m.pid)
            regions.addChild(child)
        self.tree.addTopLevelItem(regions)
        regions.setExpanded(True)

        electrodes = QTreeWidgetItem(["電極"])
        electrodes.setData(0, ROLE_KIND, "electrodes")
        for e in study.electrodes:
            child = QTreeWidgetItem([f"{e.name}　{e.role}　{e.current * 1e3:g} mA"])
            child.setData(0, ROLE_KIND, "electrode")
            child.setData(0, ROLE_KEY, e.name)
            electrodes.addChild(child)
        self.tree.addTopLevelItem(electrodes)
        electrodes.setExpanded(True)

        solver_item = QTreeWidgetItem(["ソルバ設定"])
        solver_item.setData(0, ROLE_KIND, "solver")
        self.tree.addTopLevelItem(solver_item)

        result_item = QTreeWidgetItem(["結果"])
        result_item.setData(0, ROLE_KIND, "result")
        self.tree.addTopLevelItem(result_item)

    def _on_tree_selection(self, current, previous) -> None:
        del previous
        if current is None:
            return
        kind = current.data(0, ROLE_KIND)
        key = current.data(0, ROLE_KEY)
        if kind == "region":
            self.page_region.bind(key)
            self.pages.setCurrentWidget(self.page_region)
            self.refresh_view(highlight_pid=key)
            return
        if kind == "electrode":
            self.page_electrode.bind(key)
            self.pages.setCurrentWidget(self.page_electrode)
            self.refresh_view(highlight_electrode=key)
            return
        if kind == "solver":
            self.page_solver.reload()
            self.pages.setCurrentWidget(self.page_solver)
        elif kind == "result":
            self.page_result.reload()
            self.pages.setCurrentWidget(self.page_result)
        else:
            self.page_mesh.reload()
            self.pages.setCurrentWidget(self.page_mesh)
        self.refresh_view()

    def refresh_view(
        self, highlight_pid: int | None = None, highlight_electrode: str | None = None
    ) -> None:
        """3D ビューを描き直す。"""
        if self.session.mesh is None:
            return
        self.plotter.clear()
        options = self.page_result.options()
        grid = self.session.grid()

        target = grid
        scalars = options["field"]
        mode = options["mode"]
        if mode == "slice":
            target = grid.slice(
                normal=options["slice_normal"], origin=self._slice_origin(grid, options)
            )
        elif mode == "clip":
            target = grid.clip(
                normal=options["slice_normal"],
                origin=self._slice_origin(grid, options),
                invert=True,
            )
        elif mode == "isosurface":
            field = field_by_key(scalars)
            quantity = field.iso_quantity if field else None
            if quantity is None:
                self.log("この量では等値面を作れません（領域 PID など）。全体表示に戻します")
            else:
                lo, hi = self.session.value_range(quantity)
                level = lo + (hi - lo) * options["iso_level"]
                target = self.session.iso_surface(quantity, level)
                scalars = quantity
                self.statusBar().showMessage(f"等値面 {quantity} = {level:.4g}")
        if target.n_cells == 0:
            target = grid
            scalars = options["field"]

        if scalars not in target.array_names:
            scalars = "PID" if "PID" in target.array_names else None

        self.plotter.add_mesh(
            target,
            scalars=scalars,
            cmap=options["cmap"],
            show_edges=options["edges"],
            scalar_bar_args={"title": scalars or ""} if scalars else None,
            categories=scalars == "PID",
        )

        if options["streamlines"] and self.session.solution is not None:
            try:
                seeds = self.session.default_seed_points(options["n_seeds"])
                lines = self.session.streamlines(seeds)
                if lines.n_cells:
                    self.plotter.add_mesh(lines, color="black", line_width=2)
            except Exception as exc:  # noqa: BLE001
                self.log(f"ストリームラインを引けませんでした: {exc}")

        if highlight_pid is not None:
            region = self.session.region_grid(highlight_pid)
            if region.n_cells:
                self.plotter.add_mesh(
                    region.extract_surface(), color="orange", opacity=0.85, show_edges=False
                )
        if highlight_electrode is not None:
            faces = self.session.electrode_polydata(highlight_electrode)
            if faces.n_cells:
                self.plotter.add_mesh(faces, color="red", opacity=1.0)

        self.plotter.add_axes()
        self.plotter.reset_camera()
        self.plotter.render()

    @staticmethod
    def _slice_origin(grid, options):
        bounds = grid.bounds
        axis = {"x": 0, "y": 1, "z": 2}[options["slice_normal"]]
        lo = bounds[2 * axis]
        hi = bounds[2 * axis + 1]
        centre = [
            0.5 * (bounds[0] + bounds[1]),
            0.5 * (bounds[2] + bounds[3]),
            0.5 * (bounds[4] + bounds[5]),
        ]
        centre[axis] = lo + (hi - lo) * options["slice_position"]
        return tuple(centre)


# --- プロパティパネルの各ページ ---------------------------------------------


class MeshPage(QWidget):
    """メッシュの概要（SPEC §6.2 の手順 1 の読み込みレポート）。"""

    def __init__(self, window: MainWindow) -> None:
        super().__init__()
        self.window = window
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>メッシュ</b>"))
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text)

    def reload(self) -> None:
        if self.window.session.mesh is None:
            self.text.setPlainText("（モデルが開かれていません）")
        else:
            self.text.setPlainText(self.window.session.mesh_summary())


class RegionPage(QWidget):
    """領域（PID）への σ 割り当て（SPEC §6.2 の手順 2）。"""

    def __init__(self, window: MainWindow) -> None:
        super().__init__()
        self.window = window
        self.pid: int | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>領域</b>"))
        form = QFormLayout()
        self.pid_label = QLabel("-")
        self.name_edit = QLineEdit()
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setDecimals(6)
        self.sigma_spin.setRange(0.0, 1000.0)
        self.sigma_spin.setSingleStep(0.01)
        self.sigma_spin.setSuffix(" S/m")
        form.addRow("PID", self.pid_label)
        form.addRow("名前", self.name_edit)
        form.addRow("導電率 σ", self.sigma_spin)
        layout.addLayout(form)

        hints = QGroupBox("組織物性の目安 [S/m]")
        hints_layout = QVBoxLayout(hints)
        for name, value in TISSUE_HINTS.items():
            button = QPushButton(f"{name}: {value}")
            button.clicked.connect(
                lambda _checked=False, n=name, v=value: self._apply_hint(n, v)
            )
            hints_layout.addWidget(button)
        layout.addWidget(hints)
        layout.addStretch(1)

        self.name_edit.editingFinished.connect(self._commit)
        self.sigma_spin.valueChanged.connect(self._commit)

    def bind(self, pid: int) -> None:
        self.pid = None  # コミットを抑えてから値を入れる
        session = self.window.session
        self.pid_label.setText(str(pid))
        self.name_edit.setText(session.region_name(pid))
        self.sigma_spin.setValue(session.conductivity_of(pid))
        self.pid = pid

    def _apply_hint(self, name: str, value: float) -> None:
        if self.pid is None:
            return
        self.name_edit.setText(name)
        self.sigma_spin.setValue(value)
        self._commit()

    def _commit(self) -> None:
        if self.pid is None:
            return
        self.window.session.set_conductivity(
            self.pid, self.sigma_spin.value(), self.name_edit.text()
        )
        self.window.refresh_tree()


class ElectrodePage(QWidget):
    """電極の定義（SPEC §6.2 の手順 3）。

    面を選ぶたびに**合計面積と平均電流密度**を出す。
    """

    def __init__(self, window: MainWindow) -> None:
        super().__init__()
        self.window = window
        self.name: str | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>電極</b>"))
        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.role_combo = QComboBox()
        self.role_combo.addItems(["anode", "cathode"])
        self.current_spin = QDoubleSpinBox()
        self.current_spin.setDecimals(4)
        self.current_spin.setRange(0.0, 1.0e6)
        self.current_spin.setSuffix(" mA")
        form.addRow("名前", self.name_edit)
        form.addRow("役割", self.role_combo)
        form.addRow("総電流", self.current_spin)
        layout.addLayout(form)

        layout.addWidget(QLabel("電極面（シェル PID）"))
        self.pid_list = QListWidget()
        layout.addWidget(self.pid_list)

        self.summary = QLabel("-")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.summary)

        self.name_edit.editingFinished.connect(self._commit)
        self.role_combo.currentTextChanged.connect(self._commit)
        self.current_spin.valueChanged.connect(self._commit)
        self.pid_list.itemChanged.connect(self._commit)

    def bind(self, name: str) -> None:
        session = self.window.session
        spec = session.electrode(name)
        if spec is None:
            return
        self.name = None
        self.name_edit.setText(spec.name)
        self.role_combo.setCurrentText(spec.role)
        self.current_spin.setValue(spec.current * 1e3)
        self.pid_list.clear()
        for pid in session.mesh.shell_pids():
            item = QListWidgetItem(f"PID {pid}")
            item.setData(Qt.ItemDataRole.UserRole, pid)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if pid in spec.shell_pids else Qt.CheckState.Unchecked
            )
            self.pid_list.addItem(item)
        self.name = name
        self._update_summary()

    def _commit(self) -> None:
        if self.name is None:
            return
        session = self.window.session
        spec = session.electrode(self.name)
        if spec is None:
            return
        spec.name = self.name_edit.text() or spec.name
        spec.role = self.role_combo.currentText()
        spec.current = self.current_spin.value() * 1e-3
        spec.shell_pids = [
            self.pid_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.pid_list.count())
            if self.pid_list.item(i).checkState() == Qt.CheckState.Checked
        ]
        self.name = spec.name
        self._update_summary()
        self.window.refresh_tree()
        self.window.refresh_view(highlight_electrode=spec.name)

    def _update_summary(self) -> None:
        session = self.window.session
        spec = session.electrode(self.name) if self.name else None
        if spec is None or not (spec.shell_pids or spec.element_ids):
            self.summary.setText("面が選択されていません")
            return
        try:
            info = {e["name"]: e for e in session.electrode_info()}[spec.name]
        except (GfemError, KeyError) as exc:
            self.summary.setText(str(exc).splitlines()[0])
            return
        self.summary.setText(
            f"{info['n_faces']} 面　合計面積 {info['area'] * 1e6:.1f} mm²\n"
            f"平均電流密度 {info['current_density']:.4g} A/m²"
        )


class SolverPage(QWidget):
    """ソルバ設定（SPEC §4.3）。"""

    def __init__(self, window: MainWindow) -> None:
        super().__init__()
        self.window = window
        self._loading = False

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>ソルバ設定</b>"))
        form = QFormLayout()
        self.tolerance = QLineEdit()
        self.max_iterations = QSpinBox()
        self.max_iterations.setRange(1, 10_000_000)
        self.preconditioner = QComboBox()
        self.preconditioner.addItems(["jacobi", "none"])
        self.reference = QComboBox()
        self.reference.addItems(["zero_mean", "cathode_mean"])
        form.addRow("相対残差の収束判定", self.tolerance)
        form.addRow("最大反復回数", self.max_iterations)
        form.addRow("前処理", self.preconditioner)
        form.addRow("基準電位", self.reference)
        layout.addLayout(form)
        layout.addWidget(
            QLabel(
                "ゲージ固定は射影（平均ゼロ）で行います。\n"
                "1 節点ピン留めは解を歪めるため採用しません（SPEC §2.2）。"
            )
        )
        layout.addStretch(1)

        self.tolerance.editingFinished.connect(self._commit)
        self.max_iterations.valueChanged.connect(self._commit)
        self.preconditioner.currentTextChanged.connect(self._commit)
        self.reference.currentTextChanged.connect(self._commit)

    def reload(self) -> None:
        study = self.window.session.study
        if study is None:
            return
        self._loading = True
        self.tolerance.setText(f"{study.solver.tolerance:g}")
        self.max_iterations.setValue(study.solver.max_iterations)
        self.preconditioner.setCurrentText(study.solver.preconditioner)
        self.reference.setCurrentText(study.output.reference_potential)
        self._loading = False

    def _commit(self) -> None:
        study = self.window.session.study
        if study is None or self._loading:
            return
        try:
            study.solver.tolerance = float(self.tolerance.text())
        except ValueError:
            self.tolerance.setText(f"{study.solver.tolerance:g}")
        study.solver.max_iterations = self.max_iterations.value()
        study.solver.preconditioner = self.preconditioner.currentText()
        study.output.reference_potential = self.reference.currentText()


class ResultPage(QWidget):
    """結果の表示設定（SPEC §6.2 の手順 6）。"""

    def __init__(self, window: MainWindow) -> None:
        super().__init__()
        self.window = window
        self._loading = False

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>表示</b>"))
        form = QFormLayout()
        self.field = QComboBox()
        self.mode = QComboBox()
        for label, key in VIEW_MODES:
            self.mode.addItem(label, key)
        self.cmap = QComboBox()
        self.cmap.addItems(COLORMAPS)
        form.addRow("表示量", self.field)
        form.addRow("表示モード", self.mode)
        form.addRow("カラーマップ", self.cmap)
        layout.addLayout(form)

        self.edges = QCheckBox("要素の稜線を表示")
        layout.addWidget(self.edges)

        slice_box = QGroupBox("断面 / クリップ")
        slice_layout = QVBoxLayout(slice_box)
        self.slice_normal = QComboBox()
        self.slice_normal.addItems(["x", "y", "z"])
        self.slice_position = QSlider(Qt.Orientation.Horizontal)
        self.slice_position.setRange(0, 100)
        self.slice_position.setValue(50)
        row = QHBoxLayout()
        row.addWidget(QLabel("法線"))
        row.addWidget(self.slice_normal)
        slice_layout.addLayout(row)
        slice_layout.addWidget(QLabel("位置"))
        slice_layout.addWidget(self.slice_position)
        layout.addWidget(slice_box)

        iso_box = QGroupBox("等値面")
        iso_layout = QVBoxLayout(iso_box)
        self.iso_level = QSlider(Qt.Orientation.Horizontal)
        self.iso_level.setRange(0, 100)
        self.iso_level.setValue(50)
        iso_layout.addWidget(QLabel("しきい値（値域に対する割合）"))
        iso_layout.addWidget(self.iso_level)
        layout.addWidget(iso_box)

        stream_box = QGroupBox("ストリームライン")
        stream_layout = QVBoxLayout(stream_box)
        self.streamlines = QCheckBox("電流の流線を重ねる")
        self.n_seeds = QSpinBox()
        self.n_seeds.setRange(1, 500)
        self.n_seeds.setValue(24)
        stream_layout.addWidget(self.streamlines)
        row = QHBoxLayout()
        row.addWidget(QLabel("本数"))
        row.addWidget(self.n_seeds)
        stream_layout.addLayout(row)
        layout.addWidget(stream_box)

        self.note = QLabel(NODAL_VALUE_WARNING)
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: #a05000;")
        layout.addWidget(self.note)

        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        layout.addWidget(self.summary, stretch=1)

        for widget in (self.field, self.mode, self.cmap, self.slice_normal):
            widget.currentTextChanged.connect(self._refresh)
        self.edges.toggled.connect(self._refresh)
        self.streamlines.toggled.connect(self._refresh)
        self.slice_position.valueChanged.connect(self._refresh)
        self.iso_level.valueChanged.connect(self._refresh)
        self.n_seeds.valueChanged.connect(self._refresh)

    def reload(self) -> None:
        session = self.window.session
        self._loading = True
        previous = self.field.currentData()
        self.field.clear()
        for f in session.available_fields():
            self.field.addItem(f.label, f.key)
        if previous is not None:
            index = self.field.findData(previous)
            if index >= 0:
                self.field.setCurrentIndex(index)
        self.summary.setPlainText(
            session.result_summary() if session.study is not None else ""
        )
        self._loading = False

    def options(self) -> dict:
        return {
            "field": self.field.currentData() or "PID",
            "mode": self.mode.currentData() or "volume",
            "cmap": self.cmap.currentText(),
            "edges": self.edges.isChecked(),
            "slice_normal": self.slice_normal.currentText(),
            "slice_position": self.slice_position.value() / 100.0,
            "iso_level": self.iso_level.value() / 100.0,
            "streamlines": self.streamlines.isChecked(),
            "n_seeds": self.n_seeds.value(),
        }

    def _refresh(self) -> None:
        if not self._loading:
            self.window.refresh_view()
