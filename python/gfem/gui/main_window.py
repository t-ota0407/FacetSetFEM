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

from .. import GfemError, describe_gpus, read_nastran
from .busy import REVEAL_DELAY_MS, BusyOverlay, busy_scope, run_in_thread
from .solver_thread import SolverThread
from .state import (
    DEFAULT_RESULT_FIELD,
    DISPLAY_FIELDS,
    SHAPE_FIELD,
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

#: 色分けしないときの配色。**色は物理量のためにとっておく**ので、形は無彩色で見せる
SURFACE_COLOR = "#d7d9dc"
EDGE_COLOR = "#3a3f45"
NODE_COLOR = "#1b1e21"
NODE_SIZE = 4

#: 選択中の領域の色（無彩色の中で目を引くように）
REGION_COLOR = "#f08a24"

#: 体表面に出ていない領域を選んだとき、外側をどれだけ透かすか
BURIED_OPACITY = 0.25

#: カラーバーの体裁。横置きだと窓が狭いときに数字が重なって読めなくなる
SCALAR_BAR = {
    "vertical": True,
    "position_x": 0.82,
    "position_y": 0.12,
    "width": 0.05,
    "height": 0.72,
    "n_labels": 5,
    "fmt": "%.3g",
    "title_font_size": 14,
    "label_font_size": 12,
    "color": "#1b1e21",
}

#: 背景。うっすら階調をつけると、白いモデルの輪郭が背景に溶けない
BACKGROUND_BOTTOM = "#e9edf2"
BACKGROUND_TOP = "#fdfdfe"

#: 稜線・節点を既定で出す上限（表示する三角形の枚数）。
#:
#: これを超えると線が潰れて灰色の塊になり、形が**かえって見えなくなる**。
#: 実測で頭部モデルの外表面は 87,812 面。手動で入れることは妨げない
EDGE_LIMIT = 30_000

#: これを超えたら「大きなモデル」とみなし、ローディングを待たずに出す。
#:
#: 実測（RTX A6000 / 20 コア）で、35.9 万節点だと領域を選び直すだけで 0.98 秒、
#: 1.5 万節点なら 0.05 秒。10 万節点あたりが「待たされたと感じる」境目にあたる。
BIG_MODEL_NODES = 100_000

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
        # 体表面のクリック選択（SPEC §11.1）
        self._pick_callback = None      # (point, erase|None) -> None
        self._preview_callback = None   # (point|None) -> None
        self._stroke_end_callback = None  # 1 ストロークが終わったとき
        self._pick_observers: list[int] = []
        self._world_picker = None
        self._painting: str | bool = False
        self._press_position = (0, 0)
        self._left_down = False
        self._rotating = False

        self.setWindowTitle("GenericFEM")
        self.resize(1400, 900)
        self._build_ui()
        self._build_menus()
        # 重い処理の間に出すローディング（SPEC §6.4）。3D ビューの上に重ねるため、
        # 子ウィジェットではなくウィンドウに貼りつく別ウィンドウとして作る
        self.overlay = BusyOverlay(self)
        self.refresh_all()
        self._refresh_history_actions()

    # --- 処理中の表示（SPEC §6.4）-------------------------------------------

    def busy(self, message: str, detail: str | None = None, wait_ms: int | None = None):
        """メインスレッドを塞ぐ処理を囲む。区切りで `scope.step()` を呼ぶ。

        VTK と Qt は他スレッドから触れないので、描画まわりはここを使う。
        小さなモデルでは、200 ms 以内に終わる処理に対して何も表示されない。

        `wait_ms` を渡すと、出すまでの待ちをその値にする。**段階の区切りが
        こまやかで、ふつうは一瞬で終わる処理**（電極のクリックなど）に使う。
        大きなモデルでも毎回光らせずに済み、本当に待たされるときだけ出る。
        """
        delay = self._reveal_delay() if wait_ms is None else wait_ms
        return busy_scope(self.overlay, message, self._detail(detail), delay)

    def run_busy(self, message: str, work, detail: str | None = None):
        """GIL を手放す処理（読み込み・書き出し）をワーカースレッドで走らせて待つ。

        呼び出し側から見ると同期関数のままなので、例外はそのまま飛んでくる。
        """
        return run_in_thread(
            self.overlay, message, work, self._detail(detail), self._reveal_delay()
        )

    def _reveal_delay(self) -> int | None:
        """ローディングを出すまでの待ち。大きなモデルでは待たない。

        ちらつきを避けるための 200 ms の猶予は、**速いと分かっているときだけ**
        意味がある。大きなモデルでは領域を選び直すだけでも 1 秒近くかかり、
        しかもその間は画面を更新する隙が無いので、猶予を置くと処理の最後に
        一瞬だけ光って終わる（いちばん見せたい間に出ない）。
        """
        mesh = self.session.mesh
        if mesh is not None and mesh.n_nodes >= BIG_MODEL_NODES:
            return 0
        return None

    def _detail(self, detail: str | None) -> str:
        """補足行。指定が無ければモデルの規模を出す（待ち時間の目安になる）。"""
        if detail is not None:
            return detail
        mesh = self.session.mesh
        if mesh is None:
            return ""
        return f"{mesh.n_nodes:,} 節点 / {mesh.n_elements:,} 要素"

    # --- UI の構築 ---------------------------------------------------------

    def _build_ui(self) -> None:
        from pyvistaqt import QtInteractor

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["モデル"])
        self.tree.setMinimumWidth(240)
        self.tree.currentItemChanged.connect(self._on_tree_selection)

        self.plotter = QtInteractor(self)
        # **ライトを明示的に置く。** pyvistaqt の描画器は光源を 1 つも持たず
        # （自動生成も切れている）、そのままだと陰影が付かないので、
        # どんな形も**のっぺりした影絵**になる。単体の Plotter と同じ
        # 三点照明キットを入れると、白一色でも形が読める
        self.plotter.enable_lightkit()
        self.plotter.set_background(BACKGROUND_BOTTOM, top=BACKGROUND_TOP)

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

        edit_menu = self.menuBar().addMenu("編集(&E)")
        self.action_undo = self._add_action(edit_menu, "元に戻す", self.undo, "Ctrl+Z")
        self.action_redo = self._add_action(edit_menu, "やり直す", self.redo, "Ctrl+Y")
        self._refresh_history_actions()

        run_menu = self.menuBar().addMenu("実行(&R)")
        self._add_action(run_menu, "整合性チェック", self.check_consistency, "Ctrl+K")
        self._add_action(run_menu, "solve", self.start_solve, "F5")
        self.action_cancel = self._add_action(run_menu, "中断", self.cancel_solve)
        self.action_cancel.setEnabled(False)
        run_menu.addSeparator()
        self._add_action(run_menu, "パラメータスタディ...", self.open_sweep_dialog)

        electrode_menu = self.menuBar().addMenu("電極(&L)")
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
        if path:
            self.open_mesh_asking_unit(path)

    def open_mesh_asking_unit(self, path: str | Path) -> None:
        """長さ単位を選ばせてからメッシュを読む（SPEC §3.2）。

        Nastran ファイルには単位情報が無い。**単位を黙って決めてはいけない**ので、
        ファイルを開く経路は（コマンドラインの引数を含めて）すべてここを通す。
        """
        try:
            # SPEC §3.2: 単位の選択前に寸法を見せたいので、一度そのままの値（m）で読む。
            # 選ばれた単位へは付け替えるだけにして、ファイルを 2 回パースしない
            as_metre = self.run_busy(
                "寸法を確認しています",
                lambda: read_nastran(path, length_unit="m"),
                Path(path).name,
            )
            box = as_metre.bounding_box()
            extent = tuple(hi - lo for lo, hi in zip(*box))
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc)
            return

        dialog = UnitSelector(extent, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        unit = dialog.length_unit
        self.open_mesh(path, unit, lambda: as_metre.with_length_unit(unit))

    def open_mesh(self, path: str | Path, length_unit: str, loaded=None) -> None:
        """メッシュを読み込む（テストからも呼べるようダイアログを挟まない）。

        `loaded` は読み込み済みのメッシュを返す関数（ワーカースレッドで呼ぶ）。
        """
        try:
            self.run_busy(
                "メッシュを読み込んでいます",
                lambda: self.session.open_mesh(
                    path, length_unit, None if loaded is None else loaded()
                ),
                Path(path).name,
            )
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc)
            return
        self.convergence.clear()
        self.checks_view.clear()
        self.log(f"読み込み完了: {path}")
        self.log(self.session.mesh_summary())
        self.refresh_all()
        self._refresh_history_actions()

    def open_case_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "ケースを開く", "", "ケースファイル (*.yaml *.yml);;すべて (*)"
        )
        if path:
            self.open_case(path)

    def open_case(self, path: str | Path) -> None:
        try:
            self.run_busy(
                "ケースを読み込んでいます",
                lambda: self.session.open_case(path),
                Path(path).name,
            )
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc)
            return
        self.convergence.clear()
        self.checks_view.clear()
        self.log(f"ケースを読み込みました: {path}")
        self.log(self.session.mesh_summary())
        self.refresh_all()
        self._refresh_history_actions()

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
            self.run_busy(
                "VTU を書き出しています",
                lambda: self.session.solution.write_vtu(path),
                Path(path).name,
            )
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
            self.run_busy("書き出しています", lambda: writer(path), Path(path).name)
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc)
            return
        self.log(f"書き出しました: {path}")

    def screenshot_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "スクリーンショット", "view.png", "PNG (*.png)")
        if path:
            self.screenshot(path)

    def screenshot(self, path: str | Path) -> None:
        # 画面の取り込みは OpenGL コンテキストを持つメインスレッドでしかできない
        with self.busy("スクリーンショットを保存しています", Path(path).name):
            self.plotter.screenshot(str(path))
        self.log(f"スクリーンショットを保存しました: {path}")

    # --- パラメータスタディ（SPEC §8 M7）-------------------------------------

    def open_sweep_dialog(self) -> None:
        """パラメータを振って一括で解くダイアログを開く。"""
        if self.session.study is None:
            self.log("先にモデルかケースを開いてください")
            return
        from .sweep_dialog import SweepDialog

        SweepDialog(self).exec()

    # --- 体表面のクリック選択（SPEC §11.1）----------------------------------

    #: 電極の強調表示に付ける名前。同じ名前で足し直すと差し替えになる
    ELECTRODE_ACTOR = "electrode-highlight"
    #: 塗る前の予告表示に付ける名前
    PREVIEW_ACTOR = "paint-preview"

    def refresh_electrode_highlight(self, name: str | None) -> None:
        """電極の強調表示だけを描き直す。

        3D ビュー全体の作り直しは 35 万節点で 1 秒、4 百万四面体では 3 秒以上かかる。
        電極面を選んでいる間はそこだけが変わるので、名前付きアクタを差し替える。
        """
        if self.session.mesh is None:
            return
        if self.is_painting:
            # 筆でなぞっている最中はローディングを挟まない。
            # 1 なぞりごとに `processEvents` を回すと、そのぶん筆が遅れる
            self._draw_electrode_highlight(name)
            return
        with self.busy("電極面を描いています"):
            self._draw_electrode_highlight(name)
            self._arm_picking()

    def _draw_electrode_highlight(self, name: str | None) -> None:
        self.plotter.remove_actor(self.ELECTRODE_ACTOR, render=False)
        faces = self.session.electrode_polydata(name) if name else None
        if faces is not None and faces.n_cells:
            self.plotter.add_mesh(
                faces,
                color="red",
                opacity=1.0,
                name=self.ELECTRODE_ACTOR,
                reset_camera=False,
                pickable=False,
            )
        self.plotter.render()

    def begin_patch_picking(self, callback, preview=None, stroke_end=None) -> None:
        """3D ビューのクリックで体表面の点を拾うモードに入る。

        描き直すたびに掛け直す必要があるので、コールバックを覚えておく
        （`refresh_view` が `plotter.clear()` を呼ぶため）。
        """
        self._pick_callback = callback
        self._preview_callback = preview
        self._stroke_end_callback = stroke_end
        self._arm_picking()
        self.log(
            "体表面をクリックすると電極面になります"
            "（Shift + ドラッグで塗る / Alt + ドラッグで消す / ドラッグは回転）"
        )

    def end_patch_picking(self) -> None:
        """クリック選択を終え、通常の操作に戻す。"""
        self._pick_callback = None
        self._preview_callback = None
        self._stroke_end_callback = None
        self._disarm_picking()

    def _arm_picking(self) -> None:
        """クリック選択の下ごしらえ。掛かっていても掛かっていなくても呼べる。

        **VTK のイベントを自前で拾う。** PyVista の `enable_point_picking` は
        クリック 1 回しか扱えず、ドラッグで塗る・ホバーで予告するができない。
        ピック自体は世界座標ピッカー（深度バッファ）で行う——既定のセルピッカーは
        表示中のメッシュに光線を飛ばすので、411 万セルの頭部モデルでは
        1 クリック 2.6 秒かかる。深度バッファからの逆変換なら 0.01 秒で、
        ずれは実測 0.4〜1.8 mm。**選んだ点は最寄りの面を決めるためだけに使い、
        電極の広がりは半径（cm 級）が決める**ので、この差は効かない（SPEC §11.1）。
        """
        if self._pick_callback is None or self._pick_observers:
            return
        from vtkmodules.vtkRenderingCore import vtkWorldPointPicker

        self._world_picker = vtkWorldPointPicker()
        iren = self.plotter.iren.interactor
        # 優先度を上げて、回転などの既定の操作より先に受け取る
        self._pick_observers = [
            iren.AddObserver("LeftButtonPressEvent", self._on_mouse_press, 10.0),
            iren.AddObserver("MouseMoveEvent", self._on_mouse_move, 10.0),
            iren.AddObserver("LeftButtonReleaseEvent", self._on_mouse_release, 10.0),
        ]

    def _disarm_picking(self) -> None:
        iren = self.plotter.iren.interactor
        for tag in self._pick_observers:
            iren.RemoveObserver(tag)
        self._pick_observers = []
        self._painting = False
        self._left_down = False
        self._rotating = False
        self.clear_paint_preview()

    # --- マウスの扱い -------------------------------------------------------
    #
    # 左クリック（動かさない）  = いまのモードで 1 回塗る / 消す
    # 左ドラッグ                = これまで通り回転する
    # Shift + 左ドラッグ        = 塗る
    # Alt + 左ドラッグ          = 消す
    #
    # **ドラッグをそのまま塗りに使ってはいけない。** 回転を奪うと、頭の反対側に
    # 電極を置くために毎回モードを切り替えることになる。修飾キーを押している間
    # だけ塗るなら、道具を持ち替えずに回しながら塗れる。

    def _mouse_position(self):
        iren = self.plotter.iren.interactor
        return iren.GetEventPosition()

    def _pick_world_point(self):
        """カーソルの下の点（世界座標）。モデルの上でなければ None。"""
        if self._world_picker is None:
            return None
        x, y = self._mouse_position()
        self._world_picker.Pick(x, y, 0, self.plotter.renderer)
        point = self._world_picker.GetPickPosition()
        return point if self.session.is_on_the_model(point) else None

    @staticmethod
    def _paint_mode(iren) -> str | None:
        """修飾キーから塗り方を決める。押していなければ None。"""
        if iren.GetShiftKey():
            return "add"
        if iren.GetAltKey():
            return "erase"
        return None

    #: これ以上動いたらクリックではなくドラッグとみなす [px]
    DRAG_THRESHOLD = 3

    @property
    def is_painting(self) -> bool:
        """いま筆でなぞっている最中か。"""
        return bool(self._painting)

    def _on_mouse_press(self, iren, _event) -> None:
        """左ボタンを押した。**ここではまだ回転を始めさせない。**

        VTK の既定のスタイルは押した瞬間に回転を開始し、そのときイベントの
        フォーカスを掴んでしまう。すると「離した」がこちらに届かず、
        **動かさずに離す＝クリック**を見分けられない。いったん押下を受け止め、
        実際に動いたと分かった時点で回転を始めさせる。
        """
        self._press_position = self._mouse_position()
        self._left_down = True
        self._rotating = False
        mode = self._paint_mode(iren)
        self._painting = mode or False
        if mode is not None:
            point = self._pick_world_point()
            if point is not None:
                self._pick_callback(point, mode == "erase")
        self._abort(iren, 0)

    def _on_mouse_move(self, iren, _event) -> None:
        if self._painting:
            point = self._pick_world_point()
            if point is not None:
                self._pick_callback(point, self._painting == "erase")
            self._abort(iren, 1)
            return
        if self._left_down:
            if not self._rotating and self._moved_enough():
                # ドラッグと分かった。ここから既定の回転を始めてもらう
                self._rotating = True
                self.clear_paint_preview()
                iren.GetInteractorStyle().OnLeftButtonDown()
            if not self._rotating:
                self._abort(iren, 1)
            return
        if self._preview_callback is not None:
            self._preview_callback(self._pick_world_point())

    def _on_mouse_release(self, iren, _event) -> None:
        was_painting, rotating = self._painting, self._rotating
        self._left_down = False
        self._painting = False
        self._rotating = False
        if rotating:
            return  # 回転はスタイルに任せる
        self._abort(iren, 2)
        if was_painting:
            # ストロークの間は強調表示だけを更新している。
            # 離した時点で面積・電流密度・一覧をまとめて作り直す
            if self._stroke_end_callback is not None:
                self._stroke_end_callback()
            return
        # 動かさずに離したならクリック。いまのモードで 1 回置く
        if not self._moved_enough():
            point = self._pick_world_point()
            if point is not None:
                self._pick_callback(point, None)

    def _abort(self, iren, index: int) -> None:
        """このイベントを既定の処理（回転など）に渡さない。"""
        iren.GetCommand(self._pick_observers[index]).SetAbortFlag(1)

    def _moved_enough(self) -> bool:
        x0, y0 = self._press_position
        x1, y1 = self._mouse_position()
        return max(abs(x1 - x0), abs(y1 - y0)) > self.DRAG_THRESHOLD

    # --- 塗る前の予告表示 ---------------------------------------------------

    def show_paint_preview(self, polydata, erase: bool) -> None:
        """これから塗られる（消される）範囲を薄く出す。"""
        if polydata is None or not polydata.n_cells:
            self.clear_paint_preview()
            return
        self.plotter.add_mesh(
            polydata,
            color="#2a5adb" if erase else "#ff8c1a",
            opacity=0.55,
            name=self.PREVIEW_ACTOR,
            reset_camera=False,
            pickable=False,
        )
        self.plotter.render()

    def clear_paint_preview(self) -> None:
        if self.plotter.actors.get(self.PREVIEW_ACTOR) is not None:
            self.plotter.remove_actor(self.PREVIEW_ACTOR, render=False)
            self.plotter.render()

    # --- 元に戻す / やり直す（SPEC §8 M7）-----------------------------------

    def undo(self) -> None:
        label = self.session.undo()
        if not label:
            return
        self.refresh_all()
        self._refresh_history_actions()
        self.log(f"元に戻しました: {label}")

    def redo(self) -> None:
        label = self.session.redo()
        if not label:
            return
        self.refresh_all()
        self._refresh_history_actions()
        self.log(f"やり直しました: {label}")

    def checkpoint(self, label: str) -> None:
        """編集を始める前に呼ぶ。同じ名前の編集が続いている間はまとめられる。"""
        self.session.checkpoint(label)
        self._refresh_history_actions()

    def _refresh_history_actions(self) -> None:
        """メニューに「何を戻すのか」を出す。押せるかどうかもここで決める。"""
        undo, redo = self.session.can_undo(), self.session.can_redo()
        self.action_undo.setEnabled(undo)
        self.action_redo.setEnabled(redo)
        self.action_undo.setText(
            f"元に戻す: {self.session.undo_label()}" if undo else "元に戻す"
        )
        self.action_redo.setText(
            f"やり直す: {self.session.redo_label()}" if redo else "やり直す"
        )

    # --- 電極 ---------------------------------------------------------------

    def add_electrode(self) -> None:
        if self.session.study is None:
            return
        existing = {e.name for e in self.session.study.electrodes}
        index = len(existing) + 1
        while f"E{index}" in existing:
            index += 1
        role = "anode" if not existing else "cathode"
        self.checkpoint(f"電極 E{index} を追加")
        self.session.add_electrode(f"E{index}", role=role)
        self.refresh_tree()
        self.log(f"電極 E{index} を追加しました（{role}）")

    def remove_selected_electrode(self) -> None:
        item = self.tree.currentItem()
        if item is None or item.data(0, ROLE_KIND) != "electrode":
            return
        name = item.data(0, ROLE_KEY)
        self.checkpoint(f"電極 {name} を削除")
        self.session.remove_electrode(name)
        self.refresh_tree()
        self.log(f"電極 {name} を削除しました")

    # --- 実行 ---------------------------------------------------------------

    def check_consistency(self) -> list[str] | None:
        """整合性チェック（SPEC §6.2 の手順 4）。"""
        try:
            warnings = self.run_busy("整合性をチェックしています", self.session.validate)
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
            case, study = self.run_busy("解析条件を組み立てています", self.session.prepare_solve)
            case.validate()
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc)
            return

        self.convergence.clear()
        self.convergence.set_tolerance(self.session.study.solver.tolerance)
        self.bottom.setCurrentWidget(self.convergence)
        self.log("solve を開始しました")
        self.action_cancel.setEnabled(True)

        self.solver = SolverThread(case, study, self)
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
        # solve 自体はワーカースレッドだが、結果の取り込み（領域統計・表示用データの
        # 作り直し）はメインスレッドで、規模によっては数秒かかる
        with self.busy("結果を取り込んでいます") as scope:
            self._take_solution(scope, solution)

    def _take_solution(self, scope, solution) -> None:
        self.session.set_solution(solution, self.solver.study)
        self.convergence.set_history(solution.history)
        self.checks_view.setPlainText(self.session.result_summary())
        scope.step("領域統計を集計しています")
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
        self.page_result.show_result_field()
        self.refresh_view()

    def _on_solve_failed(self, message: str) -> None:
        self.log(f"solve 失敗:\n{message}")
        QMessageBox.warning(self, "solve", message)

    # --- 表示 ---------------------------------------------------------------

    def refresh_all(self) -> None:
        # 全体をひとまとめに囲む。中の refresh_view も同じローディングを使うので、
        # 途中で消えて出直すような瞬きが起きない
        with self.busy("表示を作成しています"):
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
        """3D ビューを描き直す。

        VTK は他スレッドから触れないのでメインスレッドで行う。代わりに段階ごとに
        `scope.step()` で表示を進める（SPEC §6.4）。
        """
        if self.session.mesh is None:
            return
        with self.busy("メッシュを展開しています") as scope:
            self._draw_view(scope, highlight_pid, highlight_electrode)

    def _draw_view(self, scope, highlight_pid, highlight_electrode) -> None:
        self.plotter.clear()
        # **clear() は光源も消す。** 置き直さないと以降ずっと陰影が付かず、
        # どんな形も影絵になる（気づきにくいので、描き直しのたびに入れ直す）
        self.plotter.enable_lightkit()
        options = self.page_result.options()
        scalars = options["field"]
        mode = options["mode"]

        if mode in ("slice", "clip"):
            # 体積メッシュが要るのは断面とクリップだけ
            scope.step("断面を切り出しています")
            grid = self.session.grid()
            target = grid
        else:
            # **体積メッシュではなく外表面を描く**（SPEC §6.3）。
            # 頭部モデルで 2.7 秒 → 0.09 秒。しかも法線が付くので形が読める。
            # 等値面は Rust 側で切り出すので、空だったときの代わりの表示もこれで足りる
            target = self.session.display_surface(scalars)
            grid = target
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

        if scalars == SHAPE_FIELD or scalars not in target.array_names:
            scalars = None  # 色分けなし（PID には大小の意味が無い）

        # 選んだ領域を先に用意する。**体表面に出ていない領域**（頭部モデルの
        # 大半がそう）は、外側を半透明にしないと見えないため
        region = None
        if highlight_pid is not None:
            scope.step("選んだ領域を取り出しています")
            region = self.session.region_polydata(highlight_pid)
            if region.n_cells == 0:
                # 体表面に出ていない領域。体積メッシュから切り出す（0.5 秒）
                region = self.session.region_grid(highlight_pid).extract_surface(
                    algorithm="dataset_surface"
                )
        buried = region is not None and region.n_cells > 0 and mode == "volume" and (
            self.session.region_polydata(highlight_pid).n_cells == 0
        )

        scope.step("表示を作成しています")
        edges = options["edges"]
        if scalars is None:
            # 色分けなし。形は**陰影と稜線**で見せる（SPEC §6.3）
            self.plotter.add_mesh(
                target,
                color=SURFACE_COLOR,
                show_edges=edges,
                edge_color=EDGE_COLOR,
                line_width=1,
                specular=0.25,
                specular_power=18,
                smooth_shading=False,
                opacity=BURIED_OPACITY if buried else 1.0,
            )
        else:
            self.plotter.add_mesh(
                target,
                scalars=scalars,
                cmap=options["cmap"],
                show_edges=edges,
                edge_color=EDGE_COLOR,
                clim=self.session.field_range(scalars),
                scalar_bar_args=SCALAR_BAR,
                categories=scalars == "PID",
                opacity=BURIED_OPACITY if buried else 1.0,
            )
        if options["nodes"]:
            self.plotter.add_mesh(
                target,
                style="points",
                color=NODE_COLOR,
                point_size=NODE_SIZE,
                render_points_as_spheres=True,
                lighting=False,
            )

        if options["streamlines"] and self.session.solution is not None:
            scope.step("流線を追跡しています")
            try:
                seeds = self.session.default_seed_points(options["n_seeds"])
                lines = self.session.streamlines(seeds)
                if lines.n_cells:
                    self.plotter.add_mesh(lines, color="black", line_width=2)
            except Exception as exc:  # noqa: BLE001
                self.log(f"ストリームラインを引けませんでした: {exc}")

        if region is not None and region.n_cells:
            self.plotter.add_mesh(
                region, color=REGION_COLOR, show_edges=False, pickable=False
            )
        if highlight_electrode is not None:
            scope.step("電極の面を取り出しています")
            faces = self.session.electrode_polydata(highlight_electrode)
            if faces.n_cells:
                self.plotter.add_mesh(faces, color="red", opacity=1.0)

        scope.step("描画しています")
        self.plotter.add_axes()
        self.plotter.reset_camera()
        self.plotter.render()
        # clear() でピッキングも外れるので、選択中なら掛け直す
        self._arm_picking()

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
        self.window.checkpoint(f"領域 {self.pid} の設定を変更")
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
        #: ホバー予告をいま出している面（同じ面の上では作り直さない）
        self._hover_at = None

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

        # --- 体表面から選ぶ（SPEC §11.1）---
        # 人体モデルはソリッド要素だけで配られるのが普通で、その場合シェル PID の
        # 一覧は空になる。3D ビューを直接クリックして面を選べるようにする
        patch_box = QGroupBox("体表面から選ぶ（クリック）")
        patch_layout = QVBoxLayout(patch_box)
        self.pick_button = QPushButton("クリックで選ぶ")
        self.pick_button.setCheckable(True)
        self.pick_button.setToolTip(
            "3D ビューで体表面をクリックすると、その周りの面が電極になります。\n"
            "もう一度押すと通常の回転操作に戻ります。"
        )
        patch_layout.addWidget(self.pick_button)

        patch_form = QFormLayout()
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setDecimals(1)
        self.radius_spin.setRange(0.1, 500.0)
        self.radius_spin.setValue(25.0)
        self.radius_spin.setSingleStep(5.0)
        self.radius_spin.setSuffix(" mm")
        self.patch_pid = QComboBox()
        self.patch_pid.setToolTip(
            "外表面に出ている領域の一覧（面積の大きい順）。\n"
            "首の切断面など「外表面だが体表面ではない」面を避けるために、\n"
            "ふつうは皮膚の PID に絞り込みます。"
        )
        patch_form.addRow("ブラシの半径", self.radius_spin)
        patch_form.addRow("領域を限る", self.patch_pid)
        patch_layout.addLayout(patch_form)

        self.patch_list = QListWidget()
        self.patch_list.setMaximumHeight(90)
        mode_row = QHBoxLayout()
        self.add_mode = QPushButton("追加")
        self.erase_mode = QPushButton("削除")
        for button in (self.add_mode, self.erase_mode):
            button.setCheckable(True)
            mode_row.addWidget(button)
        self.add_mode.setChecked(True)
        self.add_mode.toggled.connect(lambda on: on and self.erase_mode.setChecked(False))
        self.erase_mode.toggled.connect(lambda on: on and self.add_mode.setChecked(False))
        patch_layout.addLayout(mode_row)
        hint = QLabel(
            "クリック＝いまのモードで 1 回。Shift＋ドラッグで塗る、"
            "Alt＋ドラッグで消す。ふつうのドラッグは回転。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        patch_layout.addWidget(hint)

        patch_layout.addWidget(self.patch_list)
        self.hand_edits = QLabel("")
        self.hand_edits.setWordWrap(True)
        patch_layout.addWidget(self.hand_edits)

        button_row = QHBoxLayout()
        self.remove_patch_button = QPushButton("選んだパッチを外す")
        self.remove_patch_button.clicked.connect(self._remove_patch)
        button_row.addWidget(self.remove_patch_button)
        self.clear_edits_button = QPushButton("手直しを消す")
        self.clear_edits_button.setToolTip(
            "1 枚ずつ足した面・外した面を捨てて、パッチだけの状態に戻します。"
        )
        self.clear_edits_button.clicked.connect(self._clear_hand_edits)
        button_row.addWidget(self.clear_edits_button)
        patch_layout.addLayout(button_row)
        layout.addWidget(patch_box)

        self.summary = QLabel("-")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.summary)

        self.name_edit.editingFinished.connect(self._commit)
        self.role_combo.currentTextChanged.connect(self._commit)
        self.current_spin.valueChanged.connect(self._commit)
        self.pid_list.itemChanged.connect(self._commit)
        self.pick_button.toggled.connect(self._toggle_picking)

    def bind(self, name: str) -> None:
        # 体表面の索引づくり（外表面 87,812 面で 1.9 秒）が初回だけここで走る。
        # 電極をツリーで選んだ瞬間なので、待たされていることを見せる（SPEC §6.4）
        with self.window.busy("電極の設定を開いています") as scope:
            self._bind(name, scope)

    def _bind(self, name: str, scope) -> None:
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
        scope.step("体表面の領域を調べています")
        self._reload_patch_pids()
        self.name = name
        self._reload_patches()
        self._reload_hand_edits()
        scope.step("面積と電流密度を集計しています")
        self._update_summary()

    # --- 体表面から選ぶ（SPEC §11.1）----------------------------------------

    def _reload_patch_pids(self) -> None:
        """外表面に出ている領域を、面積の大きい順に並べる。"""
        current = self.patch_pid.currentData()
        self.patch_pid.blockSignals(True)
        self.patch_pid.clear()
        self.patch_pid.addItem("制限しない", None)
        for row in self.window.session.surface_pid_choices():
            self.patch_pid.addItem(
                f"PID {row['pid']}（{row['area'] * 1e4:.0f} cm²）", row["pid"]
            )
        # 既定はいちばん広い領域＝ふつう皮膚。切断面を拾わないための初期値
        index = self.patch_pid.findData(current if current is not None else self._default_pid())
        self.patch_pid.setCurrentIndex(max(index, 0))
        self.patch_pid.blockSignals(False)

    def _default_pid(self):
        return self.window.session.default_surface_pid()

    def _reload_patches(self) -> None:
        """パッチ一覧を作り直す。面数と面積も出す。"""
        self.patch_list.clear()
        spec = self.window.session.electrode(self.name) if self.name else None
        if spec is None:
            return
        for i, patch in enumerate(spec.patches):
            try:
                info = self.window.session.pick_patch(patch.center, patch.radius, patch.pid)
                detail = f"{info['n_faces']} 面 {info['area'] * 1e4:.1f} cm²"
            except Exception as exc:  # noqa: BLE001 - 表示だけなので握って理由を出す
                detail = str(exc).splitlines()[0]
            centre = "(" + ", ".join(f"{v * 1e3:.0f}" for v in patch.center) + ") mm"
            limit = f" PID {patch.pid}" if patch.pid is not None else ""
            self.patch_list.addItem(
                f"{i + 1}. {centre} 半径 {patch.radius * 1e3:.0f} mm{limit}　{detail}"
            )

    def _reload_hand_edits(self) -> None:
        """手で足した面・外した面の枚数を出す。"""
        spec = self.window.session.electrode(self.name) if self.name else None
        if spec is None or not (spec.added_faces or spec.removed_faces):
            self.hand_edits.setText("")
            self.clear_edits_button.setEnabled(False)
            return
        parts = []
        if spec.added_faces:
            parts.append(f"手で足した面 {len(spec.added_faces)} 枚")
        if spec.removed_faces:
            parts.append(f"外した面 {len(spec.removed_faces)} 枚")
        self.hand_edits.setText("　".join(parts))
        self.clear_edits_button.setEnabled(True)

    def _toggle_picking(self, on: bool) -> None:
        if self.name is None:
            self.pick_button.setChecked(False)
            return
        if on:
            # 断面や等値面を出しているとカーソルの下は体内の面になる。
            # 電極はいちばん近い体表面に置かれるので、意図とずれやすい
            if self.window.page_result.options()["mode"] not in ("volume", "clip"):
                self.window.log(
                    "断面や等値面を表示中です。電極は体表面に置かれるので、"
                    "「全体」表示に戻してから選ぶことをお勧めします"
                )
            self.window.begin_patch_picking(
                self._on_picked, self._on_hover, self._on_stroke_end
            )
            self.pick_button.setText("クリックで選ぶ（実行中。押すと終了）")
        else:
            self.window.end_patch_picking()
            self.pick_button.setText("クリックで選ぶ")

    def _brush(self) -> tuple[float, int | None]:
        return self.radius_spin.value() * 1e-3, self.patch_pid.currentData()

    def _on_picked(self, point, erase: bool | None = None) -> None:
        """3D ビューが操作された。電極の面を書き換える。

        - `erase` が `None`（ふつうのクリック）: 追加モードなら**パッチを 1 枚置く**、
          削除モードならその範囲の面を外す
        - `erase` が真偽値（修飾キー＋ドラッグ）: 面を 1 枚ずつ塗る / 消す

        **クリックでパッチを置く道を残すのが要点。** 中心と半径という読める指定が
        ケースファイルに残り、後から半径だけ変えられる。手で塗ったぶんは差分として
        別に持つ（SPEC §11.1）。
        """
        if self.name is None:
            return
        session = self.window.session
        if not session.is_on_the_model(point):
            # 背景をクリックした（ピッカーは遠くの点を返す）。
            # そのまま置くと見当違いの場所に電極ができるので何もしない
            self.window.log("モデルの外をクリックしました。体表面の上をクリックしてください")
            return

        radius, pid = self._brush()
        if erase is None and not self.erase_mode.isChecked():
            self._place_patch(point, radius, pid)
            return

        erase = True if erase is None else erase
        label = "消す" if erase else "塗る"
        # ドラッグ中は同じ名前で checkpoint を積むので、**1 ストロークが 1 手**になる
        self.window.checkpoint(f"電極 {self.name} の面を{label}")
        try:
            changed = session.paint_faces(self.name, point, radius, pid, erase)
        except Exception as exc:  # noqa: BLE001
            self.window.show_error(exc)
            return
        if changed == 0:
            return  # 何も変わっていない。描き直す必要もない
        self._after_patch_change()

    def _place_patch(self, point, radius: float, pid: int | None) -> None:
        """クリックした場所にパッチを 1 枚置く。"""
        session = self.window.session
        with self.window.busy("電極面を選んでいます", wait_ms=REVEAL_DELAY_MS) as scope:
            try:
                preview = session.pick_patch(point, radius, pid)
            except Exception as exc:  # noqa: BLE001
                self.window.show_error(exc)
                return
            self.window.checkpoint(f"電極 {self.name} にパッチを追加")
            session.add_patch(self.name, point, radius, pid)
            self.window.log(
                f"電極 {self.name} にパッチを追加: "
                f"{preview['n_faces']} 面, 面積 {preview['area'] * 1e4:.2f} cm²"
            )
            self._after_patch_change(scope)

    def _on_hover(self, point) -> None:
        """カーソルの下に、これから塗られる範囲を薄く出す（SPEC §6.2 の手順 3）。"""
        if self.name is None or point is None:
            self.window.clear_paint_preview()
            self._hover_at = None
            return
        # 動くたびに作り直すと重いので、同じ面の上にいる間は作り直さない
        radius, pid = self._brush()
        try:
            info = self.window.session.pick_patch(point, 0.0, pid)
            seed = tuple(info["face_refs"][0]) if info["n_faces"] else None
        except Exception:  # noqa: BLE001 - 予告表示なので黙って諦める
            return
        if seed is not None and seed == self._hover_at:
            return
        self._hover_at = seed
        try:
            preview = self.window.session.pick_patch(point, radius, pid)
        except Exception:  # noqa: BLE001
            return
        self.window.show_paint_preview(
            self.window.session.triangles_to_polydata(preview["triangles"]),
            erase=self.erase_mode.isChecked(),
        )

    def _clear_hand_edits(self) -> None:
        if self.name is None:
            return
        with self.window.busy("手直しを消しています", wait_ms=REVEAL_DELAY_MS) as scope:
            self.window.checkpoint(f"電極 {self.name} の手直しを消す")
            self.window.session.clear_hand_edits(self.name)
            self._after_patch_change(scope)

    def _remove_patch(self) -> None:
        row = self.patch_list.currentRow()
        if self.name is None or row < 0:
            return
        with self.window.busy("電極面を外しています", wait_ms=REVEAL_DELAY_MS) as scope:
            self.window.checkpoint(f"電極 {self.name} のパッチを削除")
            self.window.session.remove_patch(self.name, row)
            self._after_patch_change(scope)

    def _on_stroke_end(self) -> None:
        """塗り終わった。面積・電流密度・一覧をまとめて作り直す。"""
        self._after_patch_change()

    def _after_patch_change(self, scope=None) -> None:
        if self.window.is_painting:
            # **ストロークの最中は強調表示だけ。** 面積の集計は 1 回 0.07 秒あり、
            # マウスが動くたびに回すと筆が付いてこない（実測 4.5 コマ/秒）。
            # 離した時点でまとめて作り直す
            self.window.refresh_electrode_highlight(self.name)
            return
        self._reload_patches()
        self._reload_hand_edits()
        if scope is not None:
            scope.step("面積と電流密度を集計しています")
        self._update_summary()
        self.window.refresh_tree()
        if scope is not None:
            scope.step("電極面を描いています")
        # クリックのたびに全体を描き直すと大きなモデルで数秒かかる。
        # 変わるのは電極の強調表示だけなので、そこだけ差し替える
        self.window.refresh_electrode_highlight(self.name)

    def _commit(self) -> None:
        if self.name is None:
            return
        session = self.window.session
        spec = session.electrode(self.name)
        if spec is None:
            return
        self.window.checkpoint(f"電極 {self.name} の設定を変更")
        new_name = self.name_edit.text().strip()
        if new_name and new_name != spec.name:
            try:
                session.rename_electrode(spec.name, new_name)
            except GfemError as exc:
                # ほかの電極と同じ名前にはできない。入力欄を元の名前に戻す
                self.name_edit.setText(spec.name)
                self.window.show_error(exc)
        elif not new_name:
            self.name_edit.setText(spec.name)
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
        # 筆で塗っただけ（パッチなし）の電極も面を持つ（SPEC §11.1）
        if spec is None or not (
            spec.shell_pids or spec.element_ids or spec.patches or spec.added_faces
        ):
            self.summary.setText("面が選択されていません")
            return
        try:
            # この電極だけを見る。まだ面を割り当てていない別の電極があっても、
            # いま編集している電極の面積は出せる
            info = session.electrode_info(spec.name)[0]
        except (GfemError, IndexError) as exc:
            self.summary.setText(str(exc).splitlines()[0])
            return
        self.summary.setText(
            f"{info['n_faces']} 面　合計面積 {info['area'] * 1e6:.1f} mm²"
            f"（{info['area'] * 1e4:.2f} cm²）\n"
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
        self.preconditioner.addItems(["jacobi", "amg", "none"])
        self.backend = QComboBox()
        self.backend.addItems(["auto", "cpu", "cpu_parallel", "cuda"])
        self.reference = QComboBox()
        self.reference.addItems(["zero_mean", "cathode_mean"])
        form.addRow("相対残差の収束判定", self.tolerance)
        form.addRow("最大反復回数", self.max_iterations)
        form.addRow("前処理", self.preconditioner)
        form.addRow("計算バックエンド", self.backend)
        form.addRow("基準電位", self.reference)
        layout.addLayout(form)
        layout.addWidget(
            QLabel(
                "ゲージ固定は射影（平均ゼロ）で行います。\n"
                "1 節点ピン留めは解を歪めるため採用しません（SPEC §2.2）。"
            )
        )
        # どのバックエンドでも答えはビット単位で同じ（CLAUDE.md 設計原則 1）。
        # 迷わせないよう、そのことと GPU の有無をここに書いておく
        self.environment = QLabel(describe_gpus())
        self.environment.setWordWrap(True)
        layout.addWidget(
            QLabel(
                "バックエンドを変えても結果は 1 ビットも変わりません（速度だけが変わります）。\n"
                "前処理はこれと違い、反復の道筋が変わるので答えが収束判定の範囲で揺れます。\n"
                "amg（代数マルチグリッド）は反復回数がメッシュの細かさにほとんど依存しなく\n"
                "なりますが、1 反復は重くなります。細かいメッシュほど有利です（SPEC §4.5）。"
            )
        )
        layout.addWidget(self.environment)
        layout.addStretch(1)

        self.tolerance.editingFinished.connect(self._commit)
        self.max_iterations.valueChanged.connect(self._commit)
        self.preconditioner.currentTextChanged.connect(self._commit)
        self.backend.currentTextChanged.connect(self._commit)
        self.reference.currentTextChanged.connect(self._commit)

    def reload(self) -> None:
        study = self.window.session.study
        if study is None:
            return
        self._loading = True
        self.tolerance.setText(f"{study.solver.tolerance:g}")
        self.max_iterations.setValue(study.solver.max_iterations)
        # ケースファイルに選択肢に無い値（`node:<ID>` の基準電位など）が入っていたときに、
        # 黙って既定値へ戻さない。戻すと、ほかの項目を触った時点で上書きされて消える
        for combo, value in (
            (self.preconditioner, study.solver.preconditioner),
            (self.backend, study.solver.backend),
            (self.reference, study.output.reference_potential),
        ):
            if combo.findText(value) < 0:
                combo.addItem(value)
            combo.setCurrentText(value)
        self._loading = False

    def _commit(self) -> None:
        study = self.window.session.study
        if study is None or self._loading:
            return
        self.window.checkpoint("ソルバ設定を変更")
        try:
            study.solver.tolerance = float(self.tolerance.text())
        except ValueError:
            self.tolerance.setText(f"{study.solver.tolerance:g}")
        study.solver.max_iterations = self.max_iterations.value()
        study.solver.preconditioner = self.preconditioner.currentText()
        study.solver.backend = self.backend.currentText()
        study.output.reference_potential = self.reference.currentText()


class ResultPage(QWidget):
    """結果の表示設定（SPEC §6.2 の手順 6）。"""

    def __init__(self, window: MainWindow) -> None:
        super().__init__()
        self.window = window
        self._loading = False
        #: 稜線の既定を決めたときのモデル（開き直したときだけ入れ直す）
        self._mesh_key = None

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
        self.nodes = QCheckBox("節点を表示")
        self.nodes.setToolTip(
            "節点を点で描きます。細かいメッシュでは潰れるので、拡大して使ってください。"
        )
        layout.addWidget(self.edges)
        layout.addWidget(self.nodes)

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
        self.nodes.toggled.connect(self._refresh)
        self.streamlines.toggled.connect(self._refresh)
        self.slice_position.valueChanged.connect(self._refresh)
        self.iso_level.valueChanged.connect(self._refresh)
        self.n_seeds.valueChanged.connect(self._refresh)

    def reload(self) -> None:
        session = self.window.session
        self._loading = True
        self._apply_size_defaults()
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

    def _apply_size_defaults(self) -> None:
        """モデルを開き直したとき、規模に合わせて稜線の既定を決める。

        細かいメッシュで稜線を全部描くと**線が潰れて灰色の塊になり、
        形がかえって見えなくなる**。小さいモデルでは出したほうが分かりやすい。
        ユーザが切り替えたあとは触らない（開き直すまで保つ）。
        """
        mesh = self.window.session.mesh
        source = None if mesh is None else (mesh.source, mesh.n_nodes)
        if source == self._mesh_key:
            return
        self._mesh_key = source
        if mesh is None:
            return
        try:
            n_triangles = len(mesh.surface_triangles())
        except Exception:  # noqa: BLE001 - 抽出できないメッシュは既定のまま
            return
        self.edges.setChecked(n_triangles <= EDGE_LIMIT)
        self.nodes.setChecked(False)

    def show_result_field(self) -> None:
        """解いた直後に、色分けを電流密度へ切り替える。

        形だけ見ていた状態から結果の表示へ自然に移る。すでに別の量を選んで
        いるときは、その選択を尊重して何もしない。
        """
        if self.field.currentData() != SHAPE_FIELD:
            return
        index = self.field.findData(DEFAULT_RESULT_FIELD)
        if index >= 0:
            self.field.setCurrentIndex(index)

    def options(self) -> dict:
        return {
            "field": self.field.currentData() or SHAPE_FIELD,
            "mode": self.mode.currentData() or "volume",
            "cmap": self.cmap.currentText(),
            "edges": self.edges.isChecked(),
            "nodes": self.nodes.isChecked(),
            "slice_normal": self.slice_normal.currentText(),
            "slice_position": self.slice_position.value() / 100.0,
            "iso_level": self.iso_level.value() / 100.0,
            "streamlines": self.streamlines.isChecked(),
            "n_seeds": self.n_seeds.value(),
        }

    def _refresh(self) -> None:
        if not self._loading:
            self.window.refresh_view()
