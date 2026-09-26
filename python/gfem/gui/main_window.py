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
ここはその表示と編集だけを担当する。右側のプロパティパネルの各ページは `pages.py`。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
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

from .. import read_nastran
from .busy import BusyOverlay, busy_scope, run_in_thread
from .pages import ElectrodePage, MeshPage, RegionPage, ResultPage, SolverPage
from .solver_thread import SolverThread
from .state import (
    SHAPE_FIELD,
    REGION_STATS_COLUMNS,
    Session,
    field_by_key,
)
from .widgets import ConvergencePlot, UnitSelector

__all__ = ["MainWindow"]

#: ツリーの項目種別
ROLE_KIND = Qt.ItemDataRole.UserRole
ROLE_KEY = Qt.ItemDataRole.UserRole + 1

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

#: これを超えたら「大きなモデル」とみなし、ローディングを待たずに出す。
#:
#: 実測（RTX A6000 / 20 コア）で、35.9 万節点だと領域を選び直すだけで 0.98 秒、
#: 1.5 万節点なら 0.05 秒。10 万節点あたりが「待たされたと感じる」境目にあたる。
BIG_MODEL_NODES = 100_000



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
    #: 置く前の電極の輪郭に付ける名前
    OUTLINE_ACTOR = "patch-outline"

    def screen_right(self) -> list[float]:
        """いまの視点での画面の横方向（世界座標の単位ベクトル）。

        矩形の電極を「画面に対してまっすぐ」置くために、幅の辺の向きとして使う
        （SPEC §11.1）。ケースファイルにはこのベクトルそのものが残るので、
        カメラを動かしても置いた電極の向きは変わらない。
        """
        camera = self.plotter.renderer.GetActiveCamera()
        return Session.screen_right(camera.GetDirectionOfProjection(), camera.GetViewUp())

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

    def show_paint_preview(self, polydata, erase: bool, outline=None) -> None:
        """これから塗られる（消される）範囲を薄く出す。

        `outline` を渡すと、置こうとしている電極の形の輪郭（閉じた折れ線。
        `Session.outline_polydata`）も重ねる。面は 1 枚単位で入るか外れるかなので、面だけでは本来の形が読めない。
        """
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
        if outline is not None and outline.n_points:
            self.plotter.add_mesh(
                outline,
                color="#b34700",
                line_width=3,
                name=self.OUTLINE_ACTOR,
                reset_camera=False,
                pickable=False,
            )
        else:
            self.plotter.remove_actor(self.OUTLINE_ACTOR, render=False)
        self.plotter.render()

    def clear_paint_preview(self) -> None:
        shown = [
            name
            for name in (self.PREVIEW_ACTOR, self.OUTLINE_ACTOR)
            if self.plotter.actors.get(name) is not None
        ]
        for name in shown:
            self.plotter.remove_actor(name, render=False)
        if shown:
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

        # **体積メッシュではなく外表面を描く**（SPEC §6.3）。
        # 頭部モデルで 2.7 秒 → 0.09 秒。しかも法線が付くので形が読める。
        # 断面・クリップ・等値面が空だったときの代わりの表示もこれで足りる
        surface = self.session.display_surface(scalars)
        target = surface
        if mode in ("slice", "clip"):
            # 切り口は Rust 側で作る。体積メッシュは組み立てない（SPEC §6.3）
            scope.step("断面を切り出しています")
            target = self.session.section(
                mode, options["slice_normal"], options["slice_position"], scalars
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
            target = surface
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
                # 体表面に出ていない領域。境界面を Rust 側で取り出す（体積メッシュは作らない）
                region = self.session.region_boundary(highlight_pid)
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
