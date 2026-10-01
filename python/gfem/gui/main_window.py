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

import textwrap
from pathlib import Path

from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtGui import QAction, QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
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

import numpy as np
import pyvista as pv

from .. import read_nastran
from ..i18n import LANGUAGES, language, set_language, tr
from .busy import BusyOverlay, busy_scope, run_in_thread
from .pages import ElectrodePage, MeshPage, RegionPage, ResultPage, SolverPage
from .solver_thread import SolverThread
from .state import (
    HIDDEN,
    SHAPE_FIELD,
    TRANSPARENT,
    VISIBLE,
    REGION_STATS_COLUMNS,
    STALE_COLOR,
    Session,
    field_by_key,
    stale_result_message,
)
from .widgets import ConvergencePlot, PartTree, UnitSelector, part_view_icon

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

#: 電極の縁の色（役割ごと）。選んでいる電極は面を赤で塗り、ほかの電極は縁だけを描く
ELECTRODE_OUTLINE_COLORS = {"anode": "#c62828", "cathode": "#1e56b8"}

#: 体表面に出ていない領域を選んだとき、外側をどれだけ透かすか
BURIED_OPACITY = 0.25

#: カラーバーの体裁。横置きだと窓が狭いときに数字が重なって読めなくなる。
#: 位置と大きさは**カラーバー専用の描画領域**（`SCALAR_BAR_STRIP`）に対する割合
SCALAR_BAR = {
    "vertical": True,
    "position_x": 0.18,
    "position_y": 0.1,
    "width": 0.22,
    "height": 0.72,
    "n_labels": 5,
    "fmt": "%.3g",
    "title_font_size": 14,
    "label_font_size": 12,
    "unconstrained_font_size": True,
    "color": "#1b1e21",
}

#: カラーバーを出している間、3D ビューの右端のこの幅 [px] をカラーバー専用にする。
#: 以前はモデルと同じ描画領域に重ねていて、モデルの右側にかぶり、途中の目盛りが
#: 暗い色のモデルの上で読めなかった（窓が狭いとバーがモデルを横切った）
SCALAR_BAR_STRIP_PX = 130


def scalar_bar_title(key: str) -> str:
    """カラーバーの見出し。量と単位を 2 行に分ける（`"|J| [A/m2]"` → `"J\\n[A/m2]"`）。

    1 行だと細い帯に収まらず端が切れる。VTK の既定のフォントには日本語が無いので、
    見出しは表示量の鍵（英字）から作る。**縦棒は落とす**——VTK の既定のフォントでは
    `|` が描かれず、しかも複数行の左寄せが崩れる（実測）。大きさであることは
    結果ページの表示量の名前で分かる。
    """
    name, bracket, unit = key.partition(" [")
    name = name.replace("|", "").strip()
    # 帯に収まる長さ（"power density" の 13 文字）を超える名前は語の切れ目で折り返す
    # （"electrode current density" は 1 行だと "density" が帯の外に切れた）
    name = "\n".join(textwrap.wrap(name, width=SCALAR_BAR_TITLE_WIDTH)) or name
    return f"{name}\n[{unit}" if bracket else name


#: カラーバーの見出しの 1 行の文字数の上限
SCALAR_BAR_TITLE_WIDTH = 13

#: 値で色分けしているときの陰影。**色は値を表すので、陰影で色を変えすぎない。**
#: 形状表示と同じ強い陰影だと、光の当たらない側面が真っ黒に近くなり、同じ値でも
#: 面の向きで違う色に見えた。環境光を足して暗くなりすぎないようにし、形が読める
#: だけの陰影を残す
SCALAR_LIGHTING = {"ambient": 0.45, "diffuse": 0.55, "specular": 0.0}

#: 背景。うっすら階調をつけると、白いモデルの輪郭が背景に溶けない
BACKGROUND_BOTTOM = "#e9edf2"
BACKGROUND_TOP = "#fdfdfe"

#: これを超えたら「大きなモデル」とみなし、ローディングを待たずに出す。
#:
#: 実測（RTX A6000 / 20 コア）で、35.9 万節点だと領域を選び直すだけで 0.98 秒、
#: 1.5 万節点なら 0.05 秒。10 万節点あたりが「待たされたと感じる」境目にあたる。
BIG_MODEL_NODES = 100_000

#: 半透明にした領域・電極の不透明度（ツリーの見せ方の列。SPEC §6.3「パーツごとの表示」）
PART_TRANSPARENT_OPACITY = 0.3
PART_TRANSPARENT_LINE_OPACITY = 0.35

#: 断面スライスを出している間、体の外形をこの不透明度で薄く重ねる。板だけだと、
#: どこを切った断面なのか分からなかった
SECTION_CONTEXT_OPACITY = 0.12

#: 流線を重ねている間の外表面の不透明度。流線は体の中を通るので、外表面が
#: 不透明だと 1 本も見えない（以前は「重ねる」を入れても何も変わらなかった）
STREAMLINE_SURFACE_OPACITY = 0.3


def draw_on_top(actor) -> None:
    """体表面にぴったり重ねる面（電極・予告）を、奥行きの比較で必ず手前に出す。

    同じ高さの面どうしはどちらが描かれるかが決まらず（z-fighting）、塗りの予告が
    上面では体表面に負けて見えず、側面でだけ見えていた（2026-09 に実機で見つけた）。
    """
    mapper = actor.GetMapper()
    mapper.SetResolveCoincidentTopologyToPolygonOffset()
    mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-2.0, -2.0)


#: 編集メニューの「元に戻す: …」に出す操作名の長さの上限（文字数）。全文はツールチップに出す
HISTORY_LABEL_LIMIT = 24


def elide(text: str, limit: int) -> str:
    """`limit` 文字を超えたら末尾を「…」にして切る。"""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def status_line(message: str, limit: int = 200) -> str:
    """ステータスバーに出す 1 行（ログの複数行のメッセージから要点を取る）。

    「エラー:」「求解失敗:」のように 1 行目が見出しだけのメッセージは、2 行目の
    中身と続けて出す。以前は 1 行目だけを出していて、ステータスバーに
    「エラー:」とだけ残り、何が起きたのか分からなかった。
    """
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    if not lines:
        return ""
    head = lines[0]
    if head.endswith((":", "：")) and len(lines) > 1:
        head = f"{head} {lines[1]}"
    return head[:limit]



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
        #: 視点を合わせたメッシュ（`Session.mesh_serial`）。開き直したときだけ合わせ直す
        self._framed_mesh: int | None = None
        #: 合わせたときの描画領域と視点。帯の出入りで合わせ直してよいかの判断に使う
        self._framed_viewport: tuple | None = None
        self._framed_camera: tuple | None = None
        #: 断面・クリップを選ぶ前の視点（やめたときに戻す）
        self._camera_before_section: tuple | None = None
        #: 閉じるときに、求解中・保存していない変更を確かめるか（テストでは切る）
        self.ask_before_closing = True

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
        return tr("{nodes:,} nodes / {elements:,} elements", nodes=mesh.n_nodes, elements=mesh.n_elements)

    # --- UI の構築 ---------------------------------------------------------

    def _build_ui(self) -> None:
        from pyvistaqt import QtInteractor

        self.tree = PartTree()
        # 2 列目は領域・電極の見せ方（表示 / 半透明 / 非表示）。目のアイコンを押すと切り替わる
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels([tr("Model"), ""])
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.viewClicked.connect(self._cycle_part_view)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._part_view_menu)
        self.tree.setMinimumWidth(260)
        # 領域は Ctrl / Shift + クリックで複数選べる（σ をまとめて入れる）
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.itemSelectionChanged.connect(self._on_tree_selection_changed)

        self.plotter = QtInteractor(self)
        # **ライトを明示的に置く。** pyvistaqt の描画器は光源を 1 つも持たず
        # （自動生成も切れている）、そのままだと陰影が付かないので、
        # どんな形も**のっぺりした影絵**になる。単体の Plotter と同じ
        # 三点照明キットを入れると、白一色でも形が読める
        self.plotter.enable_lightkit()
        self.plotter.set_background(BACKGROUND_BOTTOM, top=BACKGROUND_TOP)
        # 半透明の領域が何枚も重なっても前後を正しく描く（半透明の物が無ければ効かない）
        self.plotter.enable_depth_peeling()
        # カラーバー専用の描画領域（`_make_room_for_scalar_bar`）。背景はモデル側とそろえ、
        # マウス操作は受けない（受けると、ここでドラッグしたときに何も無い視点が回る）
        from vtkmodules.vtkRenderingCore import vtkRenderer

        self._scalar_bar_renderer = vtkRenderer()
        self._scalar_bar_renderer.SetViewport(0.0, 0.0, 0.0, 0.0)
        self._scalar_bar_renderer.SetInteractive(False)
        self._scalar_bar_renderer.GradientBackgroundOn()
        self._scalar_bar_renderer.SetBackground(pv.Color(BACKGROUND_BOTTOM).float_rgb)
        self._scalar_bar_renderer.SetBackground2(pv.Color(BACKGROUND_TOP).float_rgb)
        self.plotter.ren_win.AddRenderer(self._scalar_bar_renderer)

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
        self.stats_table.setHorizontalHeaderLabels([tr(c) for c in REGION_STATS_COLUMNS])
        self.stats_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.stats_table.verticalHeader().setVisible(False)  # 行番号は PID の列と紛らわしい

        self.bottom = QTabWidget()
        self.bottom.addTab(self.log_view, tr("Log"))
        self.bottom.addTab(self.convergence, tr("Convergence"))
        self.bottom.addTab(self.checks_view, tr("Conservation checks"))
        self.bottom.addTab(self.stats_table, tr("Region statistics"))
        self.bottom.setMaximumHeight(240)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(splitter, stretch=1)
        layout.addWidget(self.bottom)
        self.setCentralWidget(central)
        self.statusBar().showMessage(tr("Open a model to begin"))
        # 結果が古い条件のものなら、ステータスバーの右端に出し続ける（一時メッセージに消されない）
        self.stale_status = QLabel("")
        self.stale_status.setStyleSheet(f"color: {STALE_COLOR}; font-weight: bold;")
        self.stale_status.setVisible(False)
        self.statusBar().addPermanentWidget(self.stale_status)
        self._result_item = None

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu(tr("&File"))
        self._add_action(file_menu, tr("Open model..."), self.open_mesh_dialog, "Ctrl+O")
        self._add_action(file_menu, tr("Open case..."), self.open_case_dialog)
        file_menu.addSeparator()
        self._add_action(file_menu, tr("Save case..."), self.save_case_dialog, "Ctrl+S")
        self._add_action(file_menu, tr("Export results to VTU..."), self.export_vtu_dialog)
        self._add_action(
            file_menu, tr("Export region statistics to CSV..."), self.export_region_stats_dialog
        )
        self._add_action(
            file_menu, tr("Export convergence history to CSV..."), self.export_convergence_dialog
        )
        self._add_action(
            file_menu, tr("Export run summary to JSON..."), self.export_summary_dialog
        )
        self._add_action(file_menu, tr("Screenshot..."), self.screenshot_dialog)
        file_menu.addSeparator()
        self._add_action(file_menu, tr("Exit"), self.close)

        edit_menu = self.menuBar().addMenu(tr("&Edit"))
        # 「元に戻す: …」は長いと切って出すので、全文はツールチップで見せる
        edit_menu.setToolTipsVisible(True)
        self.action_undo = self._add_action(edit_menu, tr("Undo"), self.undo, "Ctrl+Z")
        self.action_redo = self._add_action(edit_menu, tr("Redo"), self.redo, "Ctrl+Y")
        self._refresh_history_actions()

        run_menu = self.menuBar().addMenu(tr("&Run"))
        self._add_action(run_menu, tr("Check consistency"), self.check_consistency, "Ctrl+K")
        self._add_action(run_menu, tr("Solve"), self.start_solve, "F5")
        self.action_cancel = self._add_action(run_menu, tr("Cancel"), self.cancel_solve)
        self.action_cancel.setEnabled(False)
        run_menu.addSeparator()
        self._add_action(run_menu, tr("Parameter study..."), self.open_sweep_dialog)

        electrode_menu = self.menuBar().addMenu(tr("E&lectrodes"))
        self._add_action(electrode_menu, tr("Add electrode"), self.add_electrode)
        self._add_action(electrode_menu, tr("Delete selected electrode"), self.remove_selected_electrode)

        # 表示言語（SPEC §6.6）。選ぶと設定に残り、次に起動したときから効く
        view_menu = self.menuBar().addMenu(tr("&View"))
        language_menu = view_menu.addMenu(tr("Language"))
        self.language_actions = {}
        for code, name in LANGUAGES:
            action = QAction(name, self)
            action.setCheckable(True)
            action.setChecked(code == language())
            action.triggered.connect(lambda _checked=False, c=code: self.choose_language(c))
            language_menu.addAction(action)
            self.language_actions[code] = action

    def _add_action(self, menu, text, slot, shortcut: str | None = None) -> QAction:
        action = QAction(text, self)
        action.triggered.connect(slot)
        if shortcut:
            action.setShortcut(shortcut)
        menu.addAction(action)
        return action

    # --- 表示言語（SPEC §6.6）----------------------------------------------

    def choose_language(self, code: str) -> None:
        """表示言語を選んだ。設定に保存し、次に起動したときから反映する。

        開いているウィンドウの文言は作ったときに訳してあるので、その場では変えない
        （全ページを作り直す仕組みに見合わない）。知らせは選んだ言語で出す。
        """
        from .language import save_language

        for c, action in self.language_actions.items():
            action.setChecked(c == code)
        if code == language():
            save_language(code)
            return
        save_language(code)
        current = language()
        set_language(code)
        try:
            title, text = tr("Language"), tr(
                "The display language will change the next time you start GenericFEM."
            )
        finally:
            set_language(current)
        QMessageBox.information(self, title, text)

    # --- ログ ---------------------------------------------------------------

    def log(self, message: str, status: bool = True) -> None:
        """ログに足す。`status` が真ならステータスバーにも要点を 1 行で出す。"""
        self.log_view.appendPlainText(message)
        if status:
            self.statusBar().showMessage(status_line(message))

    def show_error(self, exc: Exception) -> None:
        text = str(exc)
        self.log(tr("Error:") + "\n" + text)
        QMessageBox.critical(self, tr("Error"), text)

    # --- ファイル操作 -------------------------------------------------------

    def open_mesh_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Open model"), "", tr("Nastran bulk data (*.bdf *.nas);;All files (*)")
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
                tr("Checking the size"),
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
                tr("Reading the mesh"),
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
        self.log(tr("Loaded: {path}", path=path))
        self.log(self.session.mesh_summary(), status=False)
        self._leave_edited_item()
        self.refresh_all()
        self._refresh_history_actions()

    def open_case_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Open case"), "", tr("Case files (*.yaml *.yml);;All files (*)")
        )
        if path:
            self.open_case(path)

    def open_case(self, path: str | Path) -> None:
        try:
            self.run_busy(
                tr("Reading the case"),
                lambda: self.session.open_case(path),
                Path(path).name,
            )
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc)
            return
        self.convergence.clear()
        self.checks_view.clear()
        self.log(tr("Opened the case: {path}", path=path))
        self.log(self.session.mesh_summary(), status=False)
        self._leave_edited_item()
        self.refresh_all()
        self._refresh_history_actions()

    def save_case_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, tr("Save case"), "case.yaml", tr("Case files (*.yaml)")
        )
        if path:
            self.save_case(path)

    def save_case(self, path: str | Path) -> bool:
        """ケースを保存する。保存できたら True。"""
        try:
            self.session.save_case(path)
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc)
            return False
        self.log(tr("Saved the case: {path}", path=path))
        return True

    def export_vtu_dialog(self) -> None:
        if self.session.solution is None:
            QMessageBox.information(self, tr("Export VTU"), tr("Solve first (Run → Solve, F5)."))
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("Export to VTU"), "result.vtu", "VTU (*.vtu)")
        if path:
            self.export_vtu(path)

    def export_vtu(self, path: str | Path) -> None:
        try:
            self.run_busy(
                tr("Writing the VTU"),
                lambda: self.session.solution.write_vtu(path),
                Path(path).name,
            )
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc)
            return
        self.log(tr("Wrote the VTU: {path}", path=path))
        self._warn_if_stale()

    def export_region_stats_dialog(self) -> None:
        self._export_dialog(
            tr("Export region statistics"),
            "region_stats.csv",
            "CSV (*.csv)",
            self.session.write_region_stats_csv,
        )

    def export_convergence_dialog(self) -> None:
        self._export_dialog(
            tr("Export convergence history"),
            "convergence.csv",
            "CSV (*.csv)",
            self.session.write_convergence_csv,
        )

    def export_summary_dialog(self) -> None:
        self._export_dialog(
            tr("Export run summary"),
            "summary.json",
            "JSON (*.json)",
            self.session.write_summary_json,
        )

    def _export_dialog(self, title: str, default: str, filters: str, writer) -> None:
        if self.session.solution is None:
            QMessageBox.information(self, title, tr("Solve first (Run → Solve, F5)."))
            return
        path, _ = QFileDialog.getSaveFileName(self, title, default, filters)
        if not path:
            return
        try:
            self.run_busy(tr("Writing"), lambda: writer(path), Path(path).name)
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc)
            return
        self.log(tr("Wrote: {path}", path=path))
        self._warn_if_stale()

    def screenshot_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, tr("Screenshot"), "view.png", "PNG (*.png)")
        if path:
            self.screenshot(path)

    def screenshot(self, path: str | Path) -> None:
        # 画面の取り込みは OpenGL コンテキストを持つメインスレッドでしかできない
        with self.busy(tr("Saving the screenshot"), Path(path).name):
            self.plotter.screenshot(str(path))
        self.log(tr("Saved the screenshot: {path}", path=path))

    # --- パラメータスタディ（SPEC §8 M7）-------------------------------------

    def open_sweep_dialog(self) -> None:
        """パラメータを振って一括で解くダイアログを開く。"""
        if self.session.study is None:
            self.log(tr("Open a model or a case first"))
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
        with self.busy(tr("Drawing the electrode faces")):
            self._draw_electrode_highlight(name)
            self._arm_picking()

    def _draw_electrode_highlight(self, name: str | None) -> None:
        self.plotter.remove_actor(self.ELECTRODE_ACTOR, render=False)
        faces = self.session.electrode_polydata(name) if name else None
        if faces is not None and faces.n_cells:
            actor = self.plotter.add_mesh(
                faces,
                color="red",
                opacity=1.0,
                name=self.ELECTRODE_ACTOR,
                reset_camera=False,
                pickable=False,
            )
            draw_on_top(actor)
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
            tr(
                "Click the body surface to make electrode faces "
                "(Shift + drag paints / Alt + drag erases / drag rotates)"
            )
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
        draw_on_top(
            self.plotter.add_mesh(
                polydata,
                color="#2a5adb" if erase else "#ff8c1a",
                opacity=0.55,
                name=self.PREVIEW_ACTOR,
                reset_camera=False,
                pickable=False,
            )
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
        self._refresh_after_history()
        self.log(tr("Undone: {what}", what=label))

    def redo(self) -> None:
        label = self.session.redo()
        if not label:
            return
        self._refresh_after_history()
        self.log(tr("Redone: {what}", what=label))

    def apply_element_order(self) -> None:
        """解析条件の要素の次数に合わせてメッシュを入れ替え、描き直す（SPEC §2.3）。

        2 次に上げるのは大きなモデルで時間がかかるので、ローディングを出して別スレッドで行う。
        """
        try:
            changed = self.run_busy(
                tr("Changing the element order"), self.session.sync_element_order
            )
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc)
            return
        if changed:
            mesh = self.session.mesh
            self.log(
                tr(
                    "Element order {order}: {n:,} nodes",
                    order=mesh.element_order,
                    n=mesh.n_nodes,
                )
            )
            self.convergence.clear()
            self.checks_view.clear()
            self.refresh_all()
            self.page_solver.reload()
        self._refresh_history_actions()

    def _refresh_after_history(self) -> None:
        """Undo / Redo の後。ツリーと、開いているページを戻した条件で読み直す。

        以前は `refresh_all` だけで、開いているページは戻す前の値を出したまま
        （電流を戻しても欄は古い値）、戻した操作で消えた電極のページも出たままだった。
        要素の次数が戻ったときは、メッシュも戻した次数に入れ替える。
        """
        changed = False
        if self.session.element_order_out_of_sync():
            try:
                changed = self.run_busy(
                    tr("Changing the element order"), self.session.sync_element_order
                )
            except Exception as exc:  # noqa: BLE001
                self.show_error(exc)
        if changed:
            self.convergence.clear()
            self.checks_view.clear()
            self.page_mesh.reload()
        with self.busy(tr("Building the view")):
            self.refresh_tree()
            self.page_result.reload()
            self.refresh_stats_table()
            self._show_selection()
        self._refresh_history_actions()

    def _show_selection(self) -> None:
        """ツリーでいま選ばれている項目のページを開き直し、3D ビューを描き直す。

        選んでいた項目が無くなっていたら（削除した電極、Undo で消えた電極）、
        メッシュのページに戻る。**消えた電極のページを出したままにしない**——以前は
        削除した電極の名前と設定が右のパネルに残り、赤い強調表示も消えなかった。
        """
        if self.tree.selectedItems():
            self._on_tree_selection_changed()
            return
        self._leave_edited_item()
        self.refresh_view()

    def _leave_edited_item(self) -> None:
        """編集中の項目から離れてメッシュのページに戻る（項目が消えたとき・開き直したとき）。"""
        # ツリーの選択も外す。残すと、開き直したモデルで同じ PID の項目が選ばれたまま
        # （ページはメッシュなのにツリーは領域）になる
        self.tree.blockSignals(True)
        try:
            self.tree.clearSelection()
        finally:
            self.tree.blockSignals(False)
        self.page_electrode.unbind()
        self.page_mesh.reload()
        self.pages.setCurrentWidget(self.page_mesh)

    def checkpoint(self, label: str) -> None:
        """編集を始める前に呼ぶ。同じ名前の編集が続いている間はまとめられる。"""
        self.session.checkpoint(label)
        self._refresh_history_actions()

    def _refresh_history_actions(self) -> None:
        """メニューに「何を戻すのか」を出す。押せるかどうかもここで決める。"""
        undo, redo = self.session.can_undo(), self.session.can_redo()
        self.action_undo.setEnabled(undo)
        self.action_redo.setEnabled(redo)
        for action, verb, enabled, label in (
            (self.action_undo, tr("Undo"), undo, self.session.undo_label()),
            (self.action_redo, tr("Redo"), redo, self.session.redo_label()),
        ):
            # **操作名は切って出す。** 記録の名前には選んだ PID が全部入る（62 領域を
            # まとめて選ぶと「領域 1, 2, 3, …, 62 の設定を変更」）ので、そのまま出すと
            # メニューが際限なく横に伸びた。記録の名前そのものは短くしない（同じ名前の
            # 編集を 1 手にまとめる判定に使っている）。全文はツールチップとステータスバーに
            full = f"{verb}: {label}" if enabled else verb
            action.setText(
                f"{verb}: {elide(label, HISTORY_LABEL_LIMIT)}" if enabled else verb
            )
            action.setToolTip(full)
            action.setStatusTip(full if enabled else "")

    # --- 電極 ---------------------------------------------------------------

    def add_electrode(self) -> None:
        if self.session.study is None:
            return
        existing = {e.name for e in self.session.study.electrodes}
        index = len(existing) + 1
        while f"E{index}" in existing:
            index += 1
        role = "anode" if not existing else "cathode"
        self.checkpoint(tr("Add electrode {name}", name=f"E{index}"))
        self.session.add_electrode(f"E{index}", role=role)
        self.refresh_tree()
        self.log(tr("Added electrode {name} ({role})", name=f"E{index}", role=role))
        # 追加した電極を選んで、そのページを開く。以前は選択が前の項目のままで、
        # 追加した電極をツリーで探して選び直す必要があった
        self.select_item("electrode", f"E{index}")

    def select_item(self, kind: str, key) -> None:
        """ツリーの項目を選ぶ（選択が変わるので、その項目のページが開く）。"""
        for item in self._tree_items():
            if self._item_key(item) == (kind, key):
                self.tree.setCurrentItem(item)
                return

    def remove_selected_electrode(self) -> None:
        item = self.tree.currentItem()
        if item is None or item.data(0, ROLE_KIND) != "electrode":
            return
        name = item.data(0, ROLE_KEY)
        self.checkpoint(tr("Delete electrode {name}", name=name))
        self.session.remove_electrode(name)
        self.refresh_tree()
        self._show_selection()  # 消した電極は選び直せないので、メッシュのページに戻る
        self.log(tr("Deleted electrode {name}", name=name))

    # --- 実行 ---------------------------------------------------------------

    def check_consistency(self) -> list[str] | None:
        """整合性チェック（SPEC §6.2 の手順 4）。不備があれば全部を一覧で出して None。"""
        try:
            problems, warnings = self.run_busy(
                tr("Checking consistency"), self.session.problems
            )
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc)
            return None
        if problems:
            self._show_problems(problems)
            return None
        if warnings:
            self.log(tr("Consistency check: {n} warnings", n=len(warnings)))
            for w in warnings:
                self.log("  " + tr("[Warning] {text}", text=w))
        else:
            self.log(tr("Consistency check: no problems"))
        for info in self.session.electrode_info():
            self.log(
                "  "
                + tr(
                    "Electrode {name} ({role}): {n} faces, area {mm2:.1f} mm², "
                    "current density {j:.4g} A/m²",
                    name=info["name"], role=info["role"], n=info["n_faces"],
                    mm2=info["area"] * 1e6, j=info["current_density"],
                )
            )
        return warnings

    def _show_problems(self, problems: list[str]) -> None:
        """解く前に直すところを、全部まとめて見せる（ログにも残す）。

        以前は 1 件目だけをダイアログに出していて、直しては F5 を押すことを
        不備の数だけ繰り返した。
        """
        self.log(tr("Consistency check: {n} things to fix", n=len(problems)))
        for p in problems:
            self.log("  " + p.replace("\n", "\n  "))
        numbered = "\n\n".join(f"{i}. {p}" for i, p in enumerate(problems, start=1))
        QMessageBox.warning(
            self,
            tr("Consistency check"),
            tr("Fix {n} things before solving.", n=len(problems)) + "\n\n" + numbered,
        )

    def start_solve(self) -> None:
        if self.solver is not None and self.solver.isRunning():
            return
        try:
            problems, _warnings = self.run_busy(
                tr("Checking consistency"), self.session.problems
            )
            if problems:
                self._show_problems(problems)
                return
            case, study = self.run_busy(tr("Assembling the analysis"), self.session.prepare_solve)
            case.validate()
        except Exception as exc:  # noqa: BLE001
            self.show_error(exc)
            return

        self.convergence.clear()
        self.convergence.set_tolerance(self.session.study.solver.tolerance)
        self.bottom.setCurrentWidget(self.convergence)
        self.log(tr("Started solving"))
        self.action_cancel.setEnabled(True)

        self.solver = SolverThread(case, study, self)
        self.solver.progressed.connect(self._on_progress)
        self.solver.succeeded.connect(self._on_solved)
        self.solver.failed.connect(self._on_solve_failed)
        self.solver.cancelled.connect(self._on_solve_cancelled)
        self.solver.finished.connect(lambda: self.action_cancel.setEnabled(False))
        self.solver.start()

    def cancel_solve(self) -> None:
        if self.solver is not None:
            self.solver.cancel()
            self.log(tr("Requested cancellation"))

    def _on_progress(self, iteration: int, residual: float) -> None:
        self.statusBar().showMessage(
            tr("Iteration {n}   relative residual {r:.3e}", n=iteration, r=residual)
        )
        self.convergence.append(residual)

    def _on_solved(self, solution) -> None:
        # solve 自体はワーカースレッドだが、結果の取り込み（領域統計・表示用データの
        # 作り直し）はメインスレッドで、規模によっては数秒かかる
        with self.busy(tr("Loading the results")) as scope:
            self._take_solution(scope, solution)

    def _take_solution(self, scope, solution) -> None:
        self.session.set_solution(solution, self.solver.study)
        self.convergence.set_history(solution.history)
        self.checks_view.setPlainText(self.session.result_summary())
        scope.step(tr("Computing region statistics"))
        self.refresh_stats_table()
        self.bottom.setCurrentWidget(self.checks_view)
        self.log(
            tr(
                "Solved: {n} iterations, relative residual {r:.3e}, R = {ohm:.4g} Ω",
                n=solution.iterations, r=solution.residual, ohm=solution.impedance,
            )
        )
        # 完全電極モデルの電極は、板の電位と縁の集中を残す（SPEC §2.5）
        for e in solution.electrodes:
            if e["contact_impedance"] is not None:
                self.log(
                    "  "
                    + tr(
                        "{name}: plate {u:.6g} V, current density into the skin max {peak:.4g} A/m² "
                        "(mean {mean:.4g} A/m²)",
                        name=e["name"],
                        u=e["potential"],
                        peak=e["peak_current_density"],
                        mean=abs(e["mean_current_density"]),
                    )
                )
        for w in solution.warnings:
            self.log("  " + tr("[Warning] {text}", text=w))
        self.refresh_tree()
        self.page_result.reload()
        self.page_result.show_result_field()
        self.refresh_view()

    def _on_solve_failed(self, message: str) -> None:
        self.log(tr("Solve failed:") + "\n" + message)
        QMessageBox.warning(self, tr("Solve"), message)

    def _on_solve_cancelled(self) -> None:
        """利用者が中断した。**失敗ではない**ので警告のダイアログは出さない。

        以前は失敗と同じ扱いで、「求解失敗」とログに残り、警告のダイアログも出ていた。
        """
        kept = (
            tr("the previous result is kept")
            if self.session.solution is not None
            else tr("there is no result")
        )
        self.log(tr("Solving was cancelled ({kept})", kept=kept))

    # --- 閉じる -------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt の規約
        """求解中なら止めてから、保存していない変更があれば確かめてから閉じる。

        以前は何も確かめずに閉じ、求解中だとスレッドが走ったままウィンドウが
        破棄されてプロセスが異常終了していた（終了コード 127）。
        """
        if self.ask_before_closing and not self._confirm_close():
            event.ignore()
            return
        self._stop_solver()
        super().closeEvent(event)

    def _confirm_close(self) -> bool:
        """閉じてよいか利用者に確かめる。やめるなら False。"""
        buttons = QMessageBox.StandardButton
        if self.solver is not None and self.solver.isRunning():
            answer = QMessageBox.question(
                self,
                tr("Solving"),
                tr("A solve is running. Cancel it and exit?"),
                buttons.Yes | buttons.No,
                buttons.No,
            )
            if answer != buttons.Yes:
                return False
        if self.session.is_modified():
            answer = QMessageBox.question(
                self,
                tr("Unsaved changes"),
                tr("The analysis settings have unsaved changes. Save them to a case file?"),
                buttons.Save | buttons.Discard | buttons.Cancel,
                buttons.Save,
            )
            if answer == buttons.Cancel:
                return False
            if answer == buttons.Save:
                path, _ = QFileDialog.getSaveFileName(
                    self, tr("Save case"), "case.yaml", tr("Case files (*.yaml)")
                )
                if not path or not self.save_case(path):
                    return False
        return True

    def _stop_solver(self) -> None:
        """走っている求解を止め、スレッドが終わるまで待つ（次の反復で止まる）。"""
        if self.solver is None or not self.solver.isRunning():
            return
        self.solver.cancel()
        self.run_busy(tr("Stopping the calculation"), self.solver.wait)

    # --- 表示 ---------------------------------------------------------------

    def refresh_all(self) -> None:
        # 全体をひとまとめに囲む。中の refresh_view も同じローディングを使うので、
        # 途中で消えて出直すような瞬きが起きない
        with self.busy(tr("Building the view")):
            self.refresh_tree()
            # 以前はメッシュページを読み直しておらず、起動直後と読み込み直後は
            # 右のパネルが空のままだった（ツリーをクリックするまで何も出ない）
            self.page_mesh.reload()
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
        # 編集はほぼすべてここを通るので、結果が古くなったかもここで見直す
        selected = self._selected_keys()
        current = self.tree.currentItem()
        current_key = None if current is None else self._item_key(current)
        self._build_tree()
        self._restore_selection(selected, current_key)
        self.refresh_result_status()

    @staticmethod
    def _item_key(item) -> tuple:
        return (item.data(0, ROLE_KIND), item.data(0, ROLE_KEY))

    def _selected_keys(self) -> list[tuple]:
        return [self._item_key(i) for i in self.tree.selectedItems()]

    def _restore_selection(self, keys: list[tuple], current_key: tuple | None) -> None:
        """作り直したツリーで、同じ項目を選び直す（ページは選び直さない）。

        ツリーは編集のたびに作り直すので、そのままだと選択が消える。複数の領域を
        選んで σ を入れた直後に選択が消えると、続けて直せない。
        """
        if not keys:
            return
        wanted = set(keys)
        self.tree.blockSignals(True)
        try:
            for item in self._tree_items():
                key = self._item_key(item)
                if key == current_key:
                    self.tree.setCurrentItem(item, 0, QItemSelectionModel.SelectionFlag.NoUpdate)
                if key in wanted:
                    item.setSelected(True)
        finally:
            self.tree.blockSignals(False)

    def _tree_items(self):
        """ツリーの項目を上から順に全部（入れ子も含む）。"""
        stack = [self.tree.topLevelItem(i) for i in reversed(range(self.tree.topLevelItemCount()))]
        while stack:
            item = stack.pop()
            yield item
            stack.extend(item.child(i) for i in reversed(range(item.childCount())))

    def _build_tree(self) -> None:
        self.tree.clear()
        self._result_item = None
        study = self.session.study
        if study is None:
            return

        mesh_item = QTreeWidgetItem([tr("Mesh (unit: {unit})", unit=study.length_unit)])
        mesh_item.setData(0, ROLE_KIND, "mesh")
        self.tree.addTopLevelItem(mesh_item)

        regions = QTreeWidgetItem([tr("Regions (PID)")])
        regions.setData(0, ROLE_KIND, "regions")
        for m in study.materials:
            label = f"{m.pid}"
            if m.name:
                label += f" {m.name}"
            label = (
                tr("{label}   σ = {sigma:g} S/m", label=label, sigma=m.sigma)
                if m.sigma > 0
                else tr("{label}   σ not set", label=label)
            )
            child = QTreeWidgetItem([label])
            child.setData(0, ROLE_KIND, "region")
            child.setData(0, ROLE_KEY, m.pid)
            self._show_part_view(child, self.session.region_view(m.pid))
            regions.addChild(child)
        self.tree.addTopLevelItem(regions)
        regions.setExpanded(True)

        electrodes = QTreeWidgetItem([tr("Electrodes")])
        electrodes.setData(0, ROLE_KIND, "electrodes")
        for e in study.electrodes:
            child = QTreeWidgetItem(
                [tr("{name}   {role}   {ma:g} mA", name=e.name, role=e.role, ma=e.current * 1e3)]
            )
            child.setData(0, ROLE_KIND, "electrode")
            child.setData(0, ROLE_KEY, e.name)
            self._show_part_view(child, self.session.electrode_view(e.name))
            electrodes.addChild(child)
        self.tree.addTopLevelItem(electrodes)
        electrodes.setExpanded(True)

        solver_item = QTreeWidgetItem([tr("Solver settings")])
        solver_item.setData(0, ROLE_KIND, "solver")
        self.tree.addTopLevelItem(solver_item)

        result_item = QTreeWidgetItem([tr("Results")])
        result_item.setData(0, ROLE_KIND, "result")
        self.tree.addTopLevelItem(result_item)
        self._result_item = result_item

    # --- 領域・電極ごとの見せ方（SPEC §6.3「パーツごとの表示」）---------------

    #: 目のアイコンを押したときの切り替わる順
    _NEXT_VIEW = {VISIBLE: TRANSPARENT, TRANSPARENT: HIDDEN, HIDDEN: VISIBLE}

    def _show_part_view(self, item, state: str) -> None:
        item.setIcon(PartTree.VIEW_COLUMN, part_view_icon(state))
        item.setToolTip(
            PartTree.VIEW_COLUMN,
            {
                VISIBLE: tr("Shown (click: transparent)"),
                TRANSPARENT: tr("Transparent (click: hidden)"),
                HIDDEN: tr("Hidden (click: shown)"),
            }[state],
        )

    def _part_items(self, item) -> list:
        """見せ方を変える対象。押した項目が複数選択の中なら、選んでいる同じ種類の全部。"""
        kind = item.data(0, ROLE_KIND)
        selected = self.tree.selectedItems()
        if item in selected:
            return [i for i in selected if i.data(0, ROLE_KIND) == kind]
        return [item]

    def _part_view_of(self, item) -> str:
        key = item.data(0, ROLE_KEY)
        if item.data(0, ROLE_KIND) == "region":
            return self.session.region_view(key)
        return self.session.electrode_view(key)

    def set_part_view(self, items, state: str) -> None:
        """領域・電極の見せ方を変えて描き直す（見方の好みなので Undo の対象にしない）。"""
        regions = [i.data(0, ROLE_KEY) for i in items if i.data(0, ROLE_KIND) == "region"]
        electrodes = [i.data(0, ROLE_KEY) for i in items if i.data(0, ROLE_KIND) == "electrode"]
        if not regions and not electrodes:
            return
        self.session.set_region_view(regions, state)
        self.session.set_electrode_view(electrodes, state)
        for item in items:
            if item.data(0, ROLE_KIND) in ("region", "electrode"):
                self._show_part_view(item, state)
        self.page_result.update_part_controls()
        self.refresh_view()

    def _cycle_part_view(self, item) -> None:
        """目のアイコンを押した。表示 → 半透明 → 非表示 → 表示 の順に切り替える。"""
        if item.data(0, ROLE_KIND) not in ("region", "electrode"):
            return
        self.set_part_view(self._part_items(item), self._NEXT_VIEW[self._part_view_of(item)])

    def _part_view_menu(self, pos) -> None:
        """領域・電極の右クリックメニュー（表示・半透明・非表示、これだけ表示、すべて表示）。"""
        item = self.tree.itemAt(pos)
        if item is None or item.data(0, ROLE_KIND) not in ("region", "electrode"):
            return
        kind = item.data(0, ROLE_KIND)
        targets = self._part_items(item)
        siblings = [i for i in self._tree_items() if i.data(0, ROLE_KIND) == kind]
        menu = QMenu(self)
        for state, label in (
            (VISIBLE, tr("Show")),
            (TRANSPARENT, tr("Transparent")),
            (HIDDEN, tr("Hide")),
        ):
            action = menu.addAction(part_view_icon(state), label)
            action.triggered.connect(lambda _c=False, s=state: self.set_part_view(targets, s))
        menu.addSeparator()
        only = menu.addAction(tr("Show only these") if len(targets) > 1 else tr("Show only this"))
        only.triggered.connect(lambda: self._show_only(targets, siblings))
        every = menu.addAction(tr("Show all regions") if kind == "region" else tr("Show all electrodes"))
        every.triggered.connect(lambda: self.set_part_view(siblings, VISIBLE))
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _show_only(self, targets, siblings) -> None:
        others = [i for i in siblings if i not in targets]
        self.session.set_region_view(
            [i.data(0, ROLE_KEY) for i in others if i.data(0, ROLE_KIND) == "region"], HIDDEN
        )
        self.session.set_electrode_view(
            [i.data(0, ROLE_KEY) for i in others if i.data(0, ROLE_KIND) == "electrode"], HIDDEN
        )
        for i in others:
            self._show_part_view(i, HIDDEN)
        self.set_part_view(targets, VISIBLE)

    def refresh_result_status(self) -> None:
        """結果を出したときから条件が変わっていたら、古い結果だと知らせる。

        結果は消さない（解き直すまで見比べられる）。代わりに、結果ページの帯・
        ステータスバー・ツリーの「結果」の 3 か所に、何が変わったかを出す。
        条件を戻せば（Undo を含む）知らせも消える。
        """
        changes = self.session.changes_since_solve()
        text = stale_result_message(changes)
        self.page_result.show_staleness(changes)
        self.stale_status.setText(
            tr("⚠ Results are for old conditions (F5 to solve again)") if changes else ""
        )
        self.stale_status.setToolTip(text)
        self.stale_status.setVisible(bool(changes))
        item = self._result_item
        if item is not None:
            item.setText(0, tr("Results (conditions have changed)") if changes else tr("Results"))
            item.setToolTip(0, text)
            item.setForeground(0, QBrush(QColor(STALE_COLOR)) if changes else QBrush())

    def _warn_if_stale(self) -> None:
        """古い結果を書き出したら、そうだとログに残す（書き出したもの自体は正しい）。"""
        changes = self.session.changes_since_solve()
        if changes:
            self.log(
                "  "
                + tr(
                    "[Note] The exported result is for the conditions at the time of solving, "
                    "which differ from the current ones ({changes})",
                    changes=tr(", ").join(changes),
                )
            )

    def _on_tree_selection_changed(self) -> None:
        """ツリーの選択が変わった。領域だけは複数選べる。

        **種類を混ぜて選ばせない。** 領域と電極を一緒に選んでも、まとめてできる操作が
        無い。クリックした項目（current）と同じ種類だけを残す。
        """
        items = self.tree.selectedItems()
        if not items:
            return
        current = self.tree.currentItem()
        if current is None or current not in items:
            current = items[-1]
        kind = current.data(0, ROLE_KIND)
        if kind == "region":
            keep = [i for i in items if i.data(0, ROLE_KIND) == "region"]
        else:
            keep = [current]
        if len(keep) != len(items):
            self.tree.blockSignals(True)
            try:
                for item in items:
                    item.setSelected(item in keep)
            finally:
                self.tree.blockSignals(False)
        if kind == "region" and len(keep) > 1:
            pids = sorted(i.data(0, ROLE_KEY) for i in keep)  # 選んだ順ではなく PID 順
            self.page_electrode.stop_picking()
            self.page_region.bind(pids)
            self.pages.setCurrentWidget(self.page_region)
            self.refresh_view(highlight_pid=pids)
            return
        self._on_tree_selection(current)

    def _on_tree_selection(self, current) -> None:
        if current is None:
            return
        kind = current.data(0, ROLE_KIND)
        key = current.data(0, ROLE_KEY)
        if kind != "electrode":
            # 電極のページを離れたら、体表面のクリック選択も終える。掛けたままだと、
            # 領域を見ている間の 3D ビューのクリックが、見えていない電極にパッチを足す
            self.page_electrode.stop_picking()
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
        self,
        highlight_pid: int | list[int] | None = None,
        highlight_electrode: str | None = None,
    ) -> None:
        """3D ビューを描き直す。`highlight_pid` は 1 つでも複数（ツリーの複数選択）でもよい。

        VTK は他スレッドから触れないのでメインスレッドで行う。代わりに段階ごとに
        `scope.step()` で表示を進める（SPEC §6.4）。
        """
        if self.session.mesh is None:
            return
        if highlight_pid is None and highlight_electrode is None:
            # 何も指定されなければ、ツリーで選んでいる項目の強調を保つ。以前は結果の
            # 取り込みや表示の設定の変更で描き直すたびに、選んでいる電極・領域の強調が
            # 消えていた（ツリーでは選ばれたまま）
            highlight_pid, highlight_electrode = self._selected_highlight()
        with self.busy(tr("Preparing the mesh")) as scope:
            self._draw_view(scope, highlight_pid, highlight_electrode)

    def _selected_highlight(self) -> tuple[list[int] | None, str | None]:
        """ツリーで選んでいる領域（複数可）か電極。"""
        items = self.tree.selectedItems()
        pids = [i.data(0, ROLE_KEY) for i in items if i.data(0, ROLE_KIND) == "region"]
        names = [i.data(0, ROLE_KEY) for i in items if i.data(0, ROLE_KIND) == "electrode"]
        if names and self.session.electrode(names[0]) is not None:
            return None, names[0]
        return (sorted(pids) or None), None

    def _part_layers(self, target, surface, mode: str, field: str, opacity: float) -> list:
        """領域ごとの見せ方に従って、描く面を `[(面, 不透明度), …]` に分ける。

        - 全体表示: 外表面の代わりに、表示・半透明の領域の境界面を描く（非表示の領域は
          描かないので、その内側の領域が見える）
        - 断面・クリップ: 切り口と残った面を、それぞれの三角形の領域で振り分ける
        """
        from .state import TRANSPARENT, VISIBLE

        if mode == "volume" and target is surface:
            parts = self.session.part_surfaces(field)
        else:
            pids = target.cell_data["PID"] if "PID" in target.cell_data else None
            parts = {}
            if pids is not None:
                for state in (VISIBLE, TRANSPARENT):
                    wanted = [p for p in np.unique(pids) if self.session.region_view(p) == state]
                    cells = np.flatnonzero(np.isin(pids, wanted))
                    if len(cells):
                        parts[state] = target.extract_cells(cells)
        layers = []
        if VISIBLE in parts:
            layers.append((parts[VISIBLE], opacity))
        if TRANSPARENT in parts:
            layers.append((parts[TRANSPARENT], min(opacity, PART_TRANSPARENT_OPACITY)))
        return layers

    def _frame_new_mesh(self) -> None:
        """新しいメッシュを開いた直後だけ、視点をモデル全体に合わせる。

        以前は描き直すたびに `reset_camera()` を呼んでいて、ツリーで項目を選ぶ・
        カラーマップや稜線を切り替える・断面のスライダーを動かすたびに、
        **拡大して見ていたところからモデル全体の表示に戻されていた**（向きは保たれる）。
        全体に戻したいときは 3D ビューで r キーを押す（pyvista の既定）。
        """
        camera = self.plotter.renderer.GetActiveCamera()
        viewport = tuple(self.plotter.renderer.GetViewport())
        untouched = self._framed_camera == self._camera_state(camera)
        if self._framed_mesh == self.session.mesh_serial and not (
            # カラーバーの帯が出た・消えたときは、描く幅が変わるので合わせ直す。
            # ただし利用者が視点を動かしていないときだけ（動かしていれば、その視点を保つ）
            untouched and viewport != self._framed_viewport
        ):
            # 表示するものが変わる（等値面・断面など）ので、手前と奥の切り取りだけ合わせ直す
            self.plotter.renderer.reset_camera_clipping_range()
            return
        if self._framed_mesh != self.session.mesh_serial:
            self._camera_before_section = None  # 前のモデルの視点には戻さない
        self._framed_mesh = self.session.mesh_serial
        self._framed_viewport = viewport
        self.plotter.reset_camera()
        self._framed_camera = self._camera_state(camera)

    def view_section(self, kind: str, axis: str) -> None:
        """断面・クリップを選んだ・法線を変えたときに、それが見える向きに視点を向ける。

        利用者が表示モードや法線を選んだときだけ呼ぶ（位置のスライダーでは呼ばない。
        スライダーを動かすたびに視点が変わると、見比べられない）。
        """
        if self.session.mesh is None:
            return
        camera = self.plotter.renderer.GetActiveCamera()
        if self._camera_before_section is None:
            self._camera_before_section = self._camera_state(camera)
        toward, up = Session.section_view(kind, axis)
        lo, hi = self.session.model_box()
        centre = 0.5 * (lo + hi)
        camera.SetFocalPoint(*centre)
        camera.SetPosition(*(centre + float(np.linalg.norm(hi - lo)) * np.asarray(toward)))
        camera.SetViewUp(*up)
        self.plotter.reset_camera(bounds=[lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]])
        self._framed_viewport = tuple(self.plotter.renderer.GetViewport())
        self._framed_camera = self._camera_state(camera)
        self.plotter.render()

    def leave_section(self) -> None:
        """断面・クリップをやめた。断面を選ぶ前の視点に戻す。

        断面用の視点（真上から・真横から）のまま全体表示に戻ると、体を見る向きとして
        不自然だった。断面を見ている間に利用者が視点を動かしていたら、その視点を保つ。
        """
        saved, self._camera_before_section = self._camera_before_section, None
        camera = self.plotter.renderer.GetActiveCamera()
        if saved is None or self._camera_state(camera) != self._framed_camera:
            return
        position, focal, up, angle, scale = saved
        camera.SetPosition(*position)
        camera.SetFocalPoint(*focal)
        camera.SetViewUp(*up)
        camera.SetViewAngle(angle)
        camera.SetParallelScale(scale)
        self.plotter.renderer.reset_camera_clipping_range()
        self._framed_camera = self._camera_state(camera)
        self.plotter.render()

    @staticmethod
    def _camera_state(camera) -> tuple:
        return (
            camera.GetPosition(),
            camera.GetFocalPoint(),
            camera.GetViewUp(),
            camera.GetViewAngle(),
            camera.GetParallelScale(),
        )

    def _make_room_for_scalar_bar(self) -> None:
        """カラーバーを右端の専用の描画領域へ移し、モデルの描画領域をその分だけ狭める。

        カラーバーは 2D の重ね描きなので、同じ描画領域に置くとモデルにかぶる。
        別のレンダラに移せば、モデルとは決して重ならない（視点は変えない）。
        カラーバーが無いときはモデルの描画領域を全幅に戻す。
        """
        strip = self._scalar_bar_renderer
        strip.RemoveAllViewProps()
        bars = list(self.plotter.scalar_bars.values())
        for bar in bars:
            self.plotter.renderer.RemoveActor2D(bar)
            strip.AddActor2D(bar)
            # 見出しは VTK のカラーバーに描かせない。バーの幅に合わせて右寄せ・縮小され、
            # 細い帯では端が切れた。帯の左上に、左寄せの文字として置く
            title = scalar_bar_title(bar.GetTitle())  # 管理の鍵は表示量の鍵のまま
            bar.SetTitle("")
            strip.AddActor2D(self._scalar_bar_title_actor(title))
        width = max(self.plotter.interactor.width(), 1)
        right = 1.0 - min(SCALAR_BAR_STRIP_PX / width, 0.4) if bars else 1.0
        self.plotter.renderer.SetViewport(0.0, 0.0, right, 1.0)
        strip.SetViewport(right, 0.0, 1.0, 1.0) if bars else strip.SetViewport(0.0, 0.0, 0.0, 0.0)

    @staticmethod
    def _scalar_bar_title_actor(title: str):
        """カラーバーの帯の左上に置く見出し（左寄せ・上揃え）。"""
        from vtkmodules.vtkRenderingCore import vtkTextActor

        actor = vtkTextActor()
        actor.SetInput(title)
        prop = actor.GetTextProperty()
        prop.SetFontSize(SCALAR_BAR["title_font_size"])
        prop.SetColor(pv.Color(SCALAR_BAR["color"]).float_rgb)
        prop.SetJustificationToLeft()
        prop.SetVerticalJustificationToTop()
        actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        actor.SetPosition(0.08, 0.92)
        return actor

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
        iso_text = ""
        if mode in ("slice", "clip"):
            # 切り口は Rust 側で作る。体積メッシュは組み立てない（SPEC §6.3）
            scope.step(tr("Cutting the section"))
            target = self.session.section(
                mode, options["slice_normal"], options["slice_position"], scalars
            )
        elif mode == "isosurface":
            field = field_by_key(scalars)
            quantity = field.iso_quantity if field else None
            if quantity is None:
                self.log(tr("This quantity has no isosurface (e.g. region PID). Showing the whole body."))
            else:
                lo, hi = self.session.value_range(quantity)
                level = lo + (hi - lo) * options["iso_level"]
                # **表示量の鍵で貼る。** 見出しと色の範囲（モデル全体の値域）が全体表示と
                # 同じになり、等値面の色が「値域のどこを切ったか」を示す。以前は内部名
                # （`j_magnitude`）で貼っていたので色の範囲を引けず、範囲が等値面の値
                # ＝しきい値 1 点に潰れて、カラーバーの目盛りが全部同じ数字になっていた
                target = self.session.iso_surface(quantity, level, name=scalars)
                iso_text = tr("Current threshold: {level:.4g} ({field})", level=level, field=tr(field.label))
        self.page_result.show_iso_level(iso_text)
        if target.n_cells == 0:
            target = surface
            scalars = options["field"]

        if scalars == SHAPE_FIELD or scalars not in target.array_names:
            scalars = None  # 色分けなし（PID には大小の意味が無い）
        # 領域ごとに表示・半透明・非表示を選んでいれば、描く面を領域で分ける
        customised = self.session.regions_customised() and mode != "isosurface"

        # 選んだ領域を先に用意する。**体表面に出ていない領域**（頭部モデルの
        # 大半がそう）は、外側を半透明にしないと見えないため
        pids = [] if highlight_pid is None else (
            [highlight_pid] if isinstance(highlight_pid, int) else list(highlight_pid)
        )
        regions = []
        buried = False
        if pids:
            scope.step(tr("Extracting the selected regions"))
        for pid in pids:
            region = self.session.region_polydata(pid)
            if region.n_cells == 0:
                # 体表面に出ていない領域。境界面を Rust 側で取り出す（体積メッシュは作らない）
                region = self.session.region_boundary(pid)
                buried = buried or (region.n_cells > 0 and mode == "volume")
            if region.n_cells:
                regions.append(region)

        # 流線は先に引いておく。引けたら外表面を透かす（流線は体の中を通るので、
        # 外表面が不透明だと 1 本も見えない）。断面は板なので透かさなくても見える
        lines = None
        if options["streamlines"] and self.session.solution is not None:
            scope.step(tr("Tracing streamlines"))
            try:
                seeds = self.session.default_seed_points(options["n_seeds"])
                lines = self.session.streamlines(seeds)
            except Exception as exc:  # noqa: BLE001
                self.log(tr("Could not trace streamlines: {error}", error=exc))
        see_through = lines is not None and lines.n_cells > 0 and mode in ("volume", "clip")
        opacity = (
            BURIED_OPACITY if buried else STREAMLINE_SURFACE_OPACITY if see_through else 1.0
        )
        layers = [(target, opacity)]
        if customised:
            layers = self._part_layers(target, surface, mode, options["field"], opacity)
        clim = None
        if scalars is not None:
            shown = self.session.shown_region_pids() if customised and options["fit_range"] else None
            clim = self.session.field_range(scalars, shown) or self.session.field_range(scalars)

        scope.step(tr("Building the view"))
        edges = options["edges"]
        # **視点は利用者のものとして保つ。** pyvista の add_mesh は既定で視点を
        # 合わせ直すことがあるので、どれも reset_camera=False で足し、モデル全体に
        # 合わせ直すのは新しいメッシュを開いたときだけにする（下の `_frame_new_mesh`）
        for layer, layer_opacity in layers:
            # 2 次要素の面は 4 枚の三角形に割って描くので、稜線は元の要素の辺だけを線で足す
            # （show_edges に任せると分割の線まで出る。`Session.element_edges`）
            element_edges = self.session.element_edges(layer) if edges else None
            if element_edges is not None:
                self.plotter.add_mesh(
                    element_edges,
                    color=EDGE_COLOR,
                    line_width=1,
                    opacity=layer_opacity,
                    reset_camera=False,
                )
            layer_edges = edges and element_edges is None
            if scalars is None:
                # 色分けなし。形は**陰影と稜線**で見せる（SPEC §6.3）
                self.plotter.add_mesh(
                    layer,
                    color=SURFACE_COLOR,
                    show_edges=layer_edges,
                    edge_color=EDGE_COLOR,
                    line_width=1,
                    specular=0.25,
                    specular_power=18,
                    smooth_shading=False,
                    opacity=layer_opacity,
                    reset_camera=False,
                )
            else:
                self.plotter.add_mesh(
                    layer,
                    scalars=scalars,
                    cmap=options["cmap"],
                    show_edges=layer_edges,
                    edge_color=EDGE_COLOR,
                    clim=clim,
                    scalar_bar_args=SCALAR_BAR,
                    categories=scalars == "PID",
                    opacity=layer_opacity,
                    reset_camera=False,
                    **SCALAR_LIGHTING,
                )
        if options["nodes"] and layers:
            self.plotter.add_mesh(
                layers[0][0],
                style="points",
                color=NODE_COLOR,
                point_size=NODE_SIZE,
                render_points_as_spheres=True,
                lighting=False,
                reset_camera=False,
            )
        if lines is not None and lines.n_cells:
            self.plotter.add_mesh(lines, color="black", line_width=2, reset_camera=False)
        if mode == "slice" and target is not surface:
            self.plotter.add_mesh(
                surface,
                color=SURFACE_COLOR,
                opacity=SECTION_CONTEXT_OPACITY,
                pickable=False,
                reset_camera=False,
            )

        for region in regions:
            draw_on_top(
                self.plotter.add_mesh(
                    region, color=REGION_COLOR, show_edges=False, pickable=False, reset_camera=False
                )
            )
        # ほかの電極の位置も見えるように、縁を役割の色で描く（選んでいる電極は下で面を塗る）
        # クリップでは電極の縁と面も同じ平面で切る（切り落とした側に宙に浮いて見えないように）
        clip_plane = (
            self.session.section_plane(options["slice_normal"], options["slice_position"])
            if mode == "clip"
            else None
        )

        def clipped(poly):
            if clip_plane is None or not poly.n_cells:
                return poly
            origin, normal = clip_plane
            return poly.clip(normal=normal, origin=origin)

        # 非表示にした電極は描かない（選んで編集している電極だけは、非表示でも面を塗る）
        for spec in self.session.study.electrodes:
            view = self.session.electrode_view(spec.name)
            if spec.name == highlight_electrode or view == HIDDEN:
                continue
            outline = clipped(self.session.electrode_outline(spec.name))
            if outline.n_cells:
                self.plotter.add_mesh(
                    outline,
                    color=ELECTRODE_OUTLINE_COLORS.get(spec.role, "black"),
                    line_width=3,
                    lighting=False,
                    pickable=False,
                    reset_camera=False,
                    opacity=PART_TRANSPARENT_LINE_OPACITY if view == TRANSPARENT else 1.0,
                )
        if highlight_electrode is not None:
            scope.step(tr("Extracting the electrode faces"))
            faces = clipped(self.session.electrode_polydata(highlight_electrode))
            if faces.n_cells:
                faded = self.session.electrode_view(highlight_electrode) == TRANSPARENT
                draw_on_top(
                    self.plotter.add_mesh(
                        faces,
                        color="red",
                        opacity=0.5 if faded else 1.0,
                        reset_camera=False,
                    )
                )

        scope.step(tr("Rendering"))
        self.plotter.add_axes()
        self._make_room_for_scalar_bar()
        self._frame_new_mesh()
        self.plotter.render()
        # clear() でピッキングも外れるので、選択中なら掛け直す
        self._arm_picking()
