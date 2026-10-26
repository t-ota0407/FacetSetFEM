"""プロパティパネルの各ページ（SPEC §6.1 / §6.2）。

モデルツリーで選んだ項目に応じて、メインウィンドウの右側にどれか 1 つが出る。
どのページも状態は持たず、[`Session`] を読み書きするだけ（**GUI にロジックは置かない**。
SPEC §7）。描画や Undo の記録はメインウィンドウに頼む。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import FacetSetFemError, backend_choices, describe_gpus, preconditioner_choices
from ..i18n import N_, tr
from .busy import REVEAL_DELAY_MS
from .state import (
    AREA_MISMATCH_WARNING,
    DEFAULT_RESULT_FIELD,
    NODAL_VALUE_WARNING,
    SHAPE_FIELD,
    STALE_COLOR,
    TISSUE_HINTS,
    Session,
    field_by_key,
    stale_result_message,
)
from .widgets import MixedValueSpinBox

if TYPE_CHECKING:
    from .main_window import MainWindow

__all__ = ["ElectrodePage", "MeshPage", "RegionPage", "ResultPage", "SolverPage"]

COLORMAPS = ["viridis", "plasma", "coolwarm", "turbo", "gray"]

#: 稜線・節点を既定で出す上限（表示する三角形の枚数）。
#:
#: これを超えると線が潰れて灰色の塊になり、形が**かえって見えなくなる**。
#: 実測で頭部モデルの外表面は 87,812 面。手動で入れることは妨げない
EDGE_LIMIT = 30_000

#: 複数の領域を選んだとき、見出しに PID を並べる上限（それ以上は「ほか n」）
MAX_LISTED_PIDS = 12

#: クリックで置く電極の形（表示名, 内部名）。既定は従来どおり円
PATCH_SHAPES = [
    (N_("Circle"), "circle"),
    (N_("Rectangle (rounded corners allowed)"), "rect"),
]

#: 電極のモデル（表示名, ケースファイルの distribution。SPEC §2.4 / §2.5）
ELECTRODE_MODELS = [
    (N_("Uniform current density"), "uniform_current_density"),
    (N_("Complete electrode model"), "complete_electrode"),
]

#: 完全電極モデルに初めて切り替えたときの接触インピーダンス [Ω·cm²]（= 0.01 Ω·m²）。
#: 入力欄を埋めるための仮の値で、推奨値ではない（SPEC §6.2 の手順 3）
DEFAULT_CONTACT_IMPEDANCE_OHM_CM2 = 100.0

#: 指定面積と実際の面積がずれているときの説明（SPEC §11.1）
AREA_MISMATCH_HINT = N_(
    "The area of the selected faces differs from the nominal area by {limit:.0%} or more. "
    "Faces are taken or left whole, so the mesh is coarse for this electrode, or the projection "
    "onto the tangent plane is off on a curved surface. The current density uses the actual area. "
    "To be exact, refine the mesh around the electrode, or make shell elements of the electrode "
    "shape in your meshing tool and select them by shell PID."
)


def area_mismatch_hint() -> str:
    return tr(AREA_MISMATCH_HINT, limit=AREA_MISMATCH_WARNING)


def _mm_spin(value: float, lo: float, hi: float, step: float) -> QDoubleSpinBox:
    """mm 単位の長さの入力欄。"""
    spin = QDoubleSpinBox()
    spin.setDecimals(1)
    spin.setRange(lo, hi)
    spin.setValue(value)
    spin.setSingleStep(step)
    spin.setSuffix(" mm")
    return spin


def format_area(area: float) -> str:
    """面積を mm² と cm² の併記で書く（電極パッドの寸法は cm² で語られるため）。"""
    return tr("{mm2:.1f} mm² ({cm2:.2f} cm²)", mm2=area * 1e6, cm2=area * 1e4)


def describe_selection(info: dict) -> tuple[str, bool]:
    """`pick_patch` の結果を「118 面 3.98 cm²（指定 4.00 cm²、−0.5 %）」の形で書く。

    2 つ目の戻り値は、指定面積とのずれが警告のしきい値を超えているか。
    """
    text = tr("{n} faces {cm2:.2f} cm²", n=info["n_faces"], cm2=info["area"] * 1e4)
    mismatch = Session.area_mismatch(info)
    if mismatch is None:
        return text, False
    text += tr(
        " (nominal {cm2:.2f} cm², {mismatch:+.1%})", cm2=info["nominal_area"] * 1e4, mismatch=mismatch
    )
    return text, abs(mismatch) > AREA_MISMATCH_WARNING


#: 結果ページのボタンの文言。形状表示と、その前に見ていた結果の表示を行き来する
SHOW_SHAPE_LABEL = N_("Back to shape view")
SHOW_RESULT_LABEL = N_("Back to result view")

#: 3D ビューの表示モード（表示名, 内部名）
VIEW_MODES = [
    (N_("Whole body"), "volume"),
    (N_("Slice"), "slice"),
    (N_("Clip"), "clip"),
    (N_("Isosurface"), "isosurface"),
]


class ScrollPage(QScrollArea):
    """プロパティパネルのページの土台。**縦に収まらなければスクロールする。**

    以前はふつうのウィジェットで、中身がパネルの高さに収まらないと、Qt が欄を
    押しつぶす（結果ページで「古い結果」の帯が出ると法線の欄が潰れた）か、
    ウィンドウの最小の高さが中身に引っぱられて**ウィンドウが勝手に縦に伸びた**
    （電極ページで矩形を選んでパッチを置くと 900 px を超え、1100 × 700 に
    縮めようとしても 894 px より低くならなかった）。ページの中身は `body_layout` に並べる。
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        self.setWidget(body)
        self.body_layout = QVBoxLayout(body)


class MeshPage(ScrollPage):
    """メッシュの概要（SPEC §6.2 の手順 1 の読み込みレポート）。"""

    def __init__(self, window: MainWindow) -> None:
        super().__init__()
        self.window = window
        layout = self.body_layout
        layout.addWidget(QLabel("<b>" + tr("Mesh") + "</b>"))
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text)

    def reload(self) -> None:
        if self.window.session.mesh is None:
            self.text.setPlainText(tr("(no model is open)"))
        else:
            self.text.setPlainText(self.window.session.mesh_summary())


class RegionPage(ScrollPage):
    """領域（PID）への σ 割り当て（SPEC §6.2 の手順 2）。"""

    def __init__(self, window: MainWindow) -> None:
        super().__init__()
        self.window = window
        #: 編集している領域。ツリーで複数選ぶと 2 つ以上になり、σ をまとめて入れられる
        self.pids: list[int] = []

        layout = self.body_layout
        layout.addWidget(QLabel("<b>" + tr("Region") + "</b>"))
        form = QFormLayout()
        self.pid_label = QLabel("-")
        self.pid_label.setWordWrap(True)
        self.name_edit = QLineEdit()
        self.sigma_spin = MixedValueSpinBox()
        self.sigma_spin.setDecimals(6)
        self.sigma_spin.setRange(0.0, 1000.0)
        self.sigma_spin.setSingleStep(0.01)
        self.sigma_spin.setSuffix(" S/m")
        # σ = 0 は未設定を表す（Session.open_mesh）。「0 S/m」と出すと入れた値に見える
        self.sigma_spin.set_floor_text(tr("not set"))
        form.addRow("PID", self.pid_label)
        form.addRow(tr("Name"), self.name_edit)
        form.addRow(tr("Conductivity σ"), self.sigma_spin)
        layout.addLayout(form)
        hint = QLabel(tr("Ctrl / Shift + click in the tree to set σ for several regions at once."))
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        layout.addWidget(hint)

        hints = QGroupBox(tr("Typical tissue conductivities [S/m]"))
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

    @property
    def pid(self) -> int | None:
        """1 つだけ選んでいるときの PID（複数・未選択なら None）。"""
        return self.pids[0] if len(self.pids) == 1 else None

    def bind(self, pids: int | list[int]) -> None:
        """編集する領域を決める。複数なら σ をまとめて入れる。

        - σ がそろっていればその値を、ばらばらなら「値がそろっていません」を出す。
          入れた値は選んだ全部に入る
        - **名前は複数選択の間は変えられない。** 名前は領域を見分けるためのもので、
          まとめて同じにすると区別がつかなくなる（組織の目安のボタンは例外。同じ組織を
          まとめて選ぶ使い方なので、名前もそろえてよい）
        """
        pids = [pids] if isinstance(pids, int) else list(pids)
        self.pids = []  # コミットを抑えてから値を入れる
        session = self.window.session
        many = len(pids) > 1
        if many:
            shown = ", ".join(str(p) for p in pids[:MAX_LISTED_PIDS])
            more = (
                tr(" and {n} more", n=len(pids) - MAX_LISTED_PIDS)
                if len(pids) > MAX_LISTED_PIDS
                else ""
            )
            self.pid_label.setText(
                tr("{n} regions selected ({pids}{more})", n=len(pids), pids=shown, more=more)
            )
        else:
            self.pid_label.setText(str(pids[0]) if pids else "-")
        self.name_edit.setEnabled(not many)
        self.name_edit.setPlaceholderText(tr("(cannot change with several selected)") if many else "")
        self.name_edit.setText("" if many or not pids else session.region_name(pids[0]))
        sigmas = {session.conductivity_of(p) for p in pids}
        if len(sigmas) == 1:
            self.sigma_spin.set_value(sigmas.pop())
        elif sigmas:
            self.sigma_spin.set_mixed()
        self.pids = pids

    def _apply_hint(self, name: str, value: float) -> None:
        if not self.pids:
            return
        # 欄を書き直すのはボタンのときだけ（打っている途中に書き直すと文字が消える）
        self.sigma_spin.set_value(value)
        if self.pid is not None:
            self.name_edit.setText(name)
        self._set(value, name)

    def _commit(self) -> None:
        if not self.pids:
            return
        value = self.sigma_spin.value()
        if value < self.sigma_spin.floor:
            return  # 「値がそろっていません」のまま。何も入れない
        # 複数選択では名前を変えない（欄は空で押せない）
        self._set(value, self.name_edit.text() if self.pid is not None else None)

    def _set(self, sigma: float, name: str | None) -> None:
        """選んだ全部に σ（と名前）を入れる。まとめて 1 手で戻せる。"""
        label = ", ".join(str(p) for p in self.pids)
        self.window.checkpoint(tr("Change settings of region {pids}", pids=label))
        for pid in self.pids:
            self.window.session.set_conductivity(pid, sigma, name)
        self.window.refresh_tree()


class ElectrodePage(ScrollPage):
    """電極の定義（SPEC §6.2 の手順 3）。

    面を選ぶたびに**合計面積と平均電流密度**を出す。
    """

    def __init__(self, window: MainWindow) -> None:
        super().__init__()
        self.window = window
        self.name: str | None = None
        #: ホバー予告をいま出している面（同じ面の上では作り直さない）
        self._hover_at = None

        layout = self.body_layout
        layout.addWidget(QLabel("<b>" + tr("Electrode") + "</b>"))
        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.role_combo = QComboBox()
        self.role_combo.addItems(["anode", "cathode"])
        self.current_spin = QDoubleSpinBox()
        self.current_spin.setDecimals(4)
        self.current_spin.setRange(0.0, 1.0e6)
        self.current_spin.setSuffix(" mA")
        # 電極のモデル（SPEC §2.5）。接触インピーダンスは文献に合わせて Ω·cm² で入れる
        self.model_combo = QComboBox()
        for label, key in ELECTRODE_MODELS:
            self.model_combo.addItem(tr(label), key)
        self.model_combo.setToolTip(
            tr(
                "Uniform current density: the same current density flows out of every part of the\n"
                "electrode (does not show the concentration at the edge).\n"
                "Complete electrode model: the electrode plate is at one potential and a contact\n"
                "impedance sits between it and the skin, so current gathers at the edge (SPEC §2.5)."
            )
        )
        self.contact_spin = QDoubleSpinBox()
        self.contact_spin.setDecimals(3)
        self.contact_spin.setRange(0.001, 1.0e8)
        self.contact_spin.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
        self.contact_spin.setValue(DEFAULT_CONTACT_IMPEDANCE_OHM_CM2)
        self.contact_spin.setSuffix(" Ω·cm²")
        self.contact_spin.setToolTip(
            tr(
                "Resistance per area between the electrode and the skin (gel and skin surface).\n"
                "It varies many-fold with the gel and the skin condition, so fit it to the measured\n"
                "inter-electrode resistance (facetsetfem.fit_contact_impedance) rather than picking one value.\n"
                "1 Ω·m² = 10⁴ Ω·cm²."
            )
        )
        form.addRow(tr("Name"), self.name_edit)
        form.addRow(tr("Role"), self.role_combo)
        form.addRow(tr("Total current"), self.current_spin)
        form.addRow(tr("Electrode model"), self.model_combo)
        form.addRow(tr("Contact impedance"), self.contact_spin)
        layout.addLayout(form)

        layout.addWidget(QLabel(tr("Electrode faces (shell PID)")))
        self.pid_list = QListWidget()
        layout.addWidget(self.pid_list)
        # シェル要素の無いモデル（人体モデルはふつうこれ）では一覧を畳んで理由を出す
        self.no_shells = QLabel(
            tr("This model has no shell elements. Place the electrode faces with \"Pick on the body surface\" below.")
        )
        self.no_shells.setWordWrap(True)
        self.no_shells.setStyleSheet("color: #666;")
        layout.addWidget(self.no_shells)

        # --- 体表面から選ぶ（SPEC §11.1）---
        # 人体モデルはソリッド要素だけで配られるのが普通で、その場合シェル PID の
        # 一覧は空になる。3D ビューを直接クリックして面を選べるようにする
        patch_box = QGroupBox(tr("Pick on the body surface (click)"))
        patch_layout = QVBoxLayout(patch_box)
        self.pick_button = QPushButton(tr("Pick by clicking"))
        self.pick_button.setCheckable(True)
        self.pick_button.setToolTip(
            tr(
                "Click the body surface in the 3D view; the faces around it become the electrode.\n"
                "Press again to return to normal rotation."
            )
        )
        patch_layout.addWidget(self.pick_button)

        # --- 電極の形（クリックで置くもの。SPEC §11.1）---
        # 寸法は電極の仕様書に載っている言い方で入れる（円は直径、矩形は幅 × 高さ）
        shape_form = QFormLayout()
        self.shape_combo = QComboBox()
        for label, key in PATCH_SHAPES:
            self.shape_combo.addItem(tr(label), key)
        self.diameter_spin = _mm_spin(50.0, 0.1, 1000.0, 5.0)
        self.width_spin = _mm_spin(20.0, 0.1, 1000.0, 1.0)
        self.height_spin = _mm_spin(20.0, 0.1, 1000.0, 1.0)
        self.corner_spin = _mm_spin(0.0, 0.0, 500.0, 0.5)
        self.corner_spin.setToolTip(
            tr(
                "Corner radius. 0 gives sharp corners. Current concentrates at sharp corners and\n"
                "overstates the maximum |J|, so match the real electrode if its corners are rounded."
            )
        )
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setDecimals(1)
        self.angle_spin.setRange(-180.0, 180.0)
        self.angle_spin.setSingleStep(15.0)
        self.angle_spin.setWrapping(True)
        self.angle_spin.setSuffix("°")
        self.angle_spin.setToolTip(
            tr(
                "At 0° the width edge is horizontal on screen (fixed by the view when you click).\n"
                "Positive values rotate counter-clockwise as seen from outside."
            )
        )
        size_row = QHBoxLayout()
        size_row.addWidget(self.width_spin)
        size_row.addWidget(QLabel("×"))
        size_row.addWidget(self.height_spin)
        self.size_row = QWidget()
        self.size_row.setLayout(size_row)
        size_row.setContentsMargins(0, 0, 0, 0)
        self.nominal_label = QLabel("")
        shape_form.addRow(tr("Shape"), self.shape_combo)
        shape_form.addRow(tr("Diameter"), self.diameter_spin)
        shape_form.addRow(tr("Width × height"), self.size_row)
        shape_form.addRow(tr("Corner radius"), self.corner_spin)
        shape_form.addRow(tr("Angle"), self.angle_spin)
        shape_form.addRow(tr("Nominal area"), self.nominal_label)
        self.shape_form = shape_form
        patch_layout.addLayout(shape_form)

        patch_form = QFormLayout()
        self.brush_spin = _mm_spin(5.0, 0.1, 500.0, 1.0)
        self.brush_spin.setToolTip(
            tr(
                "Brush size for Shift / Alt + drag (paint / erase) and for clicks in Remove mode.\n"
                "Make it small to handle one face at a time."
            )
        )
        self.patch_pid = QComboBox()
        self.patch_pid.setToolTip(
            tr(
                "Regions on the outer surface (largest area first).\n"
                "Usually restrict to the skin PID to avoid faces that are on the outer surface\n"
                "but not on the body surface, such as the cut at the neck."
            )
        )
        patch_form.addRow(tr("Brush radius"), self.brush_spin)
        patch_form.addRow(tr("Restrict to region"), self.patch_pid)
        patch_layout.addLayout(patch_form)

        self.patch_list = QListWidget()
        # 1 枚を 2 行で出す（形と場所 / 面数・面積・指定面積とのずれ）。1 行だと
        # 肝心のずれが横スクロールの先に隠れる
        # 高さは行の高さから決める（1 枚 2 行で、2〜3 枚ぶん）。以前は 120 px 固定で、
        # 高 DPI の画面では 1 枚ぶんしか見えなかった
        line = self.patch_list.fontMetrics().lineSpacing()
        self.patch_list.setMinimumHeight(int(line * 4.8))
        self.patch_list.setMaximumHeight(int(line * 7.2))
        self.patch_list.setWordWrap(True)
        mode_row = QHBoxLayout()
        self.add_mode = QPushButton(tr("Add"))
        self.erase_mode = QPushButton(tr("Remove"))
        for button in (self.add_mode, self.erase_mode):
            button.setCheckable(True)
            mode_row.addWidget(button)
        self.add_mode.setChecked(True)
        self.add_mode.toggled.connect(lambda on: on and self.erase_mode.setChecked(False))
        self.erase_mode.toggled.connect(lambda on: on and self.add_mode.setChecked(False))
        patch_layout.addLayout(mode_row)
        hint = QLabel(
            tr(
                "Click = once in the current mode. Shift + drag paints, "
                "Alt + drag erases. A plain drag rotates."
            )
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        patch_layout.addWidget(hint)

        patch_layout.addWidget(self.patch_list)
        self.hand_edits = QLabel("")
        self.hand_edits.setWordWrap(True)
        patch_layout.addWidget(self.hand_edits)

        button_row = QHBoxLayout()
        self.reshape_patch_button = QPushButton(tr("Apply shape to patch"))
        self.reshape_patch_button.setToolTip(
            tr(
                "Replace the shape, size and angle of the patch selected in the list with the current "
                "input (the centre stays).\nSelecting a patch in the list loads its shape into the inputs."
            )
        )
        self.reshape_patch_button.clicked.connect(self._reshape_patch)
        button_row.addWidget(self.reshape_patch_button)
        self.remove_patch_button = QPushButton(tr("Remove patch"))
        self.remove_patch_button.clicked.connect(self._remove_patch)
        self.remove_patch_button.setEnabled(False)
        button_row.addWidget(self.remove_patch_button)
        self.clear_edits_button = QPushButton(tr("Clear hand edits"))
        self.clear_edits_button.setToolTip(
            tr("Discard faces added or removed one by one and go back to the patches alone.")
        )
        self.clear_edits_button.clicked.connect(self._clear_hand_edits)
        button_row.addWidget(self.clear_edits_button)
        patch_layout.addLayout(button_row)
        layout.addWidget(patch_box)

        self.summary = QLabel("-")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.summary)
        layout.addStretch(1)

        self.name_edit.editingFinished.connect(self._commit)
        self.role_combo.currentTextChanged.connect(self._commit)
        self.current_spin.valueChanged.connect(self._commit)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.contact_spin.valueChanged.connect(self._commit)
        self.pid_list.itemChanged.connect(self._commit)
        self.pick_button.toggled.connect(self._toggle_picking)
        self.shape_combo.currentIndexChanged.connect(self._on_shape_changed)
        for spin in (
            self.diameter_spin,
            self.width_spin,
            self.height_spin,
            self.corner_spin,
            self.angle_spin,
            self.brush_spin,
        ):
            spin.valueChanged.connect(self._on_shape_changed)
        self.patch_list.currentRowChanged.connect(self._load_patch_shape)
        self._on_shape_changed()
        self._fit_pid_list()

    def bind(self, name: str) -> None:
        # 体表面の索引づくり（外表面 87,812 面で 1.9 秒）が初回だけここで走る。
        # 電極をツリーで選んだ瞬間なので、待たされていることを見せる（SPEC §6.4）
        with self.window.busy(tr("Opening the electrode settings")) as scope:
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
        self.model_combo.setCurrentIndex(max(self.model_combo.findData(spec.distribution), 0))
        self.contact_spin.setValue(
            spec.contact_impedance * 1e4
            if spec.contact_impedance is not None
            else DEFAULT_CONTACT_IMPEDANCE_OHM_CM2
        )
        self.contact_spin.setEnabled(spec.is_complete)
        self.pid_list.clear()
        for pid in session.mesh.shell_pids():
            item = QListWidgetItem(f"PID {pid}")
            item.setData(Qt.ItemDataRole.UserRole, pid)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if pid in spec.shell_pids else Qt.CheckState.Unchecked
            )
            self.pid_list.addItem(item)
        self._fit_pid_list()
        scope.step(tr("Checking the regions on the body surface"))
        self._reload_patch_pids()
        self.name = name
        self._reload_patches()
        self._reload_hand_edits()
        scope.step(tr("Totalling area and current density"))
        self._update_summary()

    #: シェル PID の一覧に一度に見せる行数の上限（それ以上はスクロール）
    MAX_PID_ROWS = 6

    def _fit_pid_list(self) -> None:
        """シェル PID の一覧を行数ぶんの高さにする（空なら畳んで理由を出す）。

        ページがスクロールするようになってから、一覧が余った高さを全部取り、
        シェルの無いモデルでも空の大きな枠が「体表面から選ぶ」を下へ押し出していた。
        """
        count = self.pid_list.count()
        self.pid_list.setVisible(count > 0)
        self.no_shells.setVisible(count == 0)
        if count:
            rows = min(count, self.MAX_PID_ROWS)
            height = self.pid_list.sizeHintForRow(0) * rows + 2 * self.pid_list.frameWidth()
            self.pid_list.setFixedHeight(height)

    def stop_picking(self) -> None:
        """体表面のクリック選択を終える（掛かっていなければ何もしない）。"""
        if self.pick_button.isChecked():
            self.pick_button.setChecked(False)  # `_toggle_picking` が選択を外す

    def unbind(self) -> None:
        """どの電極も編集していない状態にする（編集していた電極が消えたとき）。"""
        self.stop_picking()
        self.name = None

    # --- 体表面から選ぶ（SPEC §11.1）----------------------------------------

    def _reload_patch_pids(self) -> None:
        """外表面に出ている領域を、面積の大きい順に並べる。"""
        current = self.patch_pid.currentData()
        self.patch_pid.blockSignals(True)
        self.patch_pid.clear()
        self.patch_pid.addItem(tr("No restriction"), None)
        for row in self.window.session.surface_pid_choices():
            self.patch_pid.addItem(
                tr("PID {pid} ({cm2:.0f} cm²)", pid=row["pid"], cm2=row["area"] * 1e4), row["pid"]
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
        session = self.window.session
        for i, patch in enumerate(spec.patches):
            warn = False
            try:
                info = session.patch_info(patch)
                detail, warn = describe_selection(info)
            except Exception as exc:  # noqa: BLE001 - 表示だけなので握って理由を出す
                detail = str(exc).splitlines()[0]
            centre = "(" + ", ".join(f"{v * 1e3:.0f}" for v in patch.center) + ") mm"
            limit = f" PID {patch.pid}" if patch.pid is not None else ""
            item = QListWidgetItem(
                f"{'⚠ ' if warn else ''}{i + 1}. {patch.describe()}   {centre}{limit}\n    {detail}"
            )
            if warn:
                item.setToolTip(area_mismatch_hint())
            self.patch_list.addItem(item)
        self.reshape_patch_button.setEnabled(False)
        self.remove_patch_button.setEnabled(False)  # 一覧で選ぶまで押せない

    def _reload_hand_edits(self) -> None:
        """手で足した面・外した面の枚数を出す。"""
        spec = self.window.session.electrode(self.name) if self.name else None
        if spec is None or not (spec.added_faces or spec.removed_faces):
            self.hand_edits.setText("")
            self.clear_edits_button.setEnabled(False)
            return
        parts = []
        if spec.added_faces:
            parts.append(tr("{n} faces added by hand", n=len(spec.added_faces)))
        if spec.removed_faces:
            parts.append(tr("{n} faces removed", n=len(spec.removed_faces)))
        self.hand_edits.setText("   ".join(parts))
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
                    tr(
                        "A slice or isosurface is shown. Electrodes are placed on the body surface, "
                        "so switch back to the whole-body view before picking."
                    )
                )
            self.window.begin_patch_picking(
                self._on_picked, self._on_hover, self._on_stroke_end
            )
            self.pick_button.setText(tr("Picking by clicking (press to stop)"))
        else:
            self.window.end_patch_picking()
            self.pick_button.setText(tr("Pick by clicking"))

    def _brush(self) -> tuple[float, int | None]:
        """筆の半径 [m] と絞り込む PID（塗る・消すとき）。"""
        return self.brush_spin.value() * 1e-3, self.patch_pid.currentData()

    def _stamp_shape(self, axis=None) -> dict:
        """クリックで置く電極の形（`Session.pick_patch` のキーワード引数。m と度）。

        矩形の `axis` を省略すると、いまの画面の横方向を使う（「画面に対してまっすぐ」）。
        """
        if self.shape_combo.currentData() != "rect":
            return {"radius": 0.5e-3 * self.diameter_spin.value()}
        return {
            "shape": "rect",
            "width": 1e-3 * self.width_spin.value(),
            "height": 1e-3 * self.height_spin.value(),
            "corner_radius": 1e-3 * self.corner_spin.value(),
            "axis": list(axis) if axis is not None else self.window.screen_right(),
            "angle": self.angle_spin.value(),
        }

    def _on_shape_changed(self, *_args) -> None:
        """形の入力が変わった。見せる欄・角の半径の上限・指定面積を合わせる。"""
        rect = self.shape_combo.currentData() == "rect"
        self.shape_form.setRowVisible(self.diameter_spin, not rect)
        for widget in (self.size_row, self.corner_spin, self.angle_spin):
            self.shape_form.setRowVisible(widget, rect)
        # 角の半径は短い辺の半分まで（それ以上は形にならない）
        self.corner_spin.setMaximum(
            0.5 * min(self.width_spin.value(), self.height_spin.value())
        )
        try:
            # 面積は向きに依らない。軸はダミーでよい（カメラを見に行かない）
            area = Session.nominal_area(**self._stamp_shape(axis=[1.0, 0.0, 0.0]))
            self.nominal_label.setText(format_area(area))
        except FacetSetFemError as exc:
            self.nominal_label.setText(str(exc).splitlines()[0])
        self._hover_at = None  # 予告を作り直させる

    def _load_patch_shape(self, row: int) -> None:
        """一覧で選んだパッチの形を入力欄に読み込む（「形を反映」で直せるように）。"""
        spec = self.window.session.electrode(self.name) if self.name else None
        valid = spec is not None and 0 <= row < len(spec.patches)
        self.reshape_patch_button.setEnabled(valid)
        self.remove_patch_button.setEnabled(valid)
        if not valid:
            return
        patch = spec.patches[row]
        widgets = (
            self.shape_combo,
            self.diameter_spin,
            self.width_spin,
            self.height_spin,
            self.corner_spin,
            self.angle_spin,
        )
        for w in widgets:
            w.blockSignals(True)
        try:
            self.shape_combo.setCurrentIndex(self.shape_combo.findData(patch.shape))
            if patch.is_rect:
                self.width_spin.setValue(patch.width * 1e3)
                self.height_spin.setValue(patch.height * 1e3)
                self.corner_spin.setMaximum(0.5e3 * min(patch.width, patch.height))
                self.corner_spin.setValue(patch.corner_radius * 1e3)
                self.angle_spin.setValue(patch.angle)
            else:
                self.diameter_spin.setValue(2e3 * patch.radius)
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._on_shape_changed()

    def _reshape_patch(self) -> None:
        """一覧で選んだパッチの形を、いまの入力に置き換える（中心と PID はそのまま）。"""
        row = self.patch_list.currentRow()
        spec = self.window.session.electrode(self.name) if self.name else None
        if spec is None or not 0 <= row < len(spec.patches):
            return
        old = spec.patches[row]
        # 矩形のまま直すなら、置いたときの向きの軸を保つ（角度はその軸から測る）
        shape = self._stamp_shape(axis=old.axis if old.is_rect else None)
        with self.window.busy(tr("Reselecting the electrode faces"), wait_ms=REVEAL_DELAY_MS) as scope:
            try:
                info = self.window.session.pick_patch(old.center, pid=old.pid, **shape)
            except Exception as exc:  # noqa: BLE001
                self.window.show_error(exc)
                return
            what = tr("Change the shape of patch {n} of electrode {name}", n=row + 1, name=self.name)
            self.window.checkpoint(what)
            self.window.session.reshape_patch(self.name, row, **shape)
            self._log_selection(what, info)
            self._after_patch_change(scope)
        self.patch_list.setCurrentRow(row)

    def _log_selection(self, what: str, info: dict) -> None:
        detail, warn = describe_selection(info)
        self.window.log(f"{what}: {detail}")
        if warn:
            self.window.log("  " + tr("[Warning] {text}", text=area_mismatch_hint()))

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
            self.window.log(tr("You clicked outside the model. Click on the body surface."))
            return

        radius, pid = self._brush()
        if erase is None and not self.erase_mode.isChecked():
            self._place_patch(point, pid)
            return

        erase = True if erase is None else erase
        label = (
            tr("Erase faces of electrode {name}", name=self.name)
            if erase
            else tr("Paint faces of electrode {name}", name=self.name)
        )
        # ドラッグ中は同じ名前で checkpoint を積むので、**1 ストロークが 1 手**になる
        self.window.checkpoint(label)
        try:
            changed = session.paint_faces(self.name, point, radius, pid, erase)
        except Exception as exc:  # noqa: BLE001
            self.window.show_error(exc)
            return
        if changed == 0:
            return  # 何も変わっていない。描き直す必要もない
        self._after_patch_change()

    def _place_patch(self, point, pid: int | None) -> None:
        """クリックした場所に、いまの形のパッチを 1 枚置く。"""
        session = self.window.session
        # 向き（画面の横方向）はクリックした瞬間の視点で決める
        shape = self._stamp_shape()
        with self.window.busy(tr("Selecting the electrode faces"), wait_ms=REVEAL_DELAY_MS) as scope:
            try:
                preview = session.pick_patch(point, pid=pid, **shape)
            except Exception as exc:  # noqa: BLE001
                self.window.show_error(exc)
                return
            what = tr("Add a patch to electrode {name}", name=self.name)
            self.window.checkpoint(what)
            session.add_patch(self.name, point, pid=pid, **shape)
            self._log_selection(what, preview)
            self._after_patch_change(scope)

    def _on_hover(self, point) -> None:
        """カーソルの下に、これから塗られる範囲を薄く出す（SPEC §6.2 の手順 3）。"""
        if self.name is None or point is None:
            self.window.clear_paint_preview()
            self._hover_at = None
            return
        # 動くたびに作り直すと重いので、同じ面の上にいる間は作り直さない
        session = self.window.session
        radius, pid = self._brush()
        erase = self.erase_mode.isChecked()
        try:
            info = session.pick_patch(point, 0.0, pid)
            seed = tuple(info["face_refs"][0]) if info["n_faces"] else None
        except Exception:  # noqa: BLE001 - 予告表示なので黙って諦める
            return
        key = None if seed is None else (seed, erase)
        if key is not None and key == self._hover_at:
            return
        self._hover_at = key
        # 追加モードのクリックは電極の形を置く。削除モードのクリックは筆で消す
        try:
            if erase:
                preview, outline = session.pick_patch(point, radius, pid), None
            else:
                preview = session.pick_patch(point, pid=pid, **self._stamp_shape())
                outline = session.outline_polydata(preview)
        except Exception:  # noqa: BLE001
            self.window.clear_paint_preview()
            return
        self.window.show_paint_preview(
            session.triangles_to_polydata(preview["triangles"]),
            erase=erase,
            outline=outline,
        )

    def _clear_hand_edits(self) -> None:
        if self.name is None:
            return
        with self.window.busy(tr("Clearing hand edits"), wait_ms=REVEAL_DELAY_MS) as scope:
            self.window.checkpoint(tr("Clear hand edits of electrode {name}", name=self.name))
            self.window.session.clear_hand_edits(self.name)
            self._after_patch_change(scope)

    def _remove_patch(self) -> None:
        row = self.patch_list.currentRow()
        if self.name is None or row < 0:
            return
        with self.window.busy(tr("Removing electrode faces"), wait_ms=REVEAL_DELAY_MS) as scope:
            self.window.checkpoint(tr("Remove a patch of electrode {name}", name=self.name))
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
            scope.step(tr("Totalling area and current density"))
        self._update_summary()
        self.window.refresh_tree()
        if scope is not None:
            scope.step(tr("Drawing the electrode faces"))
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
        self.window.checkpoint(tr("Change settings of electrode {name}", name=self.name))
        new_name = self.name_edit.text().strip()
        if new_name and new_name != spec.name:
            try:
                session.rename_electrode(spec.name, new_name)
            except FacetSetFemError as exc:
                # ほかの電極と同じ名前にはできない。入力欄を元の名前に戻す
                self.name_edit.setText(spec.name)
                self.window.show_error(exc)
        elif not new_name:
            self.name_edit.setText(spec.name)
        spec.role = self.role_combo.currentText()
        spec.current = self.current_spin.value() * 1e-3
        spec.distribution = self.model_combo.currentData()
        # 一様電流密度に戻しても接触インピーダンスは覚えておく（ケースファイルには書かない）
        if spec.is_complete:
            spec.contact_impedance = self.contact_spin.value() * 1e-4
        spec.shell_pids = [
            self.pid_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.pid_list.count())
            if self.pid_list.item(i).checkState() == Qt.CheckState.Checked
        ]
        self.name = spec.name
        self._update_summary()
        self.window.refresh_tree()
        self.window.refresh_view(highlight_electrode=spec.name)

    def _on_model_changed(self, *_args) -> None:
        """電極のモデルを切り替えた（SPEC §2.5）。接触インピーダンスの欄は CEM のときだけ使う。"""
        self.contact_spin.setEnabled(self.model_combo.currentData() == "complete_electrode")
        self._commit()

    def _update_summary(self) -> None:
        session = self.window.session
        spec = session.electrode(self.name) if self.name else None
        # 筆で塗っただけ（パッチなし）の電極も面を持つ（SPEC §11.1）
        if spec is None or not (
            spec.shell_pids or spec.element_ids or spec.patches or spec.added_faces
        ):
            self.summary.setText(tr("No faces selected"))
            return
        try:
            # この電極だけを見る。まだ面を割り当てていない別の電極があっても、
            # いま編集している電極の面積は出せる
            info = session.electrode_info(spec.name)[0]
        except (FacetSetFemError, IndexError) as exc:
            self.summary.setText(str(exc).splitlines()[0])
            return
        self.summary.setText(
            tr(
                "{n} faces   total area {area}\nMean current density {j:.4g} A/m²",
                n=info["n_faces"], area=format_area(info["area"]), j=info["current_density"],
            )
        )


def tolerance_problem(text: str) -> str | None:
    """収束判定の入力の不備（無ければ None）。範囲はコアの検証（`Case::validate`）と同じ。"""
    try:
        value = float(text)
    except ValueError:
        return tr(
            'Cannot read "{text}" as a number. Enter a value such as 1e-10. The previous value was restored.',
            text=text.strip(),
        )
    if not 0.0 < value < 1.0:
        return tr(
            "The tolerance {value:g} is out of range. Use a relative residual between 0 and 1 "
            "(default 1e-10). The previous value was restored.",
            value=value,
        )
    return None


class SolverPage(ScrollPage):
    """ソルバ設定（SPEC §4.3）。"""

    def __init__(self, window: MainWindow) -> None:
        super().__init__()
        self.window = window
        self._loading = False

        layout = self.body_layout
        layout.addWidget(QLabel("<b>" + tr("Solver settings") + "</b>"))
        form = QFormLayout()
        self.tolerance = QLineEdit()
        self.max_iterations = QSpinBox()
        self.max_iterations.setRange(1, 10_000_000)
        self.preconditioner = QComboBox()
        self.preconditioner.addItems(preconditioner_choices())
        self.backend = QComboBox()
        self.backend.addItems(backend_choices())
        self.reference = QComboBox()
        self.reference.addItems(["zero_mean", "cathode_mean"])
        # 要素の次数（SPEC §2.3）。2 は 1 次の四面体メッシュを 2 次に上げて解く
        self.element_order = QComboBox()
        self.element_order.addItems(
            [tr("1 (first-order elements)"), tr("2 (second-order elements)")]
        )
        form.addRow(tr("Element order"), self.element_order)
        form.addRow(tr("Relative residual tolerance"), self.tolerance)
        form.addRow(tr("Maximum iterations"), self.max_iterations)
        form.addRow(tr("Preconditioner"), self.preconditioner)
        form.addRow(tr("Compute backend"), self.backend)
        form.addRow(tr("Potential reference"), self.reference)
        layout.addLayout(form)
        # 収束判定が読めない・範囲外のとき、その場で理由を出す（値は元に戻す）
        self.tolerance_error = QLabel("")
        self.tolerance_error.setWordWrap(True)
        self.tolerance_error.setStyleSheet(f"color: {STALE_COLOR};")
        self.tolerance_error.setVisible(False)
        layout.addWidget(self.tolerance_error)
        self.element_order_note = QLabel(
            tr(
                "Second order adds a node at the middle of every edge of the tetrahedral mesh "
                "(about 7-8 times as many nodes). The potential becomes quadratic inside each "
                "element, so the field is more accurate on the same mesh. Electrodes and regions "
                "stay as they are (SPEC §2.3)."
            )
        )
        self.element_order_note.setWordWrap(True)
        layout.addWidget(self.element_order_note)
        layout.addWidget(
            QLabel(
                tr(
                    "The gauge is fixed by projection (zero mean).\n"
                    "Pinning a single node distorts the solution and is not used (SPEC §2.2)."
                )
            )
        )
        # どのバックエンドでも答えはビット単位で同じ（CLAUDE.md 設計原則 1）。
        # 迷わせないよう、そのことと GPU の有無をここに書いておく
        self.environment = QLabel(describe_gpus())
        self.environment.setWordWrap(True)
        layout.addWidget(
            QLabel(
                tr(
                    "Changing the backend does not change a single bit of the result (only the speed).\n"
                    "The preconditioner is different: it changes the iteration path, so the answer\n"
                    "varies within the tolerance. amg (algebraic multigrid) makes the iteration count\n"
                    "almost independent of mesh fineness, but each iteration costs more. It pays off\n"
                    "on fine meshes (SPEC §4.5)."
                )
            )
        )
        layout.addWidget(self.environment)
        layout.addStretch(1)

        self.tolerance.editingFinished.connect(self._commit)
        self.max_iterations.valueChanged.connect(self._commit)
        self.preconditioner.currentTextChanged.connect(self._commit)
        self.backend.currentTextChanged.connect(self._commit)
        self.reference.currentTextChanged.connect(self._commit)
        self.element_order.currentIndexChanged.connect(self._commit_element_order)

    def reload(self) -> None:
        study = self.window.session.study
        if study is None:
            return
        self._loading = True
        self.tolerance_error.setVisible(False)
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
        # もともと 2 次のメッシュ（10 節点の CTETRA）は 1 次に下げられないので選ばせない
        from_file = self.window.session.file_element_order()
        self.element_order.setCurrentIndex(max(study.element_order, from_file) - 1)
        self.element_order.setEnabled(from_file == 1)
        self._loading = False

    def _commit(self) -> None:
        study = self.window.session.study
        if study is None or self._loading:
            return
        self.window.checkpoint(tr("Change solver settings"))
        problem = tolerance_problem(self.tolerance.text())
        if problem is None:
            study.solver.tolerance = float(self.tolerance.text())
        else:
            # 以前は負の値や 1 以上も黙って受け付け（F5 を押すまで何も言わない）、
            # 数値でなければ黙って元に戻していた
            self.tolerance.setText(f"{study.solver.tolerance:g}")
        self.tolerance_error.setText(problem or "")
        self.tolerance_error.setVisible(problem is not None)
        study.solver.max_iterations = self.max_iterations.value()
        study.solver.preconditioner = self.preconditioner.currentText()
        study.solver.backend = self.backend.currentText()
        study.output.reference_potential = self.reference.currentText()
        # ツリーは作り直さないので、結果が古くなったかをここで見直す
        self.window.refresh_result_status()

    def _commit_element_order(self) -> None:
        """要素の次数を変える。メッシュを入れ替えるので、結果は捨てて描き直す。"""
        study = self.window.session.study
        if study is None or self._loading:
            return
        order = self.element_order.currentIndex() + 1
        if order == study.element_order:
            return
        self.window.checkpoint(tr("Change the element order"))
        study.element_order = order
        self.window.apply_element_order()


class ResultPage(ScrollPage):
    """結果の表示設定（SPEC §6.2 の手順 6）。"""

    def __init__(self, window: MainWindow) -> None:
        super().__init__()
        self.window = window
        self._loading = False
        #: 稜線の既定を決めたときのモデル（開き直したときだけ入れ直す）
        self._mesh_key = None
        #: 「形状表示に戻す」を押す前の見方（表示量・表示モード・流線）。
        #: 覚えている間はボタンが「結果の表示に戻す」になる
        self._saved_view: dict | None = None

        layout = self.body_layout
        layout.addWidget(QLabel("<b>" + tr("Display") + "</b>"))
        # 結果を出したときから条件が変わっていたら、ページの先頭で知らせる
        self.stale_label = QLabel("")
        self.stale_label.setWordWrap(True)
        self.stale_label.setStyleSheet(
            f"color: {STALE_COLOR}; background: #fff3e6; border: 1px solid {STALE_COLOR};"
            " padding: 6px;"
        )
        self.stale_label.setVisible(False)
        layout.addWidget(self.stale_label)
        form = QFormLayout()
        self.field = QComboBox()
        self.mode = QComboBox()
        for label, key in VIEW_MODES:
            self.mode.addItem(tr(label), key)
        self.cmap = QComboBox()
        self.cmap.addItems(COLORMAPS)
        form.addRow(tr("Quantity"), self.field)
        form.addRow(tr("View mode"), self.mode)
        form.addRow(tr("Colour map"), self.cmap)
        layout.addLayout(form)

        self.shape_button = QPushButton(tr(SHOW_SHAPE_LABEL))
        self.shape_button.clicked.connect(self.toggle_shape)
        layout.addWidget(self.shape_button)

        self.fit_range = QCheckBox(tr("Fit the colour range to the shown regions"))
        self.fit_range.setToolTip(
            tr(
                "Normally the colour range covers the whole model, so the same colour means the\n"
                "same value everywhere. Turn this on to use only the regions that are not hidden\n"
                "(e.g. to look inside the brain when the skin has much larger values)."
            )
        )
        self.edges = QCheckBox(tr("Show element edges"))
        self.nodes = QCheckBox(tr("Show nodes"))
        self.nodes.setToolTip(
            tr("Draws nodes as points. On fine meshes they merge, so zoom in.")
        )
        layout.addWidget(self.fit_range)
        layout.addWidget(self.edges)
        layout.addWidget(self.nodes)

        slice_box = self.slice_box = QGroupBox(tr("Slice / clip"))
        slice_layout = QVBoxLayout(slice_box)
        self.slice_normal = QComboBox()
        self.slice_normal.addItems(["x", "y", "z"])
        self.slice_position = QSlider(Qt.Orientation.Horizontal)
        self.slice_position.setRange(0, 100)
        self.slice_position.setValue(50)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("Normal")))
        row.addWidget(self.slice_normal)
        slice_layout.addLayout(row)
        slice_layout.addWidget(QLabel(tr("Position")))
        slice_layout.addWidget(self.slice_position)
        layout.addWidget(slice_box)

        iso_box = self.iso_box = QGroupBox(tr("Isosurface"))
        iso_layout = QVBoxLayout(iso_box)
        self.iso_level = QSlider(Qt.Orientation.Horizontal)
        self.iso_level.setRange(0, 100)
        self.iso_level.setValue(50)
        iso_layout.addWidget(QLabel(tr("Threshold (fraction of the value range)")))
        iso_layout.addWidget(self.iso_level)
        # 等値面を出している間だけ、しきい値の実際の値を出す。以前はステータスバーに
        # 出していて、全体表示に戻しても次のメッセージまで残っていた
        self.iso_value = QLabel("")
        iso_layout.addWidget(self.iso_value)
        layout.addWidget(iso_box)

        stream_box = self.stream_box = QGroupBox(tr("Streamlines"))
        stream_layout = QVBoxLayout(stream_box)
        self.streamlines = QCheckBox(tr("Overlay current streamlines"))
        self.n_seeds = QSpinBox()
        self.n_seeds.setRange(1, 500)
        self.n_seeds.setValue(24)
        stream_layout.addWidget(self.streamlines)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("Count")))
        row.addWidget(self.n_seeds)
        stream_layout.addLayout(row)
        layout.addWidget(stream_box)

        self.note = QLabel(tr(NODAL_VALUE_WARNING))
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
        self.fit_range.toggled.connect(self._refresh)
        self.streamlines.toggled.connect(self._refresh)
        self.slice_position.valueChanged.connect(self._refresh)
        self.iso_level.valueChanged.connect(self._refresh)
        self.n_seeds.valueChanged.connect(self._refresh)
        # 形状表示の間に見方を自分で選び直したら、覚えていた見方は捨てる
        # （もう別の見方に移っている）。稜線やカラーマップは見方に含めない
        self.field.currentIndexChanged.connect(self._forget_saved_view)
        self.mode.currentIndexChanged.connect(self._forget_saved_view)
        # 断面・クリップを選んだとき、法線を変えたときは、それが見える向きに視点を向ける
        self.mode.currentIndexChanged.connect(self._face_section)
        self.mode.currentIndexChanged.connect(self._update_enabled)
        self.streamlines.toggled.connect(self._update_enabled)
        self.slice_normal.currentIndexChanged.connect(self._face_section)
        self.streamlines.toggled.connect(self._forget_saved_view)
        self._update_shape_button()
        self._update_enabled()

    def reload(self) -> None:
        session = self.window.session
        if session.solution is None:
            self._saved_view = None  # モデルを開き直して結果が無くなった
        self._loading = True
        self._apply_size_defaults()
        previous = self.field.currentData()
        self.field.clear()
        for f in session.available_fields():
            self.field.addItem(tr(f.label), f.key)
        if previous is not None:
            index = self.field.findData(previous)
            if index >= 0:
                self.field.setCurrentIndex(index)
        self.summary.setPlainText(
            session.result_summary() if session.study is not None else ""
        )
        self._loading = False
        self._update_shape_button()
        self._update_enabled()
        self.update_part_controls()

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
        # 新しい結果が出た。形状表示の前に見ていたのは古い結果の見方なので捨てる
        self._saved_view = None
        if self.field.currentData() == SHAPE_FIELD:
            index = self.field.findData(DEFAULT_RESULT_FIELD)
            if index >= 0:
                self.field.setCurrentIndex(index)
        self._update_shape_button()

    def show_iso_level(self, text: str) -> None:
        """等値面のしきい値の実際の値（等値面を出していないときは空）。"""
        self.iso_value.setText(text)

    def show_staleness(self, changes: list[str]) -> None:
        """結果が古い条件のものなら帯を出す（`MainWindow.refresh_result_status` から）。"""
        self.stale_label.setText(f"⚠ {stale_result_message(changes)}" if changes else "")
        self.stale_label.setVisible(bool(changes))

    def _view(self) -> dict:
        """いまの見方（形状表示との行き来で覚えて戻すもの）。"""
        return {
            "field": self.field.currentData(),
            "mode": self.mode.currentData(),
            "streamlines": self.streamlines.isChecked(),
        }

    def _is_plain_shape(self) -> bool:
        """色分けなし・全体表示・流線なし（解く前の見た目）か。"""
        return self._view() == {"field": SHAPE_FIELD, "mode": "volume", "streamlines": False}

    def _apply_view(self, view: dict) -> None:
        """見方をまとめて変え、描き直しは 1 回だけにする（1 つずつだと 3 回走る）。"""
        self._loading = True
        try:
            index = self.field.findData(view["field"])
            self.field.setCurrentIndex(index if index >= 0 else 0)
            self.mode.setCurrentIndex(max(self.mode.findData(view["mode"]), 0))
            self.streamlines.setChecked(view["streamlines"])
        finally:
            self._loading = False
        self._update_shape_button()
        self.window.refresh_view()

    def toggle_shape(self) -> None:
        """ボタン 1 つで、形状表示と、その前に見ていた結果の表示を行き来する。"""
        if self._saved_view is not None:
            self.show_saved_view()
        else:
            self.show_shape()

    def show_shape(self) -> None:
        """解く前の見た目（色分けなし・全体表示・流線なし）に戻す。

        結果は捨てない（表示を切り替えるだけ）。**直前の見方（表示量・表示モード・流線）を
        覚えておき**、ボタンの「結果の表示に戻す」でそのまま戻れるようにする。
        カラーマップ・稜線・断面の位置などは見方の好みなので触らない。
        """
        if not self._is_plain_shape():
            self._saved_view = self._view()
        self._apply_view({"field": SHAPE_FIELD, "mode": "volume", "streamlines": False})

    def show_saved_view(self) -> None:
        """「形状表示に戻す」を押す前の見方に戻す。解き直しは要らない。"""
        view, self._saved_view = self._saved_view, None
        if view is not None:
            self._apply_view(view)

    def _update_enabled(self, *_args) -> None:
        """いまの表示モードで効く欄だけを押せるようにする。

        以前は全体表示の間も断面や等値面のスライダーが動かせ、動かしても何も変わらなかった。
        押せない欄には、どうすれば使えるかをツールチップで出す。
        """
        mode = self.mode.currentData()
        solved = self.window.session.solution is not None
        for box, on, hint in (
            (self.slice_box, mode in ("slice", "clip"), tr("Available in the Slice or Clip view mode.")),
            (self.iso_box, mode == "isosurface", tr("Available in the Isosurface view mode.")),
            (self.stream_box, solved, tr("Available after solving (Run → Solve, F5).")),
        ):
            box.setEnabled(on)
            box.setToolTip("" if on else hint)
        self.n_seeds.setEnabled(self.streamlines.isChecked())

    def update_part_controls(self) -> None:
        """領域の見せ方を変えたとき。色の範囲の切り替えは、隠した・透かした領域があるときだけ効く。"""
        customised = self.window.session.mesh is not None and self.window.session.regions_customised()
        self.fit_range.setEnabled(customised)
        self.fit_range.setToolTip(
            self.fit_range.toolTip() if customised
            else tr("Available when some regions are transparent or hidden (eye icons in the tree).")
        )

    def _face_section(self, *_args) -> None:
        mode = self.mode.currentData()
        if self._loading:
            return
        if mode in ("slice", "clip"):
            self.window.view_section(mode, self.slice_normal.currentText())
        else:
            self.window.leave_section()

    def _forget_saved_view(self, *_args) -> None:
        if self._loading:
            return
        self._saved_view = None
        self._update_shape_button()

    def _update_shape_button(self) -> None:
        if self._saved_view is not None:
            field = field_by_key(self._saved_view["field"])
            name = tr(field.label) if field is not None else self._saved_view["field"]
            self.shape_button.setText(tr(SHOW_RESULT_LABEL))
            self.shape_button.setToolTip(
                tr("Go back to the view before the shape view ({name}). No need to solve again.", name=name)
            )
            self.shape_button.setEnabled(True)
            return
        self.shape_button.setText(tr(SHOW_SHAPE_LABEL))
        self.shape_button.setToolTip(
            tr(
                "Turn off colouring, show the whole body and remove streamlines. The result is kept;\n"
                "press again to go back to the previous view."
            )
        )
        # すでに解く前の見た目なら、押しても何も変わらない
        self.shape_button.setEnabled(not self._is_plain_shape())

    def options(self) -> dict:
        return {
            "field": self.field.currentData() or SHAPE_FIELD,
            "mode": self.mode.currentData() or "volume",
            "cmap": self.cmap.currentText(),
            "edges": self.edges.isChecked(),
            "fit_range": self.fit_range.isChecked(),
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
