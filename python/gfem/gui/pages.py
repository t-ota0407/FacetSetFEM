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
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import GfemError, backend_choices, describe_gpus, preconditioner_choices
from .busy import REVEAL_DELAY_MS
from .state import (
    DEFAULT_RESULT_FIELD,
    NODAL_VALUE_WARNING,
    SHAPE_FIELD,
    TISSUE_HINTS,
)

if TYPE_CHECKING:
    from .main_window import MainWindow

__all__ = ["ElectrodePage", "MeshPage", "RegionPage", "ResultPage", "SolverPage"]

COLORMAPS = ["viridis", "plasma", "coolwarm", "turbo", "gray"]

#: 稜線・節点を既定で出す上限（表示する三角形の枚数）。
#:
#: これを超えると線が潰れて灰色の塊になり、形が**かえって見えなくなる**。
#: 実測で頭部モデルの外表面は 87,812 面。手動で入れることは妨げない
EDGE_LIMIT = 30_000

#: 3D ビューの表示モード（表示名, 内部名）
VIEW_MODES = [
    ("全体", "volume"),
    ("断面スライス", "slice"),
    ("クリップ", "clip"),
    ("等値面", "isosurface"),
]


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
        self.preconditioner.addItems(preconditioner_choices())
        self.backend = QComboBox()
        self.backend.addItems(backend_choices())
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

        self.shape_button = QPushButton("形状表示に戻す")
        self.shape_button.setToolTip(
            "色分けをやめ、全体表示にして流線も外します。結果は消えないので、\n"
            "表示量を選び直せばいつでも見直せます。"
        )
        self.shape_button.clicked.connect(self.show_shape)
        layout.addWidget(self.shape_button)

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

    def show_shape(self) -> None:
        """解く前の見た目（色分けなし・全体表示・流線なし）に戻す。

        結果は捨てない（表示を切り替えるだけ）。カラーマップ・稜線・断面の位置などは
        見方の好みなので触らない。3 つを 1 つずつ変えると描き直しが 3 回走るので、
        まとめて変えてから 1 回だけ描く。
        """
        self._loading = True
        self.field.setCurrentIndex(max(self.field.findData(SHAPE_FIELD), 0))
        self.mode.setCurrentIndex(self.mode.findData("volume"))
        self.streamlines.setChecked(False)
        self._loading = False
        self.window.refresh_view()

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
