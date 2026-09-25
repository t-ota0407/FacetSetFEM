"""GUI のセッション状態（SPEC §6）。

**Qt に依存しない**。GUI にロジックを置かないという方針（SPEC §7）に従い、
「何を読み込んでいて、どんな設定で、結果はどうだったか」をここに集約し、
Qt 側はこれを表示・編集するだけにする。こうするとヘッドレスでテストできる。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pyvista as pv

from .. import Case, GfemError, Mesh, Solution, Study, read_nastran
from ..study import ElectrodeSpec, Material, SurfacePatch
from ..tissues import TISSUES

__all__ = ["DisplayField", "DISPLAY_FIELDS", "SHAPE_FIELD", "Session"]


#: 色分けせず形だけを見せる表示（既定）
SHAPE_FIELD = "shape"


@dataclass(frozen=True)
class DisplayField:
    """3D ビューで切り替えられる表示量（SPEC §6.2 の手順 6）。"""

    key: str
    label: str
    #: "point" なら節点値、"cell" なら要素値
    location: str
    #: 結果が無くても表示できるか
    needs_solution: bool = True
    #: 等値面を取るときに使う量の名前。等値面にできない量は None
    iso_quantity: str | None = None

    @property
    def is_element_value(self) -> bool:
        return self.location == "cell"


#: 表示量の一覧。要素値と節点値の両方を出す（SPEC §5.1）
#:
#: 先頭が既定。**PID で色分けしない**——PID は領域を識別する番号で、
#: 大小に意味が無い。連続的なカラーマップに載せると「51 は 62 より小さい」
#: という無い情報を描いてしまう（頭部モデルは 62 領域ある）。
#: 形は陰影と稜線で見せ、色は物理量のためにとっておく。
DISPLAY_FIELDS: list[DisplayField] = [
    DisplayField(SHAPE_FIELD, "形状（色分けなし）", "none", needs_solution=False),
    DisplayField("PID", "領域 (PID) で塗り分け", "cell", needs_solution=False),
    DisplayField("phi [V]", "電位 φ [V]", "point", iso_quantity="phi"),
    DisplayField("|E| [V/m]", "電界 |E| [V/m]（要素値）", "cell", iso_quantity="e_magnitude"),
    DisplayField("|J| [A/m2]", "電流密度 |J| [A/m²]（要素値）", "cell", iso_quantity="j_magnitude"),
    DisplayField(
        "power density [W/m3]",
        "電力密度 σ|E|² [W/m³]（要素値）",
        "cell",
        iso_quantity="power_density",
    ),
    DisplayField(
        "|E| (nodal) [V/m]", "電界 |E| [V/m]（節点値・平滑化）", "point", iso_quantity="e_magnitude"
    ),
    DisplayField(
        "|J| (nodal) [A/m2]",
        "電流密度 |J| [A/m²]（節点値・平滑化）",
        "point",
        iso_quantity="j_magnitude",
    ),
]


#: 既定で解いたあとに切り替える表示量（電流密度）
DEFAULT_RESULT_FIELD = "|J| [A/m2]"


#: 表示量ごとの配列の取り出し方（`Session.grid` が貼るものと同じ）
FIELD_ARRAYS = {
    "phi [V]": lambda r: r.potential,
    "|E| [V/m]": lambda r: r.e_magnitude,
    "|J| [A/m2]": lambda r: r.j_magnitude,
    "power density [W/m3]": lambda r: r.power_density,
    "|E| (nodal) [V/m]": lambda r: r.nodal_average("e_magnitude"),
    "|J| (nodal) [A/m2]": lambda r: r.nodal_average("j_magnitude"),
}


def field_by_key(key: str) -> DisplayField | None:
    """表示量を鍵から引く。"""
    for f in DISPLAY_FIELDS:
        if f.key == key:
            return f
    return None

#: 領域統計の表の見出し（SPEC §5.2）
REGION_STATS_COLUMNS = [
    "PID",
    "名前",
    "σ [S/m]",
    "要素数",
    "体積 [cm³]",
    "電力 [mW]",
    "|J| 最大",
    "|J| 平均",
    "|J| 中央",
    "|J| 99%",
    "|E| 最大",
    "|E| 99%",
    "ホットスポット EID",
]

#: 節点値は平滑化されているという注意（SPEC §5.1 で GUI 上に明示することになっている）
NODAL_VALUE_WARNING = (
    "節点値は体積重み平均で平滑化されています。最大値の評価には要素値を使ってください。"
)

#: 組織物性の目安 [S/m]。GUI の入力補助に使う（出典を明記できるようになるまでは目安扱い）
TISSUE_HINTS: dict[str, float] = {name: t.sigma for name, t in TISSUES.items()}


#: 元に戻せる回数の上限。1 段は Study 1 つぶんの複製で、
#: 数百 KB 程度（メッシュは含まない）なので余裕を持って取ってある。
UNDO_DEPTH = 100


class Session:
    """開いているモデルと解析条件、直近の結果。"""

    def __init__(self) -> None:
        self.study: Study | None = None
        self.mesh: Mesh | None = None
        self.solution: Solution | None = None
        self._grid: pv.UnstructuredGrid | None = None
        #: 表示用の外表面（SPEC §6.3）。メッシュを開き直すまで使い回す
        self._surface: pv.PolyData | None = None
        # Undo / Redo（SPEC §8 M7）。
        # 解析条件は Study 1 つに閉じていて、メッシュも結果も含まないので、
        # 操作ごとに丸ごと複製して積むだけで済む。差分を持つ設計は
        # 「どの編集を記録し忘れたか」を追う手間に見合わない
        self._undo: list[tuple[str, Study]] = []
        self._redo: list[tuple[str, Study]] = []

    # --- Undo / Redo --------------------------------------------------------

    def checkpoint(self, label: str) -> None:
        """これから変える直前の状態を積む。編集を始める**前**に呼ぶこと。

        GUI の入力欄は 1 文字打つたびに値を反映するので、素直に積むと
        「1 文字ぶん戻る」undo が大量にできてしまう。**同じ操作が続いている間は
        まとめる**（直前の checkpoint と名前が同じなら積み直さない）。
        別の操作を挟めば名前が変わるので、そこで区切られる。
        """
        if self.study is None:
            return
        if self._undo and self._undo[-1][0] == label:
            return  # 同じ操作の続き。最初の状態だけ残す
        self._undo.append((label, copy.deepcopy(self.study)))
        if len(self._undo) > UNDO_DEPTH:
            self._undo.pop(0)
        self._redo.clear()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo_label(self) -> str:
        return self._undo[-1][0] if self._undo else ""

    def redo_label(self) -> str:
        return self._redo[-1][0] if self._redo else ""

    def undo(self) -> str:
        """1 つ前の状態に戻す。戻した操作の名前を返す。"""
        if not self._undo or self.study is None:
            return ""
        label, previous = self._undo.pop()
        self._redo.append((label, copy.deepcopy(self.study)))
        self.study = previous
        return label

    def redo(self) -> str:
        """取り消した操作をやり直す。やり直した操作の名前を返す。"""
        if not self._redo or self.study is None:
            return ""
        label, following = self._redo.pop()
        self._undo.append((label, copy.deepcopy(self.study)))
        self.study = following
        return label

    def clear_history(self) -> None:
        """履歴を捨てる（モデルやケースを開き直したとき）。"""
        self._undo.clear()
        self._redo.clear()

    # --- 読み込み -----------------------------------------------------------

    def open_mesh(self, path: str | Path, length_unit: str) -> Study:
        """メッシュを読み、既定のケースを組み立てる（SPEC §6.2 の手順 1）。"""
        self.clear_history()
        path = Path(path)
        self.mesh = read_nastran(path, length_unit=length_unit)
        self.solution = None
        self._grid = None
        self._surface = None
        self.study = Study(
            mesh_file=path,
            length_unit=length_unit,
            materials=[Material(pid=pid, sigma=0.0) for pid in self.mesh.solid_pids()],
        )
        # σ 未設定を 0 で表しておくと validate() が拾ってくれる
        return self.study

    def open_case(self, path: str | Path) -> Study:
        """ケースファイル（YAML）を開き、参照しているメッシュも読む。"""
        self.clear_history()
        study = Study.load(path)
        self.study = study
        self.mesh = study.read_mesh()
        self.solution = None
        self._grid = None
        self._surface = None
        # メッシュにあるのに YAML に無い PID を補う
        known = {m.pid for m in study.materials}
        for pid in self.mesh.solid_pids():
            if pid not in known:
                study.materials.append(Material(pid=pid, sigma=0.0))
        study.materials.sort(key=lambda m: m.pid)
        return study

    def save_case(self, path: str | Path) -> None:
        self._require_study().save(path)

    # --- 設定 ---------------------------------------------------------------

    def set_conductivity(self, pid: int, sigma: float, name: str | None = None) -> None:
        study = self._require_study()
        for m in study.materials:
            if m.pid == pid:
                m.sigma = sigma
                if name is not None:
                    m.name = name
                return
        study.materials.append(Material(pid=pid, sigma=sigma, name=name or ""))
        study.materials.sort(key=lambda m: m.pid)

    def conductivity_of(self, pid: int) -> float:
        for m in self._require_study().materials:
            if m.pid == pid:
                return m.sigma
        return 0.0

    def region_name(self, pid: int) -> str:
        for m in self._require_study().materials:
            if m.pid == pid:
                return m.name
        return ""

    def add_electrode(
        self,
        name: str,
        role: str = "anode",
        current: float = 2.0e-3,
        shell_pids: list[int] | None = None,
    ) -> ElectrodeSpec:
        study = self._require_study()
        self._require_free_name(name)
        spec = ElectrodeSpec(
            name=name, role=role, current=current, shell_pids=list(shell_pids or [])
        )
        study.electrodes.append(spec)
        return spec

    def rename_electrode(self, old: str, new: str) -> ElectrodeSpec:
        """電極の名前を変える。**ほかの電極と同じ名前にはできない。**

        電極名は結果とケースファイルで電極を指す鍵で、GUI のツリーや削除も
        名前で引いている。重複を許すと、削除で両方消える・同名の陽極と陰極が
        同じ面を指しても重なりとして捕まらない、といった壊れ方をする。
        """
        spec = self.electrode(old)
        if spec is None:
            raise GfemError(f"[設定エラー] 電極 {old} がありません")
        new = new.strip()
        if new == old:
            return spec
        self._require_free_name(new)
        spec.name = new
        return spec

    def _require_free_name(self, name: str) -> None:
        if not name.strip():
            raise GfemError(
                "[設定エラー] 電極の名前が空です\n  対処: E1・E2 のような名前を付けてください"
            )
        if self.electrode(name) is not None:
            raise GfemError(
                f"[設定エラー] 電極 {name} はすでにあります"
                "\n  対処: 電極ごとに違う名前を付けてください"
                "（名前は結果とケースファイルで電極を指す鍵です）"
            )

    def remove_electrode(self, name: str) -> None:
        study = self._require_study()
        study.electrodes = [e for e in study.electrodes if e.name != name]

    def electrode(self, name: str) -> ElectrodeSpec | None:
        for e in self._require_study().electrodes:
            if e.name == name:
                return e
        return None

    # --- 実行 ---------------------------------------------------------------

    def build_case(self) -> Case:
        """`gfem.Case` を組み立てる。σ が未設定（0）のままなら分かりやすく止める。"""
        study = self._require_study()
        missing = [m.pid for m in study.materials if m.sigma <= 0.0]
        if missing:
            raise GfemError(
                "[設定エラー] 導電率が未設定の領域があります: PID="
                + ", ".join(str(p) for p in missing)
                + "\n  対処: モデルツリーの領域を選んで σ [S/m] を入力してください"
            )
        return study.build_case(self._require_mesh())

    def validate(self) -> list[str]:
        """実行前の整合性チェック（SPEC §6.2 の手順 4）。警告の一覧を返す。"""
        return self.build_case().validate()

    def electrode_info(self, name: str | None = None) -> list[dict]:
        """電極の合計面積と平均電流密度（SPEC §6.2 の手順 3）。

        `name` を指定すると**その電極だけ**を見る。GUI は電極を 1 つずつ設定して
        いくので、まだ面を割り当てていない別の電極のせいで、いま編集している
        電極の面積が出せなくなるのを避けるため。
        """
        study = self._require_study()
        if name is None:
            return self.build_case().electrode_info()
        kept = study.electrodes
        try:
            study.electrodes = [e for e in kept if e.name == name]
            return self.build_case().electrode_info()
        finally:
            study.electrodes = kept

    def solve(self, progress: Callable[[int, float], bool] | None = None) -> Solution:
        self.set_solution(self.build_case().solve(progress=progress))
        return self.solution

    def set_solution(self, solution: Solution) -> None:
        """外（ワーカースレッド）で解いた結果を取り込む。"""
        self.solution = solution
        self._grid = None  # 表示用データに結果を貼り直す
        self._surface = None

    # --- 表示用データ -------------------------------------------------------

    def grid(self) -> pv.UnstructuredGrid:
        """表示用の `UnstructuredGrid`（結果があれば貼り付け済み）。

        SPEC §6.3 の「小規模モデル用エスケープハッチ」。節点数が増えたら
        Rust 側で表示用データを抽出する経路に切り替える。
        """
        if self._grid is not None:
            return self._grid
        mesh = self._require_mesh()
        points = mesh.points()
        # CTETRA / CHEXA / CPENTA がそのまま VTK のセル型に対応する
        cells, types = mesh.solid_cells()
        grid = pv.UnstructuredGrid(cells, types, points)
        grid.cell_data["PID"] = mesh.solid_pid_array()

        result = self.solution
        if result is not None:
            grid.point_data["phi [V]"] = result.potential
            grid.point_data["|E| (nodal) [V/m]"] = result.nodal_average("e_magnitude")
            grid.point_data["|J| (nodal) [A/m2]"] = result.nodal_average("j_magnitude")
            grid.cell_data["E [V/m]"] = result.electric_field
            grid.cell_data["J [A/m2]"] = result.current_density
            grid.cell_data["|E| [V/m]"] = result.e_magnitude
            grid.cell_data["|J| [A/m2]"] = result.j_magnitude
            grid.cell_data["power density [W/m3]"] = result.power_density
        self._grid = grid
        return grid

    def region_grid(self, pid: int) -> pv.UnstructuredGrid:
        """指定した領域だけを取り出したメッシュ（ハイライト表示用）。"""
        grid = self.grid()
        return grid.extract_cells(np.flatnonzero(grid.cell_data["PID"] == pid))

    def shell_polydata(self, pids: list[int] | None = None) -> pv.PolyData:
        """シェル要素（電極候補面）のポリゴン。`pids` を指定するとその PID だけ。"""
        mesh = self._require_mesh()
        tris, shell_pids, _ids = mesh.shell_triangles()
        if pids is not None:
            keep = np.isin(shell_pids, list(pids))
            tris, shell_pids = tris[keep], shell_pids[keep]
        return self._polydata(mesh.points(), tris, {"PID": shell_pids})

    def electrode_polydata(self, name: str) -> pv.PolyData:
        """指定した電極の面（3D ビューでの色分け用。SPEC §6.2 の手順 3）。

        シェル要素で指定した面と、体表面から選んだパッチの両方を合わせて返す。
        """
        mesh = self._require_mesh()
        spec = self.electrode(name)
        if spec is None:
            return pv.PolyData()
        tris, shell_pids, ids = mesh.shell_triangles()
        keep = np.isin(shell_pids, spec.shell_pids) | np.isin(ids, spec.element_ids)
        pieces = [tris[keep]] if len(tris) else []
        # 体表面の面は、パッチ・手で足した面・手で外した面をまとめて
        # ソルバと同じ関数に解決させる（表示と計算がずれないように）
        try:
            pieces.append(self.electrode_faces(name)["triangles"])
        except GfemError:
            pass  # まだ面が 1 枚も無いなど。表示は空でよい
        pieces = [p for p in pieces if len(p)]
        if not pieces:
            return pv.PolyData()
        return self._polydata(mesh.points(), np.vstack(pieces), {})

    # --- 体表面から選ぶ電極（SPEC §11.1）------------------------------------

    def surface_pid_choices(self) -> list[dict]:
        """外表面に顔を出している PID とその面積（面積の大きい順）。

        人体モデルでは首の切断面などで内部組織も外表面に現れるので、
        **どの PID が皮膚かを面積で見当づける**ために出す。
        """
        return self._require_mesh().surface_pids()

    def default_surface_pid(self) -> int | None:
        """パッチの PID 絞り込みの既定値。いちばん広い面（ふつう皮膚）。"""
        choices = self.surface_pid_choices()
        return choices[0]["pid"] if choices else None

    def is_on_the_model(self, point, margin_ratio: float = 0.05) -> bool:
        """クリックされた点がモデルの上とみなせるか。

        3D ビューの背景をクリックすると、ピッカーは遠くの点を返す。それを
        そのまま電極の中心にすると、いちばん近い面が選ばれて**まるで見当違いの
        場所に電極が置かれる**ので、ここで弾く。
        """
        mesh = self._require_mesh()
        box = mesh.bounding_box()
        if box is None:
            return False
        lo, hi = box
        point = [float(v) for v in point]
        if not all(np.isfinite(point)):
            return False  # ピックに失敗すると NaN が返ることがある
        diagonal = float(np.linalg.norm(np.array(hi) - np.array(lo)))
        margin = diagonal * margin_ratio
        return all(lo[k] - margin <= point[k] <= hi[k] + margin for k in range(3))

    def triangles_to_polydata(self, triangles) -> pv.PolyData:
        """`pick_patch` などが返した三角形を、そのまま描ける形にする。"""
        return self._polydata(self._require_mesh().points(), np.asarray(triangles), {})

    def pick_patch(self, center, radius: float, pid: int | None = None) -> dict:
        """体表面の 1 点を中心に、半径の中でつながっている面を選ぶ。"""
        return self._require_mesh().pick_patch(
            [float(v) for v in center], float(radius), None if pid is None else int(pid)
        )

    def add_patch(self, name: str, center, radius: float, pid: int | None = None) -> SurfacePatch:
        """電極にパッチを 1 枚足す。"""
        spec = self.electrode(name)
        if spec is None:
            raise GfemError(f"[設定エラー] 電極 {name} がありません")
        patch = SurfacePatch(
            center=tuple(float(v) for v in center),
            radius=float(radius),
            pid=None if pid is None else int(pid),
        )
        spec.patches.append(patch)
        return patch

    def remove_patch(self, name: str, index: int) -> None:
        """電極のパッチを 1 枚外す。"""
        spec = self.electrode(name)
        if spec is not None and 0 <= index < len(spec.patches):
            del spec.patches[index]

    # --- 面を 1 枚ずつ塗る / 消す（SPEC §11.1）------------------------------

    def paint_faces(self, name: str, center, radius: float, pid: int | None, erase: bool) -> int:
        """クリック（またはドラッグ）した範囲の面を、電極に足す／から外す。

        **面の一覧そのものではなく差分で持つ。** パッチの「中心と半径」という
        読める指定を残したまま縁だけ直せるようにするため。同じ面が
        `added_faces` と `removed_faces` の両方に入らないよう、必ず打ち消す。

        戻り値は実際に変わった面の枚数（0 なら何も起きていない）。
        """
        spec = self.electrode(name)
        if spec is None:
            raise GfemError(f"[設定エラー] 電極 {name} がありません")
        refs = {
            (int(e), int(f))
            for e, f in self.pick_patch(center, radius, pid)["face_refs"]
        }
        if not refs:
            return 0

        added = dict.fromkeys(tuple(f) for f in spec.added_faces)
        removed = dict.fromkeys(tuple(f) for f in spec.removed_faces)
        before = (set(added), set(removed))
        if erase:
            removed.update(dict.fromkeys(refs))
            for ref in refs:
                added.pop(ref, None)
        else:
            added.update(dict.fromkeys(refs))
            for ref in refs:
                removed.pop(ref, None)

        spec.added_faces = list(added)
        spec.removed_faces = list(removed)
        return len(set(added) ^ before[0]) + len(set(removed) ^ before[1])

    def clear_hand_edits(self, name: str) -> None:
        """手で足した面・外した面を捨てて、パッチだけの状態に戻す。"""
        spec = self.electrode(name)
        if spec is not None:
            spec.added_faces = []
            spec.removed_faces = []

    def electrode_faces(self, name: str) -> dict:
        """電極の面を**ソルバと同じ経路で**解決する（面数・面積・三角形）。

        表示だけ別に数えると、「25 cm² のつもりの電極が実は違う面積だった」が
        静かに起きる。導電率が未設定でも呼べる（SPEC §6.2 の手順 3）。
        """
        spec = self.electrode(name)
        if spec is None:
            return {"n_faces": 0, "area": 0.0, "triangles": np.zeros((0, 3), dtype=np.uint32)}
        return self._require_mesh().resolve_electrode_faces(
            shell_pids=list(spec.shell_pids),
            element_ids=list(spec.element_ids),
            patches=[p.to_dict() for p in spec.patches],
            added=[tuple(f) for f in spec.added_faces],
            removed=[tuple(f) for f in spec.removed_faces],
            name=spec.name,
        )

    def surface_polydata(self) -> pv.PolyData:
        """外表面の三角形（表示用の軽量データ。SPEC §6.3）。

        **3D ビューの「全体」表示はこれを使う。** 体積メッシュをそのまま渡すと、
        頭部モデル（411 万セル）で 1 回の描画に 2.7 秒かかるうえ、法線が無いので
        **のっぺりした影絵にしかならない**。外表面（87,812 面）なら 0.09 秒で、
        陰影が出て形が読める。
        """
        if self._surface is not None:
            return self._surface
        mesh = self._require_mesh()
        tris = mesh.surface_triangles()
        owners = mesh.surface_triangle_elements()
        poly = self._polydata(mesh.points(), tris, {"PID": mesh.solid_pid_array()[owners]})
        poly.cell_data["__owner"] = owners
        # **法線を自分で計算する。** 無いままだと（環境によっては）陰影が付かず、
        # 頭部モデルが**のっぺりした影絵**にしかならない。外表面の三角形は
        # すでに外向きの順序で並んでいるので、向きを推定し直す必要はない。
        # 点を分裂させると節点番号との対応が崩れるので split_vertices は使わない
        self._surface = poly.compute_normals(
            cell_normals=True,
            point_normals=True,
            split_vertices=False,
            consistent_normals=False,
            auto_orient_normals=False,
        )
        self._surface.cell_data["__owner"] = owners
        return self._surface

    def display_surface(self, field: str | None) -> pv.PolyData:
        """外表面に、指定した表示量を貼りつけたもの。

        要素値（|J| など）は**その面を持つソリッド要素の値**を貼る。節点値は
        節点番号がそのまま使えるので、配列をそのまま持たせる。
        """
        poly = self.surface_polydata()
        spec = field_by_key(field) if field else None
        if spec is None or self.solution is None or field in poly.array_names:
            return poly
        grid = self.grid()
        if spec.is_element_value and field in grid.cell_data:
            poly.cell_data[field] = grid.cell_data[field][poly.cell_data["__owner"]]
        elif field in grid.point_data:
            poly.point_data[field] = grid.point_data[field]
        return poly

    def field_range(self, field: str | None) -> tuple[float, float] | None:
        """表示量の値域（**モデル全体**）。

        外表面だけ、あるいは断面だけで色を決めると、**同じ色が場所によって
        違う値を指す**ことになる。全体の値域に固定すれば、全体表示と断面と
        等値面で色の意味が揃う。
        """
        if self.solution is None or field not in FIELD_ARRAYS:
            return None
        values = FIELD_ARRAYS[field](self.solution)
        return (float(np.min(values)), float(np.max(values)))

    def region_polydata(self, pid: int) -> pv.PolyData:
        """指定した領域が体表面に出している面（ハイライト表示用）。

        体積メッシュから切り出すより速く、見た目も「全体」表示と揃う。
        """
        poly = self.surface_polydata()
        return poly.extract_cells(np.flatnonzero(poly.cell_data["PID"] == pid))

    @staticmethod
    def _polydata(points, triangles, cell_arrays: dict) -> pv.PolyData:
        if len(triangles) == 0:
            return pv.PolyData()
        faces = np.hstack(
            [np.full((len(triangles), 1), 3, dtype=np.int64), triangles.astype(np.int64)]
        ).ravel()
        poly = pv.PolyData(points, faces)
        for name, values in cell_arrays.items():
            poly.cell_data[name] = values
        return poly

    # --- M3: 統計と可視化 ---------------------------------------------------

    def region_stats(self) -> list[dict]:
        """領域ごとの統計（SPEC §5.2）。まだ解いていなければ空。"""
        if self.solution is None:
            return []
        return self.solution.region_stats()

    def value_range(self, quantity: str) -> tuple[float, float]:
        """等値面のしきい値を決めるための値域。"""
        if self.solution is None:
            return (0.0, 1.0)
        return self.solution.value_range(quantity)

    def iso_surface(self, quantity: str, level: float) -> pv.PolyData:
        """等値面（Marching Tetrahedra、SPEC §6.3）。"""
        if self.solution is None:
            return pv.PolyData()
        points, triangles, scalars = self.solution.iso_surface(quantity, level)
        poly = self._polydata(points, triangles, {})
        if poly.n_cells:
            poly.point_data[quantity] = scalars
        return poly

    def streamlines(self, seeds, direction: str = "forward") -> pv.PolyData:
        """電流のストリームライン（SPEC §6.3）。折れ線をまとめた PolyData を返す。"""
        if self.solution is None or len(seeds) == 0:
            return pv.PolyData()
        lines = self.solution.streamlines(
            [list(map(float, s)) for s in seeds], direction=direction
        )
        lines = [line for line in lines if len(line) >= 2]
        if not lines:
            return pv.PolyData()
        points = np.vstack(lines)
        cells: list[int] = []
        offset = 0
        for line in lines:
            n = len(line)
            cells.append(n)
            cells.extend(range(offset, offset + n))
            offset += n
        poly = pv.PolyData()
        poly.points = points
        poly.lines = np.array(cells, dtype=np.int64)
        return poly

    def electrode_seed_points(self, name: str, count: int = 24):
        """電極面から少し内側に入った位置に、ストリームラインの種を撒く。

        面の上ちょうどだと要素の探索が境界に載って不安定になるので、
        モデルの中心へ向かってわずかに寄せる。
        """
        faces = self.electrode_polydata(name)
        if faces.n_cells == 0:
            return np.empty((0, 3))
        centres = np.asarray(faces.cell_centers().points)
        if len(centres) > count:
            step = max(1, len(centres) // count)
            centres = centres[::step][:count]
        grid = self.grid()
        model_centre = np.asarray(grid.center)
        diagonal = float(np.linalg.norm(np.array(grid.bounds[1::2]) - np.array(grid.bounds[0::2])))
        direction = model_centre - centres
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return centres + (direction / norms) * (1.0e-3 * diagonal)

    def default_seed_points(self, count: int = 24):
        """陽極があればそこから、無ければモデル中央付近から種を撒く。"""
        study = self.study
        if study is not None:
            for electrode in study.electrodes:
                if electrode.role == "anode":
                    seeds = self.electrode_seed_points(electrode.name, count)
                    if len(seeds):
                        return seeds
        grid = self.grid()
        return np.asarray([grid.center])

    def write_region_stats_csv(self, path) -> None:
        self._require_solution().write_region_stats_csv(path)

    def write_convergence_csv(self, path) -> None:
        self._require_solution().write_convergence_csv(path)

    def write_summary_json(self, path) -> None:
        """実行サマリ JSON（プロビナンス付き。SPEC §5.5）。"""
        text = None
        study = self.study
        if study is not None and study.path is not None and Path(study.path).exists():
            text = Path(study.path).read_text(encoding="utf-8")
        self._require_solution().write_summary_json(path, case_text=text)

    def write_vtu(self, path) -> None:
        self._require_solution().write_vtu(path)

    def available_fields(self) -> list[DisplayField]:
        """いま表示できる量。"""
        has = self.solution is not None
        return [f for f in DISPLAY_FIELDS if has or not f.needs_solution]

    # --- サマリ -------------------------------------------------------------

    def mesh_summary(self) -> str:
        """読み込みレポートの要約（SPEC §3.2 の単位確認表示を含む）。"""
        from .. import describe_bounding_box

        mesh = self._require_mesh()
        lines = [
            f"ファイル: {mesh.source}",
            f"長さ単位: {mesh.length_unit}",
            f"寸法: {describe_bounding_box(mesh)}",
            f"節点数: {mesh.n_nodes:,}",
            f"要素数: {mesh.n_elements:,}"
            + "（"
            + ", ".join(f"{k} {n:,}" for k, n in mesh.element_counts())
            + "）",
            f"領域 (PSOLID): {', '.join(str(p) for p in mesh.solid_pids())}",
            f"電極候補面 (PSHELL): {', '.join(str(p) for p in mesh.shell_pids())}",
        ]
        report = mesh.report
        for name, count, line in report.unsupported:
            lines.append(f"[未対応] {name} {count} 件（最初の出現: {line} 行目）")
        for w in report.warnings:
            lines.append(f"[警告] {w}")
        return "\n".join(lines)

    def result_summary(self) -> str:
        """結果と保存則チェックの要約（SPEC §5.3 / §5.4）。"""
        result = self.solution
        if result is None:
            return "（まだ解いていません）"
        c = result.conservation
        lines = [
            f"反復回数: {result.iterations}（相対残差 {result.residual:.3e}, {result.backend}）",
            f"電極間電位差 ΔV: {result.voltage:.6g} V",
            f"等価抵抗 R: {result.impedance:.6g} Ω",
            f"消費電力 P: {result.power:.6g} W",
            "",
            "電極の平均電位:",
        ]
        lines += [f"  {name}: {v:.6g} V" for name, v in result.electrode_potentials]
        lines += [
            "",
            "保存則チェック（SPEC §5.4）:",
            f"  電流保存 1ᵀf/max|f|: {c['current_balance']:.3e}",
            f"  エネルギー収支の相対誤差: {c['energy_relative_error']:.3e}",
            f"  離散残差 ‖Kφ−f‖/‖f‖: {c['discrete_residual']:.3e}",
        ]
        warnings = list(c["warnings"]) + [
            w for w in result.warnings if w not in c["warnings"]
        ]
        if warnings:
            lines.append("")
            lines += [f"[警告] {w}" for w in warnings]
        else:
            lines.append("  → すべて許容範囲内")
        timings = result.timings
        lines += [
            "",
            "所要時間 [秒]: " + ", ".join(f"{k} {v:.3f}" for k, v in timings.items()),
        ]
        return "\n".join(lines)

    def region_stats_rows(self) -> list[list[str]]:
        """領域統計を表に出すための行（SPEC §5.2）。"""
        rows = []
        for s in self.region_stats():
            j, e = s["j_magnitude"], s["e_magnitude"]
            rows.append(
                [
                    str(s["pid"]),
                    s["name"],
                    f"{s['sigma']:.4g}",
                    f"{s['n_elements']:,}",
                    # m³ → cm³（見出しの単位と必ず揃える。以前は ×1e6 のまま mm³ と書いていた）
                    f"{s['volume'] * 1e6:.4g}",
                    f"{s['power'] * 1e3:.4g}",
                    f"{j['max']:.4g}",
                    f"{j['mean']:.4g}",
                    f"{j['median']:.4g}",
                    f"{j['p99']:.4g}",
                    f"{e['max']:.4g}",
                    f"{e['p99']:.4g}",
                    str(s["hotspot"]["element_id"]),
                ]
            )
        return rows

    # --- 内部 ---------------------------------------------------------------

    def _require_solution(self) -> Solution:
        if self.solution is None:
            raise GfemError(
                "[設定エラー] まだ solve を実行していません"
                "\n  対処: 「実行 → solve」で計算してから書き出してください"
            )
        return self.solution

    def _require_mesh(self) -> Mesh:
        if self.mesh is None:
            raise GfemError(
                "[設定エラー] メッシュが読み込まれていません"
                "\n  対処: 「ファイル → モデルを開く」で .bdf / .nas を選んでください"
            )
        return self.mesh

    def _require_study(self) -> Study:
        if self.study is None:
            raise GfemError(
                "[設定エラー] モデルが開かれていません"
                "\n  対処: 「ファイル → モデルを開く」で .bdf / .nas を選んでください"
            )
        return self.study
