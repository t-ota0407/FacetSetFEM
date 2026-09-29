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
from .._gfem import check_patch
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

#: 選ばれた面の面積が指定面積からこれ以上ずれたら警告する（相対値。SPEC §11.1）。
#:
#: 面は 1 枚単位で入るか外れるかなので、縁のぶんだけ必ずずれる。2 cm 角の電極を
#: 一辺 2.6 mm の皮膚の面（頭部モデル）に置くと数 % に収まる。これを超えるなら、
#: 電極に対してメッシュが粗すぎるか、曲面で接平面への投影が効いている
AREA_MISMATCH_WARNING = 0.10

#: 組織物性の目安 [S/m]。GUI の入力補助に使う（出典を明記できるようになるまでは目安扱い）
TISSUE_HINTS: dict[str, float] = {name: t.sigma for name, t in TISSUES.items()}


#: 元に戻せる回数の上限。1 段は Study 1 つぶんの複製で、
#: 数百 KB 程度（メッシュは含まない）なので余裕を持って取ってある。
UNDO_DEPTH = 100


#: 結果が古い条件のものだと知らせるときの色（結果ページの帯・ステータスバー・ツリー）
STALE_COLOR = "#b34700"


def stale_result_message(changes: list[str]) -> str:
    """結果が古いことの説明。`changes` は `Session.changes_since_solve()` の戻り値。"""
    if not changes:
        return ""
    return (
        "この結果は求解したときの条件のものです。その後に変わった条件: "
        f"{'、'.join(changes)}。いまの条件の結果を見るには F5（実行 → 求解）で解き直してください。"
    )


def _electrode_changes(old: list[ElectrodeSpec], new: list[ElectrodeSpec]) -> list[str]:
    """電極の変更を「電極 E1 の総電流」「電極 E3 を追加」のような言葉で並べる。

    数が同じなら並び順で対応づける。GUI の改名はその場で名前だけを書き換えるので、
    名前で対応づけると改名が「削除と追加」に見えてしまう。数が違えば名前で対応づける。
    """
    def parts(a: ElectrodeSpec, b: ElectrodeSpec) -> list[str]:
        what = []
        if a.role != b.role:
            what.append("役割")
        if a.current != b.current:
            what.append("総電流")
        if a.to_dict()["faces"] != b.to_dict()["faces"]:
            what.append("面")
        if a.distribution != b.distribution:
            what.append("荷重の配り方")
        return what

    out: list[str] = []
    if len(old) == len(new):
        for a, b in zip(old, new):
            what = parts(a, b)
            if a.name != b.name:
                what.insert(0, f"名前（{a.name} → {b.name}）")
            if what:
                out.append(f"電極 {b.name} の{'・'.join(what)}")
        return out
    before = {e.name: e for e in old}
    after = {e.name: e for e in new}
    out += [f"電極 {n} を削除" for n in before if n not in after]
    out += [f"電極 {n} を追加" for n in after if n not in before]
    for name in after:
        if name in before and (what := parts(before[name], after[name])):
            out.append(f"電極 {name} の{'・'.join(what)}")
    return out


class Session:
    """開いているモデルと解析条件、直近の結果。"""

    def __init__(self) -> None:
        self.study: Study | None = None
        self.mesh: Mesh | None = None
        self.solution: Solution | None = None
        #: `solution` を解いたときの解析条件の写し。実行サマリ（SPEC §5.5）に残すのは
        #: これで、いま編集中の `study` ではない（解いた後に編集されうるため）
        self.solved_study: Study | None = None
        #: 表示量の配列（解ごとに 1 度だけ受け取る。`result_array`）
        self._arrays: dict[str, np.ndarray] = {}
        self._grid: pv.UnstructuredGrid | None = None
        #: 表示用の外表面（SPEC §6.3）。メッシュを開き直すまで使い回す
        self._surface: pv.PolyData | None = None
        #: 領域ごとの境界面（ハイライト用）。形だけなので、解き直しても使い回す
        self._region_boundaries: dict[int, pv.PolyData] = {}
        # Undo / Redo（SPEC §8 M7）。
        # 解析条件は Study 1 つに閉じていて、メッシュも結果も含まないので、
        # 操作ごとに丸ごと複製して積むだけで済む。差分を持つ設計は
        # 「どの編集を記録し忘れたか」を追う手間に見合わない
        self._undo: list[tuple[str, Study]] = []
        self._redo: list[tuple[str, Study]] = []
        #: 最後に開いた・保存したときの解析条件の写し。閉じるときに、保存していない
        #: 変更があるかをこれと比べて決める（Undo で戻せば「変更なし」に戻る）
        self._saved_study: Study | None = None
        #: メッシュを開くたびに増える番号。3D ビューはこれが変わったときだけ
        #: 視点をモデル全体に合わせ直す（それ以外の描き直しでは利用者のズームを保つ）
        self.mesh_serial = 0

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

    def open_mesh(self, path: str | Path, length_unit: str, mesh: Mesh | None = None) -> Study:
        """メッシュを読み、既定のケースを組み立てる（SPEC §6.2 の手順 1）。

        `mesh` を渡すと読み込みを省く。GUI は単位を選ばせる前に寸法を見せるため
        m として 1 度読んでいるので、それを `Mesh.with_length_unit` で付け替えて渡す
        （ファイルを 2 回パースしない）。
        """
        self.clear_history()
        path = Path(path)
        if mesh is None:
            mesh = read_nastran(path, length_unit=length_unit)
        elif mesh.length_unit != length_unit:
            raise GfemError(
                f"[内部エラー] 渡されたメッシュの単位 {mesh.length_unit} が {length_unit} と違います"
            )
        self.mesh = mesh
        self.mesh_serial += 1
        self._forget_results()
        self.study = Study(
            mesh_file=path,
            length_unit=length_unit,
            materials=[Material(pid=pid, sigma=0.0) for pid in self.mesh.solid_pids()],
        )
        # σ 未設定を 0 で表しておくと validate() が拾ってくれる
        self._mark_saved()
        return self.study

    def open_case(self, path: str | Path) -> Study:
        """ケースファイル（YAML）を開き、参照しているメッシュも読む。"""
        self.clear_history()
        study = Study.load(path)
        self.study = study
        self.mesh = study.read_mesh()
        self.mesh_serial += 1
        self._forget_results()
        # メッシュにあるのに YAML に無い PID を補う
        known = {m.pid for m in study.materials}
        for pid in self.mesh.solid_pids():
            if pid not in known:
                study.materials.append(Material(pid=pid, sigma=0.0))
        study.materials.sort(key=lambda m: m.pid)
        self._mark_saved()
        return study

    def save_case(self, path: str | Path) -> None:
        self._require_study().save(path)
        self._mark_saved()

    def _mark_saved(self) -> None:
        self._saved_study = copy.deepcopy(self.study)

    def is_modified(self) -> bool:
        """最後に開いた・保存したときから解析条件が変わっているか（閉じるときの確認に使う）。

        編集の記録ではなく中身を比べるので、Undo で元に戻せば「変更なし」になる。
        """
        return self.study is not None and self.study != self._saved_study

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

        **σ が未設定の領域があっても出す。** 面積は電極面の形だけで決まり
        （`assemble_load` は σ を見ない）、シェル要素の無いモデルでは σ を入れる前に
        電極を置いていくことが多い。以前は σ の検査に引っかかって、面積の代わりに
        「導電率が未設定」と出ていた。
        """
        study = self._require_study()
        kept = (study.electrodes, study.materials)
        try:
            if name is not None:
                study.electrodes = [e for e in kept[0] if e.name == name]
            # σ は荷重ベクトルに効かない。未設定（0）のまま渡すと Case が拒むので外す
            study.materials = []
            return study.build_case(self._require_mesh()).electrode_info()
        finally:
            study.electrodes, study.materials = kept

    def prepare_solve(self) -> tuple[Case, Study]:
        """solve に渡すケースと、**その時点の解析条件の写し**を返す。

        GUI は solve を別スレッドで回すので、終わるまでの間にも条件を編集できる。
        実行サマリ（SPEC §5.5）に残すのは解いた条件そのものでなければならないので、
        ケースを組み立てた瞬間に写しを取っておき、結果と一緒に `set_solution` へ渡す。
        """
        case = self.build_case()
        return case, copy.deepcopy(self._require_study())

    def solve(self, progress: Callable[[int, float], bool] | None = None) -> Solution:
        case, study = self.prepare_solve()
        self.set_solution(case.solve(progress=progress), study)
        return self.solution

    def set_solution(self, solution: Solution, study: Study) -> None:
        """外（ワーカースレッド）で解いた結果を、解いたときの条件の写しと一緒に取り込む。"""
        self.solution = solution
        self.solved_study = study
        self._arrays = {}
        self._grid = None  # 表示用データに結果を貼り直す
        self._surface = None

    def changes_since_solve(self) -> list[str]:
        """結果を出したときから変わった解析条件（人が読める短い言葉で）。空なら結果は最新。

        解いたときの条件の写し（`solved_study`）といまの条件を比べる。比べるのは
        **答えが変わるものだけ**:

        - 導電率、電極（役割・総電流・面・荷重の配り方・名前）
        - 収束判定と最大反復回数（答えの精度が変わる）
        - 電位の基準（φ の値がずれる。E と J は変わらない）

        バックエンドと前処理は比べない。バックエンドを変えても答えはビット単位で同じで、
        前処理を変えても収束判定の範囲でしか変わらない（SPEC §4.4 / §4.5）。領域の名前と
        書き出し先も答えを変えない。比べるので、Undo で解いたときの条件に戻せば
        「最新」に戻る。
        """
        old, new = self.solved_study, self.study
        if self.solution is None or old is None or new is None:
            return []
        changes: list[str] = []

        old_sigma = {m.pid: m.sigma for m in old.materials}
        new_sigma = {m.pid: m.sigma for m in new.materials}
        pids = sorted(p for p in old_sigma.keys() | new_sigma.keys()
                      if old_sigma.get(p) != new_sigma.get(p))
        if pids:
            changes.append(f"σ（PID {', '.join(str(p) for p in pids)}）")

        changes.extend(_electrode_changes(old.electrodes, new.electrodes))

        if old.solver.tolerance != new.solver.tolerance:
            changes.append("収束判定")
        if old.solver.max_iterations != new.solver.max_iterations:
            changes.append("最大反復回数")
        if old.output.reference_potential != new.output.reference_potential:
            changes.append("電位の基準")
        return changes

    def _forget_results(self) -> None:
        """結果と、結果から作った表示用データを捨てる（モデルを開き直したとき）。"""
        self.solution = None
        self.solved_study = None
        self._arrays = {}
        self._grid = None
        self._surface = None
        self._region_boundaries = {}

    # --- 表示用データ -------------------------------------------------------

    def result_array(self, key: str) -> np.ndarray | None:
        """表示量（`FIELD_ARRAYS` の鍵）の配列。解いていないか、表示量でなければ `None`。

        `Solution` の配列は Rust のメモリを複製せずに読み取り専用で見せたもの（SPEC §6.3）。
        **VTK に渡す前に、解ごとに 1 度だけ複製して持つ。** pyvista は読み取り専用の印を
        見ずにメモリを共有するので、そのまま渡すと表示側から結果そのものを書き換えられる。
        """
        if self.solution is None or key not in FIELD_ARRAYS:
            return None
        if key not in self._arrays:
            values = FIELD_ARRAYS[key](self.solution)
            if not values.flags.writeable:
                values = values.copy()
            self._arrays[key] = values
        return self._arrays[key]

    def model_box(self) -> tuple[np.ndarray, np.ndarray]:
        """モデルのバウンディングボックス `(最小, 最大)` [m]。"""
        lo, hi = self._require_mesh().bounding_box()
        return np.asarray(lo), np.asarray(hi)

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
            for key in FIELD_ARRAYS:
                spec = field_by_key(key)
                target = grid.cell_data if spec.is_element_value else grid.point_data
                target[key] = self.result_array(key)
            # 読み取り専用のビューは VTK に渡さない（`result_array` の説明を参照）
            grid.cell_data["E [V/m]"] = result.electric_field.copy()
            grid.cell_data["J [A/m2]"] = result.current_density.copy()
        self._grid = grid
        return grid

    def region_grid(self, pid: int) -> pv.UnstructuredGrid:
        """指定した領域だけを取り出した体積メッシュ（`grid()` を組み立てる）。

        ハイライト表示には使わない（[`region_boundary`] を使う）。PyVista の全機能で
        領域を調べたいとき用の、小規模モデル向けの経路（SPEC §6.3 のエスケープハッチ）。
        """
        grid = self.grid()
        return grid.extract_cells(np.flatnonzero(grid.cell_data["PID"] == pid))

    def region_boundary(self, pid: int) -> pv.PolyData:
        """指定した領域の境界面（体表面に出ていない領域のハイライト用。SPEC §6.3）。

        **体積メッシュは組み立てない。** 境界は Rust 側で領域の要素から直接取り出す。
        以前は体積メッシュ全体を作ってから（頭部クラスで 0.36 秒・468 MB）、領域を
        切り出して VTK に外側の面を取らせていた（選ぶたびに 0.2〜0.6 秒）。
        形だけなので、領域ごとに 1 度作ってメッシュを開き直すまで使い回す。
        """
        if pid in self._region_boundaries:
            return self._region_boundaries[pid]
        mesh = self._require_mesh()
        triangles, owners = mesh.region_surface(pid)
        poly = self._polydata(mesh.points(), triangles, {"PID": mesh.solid_pid_array()[owners]})
        if poly.n_cells:
            # 陰影のために法線を付ける（外表面と同じ理由。三角形は領域の外向きに並んでいる）
            poly = poly.compute_normals(
                cell_normals=True,
                point_normals=True,
                split_vertices=False,
                consistent_normals=False,
                auto_orient_normals=False,
            )
        self._region_boundaries[pid] = poly
        return poly

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

    def pick_patch(self, center, radius: float | None = None, pid: int | None = None, **shape) -> dict:
        """体表面の 1 点を中心に、形の中でつながっている面を選ぶ。

        形は円（`radius`）か、`shape="rect"` と `width` / `height` / `corner_radius` /
        `axis` / `angle` の矩形（`Mesh.pick_patch` と同じ。長さは m、角度は度）。
        """
        if radius is not None:
            shape["radius"] = float(radius)
        return self._require_mesh().pick_patch(
            [float(v) for v in center], pid=None if pid is None else int(pid), **shape
        )

    def patch_info(self, patch: SurfacePatch) -> dict:
        """置いてあるパッチが選ぶ面（面数・面積・指定面積・輪郭）。"""
        return self.pick_patch(patch.center, pid=patch.pid, **patch.shape_dict())

    @staticmethod
    def nominal_area(**shape) -> float:
        """形の指定どおりの面積 [m²]（メッシュに依らない）。式は Rust 側だけに置く。"""
        return float(check_patch({"center": [0.0, 0.0, 0.0], **shape})["nominal_area"])

    @staticmethod
    def screen_right(direction, view_up) -> list[float]:
        """カメラの向きと上方向から、画面の横方向（右向きの単位ベクトル）を出す。"""
        right = np.cross(np.asarray(direction, dtype=float), np.asarray(view_up, dtype=float))
        return [float(v) for v in right / np.linalg.norm(right)]

    @staticmethod
    def outline_polydata(info: dict) -> pv.PolyData | None:
        """`pick_patch` が返した輪郭を、描ける閉じた折れ線にする。

        輪郭は接平面の上にあるので、そのままだと凹んだ体表面に埋もれる。
        形の大きさの 2 % だけ法線の向きに浮かせる（見た目のためだけ）。
        """
        outline, normal = info.get("outline"), info.get("normal")
        if outline is None or normal is None or len(outline) < 2:
            return None
        points = np.asarray(outline, dtype=float)
        size = float(np.ptp(points, axis=0).max())
        points = points + np.asarray(normal, dtype=float) * (0.02 * size)
        n = len(points)
        # 点だけで PolyData を作ると点ごとに頂点セルが付き、角に粒が描かれる。線だけにする
        return pv.PolyData(points, lines=np.concatenate([[n + 1], np.arange(n), [0]]))

    @staticmethod
    def area_mismatch(info: dict) -> float | None:
        """`pick_patch` の結果の、指定面積に対する実際の面積のずれ（相対値）。"""
        nominal = info.get("nominal_area") or 0.0
        if nominal <= 0.0:
            return None  # 半径 0 の円（1 枚だけ選ぶ筆）など
        return info["area"] / nominal - 1.0

    def add_patch(
        self, name: str, center, radius: float | None = None, pid: int | None = None, **shape
    ) -> SurfacePatch:
        """電極にパッチを 1 枚足す。形の指定は `pick_patch` と同じ。"""
        spec = self.electrode(name)
        if spec is None:
            raise GfemError(f"[設定エラー] 電極 {name} がありません")
        if radius is not None:
            shape["radius"] = float(radius)
        data = {"center": [float(v) for v in center], **shape}
        if pid is not None:
            data["pid"] = int(pid)
        patch = SurfacePatch.from_dict(check_patch(data))
        spec.patches.append(patch)
        return patch

    def reshape_patch(self, name: str, index: int, **shape) -> SurfacePatch:
        """置いてあるパッチの形だけを変える（中心と PID はそのまま）。"""
        spec = self.electrode(name)
        if spec is None or not 0 <= index < len(spec.patches):
            raise GfemError(f"[設定エラー] 電極 {name} にパッチ {index + 1} がありません")
        old = spec.patches[index]
        data = {"center": list(old.center), **shape}
        if old.pid is not None:
            data["pid"] = old.pid
        patch = SurfacePatch.from_dict(check_patch(data))
        spec.patches[index] = patch
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
        if spec is None or field in poly.array_names:
            return poly
        # 体積メッシュ（`grid()`）は組み立てない。外表面に貼るのに要るのは
        # 配列だけで、大規模モデルでは体積メッシュの組み立てのほうが重い
        values = self.result_array(field)
        if values is None:
            return poly
        if spec.is_element_value:
            poly.cell_data[field] = values[poly.cell_data["__owner"]]
        else:
            poly.point_data[field] = values
        return poly

    def section_plane(self, axis: str, position: float) -> tuple[list[float], list[float]]:
        """断面・クリップの平面 `(点, 法線)`。

        `axis` は法線の向き（`"x"` / `"y"` / `"z"`）、`position` はその軸方向の
        モデルの範囲に対する割合（0〜1）。点はバウンディングボックスの中心を通る。
        """
        lo, hi = self.model_box()
        k = {"x": 0, "y": 1, "z": 2}[axis]
        origin = 0.5 * (lo + hi)
        origin[k] = lo[k] + (hi[k] - lo[k]) * position
        normal = [0.0, 0.0, 0.0]
        normal[k] = 1.0
        return origin.tolist(), normal

    def section(self, kind: str, axis: str, position: float, field: str | None) -> pv.PolyData:
        """断面（`kind="slice"`）かクリップ（`"clip"`）に表示量を貼ったもの（SPEC §6.3）。

        **切り口は Rust 側で作り、体積メッシュ（`grid()`）は組み立てない。** VTK に
        体積メッシュを渡して切ると、頭部クラスのモデルでそれだけで 468 MB、スライダーを
        1 目盛り動かすたびに 1〜2 秒かかった。クリップは軸の正の側を切り落とす。

        要素値（|J| など）は三角形の元要素の値、節点値は切り口の辺の上で補間した値を貼る。
        """
        mesh = self._require_mesh()
        origin, normal = self.section_plane(axis, position)
        data = (mesh.slice if kind == "slice" else mesh.clip)(origin, normal)
        elements = data["elements"]
        poly = self._polydata(data["points"], data["triangles"], {"PID": mesh.solid_pid_array()[elements]})
        if poly.n_cells == 0:
            return poly
        spec = field_by_key(field) if field else None
        values = self.result_array(field) if spec is not None else None
        if values is not None:
            if spec.is_element_value:
                poly.cell_data[field] = values[elements]
            else:
                a, b = data["edges"][:, 0], data["edges"][:, 1]
                poly.point_data[field] = values[a] + data["t"] * (values[b] - values[a])
        # 陰影のために法線を付ける（外表面と同じ理由。SPEC §6.3）。三角形の向きは
        # Rust 側でそろえてある（外表面は外向き、切り口は切り落とした側）
        return poly.compute_normals(
            cell_normals=True,
            point_normals=True,
            split_vertices=False,
            consistent_normals=False,
            auto_orient_normals=False,
        )

    def field_range(self, field: str | None) -> tuple[float, float] | None:
        """表示量の値域（**モデル全体**）。

        外表面だけ、あるいは断面だけで色を決めると、**同じ色が場所によって
        違う値を指す**ことになる。全体の値域に固定すれば、全体表示と断面と
        等値面で色の意味が揃う。
        """
        values = self.result_array(field)
        if values is None:
            return None
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

    def iso_surface(self, quantity: str, level: float, name: str | None = None) -> pv.PolyData:
        """等値面（Marching Tetrahedra、SPEC §6.3）。

        値は配列 `name`（省略すると `quantity`）に貼る。GUI は表示量の鍵
        （`"|J| [A/m2]"` など）で貼り、全体表示と同じ見出し・同じ色の範囲で塗る。
        """
        if self.solution is None:
            return pv.PolyData()
        points, triangles, scalars = self.solution.iso_surface(quantity, level)
        poly = self._polydata(points, triangles, {})
        if poly.n_cells:
            poly.point_data[name or quantity] = scalars
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
        lo, hi = self.model_box()
        model_centre = 0.5 * (lo + hi)
        diagonal = float(np.linalg.norm(hi - lo))
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
        lo, hi = self.model_box()
        return np.asarray([0.5 * (lo + hi)])

    def write_region_stats_csv(self, path) -> None:
        self._require_solution().write_region_stats_csv(path)

    def write_convergence_csv(self, path) -> None:
        self._require_solution().write_convergence_csv(path)

    def write_summary_json(self, path) -> None:
        """実行サマリ JSON（プロビナンス付き。SPEC §5.5）。

        解析条件は**解いたときの写し**（`solved_study`）を残す。ディスク上の
        ケースファイルを読むと、保存していない編集や解いた後の編集が反映されず、
        .bdf から開いた場合は何も残らなかった。
        """
        solution = self._require_solution()
        study = self.solved_study
        solution.write_summary_json(
            path,
            case_text=None if study is None else study.to_yaml(),
            case_file=None if study is None else study.path,
        )

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
            # 総注入電流 I で割る（M2 で max|f| から変更。SPEC §2.2 (a)）
            f"  電流保存 1ᵀf/I: {c['current_balance']:.3e}",
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
                "[設定エラー] まだ求解していません"
                "\n  対処: 「実行 → 求解」（F5）で計算してから書き出してください"
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
