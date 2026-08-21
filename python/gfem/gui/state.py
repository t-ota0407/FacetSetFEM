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
from ..study import ElectrodeSpec, Material
from ..tissues import TISSUES

__all__ = ["DisplayField", "DISPLAY_FIELDS", "Session"]


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
DISPLAY_FIELDS: list[DisplayField] = [
    DisplayField("PID", "領域 (PID)", "cell", needs_solution=False),
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
    "体積 [mm³]",
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
        spec = ElectrodeSpec(
            name=name, role=role, current=current, shell_pids=list(shell_pids or [])
        )
        self._require_study().electrodes.append(spec)
        return spec

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

    def electrode_info(self) -> list[dict]:
        """電極の合計面積と平均電流密度（SPEC §6.2 の手順 3）。"""
        return self.build_case().electrode_info()

    def solve(self, progress: Callable[[int, float], bool] | None = None) -> Solution:
        self.set_solution(self.build_case().solve(progress=progress))
        return self.solution

    def set_solution(self, solution: Solution) -> None:
        """外（ワーカースレッド）で解いた結果を取り込む。"""
        self.solution = solution
        self._grid = None  # 表示用データに結果を貼り直す

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
        """指定した電極の面（3D ビューでの色分け用。SPEC §6.2 の手順 3）。"""
        mesh = self._require_mesh()
        spec = self.electrode(name)
        if spec is None:
            return pv.PolyData()
        tris, shell_pids, ids = mesh.shell_triangles()
        keep = np.isin(shell_pids, spec.shell_pids) | np.isin(ids, spec.element_ids)
        return self._polydata(mesh.points(), tris[keep], {})

    def surface_polydata(self) -> pv.PolyData:
        """外表面の三角形（表示用の軽量データ。SPEC §6.3）。"""
        mesh = self._require_mesh()
        tris = mesh.surface_triangles()
        owners = mesh.surface_triangle_elements()
        pid = mesh.solid_pid_array()[owners]
        return self._polydata(mesh.points(), tris, {"PID": pid})

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
