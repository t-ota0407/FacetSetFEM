"""GUI のセッション状態（SPEC §6）。

**Qt に依存しない**。GUI にロジックを置かないという方針（SPEC §7）に従い、
「何を読み込んでいて、どんな設定で、結果はどうだったか」をここに集約し、
Qt 側はこれを表示・編集するだけにする。こうするとヘッドレスでテストできる。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pyvista as pv

from .. import Case, GfemError, Mesh, Solution, Study, read_nastran
from ..study import ElectrodeSpec, Material

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

    @property
    def is_element_value(self) -> bool:
        return self.location == "cell"


#: 表示量の一覧。要素値と節点値の両方を出す（SPEC §5.1）
DISPLAY_FIELDS: list[DisplayField] = [
    DisplayField("PID", "領域 (PID)", "cell", needs_solution=False),
    DisplayField("phi [V]", "電位 φ [V]", "point"),
    DisplayField("|E| [V/m]", "電界 |E| [V/m]（要素値）", "cell"),
    DisplayField("|J| [A/m2]", "電流密度 |J| [A/m²]（要素値）", "cell"),
    DisplayField("power density [W/m3]", "電力密度 σ|E|² [W/m³]（要素値）", "cell"),
    DisplayField("|E| (nodal) [V/m]", "電界 |E| [V/m]（節点値・平滑化）", "point"),
    DisplayField("|J| (nodal) [A/m2]", "電流密度 |J| [A/m²]（節点値・平滑化）", "point"),
]

#: 節点値は平滑化されているという注意（SPEC §5.1 で GUI 上に明示することになっている）
NODAL_VALUE_WARNING = (
    "節点値は体積重み平均で平滑化されています。最大値の評価には要素値を使ってください。"
)

#: 組織物性の目安 [S/m]。GUI の入力補助に使う（出典を明記できるようになるまでは目安扱い）
TISSUE_HINTS: dict[str, float] = {
    "skin": 0.10,
    "fat": 0.04,
    "muscle": 0.35,
    "bone": 0.02,
    "blood": 0.70,
}


class Session:
    """開いているモデルと解析条件、直近の結果。"""

    def __init__(self) -> None:
        self.study: Study | None = None
        self.mesh: Mesh | None = None
        self.solution: Solution | None = None
        self._grid: pv.UnstructuredGrid | None = None

    # --- 読み込み -----------------------------------------------------------

    def open_mesh(self, path: str | Path, length_unit: str) -> Study:
        """メッシュを読み、既定のケースを組み立てる（SPEC §6.2 の手順 1）。"""
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
        tets = mesh.tetrahedra().astype(np.int64)
        cells = np.hstack(
            [np.full((len(tets), 1), 4, dtype=np.int64), tets]
        ).ravel()
        types = np.full(len(tets), pv.CellType.TETRA, dtype=np.uint8)
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
            "所要時間 [秒]: "
            + ", ".join(f"{k} {v:.3f}" for k, v in timings.items()),
        ]
        return "\n".join(lines)

    # --- 内部 ---------------------------------------------------------------

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
