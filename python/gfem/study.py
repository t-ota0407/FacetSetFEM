"""ケースファイル（YAML）の読み書き（SPEC §3.3）。

Nastran ファイルは形状と領域分けだけを担い、解析条件はこの YAML が持つ。
**Nastran ファイルは書き換えない。**

YAML の解釈を Rust ではなく Python 側に置いているのは、

- `gfem-core` の依存を増やさない（CLAUDE.md 設計原則 5）
- 行番号つきの親切なエラー（設計原則 6）を Rust と Python の二重に作らずに済む
- GUI もバッチスクリプトも Python から使う（SPEC §6.5）

ため。

    >>> study = gfem.Study.load("case.yaml")
    >>> result = study.solve()
    >>> print(result.impedance)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from ._gfem import Case, GfemError, Mesh, Solution, read_nastran

__all__ = [
    "ElectrodeSpec",
    "Material",
    "OutputSpec",
    "SolverSpec",
    "Study",
]

#: 現在のケースファイル形式のバージョン
CASE_VERSION = 1

#: v1.0 で対応している電極の電流分布（SPEC §2.4）
_SUPPORTED_DISTRIBUTIONS = {"uniform_current_density"}


def _error(what: str, where_: str, hint: str) -> GfemError:
    """「何が」「どこで」「どうすれば直るか」を揃えたエラーを作る（設計原則 6）。"""
    return GfemError(f"[設定エラー] {what}\n  場所: {where_}\n  対処: {hint}")


def _require(mapping: Any, key: str, where_: str, hint: str) -> Any:
    if not isinstance(mapping, dict) or key not in mapping:
        raise _error(f"必須の項目 {key} がありません", where_, hint)
    return mapping[key]


@dataclass
class Material:
    """領域（PSOLID の PID）に割り当てる導電率。"""

    pid: int
    sigma: float
    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"pid": self.pid}
        if self.name:
            d["name"] = self.name
        d["sigma"] = self.sigma
        return d


@dataclass
class ElectrodeSpec:
    """電極 1 個の設定。"""

    name: str
    role: str
    current: float
    shell_pids: list[int] = field(default_factory=list)
    element_ids: list[int] = field(default_factory=list)
    distribution: str = "uniform_current_density"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "current": self.current,
            "distribution": self.distribution,
            "faces": {
                "shell_pids": list(self.shell_pids),
                "element_ids": list(self.element_ids),
            },
        }


#: 指定できるバックエンド（SPEC §4.4）。
#: どれを選んでも結果はビット単位で同じで、速度だけが変わる。
KNOWN_BACKENDS = frozenset({"auto", "cpu", "cpu_serial", "cpu_parallel", "parallel", "serial"})


@dataclass
class SolverSpec:
    """ソルバ設定（SPEC §4.3）。"""

    #: "auto"（規模で自動選択）/ "cpu"（逐次）/ "cpu_parallel"（rayon）
    backend: str = "auto"
    method: str = "cg"
    preconditioner: str = "jacobi"
    gauge: str = "mean_zero"
    tolerance: float = 1.0e-10
    max_iterations: int = 50_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "method": self.method,
            "preconditioner": self.preconditioner,
            "gauge": self.gauge,
            "tolerance": self.tolerance,
            "max_iterations": self.max_iterations,
        }


@dataclass
class OutputSpec:
    """出力設定（SPEC §5.5）。"""

    #: 結果の VTU（ParaView で開ける）
    vtu: str | None = None
    #: 領域統計 CSV（SPEC §5.2）
    region_stats: str | None = None
    #: 収束履歴 CSV
    convergence: str | None = None
    #: 実行サマリ JSON（プロビナンス付き）
    summary_json: str | None = None
    #: 表示に使う電位の基準
    reference_potential: str = "zero_mean"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for key in ("vtu", "region_stats", "convergence", "summary_json"):
            value = getattr(self, key)
            if value:
                d[key] = value
        d["reference_potential"] = self.reference_potential
        return d


@dataclass
class Study:
    """ケースファイル 1 つ分。メッシュの指定と解析条件をまとめて持つ。"""

    mesh_file: Path
    length_unit: str
    materials: list[Material] = field(default_factory=list)
    electrodes: list[ElectrodeSpec] = field(default_factory=list)
    solver: SolverSpec = field(default_factory=SolverSpec)
    output: OutputSpec = field(default_factory=OutputSpec)
    #: 読み込み元のケースファイル（相対パスの基準になる）
    path: Path | None = None

    # --- 入出力 ---

    @classmethod
    def load(cls, path: str | Path) -> Study:
        """ケースファイルを読む。"""
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise _error(
                f"ケースファイルを開けません: {exc}",
                str(path),
                "パスが正しいか、読み取り権限があるかを確認してください",
            ) from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise _error(
                f"YAML として読めません: {exc}",
                str(path),
                "インデントと引用符を確認してください",
            ) from exc
        return cls.from_dict(data, path=path)

    @classmethod
    def from_dict(cls, data: Any, path: Path | None = None) -> Study:
        """辞書（YAML をパースした結果）から作る。"""
        where = str(path) if path else "<ケース定義>"
        if not isinstance(data, dict):
            raise _error(
                "ケースファイルの中身がマッピングではありません",
                where,
                "SPEC §3.3 の例を参考に、version / mesh / materials / electrodes を書いてください",
            )

        version = data.get("version", CASE_VERSION)
        if version != CASE_VERSION:
            raise _error(
                f"対応していないケースファイル形式です: version={version}",
                f"{where} の version",
                f"このバージョンの gfem が読めるのは version: {CASE_VERSION} です",
            )

        mesh = _require(
            data,
            "mesh",
            where,
            "mesh: file と length_unit を書いてください",
        )
        mesh_file = _require(
            mesh, "file", f"{where} の mesh", "解析するメッシュファイルのパスを書いてください"
        )
        length_unit = _require(
            mesh,
            "length_unit",
            f"{where} の mesh",
            "mm / cm / m / inch のいずれかを書いてください（Nastran には単位情報が無いため必須）",
        )

        materials = [
            _material_from(m, f"{where} の materials[{i}]")
            for i, m in enumerate(data.get("materials") or [])
        ]
        electrodes = [
            _electrode_from(e, f"{where} の electrodes[{i}]")
            for i, e in enumerate(data.get("electrodes") or [])
        ]
        solver = _solver_from(data.get("solver") or {}, f"{where} の solver")
        output = _output_from(data.get("output") or {}, f"{where} の output")

        return cls(
            mesh_file=Path(mesh_file),
            length_unit=str(length_unit),
            materials=materials,
            electrodes=electrodes,
            solver=solver,
            output=output,
            path=path,
        )

    def to_dict(self) -> dict[str, Any]:
        """YAML に書ける辞書へ変換する。"""
        return {
            "version": CASE_VERSION,
            "mesh": {"file": str(self.mesh_file), "length_unit": self.length_unit},
            "materials": [m.to_dict() for m in self.materials],
            "electrodes": [e.to_dict() for e in self.electrodes],
            "solver": self.solver.to_dict(),
            "output": self.output.to_dict(),
        }

    def save(self, path: str | Path) -> None:
        """ケースファイルを書き出す。"""
        path = Path(path)
        text = yaml.safe_dump(
            self.to_dict(), allow_unicode=True, sort_keys=False, default_flow_style=False
        )
        path.write_text(text, encoding="utf-8")
        self.path = path

    # --- 実行 ---

    def resolved_mesh_file(self) -> Path:
        """ケースファイルからの相対パスを解決したメッシュファイル。"""
        if self.mesh_file.is_absolute() or self.path is None:
            return self.mesh_file
        return (self.path.parent / self.mesh_file).resolve()

    def read_mesh(self) -> Mesh:
        """メッシュを読み込む。"""
        return read_nastran(self.resolved_mesh_file(), length_unit=self.length_unit)

    def build_case(self, mesh: Mesh | None = None) -> Case:
        """`gfem.Case` を組み立てる。"""
        mesh = mesh if mesh is not None else self.read_mesh()
        case = Case(mesh)
        case.set_conductivity({m.pid: m.sigma for m in self.materials})
        for m in self.materials:
            if m.name:
                case.set_region_name(m.pid, m.name)
        for e in self.electrodes:
            case.add_electrode(
                e.name,
                role=e.role,
                current=e.current,
                shell_pids=e.shell_pids,
                element_ids=e.element_ids,
            )
        case.set_solver(
            tolerance=self.solver.tolerance,
            max_iterations=self.solver.max_iterations,
            preconditioner=self.solver.preconditioner,
            backend=self.solver.backend,
        )
        case.set_reference_potential(self.output.reference_potential)
        return case

    def solve(
        self,
        mesh: Mesh | None = None,
        progress: Callable[[int, float], bool] | None = None,
    ) -> Solution:
        """メッシュを読み、ケースを組み立てて解き、`output` の指定を書き出す。"""
        case = self.build_case(mesh)
        result = case.solve(progress=progress)
        self.write_outputs(result)
        return result

    def write_outputs(self, result: Solution) -> list[Path]:
        """`output` 節で指定されたファイルを書き出し、そのパスを返す。"""
        base = self.path.parent if self.path else Path.cwd()
        written: list[Path] = []
        if self.output.vtu:
            path = base / self.output.vtu
            result.write_vtu(path)
            written.append(path)
        if self.output.region_stats:
            path = base / self.output.region_stats
            result.write_region_stats_csv(path)
            written.append(path)
        if self.output.convergence:
            path = base / self.output.convergence
            result.write_convergence_csv(path)
            written.append(path)
        if self.output.summary_json:
            path = base / self.output.summary_json
            # ケースファイルの全文をプロビナンスに残す（SPEC §5.5）
            text = None
            if self.path and self.path.exists():
                text = self.path.read_text(encoding="utf-8")
            result.write_summary_json(path, case_text=text)
            written.append(path)
        return written

    def region_names(self) -> dict[int, str]:
        """PID → 領域名。GUI のツリー表示に使う（コアは名前を持たない）。"""
        return {m.pid: m.name for m in self.materials if m.name}


# --- 各節のパース -----------------------------------------------------------


def _material_from(data: Any, where_: str) -> Material:
    pid = _require(data, "pid", where_, "PSOLID の PID を書いてください")
    sigma = _require(
        data, "sigma", where_, "導電率を S/m で書いてください（例: 筋 0.35）"
    )
    try:
        return Material(pid=int(pid), sigma=float(sigma), name=str(data.get("name", "")))
    except (TypeError, ValueError) as exc:
        raise _error(
            f"pid / sigma を数値として読めません: {exc}",
            where_,
            "pid は整数、sigma は実数（S/m）で書いてください",
        ) from exc


def _electrode_from(data: Any, where_: str) -> ElectrodeSpec:
    name = _require(data, "name", where_, "電極の名前を書いてください")
    role = _require(data, "role", where_, "anode か cathode を書いてください")
    current = _require(data, "current", where_, "総電流を A で書いてください（例: 2.0e-3）")
    distribution = str(data.get("distribution", "uniform_current_density"))
    if distribution not in _SUPPORTED_DISTRIBUTIONS:
        raise _error(
            f"未対応の電流分布です: {distribution}",
            f"{where_} の distribution",
            "v1.0 は uniform_current_density（面積按分）のみ対応です（SPEC §2.4）",
        )
    faces = data.get("faces") or {}
    if not isinstance(faces, dict):
        raise _error(
            "faces がマッピングではありません",
            f"{where_} の faces",
            "shell_pids と element_ids のリストを書いてください",
        )
    shell_pids = [int(p) for p in (faces.get("shell_pids") or [])]
    element_ids = [int(i) for i in (faces.get("element_ids") or [])]
    if not shell_pids and not element_ids:
        raise _error(
            f"電極 {name} に面が指定されていません",
            f"{where_} の faces",
            "shell_pids か element_ids のどちらかは必要です",
        )
    try:
        current = float(current)
    except (TypeError, ValueError) as exc:
        raise _error(
            f"current を数値として読めません: {current!r}",
            where_,
            "総電流を A で書いてください（例: 2.0e-3）",
        ) from exc
    return ElectrodeSpec(
        name=str(name),
        role=str(role),
        current=current,
        shell_pids=shell_pids,
        element_ids=element_ids,
        distribution=distribution,
    )


def _solver_from(data: Any, where_: str) -> SolverSpec:
    spec = SolverSpec()
    if not isinstance(data, dict):
        raise _error("solver がマッピングではありません", where_, "SPEC §3.3 の例を参照してください")
    spec.backend = str(data.get("backend", spec.backend))
    spec.method = str(data.get("method", spec.method))
    spec.preconditioner = str(data.get("preconditioner", spec.preconditioner))
    spec.gauge = str(data.get("gauge", spec.gauge))
    spec.tolerance = float(data.get("tolerance", spec.tolerance))
    spec.max_iterations = int(data.get("max_iterations", spec.max_iterations))

    if spec.method != "cg":
        raise _error(
            f"未対応の解法です: {spec.method}",
            f"{where_} の method",
            "v1.0 は cg（共役勾配法）のみです（SPEC §4.3）",
        )
    if spec.gauge != "mean_zero":
        raise _error(
            f"未対応のゲージ固定です: {spec.gauge}",
            f"{where_} の gauge",
            "射影による mean_zero のみです。1 節点ピン留めは採用しません（SPEC §2.2）",
        )
    if spec.backend not in KNOWN_BACKENDS:
        raise _error(
            f"このビルドでは使えないバックエンドです: {spec.backend}",
            f"{where_} の backend",
            "auto / cpu / cpu_parallel のいずれかです。cuda は M5 で対応します（SPEC §8）",
        )
    return spec


def _output_from(data: Any, where_: str) -> OutputSpec:
    if not isinstance(data, dict):
        raise _error("output がマッピングではありません", where_, "SPEC §3.3 の例を参照してください")
    return OutputSpec(
        vtu=data.get("vtu"),
        region_stats=data.get("region_stats"),
        convergence=data.get("convergence"),
        summary_json=data.get("summary_json"),
        reference_potential=str(data.get("reference_potential", "zero_mean")),
    )
