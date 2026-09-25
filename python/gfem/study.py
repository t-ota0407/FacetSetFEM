"""ケースファイル（YAML）の読み書き（SPEC §3.3）。

Nastran ファイルは形状と領域分けだけを担い、解析条件はこの YAML が持つ。
**Nastran ファイルは書き換えない。**

YAML の解釈を Rust ではなく Python 側に置いているのは、

- `gfem-core` の依存を増やさない（CLAUDE.md 設計原則 5）
- 場所つきの親切なエラー（設計原則 6）を Rust と Python の二重に作らずに済む。
  場所はファイル名と項目の位置（`case.yaml の electrodes[1]` など）で示す。
  YAML の構文エラーだけは PyYAML が行番号を出す
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
    "SurfacePatch",
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
class SurfacePatch:
    """体表面に置く電極パッチ 1 枚（SPEC §11.1）。

    **シェル要素を持たないモデルのための指定方法。** 人体モデルはソリッド要素だけで
    配られるのが普通で、その場合シェル PID では電極を 1 つも置けない。
    GUI では 3D ビューのクリックが `center` を、スピンボックスが `radius` を決める。

    保存するのは選ばれた面の一覧ではなく**中心と半径**。同じメッシュなら同じ面が
    選ばれるうえ、ケースファイルを読んだ人が意味を追える（面番号の羅列は追えない）。
    """

    #: 中心 [m]（クリックした点）
    center: tuple[float, float, float]
    #: 半径 [m]
    radius: float
    #: この PID のソリッドが持つ面だけに限る（None なら制限しない）
    pid: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "center": [float(v) for v in self.center],
            "radius": float(self.radius),
        }
        if self.pid is not None:
            d["pid"] = int(self.pid)
        return d


@dataclass
class ElectrodeSpec:
    """電極 1 個の設定。"""

    name: str
    role: str
    current: float
    shell_pids: list[int] = field(default_factory=list)
    element_ids: list[int] = field(default_factory=list)
    patches: list[SurfacePatch] = field(default_factory=list)
    #: 手で足した体表面の面 `(EID, 面番号)`。面番号は **1 始まり**
    added_faces: list[tuple[int, int]] = field(default_factory=list)
    #: 手で外した体表面の面。パッチや added_faces より優先される
    removed_faces: list[tuple[int, int]] = field(default_factory=list)
    distribution: str = "uniform_current_density"

    def to_dict(self) -> dict[str, Any]:
        faces: dict[str, Any] = {
            "shell_pids": list(self.shell_pids),
            "element_ids": list(self.element_ids),
        }
        if self.patches:
            faces["patches"] = [p.to_dict() for p in self.patches]
        if self.added_faces:
            faces["added"] = [[int(e), int(f)] for e, f in self.added_faces]
        if self.removed_faces:
            faces["removed"] = [[int(e), int(f)] for e, f in self.removed_faces]
        return {
            "name": self.name,
            "role": self.role,
            "current": self.current,
            "distribution": self.distribution,
            "faces": faces,
        }


#: 指定できるバックエンド（SPEC §4.4）。
#: どれを選んでも結果はビット単位で同じで、速度だけが変わる。
KNOWN_BACKENDS = frozenset(
    {"auto", "cpu", "cpu_serial", "serial", "cpu_parallel", "parallel", "cuda", "gpu"}
)

#: 指定できる前処理（SPEC §4.3 / §4.5）。
#: バックエンドと違い、こちらは反復の道筋が変わるので収束判定の範囲で答えが揺れる。
KNOWN_PRECONDITIONERS = frozenset(
    {"jacobi", "diagonal", "amg", "smoothed_aggregation", "sa", "none"}
)


@dataclass
class SolverSpec:
    """ソルバ設定（SPEC §4.3）。"""

    #: "auto"（規模で自動選択）/ "cpu"（逐次）/ "cpu_parallel"（rayon）/ "cuda"（GPU）
    backend: str = "auto"
    method: str = "cg"
    #: "jacobi"（既定）/ "amg"（代数マルチグリッド）/ "none"
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
        _require_unique_names(electrodes, where)
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

    def to_yaml(self) -> str:
        """ケースファイルと同じ形の YAML 文字列。

        実行サマリ（SPEC §5.5）には、ディスク上のファイルではなくこれを残す。
        読み込んだ後に条件を変えて解くと、ファイルの中身は解いた条件と食い違うため。
        `Study.from_dict(yaml.safe_load(...))` で読み直せば同じ条件に戻る。
        """
        return yaml.safe_dump(
            self.to_dict(), allow_unicode=True, sort_keys=False, default_flow_style=False
        )

    def save(self, path: str | Path) -> None:
        """ケースファイルを書き出す。"""
        path = Path(path)
        path.write_text(self.to_yaml(), encoding="utf-8")
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
                patches=[p.to_dict() for p in e.patches],
                added=[tuple(f) for f in e.added_faces],
                removed=[tuple(f) for f in e.removed_faces],
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
        """`output` 節で指定されたファイルを書き出し、そのパスを返す。

        実行サマリには**この Study の今の中身**を解析条件として残すので、
        `result` を解いた後に条件を書き換えてから呼ばないこと（`solve` は解いた直後に呼ぶ）。
        """
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
            # 解析条件の全文をプロビナンスに残す（SPEC §5.5）。ディスク上のファイルを
            # 読むと、読み込み後に条件を変えて解いた場合に食い違う
            result.write_summary_json(path, case_text=self.to_yaml(), case_file=self.path)
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
    patches = [
        _patch_from(p, f"{where_} の faces.patches[{i}]")
        for i, p in enumerate(faces.get("patches") or [])
    ]
    added = _face_refs_from(faces.get("added"), f"{where_} の faces.added")
    removed = _face_refs_from(faces.get("removed"), f"{where_} の faces.removed")
    if not shell_pids and not element_ids and not patches and not added:
        raise _error(
            f"電極 {name} に面が指定されていません",
            f"{where_} の faces",
            "shell_pids / element_ids / patches のいずれかは必要です"
            "（シェル要素の無いモデルは patches を使ってください。SPEC §11.1）",
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
        patches=patches,
        added_faces=added,
        removed_faces=removed,
        distribution=distribution,
    )


def _require_unique_names(electrodes: list[ElectrodeSpec], where_: str) -> None:
    """電極名の重複を止める。

    電極名は結果（平均電位・実行サマリ）とケースファイルで電極を指す鍵。
    重複していると、同名の陽極と陰極が同じ面を指しても重なりとして捕まらない
    （コアの `Case::validate` も同じ検査をする。ここでは場所を添えて早めに止める）。
    """
    first: dict[str, int] = {}
    for i, e in enumerate(electrodes):
        if e.name in first:
            raise _error(
                f"電極の名前 {e.name} が重複しています",
                f"{where_} の electrodes[{i}]（electrodes[{first[e.name]}] と同じ名前）",
                "電極名は結果とケースファイルで電極を指す鍵なので、電極ごとに違う名前にしてください",
            )
        first[e.name] = i


def _face_refs_from(data: Any, where_: str) -> list[tuple[int, int]]:
    """`[[EID, 面番号], ...]` を読む。面番号は 1 始まり。"""
    if data is None:
        return []
    out: list[tuple[int, int]] = []
    for i, item in enumerate(data):
        try:
            eid, face = item
            eid, face = int(eid), int(face)
        except (TypeError, ValueError) as exc:
            raise _error(
                f"面の指定を読めません: {item!r}",
                f"{where_}[{i}]",
                "[要素 EID, 面番号] の 2 個組で書いてください（面番号は 1 始まり）",
            ) from exc
        if face < 1:
            raise _error(
                f"面番号が {face} です",
                f"{where_}[{i}]",
                "面番号は 1 始まりで書いてください（EID={} の第 1 面なら 1）".format(eid),
            )
        out.append((eid, face))
    return out


def _patch_from(data: Any, where_: str) -> SurfacePatch:
    """体表面パッチ 1 枚を読む。"""
    if not isinstance(data, dict):
        raise _error(
            "パッチがマッピングではありません",
            where_,
            "center: [x, y, z] と radius: r（m）を書いてください",
        )
    center = _require(data, "center", where_, "中心を [x, y, z]（m）で書いてください")
    radius = _require(data, "radius", where_, "半径を m で書いてください（例: 0.025）")
    try:
        center = tuple(float(v) for v in center)
        radius = float(radius)
    except (TypeError, ValueError) as exc:
        raise _error(
            f"center / radius を数値として読めません: {exc}",
            where_,
            "center は 3 個の実数、radius は実数（どちらも m）で書いてください",
        ) from exc
    if len(center) != 3:
        raise _error(
            f"center の要素数が {len(center)} 個です",
            where_,
            "center は [x, y, z] の 3 個で書いてください（単位は m）",
        )
    if not radius >= 0.0:
        raise _error(
            f"半径が負です: {radius}",
            where_,
            "半径は 0 以上で書いてください（単位は m）",
        )
    pid = data.get("pid")
    return SurfacePatch(center=center, radius=radius, pid=None if pid is None else int(pid))


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
            "auto / cpu / cpu_parallel / cuda のいずれかです。"
            "cuda が使えるかは gfem.describe_gpus() で確認できます",
        )
    if spec.preconditioner not in KNOWN_PRECONDITIONERS:
        raise _error(
            f"未知の前処理です: {spec.preconditioner}",
            f"{where_} の preconditioner",
            "jacobi / amg / none のいずれかです。amg は反復回数が"
            "メッシュの細かさにほとんど依存しなくなります（SPEC §4.5）",
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
