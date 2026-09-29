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

from ._gfem import (
    Case,
    GfemError,
    Mesh,
    Solution,
    backend_choices,
    check_patch,
    parse_backend,
    parse_preconditioner,
    preconditioner_choices,
    read_nastran,
)
from .i18n import error_text, tr

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
    return GfemError(error_text(what, where_, hint))


def _in(where_: str, part: str) -> str:
    """「<場所> の <項目>」（英語は "<項目> in <場所>"）。"""
    return tr("{part} in {where}", part=part, where=where_)


def _require(mapping: Any, key: str, where_: str, hint: str) -> Any:
    if not isinstance(mapping, dict) or key not in mapping:
        raise _error(tr("The required item {key} is missing", key=key), where_, hint)
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
    GUI では 3D ビューのクリックが `center` を、パネルの入力が形を決める。

    形は円（`radius`）か、角を丸められる矩形（`shape="rect"` と `width` / `height` /
    `corner_radius` / `axis` / `angle`）。矩形の `axis` は幅の辺の向きで、体表面の
    接平面に投影して使う。`angle` はそこから法線まわりに回す角度（度。外から見て
    反時計回りが正）。GUI はクリックしたときの画面の横方向を `axis` に入れる。

    保存するのは選ばれた面の一覧ではなく**中心と形**。同じメッシュなら同じ面が
    選ばれるうえ、ケースファイルを読んだ人が意味を追える（面番号の羅列は追えない）。
    形の解釈と値の検査は Rust 側（`check_patch`）だけが持つ。
    """

    #: 中心 [m]（クリックした点）
    center: tuple[float, float, float]
    #: 円の半径 [m]（矩形なら None）
    radius: float | None = None
    #: この PID のソリッドが持つ面だけに限る（None なら制限しない）
    pid: int | None = None
    #: "circle"（円）か "rect"（矩形）
    shape: str = "circle"
    #: 矩形の幅 [m]（`axis` の向きの辺）
    width: float | None = None
    #: 矩形の高さ [m]
    height: float | None = None
    #: 矩形の角の半径 [m]。0 なら尖った角
    corner_radius: float = 0.0
    #: 矩形の幅の辺の向き（世界座標。長さは問わない）
    axis: tuple[float, float, float] | None = None
    #: `axis` から法線まわりに回す角度 [度]
    angle: float = 0.0

    @property
    def is_rect(self) -> bool:
        return self.shape == "rect"

    def shape_dict(self) -> dict[str, Any]:
        """形だけの指定（中心と PID を除く）。`Mesh.pick_patch` のキーワード引数になる。"""
        if not self.is_rect:
            return {"radius": float(self.radius)}
        d: dict[str, Any] = {
            "shape": "rect",
            "width": float(self.width),
            "height": float(self.height),
        }
        if self.corner_radius:
            d["corner_radius"] = float(self.corner_radius)
        d["axis"] = [float(v) for v in self.axis]
        if self.angle:
            d["angle"] = float(self.angle)
        return d

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"center": [float(v) for v in self.center]}
        d.update(self.shape_dict())
        if self.pid is not None:
            d["pid"] = int(self.pid)
        return d

    def describe(self) -> str:
        """一覧に出す短い説明（例: 「矩形 20×20 mm 角R2 30°」「円 直径 50 mm」）。"""
        if not self.is_rect:
            return tr("Circle, diameter {d:.3g} mm", d=2e3 * self.radius)
        text = tr("Rectangle {w:.3g}×{h:.3g} mm", w=self.width * 1e3, h=self.height * 1e3)
        if self.corner_radius:
            text += tr(" corner R{r:.3g}", r=self.corner_radius * 1e3)
        if self.angle:
            text += f" {self.angle:.3g}°"
        return text

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurfacePatch:
        """`check_patch` が正規化した辞書から作る。"""
        axis = data.get("axis")
        return cls(
            center=tuple(float(v) for v in data["center"]),
            radius=None if data.get("radius") is None else float(data["radius"]),
            pid=None if data.get("pid") is None else int(data["pid"]),
            shape=str(data.get("shape", "circle")),
            width=None if data.get("width") is None else float(data["width"]),
            height=None if data.get("height") is None else float(data["height"]),
            corner_radius=float(data.get("corner_radius") or 0.0),
            axis=None if axis is None else tuple(float(v) for v in axis),
            angle=float(data.get("angle") or 0.0),
        )


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


@dataclass
class SolverSpec:
    """ソルバ設定（SPEC §4.3）。"""

    #: "auto"（規模で自動選択）/ "cpu"（逐次）/ "cpu_parallel"（rayon）/ "cuda"（GPU）。
    #: どれを選んでも結果はビット単位で同じで、速度だけが変わる（SPEC §4.4）
    backend: str = "auto"
    method: str = "cg"
    #: "jacobi"（既定）/ "amg"（代数マルチグリッド）/ "none"。バックエンドと違い、
    #: 反復の道筋が変わるので収束判定の範囲で答えが揺れる（SPEC §4.3 / §4.5）
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
                tr("Cannot open the case file: {error}", error=exc),
                str(path),
                tr("Check that the path is right and that the file is readable"),
            ) from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise _error(
                tr("Cannot read as YAML: {error}", error=exc),
                str(path),
                tr("Check the indentation and quotes"),
            ) from exc
        return cls.from_dict(data, path=path)

    @classmethod
    def from_dict(cls, data: Any, path: Path | None = None) -> Study:
        """辞書（YAML をパースした結果）から作る。"""
        where = str(path) if path else tr("<case definition>")
        if not isinstance(data, dict):
            raise _error(
                tr("The case file does not contain a mapping"),
                where,
                tr("Write version / mesh / materials / electrodes following the example in SPEC §3.3"),
            )

        version = data.get("version", CASE_VERSION)
        if version != CASE_VERSION:
            raise _error(
                tr("Unsupported case file format: version={version}", version=version),
                _in(where, "version"),
                tr("This version of gfem reads version: {v}", v=CASE_VERSION),
            )

        mesh = _require(
            data,
            "mesh",
            where,
            tr("Write mesh: with file and length_unit"),
        )
        mesh_file = _require(
            mesh, "file", _in(where, "mesh"), tr("Write the path of the mesh file to analyse")
        )
        length_unit = _require(
            mesh,
            "length_unit",
            _in(where, "mesh"),
            tr("Write one of mm / cm / m / inch (required because Nastran carries no unit information)"),
        )

        materials = [
            _material_from(m, _in(where, f"materials[{i}]"))
            for i, m in enumerate(data.get("materials") or [])
        ]
        electrodes = [
            _electrode_from(e, _in(where, f"electrodes[{i}]"))
            for i, e in enumerate(data.get("electrodes") or [])
        ]
        _require_unique_names(electrodes, where)
        solver = _solver_from(data.get("solver") or {}, _in(where, "solver"))
        output = _output_from(data.get("output") or {}, _in(where, "output"))

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
    pid = _require(data, "pid", where_, tr("Write the PSOLID PID"))
    sigma = _require(
        data, "sigma", where_, tr("Write the conductivity in S/m (e.g. muscle 0.35)")
    )
    try:
        return Material(pid=int(pid), sigma=float(sigma), name=str(data.get("name", "")))
    except (TypeError, ValueError) as exc:
        raise _error(
            tr("Cannot read pid / sigma as numbers: {error}", error=exc),
            where_,
            tr("Write pid as an integer and sigma as a real number (S/m)"),
        ) from exc


def _electrode_from(data: Any, where_: str) -> ElectrodeSpec:
    name = _require(data, "name", where_, tr("Write the electrode name"))
    role = _require(data, "role", where_, tr("Write anode or cathode"))
    current = _require(data, "current", where_, tr("Write the total current in A (e.g. 2.0e-3)"))
    distribution = str(data.get("distribution", "uniform_current_density"))
    if distribution not in _SUPPORTED_DISTRIBUTIONS:
        raise _error(
            tr("Unsupported current distribution: {d}", d=distribution),
            _in(where_, "distribution"),
            tr("v1.0 supports only uniform_current_density (area-weighted) (SPEC §2.4)"),
        )
    faces = data.get("faces") or {}
    if not isinstance(faces, dict):
        raise _error(
            tr("faces is not a mapping"),
            _in(where_, "faces"),
            tr("Write lists of shell_pids and element_ids"),
        )
    shell_pids = [int(p) for p in (faces.get("shell_pids") or [])]
    element_ids = [int(i) for i in (faces.get("element_ids") or [])]
    patches = [
        _patch_from(p, _in(where_, f"faces.patches[{i}]"))
        for i, p in enumerate(faces.get("patches") or [])
    ]
    added = _face_refs_from(faces.get("added"), _in(where_, "faces.added"))
    removed = _face_refs_from(faces.get("removed"), _in(where_, "faces.removed"))
    if not shell_pids and not element_ids and not patches and not added:
        raise _error(
            tr("Electrode {name} has no faces", name=name),
            _in(where_, "faces"),
            tr(
                "One of shell_pids / element_ids / patches is needed "
                "(use patches for models without shell elements; SPEC §11.1)"
            ),
        )
    try:
        current = float(current)
    except (TypeError, ValueError) as exc:
        raise _error(
            tr("Cannot read current as a number: {value}", value=repr(current)),
            where_,
            tr("Write the total current in A (e.g. 2.0e-3)"),
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
                tr("The electrode name {name} is duplicated", name=e.name),
                tr(
                    "{where} (same name as electrodes[{first}])",
                    where=_in(where_, f"electrodes[{i}]"), first=first[e.name],
                ),
                tr(
                    "The electrode name identifies the electrode in results and case files, "
                    "so give each electrode a different name"
                ),
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
                tr("Cannot read the face: {value}", value=repr(item)),
                f"{where_}[{i}]",
                tr("Write pairs of [element EID, face number] (face numbers start at 1)"),
            ) from exc
        if face < 1:
            raise _error(
                tr("The face number is {face}", face=face),
                f"{where_}[{i}]",
                tr("Face numbers start at 1 (1 for the first face of EID={eid})", eid=eid),
            )
        out.append((eid, face))
    return out


def _patch_from(data: Any, where_: str) -> SurfacePatch:
    """体表面パッチ 1 枚を読む。

    形の解釈と値の検査は Rust 側（`check_patch`）に任せ、ここは場所を添えるだけ。
    円と矩形のキーや範囲を Python にも書くと、2 か所がずれる（SPEC §6.5）。
    """
    if not isinstance(data, dict):
        raise _error(
            tr("The patch is not a mapping"),
            where_,
            tr(
                "Write center: [x, y, z] and radius: r (m); for a rectangle, shape: rect with "
                "width / height / axis (SPEC §3.3)"
            ),
        )
    try:
        normalized = check_patch(data)
    except GfemError as exc:
        # コアのエラーは「[設定エラー] 何が\n  対処: どうすれば」。場所を間に差し込む
        # （見出しはいまの表示言語。SPEC §6.6）
        text = str(exc).removeprefix(f"[{tr('Setup error')}] ")
        what, _, hint = text.partition("\n  " + tr("Fix: {hint}", hint=""))
        raise _error(
            what.strip(), where_, hint.strip() or tr("See how to write patches in SPEC §3.3")
        ) from None
    return SurfacePatch.from_dict(normalized)


def _solver_from(data: Any, where_: str) -> SolverSpec:
    spec = SolverSpec()
    if not isinstance(data, dict):
        raise _error(tr("solver is not a mapping"), where_, tr("See the example in SPEC §3.3"))
    spec.backend = str(data.get("backend", spec.backend))
    spec.method = str(data.get("method", spec.method))
    spec.preconditioner = str(data.get("preconditioner", spec.preconditioner))
    spec.gauge = str(data.get("gauge", spec.gauge))
    spec.tolerance = float(data.get("tolerance", spec.tolerance))
    spec.max_iterations = int(data.get("max_iterations", spec.max_iterations))

    if spec.method != "cg":
        raise _error(
            tr("Unsupported method: {m}", m=spec.method),
            _in(where_, "method"),
            tr("v1.0 supports only cg (conjugate gradient) (SPEC §4.3)"),
        )
    if spec.gauge != "mean_zero":
        raise _error(
            tr("Unsupported gauge fixing: {g}", g=spec.gauge),
            _in(where_, "gauge"),
            tr("Only mean_zero by projection is supported. Pinning a single node is not used (SPEC §2.2)"),
        )
    if not _parses(parse_backend, spec.backend):
        raise _error(
            tr("This build cannot use the backend: {b}", b=spec.backend),
            _in(where_, "backend"),
            tr(
                "Use one of {choices}. Check whether cuda is available with gfem.describe_gpus()",
                choices=" / ".join(backend_choices()),
            ),
        )
    if not _parses(parse_preconditioner, spec.preconditioner):
        raise _error(
            tr("Unknown preconditioner: {p}", p=spec.preconditioner),
            _in(where_, "preconditioner"),
            tr(
                "Use one of {choices}. With amg the iteration count hardly depends on mesh fineness "
                "(SPEC §4.5)",
                choices=" / ".join(preconditioner_choices()),
            ),
        )
    return spec


def _parses(parse: Callable[[str], str], value: str) -> bool:
    """Rust 側の解釈で読めるか。別名（"gpu" など）の表は Rust 側の 1 か所だけに置く。"""
    try:
        parse(value)
    except GfemError:
        return False
    return True


def _output_from(data: Any, where_: str) -> OutputSpec:
    if not isinstance(data, dict):
        raise _error(tr("output is not a mapping"), where_, tr("See the example in SPEC §3.3"))
    return OutputSpec(
        vtu=data.get("vtu"),
        region_stats=data.get("region_stats"),
        convergence=data.get("convergence"),
        summary_json=data.get("summary_json"),
        reference_potential=str(data.get("reference_potential", "zero_mean")),
    )
