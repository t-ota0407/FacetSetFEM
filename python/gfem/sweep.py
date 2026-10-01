"""パラメータスタディ（SPEC §8 M7）。

「電極電流を 1〜3 mA で振ったら R はどう変わるか」「骨の σ の不確かさが
ホットスポットにどれだけ効くか」——研究で実際に知りたいのはこういうことで、
1 ケースずつ手で書き換えて回すのは間違いのもとになる。

    >>> import gfem
    >>> study = gfem.Study.load("case.yaml")
    >>> table = gfem.sweep(study, {"electrode.E1.current": [1e-3, 2e-3, 3e-3]})
    >>> table.write_csv("sweep.csv")
    >>> table.rows[0]["impedance_ohm"]

# 振れるもの

| 書き方 | 意味 |
|---|---|
| `sigma.3` | PID 3 の導電率 [S/m] |
| `sigma.muscle` | 名前が `muscle` の領域の導電率 |
| `current` | 全電極の電流 [A]。刺激強度を振るのはたいていこれ |
| `electrode.E1.current` | 電極 `E1` だけの電流 [A] |
| `electrode.*.current` | `current` と同じ（明示的な書き方） |
| `contact_impedance` | 全電極の接触インピーダンス [Ω·m²]（全電極が完全電極モデルになる。SPEC §2.5） |
| `electrode.E1.contact_impedance` | 電極 `E1` だけの接触インピーダンス [Ω·m²]（`E1` が完全電極モデルになる） |
| `electrode.E1.distribution` | 電極 `E1` の電流の配り方（`uniform_current_density` / `complete_electrode`） |
| `solver.tolerance` / `solver.backend` / `solver.preconditioner` | ソルバ設定 |

**接触インピーダンスを入れた電極は完全電極モデルになる。** 一様電流密度のケースに
`contact_impedance` を振ると、「一様電流密度と比べてどれだけ変わるか」をそのまま見られる。
値の決め方は SPEC §2.5（実測の電極間抵抗に合わせるなら :func:`gfem.fit_contact_impedance`）。

**電極の位置**は面（`shell_pids`）の指定そのものなので、
`electrode.E1.shell_pids` に PID のリストを並べて振る。

電極 1 極だけ電流を変えると **Σ I_k = 0 が崩れて解が存在しなくなる**
（SPEC §2.2）。刺激強度を振るときは `current`（全極まとめて）を使うこと。

# 組み合わせ方

既定は**直積**（全組み合わせ）。1 対 1 で対応させたい場合は `product=False` にする。

    >>> gfem.sweep(study, {"sigma.2": [0.02, 0.04], "sigma.3": [0.30, 0.35]})
    # → 4 ケース
    >>> gfem.sweep(study, {"sigma.2": [0.02, 0.04], "sigma.3": [0.30, 0.35]}, product=False)
    # → 2 ケース

# 再現性

ケースの並びは**指定した順の直積**で決まるので、同じ入力からは常に同じ順序・
同じ結果になる（CLAUDE.md 設計原則 1）。1 ケースずつ順に解くだけで、
ケース間で状態を持ち回さない。
"""

from __future__ import annotations

import copy
import csv
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ._gfem import GfemError, Mesh
from .i18n import error_text, tr
from .study import COMPLETE, Study

__all__ = ["SweepTable", "sweep"]


def _error(what: str, where_: str, hint: str) -> GfemError:
    return GfemError(error_text(what, where_, hint))


@dataclass
class SweepTable:
    """パラメータスタディの結果表。"""

    #: 振ったパラメータの名前（指定した順）
    parameters: list[str] = field(default_factory=list)
    #: 1 ケース 1 行。振った値と結果の指標が入る
    rows: list[dict[str, Any]] = field(default_factory=list)
    #: 解けなかったケース `(パラメータの組, 理由)`
    failures: list[tuple[dict[str, Any], str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rows)

    def columns(self) -> list[str]:
        """CSV の列名。振ったパラメータが先、結果が後。"""
        if not self.rows:
            return list(self.parameters)
        rest = [k for k in self.rows[0] if k not in self.parameters]
        return list(self.parameters) + rest

    def write_csv(self, path: str | Path) -> None:
        """CSV に書き出す。列名は ASCII で単位付き（SPEC §5.5）。"""
        path = Path(path)
        columns = self.columns()
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({k: row.get(k, "") for k in columns})

    def to_dicts(self) -> list[dict[str, Any]]:
        """pandas などに渡しやすい形。"""
        return [dict(r) for r in self.rows]


def sweep(
    study: Study,
    parameters: Mapping[str, Sequence[Any]],
    *,
    product: bool = True,
    mesh: Mesh | None = None,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
    keep_going: bool = False,
) -> SweepTable:
    """`study` を土台に、`parameters` を振って一括で解く。

    元の `study` は書き換えない（ケースごとに複製する）。
    メッシュは 1 度だけ読んで全ケースで使い回す。

    `progress(index, total, values)` を渡すと 1 ケース終わるごとに呼ばれる。
    `keep_going=True` にすると、解けないケースがあっても止まらず
    [`SweepTable.failures`] に理由を残して続ける。
    """
    if not parameters:
        raise _error(
            tr("No parameters to vary were given"),
            "sweep()",
            tr('Pass something like {example}', example='{"sigma.3": [0.30, 0.35]}'),
        )
    names = list(parameters)
    values = [list(parameters[n]) for n in names]
    for name, vs in zip(names, values):
        if not vs:
            raise _error(
                tr("Parameter {name} has no values", name=name),
                "sweep()",
                tr("List at least one value"),
            )
    if product:
        combinations = list(itertools.product(*values))
    else:
        lengths = {len(v) for v in values}
        if len(lengths) != 1:
            raise _error(
                tr(
                    "With product=False every parameter must have the same number of values: {counts}",
                    counts=", ".join(
                        tr("{name} has {n}", name=n, n=len(v)) for n, v in zip(names, values)
                    ),
                ),
                "sweep()",
                tr("Make the counts equal, or use product=True (default) for the Cartesian product"),
            )
        combinations = list(zip(*values))

    # メッシュは全ケースで共通。1 度だけ読む
    shared_mesh = mesh if mesh is not None else study.read_mesh()

    table = SweepTable(parameters=names)
    total = len(combinations)
    for index, combination in enumerate(combinations):
        assignment = dict(zip(names, combination))
        candidate = copy.deepcopy(study)
        for name, value in assignment.items():
            _apply(candidate, name, value)
        try:
            case = candidate.build_case(shared_mesh)
            solution = case.solve()
        except GfemError as exc:
            if not keep_going:
                raise _error(
                    tr("Case {i}/{total} could not be solved: {error}", i=index + 1, total=total, error=exc),
                    ", ".join(f"{k}={v}" for k, v in assignment.items()),
                    tr("With keep_going=True the remaining cases keep running"),
                ) from exc
            table.failures.append((assignment, str(exc)))
            if progress is not None:
                progress(index + 1, total, assignment)
            continue

        row: dict[str, Any] = dict(assignment)
        row.update(_summarise(solution))
        table.rows.append(row)
        if progress is not None:
            progress(index + 1, total, assignment)
    return table


def _summarise(solution: Any) -> dict[str, Any]:
    """1 ケースの結果から、表に並べる指標を取り出す。

    列名は ASCII で単位付き（SPEC §5.5 の CSV の取り決めに合わせる）。
    """
    conservation = solution.conservation
    # 配列は取り出すたびに複製されるので 1 度だけ受け取り、numpy で集計する
    # （Python の組み込み max は 1 要素ずつ回るので、大規模モデルで遅い）
    potential = solution.potential
    electrodes = solution.electrodes
    return {
        "impedance_ohm": solution.impedance,
        "voltage_V": solution.voltage,
        "power_W": conservation["power_from_field"],
        # 要素内の最大（2 次要素では頂点の値。1 次要素では要素値と同じ。SPEC §5.1）
        "j_max_A_per_m2": float(solution.j_peak.max()),
        "e_max_V_per_m": float(solution.e_peak.max()),
        "phi_min_V": float(potential.min()),
        "phi_max_V": float(potential.max()),
        # 皮膚を通って流れ込む電流密度の最大（縁の集中の指標。SPEC §2.5）
        "contact_j_max_A_per_m2": max(e["peak_current_density"] for e in electrodes),
        "iterations": solution.iterations,
        "residual": solution.residual,
        "backend": solution.backend,
        "preconditioner": solution.preconditioner,
        # 保存則チェックに引っかかったケースは、表の上で見分けられるようにしておく
        "energy_relative_error": conservation["energy_relative_error"],
        "conservation_warnings": len(conservation["warnings"]),
    }


def _apply(study: Study, name: str, value: Any) -> None:
    """`name` が指すところに `value` を入れる。"""
    parts = name.split(".")
    where = tr("parameter {name}", name=name)

    if parts[0] == "sigma":
        if len(parts) != 2:
            raise _error(
                tr("A conductivity is given as sigma.<PID> or sigma.<region name>: {name}", name=name),
                where,
                tr("For example sigma.3 or sigma.muscle"),
            )
        _set_sigma(study, parts[1], float(value), where)
        return

    if parts[0] == "current" and len(parts) == 1:
        # 刺激強度。1 極だけ変えると電流保存が崩れるので、まとめて変える
        _set_electrode(study, "*", "current", value, where)
        return

    if parts[0] == "contact_impedance" and len(parts) == 1:
        _set_electrode(study, "*", "contact_impedance", value, where)
        return

    if parts[0] == "electrode":
        if len(parts) != 3:
            raise _error(
                tr("An electrode is given as electrode.<name>.<item>: {name}", name=name),
                where,
                tr("For example electrode.E1.current"),
            )
        _set_electrode(study, parts[1], parts[2], value, where)
        return

    if parts[0] == "solver":
        if len(parts) != 2 or not hasattr(study.solver, parts[1]):
            known = ", ".join(sorted(vars(study.solver)))
            raise _error(
                tr("No such solver item: {name}", name=name),
                where,
                tr("Available: {known}", known=known),
            )
        current = getattr(study.solver, parts[1])
        setattr(study.solver, parts[1], type(current)(value))
        return

    raise _error(
        tr("Don't know how to vary the parameter: {name}", name=name),
        where,
        tr(
            "Use one of current / contact_impedance / sigma.<PID> / electrode.<name>.<item> "
            "/ solver.<item>"
        ),
    )


def _set_sigma(study: Study, key: str, value: float, where: str) -> None:
    matched = [
        m
        for m in study.materials
        if (key.isdigit() and m.pid == int(key)) or (m.name and m.name == key)
    ]
    if not matched:
        known = ", ".join(
            f"{m.pid}" + (tr(" ({name})", name=m.name) if m.name else "") for m in study.materials
        )
        raise _error(
            tr("No such region: {key}", key=key),
            where,
            tr("The regions in this case are {known}", known=known),
        )
    for m in matched:
        m.sigma = value


def _set_electrode(study: Study, name: str, field_: str, value: Any, where: str) -> None:
    matched = (
        list(study.electrodes)
        if name == "*"
        else [e for e in study.electrodes if e.name == name]
    )
    if not matched:
        known = ", ".join(e.name for e in study.electrodes)
        raise _error(
            tr("No such electrode: {name}", name=name),
            where,
            tr("The electrodes in this case are {known}", known=known),
        )
    for e in matched:
        if field_ == "current":
            e.current = float(value)
        elif field_ == "shell_pids":
            e.shell_pids = [int(v) for v in _as_list(value)]
        elif field_ == "element_ids":
            e.element_ids = [int(v) for v in _as_list(value)]
        elif field_ == "role":
            e.role = str(value)
        elif field_ == "contact_impedance":
            # 接触インピーダンスを入れた電極は完全電極モデルになる（SPEC §2.5）
            e.contact_impedance = float(value)
            e.distribution = COMPLETE
        elif field_ == "distribution":
            e.distribution = str(value)
            if e.is_complete and e.contact_impedance is None:
                raise _error(
                    tr("Electrode {name} has no contact impedance", name=e.name),
                    where,
                    tr(
                        "Vary electrode.{name}.contact_impedance together with it, "
                        "or set the contact impedance in the case first",
                        name=e.name,
                    ),
                )
        else:
            raise _error(
                tr("No such electrode item: {item}", item=field_),
                where,
                tr(
                    "Use one of current / shell_pids / element_ids / role / distribution "
                    "/ contact_impedance"
                ),
            )


def _as_list(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple)):
        return value
    return [value]
