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
| `solver.tolerance` / `solver.backend` / `solver.preconditioner` | ソルバ設定 |

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
from .study import Study

__all__ = ["SweepTable", "sweep"]


def _error(what: str, where_: str, hint: str) -> GfemError:
    return GfemError(f"[設定エラー] {what}\n  場所: {where_}\n  対処: {hint}")


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
            "振るパラメータが 1 つも指定されていません",
            "sweep()",
            'たとえば {"sigma.3": [0.30, 0.35]} のように渡してください',
        )
    names = list(parameters)
    values = [list(parameters[n]) for n in names]
    for name, vs in zip(names, values):
        if not vs:
            raise _error(
                f"パラメータ {name} の値が空です",
                "sweep()",
                "少なくとも 1 つの値を並べてください",
            )
    if product:
        combinations = list(itertools.product(*values))
    else:
        lengths = {len(v) for v in values}
        if len(lengths) != 1:
            raise _error(
                "product=False では全てのパラメータの値の個数が同じでなければなりません: "
                + ", ".join(f"{n} は {len(v)} 個" for n, v in zip(names, values)),
                "sweep()",
                "個数を揃えるか、product=True（既定）で直積にしてください",
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
                    f"ケース {index + 1}/{total} が解けませんでした: {exc}",
                    ", ".join(f"{k}={v}" for k, v in assignment.items()),
                    "keep_going=True にすると、残りのケースを続けて回せます",
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
    return {
        "impedance_ohm": solution.impedance,
        "voltage_V": solution.voltage,
        "power_W": conservation["power_from_field"],
        "j_max_A_per_m2": float(max(solution.j_magnitude)),
        "e_max_V_per_m": float(max(solution.e_magnitude)),
        "phi_min_V": float(min(solution.potential)),
        "phi_max_V": float(max(solution.potential)),
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
    where = f"パラメータ {name}"

    if parts[0] == "sigma":
        if len(parts) != 2:
            raise _error(
                f"導電率の指定は sigma.<PID> か sigma.<領域名> です: {name}",
                where,
                "たとえば sigma.3 や sigma.muscle",
            )
        _set_sigma(study, parts[1], float(value), where)
        return

    if parts[0] == "current" and len(parts) == 1:
        # 刺激強度。1 極だけ変えると電流保存が崩れるので、まとめて変える
        _set_electrode(study, "*", "current", value, where)
        return

    if parts[0] == "electrode":
        if len(parts) != 3:
            raise _error(
                f"電極の指定は electrode.<名前>.<項目> です: {name}",
                where,
                "たとえば electrode.E1.current",
            )
        _set_electrode(study, parts[1], parts[2], value, where)
        return

    if parts[0] == "solver":
        if len(parts) != 2 or not hasattr(study.solver, parts[1]):
            known = ", ".join(sorted(vars(study.solver)))
            raise _error(
                f"ソルバの項目が見つかりません: {name}",
                where,
                f"使えるのは {known} です",
            )
        current = getattr(study.solver, parts[1])
        setattr(study.solver, parts[1], type(current)(value))
        return

    raise _error(
        f"振り方が分からないパラメータです: {name}",
        where,
        "current / sigma.<PID> / electrode.<名前>.<項目> / solver.<項目> "
        "のいずれかで指定してください",
    )


def _set_sigma(study: Study, key: str, value: float, where: str) -> None:
    matched = [
        m
        for m in study.materials
        if (key.isdigit() and m.pid == int(key)) or (m.name and m.name == key)
    ]
    if not matched:
        known = ", ".join(
            f"{m.pid}" + (f"（{m.name}）" if m.name else "") for m in study.materials
        )
        raise _error(
            f"領域が見つかりません: {key}",
            where,
            f"このケースにある領域は {known} です",
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
            f"電極が見つかりません: {name}",
            where,
            f"このケースにある電極は {known} です",
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
        else:
            raise _error(
                f"電極の項目が見つかりません: {field_}",
                where,
                "current / shell_pids / element_ids / role のいずれかです",
            )


def _as_list(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple)):
        return value
    return [value]
