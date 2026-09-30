"""接触インピーダンスを実測の電極間抵抗に合わせる（SPEC §2.5）。

接触インピーダンス z はゲルの種類・皮膚の湿り具合・貼ってからの時間で何倍も変わるので、
文献値を 1 つ選ぶより、**刺激装置で測った電極間抵抗に合うように決める**ほうが確実。

    >>> import gfem
    >>> study = gfem.Study.load("case.yaml")
    >>> fit = gfem.fit_contact_impedance(study, measured_ohm=1200.0)
    >>> fit.contact_impedance          # Ω·m²
    >>> fit.study.save("case_fitted.yaml")

選んだ電極（既定は全電極）を完全電極モデルにして同じ z を入れ、R(z) が実測値に
一致する z を割線法で探す。R(z) は z について単調増加で、z が大きいと
`R ≈ R_体 + z Σ_l (I_l/I)² / A_l` とほぼ直線になるので、数回の求解で収まる
（2 電極なら `Σ = 1/A_a + 1/A_c`）。1 回ごとにケースを丸ごと解くので、
大きなモデルでは時間がかかる（メッシュは 1 度だけ読んで使い回す）。
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ._gfem import GfemError, Mesh
from .i18n import error_text, tr
from .study import COMPLETE, Study

__all__ = ["ContactFit", "fit_contact_impedance"]


def _error(what: str, where_: str, hint: str) -> GfemError:
    return GfemError(error_text(what, where_, hint))


@dataclass
class ContactFit:
    """:func:`fit_contact_impedance` の結果。"""

    #: 合わせた接触インピーダンス [Ω·m²]
    contact_impedance: float
    #: そのときの電極間抵抗 [Ω]
    impedance: float
    #: 合わせた相手（実測の電極間抵抗）[Ω]
    measured: float
    #: 試した `(z [Ω·m²], R [Ω])` を試した順に
    evaluations: list[tuple[float, float]] = field(default_factory=list)
    #: 合わせた z を入れた Study の写し（元の Study は書き換えない）
    study: Study | None = None


def fit_contact_impedance(
    study: Study,
    measured_ohm: float,
    *,
    electrodes: Sequence[str] | None = None,
    mesh: Mesh | None = None,
    rtol: float = 1.0e-4,
    max_evaluations: int = 20,
    progress: Callable[[int, float, float], None] | None = None,
) -> ContactFit:
    """電極間抵抗が `measured_ohm` [Ω] になる接触インピーダンスを探す（SPEC §2.5）。

    `electrodes` に名前を並べると、その電極だけを完全電極モデルにして z を合わせる
    （既定は全電極。どれにも同じ z が入る）。`rtol` は R の相対誤差の許容値。
    `progress(回数, z, R)` を渡すと 1 回解くごとに呼ばれる。
    """
    where = "fit_contact_impedance()"
    if not (isinstance(measured_ohm, (int, float)) and math.isfinite(measured_ohm) and measured_ohm > 0):
        raise _error(
            tr("The measured resistance must be a positive finite value: {value}", value=measured_ohm),
            where,
            tr("Give the inter-electrode resistance measured with the stimulator in Ω"),
        )
    if not 0.0 < rtol < 1.0:
        raise _error(
            tr("rtol is out of range: {value}", value=rtol),
            where,
            tr("Give a relative tolerance with 0 < rtol < 1 (default 1e-4)"),
        )
    names = [e.name for e in study.electrodes]
    targets = list(electrodes) if electrodes is not None else names
    unknown = [n for n in targets if n not in names]
    if unknown or not targets:
        raise _error(
            tr("No such electrode: {name}", name=", ".join(unknown) or "-"),
            where,
            tr("The electrodes in this case are {known}", known=", ".join(names)),
        )

    shared_mesh = mesh if mesh is not None else study.read_mesh()

    def with_z(z: float) -> Study:
        candidate = copy.deepcopy(study)
        for e in candidate.electrodes:
            if e.name in targets:
                e.distribution = COMPLETE
                e.contact_impedance = z
        return candidate

    # R ≈ R_体 + z S の傾き S（面積と電流は電極の面の指定だけで決まる）
    info = with_z(1.0).build_case(shared_mesh).electrode_info()
    total = sum(e["current"] for e in info if e["current"] > 0)
    slope = sum((e["current"] / total) ** 2 / e["area"] for e in info if e["name"] in targets)

    evaluations: list[tuple[float, float]] = []

    def resistance(z: float) -> float:
        solution = with_z(z).build_case(shared_mesh).solve()
        r = solution.impedance
        evaluations.append((z, r))
        if progress is not None:
            progress(len(evaluations), z, r)
        return r

    def done(z: float, r: float) -> bool:
        return abs(r - measured_ohm) <= rtol * measured_ohm

    # 1 回目: 体の抵抗を無視した z（R はこれより必ず大きい）
    z_prev = measured_ohm / slope
    r_prev = resistance(z_prev)
    if done(z_prev, r_prev):
        return ContactFit(z_prev, r_prev, measured_ohm, evaluations, with_z(z_prev))
    # 2 回目: 傾き S で 1 歩戻る。体の抵抗だけで実測を超えていれば、z をごく小さくして確かめる
    z = z_prev - (r_prev - measured_ohm) / slope
    if z <= 0.0:
        z = z_prev * 1.0e-6
    r = resistance(z)
    if r > measured_ohm and z <= z_prev * 1.0e-6:
        raise _error(
            tr(
                "Even with an almost zero contact impedance the model's resistance ({r:.4g} Ω) "
                "exceeds the measured {m:.4g} Ω",
                r=r,
                m=measured_ohm,
            ),
            where,
            tr(
                "The body alone is more resistive than the measurement. Check the conductivities, "
                "the electrode areas and the length unit, and whether the measured value is the "
                "resistance between these electrodes"
            ),
        )
    # 割線法。実測値を挟む区間を持ち、外へ出ようとしたら対数で 2 分する
    lo = hi = None
    for zz, rr in ((z_prev, r_prev), (z, r)):
        if rr < measured_ohm:
            lo = zz if lo is None else max(lo, zz)
        else:
            hi = zz if hi is None else min(hi, zz)
    while not done(z, r):
        if len(evaluations) >= max_evaluations:
            raise _error(
                tr(
                    "The contact impedance did not converge in {n} solves (last z = {z:.4g} Ω·m², "
                    "R = {r:.4g} Ω, measured {m:.4g} Ω)",
                    n=len(evaluations),
                    z=z,
                    r=r,
                    m=measured_ohm,
                ),
                where,
                tr("Increase max_evaluations or loosen rtol"),
            )
        if r != r_prev:
            proposal = z - (r - measured_ohm) * (z - z_prev) / (r - r_prev)
        else:
            proposal = z - (r - measured_ohm) / slope
        inside = proposal > 0.0 and (lo is None or proposal > lo) and (hi is None or proposal < hi)
        if not inside:
            if lo is not None and hi is not None:
                proposal = math.sqrt(lo * hi)
            elif hi is not None:
                proposal = hi / 10.0
            else:
                proposal = lo * 10.0
        z_prev, r_prev = z, r
        z = proposal
        r = resistance(z)
        if r < measured_ohm:
            lo = z if lo is None else max(lo, z)
        else:
            hi = z if hi is None else min(hi, z)
    return ContactFit(z, r, measured_ohm, evaluations, with_z(z))
