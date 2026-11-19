"""FacetSetFEM — 経皮電気刺激シミュレーションの Python API。

GUI（`facetsetfem.gui`）もバッチスクリプトも、このモジュールを通して同じ Rust コアを叩く
（SPEC §6.5）。GUI 側にロジックは置かない。

    >>> import facetsetfem
    >>> mesh = facetsetfem.read_nastran("tests/models/slab_3layer.bdf", length_unit="mm")
    >>> case = facetsetfem.Case(mesh)
    >>> case.set_conductivity({1: 0.10, 2: 0.04, 3: 0.35})
    >>> case.add_electrode("E1", role="anode",   current=2e-3, shell_pids=[101])
    >>> case.add_electrode("E2", role="cathode", current=2e-3, shell_pids=[201])
    >>> result = case.solve()
    >>> round(result.impedance, 1)
    548.6
    >>> result.write_vtu("out.vtu")

ケースファイル（YAML）から一括で回すこともできる:

    >>> study = facetsetfem.Study.load("case.yaml")
    >>> result = study.solve()
"""

from __future__ import annotations

from ._core import (
    Case,
    FacetSetFemError,
    Mesh,
    ReadReport,
    Solution,
    SolverCancelled,
    __version__,
    backend_choices,
    cuda_available,
    cuda_check,
    cuda_devices,
    distribution_choices,
    length_units,
    metres_per_unit,
    parse_wedge_order,
    preconditioner_choices,
    read_nastran,
    wedge_order_choices,
)
from . import i18n
from .contact import ContactFit, fit_contact_impedance
from .study import ElectrodeSpec, Material, OutputSpec, SolverSpec, Study, SurfacePatch
from .i18n import tr
from .sweep import SweepTable, sweep
from .tissues import TISSUES, Tissue, load_library, tissue, tissue_names

__all__ = [
    "Case",
    "ContactFit",
    "ElectrodeSpec",
    "SurfacePatch",
    "FacetSetFemError",
    "Material",
    "Mesh",
    "OutputSpec",
    "ReadReport",
    "Solution",
    "SolverCancelled",
    "SolverSpec",
    "Study",
    "SweepTable",
    "TISSUES",
    "Tissue",
    "__version__",
    "backend_choices",
    "cuda_available",
    "cuda_check",
    "cuda_devices",
    "describe_bounding_box",
    "describe_extent",
    "describe_gpus",
    "distribution_choices",
    "fit_contact_impedance",
    "i18n",
    "length_units",
    "load_library",
    "metres_per_unit",
    "parse_wedge_order",
    "preconditioner_choices",
    "read_nastran",
    "sweep",
    "tissue",
    "tissue_names",
    "wedge_order_choices",
]

def describe_bounding_box(mesh: Mesh, unit: str | None = None) -> str:
    """バウンディングボックスを「選んだ単位」と「m」の両方で表した文字列を返す。

    単位の取り違えは結果を桁違いに狂わせるうえ、一見もっともらしい絵が出てしまう。
    SPEC §3.2 はこの確認表示を必須要件としており、GUI の単位選択ダイアログと
    読み込みレポートの両方でこの文字列を使う。
    """
    box = mesh.bounding_box()
    if box is None:
        return tr("(no nodes)")
    lo, hi = box
    return describe_extent([h - l for l, h in zip(lo, hi)], unit or mesh.length_unit)


def describe_extent(size_m, unit: str) -> str:
    """寸法 [m] を「選んだ単位」と「m」の両方で表した文字列を返す。

    GUI の単位選択ダイアログは、単位を決める前の寸法をこれで見せる。
    換算係数は Rust 側の表（:func:`metres_per_unit`）から引く。
    """
    scale = metres_per_unit(unit)
    in_unit = " x ".join(_fmt(s / scale) + f" {unit}" for s in size_m)
    in_metre = " x ".join(_fmt(s) + " m" for s in size_m)
    return f"{in_unit} = {in_metre}"


def _fmt(v: float) -> str:
    """桁数を抑えつつ、末尾の 0 を落とした表示。"""
    return f"{v:.6g}"


def describe_gpus() -> str:
    """使える GPU を人が読める形にまとめる（SPEC §8 M5）。

    GUI の「環境」表示と、バッチ実行のログ冒頭に使う。
    GPU があってもこの表示だけでは解けるとは限らない（VRAM 次第）ので、
    規模ごとの判定は :func:`cuda_check` を使う。
    """
    if not cuda_available():
        return tr("This build has no CUDA backend (CPU only)")
    devices = cuda_devices()
    if not devices:
        return tr("No CUDA-capable GPU found (solving on the CPU)")
    lines = []
    for d in devices:
        major, minor = d["compute_capability"]
        state = "" if d["usable"] else tr(" (not usable for double precision: CC below 6.0)")
        lines.append(
            tr(
                "GPU {ordinal}: {name} / CC {major}.{minor} / VRAM {free:.1f} GB free "
                "(of {total:.1f} GB){state}",
                ordinal=d["ordinal"], name=d["name"], major=major, minor=minor,
                free=d["free_memory"] / 1e9, total=d["total_memory"] / 1e9, state=state,
            )
        )
    return "\n".join(lines)
