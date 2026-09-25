"""GenericFEM — 経皮電気刺激シミュレーションの Python API。

GUI（`gfem.gui`）もバッチスクリプトも、このモジュールを通して同じ Rust コアを叩く
（SPEC §6.5）。GUI 側にロジックは置かない。

    >>> import gfem
    >>> mesh = gfem.read_nastran("tests/models/slab_3layer.bdf", length_unit="mm")
    >>> case = gfem.Case(mesh)
    >>> case.set_conductivity({1: 0.10, 2: 0.04, 3: 0.35})
    >>> case.add_electrode("E1", role="anode",   current=2e-3, shell_pids=[101])
    >>> case.add_electrode("E2", role="cathode", current=2e-3, shell_pids=[201])
    >>> result = case.solve()
    >>> round(result.impedance, 1)
    206.4
    >>> result.write_vtu("out.vtu")

ケースファイル（YAML）から一括で回すこともできる:

    >>> study = gfem.Study.load("case.yaml")
    >>> result = study.solve()
"""

from __future__ import annotations

from ._gfem import (
    Case,
    GfemError,
    Mesh,
    ReadReport,
    Solution,
    SolverCancelled,
    __version__,
    cuda_available,
    cuda_check,
    cuda_devices,
    length_units,
    read_nastran,
)
from .study import ElectrodeSpec, Material, OutputSpec, SolverSpec, Study, SurfacePatch
from .sweep import SweepTable, sweep
from .tissues import TISSUES, Tissue, load_library, tissue, tissue_names

__all__ = [
    "Case",
    "ElectrodeSpec",
    "SurfacePatch",
    "GfemError",
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
    "cuda_available",
    "cuda_check",
    "cuda_devices",
    "describe_bounding_box",
    "describe_gpus",
    "length_units",
    "load_library",
    "read_nastran",
    "sweep",
    "tissue",
    "tissue_names",
]

#: 1 単位が何メートルか。Rust 側（`gfem_core::units`）と同じ値を持つ
_METRES_PER_UNIT = {"mm": 1.0e-3, "cm": 1.0e-2, "m": 1.0, "inch": 0.0254}


def describe_bounding_box(mesh: Mesh, unit: str | None = None) -> str:
    """バウンディングボックスを「選んだ単位」と「m」の両方で表した文字列を返す。

    単位の取り違えは結果を桁違いに狂わせるうえ、一見もっともらしい絵が出てしまう。
    SPEC §3.2 はこの確認表示を必須要件としており、GUI の単位選択ダイアログと
    読み込みレポートの両方でこの文字列を使う。
    """
    box = mesh.bounding_box()
    if box is None:
        return "（節点がありません）"
    unit = unit or mesh.length_unit
    scale = _METRES_PER_UNIT[unit]
    lo, hi = box
    size_m = [h - l for l, h in zip(lo, hi)]
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
        return "このビルドには CUDA バックエンドが入っていません（CPU のみ）"
    devices = cuda_devices()
    if not devices:
        return "CUDA が使える GPU が見つかりません（CPU で解きます）"
    lines = []
    for d in devices:
        major, minor = d["compute_capability"]
        state = "" if d["usable"] else "（倍精度に使えません: CC 6.0 未満）"
        lines.append(
            f"GPU {d['ordinal']}: {d['name']} / CC {major}.{minor} / "
            f"VRAM {d['free_memory'] / 1e9:.1f} GB 空き（全 {d['total_memory'] / 1e9:.1f} GB）{state}"
        )
    return "\n".join(lines)
