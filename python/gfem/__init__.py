"""GenericFEM — 経皮電気刺激シミュレーションの Python API。

GUI（`gfem.gui`）もバッチスクリプトも、このモジュールを通して同じ Rust コアを叩く
（SPEC §6.5）。GUI 側にロジックは置かない。

M0 で使えるのは Nastran の読み込みと VTU 出力まで:

    >>> import gfem
    >>> mesh = gfem.read_nastran("tests/models/slab_3layer.bdf", length_unit="mm")
    >>> mesh.n_nodes
    936
    >>> print(gfem.describe_bounding_box(mesh))
    60 mm x 40 mm x 20 mm = 0.06 m x 0.04 m x 0.02 m

物性の割り当て・電極定義・solve は M1 で追加する（SPEC §8）。
"""

from __future__ import annotations

from ._gfem import (
    GfemError,
    Mesh,
    ReadReport,
    __version__,
    length_units,
    read_nastran,
)

__all__ = [
    "GfemError",
    "Mesh",
    "ReadReport",
    "__version__",
    "describe_bounding_box",
    "length_units",
    "read_nastran",
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
    s = f"{v:.6g}"
    return s
