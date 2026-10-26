"""組織の導電率プリセット（SPEC §8 M7 / §12 の未決事項 2）。

領域に σ を入れるとき、毎回文献を引き直すのは面倒だし写し間違いも起きる。
よく使う組織の代表値をここにまとめておく。

    >>> import facetsetfem
    >>> facetsetfem.tissue("muscle").sigma
    0.35
    >>> case.set_conductivity({1: facetsetfem.tissue("skin").sigma})

# ここの値をそのまま論文に使わないこと

**この表は「入力を組み立てる出発点」であって、文献データベースではない。**
生体組織の導電率は

- 周波数（経皮電気刺激で使う数十 Hz〜数 kHz の帯域でも数倍動く）
- 温度・含水率・測定方向（筋は繊維方向と直交方向で 4 倍以上違う）
- 個体差・部位差

で大きく変わる。ここに載せているのは**桁と相対関係を把握するための代表値**で、
各エントリに周波数帯と出典の性格を書いてある。実際の解析では、対象の周波数と
部位に合わせて文献値を確認し、[`load_library`] で自前の表に差し替えること。

権威あるデータベース（IT'IS Foundation の Tissue Properties など）の取り込みは
利用条件の確認が済んでいないため行っていない（SPEC §12 の未決事項 2）。
外部から取得したデータを使う場合は、出典とライセンスを必ず記録すること
（SPEC §11.2）。

# 自前の表を使う

    >>> library = facetsetfem.load_library("my_tissues.yaml")
    >>> library["bone"].sigma

YAML の形は次のとおり。`sigma` 以外は省略できる。

    muscle:
      sigma: 0.35
      frequency: "DC〜1 kHz"
      source: "社内測定 2026-05"
      note: "繊維方向"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ._core import FacetSetFemError
from .i18n import N_, error_text, tr

__all__ = [
    "Tissue",
    "TISSUES",
    "load_library",
    "tissue",
    "tissue_names",
]


@dataclass(frozen=True)
class Tissue:
    """組織 1 つぶんの代表値。"""

    #: 組織名（英小文字・アンダースコア区切り）
    name: str
    #: 導電率 [S/m]
    sigma: float
    #: 想定している周波数帯
    frequency: str = ""
    #: 値の出どころ
    source: str = ""
    #: 使うときの注意
    note: str = ""
    #: 表示名（同梱のプリセットは英語で持ち、表示するときに `facetsetfem.i18n.tr` を通す）
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"sigma": self.sigma}
        for key in ("frequency", "source", "note", "label"):
            value = getattr(self, key)
            if value:
                d[key] = value
        return d


def _t(name: str, label: str, sigma: float, note: str = "") -> Tissue:
    return Tissue(
        name=name,
        label=label,
        sigma=sigma,
        frequency=N_("DC to a few kHz (the band of transcutaneous stimulation)"),
        source=N_("Representative value widely used at low frequency. Literature values vary widely; check them"),
        note=note,
    )


#: 同梱のプリセット。**代表値であって文献データベースではない**（モジュール冒頭を参照）
TISSUES: dict[str, Tissue] = {
    t.name: t
    for t in [
        _t("skin", N_("Skin"), 0.10, N_("Varies by orders of magnitude with the stratum corneum and electrode wetness; lower when dry")),
        _t("fat", N_("Fat"), 0.04, N_("Subcutaneous fat. Low among tissues")),
        _t("muscle", N_("Muscle (along fibres)"), 0.35, N_("Along the fibres. Across them it drops to about 1/4")),
        _t("muscle_transverse", N_("Muscle (across fibres)"), 0.08, N_("Anisotropy is not supported yet (SPEC §9)")),
        _t("bone_cortical", N_("Cortical bone"), 0.02, N_("Among the lowest of all tissues; diverts current strongly")),
        _t("bone_cancellous", N_("Cancellous bone"), 0.08, N_("Higher than cortical bone because it contains marrow")),
        _t("blood", N_("Blood"), 0.70, N_("Depends on the haematocrit")),
        _t("csf", N_("CSF"), 1.80, N_("The highest of all body tissues")),
        _t("grey_matter", N_("Grey matter"), 0.10),
        _t("white_matter", N_("White matter"), 0.06, N_("Anisotropic along the fibre tracts")),
        _t("nerve", N_("Peripheral nerve"), 0.03, N_("Anisotropic along the fascicles")),
        _t("tendon", N_("Tendon"), 0.04),
        _t("saline_0_9", N_("Saline 0.9%"), 1.40, N_("Sometimes used in place of electrode gel")),
        _t("electrode_gel", N_("Electrode gel"), 0.50, N_("Differs a lot between products; use the datasheet value")),
    ]
}


def tissue_names() -> list[str]:
    """同梱プリセットの組織名（アルファベット順）。"""
    return sorted(TISSUES)


def tissue(name: str, library: dict[str, Tissue] | None = None) -> Tissue:
    """名前から組織を引く。無ければ候補を添えてエラーにする（設計原則 6）。"""
    table = TISSUES if library is None else library
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key in table:
        return table[key]
    # 打ち間違いを拾うために、前方一致するものを候補として出す
    close = [n for n in sorted(table) if n.startswith(key[:3])] or sorted(table)
    raise FacetSetFemError(
        error_text(
            tr("Unknown tissue name: {name}", name=repr(name)),
            tr("tissue presets"),
            tr(
                "Choose from {names} and so on, or read your own table with facetsetfem.load_library()",
                names=", ".join(close[:10]),
            ),
        )
    )


def load_library(path: str | Path) -> dict[str, Tissue]:
    """自前の組織表（YAML）を読む。

    同梱のプリセットは**上書きせず**、読み込んだ表を返す。
    同梱の値を土台にしたい場合は `dict(facetsetfem.TISSUES) | library` のように混ぜること。
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FacetSetFemError(
            error_text(
                tr("Cannot open the tissue table: {error}", error=exc),
                str(path),
                tr("Check that the path is right and that the file is readable"),
            )
        ) from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise FacetSetFemError(
            error_text(
                tr("Cannot read the tissue table as YAML: {error}", error=exc),
                str(path),
                tr("Check the indentation and quotes"),
            )
        ) from exc
    if not isinstance(data, dict):
        raise FacetSetFemError(
            error_text(
                tr("The tissue table is not a mapping"),
                str(path),
                tr('Write entries as "tissue_name: {sigma: value}"'),
            )
        )

    out: dict[str, Tissue] = {}
    for name, entry in data.items():
        where = tr("{part} in {where}", part=name, where=path)
        if isinstance(entry, (int, float)):
            entry = {"sigma": float(entry)}
        if not isinstance(entry, dict) or "sigma" not in entry:
            raise FacetSetFemError(
                error_text(
                    tr("sigma is missing"),
                    where,
                    tr('Write it like "{name}: {{sigma: 0.35}}"', name=name),
                )
            )
        try:
            sigma = float(entry["sigma"])
        except (TypeError, ValueError) as exc:
            raise FacetSetFemError(
                error_text(
                    tr("sigma is not a number: {value}", value=repr(entry["sigma"])),
                    where,
                    tr("Write the value in S/m as a number"),
                )
            ) from exc
        if not sigma > 0.0:
            raise FacetSetFemError(
                error_text(
                    tr("sigma is not positive: {value}", value=sigma),
                    where,
                    tr("The conductivity must be positive (SPEC §2.1)"),
                )
            )
        key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
        out[key] = Tissue(
            name=key,
            sigma=sigma,
            frequency=str(entry.get("frequency", "")),
            source=str(entry.get("source", "")),
            note=str(entry.get("note", "")),
            label=str(entry.get("label", "")),
        )
    return out
