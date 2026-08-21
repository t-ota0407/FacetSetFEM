"""組織の導電率プリセット（SPEC §8 M7 / §12 の未決事項 2）。

領域に σ を入れるとき、毎回文献を引き直すのは面倒だし写し間違いも起きる。
よく使う組織の代表値をここにまとめておく。

    >>> import gfem
    >>> gfem.tissue("muscle").sigma
    0.35
    >>> case.set_conductivity({1: gfem.tissue("skin").sigma})

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

    >>> library = gfem.load_library("my_tissues.yaml")
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

from ._gfem import GfemError

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
    #: 日本語名（GUI の表示用）
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
        frequency="DC〜数 kHz（経皮電気刺激の帯域）",
        source="低周波帯で広く使われる代表値。文献値の範囲は広いので要確認",
        note=note,
    )


#: 同梱のプリセット。**代表値であって文献データベースではない**（モジュール冒頭を参照）
TISSUES: dict[str, Tissue] = {
    t.name: t
    for t in [
        _t("skin", "皮膚", 0.10, "角質層の状態と電極の湿り具合で桁が動く。乾燥時はもっと低い"),
        _t("fat", "脂肪", 0.04, "皮下脂肪。組織の中では低いほう"),
        _t("muscle", "筋（繊維方向）", 0.35, "繊維方向。直交方向は 1/4 程度まで下がる"),
        _t("muscle_transverse", "筋（繊維直交方向）", 0.08, "異方性は M7 時点では未対応（SPEC §9）"),
        _t("bone_cortical", "皮質骨", 0.02, "組織の中で最も低い部類。電流を大きく迂回させる"),
        _t("bone_cancellous", "海綿骨", 0.08, "骨髄を含むぶん皮質骨より高い"),
        _t("blood", "血液", 0.70, "ヘマトクリット値に依存する"),
        _t("csf", "脳脊髄液", 1.80, "生体組織の中で最も高い"),
        _t("grey_matter", "灰白質", 0.10),
        _t("white_matter", "白質", 0.06, "線維方向に異方性がある"),
        _t("nerve", "末梢神経", 0.03, "束の走行方向に異方性がある"),
        _t("tendon", "腱", 0.04),
        _t("saline_0_9", "生理食塩水 0.9%", 1.40, "電極ゲルの代用として使うことがある"),
        _t("electrode_gel", "電極ゲル", 0.50, "製品ごとに大きく違う。仕様書の値を使うこと"),
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
    raise GfemError(
        f"[設定エラー] 知らない組織名です: {name!r}\n"
        f"  場所: 組織プリセット\n"
        f"  対処: {', '.join(close[:10])} などから選ぶか、"
        f"gfem.load_library() で自前の表を読み込んでください"
    )


def load_library(path: str | Path) -> dict[str, Tissue]:
    """自前の組織表（YAML）を読む。

    同梱のプリセットは**上書きせず**、読み込んだ表を返す。
    同梱の値を土台にしたい場合は `dict(gfem.TISSUES) | library` のように混ぜること。
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GfemError(
            f"[設定エラー] 組織表を開けません: {exc}\n"
            f"  場所: {path}\n"
            f"  対処: パスが正しいか、読み取り権限があるかを確認してください"
        ) from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise GfemError(
            f"[設定エラー] 組織表を YAML として読めません: {exc}\n"
            f"  場所: {path}\n"
            f"  対処: インデントと引用符を確認してください"
        ) from exc
    if not isinstance(data, dict):
        raise GfemError(
            f"[設定エラー] 組織表がマッピングではありません\n"
            f"  場所: {path}\n"
            f"  対処: 「組織名: {{sigma: 値}}」の形で書いてください"
        )

    out: dict[str, Tissue] = {}
    for name, entry in data.items():
        where = f"{path} の {name}"
        if isinstance(entry, (int, float)):
            entry = {"sigma": float(entry)}
        if not isinstance(entry, dict) or "sigma" not in entry:
            raise GfemError(
                f"[設定エラー] sigma がありません\n"
                f"  場所: {where}\n"
                f"  対処: 「{name}: {{sigma: 0.35}}」のように書いてください"
            )
        try:
            sigma = float(entry["sigma"])
        except (TypeError, ValueError) as exc:
            raise GfemError(
                f"[設定エラー] sigma が数値ではありません: {entry['sigma']!r}\n"
                f"  場所: {where}\n"
                f"  対処: S/m の値を数値で書いてください"
            ) from exc
        if not sigma > 0.0:
            raise GfemError(
                f"[設定エラー] sigma が正ではありません: {sigma}\n"
                f"  場所: {where}\n"
                f"  対処: 導電率は正の値でなければなりません（SPEC §2.1）"
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
