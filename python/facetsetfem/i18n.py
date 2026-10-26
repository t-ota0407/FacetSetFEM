"""表示言語（英語 / 日本語）の切り替え（SPEC §6.6）。

既定は英語。言語はプロセス全体で 1 つで、Rust コアと共有する（エラー・警告の文言も
同じ言語になる）。GUI は起動時に設定から決める。

Python 側の文言は**英語をキーにした辞書**（`facetsetfem/locale_ja.py`）から日本語訳を引く::

    from facetsetfem.i18n import tr
    tr("Open model...")                      # → "モデルを開く..."（日本語のとき）
    tr("Loaded: {path}", path=path)          # 引数は str.format で埋める

モジュールの読み込み時に決まる定数（表示量の名前など）は、そのときの言語で訳すと
後から言語を変えても直らないので、英語のまま `N_("...")` で印を付けておき、
使うときに `tr(...)` を通す。訳の付け忘れ・余った訳・`{...}` の食い違いは
`tests/test_i18n.py` がソースを走査して検出する。
"""

from __future__ import annotations

from ._core import language as _language
from ._core import set_language as _set_language
from .locale_ja import JA

__all__ = ["LANGUAGES", "N_", "error_text", "language", "set_language", "tr"]

#: 選べる言語（コード, その言語での名前）。メニューに並べる
LANGUAGES = [("en", "English"), ("ja", "日本語")]


def set_language(code: str) -> None:
    """表示言語を決める（`"en"` / `"ja"`）。Rust コアの文言も切り替わる。"""
    _set_language(code)


def language() -> str:
    """いまの表示言語（`"en"` / `"ja"`）。"""
    return _language()


def tr(text: str, /, **kwargs) -> str:
    """英語の文言 `text` を、いまの言語にする。`kwargs` があれば `str.format` で埋める。"""
    template = JA.get(text, text) if _language() == "ja" else text
    return template.format(**kwargs) if kwargs else template


def N_(text: str) -> str:  # noqa: N802 - gettext の慣習に合わせる
    """訳す文言の印だけを付ける（訳すのは使うときの `tr`）。"""
    return text


#: エラーの分類の表示名（Rust コアの `ErrorKind::label` と同じ言葉にそろえる）
_ERROR_KINDS = {
    "setup": N_("Setup error"),
    "internal": N_("Internal error"),
}


def error_text(message: str, where: str | None = None, hint: str | None = None,
               kind: str = "setup") -> str:
    """Python 側で作るエラーの文面。Rust コアの `FacetSetFemError` と同じ体裁にそろえる::

        [Setup error] 何が起きたか
          Where: どこで
          Fix: どうすれば直るか

    `message` / `where` / `hint` は訳したものを渡す（ここでは見出しだけを訳す）。
    """
    text = f"[{tr(_ERROR_KINDS[kind])}] {message}"
    if where:
        text += "\n  " + tr("Where: {where}", where=where)
    if hint:
        text += "\n  " + tr("Fix: {hint}", hint=hint)
    return text
