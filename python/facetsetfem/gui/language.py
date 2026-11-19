"""GUI の表示言語の決め方と保存（SPEC §6.6）。

起動時の言語は、強い順に

1. コマンドラインの `--lang en|ja`
2. 環境変数 `FACETSETFEM_LANG`
3. メニュー（表示 → 言語）で選んで保存した設定
4. 英語（既定）

で決める。メニューで選び直した言語は設定に残し、次に起動したときから効く
（開いているウィンドウは作り直さない。全ページを作り直す仕組みに見合わないため）。
Qt の標準のボタン（OK・Cancel・ファイルダイアログなど）は Qt に同梱の訳を読み込む。
"""

from __future__ import annotations

import os

from PySide6.QtCore import QLibraryInfo, QLocale, QSettings, QTranslator

from ..i18n import LANGUAGES, set_language

__all__ = ["apply_startup_language", "save_language", "saved_language"]

#: 設定の保存先（QSettings の組織名・アプリ名）と鍵
SETTINGS_ORG = "FacetSetFEM"
SETTINGS_APP = "FacetSetFEM"
LANGUAGE_KEY = "language"

#: Qt 同梱の訳（標準のボタン・ダイアログ）。アプリが生きている間は持っておく
_qt_translators: list[QTranslator] = []


def _settings() -> QSettings:
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def _known(code: str | None) -> str | None:
    if not code:
        return None
    head = code.strip().lower().replace("-", "_").split("_")[0]
    return head if head in {c for c, _name in LANGUAGES} else None


def saved_language() -> str | None:
    """メニューで選んで保存した言語（無ければ None）。"""
    return _known(str(_settings().value(LANGUAGE_KEY, "") or ""))


def save_language(code: str) -> None:
    """次に起動したときの言語を保存する。"""
    _settings().setValue(LANGUAGE_KEY, code)


def startup_language(cli: str | None = None) -> str:
    """起動時の言語（上の優先順位で）。"""
    return (
        _known(cli)
        or _known(os.environ.get("FACETSETFEM_LANG"))
        or saved_language()
        or "en"
    )


def apply_startup_language(app, cli: str | None = None) -> str:
    """起動時の言語を決めて、コアと Qt の標準部品に反映する。決めた言語を返す。"""
    code = startup_language(cli)
    set_language(code)
    install_qt_translation(app, code)
    return code


def install_qt_translation(app, code: str) -> None:
    """Qt の標準部品（OK / Cancel・ファイルダイアログ）の訳を読み込む。英語なら何もしない。"""
    for translator in _qt_translators:
        app.removeTranslator(translator)
    _qt_translators.clear()
    if code == "en":
        return
    path = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    for name in ("qtbase", "qt"):
        translator = QTranslator(app)
        if translator.load(QLocale(code), name, "_", path):
            app.installTranslator(translator)
            _qt_translators.append(translator)
            break

