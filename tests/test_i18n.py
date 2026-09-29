"""表示言語の切り替え（SPEC §6.6）。

既定は英語、日本語にも切り替えられる。Python 側の文言は英語をキーにした辞書
（`gfem/locale_ja.py`）から訳を引くので、ソースを走査して

- `tr("...")` / `N_("...")` に渡した英語の文言すべてに訳があること
- 使われていない訳が残っていないこと
- `{name}` の埋め込みが英語と日本語でそろっていること
- 訳を通さない日本語の文字列がソースに残っていないこと

を確かめる。ほかのテストは日本語の表示で回している（conftest.py）。
"""

from __future__ import annotations

import ast
import re
import string
from pathlib import Path

import pytest

import gfem
from gfem.i18n import tr
from gfem.locale_ja import JA

PACKAGE = Path(gfem.__file__).resolve().parent
JAPANESE = re.compile(r"[　-〿぀-ヿ一-鿿！-｠]")

#: 訳を通さなくてよい日本語（言語の名前はその言語で書く）。全角の記号・空白も日本語として扱う
ALLOWED_JAPANESE = {"日本語", "："}  # "：" はログの見出しの判定に使う


def _sources() -> list[Path]:
    return [p for p in sorted(PACKAGE.rglob("*.py")) if p.name != "locale_ja.py"]


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def _marked_texts() -> dict[str, list[str]]:
    """`tr("...")` と `N_("...")` に渡した文言 → 出てくる場所。"""
    found: dict[str, list[str]] = {}
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"tr", "N_"}
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                found.setdefault(node.args[0].value, []).append(f"{path.name}:{node.lineno}")
    return found


def _fields(template: str) -> set[str]:
    return {name for _text, name, _spec, _conv in string.Formatter().parse(template) if name}


def test_every_marked_text_has_a_japanese_translation():
    missing = {text: where for text, where in _marked_texts().items() if text not in JA}
    assert not missing, "locale_ja.py に訳がありません:\n" + "\n".join(
        f"  {where[0]}: {text!r}" for text, where in missing.items()
    )


def test_no_translation_is_left_unused():
    used = set(_marked_texts())
    unused = sorted(set(JA) - used)
    assert not unused, "使われていない訳があります:\n" + "\n".join(f"  {t!r}" for t in unused)


def test_placeholders_match_between_languages():
    wrong = [
        (en, ja) for en, ja in JA.items() if _fields(en) != _fields(ja)
    ]
    assert not wrong, "\n".join(f"  {en!r} / {ja!r}" for en, ja in wrong)


def test_no_japanese_is_hard_coded_outside_the_catalogue():
    """画面に出る日本語はすべて訳を通す（docstring とコメントは日本語のままでよい）。"""
    leftovers = []
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and JAPANESE.search(node.value)
                and node.value not in ALLOWED_JAPANESE
            ):
                leftovers.append(f"  {path.relative_to(PACKAGE)}:{node.lineno}: {node.value[:60]!r}")
    assert not leftovers, "訳を通していない日本語があります:\n" + "\n".join(leftovers)


@pytest.fixture
def english():
    before = gfem.i18n.language()
    gfem.i18n.set_language("en")
    yield
    gfem.i18n.set_language(before)


def test_english_is_the_default_of_a_fresh_process(tmp_path):
    """既定は英語（環境変数 GFEM_LANG で上書きしない限り）。"""
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "GFEM_LANG"}
    out = subprocess.run(
        [sys.executable, "-c", "import gfem; print(gfem.i18n.language())"],
        capture_output=True, text=True, env=env, check=True,
    )
    assert out.stdout.strip() == "en"
    env["GFEM_LANG"] = "ja"
    out = subprocess.run(
        [sys.executable, "-c", "import gfem; print(gfem.i18n.language())"],
        capture_output=True, text=True, env=env, check=True,
    )
    assert out.stdout.strip() == "ja"


def test_core_errors_follow_the_language(english, sample_bdf):
    mesh = gfem.read_nastran(sample_bdf, length_unit="mm")
    case = gfem.Case(mesh)
    with pytest.raises(gfem.GfemError) as excinfo:
        case.validate()
    assert not JAPANESE.search(str(excinfo.value)), str(excinfo.value)
    gfem.i18n.set_language("ja")
    with pytest.raises(gfem.GfemError) as excinfo:
        case.validate()
    assert JAPANESE.search(str(excinfo.value))


def test_tr_fills_placeholders_in_both_languages(english):
    text = "Loaded: {path}"
    if text in JA:
        assert tr(text, path="a.bdf") == "Loaded: a.bdf"
        gfem.i18n.set_language("ja")
        assert "a.bdf" in tr(text, path="a.bdf")


# --- GUI ----------------------------------------------------------------------


def test_session_texts_are_english_by_default(english, sample_bdf):
    from gfem.gui.state import Session

    s = Session()
    s.open_mesh(sample_bdf, "mm")
    summary = s.mesh_summary()
    assert summary.startswith("File: ") and "Nodes: 936" in summary
    assert not JAPANESE.search(summary)
    for pid, sigma in {1: 0.10, 2: 0.04, 3: 0.35}.items():
        s.set_conductivity(pid, sigma)
    s.add_electrode("E1", role="anode", current=2e-3, shell_pids=[101])
    s.add_electrode("E2", role="cathode", current=2e-3, shell_pids=[201])
    s.solve()
    s.set_conductivity(2, 0.05)
    assert s.changes_since_solve() == ["σ (PID 2)"]
    assert "Mean potential of each electrode:" in s.result_summary()


def test_dialogs_are_english_by_default(english, main_window):
    from gfem.gui.sweep_dialog import SweepDialog
    from gfem.gui.widgets import UnitSelector

    dialog = UnitSelector((0.06, 0.04, 0.02))
    assert dialog.windowTitle() == "Choose the length unit"
    dialog.close()
    main_window.session.open_mesh(Path(__file__).parent / "models" / "slab_3layer.bdf", "mm")
    sweep = SweepDialog(main_window)
    assert sweep.windowTitle() == "Parameter study"
    assert sweep.parameter.itemText(0) == "Stimulation current (all electrodes) [mA]"
    sweep.close()


def test_the_startup_language_follows_cli_then_env_then_settings(monkeypatch, tmp_path):
    from PySide6.QtCore import QSettings

    from gfem.gui import language as lang

    store = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(lang, "_settings", lambda: store)
    monkeypatch.delenv("GFEM_LANG", raising=False)
    assert lang.startup_language() == "en", "何も指定しなければ英語"
    lang.save_language("ja")
    assert lang.startup_language() == "ja", "メニューで選んだ言語"
    monkeypatch.setenv("GFEM_LANG", "en")
    assert lang.startup_language() == "en", "環境変数は設定より強い"
    assert lang.startup_language("ja") == "ja", "--lang がいちばん強い"


def test_choosing_a_language_in_the_menu_saves_it(main_window, monkeypatch, tmp_path):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QMessageBox

    from gfem.gui import language as lang

    store = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(lang, "_settings", lambda: store)
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda _p, title, text: shown.append(text))
    before = gfem.i18n.language()
    main_window.choose_language("en")
    assert lang.saved_language() == "en"
    assert gfem.i18n.language() == before, "開いているウィンドウの言語はその場では変えない"
    assert shown == ["The display language will change the next time you start GenericFEM."]
    assert main_window.language_actions["en"].isChecked()
    assert not main_window.language_actions["ja"].isChecked()
    main_window.choose_language(before)
