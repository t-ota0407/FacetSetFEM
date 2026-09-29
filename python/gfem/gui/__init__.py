"""PySide6 + PyVista の GUI（SPEC §6）。

起動:

    .venv/Scripts/python.exe -m gfem.gui [case.yaml | model.bdf [--unit mm]]

GUI にロジックは置かず、状態は `gfem.gui.state.Session` が持つ（SPEC §7）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

__all__ = ["main"]


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    """コマンドライン引数を解釈する。

    メッシュ（.bdf / .nas）を渡したときは、**長さ単位を黙って決めない**（SPEC §3.2）。
    `--unit` があればそれを使い、無ければ GUI で寸法を見せてから選ばせる。
    ケースファイル（.yaml）は単位を自分で持っているので `--unit` は要らない。
    """
    from .. import length_units
    from ..i18n import LANGUAGES, tr

    parser = argparse.ArgumentParser(prog="python -m gfem.gui", description=tr("GenericFEM GUI"))
    parser.add_argument(
        "path", nargs="?", type=Path, help=tr("case file (.yaml) or mesh (.bdf / .nas)")
    )
    parser.add_argument(
        "--unit",
        choices=length_units(),
        help=tr("length unit of the mesh coordinates; if omitted, the GUI shows the size and asks"),
    )
    parser.add_argument(
        "--lang",
        choices=[code for code, _name in LANGUAGES],
        help=tr("display language (default: the one chosen in View → Language, otherwise English)"),
    )
    args = parser.parse_args(argv)
    if args.unit is not None and args.path is not None and _is_case_file(args.path):
        parser.error(tr("--unit is only for opening a mesh (a case file records its own unit)"))
    return args


def _peek_language(argv: list[str]) -> str | None:
    """`--lang xx` / `--lang=xx` を引数の解釈より先に拾う。"""
    for i, arg in enumerate(argv):
        if arg == "--lang" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--lang="):
            return arg.split("=", 1)[1]
    return None


def _is_case_file(path: Path) -> bool:
    return path.suffix.lower() in {".yaml", ".yml"}


def main(argv: list[str] | None = None) -> int:
    """GUI を起動する。"""
    argv = sys.argv[1:] if argv is None else argv

    from PySide6.QtWidgets import QApplication

    from .language import apply_startup_language

    app = QApplication.instance() or QApplication(sys.argv[:1])
    # **ウィンドウを作る前に言語を決める。** 文言は作るときに訳すので、後からは変わらない。
    # --lang だけは引数を解釈する前に拾う（引数の説明もその言語で出すため）
    apply_startup_language(app, _peek_language(argv))
    args = parse_arguments(argv)

    from .main_window import MainWindow

    window = MainWindow()
    window.show()

    if args.path is not None:
        if _is_case_file(args.path):
            window.open_case(args.path)
        elif args.unit is not None:
            window.open_mesh(args.path, args.unit)
        else:
            window.open_mesh_asking_unit(args.path)

    return app.exec()
