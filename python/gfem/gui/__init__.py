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

    parser = argparse.ArgumentParser(prog="python -m gfem.gui", description="GenericFEM の GUI")
    parser.add_argument("path", nargs="?", type=Path, help="ケースファイル（.yaml）かメッシュ（.bdf / .nas）")
    parser.add_argument(
        "--unit",
        choices=length_units(),
        help="メッシュの座標の長さ単位。省略すると寸法を見せてから選ばせる",
    )
    args = parser.parse_args(argv)
    if args.unit is not None and args.path is not None and _is_case_file(args.path):
        parser.error("--unit はメッシュを開くときだけ使えます（ケースファイルは単位を持っています）")
    return args


def _is_case_file(path: Path) -> bool:
    return path.suffix.lower() in {".yaml", ".yml"}


def main(argv: list[str] | None = None) -> int:
    """GUI を起動する。"""
    args = parse_arguments(sys.argv[1:] if argv is None else argv)

    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
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
