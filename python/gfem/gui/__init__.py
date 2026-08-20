"""PySide6 + PyVista の GUI（SPEC §6）。

起動:

    .venv/Scripts/python.exe -m gfem.gui [model.bdf | case.yaml]

GUI にロジックは置かず、状態は `gfem.gui.state.Session` が持つ（SPEC §7）。
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """GUI を起動する。"""
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    argv = sys.argv if argv is None else [sys.argv[0], *argv]
    app = QApplication.instance() or QApplication(argv)
    window = MainWindow()
    window.show()

    for arg in argv[1:]:
        path = Path(arg)
        if path.suffix.lower() in {".yaml", ".yml"}:
            window.open_case(path)
        else:
            window.open_mesh(path, "mm")
        break

    return app.exec()
