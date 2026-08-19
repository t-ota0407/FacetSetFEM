"""pytest 共通のフィクスチャ。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> ModuleType:
    """`scripts/` 以下のスクリプトをモジュールとして読み込む。"""
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"gfem_scripts.{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def sample_bdf() -> Path:
    """リポジトリに入っている小フィールド形式のサンプルモデル。"""
    path = REPO_ROOT / "tests" / "models" / "slab_3layer.bdf"
    if not path.exists():
        pytest.fail(
            f"{path} がありません。"
            " .venv/Scripts/python.exe scripts/make_sample_mesh.py を実行してください"
        )
    return path


@pytest.fixture(scope="session")
def make_sample_mesh() -> ModuleType:
    """サンプルモデル生成スクリプト。"""
    return _load_script("make_sample_mesh")
