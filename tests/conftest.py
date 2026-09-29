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


@pytest.fixture(scope="session")
def full_face_bdf(tmp_path_factory, make_sample_mesh) -> Path:
    """上面全体を陽極、下面全体を陰極にしたサンプル。

    一次元問題になるので、各層で E = I/(σA) という解析解と直接照合できる
    （SPEC §8 M2 の 1・2）。
    """
    out = tmp_path_factory.mktemp("models") / "slab_full_face.bdf"
    make_sample_mesh.main(["-o", str(out), "--electrodes", "full"])
    return out


#: サンプルモデルの寸法 [m] と層構成（PID, 厚さ [m], σ [S/m]）。PID 順で、積む順ではない
#: （z = 0 から muscle・fat・skin の順に積んである。皮膚が電極を貼る上面）
SLAB_AREA = 0.060 * 0.040
SLAB_LAYERS = [(1, 0.002, 0.10), (2, 0.004, 0.04), (3, 0.014, 0.35)]


@pytest.fixture(scope="session")
def slab_sigma() -> dict[int, float]:
    return {pid: sigma for pid, _t, sigma in SLAB_LAYERS}

# --- Qt を使うテストの下ごしらえ --------------------------------------------


@pytest.fixture(scope="session")
def qt_app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="session")
def main_window(qt_app):
    """MainWindow はプロセス内で 1 つだけ作る。

    VTK の OpenGL コンテキストを何度も作り直すと不安定になるため、
    テストごとに作らず、開き直して状態だけ初期化する。
    """
    pytest.importorskip("pyvistaqt")
    from gfem.gui.main_window import MainWindow

    try:
        w = MainWindow()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"3D ビューを作成できません: {exc}")
    yield w
    w.ask_before_closing = False  # 保存していない変更の確認ダイアログでテストを止めない
    w.close()
