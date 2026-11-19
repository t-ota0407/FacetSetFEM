"""配布用の wheel を作る（SPEC §8 M7 / §12 の未決事項 3）。

    .venv/Scripts/python.exe scripts/build_release.py
    .venv/Scripts/python.exe scripts/build_release.py --no-cuda   # CPU だけの版

できた wheel は `dist/` に入る。渡された側は

    pip install facetsetfem-0.1.0-cp312-abi3-win_amd64.whl

だけで、Rust も CUDA Toolkit も無しに使える（GPU で走らせるには
NVIDIA ドライバと、CUDA Toolkit 12.x の NVRTC が要る。無ければ CPU で解く。SPEC §10.1）。

# なぜ PyInstaller で 1 つの exe にしないのか

SPEC §12 の未決事項 3 は「PyInstaller による単体化か、環境構築手順を配るか」
だった。wheel を採ったのは次の理由による。

- **PySide6 + VTK は単体化が重い。** 3D 表示のために VTK の
  レンダリングバックエンドを丸ごと抱き込むことになり、
  数百 MB の exe になるうえ、GPU ドライバとの相性問題が出たときに切り分けにくい
- **研究用途では Python から叩けるほうが価値が高い。** バッチでパラメータを
  振る（`facetsetfem.sweep`）のも、結果を pandas に流すのも、
  ライブラリとして入っているほうが素直（SPEC §6.5）
- **wheel は abi3 なので Python 3.12 以降で 1 つ作れば済む。**
  ビルド機と実行機で Python の細かいバージョンが違っても動く

GUI だけ配りたくなったら、この wheel を入れた仮想環境ごと固めるのが
いちばん確実で、そのときに改めて検討する。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("$", " ".join(command))
    return subprocess.run(command, check=True, cwd=ROOT, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="CUDA バックエンドを外して作る（CUDA が無い機械へ配るとき）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "dist",
        help="wheel の置き場所（既定: dist/）",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="テストを回さずに作る（普段は回すこと）",
    )
    args = parser.parse_args(argv)

    maturin = ROOT / ".venv" / "Scripts" / "maturin.exe"
    if not maturin.exists():
        maturin_path = shutil.which("maturin")
        if maturin_path is None:
            print(
                "maturin が見つかりません。\n"
                "  対処: .venv/Scripts/python.exe -m pip install maturin",
                file=sys.stderr,
            )
            return 1
        maturin = Path(maturin_path)

    if not args.skip_tests:
        print("--- テスト ---")
        run(["cargo", "test", "--release", "--workspace"])
        run([sys.executable, "-m", "pytest", "-q"])

    print("--- wheel ---")
    command = [str(maturin), "build", "--release", "--out", str(args.out)]
    if args.no_cuda:
        command += ["--no-default-features", "--features", "pyo3/extension-module"]
    # maturin は VIRTUAL_ENV と CONDA_PREFIX が両方あると止まる
    env = dict(os.environ)
    env.pop("CONDA_PREFIX", None)
    run(command, env=env)

    wheels = sorted(args.out.glob("facetsetfem-*.whl"))
    if not wheels:
        print("wheel ができていません", file=sys.stderr)
        return 1
    newest = max(wheels, key=lambda p: p.stat().st_mtime)
    size = newest.stat().st_size / 1e6
    print(f"\nできました: {newest}（{size:.1f} MB）")
    print("  CUDA:", "外してあります" if args.no_cuda else "入っています")
    print("\n渡された側の手順:")
    print(f"  pip install {newest.name}")
    print("  python -m facetsetfem.gui            # GUI")
    print("  python -c \"import facetsetfem; print(facetsetfem.describe_gpus())\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
