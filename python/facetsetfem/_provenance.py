"""実行時の作業ツリーの状態（SPEC §5.5 のプロビナンス記録）。

実行サマリに残す Git の状態は、ふつうは**ビルド時**に拡張モジュールへ埋め込んだもの
（`crates/facetsetfem-core/build_git.rs`）。ところが `maturin develop` の開発環境では、
Python のコードはビルドし直さなくても作業ツリーから直接読まれる。すると

- Python のコードを編集してから解いた結果が、ビルド時の「汚れていない」を名乗る
- コミットを進めたのに拡張モジュールをビルドし直し忘れ、古い Rust のまま解く

が起きうる。そこでサマリを書く時点の作業ツリーも記録する（拡張モジュールから呼ばれる）。
wheel で入れた環境では作業ツリーが無いので何も記録しない。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

#: 結果に効くソース（リポジトリのルートからの相対パス）。
#: SPEC.md や tests/ の編集は結果に効かないので数えない
RESULT_SOURCES = ("python/facetsetfem", "crates", "Cargo.toml", "Cargo.lock", "pyproject.toml")


def source_tree() -> tuple[str, str | None, bool | None] | None:
    """`(リポジトリのルート, HEAD, 未コミットの変更があるか)`。

    facetsetfem が git の作業ツリーから読み込まれていなければ `None`。
    git が使えないなど判定できなかった項目は `None` にする（サマリを書くのを止めない）。
    """
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    # python/facetsetfem/ から読まれている（＝開発環境の作業ツリー）ときだけ
    if here != root / "python" / "facetsetfem" or not (root / ".git").exists():
        return None
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain", "--", *RESULT_SOURCES)
    return str(root), commit, None if status is None else bool(status)


def _git(root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip()
