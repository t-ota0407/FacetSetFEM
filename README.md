# GenericFEM

経皮電気刺激（transcutaneous electrical stimulation）の生体内電流分布を解析する、有限要素法シミュレーションアプリケーション。

**仕様は [SPEC.md](SPEC.md) にあります。作業を始める前に必ず読んでください。**

---

## 現在の状態

| フェーズ | 状態 |
|---|---|
| 仕様策定 | **完了**（SPEC.md v0.1） |
| 開発環境構築 | **完了**（2026-08-18） |
| M0: 骨組み | **未着手 ← 次はここから** |
| M1: 体験一周（GUI で solve まで） | 未着手 |
| M2: 検証（解析解・収束次数） | 未着手 |
| M3 以降 | 未着手 |

マイルストーンの定義は SPEC.md §8 を参照。

---

## 開発環境

### 構築済みのもの（2026-08-18 時点で検証済み）

| 種別 | 内容 |
|---|---|
| Rust | 1.97.1 stable / `x86_64-pc-windows-msvc`（clippy・rustfmt・rust-analyzer 同梱） |
| ビルドツール | Visual Studio 2022 (17.13) + Windows SDK 10.0.19041 — リンク動作確認済み |
| Python | 3.14.3（プロジェクト内の `.venv`） |
| Python 依存 | `requirements.txt` に固定。NumPy 2.5.2 / SciPy 1.18.0 / pandas 3.0.5 / VTK 9.6.2 / PyVista 0.48.4 / PySide6 6.11.1 / maturin 1.14.1 / pytest 9.1.1 |
| PyO3 | 0.29.2（`maturin develop` → Python から import・呼び出しまで疎通確認済み） |
| 描画 | PyVista のオフスクリーン描画が RTX A6000 上で動作確認済み（OpenGL 4.5） |
| CUDA | Toolkit 11.8 / Driver 596.72 |

### ハードウェア

| 項目 | 内容 |
|---|---|
| GPU | **NVIDIA RTX A6000 × 2**（各 48 GB VRAM、Compute Capability 8.6） |
| CPU | 20 コア |
| RAM | 128 GB |

SPEC.md §4.2 の見積り（1000 万節点で GPU 常駐 2.5 GB）に対して VRAM が 48 GB あるため、**1 億節点クラスまで載る**計算になる。v1.0 の目標は 1000 万節点のまま据え置く。

RTX A6000（Ampere）の FP64 スループットは FP32 の 1/32 だが、**CG はメモリ帯域律速（768 GB/s）**なので実害はない。倍精度を使う判断に変更はない。

---

## セットアップ（新しいマシンで一から始める場合）

```bash
# 1. Rust
winget install -e --id Rustlang.Rustup

# 2. Python 仮想環境（Python 3.14 系）
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Windows で MSVC ツールチェーンを使うため、Visual Studio Build Tools（C++ ワークロード）と Windows SDK が必要。

---

## 日常的に使うコマンド

Git Bash から使う場合、Rust は PATH に入っていないことがあるので先に通す:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

| 目的 | コマンド |
|---|---|
| Rust のビルド | `cargo build --release` |
| Rust のテスト | `cargo test` |
| Lint | `cargo clippy -- -D warnings` |
| フォーマット | `cargo fmt` |
| Python 拡張のビルド（開発時） | `.venv/Scripts/maturin.exe develop --release` |
| Python のテスト | `.venv/Scripts/python.exe -m pytest` |
| GUI 起動（M1 以降） | `.venv/Scripts/python.exe -m gfem.gui` |

`maturin develop` は `VIRTUAL_ENV` を見るので、Git Bash からは環境変数を明示するのが確実:

```bash
VIRTUAL_ENV='C:\Users\t-ota\Desktop\GenericFEM\.venv' .venv/Scripts/maturin.exe develop --release
```

---

## 構成（SPEC.md §7 の計画）

```
crates/
  gfem-core/    Rust コア。CUDA・Python に一切依存しない
  gfem-cuda/    CUDA バックエンド（feature フラグ）
  gfem-py/      PyO3 バインディング
python/gfem/    高レベル Python API と PySide6 GUI
tests/          解析解ベンチ・収束性試験・サンプルモデル
```

まだディレクトリは作成していない（M0 で作る）。

---

## 次にやること（M0）

SPEC.md §8 の M0:

1. Cargo ワークスペースを作る（`gfem-core` / `gfem-py`）
2. 最小の Nastran パーサ（`GRID` / `CTETRA` / `PSOLID`、小フィールド形式 + 自由形式）
3. VTU ライタ
4. 小規模サンプルモデル（直方体の多層メッシュ）を生成するスクリプト
5. `maturin develop` で Python から読み込めるところまで疎通

---

## 未決事項

SPEC.md §12 を参照。バージョン管理（git）はまだ初期化していない。
