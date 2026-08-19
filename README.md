# GenericFEM

経皮電気刺激（transcutaneous electrical stimulation）の生体内電流分布を解析する、有限要素法シミュレーションアプリケーション。

**仕様は [SPEC.md](SPEC.md) にあります。作業を始める前に必ず読んでください。**

---

## 現在の状態

| フェーズ | 状態 |
|---|---|
| 仕様策定 | **完了**（SPEC.md v0.3） |
| 開発環境構築 | **完了**（2026-08-18） |
| M0: 骨組み | **完了**（2026-08-19） |
| M1: 体験一周（GUI で solve まで） | **未着手 ← 次はここから** |
| M2: 検証（解析解・収束次数） | 未着手 |
| M3 以降 | 未着手 |

マイルストーンの定義は SPEC.md §8 を参照。

### M0 で動くもの

```python
import gfem

mesh = gfem.read_nastran("tests/models/slab_3layer.bdf", length_unit="mm")
print(mesh)                              # <Mesh nodes=936 solids=4032 shells=48 unit=mm>
print(gfem.describe_bounding_box(mesh))  # 60 mm x 40 mm x 20 mm = 0.06 m x 0.04 m x 0.02 m
print(mesh.solid_pids())                 # [1, 2, 3]      領域（PSOLID）
print(mesh.shell_pids())                 # [101, 201]     電極候補面（PSHELL）
print(mesh.report)                       # カード件数・未対応カード・警告
mesh.write_vtu("out.vtu")                # ParaView で開ける
```

- Nastran バルクデータの読み込み（小フィールド / 大フィールド / 自由形式、継続行、
  GRID・CTETRA・CHEXA・CPENTA・CTRIA3・CQUAD4・PSOLID・PSHELL）
- 座標の SI 正規化と、単位取り違えを検出するためのバウンディングボックス表示
- 参照切れ・ID 重複・退化要素・孤立節点の検査と読み込みレポート
- VTU（zlib 圧縮つき）出力

まだ無いのは**解く部分**（アセンブリ・ソルバ・ポスト処理・GUI）。M1 で入れる。

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
| サンプルモデルの再生成 | `.venv/Scripts/python.exe scripts/make_sample_mesh.py` |
| GUI 起動（M1 以降） | `.venv/Scripts/python.exe -m gfem.gui` |

`maturin develop` は `VIRTUAL_ENV` を見るので、Git Bash からは環境変数を明示するのが確実:

```bash
VIRTUAL_ENV='C:\Users\t-ota\Desktop\GenericFEM\.venv' .venv/Scripts/maturin.exe develop --release
```

`CONDA_PREFIX` と `VIRTUAL_ENV` が両方立っていると maturin が拒否する。その場合は片方を外す:

```bash
env -u CONDA_PREFIX VIRTUAL_ENV='C:\Users\t-ota\Desktop\GenericFEM\.venv' .venv/Scripts/maturin.exe develop --release
```

Rust だけを触ったときは `cargo test`、Python から見える形まで確認するときは
`maturin develop` してから `pytest` を回す。

---

## 構成（SPEC.md §7 の計画）

```
Cargo.toml            ワークスペース
pyproject.toml        maturin（Rust 拡張）と pytest の設定
crates/
  gfem-core/          Rust コア。CUDA・Python に一切依存しない
    src/element/      要素の種類と幾何量（要素剛性は M1 以降）
    src/mesh/         メッシュ表現と ID 索引
    src/io/nastran/   バルクデータパーサ（field / reader / parse の 3 層）
    src/io/vtu.rs     VTU ライタ
  gfem-cuda/          CUDA バックエンド（M5 で作る。未作成）
  gfem-py/            PyO3 バインディング
python/gfem/          高レベル Python API（GUI は M1 で追加）
scripts/              サンプルモデル生成など開発用スクリプト
tests/                Python テストとサンプルモデル
```

---

## 次にやること（M1）

SPEC.md §8 の M1「体験一周」。ゴールは GUI で
「読込 → 単位選択 → PID に σ 入力 → シェル PID を anode/cathode に割当 → solve → 断面表示」
が通ること。

1. Tet4 の要素剛性と CSR アセンブリ（SPEC §2.3 / §4.2）
2. 射影付き Jacobi-CG（SPEC §2.2 / §4.3）— ゲージ固定は 1 節点ピン留めではなく射影
3. 電極荷重ベクトル（面積按分。SPEC §2.4）
4. 保存則チェック（SPEC §5.4）
5. PySide6 + PyVista の GUI と、ケースファイル（YAML）の保存・読込
6. numpy への zero-copy 受け渡し（SPEC §6.3）

---

## 未決事項

SPEC.md §12 を参照。
