# GenericFEM

経皮電気刺激（transcutaneous electrical stimulation）の生体内電流分布を解析する、有限要素法シミュレーションアプリケーション。

**仕様は [SPEC.md](SPEC.md) にあります。作業を始める前に必ず読んでください。**

---

## 現在の状態

| フェーズ | 状態 |
|---|---|
| 仕様策定 | **完了**（SPEC.md v0.7） |
| 開発環境構築 | **完了**（2026-08-18） |
| M0: 骨組み | **完了**（2026-08-19） |
| M1: 体験一周（GUI で solve まで） | **完了**（2026-08-20） |
| M2: 検証（解析解・収束次数） | **完了**（2026-08-20） |
| M3: ポスト処理の充実 | **完了**（2026-08-20） |
| M4: 要素拡張と CPU 並列 | **完了**（2026-08-21） |
| M5: CUDA バックエンド | **未着手 ← 次はここから** |
| M6 以降 | 未着手 |

マイルストーンの定義は SPEC.md §8 を参照。

### いま動くもの

GUI:

```bash
.venv/Scripts/python.exe -m gfem.gui tests/models/slab_3layer.yaml
```

Python API（GUI と同じコアを叩く）:

```python
import gfem

mesh = gfem.read_nastran("tests/models/slab_3layer.bdf", length_unit="mm")
print(gfem.describe_bounding_box(mesh))  # 60 mm x 40 mm x 20 mm = 0.06 m x 0.04 m x 0.02 m

case = gfem.Case(mesh)
case.set_conductivity({1: 0.10, 2: 0.04, 3: 0.35})   # PSOLID の PID ごとに σ [S/m]
case.add_electrode("E1", role="anode",   current=2e-3, shell_pids=[101])
case.add_electrode("E2", role="cathode", current=2e-3, shell_pids=[201])

result = case.solve()
print(result)                    # <Solution iterations=94 residual=7.12e-11 R=206.3933 ohm>
print(result.impedance)          # 電極間の等価抵抗 [Ω]
print(result.j_magnitude.max())  # |J| の最大 [A/m²]（要素値）

for s in result.region_stats():  # 領域ごとの統計（体積重みパーセンタイル）
    print(s["name"], s["j_magnitude"]["p99"], s["hotspot"]["element_id"])

result.write_vtu("out.vtu")              # ParaView で開ける
result.write_region_stats_csv("stats.csv")
result.write_summary_json("summary.json")  # 入力ハッシュ・Git の状態つき
```

ケースファイル（YAML）から一括で:

```python
result = gfem.Study.load("tests/models/slab_3layer.yaml").solve()
```

| 機能 | 状態 |
|---|---|
| Nastran 読み込み（小/大/自由形式、継続行、8 種のカード） | ✔ |
| 座標の SI 正規化と単位取り違えの検出表示 | ✔ |
| 参照切れ・ID 重複・退化要素・非多様体・非連結の検査 | ✔ |
| Tet4 の CSR アセンブリ | ✔ |
| 射影付き Jacobi-CG（ゲージ固定は射影。1 節点ピン留めはしない） | ✔ |
| 電極荷重（面積按分の一様電流密度） | ✔ |
| E, J, 電力密度、電極間インピーダンス、保存則チェック | ✔ |
| VTU 出力（節点値・要素値の両方、zlib 圧縮） | ✔ |
| GUI（領域ハイライト・電極選択・断面スライス・収束曲線） | ✔ |
| ケースファイル（YAML）の保存・読込 | ✔ |
| 領域統計 CSV（体積重みパーセンタイル・ホットスポット） | ✔ |
| 等値面（Marching Tetrahedra）・ストリームライン・クリップ | ✔ |
| プロビナンス JSON（入力ハッシュ・Git の状態・実行時刻） | ✔ |
| Hex8 / Penta6 の要素剛性（CHEXA / CPENTA） | ✔ |
| CORD2R / CORD2C / CORD2S の座標変換 | ✔ |
| CPU 並列（rayon。逐次とビット単位で一致） | ✔ |
| 等値面・流線・面隣接・L2 誤差評価の Hex8 / Penta6 対応 | M7 |
| CUDA | M5 |
| AMG 前処理 | M6 |

### 規模と速度（M4 実測）

20 コア、release ビルド、3 層スラブ（tol = 1e-10）。逐次と並列は**ビット単位で同じ答え**を出す。

| 節点 | 四面体 | 反復 | 逐次 [s] | 並列 [s] | 倍率 |
|---|---|---|---|---|---|
| 14,973 | 79,200 | 232 | 0.150 | 0.147 | 1.02 |
| 151,470 | 855,360 | 421 | 2.95 | 0.91 | 3.2 |
| 602,775 | 3,474,240 | 633 | 18.2 | 4.3 | 4.2 |

`solver.backend` は `auto`（既定。2 万節点以上で並列）/ `cpu` / `cpu_parallel`。

```bash
cargo test --release -p gfem-core --test scaling -- --ignored --nocapture
```

### 正しさの裏づけ（M2）

SPEC §8 M2 の解析解ベンチマーク 5 件と、メッシュ収束性試験を実装済み。

| 検証 | 結果 |
|---|---|
| 一様場・層状媒質（1 次元） | 厳密一致（誤差は丸めレベル） |
| 同心球殻（多層含む） | R の相対誤差 1.5 % |
| 半無限媒質上の点電極 | 一様半空間で ρ_a が理論値の 0.02 % 以内 |
| 二層半無限媒質（鏡像法級数） | Wenner 見かけ比抵抗比が 0.4 % 以内 |
| 収束次数 φ / E | 実測 1.93 / 0.87（理論 2 / 1） |

保存則（電流保存・エネルギー収支・離散残差）は**全ての solve で自動チェック**され、
逸脱があれば結果に警告として載る。

```bash
cargo test                                   # 通常の回帰
cargo test --release -- --include-ignored    # 二層半無限媒質を含む検証一式
```

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
| GUI 起動 | `.venv/Scripts/python.exe -m gfem.gui [model.bdf \| case.yaml]` |

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
    src/element/      形状関数 trait（shape）、tet4 / hex8 / penta6、面積分
    src/mesh/         メッシュ表現、ID 索引、外表面抽出、等値面、流線
    src/io/nastran/   バルクデータパーサ（field / reader / parse / coord の 4 層）
    src/io/vtu.rs     VTU ライタ
    src/assemble/     CSR、非ゼロ構造、剛性・荷重のアセンブリ、並列アセンブリ
    src/solver/       決定的リダクション、バックエンド trait（逐次 / 並列）、射影 CG
    src/post/         E・J・保存則チェック・領域統計・解析解との誤差評価
    src/provenance.rs 入力ハッシュ・Git の状態・実行時刻
    src/analysis.rs   SPEC §4.1 の流れを通す入口
  gfem-cuda/          CUDA バックエンド（M5 で作る。未作成）
  gfem-py/            PyO3 バインディング
python/gfem/          高レベル Python API
  study.py            ケースファイル（YAML）
  gui/                PySide6 + PyVista の GUI
scripts/              サンプルモデル生成など開発用スクリプト
tests/                Python テストとサンプルモデル
```

---

## 次にやること（M5）

SPEC.md §8 の M5「CUDA バックエンド」。目標は 1000 万節点が回ること。

1. `gfem-cuda` クレートを作り、cudarc で f64 の CSR SpMV と CG を実装する
2. CSR と作業ベクトルを VRAM に常駐させる
3. 内積は決定的な階層リダクション（CPU 版と同じ固定ブロック分割）
4. VRAM 不足・GPU 非搭載のときは CPU へ自動フォールバックし、ログに明記する
5. **CPU 版と GPU 版がビット単位で一致することをテストする**

`LinearBackend` はいま `&[f64]` のスライスを受け渡している。GPU ではベクトルが
デバイスメモリに載るので、trait に「ベクトル表現」の関連型を持たせる変更が要る見込み
（SPEC §4.4）。CG 本体が全てのベクトル演算をこの trait 経由で行っているので、
その変更は trait 定義と CG のシグネチャに閉じる。

### M4 から持ち越したもの

等値面・流線・面隣接・L2 誤差評価は Tet4 専用のまま。要素種を混ぜるには
**面共役な四面体分割**（隣り合う要素が共有する四角形面で同じ対角線を選ぶ分割）が
必要で、選び方がずれると等値面に隙間ができ流線の追跡も壊れる。M7 で対応する
（SPEC §5.1 / §8）。**解く・数える・書き出す・GUI で表示する経路は要素種を選ばない。**

---

## 未決事項

SPEC.md §12 を参照。
