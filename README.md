# GenericFEM

経皮電気刺激（transcutaneous electrical stimulation）の生体内電流分布を解析する、有限要素法シミュレーションアプリケーション。

**仕様は [SPEC.md](SPEC.md) にあります。作業を始める前に必ず読んでください。**

---

## 現在の状態

| フェーズ | 状態 |
|---|---|
| 仕様策定 | **完了**（SPEC.md v1.0） |
| 開発環境構築 | **完了**（2026-08-18） |
| M0: 骨組み | **完了**（2026-08-19） |
| M1: 体験一周（GUI で solve まで） | **完了**（2026-08-20） |
| M2: 検証（解析解・収束次数） | **完了**（2026-08-20） |
| M3: ポスト処理の充実 | **完了**（2026-08-20） |
| M4: 要素拡張と CPU 並列 | **完了**（2026-08-21） |
| M5: CUDA バックエンド | **完了**（2026-08-21） |
| M6: AMG 前処理 | **完了**（2026-08-21） |
| M7: 使い勝手の仕上げ | **完了**（2026-08-21） |

マイルストーンの定義は SPEC.md §8 を参照。

### いま動くもの

GUI:

```bash
.venv/Scripts/python.exe -m gfem.gui tests/models/slab_3layer.yaml
```

大きなモデルを開いたとき、ツリーで領域や電極を選び直したとき、表示を作り直すときは
**処理中のローディング**が出る。読み込みと書き出しはワーカースレッドで走るので、
その間もスピナーは滑らかに回り、何をしているか（読み込み中か、領域を取り出しているか、
描画中か）が文言で分かる。**200 ms 以内に終わる操作では出ない**ので、小さなモデルを
触っている間は一度も現れない（10 万節点を超えたら猶予なしで即座に出す。SPEC §6.4）。

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

**シェル要素を持たないモデル**（人体モデルはこれが普通）は、体表面から直接電極を選ぶ:

```python
mesh = gfem.read_nastran("head.nas", length_unit="mm")
print(mesh.n_shell_elements)     # 0 ← シェル PID では電極を置けない

for row in mesh.surface_pids():  # 外表面に出ている領域（面積の大きい順）
    print(row["pid"], row["n_faces"], row["area"])   # 51 49017 0.16 m² ← 皮膚

patch = mesh.pick_patch([-0.068, 1.717, -0.028], 0.025, 51)  # 中心 [m], 半径 [m], PID
print(patch["n_faces"], patch["area"])                        # 923 面 19.5 cm²

case.add_electrode("E1", role="anode", current=2e-3,
                   patches=[{"center": [-0.068, 1.717, -0.028], "radius": 0.025, "pid": 51}])
```

GUI では「クリックで選ぶ」を押して 3D ビューの体表面をクリックすると、その周りの面が
電極になる（半径と絞り込む領域はパネルで指定）。**PID で絞り込むのが大事**で、
首の切断面のように「外表面だが体表面ではない」面を避けるため（SPEC §11.1）。
72.7 万節点 / 411 万四面体の頭部モデルで、1 クリックから強調表示まで 0.13 秒。

縁は**筆で 1 枚ずつ**整える:

| 操作 | 動き |
|---|---|
| クリック | パッチを 1 枚置く（中心＋半径） |
| Shift + ドラッグ | 面を塗り足す |
| Alt + ドラッグ | 面を消す |
| ふつうのドラッグ | 回転（従来どおり） |

半径を最小にすればちょうど 1 枚だけ選ばれる。カーソルの下には**これから塗られる範囲**が
薄く出る。手直しはケースファイルに**差分**（`added` / `removed`）として残るので、
「中心と半径」という読める指定はそのまま——1 ストロークは Undo 1 回で戻せる。

ケースファイル（YAML）から一括で:

```python
result = gfem.Study.load("tests/models/slab_3layer.yaml").solve()
```

パラメータスタディ（M7）。σ・刺激電流・電極の面・ソルバ設定を振って一括で解く:

```python
study = gfem.Study.load("tests/models/slab_3layer.yaml")
table = gfem.sweep(study, {"sigma.2": [0.02, 0.04], "current": [1e-3, 2e-3]})
table.write_csv("sweep.csv")
print(table.rows[0]["impedance_ohm"])
```

**刺激強度は `current`（全電極まとめて）で振ること。** 1 極だけ変えると Σ I_k = 0 が
崩れて解が存在しなくなる（SPEC §2.2）。GUI からは「実行 → パラメータスタディ...」。

組織物性のプリセット（M7）:

```python
gfem.tissue("muscle").sigma     # 0.35
gfem.tissue_names()             # 使える名前の一覧
library = gfem.load_library("my_tissues.yaml")   # 自前の表に差し替える
```

**同梱の値は桁と相対関係を掴むための出発点で、文献データベースではない。**
周波数・温度・測定方向で大きく変わるので、実際の解析では文献値を確認すること
（SPEC §12 の未決事項 2）。

| 機能 | 状態 |
|---|---|
| Nastran 読み込み（小/大/自由形式、継続行、8 種のカード） | ✔ |
| 座標の SI 正規化と単位取り違えの検出表示 | ✔ |
| 参照切れ・ID 重複・退化要素・非多様体・非連結の検査 | ✔ |
| Tet4 の CSR アセンブリ | ✔ |
| 射影付き前処理付き CG（ゲージ固定は射影。1 節点ピン留めはしない） | ✔ |
| 電極荷重（面積按分の一様電流密度） | ✔ |
| 体表面から電極を選ぶ（3D ビューのクリック。シェル要素の無いモデル用） | ✔ |
| 電極面を筆で 1 枚ずつ足す・外す（差分でケースファイルに残る） | ✔ |
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
| CUDA（f64・VRAM 常駐・CPU とビット単位で一致） | ✔ |
| AMG 前処理（平滑化凝集。CPU / GPU 両対応） | ✔ |
| 等値面・流線・面隣接・L2 誤差評価の Hex8 / Penta6 対応（面共役な四面体分割） | ✔ |
| 流線の始点探索の空間分割 | ✔ |
| Undo / Redo | ✔ |
| 処理中のローディング表示（読み込み・書き出しはワーカースレッド） | ✔ |
| 3D ビューは外表面を陰影付きで描画（結果があれば値で色分け） | ✔ |
| パラメータスタディ（Python API と GUI） | ✔ |
| 組織物性プリセット | ✔ |

### 規模と速度（M5 実測）

RTX A6000 / 20 コア、release ビルド、3 層スラブ（tol = 1e-12）。
**3 つのバックエンドはビット単位で同じ答え**を出す。速度だけが変わる。

| 節点 | 四面体 | 反復 | 逐次 [s] | 並列 [s] | GPU [s] | CG だけ 逐次 / 並列 / GPU [s] |
|---|---|---|---|---|---|---|
| 14,973 | 79,200 | 287 | 0.16 | 0.20 | 0.20 | 0.10 / 0.15 / 0.14 |
| 151,470 | 855,360 | 517 | 3.15 | 1.02 | 1.33 | 2.63 / 0.51 / 0.81 |
| 301,070 | 1,715,472 | 638 | 7.87 | 2.18 | 2.09 | 6.82 / 1.13 / 1.04 |
| 602,775 | 3,474,240 | 776 | 20.2 | 4.88 | 3.83 | 18.1 / 2.73 / 1.67 |

1000 万節点（10,022,646 節点 / 59,099,040 四面体、tol = 1e-9）は GPU で **57 秒**
（準備 25 / アセンブリ 10 / CG 17.5 / ポスト 4）、VRAM 6.3 GB、解析解との相対誤差 6.5e-14。

`solver.backend` は `auto` / `cpu` / `cpu_parallel` / `cuda`。
`auto` は 2 万節点以上で CPU 並列、25 万節点以上で GPU（使えれば）を選ぶ。
**25 万節点より小さいうちは CPU 並列のほうが速い**——GPU は万能ではない。

### 前処理（M6 実測）

`solver.preconditioner` は `jacobi`（既定）/ `amg` / `none`。
**バックエンドと違い、前処理を変えると反復の道筋が変わる**ので、答えは
収束判定の範囲内でしか一致しない（同じ前処理を選べばビット単位で一致する）。

GPU での比較（時間は CG のみ。AMG は階層構築を含む）:

| 節点 | Jacobi 反復 | Jacobi [s] | AMG 反復 | AMG [s] | うち階層 [s] | 倍率 |
|---|---|---|---|---|---|---|
| 41,055 | 342 | 0.20 | 34 | 0.07 | 0.04 | 2.8 |
| 151,470 | 517 | 0.55 | 41 | 0.20 | 0.13 | 2.7 |
| 602,775 | 776 | 1.51 | 36 | 0.75 | 0.49 | 2.0 |
| 1,519,560 | 1,017 | 2.74 | 49 | 1.63 | 1.23 | 1.7 |

1000 万節点では **1,423 反復 → 42 反復**、GPU の CG が 17.5 秒 → 9.2 秒、全体 57 → 50 秒。
**反復回数が規模にほとんど依存しなくなる**のが AMG を入れた目的で、そこは達成できている。
一方で**階層の構築が新しい律速**になった（150 万節点で 1.63 秒のうち 1.23 秒。
反復そのものは 0.4 秒で Jacobi の 6.8 倍速い）。

導電率コントラストが大きいケース（骨 0.02 と筋肉 0.35 で 17 倍）では Jacobi 168 回 →
AMG 22 回。**小さい問題では Jacobi のほうが速い**——階層構築の元が取れないため。

```bash
cargo test --release -p gfem-core --test scaling -- --ignored --nocapture
cargo test --release -p gfem-core --test amg -- --ignored --nocapture      # AMG と Jacobi の比較
cargo test --release -p gfem-cuda -- --ignored --nocapture   # GPU の計測と 1000 万節点
```

使える GPU は Python から確認できる:

```python
import gfem
print(gfem.describe_gpus())
print(gfem.cuda_check(10_000_000))   # この規模を GPU に載せられるか（None なら OK）
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
| 配布用 wheel の作成 | `.venv/Scripts/python.exe scripts/build_release.py` |

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
    src/mesh/         メッシュ表現、ID 索引、外表面抽出、等値面、流線、
                      面共役な四面体分割（tetview）、要素探索の一様格子（spatial）
    src/io/nastran/   バルクデータパーサ（field / reader / parse / coord の 4 層）
    src/io/vtu.rs     VTU ライタ
    src/assemble/     CSR、非ゼロ構造、剛性・荷重のアセンブリ、並列アセンブリ
    src/solver/       決定的リダクション、バックエンド trait（逐次 / 並列）、射影 CG、
                      外部バックエンドの差し込み口（external.rs）
    src/solver/amg/   代数マルチグリッド（凝集・補間・疎行列積・V サイクル）
    src/post/         E・J・保存則チェック・領域統計・解析解との誤差評価
    src/provenance.rs 入力ハッシュ・Git の状態・実行時刻
    src/analysis.rs   SPEC §4.1 の流れを通す入口
  gfem-cuda/          CUDA バックエンド。gfem-core にのみ依存する
    src/kernels.cu    f64 の CSR SpMV・ベクトル演算・決定的リダクション
    src/backend.rs    LinearBackend の実装（CSR と作業ベクトルは VRAM 常駐）
  gfem-py/            PyO3 バインディング（cuda フィーチャで GPU 対応）
python/gfem/          高レベル Python API
  study.py            ケースファイル（YAML）
  sweep.py            パラメータスタディ
  tissues.py          組織物性プリセット
  gui/                PySide6 + PyVista の GUI
    busy.py           処理中のローディング表示とワーカースレッド（SPEC §6.4）
scripts/              サンプルモデル生成、配布用 wheel の作成
tests/                Python テストとサンプルモデル
```

---

## 次にやること

v1.0 のマイルストーンは全て終わった。次に効くのは**速度ではなく前後処理**（下記）と、
SPEC §9 に穴を空けてある拡張（異方性導電率・2 次要素・完全電極モデル）。

### M5・M6 で見えた課題（速度）

1000 万節点の内訳は AMG で **準備 25 秒 / アセンブリ 11 秒 / CG 9 秒 / ポスト 4 秒**。
CG を速くした結果、**外表面抽出（準備の大半）が CG の 2.8 倍まで相対的に重くなった。**
次に効くのは前後処理の並列化で、CG のさらなる高速化ではない。

AMG 自体の残りの伸びしろは階層構築（1000 万節点で 7.1 秒）。凝集は決定性のため
逐次でなければならないが、最細レベル行列の複製と VRAM への 2 回目の転送は
CG が持っているものを共有すれば省ける（SPEC §4.5 の末尾）。

### M7 で見つけた既存の不具合（流線）

四面体分割を通したことで、M6 までの流線追跡が 2 つの壊れ方をしていたことが分かった。

1. **入ってきた面を出口の候補から無条件に外していた。** そこが本当に最も近い出口の
   ときに、より遠い面まで突き抜けて**メッシュの外に点が出る**
2. **稜線にちょうど載った始点で破綻していた。** 場が外向きの四面体を選んでしまい、
   実測では 10²⁹ m 先へ飛んだ。GUI は始点を格子状に置くので日常的に踏む経路だった

どちらも直し、「メッシュの外に点が出ていないか」を見張る回帰テストを入れた
（SPEC §6.3）。

---

## 配布

```bash
.venv/Scripts/python.exe scripts/build_release.py           # テスト → wheel
.venv/Scripts/python.exe scripts/build_release.py --no-cuda # CPU だけの版
```

`dist/` に abi3 wheel（Python 3.12 以降で共通）ができる。受け取った側は

```bash
pip install gfem-0.1.0-cp312-abi3-win_amd64.whl
python -m gfem.gui
```

だけでよく、**Rust も CUDA Toolkit も要らない**（GPU で走らせるには NVIDIA ドライバは要る）。
PyInstaller で 1 つの exe に固めない理由は SPEC §12 の未決事項 3 に書いてある。

---

## 未決事項

SPEC.md §12 を参照。残っているのは組織物性の文献ソース（IT'IS の利用条件）と、
CQUAD4 が大きく非平面のときの警告しきい値の 2 つ。
