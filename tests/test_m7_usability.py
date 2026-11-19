"""M7: 使い勝手の仕上げ（SPEC §5.1 / §8 M7）。

1. 等値面・流線が Hex8 / Penta6 を含むメッシュでも使える（面共役な四面体分割）
2. 始点の探索が空間分割で速い
3. Undo / Redo
4. パラメータスタディ
5. 組織物性プリセット
"""

from __future__ import annotations

import numpy as np
import pytest

import facetsetfem
from facetsetfem.gui.state import Session
from test_m4_elements_and_backends import write_deck

CURRENT = 2.0e-3
SLAB_SIGMA = {1: 0.10, 2: 0.04, 3: 0.35}


def _solved(mesh):
    case = facetsetfem.Case(mesh)
    case.set_conductivity(SLAB_SIGMA)
    case.add_electrode("E1", role="anode", current=CURRENT, shell_pids=[101])
    case.add_electrode("E2", role="cathode", current=CURRENT, shell_pids=[201])
    case.set_solver(tolerance=1.0e-12)
    return case.solve()


# --- 1. 要素種を選ばないポスト処理 -------------------------------------------


@pytest.mark.parametrize("kind", ["tet", "hex", "penta"])
def test_isosurfaces_work_for_every_element_type(tmp_path, kind):
    """M6 までは CTETRA 以外でエラーになっていた（SPEC §5.1）。"""
    mesh = write_deck(tmp_path, kind, nx=4, ny=3)
    result = _solved(mesh)

    lo, hi = result.value_range("phi")
    points, triangles, scalars = result.iso_surface("phi", 0.5 * (lo + hi))
    assert len(triangles) > 0, f"{kind}: 等値面が空"
    assert points.shape[1] == 3
    assert triangles.shape[1] == 3
    assert len(scalars) == len(points)
    # 三角形の頂点番号が範囲内
    assert triangles.max() < len(points)


@pytest.mark.parametrize("kind", ["tet", "hex", "penta"])
def test_the_isosurface_of_a_layered_slab_is_a_flat_cross_section(tmp_path, kind):
    """一次元問題なので φ の等値面は水平な断面になる。

    対角線の選び方が要素の両側で食い違うと、面積が理論値からずれる。
    """
    mesh = write_deck(tmp_path, kind, nx=4, ny=3)
    result = _solved(mesh)
    lo, hi = result.value_range("phi")
    points, triangles, _ = result.iso_surface("phi", 0.5 * (lo + hi))

    # 全頂点がほぼ同じ z に載る
    z = points[:, 2]
    assert z.max() - z.min() < 1e-6, f"{kind}: 等値面が平面でない（Δz = {z.max() - z.min()}）"

    # 面積は断面（60 × 40 mm）と一致する
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    expected = 0.060 * 0.040
    assert abs(area - expected) < 1e-6 * expected, f"{kind}: 面積 {area}（{expected} のはず）"


@pytest.mark.parametrize("kind", ["tet", "hex", "penta"])
def test_streamlines_work_for_every_element_type(tmp_path, kind):
    mesh = write_deck(tmp_path, kind, nx=4, ny=3)
    result = _solved(mesh)
    lo, hi = mesh.bounding_box()

    seeds = [
        [0.060 * fx, 0.040 * fy, 0.0195]
        for fx in (0.3, 0.5, 0.7)
        for fy in (0.4, 0.6)
    ]
    lines = result.streamlines(seeds)
    assert len(lines) == len(seeds), f"{kind}: 始点が見つからなかったものがある"
    for line in lines:
        assert line.shape[1] == 3
        assert len(line) >= 2
        # メッシュの外に出ない
        assert np.all(line >= np.array(lo) - 1e-9), f"{kind}: 下限を割った"
        assert np.all(line <= np.array(hi) + 1e-9), f"{kind}: 上限を超えた"
        # 上面から入れた電流は下面まで抜ける
        assert line[-1][2] < 1e-6, f"{kind}: 下面に届いていない（終点 {line[-1]}）"


def test_streamlines_are_repeatable(tmp_path):
    mesh = write_deck(tmp_path, "hex", nx=4, ny=3)
    result = _solved(mesh)
    seeds = [[0.02, 0.018, 0.0195]]
    first = result.streamlines(seeds)[0]
    for _ in range(3):
        again = result.streamlines(seeds)[0]
        assert np.array_equal(first, again)


# --- 3. Undo / Redo ---------------------------------------------------------


@pytest.fixture
def session(sample_bdf):
    s = Session()
    s.open_mesh(sample_bdf, "mm")
    return s


def test_undo_and_redo_walk_the_history(session):
    assert not session.can_undo()
    assert not session.can_redo()

    session.checkpoint("領域 1 の設定を変更")
    session.set_conductivity(1, 0.10)
    session.checkpoint("領域 2 の設定を変更")
    session.set_conductivity(2, 0.04)
    assert session.conductivity_of(1) == 0.10
    assert session.conductivity_of(2) == 0.04

    assert session.undo_label() == "領域 2 の設定を変更"
    assert session.undo() == "領域 2 の設定を変更"
    assert session.conductivity_of(2) == 0.0
    assert session.conductivity_of(1) == 0.10

    assert session.undo() == "領域 1 の設定を変更"
    assert session.conductivity_of(1) == 0.0
    assert not session.can_undo()

    assert session.redo() == "領域 1 の設定を変更"
    assert session.conductivity_of(1) == 0.10
    assert session.redo() == "領域 2 の設定を変更"
    assert session.conductivity_of(2) == 0.04
    assert not session.can_redo()


def test_repeated_edits_of_the_same_field_collapse_into_one_step(session):
    """入力欄は 1 文字ごとに値を反映するので、まとめないと undo が使い物にならない。"""
    for value in (0.01, 0.05, 0.10):
        session.checkpoint("領域 1 の設定を変更")
        session.set_conductivity(1, value)
    assert session.conductivity_of(1) == 0.10

    session.undo()
    # 一気に「編集を始める前」へ戻る
    assert session.conductivity_of(1) == 0.0
    assert not session.can_undo()


def test_a_new_edit_after_undo_drops_the_redo_branch(session):
    session.checkpoint("領域 1 の設定を変更")
    session.set_conductivity(1, 0.10)
    session.undo()
    assert session.can_redo()

    session.checkpoint("領域 2 の設定を変更")
    session.set_conductivity(2, 0.04)
    assert not session.can_redo(), "枝分かれした先のやり直しは残さない"


def test_electrodes_can_be_undone(session):
    session.checkpoint("電極 E1 を追加")
    session.add_electrode("E1", role="anode", current=CURRENT)
    assert len(session.study.electrodes) == 1
    session.undo()
    assert len(session.study.electrodes) == 0
    session.redo()
    assert [e.name for e in session.study.electrodes] == ["E1"]


def test_opening_another_model_clears_the_history(session, sample_bdf):
    session.checkpoint("領域 1 の設定を変更")
    session.set_conductivity(1, 0.10)
    assert session.can_undo()
    session.open_mesh(sample_bdf, "mm")
    assert not session.can_undo(), "別のモデルの履歴に戻れてはいけない"
    assert not session.can_redo()


def test_the_gui_exposes_undo_and_redo(main_window, sample_bdf):
    window = main_window
    window.open_mesh(sample_bdf, "mm")

    assert not window.action_undo.isEnabled()
    window.checkpoint("領域 1 の設定を変更")
    window.session.set_conductivity(1, 0.10)
    assert window.action_undo.isEnabled()
    assert "領域 1" in window.action_undo.text()

    window.undo()
    assert window.session.conductivity_of(1) == 0.0
    assert window.action_redo.isEnabled()
    window.redo()
    assert window.session.conductivity_of(1) == 0.10


# --- 4. パラメータスタディ ---------------------------------------------------


@pytest.fixture
def slab_study(tmp_path, sample_bdf, repo_root):
    import shutil

    shutil.copy(sample_bdf, tmp_path / "slab_3layer.bdf")
    shutil.copy(repo_root / "tests" / "models" / "slab_3layer.yaml", tmp_path / "case.yaml")
    return facetsetfem.Study.load(tmp_path / "case.yaml")


def test_a_sweep_covers_the_cartesian_product(slab_study):
    table = facetsetfem.sweep(
        slab_study, {"sigma.2": [0.02, 0.04], "current": [1.0e-3, 2.0e-3]}
    )
    assert len(table) == 4
    assert table.parameters == ["sigma.2", "current"]
    assert not table.failures

    # 線形問題なので R は電流に依らず、ΔV と |J| は電流に比例する
    by_key = {(r["sigma.2"], r["current"]): r for r in table.rows}
    for sigma in (0.02, 0.04):
        low = by_key[(sigma, 1.0e-3)]
        high = by_key[(sigma, 2.0e-3)]
        assert abs(low["impedance_ohm"] - high["impedance_ohm"]) < 1e-9 * low["impedance_ohm"]
        assert abs(high["voltage_V"] - 2 * low["voltage_V"]) < 1e-9 * abs(low["voltage_V"])
        assert abs(high["j_max_A_per_m2"] - 2 * low["j_max_A_per_m2"]) < 1e-9 * low["j_max_A_per_m2"]

    # σ を上げれば抵抗は下がる
    assert by_key[(0.04, 1.0e-3)]["impedance_ohm"] < by_key[(0.02, 1.0e-3)]["impedance_ohm"]


def test_a_sweep_does_not_touch_the_original_study(slab_study):
    before = [m.sigma for m in slab_study.materials]
    facetsetfem.sweep(slab_study, {"sigma.2": [0.01, 0.02]})
    assert [m.sigma for m in slab_study.materials] == before


def test_pairwise_sweeps_line_the_values_up(slab_study):
    table = facetsetfem.sweep(
        slab_study,
        {"sigma.1": [0.08, 0.12], "sigma.3": [0.30, 0.40]},
        product=False,
    )
    assert len(table) == 2
    assert [(r["sigma.1"], r["sigma.3"]) for r in table.rows] == [(0.08, 0.30), (0.12, 0.40)]


def test_pairwise_sweeps_need_matching_lengths(slab_study):
    with pytest.raises(facetsetfem.FacetSetFemError) as excinfo:
        facetsetfem.sweep(slab_study, {"sigma.1": [0.1], "sigma.3": [0.3, 0.4]}, product=False)
    assert "個数" in str(excinfo.value)


def test_an_unknown_parameter_says_what_is_available(slab_study):
    with pytest.raises(facetsetfem.FacetSetFemError) as excinfo:
        facetsetfem.sweep(slab_study, {"sigma.99": [0.1]})
    message = str(excinfo.value)
    assert "99" in message
    assert "1" in message and "2" in message and "3" in message


def test_sweeps_can_keep_going_past_a_broken_case(slab_study):
    # 片方の電極だけ電流を変えると Σ I = 0 が崩れる（SPEC §2.2）
    table = facetsetfem.sweep(
        slab_study,
        {"electrode.E1.current": [2.0e-3, 3.0e-3]},
        keep_going=True,
    )
    assert len(table) == 1, "つり合っているケースだけが残るはず"
    assert len(table.failures) == 1
    assert "電流" in table.failures[0][1]


def test_a_sweep_writes_a_csv(slab_study, tmp_path):
    table = facetsetfem.sweep(slab_study, {"current": [1.0e-3, 2.0e-3]})
    path = tmp_path / "sweep.csv"
    table.write_csv(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3  # 見出し + 2 ケース
    header = lines[0].split(",")
    assert header[0] == "current"
    assert "impedance_ohm" in header
    # 列名は ASCII（SPEC §5.5）
    assert all(c.isascii() for c in lines[0])


def test_a_sweep_reports_progress(slab_study):
    seen = []
    facetsetfem.sweep(
        slab_study,
        {"current": [1.0e-3, 2.0e-3, 3.0e-3]},
        progress=lambda i, n, values: seen.append((i, n)),
    )
    assert seen == [(1, 3), (2, 3), (3, 3)]


# --- 5. 組織物性プリセット ---------------------------------------------------


def test_the_presets_cover_the_usual_tissues():
    names = facetsetfem.tissue_names()
    for expected in ("skin", "fat", "muscle", "bone_cortical", "blood"):
        assert expected in names
    for name in names:
        t = facetsetfem.tissue(name)
        assert t.sigma > 0.0
        # 「どこから来た値か」が必ず添えてある（SPEC §11.2）
        assert t.source
        assert t.frequency


def test_the_preset_ordering_matches_physical_intuition():
    # 骨は脂肪より通しにくく、脳脊髄液は最も通しやすい
    assert facetsetfem.tissue("bone_cortical").sigma < facetsetfem.tissue("fat").sigma
    assert facetsetfem.tissue("fat").sigma < facetsetfem.tissue("muscle").sigma
    assert facetsetfem.tissue("muscle").sigma < facetsetfem.tissue("blood").sigma
    assert facetsetfem.tissue("blood").sigma < facetsetfem.tissue("csf").sigma


def test_an_unknown_tissue_suggests_candidates():
    with pytest.raises(facetsetfem.FacetSetFemError) as excinfo:
        facetsetfem.tissue("muscel")
    message = str(excinfo.value)
    assert "muscel" in message
    assert "muscle" in message


def test_tissue_names_are_normalised():
    assert facetsetfem.tissue("MUSCLE").name == "muscle"
    assert facetsetfem.tissue(" Grey Matter ").name == "grey_matter"
    assert facetsetfem.tissue("grey-matter").name == "grey_matter"


def test_a_custom_library_can_replace_the_presets(tmp_path):
    path = tmp_path / "tissues.yaml"
    path.write_text(
        "bone:\n"
        "  sigma: 0.0123\n"
        "  source: 社内測定 2026-05\n"
        "  frequency: 1 kHz\n"
        "gel: 0.5\n",
        encoding="utf-8",
    )
    library = facetsetfem.load_library(path)
    assert set(library) == {"bone", "gel"}
    assert library["bone"].sigma == 0.0123
    assert library["bone"].source == "社内測定 2026-05"
    assert library["gel"].sigma == 0.5
    # 同梱プリセットは書き換わらない
    assert "gel" not in facetsetfem.TISSUES


def test_a_broken_library_says_where_the_problem_is(tmp_path):
    path = tmp_path / "tissues.yaml"
    path.write_text("bone:\n  note: 値を書き忘れた\n", encoding="utf-8")
    with pytest.raises(facetsetfem.FacetSetFemError) as excinfo:
        facetsetfem.load_library(path)
    message = str(excinfo.value)
    assert "sigma" in message
    assert "bone" in message


def test_a_non_positive_conductivity_is_rejected(tmp_path):
    path = tmp_path / "tissues.yaml"
    path.write_text("bone:\n  sigma: 0.0\n", encoding="utf-8")
    with pytest.raises(facetsetfem.FacetSetFemError) as excinfo:
        facetsetfem.load_library(path)
    assert "正では" in str(excinfo.value)


def test_the_gui_hints_come_from_the_presets():
    from facetsetfem.gui.state import TISSUE_HINTS

    assert TISSUE_HINTS["muscle"] == facetsetfem.tissue("muscle").sigma
    assert set(TISSUE_HINTS) == set(facetsetfem.tissue_names())


# --- パラメータスタディの GUI -----------------------------------------------


def test_the_sweep_dialog_offers_the_parameters_of_the_case(slab_study):
    from facetsetfem.gui.sweep_dialog import parameter_choices

    keys = [key for _label, key in parameter_choices(slab_study)]
    assert "current" in keys
    for m in slab_study.materials:
        assert f"sigma.{m.pid}" in keys
    for e in slab_study.electrodes:
        assert f"electrode.{e.name}.current" in keys


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.02, 0.04, 0.06", [0.02, 0.04, 0.06]),
        ("1e-3 2e-3", [1e-3, 2e-3]),
        ("0.02:0.06:0.01", [0.02, 0.03, 0.04, 0.05, 0.06]),
    ],
)
def test_the_dialog_reads_values_in_the_usual_notations(text, expected):
    from facetsetfem.gui.sweep_dialog import parse_values

    got = parse_values(text)
    assert len(got) == len(expected)
    assert all(abs(a - b) < 1e-12 for a, b in zip(got, expected))


@pytest.mark.parametrize("text", ["", "0.02:0.06", "0.02:0.06:0", "あ"])
def test_the_dialog_rejects_values_it_cannot_read(text):
    from facetsetfem.gui.sweep_dialog import parse_values

    with pytest.raises(ValueError):
        parse_values(text)


def test_the_sweep_dialog_runs_and_fills_the_table(main_window, sample_bdf, tmp_path):
    from facetsetfem.gui.sweep_dialog import SweepDialog

    window = main_window
    window.open_mesh(sample_bdf, "mm")
    for pid, sigma in SLAB_SIGMA.items():
        window.session.set_conductivity(pid, sigma)
    window.session.add_electrode("E1", "anode", CURRENT, [101])
    window.session.add_electrode("E2", "cathode", CURRENT, [201])

    dialog = SweepDialog(window)
    dialog.parameter.setCurrentIndex(
        [dialog.parameter.itemData(i) for i in range(dialog.parameter.count())].index(
            "current"
        )
    )
    dialog.values.setText("1, 2")  # 電流は電極のページと同じ mA で入れる
    dialog.start()
    assert dialog._thread is not None
    dialog._thread.wait(60_000)
    # シグナルはキュー接続なので、イベントを回して受け取る
    from PySide6.QtWidgets import QApplication

    QApplication.processEvents()

    assert dialog.table is not None
    assert len(dialog.table) == 2
    assert [row["current"] for row in dialog.table.rows] == pytest.approx([1e-3, 2e-3])
    assert dialog.result_table.rowCount() == 2
    # 表の見出しは日本語と単位、振った値は入力と同じ mA で出す
    assert dialog.result_table.horizontalHeaderItem(0).text() == "刺激電流（全電極） [mA]"
    assert dialog.result_table.item(1, 0).text() == "2"
    headings = [
        dialog.result_table.horizontalHeaderItem(c).text()
        for c in range(dialog.result_table.columnCount())
    ]
    assert "等価抵抗 R [Ω]" in headings and "impedance_ohm" not in headings
    assert dialog.save_button.isEnabled()

    path = tmp_path / "sweep.csv"
    dialog.table.write_csv(path)
    assert path.exists()
    dialog.close()


# --- 配布（wheel の依存関係。SPEC §12 の 3）---------------------------------

#: import 名 → それを入れる配布パッケージ名。
#: vtkmodules は pyvista が必ず連れてくるので、pyvista を宣言していれば足りる
_DISTRIBUTION_OF = {
    "numpy": "numpy",
    "yaml": "PyYAML",
    "pyvista": "pyvista",
    "pyvistaqt": "pyvistaqt",
    "PySide6": "PySide6",
    # vtkmodules を提供するのは vtk の配布パッケージ（pyvista はそれに依存しているだけ）
    "vtkmodules": "vtk",
}


def _third_party_imports() -> set[str]:
    """facetsetfem パッケージが import しているサードパーティのモジュール（トップレベル名）。"""
    import ast
    import sys
    from pathlib import Path

    package = Path(facetsetfem.__file__).parent
    found: set[str] = set()
    for source in package.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top not in sys.stdlib_module_names and top not in {"facetsetfem", "__future__"}:
                    found.add(top)
    return found


def test_the_wheel_declares_everything_the_package_imports(repo_root):
    """`pip install` した人が GUI まで動かせること。

    以前は GUI が使う pyvistaqt が依存に無く、wheel を入れただけでは
    `python -m facetsetfem.gui` が import で落ちた。逆に使っていない pandas が入っていた。
    """
    import tomllib

    declared = {
        requirement.split(">")[0].split("=")[0].split("<")[0].strip()
        for requirement in tomllib.loads(
            (repo_root / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]["dependencies"]
    }
    imported = _third_party_imports()
    unknown = imported - set(_DISTRIBUTION_OF)
    assert not unknown, f"配布パッケージとの対応が未登録の import: {sorted(unknown)}"
    needed = {_DISTRIBUTION_OF[name] for name in imported}
    assert needed <= declared, f"pyproject.toml に足りない依存: {sorted(needed - declared)}"
    assert declared <= needed, f"使っていない依存: {sorted(declared - needed)}"
