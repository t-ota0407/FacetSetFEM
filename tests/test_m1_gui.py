"""GUI（SPEC §6）の受け入れテスト。

GUI にロジックを置かない方針なので、検証の主眼は

- `Session`（Qt 非依存の状態）が SPEC §6.2 の操作フローを一通り回せること
- Qt 側がそれを正しく表示・編集し、solve を別スレッドで回せること

の 2 点。3D 描画には OpenGL コンテキストが要るため、それが取れない環境では
Qt 側のテストを自動的にスキップする。
"""

from __future__ import annotations

import pytest

import gfem
from gfem.gui.state import DISPLAY_FIELDS, Session

SIGMA = {1: 0.10, 2: 0.04, 3: 0.35}
NAMES = {1: "skin", 2: "fat", 3: "muscle"}


# --- Qt を使わない部分 ------------------------------------------------------


@pytest.fixture
def session(sample_bdf) -> Session:
    """SPEC §6.2 の手順 1〜3 まで進めたセッション。"""
    s = Session()
    s.open_mesh(sample_bdf, "mm")
    for pid, sigma in SIGMA.items():
        s.set_conductivity(pid, sigma, NAMES[pid])
    s.add_electrode("E1", "anode", 2.0e-3, [101])
    s.add_electrode("E2", "cathode", 2.0e-3, [201])
    return s


def test_session_walks_through_the_workflow(session):
    # 手順 1: 読み込みレポート（単位の確認表示を含む。SPEC §3.2）
    summary = session.mesh_summary()
    assert "60 mm x 40 mm x 20 mm" in summary
    assert "0.06 m x 0.04 m x 0.02 m" in summary

    # 手順 2: 領域に σ
    assert session.conductivity_of(2) == 0.04
    assert session.region_name(3) == "muscle"

    # 手順 3: 電極の合計面積と平均電流密度
    info = {e["name"]: e for e in session.electrode_info()}
    assert info["E1"]["area"] == pytest.approx(300e-6, rel=1e-12)

    # 手順 4: 整合性チェック
    assert session.validate() == []

    # 手順 5: solve
    result = session.solve()
    assert result.iterations > 0

    # 手順 6: 結果の表示
    grid = session.grid()
    assert "phi [V]" in grid.point_data
    assert "|J| [A/m2]" in grid.cell_data
    assert grid.point_data["phi [V]"] == pytest.approx(result.potential)
    assert [f.key for f in session.available_fields()] == [f.key for f in DISPLAY_FIELDS]
    assert "等価抵抗" in session.result_summary()


def test_missing_conductivity_is_caught_before_solving(sample_bdf):
    s = Session()
    s.open_mesh(sample_bdf, "mm")
    s.add_electrode("E1", "anode", 2.0e-3, [101])
    s.add_electrode("E2", "cathode", 2.0e-3, [201])
    with pytest.raises(gfem.GfemError) as excinfo:
        s.validate()
    assert "導電率が未設定" in str(excinfo.value)


def test_electrode_names_stay_unique(session):
    """電極名はツリー・削除・結果で電極を引く鍵なので、重複を作らせない。"""
    with pytest.raises(gfem.GfemError, match="すでにあります"):
        session.add_electrode("E1")
    with pytest.raises(gfem.GfemError, match="すでにあります"):
        session.rename_electrode("E2", "E1")
    with pytest.raises(gfem.GfemError, match="空"):
        session.rename_electrode("E2", "  ")
    session.rename_electrode("E2", "E3")
    assert [e.name for e in session.study.electrodes] == ["E1", "E3"]


def test_region_stats_table_volume_is_in_the_unit_of_its_heading(session):
    """体積の見出しと値の単位が揃っていること。

    以前は見出しが mm³ なのに値は m³ × 1e6（＝ cm³）で、1000 倍ずれていた。
    サンプルモデルは 60 mm × 40 mm × 20 mm = 48 cm³。
    """
    from gfem.gui.state import REGION_STATS_COLUMNS

    session.solve()
    column = REGION_STATS_COLUMNS.index("体積 [cm³]")
    total = sum(float(row[column]) for row in session.region_stats_rows())
    assert total == pytest.approx(48.0, rel=1e-3)


def test_summary_records_the_conditions_that_were_solved(session, tmp_path):
    """GUI の実行サマリにも、**解いたときの条件**が残る（SPEC §5.5）。

    以前はディスク上のケースファイルを読み直していたので、.bdf から開いた
    セッションでは条件が 1 行も残らず、解いた後の編集も区別できなかった。
    """
    import json

    import yaml

    session.solve()
    session.set_conductivity(3, 0.99)  # 解いた後の編集はサマリに入らない
    out = tmp_path / "summary.json"
    session.write_summary_json(out)

    text = json.loads(out.read_text(encoding="utf-8"))["provenance"]["case_text"]
    solved = gfem.Study.from_dict(yaml.safe_load(text))
    assert {m.pid: m.sigma for m in solved.materials} == SIGMA
    assert [e.name for e in solved.electrodes] == ["E1", "E2"]


def test_region_and_electrode_geometry_for_display(session):
    # 領域ハイライト
    region = session.region_grid(3)
    assert region.n_cells == 2304
    # 電極面
    assert session.electrode_polydata("E1").n_cells == 24
    assert session.electrode_polydata("E2").n_cells == 24
    # 電極候補面すべて
    assert session.shell_polydata().n_cells == 48
    # 外表面（SPEC §6.3 の軽量データ）
    assert session.surface_polydata().n_cells == 944


def test_slice_of_the_result(session):
    session.solve()
    grid = session.grid()
    cut = grid.slice(normal="x", origin=grid.center)
    assert cut.n_cells > 0
    assert "phi [V]" in cut.point_data


def test_session_round_trips_through_a_case_file(session, tmp_path):
    out = tmp_path / "case.yaml"
    session.save_case(out)

    reopened = Session()
    reopened.open_case(out)
    assert reopened.conductivity_of(2) == 0.04
    assert [e.name for e in reopened.study.electrodes] == ["E1", "E2"]
    assert reopened.mesh.n_nodes == session.mesh.n_nodes
    # 同じ結果が出る
    assert reopened.solve().impedance == pytest.approx(session.solve().impedance, rel=0)


def test_opening_a_case_fills_in_regions_missing_from_the_yaml(session, tmp_path):
    """YAML に書かれていない PID も領域一覧に出す（σ を割り当てられるように）。"""
    session.study.materials = [m for m in session.study.materials if m.pid != 3]
    out = tmp_path / "partial.yaml"
    session.save_case(out)

    reopened = Session()
    reopened.open_case(out)
    assert [m.pid for m in reopened.study.materials] == [1, 2, 3]
    assert reopened.conductivity_of(3) == 0.0


# --- Qt を使う部分 ----------------------------------------------------------


@pytest.fixture
def window(main_window, sample_bdf):
    """SPEC §6.2 の手順 1〜3 まで進めた状態のウィンドウ。"""
    w = main_window
    w.open_mesh(sample_bdf, "mm")  # セッションごと初期化される
    for pid, sigma in SIGMA.items():
        w.session.set_conductivity(pid, sigma, NAMES[pid])
    w.session.add_electrode("E1", "anode", 2.0e-3, [101])
    w.session.add_electrode("E2", "cathode", 2.0e-3, [201])
    w.refresh_tree()
    return w


def test_tree_shows_the_model_structure(window):
    labels = [
        window.tree.topLevelItem(i).text(0)
        for i in range(window.tree.topLevelItemCount())
    ]
    assert labels == ["Mesh（単位: mm）", "領域 (PID)", "電極", "ソルバ設定", "結果"]

    regions = window.tree.topLevelItem(1)
    assert regions.childCount() == 3
    assert "skin" in regions.child(0).text(0)
    assert "0.1 S/m" in regions.child(0).text(0)

    electrodes = window.tree.topLevelItem(2)
    assert electrodes.childCount() == 2
    assert "anode" in electrodes.child(0).text(0)
    assert "2 mA" in electrodes.child(0).text(0)


def test_region_page_edits_the_conductivity(window):
    window.page_region.bind(2)
    assert window.page_region.sigma_spin.value() == pytest.approx(0.04)
    window.page_region.sigma_spin.setValue(0.05)
    assert window.session.conductivity_of(2) == pytest.approx(0.05)


def test_electrode_page_shows_area_and_current_density(window):
    window.page_electrode.bind("E1")
    text = window.page_electrode.summary.text()
    assert "24 面" in text
    assert "300.0 mm²" in text
    assert "6.667 A/m²" in text


def test_renaming_an_electrode_to_an_existing_name_is_refused(window, monkeypatch):
    shown: list[str] = []
    monkeypatch.setattr(
        "gfem.gui.main_window.QMessageBox.critical", lambda *args: shown.append(args[2])
    )
    page = window.page_electrode
    page.bind("E2")
    page.name_edit.setText("E1")
    page._commit()

    assert [e.name for e in window.session.study.electrodes] == ["E1", "E2"]
    assert page.name_edit.text() == "E2"  # 入力欄も元の名前に戻る
    assert shown and "E1" in shown[0]

    # 重複しない名前なら変えられる
    page.name_edit.setText("陰極")
    page._commit()
    assert [e.name for e in window.session.study.electrodes] == ["E1", "陰極"]


def test_solver_page_keeps_a_node_reference_from_the_case_file(window):
    """選択肢に無い基準電位（`node:<ID>`）を、ほかの項目の編集で上書きしない。

    以前は他の項目を 1 つ触った時点で `zero_mean` などに置き換わっていた。
    """
    window.session.study.output.reference_potential = "node:1"
    page = window.page_solver
    page.reload()
    assert page.reference.currentText() == "node:1"

    page.max_iterations.setValue(12_345)  # 別の項目を編集する
    assert window.session.study.solver.max_iterations == 12_345
    assert window.session.study.output.reference_potential == "node:1"


def test_consistency_check_reports_electrode_geometry(window):
    assert window.check_consistency() == []
    log = window.log_view.toPlainText()
    assert "整合性チェック: 問題ありません" in log
    assert "電流密度" in log


def test_solve_runs_in_a_worker_thread_and_updates_the_ui(window, qt_app):
    window.start_solve()
    assert window.solver is not None
    assert window.solver.wait(60_000), "solve がタイムアウトしました"
    qt_app.processEvents()

    result = window.session.solution
    assert result is not None
    assert result.iterations > 0
    # 収束曲線に全反復ぶんの履歴が入っている
    assert len(window.convergence.history) == result.iterations + 1
    # 保存則チェックの表示
    checks = window.checks_view.toPlainText()
    assert "保存則チェック" in checks
    assert "すべて許容範囲内" in checks
    assert "R = " in window.log_view.toPlainText()


def test_view_and_screenshot(window, tmp_path):
    window.refresh_view()
    index = window.page_result.mode.findData("slice")
    window.page_result.mode.setCurrentIndex(index)
    window.refresh_view()
    png = tmp_path / "view.png"
    window.screenshot(png)
    assert png.exists() and png.stat().st_size > 0


def test_unit_selector_shows_both_units(qt_app):
    from gfem.gui.widgets import format_extent

    text = format_extent((60.0, 40.0, 20.0), "mm")
    assert "60 mm" in text
    assert "0.06 m" in text
