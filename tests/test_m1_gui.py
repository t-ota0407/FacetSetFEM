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
