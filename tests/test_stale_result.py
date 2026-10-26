"""条件を変えたら、古い結果だと知らせる（SPEC §6.2 の手順 6）。

結果は解いたときの条件の写し（`Session.solved_study`）と一緒に持っている。いまの条件と
比べ、答えが変わる条件が違っていれば「古い結果」として結果ページ・ステータスバー・
ツリーに出す。結果そのものは消さない（解き直すまで見比べられる）。
"""

from __future__ import annotations

import copy

import pytest

from facetsetfem.gui.state import Session

SIGMA = {1: 0.10, 2: 0.04, 3: 0.35}
CURRENT = 2.0e-3


def _setup(session: Session) -> None:
    for pid, sigma in SIGMA.items():
        session.set_conductivity(pid, sigma)
    session.add_electrode("E1", role="anode", current=CURRENT, shell_pids=[101])
    session.add_electrode("E2", role="cathode", current=CURRENT, shell_pids=[201])


@pytest.fixture(scope="module")
def solved(sample_bdf) -> Session:
    """1 度だけ解いたセッション。各テストは条件の写しを差し替えて使う。"""
    s = Session()
    s.open_mesh(sample_bdf, "mm")
    _setup(s)
    s.solve()
    return s


@pytest.fixture
def session(solved) -> Session:
    # 解くのは重いので、結果はそのまま、いまの条件だけを解いたときの写しに戻す
    solved.study = copy.deepcopy(solved.solved_study)
    solved.clear_history()
    return solved


def test_nothing_is_stale_before_or_right_after_solving(sample_bdf, session):
    fresh = Session()
    fresh.open_mesh(sample_bdf, "mm")
    _setup(fresh)
    assert fresh.changes_since_solve() == [], "解く前は古いも新しいもない"
    assert session.changes_since_solve() == []


def test_a_conductivity_change_is_named(session):
    session.set_conductivity(2, 0.05)
    assert session.changes_since_solve() == ["σ（PID 2）"]
    # 比べているので、元の値に戻せば最新に戻る
    session.set_conductivity(2, SIGMA[2])
    assert session.changes_since_solve() == []


def test_electrode_changes_are_named(session):
    spec = session.electrode("E1")
    spec.current = 3.0e-3
    spec.shell_pids = []
    session.add_patch("E1", (0.030, 0.020, 0.020), 0.010)
    assert session.changes_since_solve() == ["電極 E1 の総電流・面"]


def test_a_rename_is_not_mistaken_for_removal_and_addition(session):
    session.rename_electrode("E2", "GND")
    assert session.changes_since_solve() == ["電極 GND の名前（E2 → GND）"]


def test_adding_and_removing_electrodes_is_named(session):
    session.add_electrode("E3", role="cathode", current=1.0e-3, shell_pids=[201])
    assert session.changes_since_solve() == ["電極 E3 を追加"]
    session.remove_electrode("E3")
    session.remove_electrode("E2")
    assert session.changes_since_solve() == ["電極 E2 を削除"]


def test_solver_settings_that_change_the_answer_are_named(session):
    session.study.solver.tolerance = 1.0e-6
    session.study.solver.max_iterations = 10
    session.study.output.reference_potential = "cathode_mean"
    assert session.changes_since_solve() == ["収束判定", "最大反復回数", "電位の基準"]


def test_backend_and_preconditioner_do_not_make_the_result_stale(session):
    """バックエンドを変えても答えはビット単位で同じ。前処理は収束判定の範囲でしか変わらない。"""
    session.study.solver.backend = "cpu"
    session.study.solver.preconditioner = "amg"
    session.study.materials[0].name = "皮膚"  # 領域の名前も答えを変えない
    session.study.output.vtu = "elsewhere.vtu"
    assert session.changes_since_solve() == []


def test_undo_back_to_the_solved_conditions_makes_it_fresh_again(session):
    session.checkpoint("領域 2 の設定を変更")
    session.set_conductivity(2, 0.08)
    assert session.changes_since_solve()
    session.undo()
    assert session.changes_since_solve() == []


def test_editing_while_solving_is_caught(sample_bdf):
    """解いている間（別スレッド）の編集も拾う。写しはケースを組み立てた瞬間に取る。"""
    s = Session()
    s.open_mesh(sample_bdf, "mm")
    _setup(s)
    case, study = s.prepare_solve()
    s.set_conductivity(3, 0.5)  # 解いている最中に編集した
    s.set_solution(case.solve(), study)
    assert s.changes_since_solve() == ["σ（PID 3）"]


# --- GUI ---------------------------------------------------------------------


@pytest.fixture
def window(main_window, sample_bdf, qt_app):
    w = main_window
    w.open_mesh(sample_bdf, "mm")
    _setup(w.session)
    w.refresh_tree()
    w.start_solve()
    assert w.solver.wait(60_000)
    qt_app.processEvents()
    return w


def _shown(widget) -> bool:
    return not widget.isHidden()


def test_a_fresh_result_shows_no_warning(window):
    assert window.session.changes_since_solve() == []
    assert not _shown(window.page_result.stale_label)
    assert not _shown(window.stale_status)
    assert window._result_item.text(0) == "結果"


def test_changing_sigma_in_the_panel_flags_the_result(window, qt_app):
    page = window.page_region
    page.bind(2)
    page.sigma_spin.setValue(0.05)

    assert _shown(window.page_result.stale_label)
    assert "σ（PID 2）" in window.page_result.stale_label.text()
    assert "F5" in window.page_result.stale_label.text()
    assert _shown(window.stale_status)
    assert window._result_item.text(0) == "結果（条件が変わっています）"
    assert "σ（PID 2）" in window._result_item.toolTip(0)
    # 結果は消さない（解き直すまで見比べられる）
    assert window.session.solution is not None

    # 解き直せば知らせは消える
    window.start_solve()
    assert window.solver.wait(60_000)
    qt_app.processEvents()
    assert not _shown(window.page_result.stale_label)
    assert not _shown(window.stale_status)
    assert window._result_item.text(0) == "結果"


def test_the_solver_page_also_flags_the_result(window):
    """ソルバ設定のページはツリーを作り直さないので、別に見直している。"""
    page = window.page_solver
    page.reload()
    page.tolerance.setText("1e-6")
    page._commit()
    assert "収束判定" in window.page_result.stale_label.text()
    assert _shown(window.stale_status)


def test_undo_clears_the_warning(window):
    window.page_region.bind(1)
    window.page_region.sigma_spin.setValue(0.2)
    assert _shown(window.stale_status)
    window.undo()
    assert not _shown(window.stale_status)
    assert window._result_item.text(0) == "結果"


def test_exporting_a_stale_result_says_so(window, tmp_path):
    """古い結果の書き出しは止めない（中身は求解時の条件どおりで正しい）が、ログに残す。"""
    window.export_vtu(tmp_path / "fresh.vtu")
    assert "[注意]" not in window.log_view.toPlainText().splitlines()[-1]
    window.page_region.bind(3)
    window.page_region.sigma_spin.setValue(0.3)
    window.export_vtu(tmp_path / "stale.vtu")
    last = window.log_view.toPlainText().splitlines()[-1]
    assert "[注意]" in last and "σ（PID 3）" in last
