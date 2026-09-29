"""複数の領域に σ をまとめて入れる（SPEC §6.2 の手順 2）。

人体モデルは領域が数十あり（頭部モデルで 62）、同じ組織が複数の PID に分かれていることも
多い。1 つずつ入れるのは手間なので、ツリーで Ctrl / Shift + クリックして複数選び、
σ をまとめて入れられるようにした。
"""

from __future__ import annotations

import pytest

from gfem.i18n import tr
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtTest import QTest

from gfem.gui.widgets import MixedValueSpinBox

SIGMA = {1: 0.10, 2: 0.04, 3: 0.35}


@pytest.fixture
def window(main_window, sample_bdf):
    w = main_window
    w.open_mesh(sample_bdf, "mm")
    for pid, sigma in SIGMA.items():
        w.session.set_conductivity(pid, sigma)
    w.session.add_electrode("E1", role="anode", current=2e-3, shell_pids=[101])
    w.session.add_electrode("E2", role="cathode", current=2e-3, shell_pids=[201])
    w.refresh_tree()
    return w


def _item(window, kind, key=None):
    from gfem.gui.main_window import ROLE_KEY, ROLE_KIND

    it = window.tree.findItems("", Qt.MatchFlag.MatchContains | Qt.MatchFlag.MatchRecursive)
    for item in it:
        if item.data(0, ROLE_KIND) == kind and (key is None or item.data(0, ROLE_KEY) == key):
            return item
    raise LookupError((kind, key))


def _select(window, *items):
    """Ctrl + クリックで選んだのと同じ状態にする（最後の項目が current）。"""
    window.tree.clearSelection()
    for item in items:
        window.tree.setCurrentItem(item, 0, QItemSelectionModel.SelectionFlag.Select)


def test_several_regions_can_be_selected_and_share_one_sigma(window):
    _select(window, _item(window, "region", 1), _item(window, "region", 3))
    page = window.page_region
    assert window.pages.currentWidget() is page
    assert page.pids == [1, 3]
    assert "2 領域を選択中（1, 3）" in page.pid_label.text()
    # σ がばらばらなら「値がそろっていません」
    assert page.sigma_spin.is_mixed
    assert page.sigma_spin.text() == tr(MixedValueSpinBox.MIXED_TEXT)
    # 名前は複数選択の間は変えられない（まとめて同じ名前にすると見分けられない）
    assert not page.name_edit.isEnabled()

    page.sigma_spin.setValue(0.2)
    assert window.session.conductivity_of(1) == pytest.approx(0.2)
    assert window.session.conductivity_of(3) == pytest.approx(0.2)
    assert window.session.conductivity_of(2) == pytest.approx(SIGMA[2]), "選んでいない領域は触らない"
    # 作り直したツリーでも選択が残っている（続けて直せる）
    assert page.pids == [1, 3]
    assert {i.data(0, Qt.ItemDataRole.UserRole + 1) for i in window.tree.selectedItems()} == {1, 3}


def test_typing_into_a_mixed_field_sets_every_selected_region(window):
    _select(window, _item(window, "region", 1), _item(window, "region", 2))
    page = window.page_region
    spin = page.sigma_spin
    spin.lineEdit().selectAll()
    QTest.keyClicks(spin, "0.25")
    QTest.keyClick(spin, Qt.Key.Key_Return)
    assert spin.value() == pytest.approx(0.25), "打っている途中で欄が書き直されていない"
    assert window.session.conductivity_of(1) == pytest.approx(0.25)
    assert window.session.conductivity_of(2) == pytest.approx(0.25)


def test_the_down_arrow_on_a_mixed_field_changes_nothing(window):
    """「そろっていない」から下矢印では何も入れない（番兵の値が漏れて σ < 0 にならない）。"""
    _select(window, _item(window, "region", 1), _item(window, "region", 2))
    window.page_region.sigma_spin.stepBy(-1)
    assert window.session.conductivity_of(1) == pytest.approx(SIGMA[1])
    assert window.session.conductivity_of(2) == pytest.approx(SIGMA[2])


def test_matching_values_are_shown_as_is(window):
    window.session.set_conductivity(3, SIGMA[1])
    window.refresh_tree()
    _select(window, _item(window, "region", 1), _item(window, "region", 3))
    spin = window.page_region.sigma_spin
    assert not spin.is_mixed
    assert spin.value() == pytest.approx(SIGMA[1])


def test_a_tissue_hint_sets_sigma_and_name_for_all(window):
    from gfem.gui.state import TISSUE_HINTS

    name, value = next(iter(TISSUE_HINTS.items()))
    _select(window, _item(window, "region", 2), _item(window, "region", 3))
    window.page_region._apply_hint(name, value)
    for pid in (2, 3):
        assert window.session.conductivity_of(pid) == pytest.approx(value)
        assert window.session.region_name(pid) == name
    assert window.session.region_name(1) == ""


def test_setting_several_regions_is_one_undo_step(window):
    _select(window, _item(window, "region", 1), _item(window, "region", 2), _item(window, "region", 3))
    window.page_region.sigma_spin.setValue(0.5)
    window.undo()
    for pid, sigma in SIGMA.items():
        assert window.session.conductivity_of(pid) == pytest.approx(sigma)


def test_regions_and_other_items_are_not_mixed(window):
    """領域と電極を一緒に選んでもまとめてできる操作は無い。クリックした方の種類だけ残す。"""
    _select(window, _item(window, "region", 1), _item(window, "electrode", "E1"))
    kinds = {i.data(0, Qt.ItemDataRole.UserRole) for i in window.tree.selectedItems()}
    assert kinds == {"electrode"}
    assert window.pages.currentWidget() is window.page_electrode

    _select(window, _item(window, "electrode", "E1"), _item(window, "region", 1), _item(window, "region", 2))
    kinds = {i.data(0, Qt.ItemDataRole.UserRole) for i in window.tree.selectedItems()}
    assert kinds == {"region"}
    assert window.page_region.pids == [1, 2]


def test_every_selected_region_is_highlighted(window, monkeypatch):
    """選んだ領域をすべて強調表示する（どれを選んでいるか目で確かめられる）。"""
    seen = []
    original = window.refresh_view
    monkeypatch.setattr(
        window, "refresh_view", lambda **kw: (seen.append(kw.get("highlight_pid")), original(**kw))
    )
    _select(window, _item(window, "region", 3), _item(window, "region", 1))
    assert seen[-1] == [1, 3]
    before = len(window.plotter.actors)
    window.refresh_view(highlight_pid=[1])
    assert len(window.plotter.actors) == before - 1, "領域 1 つぶんの強調が減る"


def test_the_stale_result_names_every_changed_region(window, qt_app):
    window.start_solve()
    assert window.solver.wait(60_000)
    qt_app.processEvents()
    _select(window, _item(window, "region", 1), _item(window, "region", 2))
    window.page_region.sigma_spin.setValue(0.3)
    assert "σ（PID 1, 2）" in window.page_result.stale_label.text()


def test_a_single_region_still_edits_its_name(window):
    _select(window, _item(window, "region", 2))
    page = window.page_region
    assert page.pid == 2 and page.name_edit.isEnabled()
    page.name_edit.setText("脂肪")
    page.name_edit.editingFinished.emit()
    assert window.session.region_name(2) == "脂肪"
