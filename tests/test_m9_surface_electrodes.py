"""体表面から選ぶ電極（SPEC §11.1）。

シェル要素（CTRIA3 / CQUAD4）を持たないモデルでも電極を置けるようにするための経路。
人体モデルはソリッド要素だけで配られるのが普通で、それまでは電極を 1 つも
定義できなかった。

物理そのもの（シェルで指定したときとビット単位で同じ荷重になること）は Rust 側の
`crates/gfem-core/tests/electrode_patch.rs` で確かめている。ここでは
**Python から使ったときに一貫しているか**を見る。
"""

from __future__ import annotations

import numpy as np
import pytest

import gfem
from gfem.gui.state import Session

SIGMA = {1: 0.10, 2: 0.04, 3: 0.35}
CURRENT = 2.0e-3


@pytest.fixture
def session(sample_bdf) -> Session:
    s = Session()
    s.open_mesh(sample_bdf, "mm")
    for pid, sigma in SIGMA.items():
        s.set_conductivity(pid, sigma)
    return s


def test_the_outer_surface_lists_the_regions_it_exposes(session):
    """どの PID が体表面に出ているかを面積つきで出せること。

    首の切断面のように「外表面だが体表面ではない」面を避けるために、
    GUI はこの一覧から絞り込み先を選ばせる。
    """
    rows = session.surface_pid_choices()
    assert [r["pid"] for r in rows] == [3, 1, 2]  # 面積の大きい順
    assert rows[0]["area"] > rows[-1]["area"]
    assert sum(r["n_faces"] for r in rows) == 944  # 外表面の三角形の総数
    # 既定はいちばん広い領域（人体モデルなら皮膚にあたる）
    assert session.default_surface_pid() == 3


def test_a_click_selects_a_connected_patch(session):
    """クリック点の周りの面が、体表面伝いに広がって選ばれること。"""
    click = (0.030, 0.020, 0.020)  # 上面の中央 [m]
    small = session.pick_patch(click, 0.005)
    large = session.pick_patch(click, 0.015)
    assert 0 < small["n_faces"] < large["n_faces"]
    assert small["area"] < large["area"]
    # 強調表示用の三角形が面数と対応している（四角形面は 2 枚に割れる）
    assert len(small["triangles"]) >= small["n_faces"]
    # 半径ゼロでもクリックした面は必ず 1 枚選ばれる
    assert session.pick_patch(click, 0.0)["n_faces"] == 1


def test_the_patch_area_shown_in_the_gui_is_what_the_solver_uses(session):
    """GUI に出す面積と、ソルバが実際に使う面積が一致すること。

    ここがずれると「25 cm² のつもりで置いた電極が実は違う面積だった」ことになり、
    電流密度の評価が静かに狂う。
    """
    top = session.pick_patch((0.030, 0.020, 0.020), 0.012)
    bottom = session.pick_patch((0.030, 0.020, 0.000), 0.012)
    session.add_electrode("E1", role="anode", current=CURRENT)
    session.add_electrode("E2", role="cathode", current=CURRENT)
    session.add_patch("E1", (0.030, 0.020, 0.020), 0.012)
    session.add_patch("E2", (0.030, 0.020, 0.000), 0.012)

    info = {e["name"]: e for e in session.electrode_info()}
    assert info["E1"]["area"] == pytest.approx(top["area"], rel=0, abs=0)
    assert info["E1"]["n_faces"] == top["n_faces"]
    assert info["E2"]["area"] == pytest.approx(bottom["area"], rel=0, abs=0)
    assert info["E1"]["current_density"] == pytest.approx(CURRENT / top["area"], rel=1e-12)


def test_a_model_without_shells_can_be_solved(session, tmp_path, make_sample_mesh):
    """本題。シェル要素を持たないモデルでも解けること。"""
    solid_only = tmp_path / "solid_only.bdf"
    make_sample_mesh.main(["-o", str(solid_only), "--electrodes", "none"])
    mesh = gfem.read_nastran(solid_only, length_unit="mm")
    assert mesh.n_shell_elements == 0, "シェルの無いモデルを作れていない"

    case = gfem.Case(mesh)
    case.set_conductivity(SIGMA)
    case.add_electrode(
        "E1", role="anode", current=CURRENT,
        patches=[{"center": [0.030, 0.020, 0.020], "radius": 0.012}],
    )
    case.add_electrode(
        "E2", role="cathode", current=CURRENT,
        patches=[{"center": [0.030, 0.020, 0.000], "radius": 0.012}],
    )
    assert case.validate() == []
    result = case.solve()
    assert result.iterations > 0
    assert result.impedance > 0
    # 保存則は毎回自動チェックされる（SPEC §5.4）
    assert not [w for w in result.warnings if "保存" in w]


def test_patches_survive_a_case_file_round_trip(session, tmp_path):
    """ケースファイルに残るのは面の一覧ではなく中心と半径（人が読める形）。"""
    session.add_electrode("E1", role="anode", current=CURRENT)
    session.add_patch("E1", (0.030, 0.020, 0.020), 0.012, pid=3)
    session.add_electrode("E2", role="cathode", current=CURRENT)
    session.add_patch("E2", (0.030, 0.020, 0.000), 0.012)

    path = tmp_path / "case.yaml"
    session.save_case(path)
    text = path.read_text(encoding="utf-8")
    assert "patches" in text and "radius" in text

    reopened = Session()
    reopened.open_case(path)
    patch = reopened.study.electrodes[0].patches[0]
    assert patch.center == pytest.approx((0.030, 0.020, 0.020))
    assert patch.radius == pytest.approx(0.012)
    assert patch.pid == 3
    assert reopened.study.electrodes[1].patches[0].pid is None
    # 同じ面が選ばれる
    assert reopened.electrode_info()[0]["area"] == session.electrode_info()[0]["area"]


def test_a_bad_patch_says_what_to_fix(session):
    """存在しない領域に絞り込んだら、理由と直し方が分かること（設計原則 6）。"""
    session.add_electrode("E1", role="anode", current=CURRENT)
    session.add_patch("E1", (0.030, 0.020, 0.020), 0.012, pid=999)
    session.add_electrode("E2", role="cathode", current=CURRENT, shell_pids=[201])
    with pytest.raises(gfem.GfemError) as excinfo:
        session.validate()
    message = str(excinfo.value)
    assert "1 枚も選べませんでした" in message
    assert "PID=999" in message
    assert "体表面に露出しているか" in message


def test_an_electrode_without_any_face_mentions_patches(session):
    """面が無いときの案内に、クリックで選ぶ道も出ること。"""
    session.add_electrode("E1", role="anode", current=CURRENT)
    session.add_electrode("E2", role="cathode", current=CURRENT, shell_pids=[201])
    with pytest.raises(gfem.GfemError) as excinfo:
        session.validate()
    assert "パッチ" in str(excinfo.value)


# --- GUI ---------------------------------------------------------------------


@pytest.fixture
def window(main_window, sample_bdf):
    w = main_window
    w.open_mesh(sample_bdf, "mm")
    for pid, sigma in SIGMA.items():
        w.session.set_conductivity(pid, sigma)
    w.session.add_electrode("E1", role="anode", current=CURRENT)
    w.session.add_electrode("E2", role="cathode", current=CURRENT)
    w.refresh_tree()
    return w


def test_an_electrode_painted_only_by_brush_shows_its_area(window):
    """パッチを置かず筆で塗っただけの電極でも、面数と面積を出す。

    以前は「面が選択されていません」と表示していた（パッチ・シェルの有無しか見ていなかった）。
    """
    window.session.paint_faces("E1", (0.010, 0.010, 0.020), 0.003, None, erase=False)
    spec = window.session.electrode("E1")
    assert spec.added_faces and not spec.patches

    page = window.page_electrode
    page.bind("E1")
    text = page.summary.text()
    assert "面が選択されていません" not in text
    assert f"{len(spec.added_faces)} 面" in text


def test_clicking_the_view_adds_a_patch(window):
    """3D ビューのクリック（＝コールバック）で電極面が増えること。"""
    page = window.page_electrode
    page.bind("E1")
    page.diameter_spin.setValue(24.0)  # mm（直径）
    page.patch_pid.setCurrentIndex(0)  # 制限しない

    page._on_picked(np.array([0.030, 0.020, 0.020]))

    spec = window.session.electrode("E1")
    assert len(spec.patches) == 1
    assert spec.patches[0].radius == pytest.approx(0.012)
    # 一覧に面数と面積が出る
    assert page.patch_list.count() == 1
    assert "cm²" in page.patch_list.item(0).text()
    # 電極の要約も更新される
    assert "面" in page.summary.text()
    # 3D ビューの強調表示に入っている
    assert window.session.electrode_polydata("E1").n_cells > 0


def test_the_pid_filter_offers_the_exposed_regions(window):
    page = window.page_electrode
    page.bind("E1")
    labels = [page.patch_pid.itemText(i) for i in range(page.patch_pid.count())]
    assert labels[0] == "制限しない"
    assert any("PID 3" in text for text in labels)
    # 既定はいちばん広い領域（人体モデルなら皮膚）
    assert page.patch_pid.currentData() == 3


def test_removing_a_patch(window):
    page = window.page_electrode
    page.bind("E1")
    page._on_picked(np.array([0.030, 0.020, 0.020]))
    assert page.patch_list.count() == 1
    page.patch_list.setCurrentRow(0)
    page._remove_patch()
    assert page.patch_list.count() == 0
    assert window.session.electrode("E1").patches == []


def test_picking_mode_can_be_turned_on_and_off(window):
    page = window.page_electrode
    page.bind("E1")
    page.pick_button.setChecked(True)
    assert window._pick_callback is not None
    page.pick_button.setChecked(False)
    assert window._pick_callback is None


def test_faces_can_be_painted_and_erased_one_by_one(session):
    """面を 1 枚ずつ足す・外すができること（SPEC §11.1）。

    半径を最小にすればちょうど 1 枚選ばれるので、**細かい調整も同じ道具**でできる。
    """
    session.add_electrode("E1", role="anode", current=CURRENT)
    click = (0.030, 0.020, 0.020)
    session.add_patch("E1", click, 0.008)
    base = session.electrode_faces("E1")["n_faces"]

    # 1 枚だけ足す
    added = session.paint_faces("E1", (0.010, 0.010, 0.020), 0.0, None, erase=False)
    assert added == 1
    assert session.electrode_faces("E1")["n_faces"] == base + 1
    assert len(session.electrode("E1").added_faces) == 1

    # 同じ 1 枚を外すと、足した記録は消えて外した記録が残る（両方に載らない）
    session.paint_faces("E1", (0.010, 0.010, 0.020), 0.0, None, erase=True)
    spec = session.electrode("E1")
    assert spec.added_faces == []
    assert len(spec.removed_faces) == 1
    assert session.electrode_faces("E1")["n_faces"] == base

    # パッチの中の 1 枚を外すと減る
    session.paint_faces("E1", click, 0.0, None, erase=True)
    assert session.electrode_faces("E1")["n_faces"] == base - 1


def test_hand_edits_survive_a_case_file_round_trip(session, tmp_path):
    """手直しはケースファイルに**差分として**残ること。

    解決後の面を全部書くと 1 電極で 6〜12 KB になり、YAML が読めなくなる。
    """
    session.add_electrode("E1", role="anode", current=CURRENT)
    session.add_patch("E1", (0.030, 0.020, 0.020), 0.008)
    session.paint_faces("E1", (0.010, 0.010, 0.020), 0.002, None, erase=False)
    session.paint_faces("E1", (0.030, 0.020, 0.020), 0.0, None, erase=True)
    session.add_electrode("E2", role="cathode", current=CURRENT, shell_pids=[201])
    before = session.electrode_faces("E1")

    path = tmp_path / "case.yaml"
    session.save_case(path)
    text = path.read_text(encoding="utf-8")
    assert "added" in text and "removed" in text

    reopened = Session()
    reopened.open_case(path)
    after = reopened.electrode_faces("E1")
    assert after["n_faces"] == before["n_faces"]
    assert after["area"] == before["area"]
    # 面番号は 1 始まりで書かれている（エラーメッセージと同じ数え方）
    assert min(f for _e, f in reopened.electrode("E1").added_faces) >= 1


def test_what_the_panel_shows_is_what_the_solver_uses(session):
    """手直しを入れても、表示とソルバの面積が一致すること。"""
    session.add_electrode("E1", role="anode", current=CURRENT)
    session.add_electrode("E2", role="cathode", current=CURRENT, shell_pids=[201])
    session.add_patch("E1", (0.030, 0.020, 0.020), 0.008)
    session.paint_faces("E1", (0.010, 0.010, 0.020), 0.003, None, erase=False)
    session.paint_faces("E1", (0.030, 0.020, 0.020), 0.0, None, erase=True)

    shown = session.electrode_faces("E1")
    used = session.electrode_info("E1")[0]
    assert shown["n_faces"] == used["n_faces"]
    assert shown["area"] == used["area"]


def test_clearing_hand_edits_goes_back_to_the_patch(session):
    session.add_electrode("E1", role="anode", current=CURRENT)
    session.add_patch("E1", (0.030, 0.020, 0.020), 0.008)
    base = session.electrode_faces("E1")["n_faces"]
    session.paint_faces("E1", (0.010, 0.010, 0.020), 0.003, None, erase=False)
    assert session.electrode_faces("E1")["n_faces"] > base
    session.clear_hand_edits("E1")
    assert session.electrode_faces("E1")["n_faces"] == base


def test_a_click_off_the_model_is_ignored(session):
    """背景をクリックしたとき（ピッカーは遠くの点を返す）に置かないこと。

    そのまま置くと「いちばん近い面」が選ばれて、**まるで見当違いの場所に
    電極ができる**。3D ビューの外れをクリックするのは日常的に起きる。
    """
    assert session.is_on_the_model((0.030, 0.020, 0.020))       # モデルの上
    assert session.is_on_the_model((0.030, 0.020, 0.0205))      # すぐ外側は許す
    assert not session.is_on_the_model((0.030, 0.020, 5.0))     # 遠い背景
    assert not session.is_on_the_model((float("nan"),) * 3)     # ピック失敗


def test_a_failed_pick_is_refused_by_the_core(session):
    """NaN が来たら、黙って先頭の面を選ばずに止まること（設計原則 6）。"""
    with pytest.raises(gfem.GfemError) as excinfo:
        session.pick_patch((float("nan"), 0.0, 0.0), 0.01)
    assert "数値になっていません" in str(excinfo.value)


def test_clicking_the_background_adds_nothing(window):
    page = window.page_electrode
    page.bind("E1")
    page._on_picked(np.array([0.030, 0.020, 5.0]))  # はるか上空
    assert window.session.electrode("E1").patches == []
    assert "モデルの外" in window.log_view.toPlainText()


def test_a_real_mouse_click_selects_faces(window, qt_app):
    """本物のマウスイベントで面が選べること。

    コールバックを直接呼ぶだけでは配線（PyVista のピッキング）を確かめられない。
    実際、描き直しのたびに掛け直す実装で**二重に有効化して例外**になっていたのを
    ここで見つけたので、回帰として残す。
    """
    page = window.page_electrode
    page.bind("E1")
    page.diameter_spin.setValue(16.0)
    window.show()
    qt_app.processEvents()
    page.pick_button.setChecked(True)
    qt_app.processEvents()

    plotter = window.plotter
    size = plotter.render_window.GetSize()
    if min(size) <= 1:
        pytest.skip("3D ビューを描画できない環境です")
    iren = plotter.iren.interactor
    iren.SetEventPosition(size[0] // 2, size[1] // 2)
    iren.LeftButtonPressEvent()
    iren.LeftButtonReleaseEvent()
    qt_app.processEvents()

    assert len(window.session.electrode("E1").patches) == 1, "クリックで選べていない"
    # 掛け直しは何度呼んでも例外にならないこと
    window._arm_picking()
    window._arm_picking()
    page.pick_button.setChecked(False)
    window.hide()


def _drag(window, qt_app, path, shift=False, alt=False):
    """3D ビューを本物のマウスイベントでなぞる。"""
    iren = window.plotter.iren.interactor
    for i, (x, y) in enumerate(path):
        iren.SetEventPosition(int(x), int(y))
        iren.SetShiftKey(1 if shift else 0)
        iren.SetAltKey(1 if alt else 0)
        if i == 0:
            iren.LeftButtonPressEvent()
        else:
            iren.MouseMoveEvent()
    iren.LeftButtonReleaseEvent()
    qt_app.processEvents()


@pytest.fixture
def painting_window(window, qt_app):
    """クリック選択モードに入り、3D ビューが描かれた状態のウィンドウ。"""
    page = window.page_electrode
    page.bind("E1")
    page.shape_combo.setCurrentIndex(page.shape_combo.findData("circle"))
    page.diameter_spin.setValue(16.0)
    page.brush_spin.setValue(8.0)
    # 他のテストが断面表示のまま終えていることがある。
    # 体表面をクリックしたいので「全体」に戻してから始める
    window.page_result.mode.setCurrentIndex(window.page_result.mode.findData("volume"))
    window.refresh_view()
    window.plotter.reset_camera()
    window.show()
    qt_app.processEvents()
    if min(window.plotter.render_window.GetSize()) <= 1:
        pytest.skip("3D ビューを描画できない環境です")
    page.pick_button.setChecked(True)
    qt_app.processEvents()
    yield window
    page.pick_button.setChecked(False)
    window.hide()


def test_dragging_with_shift_paints_faces(painting_window, qt_app):
    """Shift＋ドラッグで面が増えること。"""
    window = painting_window
    w, h = window.plotter.render_window.GetSize()
    # まずクリックでパッチを置き、それから塗り足す
    _drag(window, qt_app, [(w // 2, h // 2)])
    before = window.session.electrode_faces("E1")["n_faces"]
    assert before > 1, "クリックでパッチが置けていない"

    _drag(
        window, qt_app,
        [(w // 2 + dx, h // 2) for dx in range(0, 40, 8)],
        shift=True,
    )
    spec = window.session.electrode("E1")
    assert spec.added_faces, "Shift+ドラッグで面が足されていない"
    assert window.session.electrode_faces("E1")["n_faces"] > before
    # 塗り終わりに面積の表示も追いつく
    assert "cm²" in window.page_electrode.summary.text()
    assert "手で足した面" in window.page_electrode.hand_edits.text()


def test_dragging_with_alt_erases_faces(painting_window, qt_app):
    window = painting_window
    w, h = window.plotter.render_window.GetSize()
    _drag(window, qt_app, [(w // 2, h // 2)])
    before = window.session.electrode_faces("E1")["n_faces"]

    window.page_electrode.brush_spin.setValue(0.5)  # 細い筆で縁だけ削る
    _drag(window, qt_app, [(w // 2, h // 2), (w // 2 + 4, h // 2)], alt=True)
    assert window.session.electrode("E1").removed_faces, "Alt+ドラッグで外れていない"
    assert window.session.electrode_faces("E1")["n_faces"] < before


def test_a_plain_drag_still_rotates(painting_window, qt_app):
    """ふつうのドラッグは回転のまま（塗りに奪われないこと）。

    奪うと、頭の反対側に電極を置くたびにモードを切り替えることになる。
    """
    window = painting_window
    w, h = window.plotter.render_window.GetSize()
    _drag(window, qt_app, [(w // 2, h // 2)])
    faces = window.session.electrode_faces("E1")["n_faces"]
    camera = tuple(window.plotter.camera.position)

    _drag(window, qt_app, [(w // 2 + d, h // 2 + d // 2) for d in range(0, 100, 10)])
    assert tuple(window.plotter.camera.position) != camera, "回転していない"
    assert window.session.electrode_faces("E1")["n_faces"] == faces, "回転で面が変わった"


def test_one_stroke_is_one_undo_step(painting_window, qt_app):
    """1 ストロークが 1 手であること（1 なぞりごとに戻るのは煩わしい）。"""
    window = painting_window
    w, h = window.plotter.render_window.GetSize()
    _drag(window, qt_app, [(w // 2, h // 2)])
    after_patch = window.session.electrode_faces("E1")["n_faces"]

    _drag(window, qt_app, [(w // 2 + dx, h // 2) for dx in range(0, 40, 8)], shift=True)
    assert window.session.electrode_faces("E1")["n_faces"] > after_patch

    window.undo()
    assert window.session.electrode("E1").added_faces == []
    assert window.session.electrode_faces("E1")["n_faces"] == after_patch


def test_hovering_previews_what_will_be_painted(painting_window, qt_app):
    """塗る前に、どこが塗られるかが見えること。"""
    window = painting_window
    w, h = window.plotter.render_window.GetSize()
    iren = window.plotter.iren.interactor
    iren.SetEventPosition(w // 2, h // 2)
    iren.MouseMoveEvent()
    qt_app.processEvents()
    assert window.plotter.actors.get(window.PREVIEW_ACTOR) is not None
    # 面はまだ増えない（予告なので）
    with pytest.raises(gfem.GfemError):
        window.session.electrode_faces("E1")


def test_adding_a_patch_can_be_undone(window):
    page = window.page_electrode
    page.bind("E1")
    page._on_picked(np.array([0.030, 0.020, 0.020]))
    assert len(window.session.electrode("E1").patches) == 1
    window.undo()
    assert window.session.electrode("E1").patches == []
