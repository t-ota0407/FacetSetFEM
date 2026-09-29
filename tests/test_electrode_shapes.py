"""矩形（角丸可）の電極パッチ（SPEC §11.1）。

実験で使う電極は 2 cm × 2 cm の正方形のような矩形が多い。円のパッチしか無いと
形も面積も合わせられないので、形・寸法・角の半径・向きを選べるようにした。

面の選び方そのもの（面積・向き・角丸・回り込まないこと）は Rust 側の
`crates/gfem-core/tests/electrode_shape.rs` で確かめている。ここでは Python と GUI から
使ったときに一貫しているか（ケースファイルに残るか、画面に対してまっすぐ置けるか、
指定面積とのずれを見せるか）を見る。
"""

from __future__ import annotations

import numpy as np
import pytest

import gfem
from gfem.gui.pages import describe_selection
from gfem.gui.state import Session

SIGMA = {1: 0.10, 2: 0.04, 3: 0.35}
CURRENT = 2.0e-3
#: サンプルの上面（60 mm × 40 mm、z = 20 mm）の中央 [m]
TOP = (0.030, 0.020, 0.020)
BOTTOM = (0.030, 0.020, 0.000)
SQUARE = {"shape": "rect", "width": 0.02, "height": 0.02, "axis": [1.0, 0.0, 0.0]}


@pytest.fixture
def session(sample_bdf) -> Session:
    s = Session()
    s.open_mesh(sample_bdf, "mm")
    for pid, sigma in SIGMA.items():
        s.set_conductivity(pid, sigma)
    return s


def test_a_square_is_picked_with_about_its_nominal_area(session):
    """2 cm 角を置くと、指定面積（4 cm²）に近い面が選ばれ、輪郭と法線も返ること。"""
    info = session.pick_patch(TOP, **SQUARE)
    assert info["shape"] == "rect"
    assert info["nominal_area"] == pytest.approx(4.0e-4, rel=1e-12)
    # サンプルは粗い（面 1 枚が約 9 mm²）ので縁のぶん数 % ずれる
    assert info["area"] == pytest.approx(4.0e-4, rel=0.15)
    assert info["outline"].shape == (4, 3)  # 尖った角の矩形は 4 点
    assert np.allclose(info["normal"], [0.0, 0.0, 1.0])
    # 輪郭は指定どおりの 2 cm 角
    xy = info["outline"][:, :2]
    assert np.ptp(xy, axis=0) == pytest.approx([0.02, 0.02])


def test_the_circle_keeps_working_as_before(session):
    """円は従来どおり半径だけで置ける（既存のケースファイル・スクリプトを壊さない）。"""
    info = session.pick_patch(TOP, 0.012)
    assert info["shape"] == "circle"
    assert info["nominal_area"] == pytest.approx(np.pi * 0.012**2)
    assert info["outline"].shape[1] == 3


def test_rectangles_survive_a_case_file_round_trip(session, tmp_path):
    """ケースファイルには形・寸法・角の半径・向きが人の読める形で残ること。"""
    session.add_electrode("E1", role="anode", current=CURRENT)
    session.add_patch(
        "E1", TOP, pid=1, shape="rect", width=0.03, height=0.02, corner_radius=0.004,
        axis=[0.0, 1.0, 0.0], angle=30.0,
    )
    session.add_electrode("E2", role="cathode", current=CURRENT)
    session.add_patch("E2", BOTTOM, 0.012)

    path = tmp_path / "case.yaml"
    session.save_case(path)
    text = path.read_text(encoding="utf-8")
    assert "shape: rect" in text and "corner_radius" in text and "axis" in text
    # 円のパッチに矩形のキーは出ない（読んで意味が追えるように）
    circle_part = text.split("name: E2")[1]
    assert "width" not in circle_part and "radius: 0.012" in circle_part

    reopened = Session()
    reopened.open_case(path)
    patch = reopened.study.electrodes[0].patches[0]
    assert patch.is_rect
    assert (patch.width, patch.height, patch.corner_radius) == pytest.approx((0.03, 0.02, 0.004))
    assert patch.axis == pytest.approx((0.0, 1.0, 0.0))
    assert patch.angle == pytest.approx(30.0)
    assert patch.pid == 1
    assert not reopened.study.electrodes[1].patches[0].is_rect
    # 同じ面が選ばれる
    assert reopened.electrode_info()[0]["area"] == session.electrode_info()[0]["area"]


def _write_case(tmp_path, sample_bdf, patch: str):
    path = tmp_path / "case.yaml"
    path.write_text(
        f"""\
mesh:
  file: {sample_bdf.as_posix()}
  length_unit: mm
materials:
  - {{pid: 1, sigma: 0.1}}
  - {{pid: 2, sigma: 0.04}}
  - {{pid: 3, sigma: 0.35}}
electrodes:
  - name: E1
    role: anode
    current: 2.0e-3
    faces:
      patches:
        - {patch}
  - name: E2
    role: cathode
    current: 2.0e-3
    faces:
      patches:
        - {{center: [0.03, 0.02, 0.0], radius: 0.012}}
""",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("patch", "needles"),
    [
        (
            "{center: [0.03, 0.02, 0.02], shape: rect, height: 0.02, axis: [1, 0, 0]}",
            ["width がありません", "electrodes[0] の faces.patches[0]"],
        ),
        (
            "{center: [0.03, 0.02, 0.02], shape: rect, width: 0.02, height: 0.01,"
            " corner_radius: 0.008, axis: [1, 0, 0]}",
            ["角の半径", "短い辺の半分", "対処"],
        ),
        (
            "{center: [0.03, 0.02, 0.02], shape: square, width: 0.02}",
            ["\"square\"", "rect"],
        ),
        (
            "{center: [0.03, 0.02, 0.02], radius: 0.01, width: 0.02}",
            ["width は使えません"],
        ),
        (
            "{center: [0.03, 0.02, 0.02], shape: rect, width: 0.02, height: 0.02}",
            ["axis がありません"],
        ),
    ],
)
def test_a_bad_patch_in_a_case_file_says_where_and_how(tmp_path, sample_bdf, patch, needles):
    """ケースファイルのパッチの誤りは、場所と直し方つきで止まること（設計原則 6）。

    形の解釈は Rust 側だけにあり、Python は場所を添えるだけ。
    """
    path = _write_case(tmp_path, sample_bdf, patch)
    with pytest.raises(gfem.GfemError) as info:
        gfem.Study.load(path)
    message = str(info.value)
    assert "場所:" in message
    for needle in needles:
        assert needle in message, message


def test_a_mistyped_key_is_refused(session):
    """キーの打ち間違い（widht）を黙って無視しないこと。黙ると既定の円になる。"""
    with pytest.raises(gfem.GfemError, match="使えないキー"):
        session.pick_patch(TOP, shape="rect", widht=0.02, height=0.02, axis=[1, 0, 0])


def test_a_model_with_rectangle_electrodes_can_be_solved(session):
    """矩形の電極で解けて、電極の面積はクリックで見えたものと同じであること。"""
    session.add_electrode("E1", role="anode", current=CURRENT)
    session.add_electrode("E2", role="cathode", current=CURRENT)
    session.add_patch("E1", TOP, **SQUARE, corner_radius=0.003)
    session.add_patch("E2", BOTTOM, **SQUARE)
    picked = session.pick_patch(TOP, **SQUARE, corner_radius=0.003)
    info = {e["name"]: e for e in session.electrode_info()}
    assert info["E1"]["area"] == picked["area"]
    result = session.solve()
    assert result.impedance > 0
    assert not [w for w in result.warnings if "保存" in w]


def test_a_mismatch_with_the_nominal_area_is_flagged(session):
    """電極に対してメッシュが粗すぎると警告する。面積は実面積で計算されている。"""
    fine = session.pick_patch(TOP, **SQUARE)
    text, warn = describe_selection(fine)
    assert "指定 4.00 cm²" in text and not warn
    # 3 mm 角は面 1 枚（約 9 mm²）とほぼ同じ大きさ。指定 9 mm² から大きくずれる
    tiny = session.pick_patch(TOP, shape="rect", width=0.003, height=0.003, axis=[1, 0, 0])
    text, warn = describe_selection(tiny)
    assert warn, text
    # 半径 0 の円（筆の 1 枚選び）は比べない
    assert describe_selection(session.pick_patch(TOP, 0.0))[1] is False


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
    page = w.page_electrode
    page.bind("E1")
    page.patch_pid.setCurrentIndex(0)  # 制限しない
    yield w
    # MainWindow はテスト間で共有している。形と視点を既定（円・斜め上から）に戻す
    page.shape_combo.setCurrentIndex(page.shape_combo.findData("circle"))
    w.plotter.view_isometric()


def _choose_rect(page, width=20.0, height=20.0, corner=0.0, angle=0.0):
    page.shape_combo.setCurrentIndex(page.shape_combo.findData("rect"))
    page.width_spin.setValue(width)
    page.height_spin.setValue(height)
    page.corner_spin.setValue(corner)
    page.angle_spin.setValue(angle)


def test_the_panel_shows_only_the_fields_of_the_chosen_shape(window):
    page = window.page_electrode
    page.shape_combo.setCurrentIndex(page.shape_combo.findData("circle"))
    assert page.shape_form.isRowVisible(page.diameter_spin)
    assert not page.shape_form.isRowVisible(page.size_row)
    _choose_rect(page)
    assert not page.shape_form.isRowVisible(page.diameter_spin)
    assert page.shape_form.isRowVisible(page.size_row)
    assert page.shape_form.isRowVisible(page.corner_spin)
    assert page.shape_form.isRowVisible(page.angle_spin)
    # 指定面積が出る（電極の仕様書と照らし合わせられるように）
    assert "400.0 mm²" in page.nominal_label.text()
    # 角の半径は短い辺の半分まで
    _choose_rect(page, width=20.0, height=10.0)
    page.corner_spin.setValue(9.0)
    assert page.corner_spin.value() == pytest.approx(5.0)


def test_a_click_places_a_rectangle_square_to_the_screen(window):
    """「画面に対してまっすぐ」: 真上から見て置くと、幅の辺が画面の横（x）にそろう。"""
    page = window.page_electrode
    window.plotter.view_xy()  # 真上から、画面の上が +y
    _choose_rect(page, width=30.0, height=10.0)
    page._on_picked(np.array(TOP))

    patch = window.session.electrode("E1").patches[0]
    assert patch.is_rect
    assert patch.axis == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
    assert (patch.width, patch.height) == pytest.approx((0.03, 0.01))
    # 選ばれた面は x に長い
    faces = window.session.electrode_polydata("E1")
    extent = np.ptp(faces.points[np.unique(faces.faces.reshape(-1, 4)[:, 1:])], axis=0)
    assert extent[0] > 2 * extent[1]
    # 一覧に形と指定面積が出る
    text = page.patch_list.item(0).text()
    assert "矩形 30×10 mm" in text and "指定 3.00 cm²" in text


def test_a_patch_can_be_reshaped_from_the_list(window):
    """一覧で選ぶと形が読み込まれ、直して「形を反映」できる。向きの軸は保たれる。"""
    page = window.page_electrode
    window.plotter.view_xy()
    _choose_rect(page, width=30.0, height=10.0)
    page._on_picked(np.array(TOP))
    before = window.session.electrode("E1").patches[0]

    # 別の形を入れてから一覧で選ぶと、そのパッチの形が読み戻される
    page.shape_combo.setCurrentIndex(page.shape_combo.findData("circle"))
    page.patch_list.setCurrentRow(0)
    assert page.shape_combo.currentData() == "rect"
    assert page.width_spin.value() == pytest.approx(30.0)
    assert page.reshape_patch_button.isEnabled()

    # 視点を変えても、反映したときの向きは置いたときの軸から測る
    window.plotter.view_yz()
    page.angle_spin.setValue(90.0)
    page.corner_spin.setValue(2.0)
    page._reshape_patch()
    after = window.session.electrode("E1").patches[0]
    assert after.center == before.center
    assert after.axis == pytest.approx(before.axis)
    assert after.angle == pytest.approx(90.0)
    assert after.corner_radius == pytest.approx(0.002)
    assert page.patch_list.currentRow() == 0
    # 1 手で戻せる
    window.undo()
    assert window.session.electrode("E1").patches[0].angle == pytest.approx(0.0)


def test_hovering_shows_the_outline_of_the_electrode(window, qt_app):
    """置く前に、電極の形の輪郭が見えること（面だけでは本来の形が読めない）。"""
    page = window.page_electrode
    window.page_result.mode.setCurrentIndex(window.page_result.mode.findData("volume"))
    window.refresh_view()
    window.plotter.view_xy()
    window.show()
    qt_app.processEvents()
    try:
        w, h = window.plotter.render_window.GetSize()
        if min(w, h) <= 1:
            pytest.skip("3D ビューを描画できない環境です")
        _choose_rect(page, corner=3.0)
        page.pick_button.setChecked(True)
        iren = window.plotter.iren.interactor
        iren.SetEventPosition(w // 2, h // 2)
        iren.MouseMoveEvent()
        qt_app.processEvents()
        assert window.plotter.actors.get(window.PREVIEW_ACTOR) is not None
        assert window.plotter.actors.get(window.OUTLINE_ACTOR) is not None
        # 削除モードは筆で消すので、電極の輪郭は出さない
        page.erase_mode.setChecked(True)
        iren.MouseMoveEvent()
        qt_app.processEvents()
        assert window.plotter.actors.get(window.OUTLINE_ACTOR) is None
    finally:
        page.erase_mode.setChecked(False)
        page.add_mode.setChecked(True)
        page.pick_button.setChecked(False)
        window.hide()


def test_looking_at_the_surface_edge_on_is_refused_kindly(window):
    """体表面を真横から見て矩形を置こうとしたら、辺の向きが決まらないと伝えて置かない。"""
    page = window.page_electrode
    window.plotter.view_xz()  # 真横から。画面の横は x、上面の法線 z は画面の上
    _choose_rect(page)
    # 画面の横方向を上面の法線と同じ向きにしてしまう視点
    window.plotter.camera.up = (1.0, 0.0, 0.0)
    errors = []
    original = window.show_error
    window.show_error = errors.append
    try:
        page._on_picked(np.array(TOP))
    finally:
        window.show_error = original
    assert window.session.electrode("E1").patches == []
    assert errors and "正面から見る" in str(errors[0])
