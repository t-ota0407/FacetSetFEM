"""要素種の拡張・座標系・CPU 並列の Python 側からの確認（SPEC §8 M4）。

Rust 側でも同じ性質を検証しているが（`crates/facetsetfem-core/tests/elements.rs`,
`parallel.rs`）、ここでは**研究で実際にやる形**——Nastran ファイルを読んで
Python API で解く——で M4 の 3 つの成果が使えることを確かめる。

1. CHEXA / CPENTA が CTETRA と同じように解ける
2. CORD2R / CORD2C で定義した GRID が基本座標系へ変換される
3. `backend="cpu_parallel"` が逐次とビット単位で同じ答えを出す
"""

from __future__ import annotations

import numpy as np
import pytest

import facetsetfem

#: 電極電流 [A]
CURRENT = 2.0e-3
#: スラブの平面寸法 [mm]（ファイルは mm 単位で書く）
LX_MM, LY_MM = 60.0, 40.0
#: 層構成（PID, 厚さ [mm], σ [S/m], 分割数）
LAYERS = [(1, 2.0, 0.10, 1), (2, 4.0, 0.04, 1), (3, 14.0, 0.35, 2)]
#: 電極面のシェル PID
TOP_PID, BOTTOM_PID = 101, 201

SIGMA = {pid: sigma for pid, _t, sigma, _d in LAYERS}
AREA_M2 = (LX_MM * 1e-3) * (LY_MM * 1e-3)
#: 一次元の厳密解（層の直列抵抗）[Ω]
EXACT_R = sum((t * 1e-3) / (sigma * AREA_M2) for _pid, t, sigma, _d in LAYERS)


# --- デッキの生成 -------------------------------------------------------


def _grid_lines(nx: int, ny: int) -> tuple[list[str], list[float], int]:
    """格子点の GRID カードと、各段の z 座標 [mm] を作る。"""
    z_levels = [0.0]
    for _pid, thickness, _sigma, divisions in LAYERS:
        base = z_levels[-1]
        for i in range(divisions):
            z_levels.append(base + thickness * (i + 1) / divisions)
    lines = []
    for k, z in enumerate(z_levels):
        for j in range(ny + 1):
            for i in range(nx + 1):
                nid = 1 + i + (nx + 1) * (j + (ny + 1) * k)
                x = LX_MM * i / nx
                y = LY_MM * j / ny
                lines.append(f"GRID,{nid},,{x},{y},{z}")
    return lines, z_levels, len(z_levels)


def _layer_pid(k: int) -> int:
    """z 方向 k 段目（0 始まり）の要素が属する層の PID。"""
    remaining = k
    for pid, _t, _sigma, divisions in LAYERS:
        if remaining < divisions:
            return pid
        remaining -= divisions
    raise AssertionError("段数が層の分割数の合計を超えています")


def _signed_volume(cells) -> float:
    """格子番号 4 点の符号付き体積（並びの向きだけ見たいので格子座標で十分）。"""
    (a, b, c, d) = [np.array(p, dtype=float) for p in cells]
    return float(np.dot(np.cross(b - a, c - a), d - a)) / 6.0


def slab_deck(kind: str, nx: int = 3, ny: int = 2) -> str:
    """3 層スラブのデッキ。`kind` は "hex" / "penta" / "tet"。"""
    lines, _z, n_levels = _grid_lines(nx, ny)
    for pid, _t, _sigma, _d in LAYERS:
        lines.append(f"PSOLID,{pid},{pid}")
    node = lambda i, j, k: 1 + i + (nx + 1) * (j + (ny + 1) * k)  # noqa: E731

    eid = 1
    for k in range(n_levels - 1):
        pid = _layer_pid(k)
        for j in range(ny):
            for i in range(nx):
                # 下面 4 点（+z から見て反時計回り）と、その真上の 4 点
                lo = [node(i, j, k), node(i + 1, j, k), node(i + 1, j + 1, k), node(i, j + 1, k)]
                hi = [n + (nx + 1) * (ny + 1) for n in lo]
                if kind == "hex":
                    g = lo + hi
                    lines.append("CHEXA,{},{},{},{},{},{},{},{},+C{}".format(eid, pid, *g[:6], eid))
                    lines.append(f"+C{eid},{g[6]},{g[7]}")
                    eid += 1
                elif kind == "penta":
                    for t in ([0, 1, 2], [0, 2, 3]):
                        g = [lo[n] for n in t] + [hi[n] for n in t]
                        lines.append("CPENTA,{},{},{},{},{},{},{},{}".format(eid, pid, *g))
                        eid += 1
                elif kind == "tet":
                    # Kuhn 分割。座標の格子番号が分かるので、符号付き体積が
                    # 正になる並びに直してから書く（反転要素の警告を出さないため）
                    ijk = [
                        (i, j, k), (i + 1, j, k), (i, j + 1, k), (i + 1, j + 1, k),
                        (i, j, k + 1), (i + 1, j, k + 1),
                        (i, j + 1, k + 1), (i + 1, j + 1, k + 1),
                    ]
                    corner = [node(*c) for c in ijk]
                    for t in ([0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7],
                              [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]):
                        if _signed_volume([ijk[n] for n in t]) < 0:
                            t = [t[0], t[1], t[3], t[2]]
                        lines.append(
                            "CTETRA,{},{},{},{},{},{}".format(
                                eid, pid, *[corner[n] for n in t]
                            )
                        )
                        eid += 1
                else:
                    raise ValueError(kind)

    # 電極面（上面全体を陽極、下面全体を陰極）
    lines.append(f"PSHELL,{TOP_PID},{TOP_PID},0.1")
    lines.append(f"PSHELL,{BOTTOM_PID},{BOTTOM_PID},0.1")
    for k, pid in ((n_levels - 1, TOP_PID), (0, BOTTOM_PID)):
        for j in range(ny):
            for i in range(nx):
                q = [node(i, j, k), node(i + 1, j, k), node(i + 1, j + 1, k), node(i, j + 1, k)]
                if kind == "penta" or kind == "tet":
                    lines.append("CTRIA3,{},{},{},{},{}".format(eid, pid, q[0], q[1], q[2]))
                    eid += 1
                    lines.append("CTRIA3,{},{},{},{},{}".format(eid, pid, q[0], q[2], q[3]))
                    eid += 1
                else:
                    lines.append("CQUAD4,{},{},{},{},{},{}".format(eid, pid, *q))
                    eid += 1
    return "\n".join(lines) + "\n"


def write_deck(tmp_path, kind: str, **kwargs) -> "facetsetfem.Mesh":
    path = tmp_path / f"slab_{kind}.bdf"
    path.write_text(slab_deck(kind, **kwargs), encoding="utf-8")
    return facetsetfem.read_nastran(path, length_unit="mm")


def solved(mesh: "facetsetfem.Mesh", **solver) -> "facetsetfem.Solution":
    case = facetsetfem.Case(mesh)
    case.set_conductivity(SIGMA)
    case.add_electrode("E1", role="anode", current=CURRENT, shell_pids=[TOP_PID])
    case.add_electrode("E2", role="cathode", current=CURRENT, shell_pids=[BOTTOM_PID])
    case.set_solver(tolerance=1e-13, **solver)
    case.set_reference_potential("cathode_mean")
    return case.solve()


# --- 1. 要素種 ---------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "card"), [("tet", "CTETRA"), ("hex", "CHEXA"), ("penta", "CPENTA")]
)
def test_every_solid_element_type_solves_the_layered_slab(tmp_path, kind, card):
    """一次元問題なので、どの要素種でも同じ厳密解に一致する（SPEC §2.3）。"""
    mesh = write_deck(tmp_path, kind)
    counts = dict(mesh.element_counts())
    assert counts.get(card, 0) > 0, f"{card} が読めていません: {counts}"

    result = solved(mesh)
    assert result.residual < 1e-13

    # 等価抵抗が層の直列抵抗に一致する
    assert result.impedance == pytest.approx(EXACT_R, rel=1e-9)
    # 体積の総和はモデルの体積
    volume = (LX_MM * 1e-3) * (LY_MM * 1e-3) * (sum(t for _p, t, _s, _d in LAYERS) * 1e-3)
    assert result.element_volume.sum() == pytest.approx(volume, rel=1e-12)

    # 各層で |E| = I/(σA)、|J| = I/A
    pid = mesh.solid_pid_array()
    sigma_of = np.zeros(mesh.n_solid_elements)
    for region, sigma in SIGMA.items():
        sigma_of[pid == region] = sigma
    assert np.allclose(result.e_magnitude, CURRENT / (sigma_of * AREA_M2), rtol=1e-8)
    assert np.allclose(result.j_magnitude, CURRENT / AREA_M2, rtol=1e-8)


@pytest.mark.parametrize(("kind", "cell_type"), [("hex", 12), ("penta", 13)])
def test_mixed_element_types_are_written_to_vtu(tmp_path, kind, cell_type):
    """CHEXA は VTK_HEXAHEDRON、CPENTA は VTK_WEDGE として出る（SPEC §5.5）。"""
    pv = pytest.importorskip("pyvista")
    mesh = write_deck(tmp_path, kind)
    result = solved(mesh)
    out = tmp_path / f"{kind}.vtu"
    result.write_vtu(out)

    # セルはソリッド → シェルの順に並ぶ（要素値配列と同じ並び）
    grid = pv.read(out)
    solids = grid.celltypes[: mesh.n_solid_elements]
    shells = grid.celltypes[mesh.n_solid_elements :]
    assert set(solids.tolist()) == {cell_type}
    assert set(shells.tolist()) <= {5, 9}, "電極面が VTK_TRIANGLE / VTK_QUAD でない"
    assert len(shells) == mesh.n_shell_elements
    # 要素値も一緒に入っている（SPEC §5.5）
    assert "|J| [A/m2]" in grid.cell_data
    assert np.allclose(grid.cell_data["|J| [A/m2]"][: mesh.n_solid_elements],
                       CURRENT / AREA_M2, rtol=1e-8)


# --- 2. 座標系 ---------------------------------------------------------


def _read_grids(tmp_path, deck: str) -> np.ndarray:
    """GRID だけのデッキを読んで、基本座標系の節点座標 [m] を返す。

    ソルバは通さないので、四面体を 1 つだけ足して読めるようにする。
    """
    path = tmp_path / "coord.bdf"
    path.write_text(deck, encoding="utf-8")
    return facetsetfem.read_nastran(path, length_unit="mm").points()


#: 直交座標系。CID=10 は 90° 回した系、CID=20 は平行移動しただけの系
CORD_DECK_HEAD = """CORD2R,10,,0.0,0.0,0.0,0.0,0.0,1.0,+A
+A,0.0,1.0,0.0
CORD2R,20,,10.0,20.0,30.0,10.0,20.0,31.0,+B
+B,11.0,20.0,30.0
CORD2C,30,,0.0,0.0,0.0,0.0,0.0,1.0,+C
+C,1.0,0.0,0.0
PSOLID,1,1
"""


def test_cord2r_rotates_and_translates_grid_points(tmp_path):
    """CORD2R で定義した GRID が基本座標系に直される（SPEC §3.1）。"""
    deck = CORD_DECK_HEAD + "\n".join(
        [
            "GRID,1,,1.0,0.0,0.0",  # 基本座標系
            "GRID,2,10,1.0,0.0,0.0",  # z 軸まわりに 90°
            "GRID,3,20,1.0,2.0,3.0",  # 平行移動のみ
            "GRID,4,,0.0,0.0,5.0",
            "CTETRA,1,1,1,2,3,4",
        ]
    ) + "\n"
    points = _read_grids(tmp_path, deck)
    # mm → m の変換も済んでいる
    assert np.allclose(points[0], [1e-3, 0.0, 0.0])
    assert np.allclose(points[1], [0.0, 1e-3, 0.0])
    assert np.allclose(points[2], [11e-3, 22e-3, 33e-3])


def test_cord2c_reads_angles_in_degrees(tmp_path):
    """CORD2C の GRID は (r, θ[度], z)（SPEC §3.1）。"""
    deck = CORD_DECK_HEAD + "\n".join(
        [
            "GRID,1,30,2.0,0.0,0.0",
            "GRID,2,30,2.0,90.0,0.0",
            "GRID,3,30,2.0,180.0,5.0",
            "GRID,4,,0.0,0.0,9.0",
            "CTETRA,1,1,1,2,3,4",
        ]
    ) + "\n"
    points = _read_grids(tmp_path, deck)
    assert np.allclose(points[0], [2e-3, 0.0, 0.0])
    assert np.allclose(points[1], [0.0, 2e-3, 0.0], atol=1e-15)
    assert np.allclose(points[2], [-2e-3, 0.0, 5e-3], atol=1e-15)


def test_unknown_coordinate_system_is_reported(tmp_path):
    """存在しない CP を参照したら、どの GRID かを示して止まる（設計原則 6）。"""
    deck = "PSOLID,1,1\nGRID,7,99,1.0,2.0,3.0\n"
    path = tmp_path / "bad_cp.bdf"
    path.write_text(deck, encoding="utf-8")
    with pytest.raises(facetsetfem.FacetSetFemError) as excinfo:
        facetsetfem.read_nastran(path, length_unit="mm")
    message = str(excinfo.value)
    assert "99" in message, message
    assert "7" in message, message


def test_inverted_hex_is_reported_at_read_time(tmp_path):
    """ヤコビアンが正でない CHEXA は、読み込み時に EID を挙げて止まる（SPEC §2.3）。"""
    deck = slab_deck("hex")
    # 最初の CHEXA だけ底面と上面を入れ替える
    lines = deck.splitlines()
    i = next(n for n, line in enumerate(lines) if line.startswith("CHEXA"))
    head = lines[i].split(",")
    cont = lines[i + 1].split(",")
    g = head[3:9] + cont[1:3]
    lines[i] = ",".join(head[:3] + g[4:] + g[:2] + [head[9]])
    lines[i + 1] = ",".join([cont[0]] + g[2:4])
    path = tmp_path / "inverted.bdf"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(facetsetfem.FacetSetFemError) as excinfo:
        facetsetfem.read_nastran(path, length_unit="mm")
    message = str(excinfo.value)
    assert "ヤコビアン" in message, message
    assert "CHEXA" in message, message
    assert head[1] in message, message  # EID


@pytest.mark.parametrize("kind", ["tet", "hex", "penta"])
def test_solid_cells_feed_pyvista_directly(tmp_path, kind):
    """要素種が混ざっていても GUI の表示経路が通る（SPEC §6.3）。"""
    pv = pytest.importorskip("pyvista")
    mesh = write_deck(tmp_path, kind)
    cells, types = mesh.solid_cells()
    grid = pv.UnstructuredGrid(cells, types, mesh.points())

    assert grid.n_cells == mesh.n_solid_elements
    assert set(types.tolist()) == {{"tet": 10, "hex": 12, "penta": 13}[kind]}
    # 体積が一致する ＝ 節点の並びが VTK と食い違っていない
    volume = (LX_MM * 1e-3) * (LY_MM * 1e-3) * (sum(t for _p, t, _s, _d in LAYERS) * 1e-3)
    assert grid.volume == pytest.approx(volume, rel=1e-9)


# --- 五面体の節点の並びと VTK の版（SPEC §5.5）--------------------------------
#
# VTK 9.7 で VTK_WEDGE の並びの約束が逆になった。並びを切り替える版の境目は Rust 側の
# 1 か所（`WedgeOrder::for_vtk`）にだけ置き、ここでは境目を書かずに「入っている VTK で
# 体積が正になる並び」を実際に確かめる。同じテストを VTK 9.6 と 9.7 の両方で回して
# 確かめた（SPEC §8 の「2026-11」）。


def _signed_cell_volumes(grid) -> np.ndarray:
    """VTK が計算する符号付きのセル体積（vtkCellSizeFilter）。裏返ったセルは負になる。"""
    return grid.compute_cell_sizes(length=False, area=False, volume=True).cell_data["Volume"]


def test_exactly_one_wedge_order_suits_the_installed_vtk(tmp_path):
    """入っている VTK で体積が正になる並びはちょうど 1 つで、省略時はそれを選ぶ。"""
    pv = pytest.importorskip("pyvista")
    mesh = write_deck(tmp_path, "penta")
    points = mesh.points()
    suits = []
    for order in facetsetfem.wedge_order_choices():
        cells, types = mesh.solid_cells(wedge_order=order)
        volumes = _signed_cell_volumes(pv.UnstructuredGrid(cells, types, points))
        assert np.all(volumes > 0) or np.all(volumes < 0), f"{order}: 向きが揃っていません"
        if np.all(volumes > 0):
            suits.append(order)
    assert len(suits) == 1, suits

    chosen, _ = mesh.solid_cells()
    expected, _ = mesh.solid_cells(wedge_order=suits[0])
    assert np.array_equal(chosen, expected)


def test_a_vtu_reads_the_right_way_round_only_in_a_matching_vtk(tmp_path):
    """VTU は読む側の VTK の版で解釈が変わる。合う並びで書けば体積が正、合わなければ負。

    VTU には約束の印が無く（ファイルの版の番号も同じ）、VTK 9.7 のリーダは古い並びの
    ファイルを読み替えない。だから開く ParaView の版に合わせて書く側で選ぶ（SPEC §5.5）。
    """
    pv = pytest.importorskip("pyvista")
    mesh = write_deck(tmp_path, "penta")
    installed, _ = mesh.solid_cells()  # 入っている VTK に合う並び（上のテスト）
    for order in facetsetfem.wedge_order_choices():
        out = tmp_path / f"{order}.vtu"
        mesh.write_vtu(out, wedge_order=order)
        grid = pv.read(out)
        volumes = _signed_cell_volumes(grid)[grid.celltypes == 13]
        assert len(volumes) == mesh.n_solid_elements
        cells, _ = mesh.solid_cells(wedge_order=order)
        if np.array_equal(cells, installed):
            assert np.all(volumes > 0), order
        else:
            assert np.all(volumes < 0), order


def test_the_vtu_is_written_for_paraview_6_2_or_later_by_default(tmp_path):
    """VTU の既定は VTK 9.7 以降（ParaView 6.2 以降）の並び（SPEC §5.5）。"""
    pv = pytest.importorskip("pyvista")
    mesh = write_deck(tmp_path, "penta")
    assert facetsetfem.wedge_order_choices() == ["vtk9.7", "vtk9.6"]
    written = {}
    for name, order in [("default", None), ("vtk9.7", "vtk9.7"), ("vtk9.6", "vtk9.6")]:
        out = tmp_path / f"{name}.vtu"
        if order is None:
            mesh.write_vtu(out)
        else:
            mesh.write_vtu(out, wedge_order=order)
        written[name] = pv.read(out).cells
    assert np.array_equal(written["default"], written["vtk9.7"])
    assert not np.array_equal(written["default"], written["vtk9.6"])


def test_an_unknown_wedge_order_says_which_names_to_use(tmp_path):
    mesh = write_deck(tmp_path, "penta")
    out = tmp_path / "x.vtu"
    for call in (
        lambda: mesh.solid_cells(wedge_order="paraview"),
        lambda: mesh.write_vtu(out, wedge_order="paraview"),
    ):
        with pytest.raises(facetsetfem.FacetSetFemError) as excinfo:
            call()
        message = str(excinfo.value)
        assert "paraview" in message and "vtk9.7" in message and "vtk9.6" in message
        assert "ParaView 6.2" in message, message
    assert not out.exists(), "解釈できない指定で書き始めない"


def test_without_vtk_the_wedge_order_must_be_given(tmp_path, monkeypatch):
    """VTK を読み込めなければ並びを黙って決めず、指定を求める。五面体が無ければ要らない。"""
    import sys

    monkeypatch.setitem(sys.modules, "vtkmodules.vtkCommonCore", None)  # import が失敗する
    penta = write_deck(tmp_path, "penta")
    with pytest.raises(facetsetfem.FacetSetFemError) as excinfo:
        penta.solid_cells()
    assert "wedge_order" in str(excinfo.value)
    assert "対処" in str(excinfo.value)
    penta.solid_cells(wedge_order="vtk9.6")  # 指定すれば VTK の版は見ない
    write_deck(tmp_path, "tet").solid_cells()  # 五面体が無ければ並びはどれでも同じ


def test_the_case_file_chooses_the_wedge_order_of_the_vtu(tmp_path):
    """ケースファイルの `output.vtu_wedge_order` が、解いた後に書く VTU に効く（SPEC §3.3）。"""
    pv = pytest.importorskip("pyvista")
    (tmp_path / "slab_penta.bdf").write_text(slab_deck("penta"), encoding="utf-8")
    materials = "".join(f"  - {{pid: {p}, sigma: {s}}}\n" for p, s in SIGMA.items())
    (tmp_path / "case.yaml").write_text(
        "version: 1\n"
        "mesh: {file: slab_penta.bdf, length_unit: mm}\n"
        f"materials:\n{materials}"
        "electrodes:\n"
        f"  - {{name: E1, role: anode, current: {CURRENT}, faces: {{shell_pids: [{TOP_PID}]}}}}\n"
        f"  - {{name: E2, role: cathode, current: {CURRENT}, faces: {{shell_pids: [{BOTTOM_PID}]}}}}\n"
        "output: {vtu: result.vtu, vtu_wedge_order: vtk9.6}\n",
        encoding="utf-8",
    )
    study = facetsetfem.Study.load(tmp_path / "case.yaml")
    assert study.output.vtu_wedge_order == "vtk9.6"
    study.solve()

    mesh = facetsetfem.read_nastran(tmp_path / "slab_penta.bdf", length_unit="mm")
    expected, _ = mesh.solid_cells(wedge_order="vtk9.6")
    cells = pv.read(tmp_path / "result.vtu").cells
    assert np.array_equal(cells[: len(expected)], expected)


def test_gui_state_shows_a_hex_mesh(tmp_path):
    """GUI のセッション状態も要素種を選ばない（SPEC §6.3）。"""
    pytest.importorskip("pyvista")
    from facetsetfem.gui.state import Session

    path = tmp_path / "hex.bdf"
    path.write_text(slab_deck("hex"), encoding="utf-8")
    state = Session()
    state.open_mesh(path, length_unit="mm")
    grid = state.grid()
    assert grid.n_cells == state.mesh.n_solid_elements
    assert "PID" in grid.cell_data


def test_the_gui_asks_which_paraview_opens_the_vtu_only_for_wedges(tmp_path):
    """五面体があるときだけ、VTU の保存ダイアログで開く ParaView の版を選ばせる（SPEC §5.5）。

    初期選択はこのセッションで前回選んだもの → ケースの `output.vtu_wedge_order` → 既定の順。
    """
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvista")
    from facetsetfem.gui.main_window import VTU_WEDGE_FILTERS, vtu_save_filters
    from facetsetfem.gui.state import Session

    # Rust 側の選択肢には、どれも表示名がある（選択肢が増えたら表示名も足すこと）
    assert set(facetsetfem.wedge_order_choices()) == set(VTU_WEDGE_FILTERS)

    def session(kind: str) -> Session:
        path = tmp_path / f"{kind}.bdf"
        path.write_text(slab_deck(kind), encoding="utf-8")
        state = Session()
        state.open_mesh(path, length_unit="mm")
        return state

    filters, orders, selected = vtu_save_filters(session("tet"))
    assert orders == [None] and filters == [selected], "四面体だけなら並びを選ばせない"

    state = session("penta")
    filters, orders, selected = vtu_save_filters(state)
    assert orders == ["vtk9.7", "vtk9.6"]
    assert selected == filters[0] and "ParaView 6.2" in selected

    state.study.output.vtu_wedge_order = "vtk9.6"  # ケースの指定
    assert vtu_save_filters(state)[2] == filters[1]

    for pid, sigma in SIGMA.items():
        state.set_conductivity(pid, sigma)
    state.add_electrode("E1", role="anode", current=CURRENT, shell_pids=[TOP_PID])
    state.add_electrode("E2", role="cathode", current=CURRENT, shell_pids=[BOTTOM_PID])
    state.solve()
    state.write_vtu(tmp_path / "chosen.vtu", "vtk9.7")  # ダイアログで選んだ
    assert vtu_save_filters(state)[2] == filters[0], "選んだものを次の既定にする"
    state.write_vtu(tmp_path / "again.vtu")  # 省けば前回の選択
    pv = pytest.importorskip("pyvista")
    assert np.array_equal(
        pv.read(tmp_path / "again.vtu").cells, pv.read(tmp_path / "chosen.vtu").cells
    )


def test_the_file_type_chosen_in_the_save_dialog_reaches_the_vtu(
    tmp_path, main_window, qt_app, monkeypatch
):
    """保存ダイアログで選んだファイルの種類どおりの並びで VTU を書く（SPEC §5.5）。"""
    pv = pytest.importorskip("pyvista")
    from facetsetfem.gui import main_window as module

    path = tmp_path / "penta.bdf"
    path.write_text(slab_deck("penta"), encoding="utf-8")
    window = main_window
    window.open_mesh(path, "mm")
    for pid, sigma in SIGMA.items():
        window.session.set_conductivity(pid, sigma)
    window.session.add_electrode("E1", role="anode", current=CURRENT, shell_pids=[TOP_PID])
    window.session.add_electrode("E2", role="cathode", current=CURRENT, shell_pids=[BOTTOM_PID])
    window.session.solve()

    asked = {}
    out = tmp_path / "chosen.vtu"

    def save_dialog(_parent, _caption, _name, filters, selected):
        asked["filters"], asked["selected"] = filters.split(";;"), selected
        return str(out), asked["filters"][1]  # 2 つ目（ParaView 6.1 以前）を選ぶ

    monkeypatch.setattr(module.QFileDialog, "getSaveFileName", save_dialog)
    window.export_vtu_dialog()
    qt_app.processEvents()

    assert len(asked["filters"]) == 2 and asked["selected"] == asked["filters"][0]
    assert "6.1" in asked["filters"][1]
    expected, _ = window.session.mesh.solid_cells(wedge_order="vtk9.6")
    assert np.array_equal(pv.read(out).cells[: len(expected)], expected)


# --- 3. バックエンド ---------------------------------------------------


def test_parallel_backend_is_bitwise_identical_to_serial(tmp_path):
    """CLAUDE.md 設計原則 1: 並列化しても 1 ビットも変わってはいけない。"""
    mesh = write_deck(tmp_path, "tet", nx=6, ny=4)
    serial = solved(mesh, backend="cpu")
    parallel = solved(mesh, backend="cpu_parallel")

    assert serial.backend == "cpu_serial"
    assert parallel.backend == "cpu_parallel"
    assert np.array_equal(serial.potential, parallel.potential)
    assert np.array_equal(serial.electric_field, parallel.electric_field)
    assert np.array_equal(serial.current_density, parallel.current_density)
    assert serial.iterations == parallel.iterations
    assert serial.impedance == parallel.impedance


def test_case_reports_and_validates_the_backend(tmp_path):
    mesh = write_deck(tmp_path, "tet")
    case = facetsetfem.Case(mesh)
    assert case.solver["backend"] == "auto"

    case.set_solver(backend="cpu_parallel")
    assert case.solver["backend"] == "cpu_parallel"

    # cuda は M5 で受け付けるようになった
    case.set_solver(backend="cuda")
    assert case.solver["backend"] == "cuda"

    with pytest.raises(facetsetfem.FacetSetFemError) as excinfo:
        case.set_solver(backend="opencl")
    assert "opencl" in str(excinfo.value)
    # 弾かれた後も設定は元のまま
    assert case.solver["backend"] == "cuda"


def test_study_accepts_cpu_parallel(tmp_path):
    """ケースファイルからも並列を選べる（SPEC §3.3）。"""
    spec = facetsetfem.SolverSpec(backend="cpu_parallel")
    assert spec.to_dict()["backend"] == "cpu_parallel"

    path = tmp_path / "slab.bdf"
    path.write_text(slab_deck("tet"), encoding="utf-8")
    study = facetsetfem.Study(
        mesh_file=path,
        length_unit="mm",
        materials=[facetsetfem.Material(pid=pid, sigma=s) for pid, s in SIGMA.items()],
        electrodes=[
            facetsetfem.ElectrodeSpec(name="E1", role="anode", current=CURRENT, shell_pids=[TOP_PID]),
            facetsetfem.ElectrodeSpec(
                name="E2", role="cathode", current=CURRENT, shell_pids=[BOTTOM_PID]
            ),
        ],
        solver=spec,
    )
    result = study.solve()
    assert result.backend == "cpu_parallel"
    assert result.impedance == pytest.approx(EXACT_R, rel=1e-9)
