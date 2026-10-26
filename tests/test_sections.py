"""断面とクリップ（SPEC §6.3）。

以前は GUI が体積メッシュ全体を VTK の `UnstructuredGrid` にして VTK で切っていた。
頭部クラスのモデルでそれだけで 468 MB、スライダーを 1 目盛り動かすたびに 1〜2 秒
かかったので、切り口の三角形を Rust 側で作るようにした。ここでは

- 切り口の形と値が VTK で切ったものと一致すること
- 断面・クリップの表示で体積メッシュを組み立てないこと

を確かめる。
"""

from __future__ import annotations

import numpy as np
import pytest

import facetsetfem
from facetsetfem.gui.state import Session
from test_m4_elements_and_backends import write_deck

SIGMA = {1: 0.10, 2: 0.04, 3: 0.35}
#: サンプルモデルの寸法 [m]
LX, LY, LZ = 0.060, 0.040, 0.020


def _area(data) -> float:
    p, t = data["points"], data["triangles"].astype(np.int64)
    a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


@pytest.fixture
def session(sample_bdf) -> Session:
    s = Session()
    s.open_mesh(sample_bdf, "mm")
    for pid, sigma in SIGMA.items():
        s.set_conductivity(pid, sigma)
    s.add_electrode("E1", "anode", 2.0e-3, [101])
    s.add_electrode("E2", "cathode", 2.0e-3, [201])
    return s


# --- Python API（facetsetfem.Mesh） -------------------------------------------------


def test_a_slice_is_the_cross_section_of_the_model(sample_bdf):
    mesh = facetsetfem.read_nastran(sample_bdf, length_unit="mm")
    data = mesh.slice([0.0, 0.0, 0.0073], [0.0, 0.0, 1.0])
    n, m = len(data["points"]), len(data["triangles"])
    assert m > 0
    assert data["points"].shape == (n, 3)
    assert data["triangles"].shape == (m, 3)
    assert data["elements"].shape == (m,)
    assert data["edges"].shape == (n, 2)
    assert data["t"].shape == (n,)
    assert data["elements"].max() < mesh.n_solid_elements
    assert _area(data) == pytest.approx(LX * LY, rel=1e-12)
    assert np.allclose(data["points"][:, 2], 0.0073, atol=1e-15)


def test_a_clip_shows_the_kept_half_with_its_cut_face(sample_bdf):
    # x > 30 mm を切り落とすと 30 × 40 × 20 mm の箱が残る。見える面は箱の表面全部
    mesh = facetsetfem.read_nastran(sample_bdf, length_unit="mm")
    data = mesh.clip([0.030, 0.0, 0.0], [1.0, 0.0, 0.0])
    half = LX / 2
    expected = 2 * (half * LY + half * LZ + LY * LZ)
    assert _area(data) == pytest.approx(expected, rel=1e-12)
    assert data["points"][:, 0].max() <= half + 1e-15


@pytest.mark.parametrize("kind", ["tet", "hex", "penta"])
def test_sections_work_for_every_element_type(tmp_path, kind):
    """Hex8 / Penta6 は面共役な四面体分割を通して切る（隙間や重なりができない）。"""
    mesh = write_deck(tmp_path, kind, nx=4, ny=3)
    lo, hi = mesh.bounding_box()
    centre = [0.5 * (a + b) for a, b in zip(lo, hi)]
    for normal, expected in [
        ([0.0, 0.0, 1.0], (hi[0] - lo[0]) * (hi[1] - lo[1])),
        ([1.0, 0.0, 0.0], (hi[1] - lo[1]) * (hi[2] - lo[2])),
    ]:
        area = _area(mesh.slice(centre, normal))
        assert area == pytest.approx(expected, rel=1e-12), f"{kind}: 面積 {area}"


def test_a_bad_plane_is_rejected_with_a_reason(sample_bdf):
    mesh = facetsetfem.read_nastran(sample_bdf, length_unit="mm")
    with pytest.raises(facetsetfem.FacetSetFemError, match="法線"):
        mesh.slice([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    with pytest.raises(facetsetfem.FacetSetFemError, match="位置"):
        mesh.clip([float("nan"), 0.0, 0.0], [0.0, 0.0, 1.0])


# --- GUI のセッション -----------------------------------------------------------


def _integral(poly, name: str) -> float:
    """面積分 ∫ v dA（点データは三角形の 3 頂点の平均。線形なので厳密）。"""
    area = poly.compute_cell_sizes(length=False, volume=False).cell_data["Area"]
    if name in poly.cell_data:
        return float(np.sum(area * poly.cell_data[name]))
    tri = poly.faces.reshape(-1, 4)[:, 1:]
    return float(np.sum(area * poly.point_data[name][tri].mean(axis=1)))


@pytest.mark.parametrize("kind, axis, position", [
    ("slice", "z", 0.37),
    ("slice", "x", 0.5),  # 格子面にちょうど載る
    ("clip", "y", 0.3),
    ("clip", "x", 0.5),
])
def test_sections_match_what_vtk_cuts_from_the_volume_mesh(session, kind, axis, position):
    session.solve()
    origin, normal = session.section_plane(axis, position)
    grid = session.grid()
    if kind == "slice":
        reference = grid.slice(normal=normal, origin=origin)
    else:
        reference = grid.clip(normal=normal, origin=origin, invert=True).extract_surface(
            algorithm="dataset_surface"
        )
    reference = reference.triangulate()
    for field in ("phi [V]", "|J| [A/m2]"):
        ours = session.section(kind, axis, position, field)
        assert ours.area == pytest.approx(reference.area, rel=1e-12)
        assert _integral(ours, field) == pytest.approx(_integral(reference, field), rel=1e-12)
        assert "Normals" in ours.point_data  # 陰影のため（SPEC §6.3）


def test_sections_do_not_build_the_volume_mesh(session):
    session.solve()
    for kind in ("slice", "clip"):
        poly = session.section(kind, "z", 0.4, "|J| [A/m2]")
        assert poly.n_cells > 0
        assert "|J| [A/m2]" in poly.cell_data
        assert "PID" in poly.cell_data
    assert session._grid is None, "断面・クリップのために体積メッシュを組み立てた"


def test_a_plane_outside_the_model_gives_an_empty_section(session):
    assert session.section("slice", "z", 1.5, None).n_cells == 0
