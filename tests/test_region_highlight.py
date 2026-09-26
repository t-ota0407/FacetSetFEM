"""体表面に出ていない領域のハイライト（SPEC §6.3）。

以前は体積メッシュ全体を VTK の形式にしてから（頭部クラスで 0.36 秒・468 MB）、その領域を
切り出して VTK に外側の面を取らせていた。いまは領域の境界面を Rust 側で直接取り出す。
"""

from __future__ import annotations

import numpy as np
import pytest

import gfem
from gfem.gui.state import Session

#: 立方体を 6 個の四面体に分ける Kuhn 分割（局所節点 c = bx + 2 by + 4 bz）
KUHN = [(0, 1, 3, 7), (0, 1, 7, 5), (0, 5, 7, 4), (0, 3, 2, 7), (0, 6, 4, 7), (0, 2, 6, 7)]


def buried_box_deck(n: int = 4) -> str:
    """1 辺 n mm の箱。中央の (n/2)³ の立方体だけ PID 2（体表面に出ない）、残りは PID 1。"""
    node = lambda i, j, k: 1 + i + (n + 1) * (j + (n + 1) * k)  # noqa: E731
    lines = ["PSOLID,1,1", "PSOLID,2,2"]
    for k in range(n + 1):
        for j in range(n + 1):
            for i in range(n + 1):
                lines.append(f"GRID,{node(i, j, k)},,{float(i)},{float(j)},{float(k)}")
    inner = range(n // 4, n - n // 4)
    eid = 1
    for k in range(n):
        for j in range(n):
            for i in range(n):
                pid = 2 if (i in inner and j in inner and k in inner) else 1
                corner = lambda c: node(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1))  # noqa: E731
                for t in KUHN:
                    lines.append(f"CTETRA,{eid},{pid}," + ",".join(str(corner(c)) for c in t))
                    eid += 1
    return "\n".join(lines) + "\n"


@pytest.fixture
def buried_bdf(tmp_path):
    path = tmp_path / "buried.bdf"
    path.write_text(buried_box_deck(), encoding="utf-8")
    return path


@pytest.fixture
def session(buried_bdf) -> Session:
    s = Session()
    s.open_mesh(buried_bdf, "mm")
    return s


def test_region_surface_is_the_boundary_of_the_region(buried_bdf):
    mesh = gfem.read_nastran(buried_bdf, length_unit="mm")
    triangles, elements = mesh.region_surface(2)
    assert triangles.shape[1] == 3
    assert len(elements) == len(triangles)
    # 中央の 2 × 2 × 2 mm の立方体の表面（24 mm²）
    p = mesh.points()
    a, b, c = (p[triangles[:, k].astype(np.int64)] for k in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    assert area == pytest.approx(24e-6, rel=1e-12)
    assert set(mesh.solid_pid_array()[elements]) == {2}
    # 無い領域は空
    empty, owners = mesh.region_surface(99)
    assert empty.shape == (0, 3) and len(owners) == 0


def test_the_boundary_matches_what_vtk_extracts(session):
    for pid in (1, 2):
        ours = session.region_boundary(pid)
        reference = session.region_grid(pid).extract_surface(algorithm="dataset_surface")
        reference = reference.triangulate()
        assert ours.n_cells == reference.n_cells
        assert ours.area == pytest.approx(reference.area, rel=1e-12)
        assert "Normals" in ours.point_data  # 陰影のため（SPEC §6.3）


def test_a_buried_region_is_highlighted_without_the_volume_mesh(session):
    # PID 2 は体表面に出ていない（外表面からは切り出せない）
    assert session.region_polydata(2).n_cells == 0
    boundary = session.region_boundary(2)
    assert boundary.n_cells > 0
    assert set(boundary.cell_data["PID"]) == {2}
    assert session._grid is None, "ハイライトのために体積メッシュを組み立てた"
    # 形だけなので使い回す
    assert session.region_boundary(2) is boundary


def test_the_gui_highlights_a_buried_region_without_the_volume_mesh(main_window, buried_bdf):
    main_window.open_mesh(buried_bdf, "mm")
    main_window.refresh_view(highlight_pid=2)
    assert main_window.session._grid is None
