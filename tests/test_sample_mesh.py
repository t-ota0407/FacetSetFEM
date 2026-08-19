"""サンプルモデル生成器（scripts/make_sample_mesh.py）の性質を検証する。

六面体格子の四面体分割が**適合（conforming）**であることは、M1 以降の解の正しさが
直接依存する前提である。非適合メッシュでも VTU は書けてしまい、絵も一見それらしく
出るため、ここで明示的に検査しておく。

適合の判定条件:

- 内部の三角形面はちょうど 2 個の四面体に共有される
- 境界の三角形面はちょうど 1 個の四面体だけが持ち、直方体の 6 面のいずれかに載る

電極シェル（CTRIA3）は「外表面の面と一致していなければならない」（SPEC §11.1）ので、
これも合わせて検査する。
"""

from __future__ import annotations

from collections import Counter

import pytest

LX, LY = 60.0, 40.0
LZ = 20.0  # skin 2 + fat 4 + muscle 14


@pytest.fixture(scope="module")
def mesh(make_sample_mesh):
    return make_sample_mesh.build_mesh(LX, LY, nx=12, ny=8, refine=1)


def _faces(tet):
    """四面体の 4 面。節点を昇順に並べて向きを無視した鍵にする。"""
    a, b, c, d = tet
    return [
        tuple(sorted((a, b, c))),
        tuple(sorted((a, b, d))),
        tuple(sorted((a, c, d))),
        tuple(sorted((b, c, d))),
    ]


@pytest.fixture(scope="module")
def face_counts(mesh):
    counter: Counter = Counter()
    for tet in mesh.tets:
        counter.update(_faces(tet))
    return counter


def test_every_face_is_shared_by_at_most_two_tets(face_counts):
    """3 個以上に共有される面があれば、それは分割が壊れている証拠。"""
    assert set(face_counts.values()) <= {1, 2}


def test_boundary_faces_lie_on_the_box_surface(mesh, face_counts):
    """出現回数 1 の面は必ず直方体の表面上にある（＝内部に穴や割れ目がない）。"""
    limits = [(0.0, LX), (0.0, LY), (0.0, LZ)]
    for face, count in face_counts.items():
        if count != 1:
            continue
        points = [mesh.coords[n] for n in face]
        on_surface = any(
            all(p[axis] == lo for p in points) or all(p[axis] == hi for p in points)
            for axis, (lo, hi) in enumerate(limits)
        )
        assert on_surface, f"内部に境界面があります: {face}"


def test_boundary_face_area_equals_the_box_surface_area(mesh, face_counts):
    """境界面の総面積が直方体の表面積と一致する（過不足なく覆えている）。"""
    total = sum(
        _triangle_area(mesh.coords, face)
        for face, count in face_counts.items()
        if count == 1
    )
    expected = 2 * (LX * LY + LY * LZ + LZ * LX)
    assert total == pytest.approx(expected, rel=1e-12)


def test_electrode_shells_coincide_with_exterior_faces(mesh, face_counts):
    """電極面は体表面でなければならない（SPEC §11.1）。"""
    boundary = {face for face, count in face_counts.items() if count == 1}
    assert mesh.tris, "電極シェルが生成されていません"
    for tri in mesh.tris:
        assert tuple(sorted(tri)) in boundary, f"外表面に載っていない電極面: {tri}"


def test_electrode_patches_have_the_intended_area(mesh, make_sample_mesh):
    for _name, pid, (x0, x1), (y0, y1) in make_sample_mesh.ELECTRODES:
        area = sum(
            _triangle_area(mesh.coords, tri)
            for tri, q in zip(mesh.tris, mesh.tri_pids)
            if q == pid
        )
        assert area == pytest.approx((x1 - x0) * (y1 - y0), rel=1e-12)


def test_all_tetrahedra_have_positive_volume(mesh, make_sample_mesh):
    """Nastran の CTETRA は正の体積を要求する。"""
    volumes = [
        make_sample_mesh.signed_volume(*(mesh.coords[n] for n in tet))
        for tet in mesh.tets
    ]
    assert min(volumes) > 0.0
    assert sum(volumes) == pytest.approx(LX * LY * LZ, rel=1e-12)


def test_refinement_keeps_the_mesh_conforming(make_sample_mesh):
    """分割数を変えても適合性が保たれる（Kuhn 分割は格子サイズに依らない）。"""
    fine = make_sample_mesh.build_mesh(LX, LY, nx=5, ny=3, refine=2)
    counter: Counter = Counter()
    for tet in fine.tets:
        counter.update(_faces(tet))
    assert set(counter.values()) <= {1, 2}


def _triangle_area(coords, tri) -> float:
    a, b, c = (coords[n] for n in tri)
    u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    n = (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )
    return 0.5 * (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5
