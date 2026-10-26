"""結果の配列の受け渡し（SPEC §6.3 の zero-copy）。

`Solution` の配列は Rust のメモリを複製せずに、読み取り専用の numpy 配列として見せる。
以前は getter を呼ぶたびに丸ごと複製していて、1000 万節点では E・J を 1 回読むごとに
1.4 GB ずつ余計に使った。一方で節点座標（`Mesh.points`）は毎回複製する——pyvista は
読み取り専用の印を見ずにメモリを共有するので、座標を共有するとその先から書き換えられ、
以後の解析がすべて黙って狂う。
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

import facetsetfem
from facetsetfem.gui.state import Session

SIGMA = {1: 0.10, 2: 0.04, 3: 0.35}
ARRAYS = [
    "potential",
    "electric_field",
    "current_density",
    "e_magnitude",
    "j_magnitude",
    "power_density",
    "element_volume",
    "history",
]


@pytest.fixture
def solution(sample_bdf) -> facetsetfem.Solution:
    mesh = facetsetfem.read_nastran(sample_bdf, length_unit="mm")
    case = facetsetfem.Case(mesh)
    case.set_conductivity(SIGMA)
    case.add_electrode("E1", role="anode", current=2.0e-3, shell_pids=[101])
    case.add_electrode("E2", role="cathode", current=2.0e-3, shell_pids=[201])
    return case.solve()


@pytest.mark.parametrize("name", ARRAYS)
def test_result_arrays_are_shared_read_only_views(solution, name):
    a = getattr(solution, name)
    b = getattr(solution, name)
    # 呼ぶたびに複製しない（同じメモリを指す）。親は Solution そのもの
    assert np.shares_memory(a, b)
    assert a.base is solution
    # 書き換えると Rust 側の計算（領域統計・VTU）が狂うので、書き込みは禁じる
    assert not a.flags.writeable
    with pytest.raises(ValueError):
        a[0] = 0.0
    # Python 側から書き込み可に戻すこともできない
    with pytest.raises(ValueError):
        a.setflags(write=True)


def test_a_view_stays_valid_after_the_solution_is_dropped(solution):
    """配列が Solution を生かし続けるので、元の参照を消しても中身は壊れない。"""
    j = solution.j_magnitude
    expected = j.copy()
    del solution
    gc.collect()
    _ = [np.ones(100_000) for _ in range(50)]  # 解放されたメモリが再利用されやすくする
    assert np.array_equal(j, expected)


def test_views_are_the_same_values_as_before(solution):
    # 形と値は従来どおり
    n_elements = solution.mesh.n_solid_elements
    assert solution.current_density.shape == (n_elements, 3)
    assert solution.potential.shape == (solution.mesh.n_nodes,)
    assert np.allclose(
        np.linalg.norm(solution.current_density, axis=1), solution.j_magnitude, rtol=1e-12
    )


def test_mesh_points_are_an_independent_copy(sample_bdf):
    """座標を書き換えられると以後の解析がすべて狂うので、毎回複製して渡す。"""
    mesh = facetsetfem.read_nastran(sample_bdf, length_unit="mm")
    before = mesh.bounding_box()
    p = mesh.points()
    assert p.flags.writeable
    assert not np.shares_memory(p, mesh.points())
    p *= 1000.0  # mm 表示にしたい、などでその場で書き換える
    assert mesh.bounding_box() == before


def test_the_gui_does_not_hand_rust_memory_to_vtk(sample_bdf):
    """pyvista は読み取り専用の印を見ずにメモリを共有する。表示側から結果を書き換えられないよう、
    GUI は VTK に渡す前に複製する。"""
    s = Session()
    s.open_mesh(sample_bdf, "mm")
    for pid, sigma in SIGMA.items():
        s.set_conductivity(pid, sigma)
    s.add_electrode("E1", "anode", 2.0e-3, [101])
    s.add_electrode("E2", "cathode", 2.0e-3, [201])
    r = s.solve()
    rust_arrays = [r.potential, r.e_magnitude, r.j_magnitude, r.power_density,
                   r.electric_field, r.current_density]

    def shares_rust_memory(data) -> bool:
        return any(np.shares_memory(np.asarray(data), a) for a in rust_arrays)

    grid = s.grid()
    for name in list(grid.point_data.keys()) + list(grid.cell_data.keys()):
        data = grid.point_data[name] if name in grid.point_data else grid.cell_data[name]
        assert not shares_rust_memory(data), f"体積メッシュの {name} が Rust のメモリを共有している"
    surface = s.display_surface("phi [V]")
    assert not shares_rust_memory(surface.point_data["phi [V]"])
