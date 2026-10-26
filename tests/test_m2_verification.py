"""Python 側からの検証（SPEC §8 M2）。

厳密な収束次数の測定は Rust 側（`crates/facetsetfem-core/tests/convergence.rs`）で行う。
ここでは**研究で実際にやる形**——結果配列と要素重心を numpy で受け取り、
解析解と突き合わせる——が Python API だけで完結することを確かめる。
"""

from __future__ import annotations

import numpy as np
import pytest

import facetsetfem
from conftest import SLAB_AREA, SLAB_LAYERS

CURRENT = 2.0e-3


@pytest.fixture(scope="module")
def layered_solution(full_face_bdf, slab_sigma):
    """上下面全体を電極にした 3 層スラブの解。一次元の厳密解を持つ。"""
    mesh = facetsetfem.read_nastran(full_face_bdf, length_unit="mm")
    case = facetsetfem.Case(mesh)
    case.set_conductivity(slab_sigma)
    case.add_electrode("E1", role="anode", current=CURRENT, shell_pids=[101])
    case.add_electrode("E2", role="cathode", current=CURRENT, shell_pids=[201])
    case.set_solver(tolerance=1e-13)
    case.set_reference_potential("cathode_mean")
    return mesh, case.solve()


def test_element_centroids_line_up_with_the_element_values(layered_solution):
    mesh, result = layered_solution
    centroids = mesh.element_centroids()
    assert centroids.shape == (mesh.n_solid_elements, 3)
    # 重心は必ずモデルの内側にある
    (lo, hi) = mesh.bounding_box()
    assert np.all(centroids > np.array(lo) - 1e-12)
    assert np.all(centroids < np.array(hi) + 1e-12)
    # 要素体積の総和はモデルの体積
    assert result.element_volume.sum() == pytest.approx(0.060 * 0.040 * 0.020, rel=1e-12)


def test_field_matches_the_analytic_solution_elementwise(layered_solution):
    """各層で E = I/(σA)、|J| = I/A（SPEC §8 M2 の 2）。"""
    mesh, result = layered_solution
    pid = mesh.solid_pid_array()
    sigma_of_element = np.zeros(mesh.n_solid_elements)
    for region, _thickness, sigma in SLAB_LAYERS:
        sigma_of_element[pid == region] = sigma

    exact_e = CURRENT / (sigma_of_element * SLAB_AREA)
    exact_j = np.full(mesh.n_solid_elements, CURRENT / SLAB_AREA)

    # 1 次要素は線形場を厳密に再現するので、残るのはソルバの収束誤差だけ
    assert np.allclose(result.e_magnitude, exact_e, rtol=1e-9)
    assert np.allclose(result.j_magnitude, exact_j, rtol=1e-9)

    # 電流は下向き（陽極 → 陰極）で、面内成分は無い
    assert np.allclose(result.current_density[:, :2], 0.0, atol=1e-9 * exact_j.max())
    assert np.all(result.current_density[:, 2] < 0.0)


def test_potential_matches_the_analytic_profile(layered_solution):
    """φ(z) は層ごとに傾きの違う折れ線になる。"""
    mesh, result = layered_solution
    z = mesh.points()[:, 2]

    # 層は z = 0 から muscle・fat・skin の順に積まれている（皮膚が電極を貼る上面）。
    # 陰極は下面なので φ(0) = 0 とし、上へ積み上げる
    def exact(height: float) -> float:
        phi = 0.0
        lower = 0.0
        for _pid, thickness, sigma in reversed(SLAB_LAYERS):
            upper = lower + thickness
            if height > lower:
                phi += CURRENT / (sigma * SLAB_AREA) * (min(height, upper) - lower)
            lower = upper
        return phi

    expected = np.array([exact(v) for v in z])
    scale = expected.max()
    assert np.allclose(result.potential, expected, atol=1e-9 * scale)


def test_l2_error_computed_in_numpy_is_at_the_rounding_level(layered_solution):
    """研究で使う形の誤差評価（体積重みの L2 ノルム）を Python 側で組み立てる。"""
    mesh, result = layered_solution
    pid = mesh.solid_pid_array()
    volume = result.element_volume

    exact = np.zeros((mesh.n_solid_elements, 3))
    for region, _thickness, sigma in SLAB_LAYERS:
        exact[pid == region, 2] = -CURRENT / (sigma * SLAB_AREA)

    diff = result.electric_field - exact
    absolute = np.sqrt(np.sum(volume * np.sum(diff**2, axis=1)))
    reference = np.sqrt(np.sum(volume * np.sum(exact**2, axis=1)))
    relative = absolute / reference
    assert relative < 1e-9, f"E の相対 L2 誤差 {relative}"


def test_conservation_and_impedance_agree_with_theory(layered_solution):
    _mesh, result = layered_solution
    expected_r = sum(t / (s * SLAB_AREA) for _p, t, s in SLAB_LAYERS)
    assert result.impedance == pytest.approx(expected_r, rel=1e-8)
    # 消費電力 = I²R
    assert result.power == pytest.approx(CURRENT**2 * expected_r, rel=1e-8)

    c = result.conservation
    assert c["warnings"] == []
    # 電流保存は総注入電流で正規化した無次元量
    assert abs(c["current_balance"]) < 1e-12
    assert c["energy_relative_error"] < 1e-9
