"""M1（体験一周）の受け入れテスト: Python API から解くところまで。

SPEC §8 の M1 が満たすべきこと:

1. Tet4 アセンブリ + 射影付き Jacobi-CG で解ける
2. 電極荷重が面積按分になっている（SPEC §2.4）
3. 保存則チェックが自動で走る（SPEC §5.4）
4. 実行前の整合性チェックが不備を親切に教える（SPEC §6.2）
5. 進捗の通知とキャンセルができる（SPEC §6.2 の手順 5）
6. 同一入力からビット単位で同じ結果が出る（CLAUDE.md 設計原則 1）
"""

from __future__ import annotations

import numpy as np
import pytest

import facetsetfem
from conftest import SLAB_AREA, SLAB_LAYERS

CURRENT = 2.0e-3  # A


def build_case(bdf, sigma, current=CURRENT, tolerance=1.0e-12):
    mesh = facetsetfem.read_nastran(bdf, length_unit="mm")
    case = facetsetfem.Case(mesh)
    case.set_conductivity(sigma)
    case.add_electrode("E1", role="anode", current=current, shell_pids=[101])
    case.add_electrode("E2", role="cathode", current=current, shell_pids=[201])
    case.set_solver(tolerance=tolerance)
    return case


# --- 解析解との照合 ---------------------------------------------------------


def test_layered_slab_matches_the_one_dimensional_solution(full_face_bdf, slab_sigma):
    """上下面全体に電流を流すと一次元問題になる（SPEC §8 M2 の 2）。

    各層で E = I/(σA)、|J| はどの層でも I/A で一定、R = Σ t_i/(σ_i A)。
    1 次要素は線形場を厳密に再現するので、誤差はソルバの収束分だけ。
    """
    case = build_case(full_face_bdf, slab_sigma, tolerance=1e-13)
    result = case.solve()
    pid = case.mesh.solid_pid_array()

    for region, _thickness, sigma in SLAB_LAYERS:
        expected = CURRENT / (sigma * SLAB_AREA)
        e = result.e_magnitude[pid == region]
        assert e.size > 0
        assert np.allclose(e, expected, rtol=1e-8), f"PID {region}: {e.min()}..{e.max()}"

    # 電流連続: |J| はどの層でも同じ
    assert np.allclose(result.j_magnitude, CURRENT / SLAB_AREA, rtol=1e-8)

    expected_r = sum(t / (s * SLAB_AREA) for _p, t, s in SLAB_LAYERS)
    assert result.impedance == pytest.approx(expected_r, rel=1e-8)
    # 消費電力 = I²R
    assert result.power == pytest.approx(CURRENT**2 * expected_r, rel=1e-8)


def test_field_scales_linearly_with_current(full_face_bdf, slab_sigma):
    base = build_case(full_face_bdf, slab_sigma).solve()
    doubled = build_case(full_face_bdf, slab_sigma, current=2 * CURRENT).solve()
    assert np.allclose(doubled.e_magnitude, 2 * base.e_magnitude, rtol=1e-9)
    # 抵抗は電流に依らない
    assert doubled.impedance == pytest.approx(base.impedance, rel=1e-9)


# --- 保存則チェック（SPEC §5.4）--------------------------------------------


def test_conservation_checks_run_automatically(sample_bdf, slab_sigma):
    result = build_case(sample_bdf, slab_sigma).solve()
    c = result.conservation
    assert c["warnings"] == []
    assert abs(c["current_balance"]) < 1e-12  # 1ᵀf = 0
    assert c["energy_relative_error"] < 1e-9  # ∫σ|E|²dΩ と fᵀφ
    assert c["discrete_residual"] < 1e-11  # ‖Kφ − f‖/‖f‖
    assert result.warnings == []


def test_electrode_load_is_area_weighted(sample_bdf, slab_sigma):
    """SPEC §2.4: 一様電流密度 J = I/|Γ| を面積按分で配る。"""
    case = build_case(sample_bdf, slab_sigma)
    info = {e["name"]: e for e in case.electrode_info()}
    # 15 mm x 20 mm のパッチ = 300 mm²
    assert info["E1"]["area"] == pytest.approx(300e-6, rel=1e-12)
    assert info["E1"]["current_density"] == pytest.approx(CURRENT / 300e-6, rel=1e-12)
    # 陰極は流出（負）
    assert info["E2"]["current"] < 0
    assert info["E1"]["n_faces"] == 24


# --- 整合性チェック（SPEC §6.2）--------------------------------------------


def test_missing_conductivity_is_reported(sample_bdf):
    mesh = facetsetfem.read_nastran(sample_bdf, length_unit="mm")
    case = facetsetfem.Case(mesh)
    case.set_conductivity({1: 0.1})  # PID 2, 3 が未設定
    case.add_electrode("E1", role="anode", current=CURRENT, shell_pids=[101])
    case.add_electrode("E2", role="cathode", current=CURRENT, shell_pids=[201])
    with pytest.raises(facetsetfem.FacetSetFemError) as excinfo:
        case.validate()
    message = str(excinfo.value)
    assert "導電率が未設定" in message
    assert "PID=2, 3" in message
    assert "対処" in message


def test_unbalanced_current_is_rejected(sample_bdf, slab_sigma):
    """全 Neumann 境界なので Σ I_k = 0 でないと解が存在しない（SPEC §2.2）。"""
    mesh = facetsetfem.read_nastran(sample_bdf, length_unit="mm")
    case = facetsetfem.Case(mesh)
    case.set_conductivity(slab_sigma)
    case.add_electrode("E1", role="anode", current=2e-3, shell_pids=[101])
    case.add_electrode("E2", role="cathode", current=1e-3, shell_pids=[201])
    with pytest.raises(facetsetfem.FacetSetFemError) as excinfo:
        case.solve()
    assert "一致しません" in str(excinfo.value)


def test_missing_electrode_role_is_rejected(sample_bdf, slab_sigma):
    mesh = facetsetfem.read_nastran(sample_bdf, length_unit="mm")
    case = facetsetfem.Case(mesh)
    case.set_conductivity(slab_sigma)
    case.add_electrode("E1", role="anode", current=CURRENT, shell_pids=[101])
    with pytest.raises(facetsetfem.FacetSetFemError) as excinfo:
        case.validate()
    assert "cathode" in str(excinfo.value)


def test_unknown_shell_pid_is_reported(sample_bdf, slab_sigma):
    mesh = facetsetfem.read_nastran(sample_bdf, length_unit="mm")
    case = facetsetfem.Case(mesh)
    case.set_conductivity(slab_sigma)
    case.add_electrode("E1", role="anode", current=CURRENT, shell_pids=[999])
    case.add_electrode("E2", role="cathode", current=CURRENT, shell_pids=[201])
    with pytest.raises(facetsetfem.FacetSetFemError) as excinfo:
        case.validate()
    assert "PID=999" in str(excinfo.value)


def test_the_documented_example_gives_the_documented_numbers(sample_bdf, repo_root):
    """README と `facetsetfem` の docstring に載せている例の出力が、実際の値と一致すること。

    以前は docstring の等価抵抗が 83.0 Ω のまま古くなっていた（実際は 206.4 Ω）。
    結果はビット単位で再現する（CLAUDE.md 設計原則 1）ので、表示の文字列ごと比べられる。
    """
    mesh = facetsetfem.read_nastran(sample_bdf, length_unit="mm")
    case = facetsetfem.Case(mesh)
    case.set_conductivity({1: 0.10, 2: 0.04, 3: 0.35})
    case.add_electrode("E1", role="anode", current=2e-3, shell_pids=[101])
    case.add_electrode("E2", role="cathode", current=2e-3, shell_pids=[201])
    result = case.solve()

    for name in ("README.md", "README.ja.md"):  # 英語版と日本語版
        readme = (repo_root / name).read_text(encoding="utf-8")
        assert repr(result) in readme, f"{name} の例を {repr(result)} に直してください"
    assert f">>> round(result.impedance, 1)\n    {round(result.impedance, 1)}" in facetsetfem.__doc__


def test_duplicate_electrode_names_are_rejected(sample_bdf, slab_sigma):
    """同名の陽極と陰極が同じ面を指しても、以前は検証を素通りしていた。

    荷重が打ち消し合って 0 反復で「収束」し、R = 0 Ω・|J| = 0 の結果が
    警告なしで返っていた。電極名の重複そのものを止める。
    """
    mesh = facetsetfem.read_nastran(sample_bdf, length_unit="mm")
    case = facetsetfem.Case(mesh)
    case.set_conductivity(slab_sigma)
    case.add_electrode("E", role="anode", current=CURRENT, shell_pids=[101])
    case.add_electrode("E", role="cathode", current=CURRENT, shell_pids=[101])
    with pytest.raises(facetsetfem.FacetSetFemError) as excinfo:
        case.solve()
    assert "同じ名前の電極" in str(excinfo.value)


def test_bad_role_name_lists_the_choices(sample_bdf):
    mesh = facetsetfem.read_nastran(sample_bdf, length_unit="mm")
    case = facetsetfem.Case(mesh)
    with pytest.raises(facetsetfem.FacetSetFemError) as excinfo:
        case.add_electrode("E1", role="positive", current=CURRENT, shell_pids=[101])
    assert "anode / cathode" in str(excinfo.value)


# --- 進捗とキャンセル（SPEC §6.2 の手順 5）---------------------------------


def test_progress_callback_receives_the_convergence_history(sample_bdf, slab_sigma):
    seen: list[tuple[int, float]] = []

    def progress(iteration: int, residual: float) -> bool:
        seen.append((iteration, residual))
        return True

    result = build_case(sample_bdf, slab_sigma).solve(progress=progress)
    assert len(seen) == result.iterations
    assert [it for it, _ in seen] == list(range(1, result.iterations + 1))
    # 収束履歴は初期残差 + 各反復
    assert result.history.size == result.iterations + 1
    assert result.history[-1] < result.history[0]


def test_progress_callback_can_cancel(sample_bdf, slab_sigma):
    case = build_case(sample_bdf, slab_sigma)
    with pytest.raises(facetsetfem.SolverCancelled):
        case.solve(progress=lambda it, res: it < 3)


def test_exception_in_the_callback_propagates(sample_bdf, slab_sigma):
    case = build_case(sample_bdf, slab_sigma)

    def boom(iteration: int, residual: float) -> bool:
        raise ValueError("テスト用の例外")

    with pytest.raises(ValueError, match="テスト用の例外"):
        case.solve(progress=boom)


# --- 再現性（CLAUDE.md 設計原則 1）-----------------------------------------


def test_repeated_runs_are_bitwise_identical(sample_bdf, slab_sigma):
    a = build_case(sample_bdf, slab_sigma).solve()
    b = build_case(sample_bdf, slab_sigma).solve()
    assert np.array_equal(a.potential, b.potential)
    assert np.array_equal(a.electric_field, b.electric_field)
    assert np.array_equal(a.history, b.history)
    assert a.iterations == b.iterations


# --- 基準電位（SPEC §2.2 / §3.3）-------------------------------------------


def test_reference_potential_shifts_but_does_not_change_the_field(sample_bdf, slab_sigma):
    zero_mean = build_case(sample_bdf, slab_sigma)
    zero_mean.set_reference_potential("zero_mean")
    a = zero_mean.solve()

    cathode = build_case(sample_bdf, slab_sigma)
    cathode.set_reference_potential("cathode_mean")
    b = cathode.solve()

    # 平均ゼロ基準では φ の平均が 0
    assert a.potential.mean() == pytest.approx(0.0, abs=1e-15)
    # 陰極基準では陰極面の平均電位が 0
    assert dict(b.electrode_potentials)["E2"] == pytest.approx(0.0, abs=1e-15)
    # 差は定数シフトだけ。E, J は基準に依存しない
    assert np.allclose(a.potential - b.potential, (a.potential - b.potential)[0])
    assert np.allclose(a.e_magnitude, b.e_magnitude, rtol=1e-12)


def test_node_reference_zeroes_that_node(sample_bdf, slab_sigma):
    case = build_case(sample_bdf, slab_sigma)
    case.set_reference_potential("node:1")
    result = case.solve()
    assert result.potential[0] == pytest.approx(0.0, abs=1e-15)


# --- 結果の受け渡し ---------------------------------------------------------


def test_result_arrays_have_the_expected_shapes(sample_bdf, slab_sigma):
    case = build_case(sample_bdf, slab_sigma)
    result = case.solve()
    mesh = case.mesh
    assert result.potential.shape == (mesh.n_nodes,)
    assert result.electric_field.shape == (mesh.n_solid_elements, 3)
    assert result.current_density.shape == (mesh.n_solid_elements, 3)
    assert result.e_magnitude.shape == (mesh.n_solid_elements,)
    assert result.element_volume.sum() == pytest.approx(0.060 * 0.040 * 0.020, rel=1e-12)
    # 節点外挿は表示用（SPEC §5.1）
    nodal = result.nodal_average("j_magnitude")
    assert nodal.shape == (mesh.n_nodes,)
    # 平滑化されているので要素値の最大を超えない
    assert nodal.max() <= result.j_magnitude.max()


def test_result_vtu_is_readable_by_vtk(sample_bdf, slab_sigma, tmp_path):
    pv = pytest.importorskip("pyvista")
    result = build_case(sample_bdf, slab_sigma).solve()
    out = tmp_path / "result.vtu"
    result.write_vtu(out)

    grid = pv.read(out)
    assert set(grid.point_data.keys()) >= {"phi [V]"}
    assert set(grid.cell_data.keys()) >= {"E [V/m]", "J [A/m2]", "|J| [A/m2]", "PID"}
    assert grid.point_data["phi [V]"] == pytest.approx(result.potential)
    # 要素値はソリッド → シェルの順に並び、シェル側は 0 で埋まる
    assert grid.n_cells == grid.n_cells
    assert grid.cell_data["|J| [A/m2]"].max() == pytest.approx(result.j_magnitude.max())


def test_surface_extraction_for_display(sample_bdf):
    """GUI には四面体そのものではなく外表面の三角形だけを渡す（SPEC §6.3）。"""
    mesh = facetsetfem.read_nastran(sample_bdf, length_unit="mm")
    tris = mesh.surface_triangles()
    owners = mesh.surface_triangle_elements()
    assert tris.ndim == 2 and tris.shape[1] == 3
    assert tris.shape[0] == owners.shape[0]
    assert tris.max() < mesh.n_nodes
    # 直方体の表面積 = 2(60x40 + 40x20 + 20x60) mm²
    expected = 2 * (0.060 * 0.040 + 0.040 * 0.020 + 0.020 * 0.060)
    assert mesh.surface_area() == pytest.approx(expected, rel=1e-12)


def test_solution_reports_timings_and_backend(sample_bdf, slab_sigma):
    result = build_case(sample_bdf, slab_sigma).solve()
    assert result.backend == "cpu_serial"
    timings = result.timings
    assert set(timings) == {"prepare", "assemble", "solve", "post", "total"}
    assert timings["total"] >= timings["solve"]
