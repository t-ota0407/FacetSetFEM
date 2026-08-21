"""CUDA バックエンドの Python 側からの確認（SPEC §8 M5）。

Rust 側（`crates/gfem-cuda/tests/`）がビット単位一致そのものを検証する。
ここでは**研究で実際にやる形**——ケースファイルに `backend: cuda` と書いて解く——が
Python API で完結すること、GPU が無い／足りないときに黙って壊れないことを確かめる。

GPU が無い環境ではテストごとに `skip` する。
"""

from __future__ import annotations

import numpy as np
import pytest

import gfem

from test_m4_elements_and_backends import (
    AREA_M2,
    CURRENT,
    EXACT_R,
    SIGMA,
    BOTTOM_PID,
    TOP_PID,
    slab_deck,
    write_deck,
)


def gpu_ready(n_nodes: int = 1) -> bool:
    """この規模を GPU で解けるか。"""
    return gfem.cuda_available() and gfem.cuda_check(n_nodes) is None


requires_gpu = pytest.mark.skipif(not gpu_ready(), reason="使える GPU が無い")


# --- 環境の問い合わせ（GPU が無くても動くこと） -------------------------


def test_describe_gpus_always_returns_something():
    text = gfem.describe_gpus()
    assert isinstance(text, str) and text
    if gfem.cuda_available() and gfem.cuda_devices():
        assert "GPU 0" in text
    else:
        # 無いなら無いと書く（黙って空にしない）
        assert "CPU" in text or "見つかりません" in text


def test_device_list_has_the_expected_shape():
    for d in gfem.cuda_devices():
        assert set(d) == {
            "ordinal",
            "name",
            "compute_capability",
            "total_memory",
            "free_memory",
            "usable",
        }
        assert len(d["compute_capability"]) == 2
        assert d["free_memory"] <= d["total_memory"]
        assert d["usable"] == (d["compute_capability"] >= (6, 0))


def test_absurd_problem_sizes_are_rejected_with_a_reason():
    """VRAM に載らない規模は、理由つきで断られる（SPEC §4.4）。"""
    reason = gfem.cuda_check(1_000_000_000)
    assert reason is not None
    assert "VRAM" in reason or "GPU" in reason


# --- 実際に解く ---------------------------------------------------------


@requires_gpu
def test_cuda_gives_the_same_answer_as_the_cpu(tmp_path):
    """CLAUDE.md 設計原則 1: バックエンドを変えても 1 ビットも変わらない。"""
    mesh = write_deck(tmp_path, "tet", nx=6, ny=4)

    def solved(backend):
        case = gfem.Case(mesh)
        case.set_conductivity(SIGMA)
        case.add_electrode("E1", role="anode", current=CURRENT, shell_pids=[TOP_PID])
        case.add_electrode("E2", role="cathode", current=CURRENT, shell_pids=[BOTTOM_PID])
        case.set_solver(tolerance=1e-13, backend=backend)
        case.set_reference_potential("cathode_mean")
        return case.solve()

    cpu = solved("cpu")
    gpu = solved("cuda")

    assert gpu.backend == "cuda"
    assert np.array_equal(cpu.potential, gpu.potential)
    assert np.array_equal(cpu.electric_field, gpu.electric_field)
    assert np.array_equal(cpu.current_density, gpu.current_density)
    assert cpu.iterations == gpu.iterations
    assert cpu.impedance == gpu.impedance
    assert gpu.impedance == pytest.approx(EXACT_R, rel=1e-9)


@requires_gpu
def test_study_can_ask_for_cuda(tmp_path):
    """ケースファイルからも GPU を選べる（SPEC §3.3）。"""
    path = tmp_path / "slab.bdf"
    path.write_text(slab_deck("tet"), encoding="utf-8")
    study = gfem.Study(
        mesh_file=path,
        length_unit="mm",
        materials=[gfem.Material(pid=pid, sigma=s) for pid, s in SIGMA.items()],
        electrodes=[
            gfem.ElectrodeSpec(name="E1", role="anode", current=CURRENT, shell_pids=[TOP_PID]),
            gfem.ElectrodeSpec(
                name="E2", role="cathode", current=CURRENT, shell_pids=[BOTTOM_PID]
            ),
        ],
        solver=gfem.SolverSpec(backend="cuda", tolerance=1e-13),
    )
    result = study.solve()
    assert result.backend == "cuda"
    assert result.impedance == pytest.approx(EXACT_R, rel=1e-9)
    # 電流密度は一次元の厳密解どおり
    assert np.allclose(result.j_magnitude, CURRENT / AREA_M2, rtol=1e-8)


@requires_gpu
def test_auto_stays_on_the_cpu_for_small_meshes(tmp_path):
    """小さい問題は転送のほうが高くつくので GPU に載せない（SPEC §4.4）。"""
    mesh = write_deck(tmp_path, "tet")
    assert mesh.n_nodes < 100_000
    case = gfem.Case(mesh)
    case.set_conductivity(SIGMA)
    case.add_electrode("E1", role="anode", current=CURRENT, shell_pids=[TOP_PID])
    case.add_electrode("E2", role="cathode", current=CURRENT, shell_pids=[BOTTOM_PID])
    case.set_solver(backend="auto")
    result = case.solve()
    assert result.backend.startswith("cpu")


def test_falling_back_to_the_cpu_is_never_silent(tmp_path):
    """GPU を明示したのに使えないときは、結果に理由が残る（SPEC §4.4）。"""
    if gpu_ready():
        pytest.skip("GPU が使えるのでフォールバックは起きない")
    mesh = write_deck(tmp_path, "tet")
    case = gfem.Case(mesh)
    case.set_conductivity(SIGMA)
    case.add_electrode("E1", role="anode", current=CURRENT, shell_pids=[TOP_PID])
    case.add_electrode("E2", role="cathode", current=CURRENT, shell_pids=[BOTTOM_PID])
    case.set_solver(backend="cuda")
    result = case.solve()

    assert result.backend.startswith("cpu")
    assert any("CPU で解きます" in w for w in result.warnings), result.warnings
