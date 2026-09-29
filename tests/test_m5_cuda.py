"""CUDA バックエンドの Python 側からの確認（SPEC §8 M5）。

Rust 側（`crates/gfem-cuda/tests/`）がビット単位一致そのものを検証する。
ここでは**研究で実際にやる形**——ケースファイルに `backend: cuda` と書いて解く——が
Python API で完結すること、GPU が無い／足りないときに黙って壊れないことを確かめる。

GPU が無い環境ではテストごとに `skip` する。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

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


#: GPU を隠した別プロセスで `backend: cuda` を解くスクリプト。
#:
#: CUDA の初期化はプロセスにつき 1 回なので、すでに GPU を掴んだテストの
#: プロセスの中では GPU を隠せない。
_FALLBACK_SCRIPT = r"""
import json, sys
import gfem

mesh = gfem.read_nastran(sys.argv[1], length_unit="mm")
case = gfem.Case(mesh)
case.set_conductivity({1: 0.10, 2: 0.04, 3: 0.35})
case.add_electrode("E1", role="anode", current=2e-3, shell_pids=[101])
case.add_electrode("E2", role="cathode", current=2e-3, shell_pids=[201])
case.set_solver(backend="cuda")
result = case.solve()
summary = json.loads(result.summary_json())
print(json.dumps({
    "backend": result.backend,
    "backend_setting": summary["solver"]["backend_setting"],
    "warnings": result.warnings,
    "potential": [float(v) for v in result.potential],
}))
"""


def test_falling_back_to_the_cpu_is_never_silent(sample_bdf):
    """GPU を明示したのに使えないときは、CPU で解いて理由を結果に残す（SPEC §4.4）。

    以前はこの経路が「CUDA バックエンドが渡されていません」というエラーで止まり、
    本当の理由（GPU が無い・VRAM が足りない）も失われていた。GPU のある開発機では
    フォールバックが起きずに skip されていたので気づけなかった。
    **GPU を隠した別プロセスで必ず走らせる**（`CUDA_VISIBLE_DEVICES=-1`。
    Windows では空文字の環境変数は未設定と同じになるので -1 を使う）。
    """
    # テストは日本語の表示で回す（conftest.py）。子プロセスにも引き継ぐ
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONUTF8="1", GFEM_LANG="ja")
    done = subprocess.run(
        [sys.executable, "-c", _FALLBACK_SCRIPT, str(sample_bdf)],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout.strip().splitlines()[-1])

    assert out["backend"].startswith("cpu")
    # 何を頼んだかは記録に残る（SPEC §5.5 のプロビナンス）
    assert out["backend_setting"] == "cuda"
    assert any("CPU で解きます" in w for w in out["warnings"]), out["warnings"]

    # 代わりに使った CPU のソルバは、ふつうに CPU で解いたのとビット単位で同じ答えを出す
    mesh = gfem.read_nastran(sample_bdf, length_unit="mm")
    case = gfem.Case(mesh)
    case.set_conductivity({1: 0.10, 2: 0.04, 3: 0.35})
    case.add_electrode("E1", role="anode", current=2e-3, shell_pids=[101])
    case.add_electrode("E2", role="cathode", current=2e-3, shell_pids=[201])
    case.set_solver(backend="auto")
    assert out["potential"] == case.solve().potential.tolist()


@requires_gpu
def test_amg_is_checked_against_the_vram_it_really_needs():
    """AMG は階層（粗いレベルの A と全レベルの P・R）も VRAM に置くので、見積りに入れる（SPEC §4.5）。

    以前は前処理を見ずに見積もっていたので、Jacobi なら載る規模で AMG を選ぶと、
    見積りを通ったあと解いている途中で VRAM が尽きて止まった。
    Jacobi なら載るが AMG なら載らない規模を、いまの空き VRAM から選んで確かめる。
    """
    free = max(d["free_memory"] for d in gfem.cuda_devices() if d["usable"])
    csr_per_node = 8 + 15 * 12  # 行ポインタ + 節点あたり 15 非ゼロ（gfem-cuda の見積りと同じ）
    jacobi_per_node = csr_per_node + 7 * 8
    amg_per_node = jacobi_per_node + 1.4 * csr_per_node
    n = int(0.9 * free / ((jacobi_per_node + amg_per_node) / 2))

    assert gfem.cuda_check(n) is None
    reason = gfem.cuda_check(n, preconditioner="amg")
    assert reason is not None and "VRAM" in reason
