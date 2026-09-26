"""M6: AMG 前処理（SPEC §4.5 / §8 M6）。

数学的な性質（V サイクルの対称正定値性、近零空間の保存、バックエンド間の
ビット単位一致）は Rust 側の `crates/gfem-core/tests/amg.rs` で固めてある。
ここで見るのは **Python から使えるか**と**ケースファイルから選べるか**。
"""

from __future__ import annotations

import json
import shutil

import pytest

import gfem
from gfem.study import Study

CURRENT = 2.0e-3  # A


def build_case(bdf, sigma, preconditioner):
    mesh = gfem.read_nastran(bdf, length_unit="mm")
    case = gfem.Case(mesh)
    case.set_conductivity(sigma)
    case.add_electrode("E1", role="anode", current=CURRENT, shell_pids=[101])
    case.add_electrode("E2", role="cathode", current=CURRENT, shell_pids=[201])
    case.set_solver(tolerance=1.0e-11, preconditioner=preconditioner, backend="cpu")
    return case


@pytest.fixture(scope="module")
def solved(sample_bdf, slab_sigma):
    """Jacobi と AMG の両方で 1 回ずつ解いた結果。"""
    return {
        name: build_case(sample_bdf, slab_sigma, name).solve()
        for name in ("jacobi", "amg")
    }


# --- 選択肢として通っていること ---------------------------------------------


def test_amg_is_a_known_preconditioner():
    assert "amg" in gfem.preconditioner_choices()
    assert "jacobi" in gfem.preconditioner_choices()


def test_set_solver_accepts_amg(sample_bdf, slab_sigma):
    case = build_case(sample_bdf, slab_sigma, "amg")
    assert case.solver["preconditioner"] == "amg"
    case.set_solver(preconditioner="jacobi")
    assert case.solver["preconditioner"] == "jacobi"


def test_an_unknown_preconditioner_says_what_is_available(sample_bdf, slab_sigma):
    case = build_case(sample_bdf, slab_sigma, "jacobi")
    with pytest.raises(gfem.GfemError) as excinfo:
        case.set_solver(preconditioner="ilu")
    message = str(excinfo.value)
    assert "ilu" in message
    assert "amg" in message, "使える選択肢を示すこと（設計原則 6）"


# --- 解いた結果 -------------------------------------------------------------


def test_amg_and_jacobi_reach_the_same_answer(solved):
    jacobi, amg = solved["jacobi"], solved["amg"]
    assert amg.preconditioner == "amg"
    assert jacobi.preconditioner == "jacobi"

    scale = max(abs(v) for v in jacobi.potential)
    worst = max(abs(a - b) for a, b in zip(jacobi.potential, amg.potential))
    assert worst < 1e-7 * scale, f"電位の差 {worst}（振幅 {scale}）"
    assert abs(jacobi.impedance - amg.impedance) < 1e-7 * abs(jacobi.impedance)


def test_amg_needs_fewer_iterations(solved):
    jacobi, amg = solved["jacobi"], solved["amg"]
    assert amg.iterations < jacobi.iterations, (
        f"AMG {amg.iterations} 反復 / Jacobi {jacobi.iterations} 反復"
    )


def test_amg_reports_its_hierarchy(solved):
    info = solved["amg"].amg
    assert info is not None
    levels = info["levels"]
    assert len(levels) >= 2, f"階層が 1 段しかない: {levels}"
    sizes = [n for n, _nnz in levels]
    assert sizes == sorted(sizes, reverse=True), f"粗くなっていない: {sizes}"
    assert sizes[0] > sizes[-1]
    # V サイクル 1 回のコストが最細レベルの数倍で収まっていること
    assert 1.0 <= info["operator_complexity"] < 3.0
    assert info["direct_coarse_solve"] is True
    assert info["setup_seconds"] >= 0.0


def test_jacobi_reports_no_hierarchy(solved):
    assert solved["jacobi"].amg is None


def test_the_summary_json_records_the_hierarchy(solved):
    """「この図はどの設定で出たか」を後から追えること（SPEC §5.5）。"""
    summary = json.loads(solved["amg"].summary_json())
    solver = summary["solver"]
    assert solver["preconditioner"] == "amg"
    assert solver["preconditioner_setup_s"] >= 0.0
    assert len(solver["amg"]["levels"]) >= 2
    assert solver["amg"]["levels"][0]["n_rows"] > 0
    assert solver["amg"]["direct_coarse_solve"] is True

    jacobi_summary = json.loads(solved["jacobi"].summary_json())
    assert jacobi_summary["solver"]["preconditioner"] == "jacobi"
    assert "amg" not in jacobi_summary["solver"]


# --- ケースファイル ---------------------------------------------------------


@pytest.fixture
def case_dir(tmp_path, sample_bdf, repo_root):
    shutil.copy(sample_bdf, tmp_path / "slab_3layer.bdf")
    shutil.copy(repo_root / "tests" / "models" / "slab_3layer.yaml", tmp_path / "case.yaml")
    return tmp_path


def test_a_case_file_can_ask_for_amg(case_dir):
    path = case_dir / "case.yaml"
    text = path.read_text(encoding="utf-8")
    assert "preconditioner: jacobi" in text
    path.write_text(
        text.replace("preconditioner: jacobi", "preconditioner: amg"), encoding="utf-8"
    )

    study = Study.load(path)
    assert study.solver.preconditioner == "amg"
    result = study.solve()
    assert result.preconditioner == "amg"
    assert result.amg is not None


def test_an_unknown_preconditioner_is_rejected_when_the_case_is_loaded(case_dir):
    """M5 までは前処理だけ検証が抜けていて、ソルブ直前まで気づけなかった。"""
    path = case_dir / "case.yaml"
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace("preconditioner: jacobi", "preconditioner: ilu"), encoding="utf-8"
    )
    with pytest.raises(gfem.GfemError) as excinfo:
        Study.load(path)
    message = str(excinfo.value)
    assert "ilu" in message
    assert "amg" in message


def test_the_sample_case_documents_the_choice(repo_root):
    text = (repo_root / "tests" / "models" / "slab_3layer.yaml").read_text(encoding="utf-8")
    line = next(l for l in text.splitlines() if "preconditioner:" in l)
    assert "amg" in line, f"選べることが見えていない: {line}"
