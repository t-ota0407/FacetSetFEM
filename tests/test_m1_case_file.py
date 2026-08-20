"""ケースファイル（YAML）の読み書き（SPEC §3.3）。

Nastran ファイルは形状と領域分けだけを担い、解析条件は YAML が持つ。
Nastran ファイルは書き換えない、という分担を守れているかを確認する。
"""

from __future__ import annotations

import shutil

import pytest
import yaml

import gfem
from conftest import SLAB_AREA, SLAB_LAYERS

#: SPEC §3.3 に載っているケースファイルの形（サンプルモデルに合わせた値）
CASE_YAML = """
version: 1

mesh:
  file: slab_3layer.bdf
  length_unit: mm

materials:
  - pid: 1
    name: skin
    sigma: 0.10
  - pid: 2
    name: fat
    sigma: 0.04
  - pid: 3
    name: muscle
    sigma: 0.35

electrodes:
  - name: E1
    role: anode
    current: 2.0e-3
    distribution: uniform_current_density
    faces:
      shell_pids: [101]
      element_ids: []

  - name: E2
    role: cathode
    current: 2.0e-3
    distribution: uniform_current_density
    faces:
      shell_pids: [201]

solver:
  backend: auto
  method: cg
  preconditioner: jacobi
  gauge: mean_zero
  tolerance: 1.0e-12
  max_iterations: 50000

output:
  vtu: result.vtu
  reference_potential: cathode_mean
"""


@pytest.fixture
def case_dir(tmp_path, sample_bdf):
    """ケースファイルとメッシュを同じディレクトリに置いた作業場所。"""
    shutil.copy(sample_bdf, tmp_path / "slab_3layer.bdf")
    (tmp_path / "case.yaml").write_text(CASE_YAML, encoding="utf-8")
    return tmp_path


def test_loads_the_spec_example(case_dir):
    study = gfem.Study.load(case_dir / "case.yaml")
    assert study.length_unit == "mm"
    assert [m.pid for m in study.materials] == [1, 2, 3]
    assert study.region_names() == {1: "skin", 2: "fat", 3: "muscle"}
    assert [e.name for e in study.electrodes] == ["E1", "E2"]
    assert study.electrodes[0].role == "anode"
    assert study.electrodes[1].shell_pids == [201]
    assert study.solver.tolerance == 1.0e-12
    assert study.output.reference_potential == "cathode_mean"


def test_mesh_path_is_relative_to_the_case_file(case_dir):
    study = gfem.Study.load(case_dir / "case.yaml")
    assert study.resolved_mesh_file() == (case_dir / "slab_3layer.bdf").resolve()
    mesh = study.read_mesh()
    assert mesh.n_nodes == 936


def test_solving_from_a_case_file_writes_the_requested_output(case_dir):
    study = gfem.Study.load(case_dir / "case.yaml")
    result = study.solve()
    assert result.conservation["warnings"] == []
    # output.vtu が書かれている
    assert (case_dir / "result.vtu").exists()
    # reference_potential: cathode_mean が効いている
    assert dict(result.electrode_potentials)["E2"] == pytest.approx(0.0, abs=1e-15)


def test_case_file_and_python_api_agree(case_dir, slab_sigma):
    from_file = gfem.Study.load(case_dir / "case.yaml").solve()

    mesh = gfem.read_nastran(case_dir / "slab_3layer.bdf", length_unit="mm")
    case = gfem.Case(mesh)
    case.set_conductivity(slab_sigma)
    case.add_electrode("E1", role="anode", current=2e-3, shell_pids=[101])
    case.add_electrode("E2", role="cathode", current=2e-3, shell_pids=[201])
    case.set_solver(tolerance=1e-12)
    case.set_reference_potential("cathode_mean")
    from_api = case.solve()

    assert from_file.impedance == pytest.approx(from_api.impedance, rel=0)
    assert from_file.iterations == from_api.iterations


def test_round_trip_preserves_the_settings(case_dir, tmp_path):
    study = gfem.Study.load(case_dir / "case.yaml")
    out = tmp_path / "saved.yaml"
    study.save(out)

    written = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert written["version"] == 1
    assert written["mesh"]["length_unit"] == "mm"
    assert written["electrodes"][0]["faces"]["shell_pids"] == [101]

    again = gfem.Study.load(out)
    assert again.to_dict() == study.to_dict()


def test_nastran_file_is_never_modified(case_dir):
    """解析条件は YAML に持ち、メッシュファイルは触らない（SPEC §3.3）。"""
    bdf = case_dir / "slab_3layer.bdf"
    before = bdf.read_bytes()
    gfem.Study.load(case_dir / "case.yaml").solve()
    assert bdf.read_bytes() == before


def test_full_face_case_matches_the_analytic_solution(tmp_path, full_face_bdf):
    """ケースファイル経由でも一次元解析解に一致する（SPEC §8 M2）。"""
    shutil.copy(full_face_bdf, tmp_path / "slab.bdf")
    text = CASE_YAML.replace("slab_3layer.bdf", "slab.bdf").replace(
        "  vtu: result.vtu\n", ""
    )
    (tmp_path / "case.yaml").write_text(text, encoding="utf-8")

    result = gfem.Study.load(tmp_path / "case.yaml").solve()
    expected_r = sum(t / (s * SLAB_AREA) for _p, t, s in SLAB_LAYERS)
    assert result.impedance == pytest.approx(expected_r, rel=1e-8)


# --- エラーメッセージ -------------------------------------------------------


def _load_bad(tmp_path, text: str):
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(gfem.GfemError) as excinfo:
        gfem.Study.load(path)
    return str(excinfo.value)


def test_missing_length_unit_is_reported(tmp_path):
    message = _load_bad(tmp_path, "version: 1\nmesh:\n  file: m.bdf\n")
    assert "length_unit" in message
    assert "mm / cm / m / inch" in message
    assert "場所" in message and "対処" in message


def test_missing_mesh_section_is_reported(tmp_path):
    message = _load_bad(tmp_path, "version: 1\n")
    assert "mesh" in message


def test_unknown_version_is_reported(tmp_path):
    message = _load_bad(tmp_path, "version: 99\nmesh:\n  file: m.bdf\n  length_unit: mm\n")
    assert "version=99" in message


def test_unsupported_distribution_is_reported(tmp_path):
    message = _load_bad(
        tmp_path,
        "version: 1\n"
        "mesh: {file: m.bdf, length_unit: mm}\n"
        "electrodes:\n"
        "  - name: E1\n"
        "    role: anode\n"
        "    current: 1.0e-3\n"
        "    distribution: equal_nodal_split\n"
        "    faces: {shell_pids: [101]}\n",
    )
    assert "equal_nodal_split" in message
    assert "uniform_current_density" in message
    assert "SPEC §2.4" in message


def test_electrode_without_faces_is_reported(tmp_path):
    message = _load_bad(
        tmp_path,
        "version: 1\n"
        "mesh: {file: m.bdf, length_unit: mm}\n"
        "electrodes:\n"
        "  - name: E1\n"
        "    role: anode\n"
        "    current: 1.0e-3\n"
        "    faces: {}\n",
    )
    assert "面が指定されていません" in message


def test_pin_gauge_is_refused(tmp_path):
    """1 節点ピン留めは採用しない（SPEC §2.2）。"""
    message = _load_bad(
        tmp_path,
        "version: 1\nmesh: {file: m.bdf, length_unit: mm}\nsolver: {gauge: pin_node}\n",
    )
    assert "pin_node" in message
    assert "SPEC §2.2" in message


def test_cuda_backend_says_when_it_arrives(tmp_path):
    message = _load_bad(
        tmp_path,
        "version: 1\nmesh: {file: m.bdf, length_unit: mm}\nsolver: {backend: cuda}\n",
    )
    assert "cuda" in message
    assert "M5" in message


def test_broken_yaml_is_reported(tmp_path):
    message = _load_bad(tmp_path, "version: 1\n  mesh: [unbalanced\n")
    assert "YAML" in message


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(gfem.GfemError) as excinfo:
        gfem.Study.load(tmp_path / "does_not_exist.yaml")
    assert "開けません" in str(excinfo.value)
