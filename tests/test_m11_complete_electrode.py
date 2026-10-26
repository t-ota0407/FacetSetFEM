"""完全電極モデル（SPEC §2.5 / §8 M11）の Python API・ケースファイル・パラメータスタディ・GUI。

物理の検証（厳密解・極限・保存則・再現性・正方形電極の解析解）は Rust の
`crates/facetsetfem-core/tests/complete_electrode.rs` にある。ここは Python から使う経路を見る。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import facetsetfem
from facetsetfem.gui.state import FIELD_ARRAYS, Session, field_by_key

from conftest import SLAB_AREA, SLAB_LAYERS

CURRENT = 2.0e-3
Z = 0.01  # Ω·m²


def _body_resistance() -> float:
    """上下面が丸ごと電極の 3 層スラブの抵抗（直列、SPEC §8 M2 の 2）。"""
    return sum(t / (sigma * SLAB_AREA) for _pid, t, sigma in SLAB_LAYERS)


def _full_face_case(full_face_bdf, slab_sigma, anode=None, cathode=None) -> facetsetfem.Case:
    mesh = facetsetfem.read_nastran(full_face_bdf, length_unit="mm")
    case = facetsetfem.Case(mesh)
    case.set_conductivity(slab_sigma)
    for name, role, pid, z in (("E1", "anode", 101, anode), ("E2", "cathode", 201, cathode)):
        if z is None:
            case.add_electrode(name, role=role, current=CURRENT, shell_pids=[pid])
        else:
            case.add_electrode(
                name, role=role, current=CURRENT, shell_pids=[pid],
                distribution="complete_electrode", contact_impedance=z,
            )
    case.set_solver(tolerance=1e-12)
    return case


def test_the_choices_come_from_the_core():
    assert facetsetfem.distribution_choices() == ["uniform_current_density", "complete_electrode"]


def test_a_full_face_slab_adds_the_contact_resistance(full_face_bdf, slab_sigma):
    # 1 次元の直列抵抗: R = R_体 + z/A + z/A（1 次要素で厳密。SPEC §2.5）
    result = _full_face_case(full_face_bdf, slab_sigma, anode=Z, cathode=Z).solve()
    expected = _body_resistance() + 2 * Z / SLAB_AREA
    assert result.impedance == pytest.approx(expected, rel=1e-9)
    assert result.n_solves == 1
    electrodes = result.electrodes
    assert [e["distribution"] for e in electrodes] == ["complete_electrode"] * 2
    for e in electrodes:
        assert e["contact_impedance"] == Z
        assert e["peak_current_density"] == pytest.approx(CURRENT / SLAB_AREA, rel=1e-8)
        assert abs(e["current_from_field"]) == pytest.approx(CURRENT, rel=1e-9)
    # 板の電位と皮膚の平均電位の差は I z / A
    anode = electrodes[0]
    assert anode["potential"] - anode["mean_surface_potential"] == pytest.approx(
        CURRENT * Z / SLAB_AREA, rel=1e-8
    )
    c = result.conservation
    assert c["contact_power"] == pytest.approx(CURRENT**2 * 2 * Z / SLAB_AREA, rel=1e-8)
    assert c["electrode_current_error"] < 1e-12
    assert not c["warnings"]


def test_uniform_electrodes_are_unchanged(full_face_bdf, slab_sigma):
    result = _full_face_case(full_face_bdf, slab_sigma).solve()
    assert result.impedance == pytest.approx(_body_resistance(), rel=1e-9)
    assert result.conservation["contact_power"] == 0.0
    assert [e["contact_impedance"] for e in result.electrodes] == [None, None]


def test_the_electrode_current_density_is_a_nodal_array(full_face_bdf, slab_sigma):
    result = _full_face_case(full_face_bdf, slab_sigma, anode=Z, cathode=None).solve()
    j = result.electrode_current_density
    assert j.shape == (result.mesh.n_nodes,)
    # 陽極（流入）は正、陰極は負、電極の外は 0
    assert j.max() == pytest.approx(CURRENT / SLAB_AREA, rel=1e-8)
    assert j.min() == pytest.approx(-CURRENT / SLAB_AREA, rel=1e-12)
    assert np.count_nonzero(j == 0.0) > 0


def test_the_vtu_carries_the_electrode_current_density(full_face_bdf, slab_sigma, tmp_path):
    pv = pytest.importorskip("pyvista")
    result = _full_face_case(full_face_bdf, slab_sigma, anode=Z, cathode=Z).solve()
    path = tmp_path / "cem.vtu"
    result.write_vtu(path)
    grid = pv.read(path)
    assert "electrode current density [A/m2]" in grid.point_data
    assert grid.point_data["electrode current density [A/m2]"] == pytest.approx(
        result.electrode_current_density
    )


def test_the_summary_json_records_the_model(full_face_bdf, slab_sigma):
    result = _full_face_case(full_face_bdf, slab_sigma, anode=Z, cathode=Z).solve()
    summary = json.loads(result.summary_json())
    e1 = summary["electrodes"][0]
    assert e1["distribution"] == "complete_electrode"
    assert e1["contact_impedance_ohm_m2"] == Z
    assert "electrode_potential_V" in e1
    assert summary["solver"]["gauge"] == "none"
    assert summary["solver"]["n_solves"] == 1
    assert summary["conservation"]["contact_power_W"] > 0


def test_a_contact_impedance_without_the_model_is_rejected(full_face_bdf, slab_sigma):
    case = _full_face_case(full_face_bdf, slab_sigma)
    with pytest.raises(facetsetfem.FacetSetFemError, match="complete_electrode"):
        case.add_electrode("E3", role="anode", current=CURRENT, shell_pids=[101], contact_impedance=Z)
    with pytest.raises(facetsetfem.FacetSetFemError, match="接触インピーダンス"):
        case.add_electrode(
            "E3", role="anode", current=CURRENT, shell_pids=[101], distribution="complete_electrode"
        )


# --- ケースファイル -----------------------------------------------------------


def _case_yaml(electrode_lines: str) -> str:
    return (
        "version: 1\n"
        "mesh: {file: m.bdf, length_unit: mm}\n"
        "electrodes:\n"
        "  - name: E1\n"
        "    role: anode\n"
        "    current: 1.0e-3\n" + electrode_lines + "    faces: {shell_pids: [101]}\n"
    )


def _load(tmp_path, text: str) -> facetsetfem.Study:
    path = tmp_path / "case.yaml"
    path.write_text(text, encoding="utf-8")
    return facetsetfem.Study.load(path)


def test_the_case_file_round_trips_the_model(tmp_path):
    study = _load(
        tmp_path, _case_yaml("    distribution: complete_electrode\n    contact_impedance: 0.02\n")
    )
    e = study.electrodes[0]
    assert e.is_complete and e.contact_impedance == 0.02
    again = facetsetfem.Study.from_dict(__import__("yaml").safe_load(study.to_yaml()))
    assert again.electrodes[0].contact_impedance == 0.02
    assert again.electrodes[0].distribution == "complete_electrode"


def test_a_remembered_contact_impedance_is_not_written_for_uniform(tmp_path):
    # GUI で一様電流密度に戻しても値は覚えておくが、ケースファイルには書かない
    study = _load(tmp_path, _case_yaml(""))
    e = study.electrodes[0]
    e.contact_impedance = 0.05
    assert "contact_impedance" not in e.to_dict()
    e.distribution = "complete_electrode"
    assert e.to_dict()["contact_impedance"] == 0.05


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        ("    distribution: complete_electrode\n", "contact_impedance が必要"),
        ("    contact_impedance: 0.01\n", "distribution が uniform_current_density"),
        ("    distribution: complete_electrode\n    contact_impedance: 0\n", "正の有限値"),
        ("    distribution: complete_electrode\n    contact_impedance: abc\n", "数値として読めません"),
        ("    distribution: equipotential\n", "complete_electrode"),
    ],
)
def test_bad_case_files_are_reported(tmp_path, lines, expected):
    with pytest.raises(facetsetfem.FacetSetFemError) as info:
        _load(tmp_path, _case_yaml(lines))
    message = str(info.value)
    assert expected in message
    assert "electrodes[0]" in message, "どこが悪いか書いていない"


# --- パラメータスタディと合わせ込み ----------------------------------------


def _full_face_study(full_face_bdf, slab_sigma) -> facetsetfem.Study:
    study = facetsetfem.Study(mesh_file=full_face_bdf, length_unit="mm")
    study.materials = [facetsetfem.Material(pid=pid, sigma=s) for pid, s in slab_sigma.items()]
    study.electrodes = [
        facetsetfem.ElectrodeSpec("E1", "anode", CURRENT, shell_pids=[101]),
        facetsetfem.ElectrodeSpec("E2", "cathode", CURRENT, shell_pids=[201]),
    ]
    study.solver.tolerance = 1e-12
    return study


def test_sweeping_the_contact_impedance_makes_the_electrodes_complete(full_face_bdf, slab_sigma):
    study = _full_face_study(full_face_bdf, slab_sigma)
    table = facetsetfem.sweep(study, {"contact_impedance": [0.01, 0.02]})
    body = _body_resistance()
    for row in table.rows:
        z = row["contact_impedance"]
        assert row["impedance_ohm"] == pytest.approx(body + 2 * z / SLAB_AREA, rel=1e-9)
        assert row["contact_j_max_A_per_m2"] == pytest.approx(CURRENT / SLAB_AREA, rel=1e-8)
    # 元の Study は書き換えない
    assert not study.electrodes[0].is_complete


def test_one_electrode_can_be_swept_alone(full_face_bdf, slab_sigma):
    study = _full_face_study(full_face_bdf, slab_sigma)
    table = facetsetfem.sweep(study, {"electrode.E1.contact_impedance": [0.01]})
    assert table.rows[0]["impedance_ohm"] == pytest.approx(
        _body_resistance() + 0.01 / SLAB_AREA, rel=1e-9
    )


def test_switching_to_the_model_without_a_value_is_reported(full_face_bdf, slab_sigma):
    study = _full_face_study(full_face_bdf, slab_sigma)
    with pytest.raises(facetsetfem.FacetSetFemError, match="接触インピーダンスがありません"):
        facetsetfem.sweep(study, {"electrode.E1.distribution": ["complete_electrode"]})


def test_the_contact_impedance_can_be_fitted_to_a_measured_resistance(full_face_bdf, slab_sigma):
    study = _full_face_study(full_face_bdf, slab_sigma)
    z_true = 0.0123
    measured = _body_resistance() + 2 * z_true / SLAB_AREA
    seen = []
    fit = facetsetfem.fit_contact_impedance(study, measured, progress=lambda i, z, r: seen.append(i))
    assert fit.contact_impedance == pytest.approx(z_true, rel=1e-3)
    assert fit.impedance == pytest.approx(measured, rel=1e-4)
    assert len(fit.evaluations) <= 4, fit.evaluations
    assert seen == list(range(1, len(fit.evaluations) + 1))
    # 合わせた値を入れた写しが返り、元は書き換えない
    assert all(e.contact_impedance == fit.contact_impedance for e in fit.study.electrodes)
    assert study.electrodes[0].contact_impedance is None


def test_a_measurement_below_the_body_resistance_is_reported(full_face_bdf, slab_sigma):
    study = _full_face_study(full_face_bdf, slab_sigma)
    with pytest.raises(facetsetfem.FacetSetFemError, match="実測"):
        facetsetfem.fit_contact_impedance(study, 0.5 * _body_resistance())


# --- GUI -------------------------------------------------------------------


def test_the_session_names_a_contact_impedance_change(sample_bdf, slab_sigma):
    session = Session()
    session.open_mesh(sample_bdf, "mm")
    for pid, sigma in slab_sigma.items():
        session.set_conductivity(pid, sigma)
    session.add_electrode("E1", role="anode", current=CURRENT, shell_pids=[101])
    session.add_electrode("E2", role="cathode", current=CURRENT, shell_pids=[201])
    for name in ("E1", "E2"):
        spec = session.electrode(name)
        spec.distribution = "complete_electrode"
        spec.contact_impedance = Z
    session.solve()
    assert session.changes_since_solve() == []
    session.electrode("E1").contact_impedance = 2 * Z
    assert session.changes_since_solve() == ["電極 E1 の接触インピーダンス"]
    summary = session.result_summary()
    assert "完全電極モデル" in summary
    assert "接触面での消費電力" in summary
    # 皮膚へ流れ込む電流密度を 3D ビューに出せる
    field = field_by_key("electrode current density [A/m2]")
    assert field is not None and field.location == "point"
    values = FIELD_ARRAYS[field.key](session.solution)
    assert values.max() > 0 > values.min()


@pytest.fixture
def window(main_window, sample_bdf, slab_sigma):
    w = main_window
    w.open_mesh(sample_bdf, "mm")
    for pid, sigma in slab_sigma.items():
        w.session.set_conductivity(pid, sigma)
    w.session.add_electrode("E1", "anode", CURRENT, [101])
    w.session.add_electrode("E2", "cathode", CURRENT, [201])
    w.refresh_tree()
    return w


def test_the_electrode_page_switches_the_model(window):
    page = window.page_electrode
    page.bind("E1")
    spec = window.session.electrode("E1")
    assert page.model_combo.currentData() == "uniform_current_density"
    assert not page.contact_spin.isEnabled()

    page.model_combo.setCurrentIndex(page.model_combo.findData("complete_electrode"))
    assert page.contact_spin.isEnabled()
    assert spec.is_complete
    # 初めて切り替えたときは 100 Ω·cm² = 0.01 Ω·m²（入力欄を埋めるための仮の値）
    assert spec.contact_impedance == pytest.approx(0.01)
    page.contact_spin.setValue(250.0)
    assert spec.contact_impedance == pytest.approx(0.025)

    # 一様電流密度に戻しても値は覚えておき、もう一度切り替えると戻ってくる
    page.model_combo.setCurrentIndex(page.model_combo.findData("uniform_current_density"))
    assert not spec.is_complete and spec.contact_impedance == pytest.approx(0.025)
    page.bind("E1")
    assert page.contact_spin.value() == pytest.approx(250.0)
    # 元に戻す（MainWindow はテスト間で共有している）
    page.model_combo.setCurrentIndex(page.model_combo.findData("uniform_current_density"))
