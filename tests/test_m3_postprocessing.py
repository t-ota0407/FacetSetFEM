"""M3（ポスト処理の充実）の受け入れテスト。

SPEC §8 の M3 が満たすべきこと:

1. 領域統計 CSV（体積重みパーセンタイル。SPEC §5.2）
2. 等値面（Marching Tetrahedra）、ストリームライン、クリップ（SPEC §6.3）
3. 節点値／要素値の切り替えと、最大値評価に関する注意表示（SPEC §5.1）
4. プロビナンス JSON（SPEC §5.5）
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess

import numpy as np
import pytest
import yaml

import facetsetfem
from conftest import SLAB_AREA, SLAB_LAYERS

CURRENT = 2.0e-3


@pytest.fixture(scope="module")
def solution(sample_bdf, slab_sigma):
    """上面に 2 枚のパッチ電極を置いた 3 層スラブの解。"""
    mesh = facetsetfem.read_nastran(sample_bdf, length_unit="mm")
    case = facetsetfem.Case(mesh)
    case.set_conductivity(slab_sigma)
    for pid, name in [(1, "skin"), (2, "fat"), (3, "muscle")]:
        case.set_region_name(pid, name)
    case.add_electrode("E1", role="anode", current=CURRENT, shell_pids=[101])
    case.add_electrode("E2", role="cathode", current=CURRENT, shell_pids=[201])
    case.set_solver(tolerance=1e-12)
    return mesh, case.solve()


# --- 領域統計（SPEC §5.2）--------------------------------------------------


def test_region_stats_cover_every_region(solution):
    _mesh, result = solution
    stats = result.region_stats()
    assert [s["pid"] for s in stats] == [1, 2, 3]
    assert [s["name"] for s in stats] == ["skin", "fat", "muscle"]

    for s, (_pid, thickness, sigma) in zip(stats, SLAB_LAYERS):
        assert s["sigma"] == pytest.approx(sigma)
        assert s["volume"] == pytest.approx(SLAB_AREA * thickness, rel=1e-12)
        for key in ("max", "mean", "median", "p95", "p99"):
            assert s["j_magnitude"][key] > 0.0
            assert s["e_magnitude"][key] > 0.0
        # |E| = |J| / σ が要素ごとに成り立つので、最大値も比例する
        assert s["e_magnitude"]["max"] == pytest.approx(
            s["j_magnitude"]["max"] / sigma, rel=1e-12
        )


def test_quantiles_are_ordered_and_bounded(solution):
    _mesh, result = solution
    for s in result.region_stats():
        j = s["j_magnitude"]
        assert j["median"] <= j["p95"] <= j["p99"] <= j["max"]
        assert j["mean"] <= j["max"]


def test_region_powers_sum_to_the_total(solution):
    _mesh, result = solution
    total = sum(s["power"] for s in result.region_stats())
    assert total == pytest.approx(result.power, rel=1e-9)
    # 消費電力 = I²R
    assert total == pytest.approx(CURRENT**2 * result.impedance, rel=1e-8)


def test_hotspot_identifies_a_real_element(solution):
    mesh, result = solution
    (lo, hi) = mesh.bounding_box()
    for s in result.region_stats():
        spot = s["hotspot"]
        assert spot["element_id"] > 0
        centroid = np.array(spot["centroid"])
        assert np.all(centroid > np.array(lo))
        assert np.all(centroid < np.array(hi))
        # ホットスポットの値はその領域の最大値
        assert spot["j_magnitude"] == pytest.approx(s["j_magnitude"]["max"])


def test_region_stats_csv_can_be_read_by_a_csv_reader(solution, tmp_path):
    _mesh, result = solution
    path = tmp_path / "stats.csv"
    result.write_region_stats_csv(path)

    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[0]["pid"] == "1"
    assert rows[0]["name"] == "skin"
    assert float(rows[2]["volume_m3"]) == pytest.approx(SLAB_AREA * 0.014, rel=1e-9)
    # 単位が列名に入っている（pandas でそのまま扱えるように）
    assert "J_p99_A_per_m2" in rows[0]
    assert "hotspot_x_m" in rows[0]


def test_convergence_csv_matches_the_history(solution, tmp_path):
    _mesh, result = solution
    path = tmp_path / "convergence.csv"
    result.write_convergence_csv(path)
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(result.history)
    assert float(rows[-1]["relative_residual"]) == pytest.approx(result.residual)


# --- プロビナンス JSON（SPEC §5.5）------------------------------------------


def test_summary_json_records_the_provenance(solution, tmp_path):
    _mesh, result = solution
    path = tmp_path / "summary.json"
    result.write_summary_json(path, case_text="version: 1\nmesh:\n  file: slab.bdf\n")

    data = json.loads(path.read_text(encoding="utf-8"))
    provenance = data["provenance"]
    # 入力ファイルのハッシュ（SHA-256 は 64 桁の 16 進）
    assert len(provenance["mesh_sha256"]) == 64
    assert provenance["facetsetfem_version"] == facetsetfem.__version__
    assert provenance["timestamp_utc"].endswith("Z")
    # ケースファイル全文
    assert "version: 1" in provenance["case_text"]
    # コミットハッシュだけでなく「作業ツリーが汚れていたか」も残す
    assert "git_dirty" in provenance


def test_the_mesh_hash_is_taken_when_the_mesh_is_read(tmp_path, sample_bdf, slab_sigma):
    """サマリのメッシュのハッシュは、解いたメッシュを**読んだときの**もの（SPEC §5.5）。

    以前はサマリを書く時点でファイルを読み直していたので、解いた後にファイルが
    書き換えられると、解いていないメッシュのハッシュが残った。
    """
    bdf = tmp_path / "slab.bdf"
    shutil.copy(sample_bdf, bdf)
    original = hashlib.sha256(bdf.read_bytes()).hexdigest()

    mesh = facetsetfem.read_nastran(bdf, length_unit="mm")
    assert mesh.report.sha256 == original  # sha256sum と同じ値
    case = facetsetfem.Case(mesh)
    case.set_conductivity(slab_sigma)
    case.add_electrode("E1", role="anode", current=CURRENT, shell_pids=[101])
    case.add_electrode("E2", role="cathode", current=CURRENT, shell_pids=[201])
    result = case.solve()

    bdf.write_text(bdf.read_text(encoding="utf-8") + "$ 解いた後の編集\n", encoding="utf-8")
    data = json.loads(result.summary_json())
    assert data["provenance"]["mesh_sha256"] == original


def test_the_summary_records_the_conditions_that_were_solved(tmp_path, sample_bdf):
    """解析条件は、ディスク上のケースファイルではなく**解いた条件そのもの**を残す。

    以前はファイルを読み直していたので、読み込んだ後にスクリプトや GUI で
    条件を変えて解くと、サマリの条件が実際と食い違った。
    """
    shutil.copy(sample_bdf, tmp_path / "slab.bdf")
    case_yaml = tmp_path / "case.yaml"
    case_yaml.write_text(
        "version: 1\n"
        "mesh: {file: slab.bdf, length_unit: mm}\n"
        "materials:\n"
        "  - {pid: 1, sigma: 0.10}\n"
        "  - {pid: 2, sigma: 0.04}\n"
        "  - {pid: 3, sigma: 0.35}\n"
        "electrodes:\n"
        "  - {name: E1, role: anode, current: 2.0e-3, faces: {shell_pids: [101]}}\n"
        "  - {name: E2, role: cathode, current: 2.0e-3, faces: {shell_pids: [201]}}\n"
        "output: {summary_json: summary.json}\n",
        encoding="utf-8",
    )
    study = facetsetfem.Study.load(case_yaml)
    study.materials[2].sigma = 0.50  # 読み込んだ後に条件を変えて解く
    study.solve()

    provenance = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))[
        "provenance"
    ]
    solved = facetsetfem.Study.from_dict(yaml.safe_load(provenance["case_text"]))
    assert {m.pid: m.sigma for m in solved.materials} == {1: 0.10, 2: 0.04, 3: 0.50}
    assert provenance["case_file"] == str(case_yaml)


def test_the_git_state_describes_this_build(solution):
    """サマリの Git の状態が、この拡張モジュールを作ったソースと一致すること。

    以前は build.rs がソースの編集で走り直さず、編集済みのソースから作った
    バイナリが `git_dirty: false` を名乗っていた。開発環境（作業ツリーから
    読み込んでいる）でだけ確かめられる。
    """
    _mesh, result = solution
    provenance = json.loads(result.summary_json())["provenance"]
    tree = provenance.get("source_tree")
    if tree is None:
        pytest.skip("作業ツリーの外で動いている（wheel で入れた環境など）")

    assert tree["same_commit_as_build"] is True, (
        "拡張モジュールのビルド後にコミットが進んでいます。maturin develop してください"
    )
    # Python 拡張の git_dirty は facetsetfem-core だけでなく 3 クレート全部のソースを見る
    scope = provenance["git_dirty_scope"]
    assert "crates/facetsetfem-py/src" in scope and "crates/facetsetfem-core/src" in scope
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *scope],
        cwd=tree["path"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout
    assert provenance["git_dirty"] == bool(status.strip()), (
        "埋め込まれた git_dirty が作業ツリーと食い違います"
        "（ソースを編集した後なら maturin develop してください）:\n" + status
    )


def test_summary_json_records_the_run(solution):
    _mesh, result = solution
    data = json.loads(result.summary_json())
    assert data["solver"]["backend"] == "cpu_serial"
    assert data["solver"]["gauge"] == "mean_zero"
    assert data["solver"]["converged"] is True
    assert data["conservation"]["passed"] is True
    assert data["results"]["impedance_ohm"] == pytest.approx(result.impedance)
    assert len(data["region_stats"]) == 3
    assert len(data["electrodes"]) == 2
    assert set(data["timings"]) == {
        "prepare_s",
        "assemble_s",
        "solve_s",
        "post_s",
        "total_s",
    }


# --- 等値面（SPEC §6.3）-----------------------------------------------------


def test_iso_surface_lies_on_the_requested_level(solution):
    _mesh, result = solution
    lo, hi = result.value_range("j_magnitude")
    level = lo + 0.5 * (hi - lo)
    points, triangles, scalars = result.iso_surface("j_magnitude", level)

    assert points.shape[1] == 3
    assert triangles.shape[1] == 3
    assert len(scalars) == len(points)
    assert triangles.max() < len(points)
    # 頂点の値は全てしきい値に等しい（線形補間で作っているので厳密）
    assert np.allclose(scalars, level, rtol=1e-9)


def test_iso_surface_of_the_potential(solution):
    _mesh, result = solution
    lo, hi = result.value_range("phi")
    points, triangles, _ = result.iso_surface("phi", 0.5 * (lo + hi))
    assert len(triangles) > 0
    # 等値面はモデルの中に収まる
    assert points[:, 2].min() >= -1e-12
    assert points[:, 2].max() <= 0.020 + 1e-12


def test_iso_surface_outside_the_range_is_empty(solution):
    _mesh, result = solution
    _lo, hi = result.value_range("phi")
    _points, triangles, _ = result.iso_surface("phi", hi * 10.0 + 1.0)
    assert len(triangles) == 0


def test_unknown_quantity_lists_the_choices(solution):
    _mesh, result = solution
    with pytest.raises(facetsetfem.FacetSetFemError) as excinfo:
        result.iso_surface("temperature", 1.0)
    assert "j_magnitude" in str(excinfo.value)


# --- ストリームライン（SPEC §6.3）-------------------------------------------


def test_streamlines_follow_the_current(solution):
    """陽極から潜って陰極へ戻る、という経皮電気刺激の典型的な経路になる。

    2 枚の電極が同じ面（上面）にあるので、流線はいったん組織内へ潜り、
    横に走ってから陰極で表面に戻る。
    """
    mesh, result = solution
    # 陽極パッチ（x = 5〜20 mm）のすぐ下から流す
    seeds = [[x, 0.020, 0.0195] for x in (0.008, 0.012, 0.016)]
    lines = result.streamlines(seeds)
    assert len(lines) == len(seeds)

    (lo, hi) = mesh.bounding_box()
    for line in lines:
        assert line.shape[1] == 3
        assert len(line) >= 2
        # モデルの外に出ない
        assert np.all(line >= np.array(lo) - 1e-9)
        assert np.all(line <= np.array(hi) + 1e-9)
        # いったん始点より深く潜る
        assert line[:, 2].min() < line[0][2] - 1e-4, "組織内に潜っていません"
        # 陰極パッチ（x = 40〜55 mm）の側で表面に戻る
        assert line[-1][2] == pytest.approx(0.020, abs=1e-9), "上面に戻っていません"
        assert 0.040 <= line[-1][0] <= 0.055, f"終点 x = {line[-1][0]}"


def test_backward_streamlines_go_upstream(solution):
    _mesh, result = solution
    seeds = [[0.030, 0.020, 0.010]]
    forward = result.streamlines(seeds, direction="forward")[0]
    backward = result.streamlines(seeds, direction="backward")[0]
    # 同じ点から出て逆向きに進む
    assert np.allclose(forward[0], backward[0])
    assert not np.allclose(forward[-1], backward[-1])


def test_seed_outside_the_mesh_is_skipped(solution):
    _mesh, result = solution
    lines = result.streamlines([[0.030, 0.020, 0.010], [10.0, 10.0, 10.0]])
    assert len(lines) == 1


def test_unknown_direction_is_rejected(solution):
    _mesh, result = solution
    with pytest.raises(facetsetfem.FacetSetFemError) as excinfo:
        result.streamlines([[0.03, 0.02, 0.01]], direction="sideways")
    assert "forward / backward" in str(excinfo.value)


# --- ケースファイルからの一括出力 -------------------------------------------


def test_case_file_writes_all_requested_outputs(tmp_path, sample_bdf):
    import shutil

    shutil.copy(sample_bdf, tmp_path / "slab.bdf")
    (tmp_path / "case.yaml").write_text(
        """
version: 1
mesh: {file: slab.bdf, length_unit: mm}
materials:
  - {pid: 1, name: skin, sigma: 0.10}
  - {pid: 2, name: fat, sigma: 0.04}
  - {pid: 3, name: muscle, sigma: 0.35}
electrodes:
  - {name: E1, role: anode, current: 2.0e-3, faces: {shell_pids: [101]}}
  - {name: E2, role: cathode, current: 2.0e-3, faces: {shell_pids: [201]}}
output:
  vtu: result.vtu
  region_stats: stats.csv
  convergence: convergence.csv
  summary_json: summary.json
""",
        encoding="utf-8",
    )

    study = facetsetfem.Study.load(tmp_path / "case.yaml")
    study.solve()
    for name in ("result.vtu", "stats.csv", "convergence.csv", "summary.json"):
        assert (tmp_path / name).exists(), f"{name} が書き出されていません"

    # 領域名がケースファイルから統計に伝わっている
    data = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert [r["name"] for r in data["region_stats"]] == ["skin", "fat", "muscle"]
    # ケースファイルの全文がプロビナンスに残っている
    assert "region_stats: stats.csv" in data["provenance"]["case_text"]
