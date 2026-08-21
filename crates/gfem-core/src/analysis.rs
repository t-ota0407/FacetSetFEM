//! 解析の実行（SPEC §4.1 の全体の流れ）。
//!
//! ```text
//! Nastran 読込
//!   → トポロジ構築（外表面抽出）
//!   → 検証（ケースの整合性・電流保存）
//!   → CSR 構造の構築
//!   → 剛性のアセンブリ
//!   → 電極荷重ベクトルの構築
//!   → 射影付き前処理付き CG
//!   → ポスト処理（E, J, 保存則チェック）
//! ```
//!
//! 読み込みは [`crate::io::nastran`]、出力は [`write_analysis_vtu`]。

use crate::assemble::{LinearSystem, assemble, load::ElectrodeGeometry};
use crate::case::{Backend, Case, ElectrodeRole, ReferencePotential};
use crate::error::{ErrorKind, GfemError, Result};
use crate::io::json::Json;
use crate::io::vtu::{DataArray, VtuData, VtuOptions, write_vtu};
use crate::mesh::Mesh;
use crate::mesh::topology::{Surface, extract_surface};
use crate::post::{
    ConservationReport, FieldResult, RegionStats, check_conservation, compute_fields,
    pad_for_shells, region_stats, write_convergence_csv, write_region_stats_csv,
};
use crate::provenance::Provenance;
use crate::solver::backend::{CpuParallel, CpuSerial, LinearBackend};
use crate::solver::cg::{self, CgReport, ProgressFn};
use std::path::Path;
use std::time::Instant;

/// 各段階の所要時間 [秒]。出力 JSON のプロビナンス（SPEC §5.5）に記録する。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Timings {
    /// 外表面抽出と検証
    pub prepare: f64,
    /// CSR 構築 + 剛性 + 荷重
    pub assemble: f64,
    /// CG
    pub solve: f64,
    /// ポスト処理
    pub post: f64,
    /// 合計
    pub total: f64,
}

/// 解析の結果一式。
#[derive(Debug, Clone)]
pub struct Analysis {
    /// 節点電位 φ [V]。表示基準（[`ReferencePotential`]）を適用済み
    pub potential: Vec<f64>,
    /// 要素ごとの E, J, 電力密度など
    pub fields: FieldResult,
    /// 保存則チェックと電極間インピーダンス
    pub conservation: ConservationReport,
    /// CG の収束状況
    pub solver: CgReport,
    /// 電極ごとの幾何情報
    pub electrodes: Vec<ElectrodeGeometry>,
    /// 外表面
    pub surface: Surface,
    /// 実行中に出た警告（ケース検証 + 保存則チェック）
    pub warnings: Vec<String>,
    /// 所要時間
    pub timings: Timings,
}

impl Analysis {
    /// 電極間の等価抵抗 [Ω]（SPEC §5.3）。
    pub fn impedance(&self) -> f64 {
        self.conservation.impedance
    }

    /// 電位の最小値と最大値 [V]。
    pub fn potential_range(&self) -> (f64, f64) {
        self.potential
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            })
    }

    /// 領域（PID）ごとの統計（SPEC §5.2）。
    pub fn region_stats(&self, mesh: &Mesh, case: &Case) -> Result<Vec<RegionStats>> {
        region_stats(mesh, case, &self.fields)
    }

    /// 領域統計を CSV に書き出す（SPEC §5.5）。
    pub fn write_region_stats_csv(
        &self,
        path: impl AsRef<Path>,
        mesh: &Mesh,
        case: &Case,
    ) -> Result<()> {
        write_region_stats_csv(path, &self.region_stats(mesh, case)?)
    }

    /// 収束履歴を CSV に書き出す（SPEC §5.5）。
    pub fn write_convergence_csv(&self, path: impl AsRef<Path>) -> Result<()> {
        write_convergence_csv(path, &self.solver.history)
    }

    /// 実行サマリを JSON にする（SPEC §5.5）。
    ///
    /// プロビナンス（入力ハッシュ・Git の状態・実行時刻）とソルバの挙動、
    /// 保存則チェックの結果、領域統計をひとまとめにする。
    /// 「この図はどの入力から出たか」を後から追えるようにするための記録。
    pub fn summary_json(&self, mesh: &Mesh, case: &Case, provenance: &Provenance) -> Result<Json> {
        let mut root = Json::object();
        root.set("provenance", provenance.to_json());

        let mut mesh_json = Json::object();
        mesh_json
            .set("length_unit", mesh.length_unit.symbol())
            .set("n_nodes", mesh.nodes.len())
            .set("n_solid_elements", mesh.n_solid_elements())
            .set("n_shell_elements", mesh.n_shell_elements());
        if let Some((lo, hi)) = mesh.bounding_box() {
            mesh_json
                .set("bounding_box_min_m", lo.to_vec())
                .set("bounding_box_max_m", hi.to_vec());
        }
        root.set("mesh", mesh_json);

        let materials: Vec<Json> = case
            .conductivity
            .iter()
            .map(|(&pid, sigma)| {
                let mut m = Json::object();
                m.set("pid", pid)
                    .set(
                        "name",
                        case.region_names.get(&pid).cloned().unwrap_or_default(),
                    )
                    .set("sigma_S_per_m", sigma.scalar());
                m
            })
            .collect();
        root.set("materials", Json::Array(materials));

        let electrodes: Vec<Json> = self
            .electrodes
            .iter()
            .map(|e| {
                let mut j = Json::object();
                j.set("name", e.name.clone())
                    .set("role", e.role.label())
                    .set("current_A", e.signed_current)
                    .set("n_faces", e.n_faces)
                    .set("area_m2", e.area)
                    .set("current_density_A_per_m2", e.current_density)
                    .set("mean_potential_V", e.mean_potential(&self.potential));
                j
            })
            .collect();
        root.set("electrodes", Json::Array(electrodes));

        let mut solver = Json::object();
        solver
            .set("backend", self.solver.backend)
            .set("backend_setting", case.solver.backend.label())
            .set("method", "projected_preconditioned_cg")
            .set("gauge", "mean_zero")
            .set(
                "preconditioner",
                match case.solver.preconditioner {
                    crate::case::Preconditioner::Jacobi => "jacobi",
                    _ => "none",
                },
            )
            .set("tolerance", case.solver.tolerance)
            .set("max_iterations", case.solver.max_iterations)
            .set("iterations", self.solver.iterations)
            .set("relative_residual", self.solver.relative_residual)
            .set("converged", self.solver.converged);
        root.set("solver", solver);

        let c = &self.conservation;
        let mut checks = Json::object();
        checks
            .set("current_balance", c.current_balance)
            .set("power_from_field_W", c.power_from_field)
            .set("power_from_load_W", c.power_from_load)
            .set("energy_relative_error", c.energy_relative_error)
            .set("discrete_residual", c.discrete_residual)
            .set("passed", c.is_clean());
        root.set("conservation", checks);

        let (phi_min, phi_max) = self.potential_range();
        let mut results = Json::object();
        results
            .set("voltage_V", c.voltage)
            .set("impedance_ohm", c.impedance)
            .set("total_current_A", c.total_current)
            .set("power_W", c.power_from_field)
            .set("potential_min_V", phi_min)
            .set("potential_max_V", phi_max)
            .set(
                "j_magnitude_max_A_per_m2",
                self.fields
                    .j_magnitude
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max),
            )
            .set(
                "e_magnitude_max_V_per_m",
                self.fields
                    .e_magnitude
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max),
            );
        root.set("results", results);

        let regions: Vec<Json> = self
            .region_stats(mesh, case)?
            .iter()
            .map(region_stats_json)
            .collect();
        root.set("region_stats", Json::Array(regions));

        let t = &self.timings;
        let mut timings = Json::object();
        timings
            .set("prepare_s", t.prepare)
            .set("assemble_s", t.assemble)
            .set("solve_s", t.solve)
            .set("post_s", t.post)
            .set("total_s", t.total);
        root.set("timings", timings);

        root.set("warnings", self.warnings.clone());
        Ok(root)
    }

    /// 実行サマリ JSON をファイルに書き出す。
    pub fn write_summary_json(
        &self,
        path: impl AsRef<Path>,
        mesh: &Mesh,
        case: &Case,
        provenance: &Provenance,
    ) -> Result<()> {
        let path = path.as_ref();
        let text = self
            .summary_json(mesh, case, provenance)?
            .to_pretty_string();
        std::fs::write(path, text).map_err(|e| {
            GfemError::io(path, "実行サマリ JSON を書き出せません", e)
                .with_hint("出力先のディレクトリが存在し、書き込み権限があるかを確認してください")
        })
    }
}

fn region_stats_json(s: &RegionStats) -> Json {
    let quantity = |q: &crate::post::QuantityStats| {
        let mut j = Json::object();
        j.set("max", q.max)
            .set("mean", q.mean)
            .set("median", q.median)
            .set("p95", q.p95)
            .set("p99", q.p99);
        j
    };
    let mut hotspot = Json::object();
    hotspot
        .set("element_id", s.hotspot.element_id)
        .set("centroid_m", s.hotspot.centroid.to_vec())
        .set("j_magnitude_A_per_m2", s.hotspot.j_magnitude)
        .set("e_magnitude_V_per_m", s.hotspot.e_magnitude);

    let mut j = Json::object();
    j.set("pid", s.pid)
        .set("name", s.name.clone())
        .set("sigma_S_per_m", s.sigma)
        .set("n_elements", s.n_elements)
        .set("volume_m3", s.volume)
        .set("power_W", s.power)
        .set("j_magnitude_A_per_m2", quantity(&s.current_density))
        .set("e_magnitude_V_per_m", quantity(&s.electric_field))
        .set("hotspot", hotspot);
    j
}

/// ケースで指定されたバックエンドで解く。
///
/// バックエンドを変えても**結果はビット単位で同じ**（CLAUDE.md 設計原則 1）。
/// `Auto` は規模を見て逐次と並列を切り替えるだけで、答えには影響しない。
pub fn solve(mesh: &Mesh, case: &Case, progress: Option<ProgressFn<'_>>) -> Result<Analysis> {
    match case.solver.backend.resolve(mesh.nodes.len()) {
        Backend::CpuParallel => run(&CpuParallel, mesh, case, progress, true),
        _ => run(&CpuSerial, mesh, case, progress, false),
    }
}

/// バックエンドを明示して解く（アセンブリは逐次）。
pub fn solve_with_backend<B: LinearBackend>(
    backend: &B,
    mesh: &Mesh,
    case: &Case,
    progress: Option<ProgressFn<'_>>,
) -> Result<Analysis> {
    run(backend, mesh, case, progress, false)
}

fn run<B: LinearBackend>(
    backend: &B,
    mesh: &Mesh,
    case: &Case,
    progress: Option<ProgressFn<'_>>,
    parallel_assembly: bool,
) -> Result<Analysis> {
    let started = Instant::now();

    // --- 準備 ---
    let surface = extract_surface(mesh)?;
    let mut warnings = case.validate(mesh, &surface)?;
    let prepare = started.elapsed().as_secs_f64();

    // --- アセンブリ ---
    let mark = Instant::now();
    let LinearSystem {
        matrix,
        load,
        electrodes,
    } = assemble(mesh, case, parallel_assembly)?;
    let assemble_time = mark.elapsed().as_secs_f64();

    // --- 求解 ---
    let mark = Instant::now();
    let mut potential = vec![0.0; mesh.nodes.len()];
    let report = cg::solve(
        backend,
        &matrix,
        &load,
        &mut potential,
        &case.solver,
        progress,
    )?;
    let solve_time = mark.elapsed().as_secs_f64();

    if !report.converged {
        let reason = report.failure_reason().unwrap_or_default();
        let kind = if report.cancelled {
            ErrorKind::Cancelled
        } else {
            ErrorKind::Convergence
        };
        return Err(
            GfemError::new(kind, format!("CG が収束しませんでした: {reason}")).with_hint(
                if report.cancelled {
                    "中断された時点の解は使えません。もう一度実行してください".to_string()
                } else {
                    format!(
                        "最大反復回数を増やす、収束判定を緩める（現在 {:.0e}）、\
                     または導電率のコントラストとメッシュの品質を見直してください",
                        case.solver.tolerance
                    )
                },
            ),
        );
    }

    // --- 表示基準の適用（SPEC §2.2 の「基準点の付け替え」）---
    apply_reference(mesh, &mut potential, &electrodes, case.reference)?;

    // --- ポスト処理 ---
    let mark = Instant::now();
    let fields = compute_fields(mesh, &case.conductivity, &potential)?;
    let conservation = check_conservation(&matrix, &load, &potential, &fields, &electrodes);
    warnings.extend(conservation.warnings.iter().cloned());
    let post = mark.elapsed().as_secs_f64();

    Ok(Analysis {
        potential,
        fields,
        conservation,
        solver: report,
        electrodes,
        surface,
        warnings,
        timings: Timings {
            prepare,
            assemble: assemble_time,
            solve: solve_time,
            post,
            total: started.elapsed().as_secs_f64(),
        },
    })
}

/// 電位の基準点を付け替える。
///
/// E と J は基準に依存しないので、この操作は表示のためだけのもの（SPEC §2.2）。
fn apply_reference(
    mesh: &Mesh,
    phi: &mut [f64],
    electrodes: &[ElectrodeGeometry],
    reference: ReferencePotential,
) -> Result<()> {
    let shift = match reference {
        // ソルバの射影で既に平均ゼロになっている
        ReferencePotential::ZeroMean => return Ok(()),
        ReferencePotential::CathodeMean => {
            let mut sum = 0.0;
            let mut weight = 0.0;
            for e in electrodes
                .iter()
                .filter(|e| e.role == ElectrodeRole::Cathode)
            {
                let w = e.signed_current.abs();
                sum += w * e.mean_potential(phi);
                weight += w;
            }
            if weight == 0.0 {
                return Err(GfemError::new(
                    ErrorKind::Case,
                    "基準電位に cathode_mean が指定されていますが、陰極がありません",
                ));
            }
            sum / weight
        }
        ReferencePotential::Node(id) => {
            let Some(index) = mesh.nodes.ids().iter().position(|&n| n == id) else {
                return Err(GfemError::new(
                    ErrorKind::Case,
                    format!("基準電位に指定された節点 ID {id} がメッシュにありません"),
                )
                .with_hint("存在する GRID の ID を指定してください"));
            };
            phi[index]
        }
    };
    for v in phi.iter_mut() {
        *v -= shift;
    }
    Ok(())
}

/// 解析結果を VTU として書き出す（SPEC §5.1 / §5.5）。
///
/// 節点値と要素値の**両方**を出す。節点値は体積重み平均で平滑化されているため、
/// 最大値の評価には要素値（`|E|`, `|J|`）を使うこと。
pub fn write_analysis_vtu(
    path: impl AsRef<Path>,
    mesh: &Mesh,
    analysis: &Analysis,
    opts: &VtuOptions,
) -> Result<()> {
    let fields = &analysis.fields;
    let volumes = &fields.volume;

    // 節点値（表示用に平滑化した量）
    let nodal_e = crate::post::nodal_average(mesh, &fields.e_magnitude, volumes, 1);
    let nodal_j = crate::post::nodal_average(mesh, &fields.j_magnitude, volumes, 1);

    // 要素値。VTU のセルはソリッド → シェルの順なので、シェル側は 0 で埋める
    let cell_e = pad_for_shells(mesh, &fields.electric_field, 3, 0.0);
    let cell_j = pad_for_shells(mesh, &fields.current_density, 3, 0.0);
    let cell_e_mag = pad_for_shells(mesh, &fields.e_magnitude, 1, 0.0);
    let cell_j_mag = pad_for_shells(mesh, &fields.j_magnitude, 1, 0.0);
    let cell_power = pad_for_shells(mesh, &fields.power_density, 1, 0.0);
    let pid: Vec<i64> = mesh
        .blocks()
        .flat_map(|b| b.pids().iter().map(|&p| p as i64))
        .collect();

    let data = VtuData {
        point_data: vec![
            DataArray::scalars_f64("phi [V]", &analysis.potential),
            DataArray::scalars_f64("|E| (nodal) [V/m]", &nodal_e),
            DataArray::scalars_f64("|J| (nodal) [A/m2]", &nodal_j),
        ],
        cell_data: vec![
            DataArray::vectors_f64("E [V/m]", &cell_e),
            DataArray::vectors_f64("J [A/m2]", &cell_j),
            DataArray::scalars_f64("|E| [V/m]", &cell_e_mag),
            DataArray::scalars_f64("|J| [A/m2]", &cell_j_mag),
            DataArray::scalars_f64("power density [W/m3]", &cell_power),
            DataArray::scalars_i64("PID", &pid),
        ],
    };
    write_vtu(path, mesh, &data, opts)
}
