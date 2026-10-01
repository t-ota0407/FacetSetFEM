//! 解析の実行（SPEC §4.1 の全体の流れ）。
//!
//! ```text
//! Nastran 読込
//!   → トポロジ構築（外表面抽出）
//!   → 検証（ケースの整合性・電流保存）
//!   → CSR 構造の構築
//!   → 剛性のアセンブリ
//!   → 電極荷重ベクトルの構築（CEM の電極があれば接触インピーダンスの項も。SPEC §2.5）
//!   → 射影付き前処理付き CG（CEM があれば射影なしで重ね合わせ。[`crate::contact`]）
//!   → ポスト処理（E, J, 保存則チェック）
//! ```
//!
//! 読み込みは [`crate::io::nastran`]、出力は [`write_analysis_vtu`]。

use crate::assemble::{LinearSystem, assemble, load::ElectrodeGeometry};
use crate::case::{Backend, Case, ElectrodeRole, FaceLookup, ReferencePotential};
use crate::contact::{self, ElectrodeState};
use crate::error::{ErrorKind, GfemError, Result};
use crate::io::json::Json;
use crate::io::vtu::{DataArray, VtuData, VtuOptions, write_vtu};
use crate::mesh::Mesh;
use crate::mesh::topology::extract_surface;
use crate::post::{
    ConservationReport, FieldResult, RegionStats, check_conservation_with, compute_fields,
    pad_for_shells, region_stats, write_convergence_csv, write_region_stats_csv,
};
use crate::provenance::Provenance;
use crate::solver::backend::{CpuParallel, CpuSerial, LinearBackend};
use crate::solver::cg::{CgReport, Gauge, ProgressFn};
use crate::solver::external::{BackendSolver, LinearSolver};
use std::path::Path;
use std::time::Instant;

/// 各段階の所要時間 [秒]。出力 JSON のプロビナンス（SPEC §5.5）に記録する。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Timings {
    /// 外表面抽出と検証（外表面を作り置きして [`solve_prepared`] で解いた場合は検証だけ）
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
    /// CG の収束状況。重ね合わせで何度か解いたときは、反復回数の合計・残差の最大・
    /// つないだ収束履歴（SPEC §4.3）
    pub solver: CgReport,
    /// 右辺ごとの `(反復回数, 相対残差)`（解いた順。一様電流密度だけなら 1 本）
    pub solves: Vec<(usize, f64)>,
    /// 解いた系が特異（全 Neumann）か正則（CEM の電極がある）か
    pub gauge: Gauge,
    /// 電極ごとの幾何情報（CEM の電極には板の電位 `terminal_potential` が入っている）
    pub electrodes: Vec<ElectrodeGeometry>,
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

    /// 電極ごとの電位・電流・接触電流密度の最大・接触面の消費電力（SPEC §2.5）。
    pub fn electrode_states(&self) -> Vec<ElectrodeState> {
        contact::electrode_states(&self.electrodes, &self.potential)
    }

    /// 電極面の節点ごとの、皮膚を通って流れ込む電流密度 [A/m²]（電極の外は 0。SPEC §5.5）。
    pub fn electrode_current_density(&self, mesh: &Mesh) -> Vec<f64> {
        contact::electrode_current_density(mesh, &self.electrodes, &self.potential)
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
        let stats = self.region_stats(mesh, case)?;
        Ok(self.summary_json_with_stats(mesh, case, provenance, &stats))
    }

    /// [`summary_json`](Self::summary_json) の、計算済みの領域統計を渡す版。
    ///
    /// 領域統計は大規模モデルで数百ミリ秒かかる（領域ごとのソート）ので、
    /// 表示・CSV・サマリで使い回す呼び出し側（Python の `Solution`）はこちらを使う。
    pub fn summary_json_with_stats(
        &self,
        mesh: &Mesh,
        case: &Case,
        provenance: &Provenance,
        stats: &[RegionStats],
    ) -> Json {
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

        let states = self.electrode_states();
        let electrodes: Vec<Json> = self
            .electrodes
            .iter()
            .zip(&states)
            .map(|(e, state)| {
                let mut j = Json::object();
                j.set("name", e.name.clone())
                    .set("role", e.role.label())
                    .set("distribution", e.model.label())
                    .set("current_A", e.signed_current)
                    .set("n_faces", e.n_faces)
                    .set("area_m2", e.area)
                    .set("current_density_A_per_m2", e.current_density)
                    .set("mean_potential_V", state.mean_surface_potential)
                    .set("peak_current_density_A_per_m2", state.peak_current_density);
                // 完全電極モデル（SPEC §2.5）の電極だけが持つ量
                if let Some(z) = e.contact_impedance() {
                    j.set("contact_impedance_ohm_m2", z)
                        .set("electrode_potential_V", state.potential)
                        .set("current_from_field_A", state.current)
                        .set("contact_power_W", state.contact_power);
                }
                j
            })
            .collect();
        root.set("electrodes", Json::Array(electrodes));

        let mut solver = Json::object();
        solver
            .set("backend", self.solver.backend)
            .set("backend_setting", case.solver.backend.label())
            .set(
                "method",
                match self.gauge {
                    Gauge::MeanZero => "projected_preconditioned_cg",
                    Gauge::NotNeeded => "preconditioned_cg_superposition",
                },
            )
            .set("gauge", self.gauge.label())
            .set("preconditioner", case.solver.preconditioner.label())
            .set("tolerance", case.solver.tolerance)
            .set("max_iterations", case.solver.max_iterations)
            .set("iterations", self.solver.iterations)
            .set("relative_residual", self.solver.relative_residual)
            .set("converged", self.solver.converged)
            .set("preconditioner_setup_s", self.solver.setup_seconds)
            .set("n_solves", self.solves.len());
        if self.solves.len() > 1 {
            // 重ね合わせで何度か解いた（SPEC §2.5）。右辺ごとの反復回数と残差を残す
            let solves: Vec<Json> = self
                .solves
                .iter()
                .map(|&(iterations, residual)| {
                    let mut j = Json::object();
                    j.set("iterations", iterations)
                        .set("relative_residual", residual);
                    j
                })
                .collect();
            solver.set("solves", Json::Array(solves));
        }
        if let Some(info) = &self.solver.amg {
            // 「どんな階層で解いたか」は後から結果を読み解くのに要る（SPEC §4.5 / §5.5）
            let levels: Vec<Json> = info
                .levels
                .iter()
                .map(|&(n_rows, nnz)| {
                    let mut j = Json::object();
                    j.set("n_rows", n_rows).set("nnz", nnz);
                    j
                })
                .collect();
            let mut amg = Json::object();
            amg.set("levels", Json::Array(levels))
                .set("operator_complexity", info.operator_complexity)
                .set("grid_complexity", info.grid_complexity)
                .set("direct_coarse_solve", info.direct_coarse_solve)
                .set("strength_threshold", info.strength_threshold);
            solver.set("amg", amg);
        }
        root.set("solver", solver);

        let c = &self.conservation;
        let mut checks = Json::object();
        checks
            .set("current_balance", c.current_balance)
            .set("electrode_current_error", c.electrode_current_error)
            .set("power_from_field_W", c.power_from_field)
            .set("contact_power_W", c.contact_power)
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
            .set("contact_power_W", c.contact_power)
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

        let regions: Vec<Json> = stats.iter().map(region_stats_json).collect();
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
        root
    }

    /// 実行サマリ JSON をファイルに書き出す。
    pub fn write_summary_json(
        &self,
        path: impl AsRef<Path>,
        mesh: &Mesh,
        case: &Case,
        provenance: &Provenance,
    ) -> Result<()> {
        write_json(path.as_ref(), &self.summary_json(mesh, case, provenance)?)
    }

    /// [`write_summary_json`](Self::write_summary_json) の、計算済みの領域統計を渡す版。
    pub fn write_summary_json_with_stats(
        &self,
        path: impl AsRef<Path>,
        mesh: &Mesh,
        case: &Case,
        provenance: &Provenance,
        stats: &[RegionStats],
    ) -> Result<()> {
        write_json(
            path.as_ref(),
            &self.summary_json_with_stats(mesh, case, provenance, stats),
        )
    }
}

fn write_json(path: &Path, json: &Json) -> Result<()> {
    std::fs::write(path, json.to_pretty_string()).map_err(|e| {
        GfemError::io(
            path,
            crate::tr!(
                "Cannot write the run summary JSON",
                "実行サマリ JSON を書き出せません"
            ),
            e,
        )
        .with_hint(crate::tr!(
            "Check that the output directory exists and that you have write permission",
            "出力先のディレクトリが存在し、書き込み権限があるかを確認してください"
        ))
    })
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

/// ケースで指定されたバックエンドで解く（CPU のみ）。
///
/// バックエンドを変えても**結果はビット単位で同じ**（CLAUDE.md 設計原則 1）。
/// `Auto` は規模を見て逐次と並列を切り替えるだけで、答えには影響しない。
///
/// CUDA は `gfem-core` からは見えないので、GPU で解くときは
/// [`solve_with`] に `gfem-cuda` のソルバを渡す。
pub fn solve(mesh: &Mesh, case: &Case, progress: Option<ProgressFn<'_>>) -> Result<Analysis> {
    solve_with(mesh, case, progress, None)
}

/// 外部のソルバ（`gfem-cuda` など）を差し込んで解く。
///
/// `solver` が `None` ならケースの設定どおり CPU で解く。`Some` なら
/// **そのソルバで解く**——バックエンドの選択は呼び出し側の責任になる。
/// GPU が使えるかどうかの判定とフォールバックは `gfem-cuda` 側で行い、
/// ここには「使えると決まったソルバ」だけが渡ってくる（SPEC §4.4）。
///
/// **アセンブリは常に CPU で行う。** GPU で解く場合でも行列は主記憶で組み立て、
/// できあがった CSR を VRAM へ送る。アセンブリを並列にするかは節点数だけで決める
/// （`solver.backend` の指定とは独立。GPU で解くのに組み立てが逐次では意味がない）。
pub fn solve_with(
    mesh: &Mesh,
    case: &Case,
    progress: Option<ProgressFn<'_>>,
    solver: Option<&dyn LinearSolver>,
) -> Result<Analysis> {
    let started = Instant::now();
    let surface = extract_surface(mesh)?;
    let lookup = FaceLookup::new(mesh, &surface);
    let extract = started.elapsed().as_secs_f64();
    dispatch(&lookup, case, progress, solver, extract)
}

/// 外表面と電極面の索引を作り置きして解く（同じメッシュで何度も解く場合）。
///
/// 大規模モデルでは外表面の抽出が solve の準備の大半を占める（1000 万節点で 1.8 秒。
/// SPEC §8 の「2026-09」）。GUI・Python の `Mesh` やパラメータスタディは
/// 外表面と [`crate::case::SurfaceIndex`] を 1 度だけ作り、ここへ渡す。
/// `lookup` の外表面は、そのメッシュから [`extract_surface`] で抽出したものであること。
///
/// ソルバの扱いは [`solve_with`] と同じ。
pub fn solve_prepared(
    lookup: &FaceLookup,
    case: &Case,
    progress: Option<ProgressFn<'_>>,
    solver: Option<&dyn LinearSolver>,
) -> Result<Analysis> {
    dispatch(lookup, case, progress, solver, 0.0)
}

/// バックエンドを選んで解く。`extract` は外表面の抽出にかかった時間（準備の時間に足す）。
fn dispatch(
    lookup: &FaceLookup,
    case: &Case,
    progress: Option<ProgressFn<'_>>,
    solver: Option<&dyn LinearSolver>,
    extract: f64,
) -> Result<Analysis> {
    let n = lookup.mesh().nodes.len();
    let resolved = case.solver.backend.resolve(n);
    match solver {
        // 外部ソルバのときは規模だけでアセンブリの並列度を決める
        Some(s) => {
            let parallel = Backend::Auto.resolve(n) == Backend::CpuParallel;
            run_with(s, lookup, case, progress, parallel, extract)
        }
        None if resolved == Backend::Cuda => Err(GfemError::new(
            ErrorKind::Case,
            crate::tr!(
                "backend: cuda was specified, but no CUDA backend was supplied",
                "backend: cuda が指定されましたが、CUDA バックエンドが渡されていません"
            ),
        )
        .with_hint(crate::tr!(
            "gfem-core on its own does not depend on CUDA (SPEC §7). Use the Python API, \
             or pass a gfem_cuda solver to solve_with",
            "gfem-core 単体は CUDA に依存しません（SPEC §7）。Python API から使うか、\
             gfem_cuda のソルバを solve_with に渡してください"
        ))),
        None if resolved == Backend::CpuParallel => run_with(
            &BackendSolver(CpuParallel),
            lookup,
            case,
            progress,
            true,
            extract,
        ),
        None => run_with(
            &BackendSolver(CpuSerial),
            lookup,
            case,
            progress,
            false,
            extract,
        ),
    }
}

/// バックエンドを明示して解く（アセンブリは逐次）。
///
/// テストで「このバックエンドで解いた結果」を直に比べるための入口。
pub fn solve_with_backend<B: LinearBackend>(
    backend: B,
    mesh: &Mesh,
    case: &Case,
    progress: Option<ProgressFn<'_>>,
) -> Result<Analysis> {
    let started = Instant::now();
    let surface = extract_surface(mesh)?;
    let lookup = FaceLookup::new(mesh, &surface);
    let extract = started.elapsed().as_secs_f64();
    run_with(
        &BackendSolver(backend),
        &lookup,
        case,
        progress,
        false,
        extract,
    )
}

/// 2 次要素（Tet10）での求解は、組み立てと後処理が対応するまで止める（M12 の途中）。
///
/// 要素・読み込み・2 次への変換・電極荷重（6 節点の面）は入ったが、剛性の組み立てと
/// 後処理（電界の代表値・保存則の検査・等値面など）がまだ 2 次要素に対応していない。
/// 保存則の検査は要素中心の 1 点で電力を数えているので、2 次要素では必ず誤った警告になる。
/// 対応が揃うまでは入口で止める。
fn reject_unfinished_quadratic(mesh: &Mesh) -> Result<()> {
    if mesh.solids.iter().any(|b| b.kind().is_quadratic()) {
        return Err(GfemError::unsupported(crate::tr!(
            "Solving with second-order elements (10-node CTETRA) is not available yet",
            "2 次要素（10 節点の CTETRA）での求解はまだ使えません"
        ))
        .with_hint(crate::tr!(
            "Assembly and post-processing for second-order elements are still being \
             implemented. Solve with first-order elements for now",
            "2 次要素に対応した組み立てと後処理を実装中です。今は 1 次要素で解いてください"
        )));
    }
    Ok(())
}

fn run_with(
    solver: &dyn LinearSolver,
    lookup: &FaceLookup,
    case: &Case,
    progress: Option<ProgressFn<'_>>,
    parallel_assembly: bool,
    extract: f64,
) -> Result<Analysis> {
    let started = Instant::now();
    let mesh = lookup.mesh();

    // --- 準備 ---
    // 1 次と 2 次の混在・中点の食い違いは、解けてしまうが答えが黙って狂う（SPEC §2.3）。
    // 1 次要素だけのメッシュでは何もせずに返る
    crate::mesh::check_orders(mesh)?;
    reject_unfinished_quadratic(mesh)?;
    // 検証と荷重の組み立てで同じ下ごしらえ（パッチ選択の索引など）を使う
    let mut warnings = case.validate_with(lookup)?;
    let prepare = extract + started.elapsed().as_secs_f64();

    // --- アセンブリ ---
    let mark = Instant::now();
    let LinearSystem {
        matrix,
        load,
        mut electrodes,
        gauge,
    } = assemble(mesh, case, lookup, parallel_assembly)?;
    let assemble_time = mark.elapsed().as_secs_f64();
    // 縁のピークがメッシュで決まってしまう電極を知らせる（SPEC §2.5）
    warnings.extend(contact::resolution_warnings(&electrodes));

    // --- 求解 ---
    let mark = Instant::now();
    // 2 次要素では AMG の θ を 2 次要素向けにする（SPEC §4.5。1 次要素では何も変えない）
    let options = case.solver.for_mesh(mesh);
    let (mut potential, reports, rhs_scale) = match gauge {
        Gauge::MeanZero => {
            let mut potential = vec![0.0; mesh.nodes.len()];
            let report = solver.solve(&matrix, &load, &mut potential, &options, progress)?;
            (potential, vec![report], None)
        }
        // 完全電極モデル: 電極の電位を仮に決めた系を重ね合わせる（SPEC §2.5）
        Gauge::NotNeeded => {
            let solution = contact::solve_superposition(
                solver,
                &matrix,
                &load,
                &electrodes,
                &options,
                progress,
            )?;
            for (e, &u) in electrodes.iter_mut().zip(&solution.terminal_potentials) {
                e.terminal_potential = u;
            }
            (
                solution.potential,
                solution.reports,
                Some(solution.rhs_scale),
            )
        }
    };
    let solve_time = mark.elapsed().as_secs_f64();

    if let Some(report) = reports.iter().find(|r| !r.converged) {
        let reason = report.failure_reason().unwrap_or_default();
        let kind = if report.cancelled {
            ErrorKind::Cancelled
        } else {
            ErrorKind::Convergence
        };
        return Err(GfemError::new(
            kind,
            crate::tr!(
                "CG did not converge: {reason}",
                "CG が収束しませんでした: {reason}"
            ),
        )
        .with_hint(if report.cancelled {
            crate::tr!(
                "The solution at the point of cancellation cannot be used. Run it again",
                "中断された時点の解は使えません。もう一度実行してください"
            )
        } else {
            crate::tr!(
                "Increase the maximum number of iterations, loosen the convergence tolerance \
                     (currently {:.0e}), or review the conductivity contrast and the mesh quality",
                "最大反復回数を増やす、収束判定を緩める（現在 {:.0e}）、\
                     または導電率のコントラストとメッシュの品質を見直してください",
                case.solver.tolerance
            )
        }));
    }

    let report = if reports.len() == 1 {
        reports[0].clone()
    } else {
        contact::merge_reports(&reports)
    };
    let solves: Vec<(usize, f64)> = reports
        .iter()
        .map(|r| (r.iterations, r.relative_residual))
        .collect();
    drop(reports);

    // --- 表示基準の適用（SPEC §2.2 の「基準点の付け替え」）---
    apply_reference(mesh, &mut potential, &mut electrodes, case.reference, gauge)?;

    // --- ポスト処理 ---
    let mark = Instant::now();
    let fields = compute_fields(mesh, &case.conductivity, &potential)?;
    // CEM では電極の電位を入れた右辺で離散残差を見る（SPEC §5.4）。付け替えた後の電位で作る
    let full_rhs = match gauge {
        Gauge::MeanZero => None,
        Gauge::NotNeeded => Some(contact::full_rhs(&load, &electrodes)),
    };
    let conservation = check_conservation_with(
        &matrix,
        full_rhs.as_deref().unwrap_or(&load),
        &potential,
        &fields,
        &electrodes,
        case.solver.tolerance,
        rhs_scale,
    );
    warnings.extend(conservation.warnings.iter().cloned());
    // AMG の最粗レベルを直接解けなかった（SPEC §4.5）。答えは変わらないが、
    // 反復が増えるうえ粗化がうまく進んでいない兆候なので知らせる
    if let Some(info) = &report.amg
        && !info.direct_coarse_solve
    {
        let (n_coarse, _) = info.levels.last().copied().unwrap_or_default();
        warnings.push(crate::tr!(
            "The coarsest AMG level ({n_coarse} nodes) could not be solved directly, so smoothing \
             was used instead. The answer is unchanged, but more iterations are needed (SPEC §4.5)",
            "AMG の最粗レベル（{n_coarse} 節点）を直接解けなかったため、平滑化で代用しました。\
             答えは変わりませんが反復回数が増えます（SPEC §4.5）"
        ));
    }
    let post = mark.elapsed().as_secs_f64();

    Ok(Analysis {
        potential,
        fields,
        conservation,
        solver: report,
        solves,
        gauge,
        electrodes,
        warnings,
        timings: Timings {
            prepare,
            assemble: assemble_time,
            solve: solve_time,
            post,
            total: extract + started.elapsed().as_secs_f64(),
        },
    })
}

/// 電位の基準点を付け替える。
///
/// E と J は基準に依存しないので、この操作は表示のためだけのもの（SPEC §2.2）。
/// 完全電極モデルの板の電位も同じだけずらす（φ と U を一緒にずらせば、電流も
/// 保存則チェックも変わらない。SPEC §5.4）。
fn apply_reference(
    mesh: &Mesh,
    phi: &mut [f64],
    electrodes: &mut [ElectrodeGeometry],
    reference: ReferencePotential,
    gauge: Gauge,
) -> Result<()> {
    let shift = match reference {
        // 全 Neumann ならソルバの射影で既に平均ゼロになっている
        ReferencePotential::ZeroMean if gauge == Gauge::MeanZero => return Ok(()),
        // 完全電極モデルは基準の電極の板が 0 V で解けているので、平均を落とす
        ReferencePotential::ZeroMean => crate::solver::reduce::sum(phi) / phi.len().max(1) as f64,
        ReferencePotential::CathodeMean => {
            if !electrodes.iter().any(|e| e.role == ElectrodeRole::Cathode) {
                return Err(GfemError::new(
                    ErrorKind::Case,
                    crate::tr!(
                        "The reference potential is cathode_mean, but there is no cathode",
                        "基準電位に cathode_mean が指定されていますが、陰極がありません"
                    ),
                ));
            }
            crate::post::weighted_potential(electrodes, phi, ElectrodeRole::Cathode)
        }
        ReferencePotential::Node(id) => {
            let Some(index) = mesh.nodes.ids().iter().position(|&n| n == id) else {
                return Err(GfemError::new(
                    ErrorKind::Case,
                    crate::tr!(
                        "Node ID {id} given for the reference potential is not in the mesh",
                        "基準電位に指定された節点 ID {id} がメッシュにありません"
                    ),
                )
                .with_hint(crate::tr!(
                    "Specify the ID of an existing GRID",
                    "存在する GRID の ID を指定してください"
                )));
            };
            phi[index]
        }
    };
    for v in phi.iter_mut() {
        *v -= shift;
    }
    for e in electrodes.iter_mut() {
        if let Some(u) = e.terminal_potential.as_mut() {
            *u -= shift;
        }
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
    let electrode_j = analysis.electrode_current_density(mesh);

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
            // 皮膚を通って流れ込む電流密度。電極の外は 0（SPEC §2.5 / §5.5）
            DataArray::scalars_f64("electrode current density [A/m2]", &electrode_j),
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
