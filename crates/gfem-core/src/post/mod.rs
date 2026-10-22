//! ポスト処理: 導出量と保存則チェック（SPEC §5）、領域統計（SPEC §5.2）、
//! 解析解との誤差評価（SPEC §8 M2）。

pub mod error;
pub mod stats;
pub mod table;

pub use error::{
    ErrorNorms, convergence_order, l2_error_electric_field, l2_error_field, l2_error_potential,
    mean_element_size,
};
pub use stats::{
    Hotspot, QuantityStats, RegionStats, region_stats, total_stats, write_convergence_csv,
    write_region_stats_csv,
};
pub use table::{write_element_table, write_node_table};

use crate::assemble::csr::CsrMatrix;
use crate::assemble::load::ElectrodeGeometry;
use crate::case::ElectrodeRole;
use crate::element::shape::{self, MAX_ELEMENT_NODES, ShapeFunctions};
use crate::element::{ElementKind, Hex8, Penta6, tet4, tet10};
use crate::error::{GfemError, Result};
use crate::geom::{dot, norm};
use crate::material::Conductivity;
use crate::mesh::{Mesh, PropId};
use crate::solver::reduce;
use std::collections::BTreeMap;

/// エネルギー収支の相対誤差の許容値（SPEC §5.4）。
pub const ENERGY_TOLERANCE: f64 = 1.0e-6;

/// 電流保存の許容値（SPEC §2.2）。ケースの検証と同じ値を使う。
pub use crate::case::CURRENT_BALANCE_TOLERANCE;

/// 離散残差 ‖Kφ − f‖/‖f‖ が収束判定しきい値の何倍を超えたら警告するか（SPEC §5.4）。
///
/// CG は残差を漸化式で更新しながら収束を判定するので、丸めが積もると本当の残差
/// ‖Kφ − f‖ とずれていく。「収束した」と報告したのに本当の残差が桁違いに大きければ、
/// その解は報告どおりの精度ではない。以前はこの値を記録するだけで警告しなかった。
pub const DISCRETE_RESIDUAL_FACTOR: f64 = 100.0;

/// 離散残差の警告の下限。倍精度の丸めだけでこの程度にはなるので、
/// 収束判定をこれより厳しくしても警告は出さない。
pub const DISCRETE_RESIDUAL_FLOOR: f64 = 1.0e-12;

/// 要素ごとの導出量（SPEC §5.1）。
///
/// 並びは [`Mesh::iter_solids`] の通し番号順。
#[derive(Debug, Clone, Default)]
pub struct FieldResult {
    /// ソリッド要素数
    pub n_elements: usize,
    /// 電界 E = −∇φ [V/m]。3 成分ずつ
    pub electric_field: Vec<f64>,
    /// 電流密度 J = σE [A/m²]。3 成分ずつ
    pub current_density: Vec<f64>,
    /// |E| [V/m]
    pub e_magnitude: Vec<f64>,
    /// |J| [A/m²]
    pub j_magnitude: Vec<f64>,
    /// 電力密度 σ|E|² [W/m³]。熱連成（SPEC §9）の熱源項にそのまま使える。
    /// E が要素内で変わる要素（Hex8 / Penta6 / Tet10）では**要素内の平均**
    /// （`∫σ|E|² dΩ / V`。剛性と同じ積分点で積分する）。
    /// どちらでも `電力密度 × 体積` を足せば消費電力になる
    pub power_density: Vec<f64>,
    /// 要素体積 [m³]
    pub volume: Vec<f64>,
    /// 要素内の |E| の最大 [V/m]。2 次要素のメッシュだけ（1 次要素では空。要素内で一定か、
    /// 代表値で足りるため）。最大値・ホットスポットの評価に使う（SPEC §5.1 / §5.2）
    pub e_peak: Vec<f64>,
    /// 要素内の |J| の最大 [A/m²]。[`e_peak`](Self::e_peak) と同じく 2 次要素のメッシュだけ
    pub j_peak: Vec<f64>,
}

impl FieldResult {
    /// 全領域の消費電力 ∫σ|E|² dΩ [W]。
    pub fn total_power(&self) -> f64 {
        let terms: Vec<f64> = self
            .power_density
            .iter()
            .zip(&self.volume)
            .map(|(p, v)| p * v)
            .collect();
        reduce::sum(&terms)
    }

    /// 解析対象の総体積 [m³]。
    pub fn total_volume(&self) -> f64 {
        reduce::sum(&self.volume)
    }

    /// 要素 `e` の中での |J| の最大。2 次要素は頂点での最大、1 次要素は要素値そのもの。
    pub fn j_peak_of(&self, e: usize) -> f64 {
        self.j_peak.get(e).copied().unwrap_or(self.j_magnitude[e])
    }

    /// 要素 `e` の中での |E| の最大（[`j_peak_of`](Self::j_peak_of) と同じ扱い）。
    pub fn e_peak_of(&self, e: usize) -> f64 {
        self.e_peak.get(e).copied().unwrap_or(self.e_magnitude[e])
    }
}

/// 節点電位から要素ごとの E・J などを求める（SPEC §5.1）。
pub fn compute_fields(
    mesh: &Mesh,
    conductivity: &BTreeMap<PropId, Conductivity>,
    phi: &[f64],
) -> Result<FieldResult> {
    use rayon::prelude::*;
    let n = mesh.n_solid_elements();
    // 要素内の最大は 2 次要素のメッシュだけで持つ（1 次要素では要素値で足りる）
    let n_peaks = if mesh.solids.iter().any(|b| b.kind().is_quadratic()) {
        n
    } else {
        0
    };
    let mut out = FieldResult {
        n_elements: n,
        electric_field: vec![0.0; 3 * n],
        current_density: vec![0.0; 3 * n],
        e_magnitude: vec![0.0; n],
        j_magnitude: vec![0.0; n],
        power_density: vec![0.0; n],
        volume: vec![0.0; n],
        e_peak: vec![0.0; n_peaks],
        j_peak: vec![0.0; n_peaks],
    };
    let peaks = n_peaks > 0;

    // 要素ごとに独立なので、出力を塊ごとに切り分けて並列に書く。
    // 各要素の値は逐次と同じ計算なので、結果はビット単位で変わらない
    let results: Vec<Result<Vec<(f64, f64)>>> = (
        out.electric_field.par_chunks_mut(3 * FIELD_CHUNK),
        out.current_density.par_chunks_mut(3 * FIELD_CHUNK),
        out.e_magnitude.par_chunks_mut(FIELD_CHUNK),
        out.j_magnitude.par_chunks_mut(FIELD_CHUNK),
        out.power_density.par_chunks_mut(FIELD_CHUNK),
        out.volume.par_chunks_mut(FIELD_CHUNK),
    )
        .into_par_iter()
        .enumerate()
        .map(|(chunk, (field, current, e_mag, j_mag, power, volume))| {
            let start = chunk * FIELD_CHUNK;
            let mut found = Vec::with_capacity(if peaks { volume.len() } else { 0 });
            for k in 0..volume.len() {
                let f = element_field(mesh, conductivity, phi, &mesh.solid(start + k))?;
                field[3 * k..3 * k + 3].copy_from_slice(&f.field);
                current[3 * k..3 * k + 3].copy_from_slice(&f.current);
                e_mag[k] = f.e_magnitude;
                j_mag[k] = f.j_magnitude;
                power[k] = f.power_density;
                volume[k] = f.volume;
                if peaks {
                    found.push((f.e_peak, f.j_peak));
                }
            }
            Ok(found)
        })
        .collect();
    // 塊は通し番号の順に並んでいて、塊の中では最初の不正な要素で止まるので、
    // 逐次と同じく「通し番号がいちばん小さい要素のエラー」を返す
    let mut at = 0;
    for r in results {
        for (e_peak, j_peak) in r? {
            out.e_peak[at] = e_peak;
            out.j_peak[at] = j_peak;
            at += 1;
        }
    }
    Ok(out)
}

/// [`compute_fields`] を並列に回すときの、1 タスクあたりの要素数。
const FIELD_CHUNK: usize = 4096;

/// 要素 1 個ぶんの導出量。
struct ElementField {
    field: [f64; 3],
    current: [f64; 3],
    e_magnitude: f64,
    j_magnitude: f64,
    power_density: f64,
    volume: f64,
    /// 要素内の |E|・|J| の最大（1 次要素では要素値と同じ）
    e_peak: f64,
    j_peak: f64,
}

fn element_field(
    mesh: &Mesh,
    conductivity: &BTreeMap<PropId, Conductivity>,
    phi: &[f64],
    e: &crate::mesh::ElementRef<'_>,
) -> Result<ElementField> {
    let Some(sigma) = conductivity.get(&e.pid) else {
        return Err(GfemError::new(
            crate::error::ErrorKind::Case,
            crate::tr!(
                "No conductivity is set for PID={}",
                "PID={} に導電率が設定されていません",
                e.pid
            ),
        ));
    };
    let n = e.conn.len();
    let mut points = [[0.0f64; 3]; MAX_ELEMENT_NODES];
    mesh.gather_coords(e.conn, &mut points[..n]);
    let mut nodal = [0.0f64; MAX_ELEMENT_NODES];
    for (k, &node) in e.conn.iter().enumerate() {
        nodal[k] = phi[node as usize];
    }

    // Tet4 は E が要素内で一定。Hex8 / Penta6 は要素内で変化するので、
    // 要素の代表値としては**中心での値**を採り、電力は剛性と同じ積分点で積分する（SPEC §5.1）。
    // 戻り値の 3 つ目は電力 ∫σ|E|²dΩ [W]（Tet4 は `None` で、代表値から σ|E|²·V）
    let (field, volume, power) = match e.kind {
        ElementKind::Tet4 => {
            let p = mesh.tet4_coords(e.conn);
            let shape = tet4::tet4_shape(&p);
            let nodal = [nodal[0], nodal[1], nodal[2], nodal[3]];
            (
                tet4::tet4_electric_field(&shape, &nodal),
                shape.volume,
                None,
            )
        }
        ElementKind::Hex8 => (
            shape::electric_field::<Hex8>(&points[..n], &nodal[..n], Hex8::CENTRE),
            shape::volume::<Hex8>(&points[..n])?,
            Some(shape::power_with_sign::<Hex8>(
                &points[..n],
                &nodal[..n],
                sigma,
                1.0,
            )?),
        ),
        ElementKind::Penta6 => (
            shape::electric_field::<Penta6>(&points[..n], &nodal[..n], Penta6::CENTRE),
            shape::volume::<Penta6>(&points[..n])?,
            Some(shape::power_with_sign::<Penta6>(
                &points[..n],
                &nodal[..n],
                sigma,
                1.0,
            )?),
        ),
        // 2 次要素は E が要素内で 1 次式で変わる。代表値は中心の値、電力は積分点で積分した
        // 要素内の平均、最大は頂点で取る（element::tet10::field_summary）
        ElementKind::Tet10 => {
            let s = tet10::field_summary(&points[..n], &nodal[..n], sigma)?;
            let current = sigma.apply(s.centre_field);
            return Ok(ElementField {
                field: s.centre_field,
                current,
                e_magnitude: norm(s.centre_field),
                j_magnitude: norm(current),
                power_density: s.power / s.volume,
                volume: s.volume,
                e_peak: s.e_peak,
                j_peak: s.j_peak,
            });
        }
        other => {
            return Err(GfemError::unsupported(crate::tr!(
                "{} cannot be treated as a solid element",
                "{} はソリッド要素として扱えません",
                other.card_name()
            )));
        }
    };
    let current = sigma.apply(field);
    let (e_magnitude, j_magnitude) = (norm(field), norm(current));
    Ok(ElementField {
        field,
        current,
        e_magnitude,
        j_magnitude,
        // σ|E|² = E·(σE) = E·J。異方性でも成り立つ形にしておく。
        // E が要素内で変わる要素は、積分した電力の要素内の平均
        power_density: match power {
            Some(power) => power / volume,
            None => dot(field, current),
        },
        volume,
        e_peak: e_magnitude,
        j_peak: j_magnitude,
    })
}

/// 要素値を体積重み平均で節点値に外挿する（SPEC §5.1）。
///
/// 表示を滑らかにするための量であり、**最大値の評価には要素値を使うべき**
/// （節点値は平滑化されているため過小評価になる）。この注意は GUI にも出す。
pub fn nodal_average(
    mesh: &Mesh,
    values: &[f64],
    volumes: &[f64],
    n_components: usize,
) -> Vec<f64> {
    let n_nodes = mesh.nodes.len();
    let mut acc = vec![0.0; n_nodes * n_components];
    let mut weight = vec![0.0; n_nodes];
    for e in mesh.iter_solids() {
        let v = volumes[e.index];
        for &node in e.conn {
            weight[node as usize] += v;
            for c in 0..n_components {
                acc[node as usize * n_components + c] += v * values[e.index * n_components + c];
            }
        }
    }
    for i in 0..n_nodes {
        if weight[i] > 0.0 {
            for c in 0..n_components {
                acc[i * n_components + c] /= weight[i];
            }
        }
    }
    acc
}

/// 要素値を「ソリッド + シェル」の並びに広げる。
///
/// VTU のセル順序はソリッド → シェル（[`Mesh::blocks`]）なので、
/// シェル側は `fill` で埋める。
pub fn pad_for_shells(mesh: &Mesh, values: &[f64], n_components: usize, fill: f64) -> Vec<f64> {
    let mut out = values.to_vec();
    out.resize(mesh.n_elements() * n_components, fill);
    out
}

/// 保存則チェックの結果（SPEC §5.4）と電極間インピーダンス（SPEC §5.3）。
#[derive(Debug, Clone, Default)]
pub struct ConservationReport {
    /// 電流保存の残差 `1ᵀf / I`。**総注入電流で正規化**した無次元量
    ///
    /// 「差し引き何アンペア余っているか」を注入電流と比べた値なので、
    /// メッシュを細かくしても意味が変わらない。
    ///
    /// 完全電極モデル（SPEC §5.4）では、φ から求め直した全電極の電流の総和を総注入電流で割ったもの。
    pub current_balance: f64,
    /// 完全電極モデルで φ から求め直した電流と指定値のずれの最大（総注入電流で割る。SPEC §5.4）。
    /// CEM の電極が無ければ 0
    pub electrode_current_error: f64,
    /// ∫σ|E|² dΩ [W]
    pub power_from_field: f64,
    /// 接触面での消費電力 `Σ_l (1/z_l)∫(U_l − φ)² dS` [W]（SPEC §5.4）。CEM の電極が無ければ 0
    pub contact_power: f64,
    /// `fᵀφ` [W]。電極から注入された電力（CEM の電極は `I_l U_l`）
    pub power_from_load: f64,
    /// 注入された電力と消費（体内 + 接触面）の相対誤差
    pub energy_relative_error: f64,
    /// `‖Kφ − f‖ / ‖f‖`
    pub discrete_residual: f64,
    /// 電極ごとの電位 [V]（一様電流密度は面積重み平均電位、CEM は板の電位。SPEC §5.3）
    pub electrode_potentials: Vec<(String, f64)>,
    /// 電極間電位差 ΔV [V]
    pub voltage: f64,
    /// 等価抵抗 R = ΔV / I [Ω]
    pub impedance: f64,
    /// 総注入電流 [A]
    pub total_current: f64,
    /// 許容値を超えた項目の警告
    pub warnings: Vec<String>,
}

impl ConservationReport {
    /// 警告が 1 つも無いか。
    pub fn is_clean(&self) -> bool {
        self.warnings.is_empty()
    }
}

/// 保存則チェックを実行する（SPEC §5.4）。毎回自動で走らせる想定。
///
/// `tolerance` は CG の収束判定しきい値（相対残差）。離散残差の警告に使う。
pub fn check_conservation(
    matrix: &CsrMatrix,
    load: &[f64],
    phi: &[f64],
    fields: &FieldResult,
    electrodes: &[ElectrodeGeometry],
    tolerance: f64,
) -> ConservationReport {
    check_conservation_with(matrix, load, phi, fields, electrodes, tolerance, None)
}

/// [`check_conservation`] の、離散残差を割る右辺の大きさを指定する版。
///
/// 完全電極モデル（SPEC §2.5 / §5.4）では、`load` に電極の電位を入れた右辺を渡し、
/// `rhs_scale` に CG が抑えた右辺の大きさ（[`crate::contact::ContactSolution::rhs_scale`]）を渡す。
/// `None` なら `‖load‖` で割る。
pub fn check_conservation_with(
    matrix: &CsrMatrix,
    load: &[f64],
    phi: &[f64],
    fields: &FieldResult,
    electrodes: &[ElectrodeGeometry],
    tolerance: f64,
    rhs_scale: Option<f64>,
) -> ConservationReport {
    let mut warnings = Vec::new();
    let residual_limit = (DISCRETE_RESIDUAL_FACTOR * tolerance).max(DISCRETE_RESIDUAL_FLOOR);
    let norm_f = rhs_scale.unwrap_or_else(|| reduce::norm2(load));
    let total_current: f64 = electrodes
        .iter()
        .filter(|e| e.role == ElectrodeRole::Anode)
        .map(|e| e.signed_current)
        .sum();
    // 完全電極モデル（SPEC §2.5）で解いたか。解いた後は板の電位が入っている
    let complete = electrodes.iter().any(|e| e.terminal_potential.is_some());

    let power_from_field = fields.total_power();
    let (current_balance, electrode_current_error, power_from_load, contact_power) = if complete {
        // 1'. 電流保存（SPEC §5.4 の CEM の読み替え）: φ から電極ごとの電流を求め直す
        let states = crate::contact::electrode_states(electrodes, phi);
        let scale = if total_current > 0.0 {
            total_current
        } else {
            1.0
        };
        let sum: f64 = states.iter().map(|s| s.current).sum();
        let current_balance = sum / scale;
        // 基準以外の電極は重ね合わせで電流を決めているので、丸めの範囲で指定値に一致するはず
        let reference = crate::contact::reference_electrode(electrodes);
        let electrode_current_error = electrodes
            .iter()
            .zip(&states)
            .enumerate()
            .filter(|&(l, _)| Some(l) != reference)
            .map(|(_, (e, s))| (s.current - e.signed_current).abs() / scale)
            .fold(0.0f64, f64::max);
        let tol = crate::contact::CONTACT_CURRENT_TOLERANCE;
        if electrode_current_error > tol {
            warnings.push(crate::tr!(
                "The electrode currents recomputed from the potential do not match the settings: \
                 largest deviation {electrode_current_error:.3e} (relative to the injected current; \
                 tolerance {tol:.0e}). The superposition was assembled incorrectly",
                "電位から求め直した電極の電流が設定と合いません: ずれの最大 {electrode_current_error:.3e}\
                 （総注入電流に対する比。許容 {tol:.0e}）。重ね合わせの組み立ての誤りです"
            ));
        }
        // 総和（＝基準の電極の電流のずれ）は −1ᵀ(Aφ − f)/I に等しく、CG の残差で決まる。
        // 収束判定から見込まれる大きさ |1ᵀr| ≲ ‖r‖ ≤ limit × (右辺の大きさ) と比べる
        let balance_limit = (residual_limit * norm_f / scale).max(tol);
        if current_balance.abs() > balance_limit {
            warnings.push(crate::tr!(
                "The total of the electrode currents recomputed from the potential is \
                 {current_balance:.3e} of the injected current, more than the convergence tolerance \
                 leads one to expect ({balance_limit:.0e}). Solve again with a tighter tolerance",
                "電位から求め直した電極の電流の総和が総注入電流の {current_balance:.3e} で、\
                 収束判定から見込まれる大きさ（{balance_limit:.0e}）を超えています。\
                 収束判定を厳しくして解き直してください"
            ));
        }
        // 2'. 電極から入る電力: 一様電流密度は J ∫φ dS、CEM は I_l U_l
        let power_from_load: f64 = electrodes
            .iter()
            .zip(&states)
            .map(|(e, s)| match e.terminal_potential {
                Some(u) => s.current * u,
                None => e.current_density * e.integrate(phi),
            })
            .sum();
        let contact_power: f64 = states.iter().map(|s| s.contact_power).sum();
        (
            current_balance,
            electrode_current_error,
            power_from_load,
            contact_power,
        )
    } else {
        // 1. 電流保存: 1ᵀf = 0（SPEC §2.2 の適合条件）
        //
        // 正規化には**総注入電流**を使う。SPEC v0.3 までは max|f| で割っていたが、
        // これは節点あたりの荷重で割ることになるためメッシュ依存になる:
        // 細かくすると max|f| は節点数に反比例して小さくなる一方、総和の丸め誤差は
        // √N で増えるので、同じ精度で解いているのに判定だけが厳しくなってしまう。
        // 総注入電流で割れば「差し引き何 A 余っているか」を測ることになり、
        // 分割によらず同じ意味になる。
        let sum_f = reduce::sum(load);
        let scale = if total_current > 0.0 {
            total_current
        } else {
            load.iter().fold(0.0f64, |m, v| m.max(v.abs()))
        };
        let current_balance = if scale > 0.0 { sum_f / scale } else { 0.0 };
        if current_balance.abs() > CURRENT_BALANCE_TOLERANCE {
            warnings.push(crate::tr!(
                "The current conservation residual is too large: 1ᵀf/I = {current_balance:.3e} \
                 (tolerance {CURRENT_BALANCE_TOLERANCE:.0e})",
                "電流保存の残差が大きすぎます: 1ᵀf/I = {current_balance:.3e}（許容 {CURRENT_BALANCE_TOLERANCE:.0e}）"
            ));
        }
        // 2. エネルギー収支: ∫σ|E|²dΩ と fᵀφ
        (current_balance, 0.0, reduce::dot(load, phi), 0.0)
    };

    let consumed = power_from_field + contact_power;
    let scale = consumed.abs().max(power_from_load.abs());
    let energy_relative_error = if scale > 0.0 {
        (consumed - power_from_load).abs() / scale
    } else {
        0.0
    };
    if energy_relative_error > ENERGY_TOLERANCE {
        warnings.push(if complete {
            crate::tr!(
                "The energy balance does not match: ∫σ|E|²dΩ + contact losses = {consumed:.6e} W, \
                 power delivered by the electrodes = {power_from_load:.6e} W, \
                 relative error {energy_relative_error:.3e} (tolerance {ENERGY_TOLERANCE:.0e})",
                "エネルギー収支が合いません: ∫σ|E|²dΩ + 接触面の消費 = {consumed:.6e} W、\
                 電極から入る電力 = {power_from_load:.6e} W、\
                 相対誤差 {energy_relative_error:.3e}（許容 {ENERGY_TOLERANCE:.0e}）"
            )
        } else {
            crate::tr!(
                "The energy balance does not match: ∫σ|E|²dΩ = {power_from_field:.6e} W, fᵀφ = {power_from_load:.6e} W, \
                 relative error {energy_relative_error:.3e} (tolerance {ENERGY_TOLERANCE:.0e})",
                "エネルギー収支が合いません: ∫σ|E|²dΩ = {power_from_field:.6e} W, fᵀφ = {power_from_load:.6e} W, \
                 相対誤差 {energy_relative_error:.3e}（許容 {ENERGY_TOLERANCE:.0e}）"
            )
        });
    }

    // 3. 離散残差: ‖Kφ − f‖/‖f‖（CEM では `load` に電極の電位を入れた右辺を渡す。SPEC §5.4）
    let mut kphi = vec![0.0; phi.len()];
    matrix.spmv(phi, &mut kphi);
    let diff: Vec<f64> = kphi.iter().zip(load).map(|(a, b)| a - b).collect();
    let discrete_residual = if norm_f > 0.0 {
        reduce::norm2(&diff) / norm_f
    } else {
        0.0
    };
    if discrete_residual > residual_limit {
        warnings.push(crate::tr!(
            "The discrete residual ‖Kφ−f‖/‖f‖ = {discrete_residual:.3e} exceeds what the convergence \
             tolerance {tolerance:.0e} leads one to expect (at most {residual_limit:.0e}). Rounding errors \
             may have accumulated in the CG residual updates, so solve again with a looser tolerance \
             or a different preconditioner",
            "離散残差 ‖Kφ−f‖/‖f‖ = {discrete_residual:.3e} が、収束判定 {tolerance:.0e} から\
             見込まれる大きさ（{residual_limit:.0e} 以下）を超えています。CG の残差の更新に\
             丸めが積もっている可能性があるので、収束判定を緩めるか前処理を変えて解き直してください"
        ));
    }

    // 4. 電極間インピーダンス（SPEC §5.3）。CEM の電極は板の電位を使うので接触抵抗を含む
    let electrode_potentials: Vec<(String, f64)> = electrodes
        .iter()
        .map(|e| (e.name.clone(), e.potential(phi)))
        .collect();
    let anode = weighted_potential(electrodes, phi, ElectrodeRole::Anode);
    let cathode = weighted_potential(electrodes, phi, ElectrodeRole::Cathode);
    let voltage = anode - cathode;
    let impedance = if total_current != 0.0 {
        voltage / total_current
    } else {
        f64::NAN
    };

    ConservationReport {
        current_balance,
        electrode_current_error,
        power_from_field,
        contact_power,
        power_from_load,
        energy_relative_error,
        discrete_residual,
        electrode_potentials,
        voltage,
        impedance,
        total_current,
        warnings,
    }
}

/// 同じ役割の電極を電流で重み付けした平均電位（その役割の電極が無ければ 0）。
///
/// 電極間インピーダンス（SPEC §5.3）と、基準電位 `cathode_mean` の付け替え
/// （[`crate::analysis`]）の両方で使う。電極の電位は [`ElectrodeGeometry::potential`]
/// （CEM は板の電位、一様電流密度は面積重み平均電位）。
pub(crate) fn weighted_potential(
    electrodes: &[ElectrodeGeometry],
    phi: &[f64],
    role: ElectrodeRole,
) -> f64 {
    let mut sum = 0.0;
    let mut weight = 0.0;
    for e in electrodes.iter().filter(|e| e.role == role) {
        let w = e.signed_current.abs();
        sum += w * e.potential(phi);
        weight += w;
    }
    if weight > 0.0 { sum / weight } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::nastran::read_nastran_from;
    use crate::units::LengthUnit;
    use std::path::Path;

    fn single_tet() -> Mesh {
        read_nastran_from(
            "GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
GRID,4,,0.0,0.0,1.0
PSOLID,1,1
CTETRA,10,1,1,2,3,4
"
            .as_bytes(),
            Path::new("<test>"),
            LengthUnit::Meter,
        )
        .unwrap()
        .0
    }

    fn sigma(value: f64) -> BTreeMap<PropId, Conductivity> {
        let mut m = BTreeMap::new();
        m.insert(1, Conductivity::isotropic(value).unwrap());
        m
    }

    #[test]
    fn linear_potential_gives_a_uniform_field() {
        let mesh = single_tet();
        // φ = −5x なので E = −∇φ = (5, 0, 0)
        let phi = vec![0.0, -5.0, 0.0, 0.0];
        let f = compute_fields(&mesh, &sigma(0.4), &phi).unwrap();
        assert_eq!(f.n_elements, 1);
        assert!((f.electric_field[0] - 5.0).abs() < 1e-14);
        assert!(f.electric_field[1].abs() < 1e-14);
        assert!((f.e_magnitude[0] - 5.0).abs() < 1e-14);
        // J = σE
        assert!((f.current_density[0] - 2.0).abs() < 1e-14);
        assert!((f.j_magnitude[0] - 2.0).abs() < 1e-14);
        // σ|E|² = 0.4 * 25 = 10 W/m³
        assert!((f.power_density[0] - 10.0).abs() < 1e-13);
        assert!((f.volume[0] - 1.0 / 6.0).abs() < 1e-15);
        assert!((f.total_power() - 10.0 / 6.0).abs() < 1e-14);
    }

    #[test]
    fn constant_potential_gives_no_field() {
        let mesh = single_tet();
        let phi = vec![3.7; 4];
        let f = compute_fields(&mesh, &sigma(0.4), &phi).unwrap();
        assert!(f.e_magnitude[0] < 1e-15);
        assert_eq!(f.total_power(), 0.0);
    }

    #[test]
    fn nodal_average_of_a_constant_element_field() {
        let mesh = single_tet();
        let values = vec![7.0];
        let volumes = vec![1.0 / 6.0];
        let nodal = nodal_average(&mesh, &values, &volumes, 1);
        assert_eq!(nodal.len(), 4);
        assert!(nodal.iter().all(|v| (v - 7.0).abs() < 1e-15));
    }

    #[test]
    fn padding_extends_element_values_over_shells() {
        let mesh = read_nastran_from(
            "GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
GRID,4,,0.0,0.0,1.0
PSOLID,1,1
PSHELL,101,1,0.1
CTETRA,10,1,1,2,3,4
CTRIA3,100,101,1,3,2
"
            .as_bytes(),
            Path::new("<test>"),
            LengthUnit::Meter,
        )
        .unwrap()
        .0;
        let padded = pad_for_shells(&mesh, &[1.0], 1, 0.0);
        assert_eq!(padded, vec![1.0, 0.0]);
    }
}
