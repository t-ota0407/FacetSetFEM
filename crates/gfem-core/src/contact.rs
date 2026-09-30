//! 完全電極モデル（CEM。SPEC §2.5）: 接触インピーダンスの項の組み立て、重ね合わせによる求解、
//! 電極ごとの電位・電流・接触電流密度。
//!
//! 電極 l の板の電位を U_l、接触インピーダンスを z_l [Ω·m²] として、
//!
//! ```text
//! (K + Σ_l M_l / z_l) φ = f_u + Σ_l (U_l / z_l) m_l          (SPEC §2.5)
//! I_l = (U_l A_l − m_lᵀ φ) / z_l
//! ```
//!
//! を、U_l を未知数に加えずに**重ね合わせ**で解く。U_l の行を足した拡大系にすると、
//! その行が電極面の全節点とつながって GPU の SpMV と AMG の粗い格子で重い密な行になるため。
//!
//! 1. 基準の電極 r（最初の陰極）の電位を 0 に固定する
//! 2. r 以外の各 CEM 電極 k について、U_k = 1 V・ほかは 0 V で `A φ^k = m_k / z_k` を解く
//! 3. 一様電流密度の電極があれば、全電極 0 V で `A φ^u = f_u` を解く
//! 4. コンダクタンス行列 `G_lk = (δ_lk A_l − m_lᵀ φ^k) / z_l` で `G U = I − c` を解く
//! 5. `φ = φ^u + Σ_k U_k φ^k`
//!
//! どれも同じ行列なので、前処理は 1 回だけ作る（[`LinearSolver::solve_many`]）。
//! 加算の順序はすべて「電極の並び → 面・節点の通し番号の昇順」で固定してある（設計原則 1）。

use crate::assemble::csr::CsrMatrix;
use crate::assemble::load::ElectrodeGeometry;
use crate::case::{ElectrodeRole, SolverOptions};
use crate::error::{GfemError, Result};
use crate::mesh::Mesh;
use crate::solver::cg::{CgReport, Gauge, ProgressFn};
use crate::solver::external::LinearSolver;
use crate::solver::reduce;

/// 求め直した電極の電流と指定値のずれの許容値（総注入電流に対する比。SPEC §5.4）。
///
/// 基準以外の電極の電流は重ね合わせの小さな連立方程式で決めるので、丸めしか入らない。
/// これを超えたら組み立ての誤り。基準の電極の電流（＝全電極の電流の総和のずれ）は
/// CG の残差で決まるので、別の許容値で見る（[`crate::post::check_conservation`]）。
pub const CONTACT_CURRENT_TOLERANCE: f64 = 1.0e-9;

/// 接触インピーダンスの項 `Σ_l M_l / z_l` を行列に足す（SPEC §2.5）。
///
/// 電極面の節点は同じソリッド要素の節点なので、足す位置は K の非ゼロ構造に必ずある。
/// 無ければ組み立ての誤りなので、内部エラーにする。
pub fn add_contact_terms(matrix: &mut CsrMatrix, electrodes: &[ElectrodeGeometry]) -> Result<()> {
    for e in electrodes {
        let Some(z) = e.contact_impedance() else {
            continue;
        };
        for face in &e.contact_faces {
            let nodes = face.nodes();
            for (a, &i) in nodes.iter().enumerate() {
                for (b, &j) in nodes.iter().enumerate() {
                    let Some(k) = matrix.position(i as usize, j) else {
                        return Err(GfemError::internal(crate::tr!(
                            "The contact-impedance term of electrode {} falls outside the sparsity \
                             pattern of the stiffness matrix (nodes {i} and {j})",
                            "電極 {} の接触インピーダンスの項が、剛性行列の非ゼロ構造の外に出ました\
                             （節点 {i} と {j}）",
                            e.name
                        ))
                        .with_hint(crate::tr!(
                            "Every electrode face must be a face of a solid element (SPEC §11.1)",
                            "電極面はどれもソリッド要素の面である必要があります（SPEC §11.1）"
                        )));
                    };
                    matrix.values_mut()[k] += face.mass[a][b] / z;
                }
            }
        }
    }
    Ok(())
}

/// 重ね合わせで解いた結果。
#[derive(Debug, Clone)]
pub struct ContactSolution {
    /// 節点電位 φ [V]（基準の電極の板が 0 V。表示基準は呼び出し側が付け替える）
    pub potential: Vec<f64>,
    /// 電極ごとの板の電位 U_l [V]。一様電流密度の電極は `None`
    pub terminal_potentials: Vec<Option<f64>>,
    /// 右辺ごとの CG の報告（解いた順）
    pub reports: Vec<CgReport>,
    /// CG が抑えた右辺の大きさ `Σ_k |U_k| ‖f^k‖ + ‖f_u‖`（U は基準の電極を 0 V とした値）。
    ///
    /// 足し合わせた解の残差は `Σ_k U_k r_k + r_u` なので、`‖r‖ ≤ tol × これ`。
    /// 離散残差をこれで割れば、表示の基準電位（§2.2）の付け替えに左右されない（SPEC §5.4）
    pub rhs_scale: f64,
}

/// 解く右辺の数（SPEC §2.5 の「解く回数」）。
pub fn n_solves(electrodes: &[ElectrodeGeometry]) -> usize {
    let n_complete = electrodes
        .iter()
        .filter(|e| e.contact_impedance().is_some())
        .count();
    let has_uniform = electrodes.iter().any(|e| e.contact_impedance().is_none());
    n_complete.saturating_sub(1) + usize::from(has_uniform)
}

/// 基準の電極（電位を 0 に固定する CEM の電極）の通し番号。
///
/// CEM の電極のうち最初の陰極、陰極が無ければ最初の CEM の電極。CEM の電極が無ければ `None`。
pub fn reference_electrode(electrodes: &[ElectrodeGeometry]) -> Option<usize> {
    let complete = |e: &ElectrodeGeometry| e.contact_impedance().is_some();
    electrodes
        .iter()
        .position(|e| complete(e) && e.role == ElectrodeRole::Cathode)
        .or_else(|| electrodes.iter().position(complete))
}

/// 重ね合わせで解く（SPEC §2.5）。
///
/// `matrix` は接触インピーダンスの項を足した `K + Σ M_l/z_l`、`uniform_load` は一様電流密度の
/// 電極の荷重 f_u。収束しなかった右辺があれば、そこまでの報告を入れて返す
/// （`potential` は意味を持たない。呼び出し側が報告を見てエラーにする）。
pub fn solve_superposition(
    solver: &dyn LinearSolver,
    matrix: &CsrMatrix,
    uniform_load: &[f64],
    electrodes: &[ElectrodeGeometry],
    options: &SolverOptions,
    progress: Option<ProgressFn<'_>>,
) -> Result<ContactSolution> {
    let n = matrix.n_rows();
    let reference = reference_electrode(electrodes).ok_or_else(|| {
        GfemError::internal(crate::tr!(
            "There is no complete-electrode-model electrode to superpose",
            "重ね合わせる完全電極モデルの電極がありません"
        ))
    })?;
    // 電位を未知数にする電極（基準以外の CEM の電極）
    let basis: Vec<usize> = (0..electrodes.len())
        .filter(|&l| l != reference && electrodes[l].contact_impedance().is_some())
        .collect();
    let has_uniform = electrodes.iter().any(|e| e.contact_impedance().is_none());

    // 右辺: 基準以外の CEM 電極を 1 V にしたもの → 一様電流密度の電極の荷重
    let mut rhs: Vec<Vec<f64>> = Vec::with_capacity(basis.len() + 1);
    for &k in &basis {
        let e = &electrodes[k];
        let z = e.contact_impedance().expect("CEM の電極");
        let mut f = vec![0.0; n];
        for &(node, w) in &e.node_weights {
            f[node as usize] = w / z;
        }
        rhs.push(f);
    }
    if has_uniform {
        rhs.push(uniform_load.to_vec());
    }
    let rhs_norms: Vec<f64> = rhs.iter().map(|f| reduce::norm2(f)).collect();
    let mut solutions: Vec<Vec<f64>> = vec![vec![0.0; n]; rhs.len()];
    let reports = {
        let rhs_refs: Vec<&[f64]> = rhs.iter().map(|v| v.as_slice()).collect();
        let mut sol_refs: Vec<&mut [f64]> =
            solutions.iter_mut().map(|v| v.as_mut_slice()).collect();
        solver.solve_many(
            matrix,
            &rhs_refs,
            &mut sol_refs,
            options,
            Gauge::NotNeeded,
            progress,
        )?
    };
    drop(rhs);
    let mut terminal_potentials: Vec<Option<f64>> = vec![None; electrodes.len()];
    if reports.len() < solutions.len() || reports.iter().any(|r| !r.converged) {
        return Ok(ContactSolution {
            potential: vec![0.0; n],
            terminal_potentials,
            reports,
            rhs_scale: 0.0,
        });
    }
    let uniform = if has_uniform { solutions.pop() } else { None };

    // コンダクタンス行列 G と、一様電流密度の電極による電流 c（SPEC §2.5 の手順 4）
    let m = basis.len();
    let mut g = vec![0.0f64; m * m];
    let mut b = vec![0.0f64; m];
    for (row, &l) in basis.iter().enumerate() {
        let e = &electrodes[l];
        let z = e.contact_impedance().expect("CEM の電極");
        for (col, phi_k) in solutions.iter().enumerate() {
            let delta = if row == col { e.area } else { 0.0 };
            g[row * m + col] = (delta - e.integrate(phi_k)) / z;
        }
        let c = uniform
            .as_ref()
            .map_or(0.0, |phi_u| -e.integrate(phi_u) / z);
        b[row] = e.signed_current - c;
    }
    let u = solve_dense(&mut g, &mut b, m)?;

    // φ = φ^u + Σ_k U_k φ^k（節点ごとに独立で、足す順序は電極の並びで固定）
    let mut potential = uniform.unwrap_or_else(|| vec![0.0; n]);
    for (phi_k, &u_k) in solutions.iter().zip(&u) {
        for (p, &v) in potential.iter_mut().zip(phi_k) {
            *p += u_k * v;
        }
    }
    terminal_potentials[reference] = Some(0.0);
    for (&l, &u_l) in basis.iter().zip(&u) {
        terminal_potentials[l] = Some(u_l);
    }
    let rhs_scale = u
        .iter()
        .zip(&rhs_norms)
        .map(|(u_k, norm)| u_k.abs() * norm)
        .sum::<f64>()
        + if has_uniform {
            rhs_norms[basis.len()]
        } else {
            0.0
        };
    Ok(ContactSolution {
        potential,
        terminal_potentials,
        reports,
        rhs_scale,
    })
}

/// 小さな密行列の連立方程式 `G x = b` を部分ピボットのガウス消去で解く（行優先、`n × n`）。
///
/// 行の入れ替えは「絶対値が最大の行、同じなら上の行」で決めるので、同じ入力から常に同じ答えになる。
/// G は理論上は対称正定値（コンダクタンス行列）だが、CG の収束判定の範囲でわずかに非対称になりうる
/// ので Cholesky は使わない。
fn solve_dense(g: &mut [f64], b: &mut [f64], n: usize) -> Result<Vec<f64>> {
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&x, &y| {
                g[x * n + col]
                    .abs()
                    .partial_cmp(&g[y * n + col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(y.cmp(&x))
            })
            .expect("列が空");
        let p = g[pivot * n + col];
        if p == 0.0 || !p.is_finite() {
            return Err(GfemError::internal(crate::tr!(
                "The conductance matrix of the complete electrode model is singular (column {col})",
                "完全電極モデルのコンダクタンス行列が特異です（列 {col}）"
            ))
            .with_hint(crate::tr!(
                "Check whether two electrodes share all of their faces, or whether a contact \
                 impedance is extremely large",
                "2 つの電極が同じ面を指していないか、接触インピーダンスが極端に大きくないかを確認してください"
            )));
        }
        if pivot != col {
            for k in 0..n {
                g.swap(col * n + k, pivot * n + k);
            }
            b.swap(col, pivot);
        }
        for row in (col + 1)..n {
            let factor = g[row * n + col] / g[col * n + col];
            if factor == 0.0 {
                continue;
            }
            for k in col..n {
                g[row * n + k] -= factor * g[col * n + k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut s = b[row];
        for k in (row + 1)..n {
            s -= g[row * n + k] * x[k];
        }
        x[row] = s / g[row * n + row];
    }
    Ok(x)
}

/// 右辺 `f_u + Σ_l (U_l / z_l) m_l`（保存則チェックの離散残差に使う。SPEC §5.4）。
///
/// `electrodes` の `terminal_potential` を使うので、表示基準を付け替えた後に作ること
/// （φ と U に同じ定数を足しても、行列と右辺の差は変わらない）。
pub fn full_rhs(uniform_load: &[f64], electrodes: &[ElectrodeGeometry]) -> Vec<f64> {
    let mut f = uniform_load.to_vec();
    for e in electrodes {
        let (Some(z), Some(u)) = (e.contact_impedance(), e.terminal_potential) else {
            continue;
        };
        for &(node, w) in &e.node_weights {
            f[node as usize] += u * w / z;
        }
    }
    f
}

/// 解いた後の電極 1 個の状態（SPEC §2.5 / §5.5）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElectrodeState {
    /// 電極の電位 [V]。CEM は板の電位 U_l、一様電流密度は面積重み平均電位
    pub potential: f64,
    /// 電極の下の皮膚の面積重み平均電位 [V]
    pub mean_surface_potential: f64,
    /// φ から求め直した流入電流 [A]（符号付き）。一様電流密度では指定値そのもの
    pub current: f64,
    /// 皮膚を通って流れ込む電流密度の最大 [A/m²]（電極の向き＝陽極なら流入・陰極なら流出を正にした大きさ）。
    /// CEM は節点値 `(U_l − φ_i)/z_l` の最大、一様電流密度は `|I|/A`
    pub peak_current_density: f64,
    /// 接触面での消費電力 `(1/z)∫(U − φ)² dS` [W]。一様電流密度では 0
    pub contact_power: f64,
}

/// 電極ごとの状態を求める。
pub fn electrode_states(electrodes: &[ElectrodeGeometry], phi: &[f64]) -> Vec<ElectrodeState> {
    electrodes.iter().map(|e| electrode_state(e, phi)).collect()
}

fn electrode_state(e: &ElectrodeGeometry, phi: &[f64]) -> ElectrodeState {
    let mean_surface_potential = e.mean_potential(phi);
    let (Some(z), Some(u)) = (e.contact_impedance(), e.terminal_potential) else {
        return ElectrodeState {
            potential: mean_surface_potential,
            mean_surface_potential,
            current: e.signed_current,
            peak_current_density: e.current_density.abs(),
            contact_power: 0.0,
        };
    };
    let current = (u * e.area - e.integrate(phi)) / z;
    let sign = e.role.sign();
    let peak_current_density = e
        .node_weights
        .iter()
        .map(|&(node, _)| sign * (u - phi[node as usize]) / z)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut contact_power = 0.0;
    for face in &e.contact_faces {
        let nodes = face.nodes();
        for (a, &i) in nodes.iter().enumerate() {
            for (b, &j) in nodes.iter().enumerate() {
                contact_power += (u - phi[i as usize]) * face.mass[a][b] * (u - phi[j as usize]);
            }
        }
    }
    ElectrodeState {
        potential: u,
        mean_surface_potential,
        current,
        peak_current_density,
        contact_power: contact_power / z,
    }
}

/// 電極面の節点ごとの、皮膚を通って流れ込む電流密度 [A/m²]（流入を正。VTU の節点値。SPEC §5.5）。
///
/// CEM は `(U_l − φ_i)/z_l`、一様電流密度は `I/A`。電極の外は 0。2 つの電極が節点を
/// 共有していれば、後に並ぶ電極の値が残る。
pub fn electrode_current_density(
    mesh: &Mesh,
    electrodes: &[ElectrodeGeometry],
    phi: &[f64],
) -> Vec<f64> {
    let mut out = vec![0.0; mesh.nodes.len()];
    for e in electrodes {
        match (e.contact_impedance(), e.terminal_potential) {
            (Some(z), Some(u)) => {
                for &(node, _) in &e.node_weights {
                    out[node as usize] = (u - phi[node as usize]) / z;
                }
            }
            _ => {
                for &(node, _) in &e.node_weights {
                    out[node as usize] = e.current_density;
                }
            }
        }
    }
    out
}

/// 縁の分解能の警告（SPEC §2.5 の「縁の分解能の目安」）。
///
/// 縁で電流が集まる幅 ℓ = zσ より電極面の要素が大きいと、縁のピークがメッシュで決まる。
pub fn resolution_warnings(electrodes: &[ElectrodeGeometry]) -> Vec<String> {
    let mut warnings = Vec::new();
    for e in electrodes {
        let Some(z) = e.contact_impedance() else {
            continue;
        };
        if e.n_faces == 0 || e.underlying_sigma <= 0.0 {
            continue;
        }
        let length = z * e.underlying_sigma;
        let h = (2.0 * e.area / e.n_faces as f64).sqrt();
        if h > length {
            warnings.push(crate::tr!(
                "The faces of electrode {} (about {:.3} mm) are larger than the width over which \
                 current gathers at its edge, ℓ = zσ = {:.3} mm, so the peak contact current density \
                 depends on the mesh. Refine the mesh at the electrode edge before relying on the peak \
                 (SPEC §2.5)",
                "電極 {} の面（約 {:.3} mm）が、縁で電流が集まる幅 ℓ = zσ = {:.3} mm より大きいので、\
                 接触電流密度のピークがメッシュで決まります。ピークの値を使う前に、電極の縁のメッシュを\
                 細かくしてください（SPEC §2.5）",
                e.name,
                h * 1e3,
                length * 1e3
            ));
        }
    }
    warnings
}

/// 右辺ごとの報告を 1 つにまとめる（`Analysis::solver`。SPEC §4.3）。
///
/// 反復回数は合計、残差は最大、収束履歴は順につなぐ。前処理の準備時間は 1 本目に入っている。
pub fn merge_reports(reports: &[CgReport]) -> CgReport {
    let first = reports.first().expect("報告が 1 つも無い");
    let mut merged = first.clone();
    for r in &reports[1..] {
        merged.iterations += r.iterations;
        merged.relative_residual = merged.relative_residual.max(r.relative_residual);
        merged.converged &= r.converged;
        merged.stalled |= r.stalled;
        merged.cancelled |= r.cancelled;
        merged.history.extend_from_slice(&r.history);
        merged.setup_seconds += r.setup_seconds;
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dense_solve_is_exact_for_a_small_system() {
        let mut g = vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0];
        let x_true = [0.3, -1.2, 2.5];
        let mut b: Vec<f64> = (0..3)
            .map(|i| (0..3).map(|j| g[i * 3 + j] * x_true[j]).sum())
            .collect();
        let x = solve_dense(&mut g, &mut b, 3).unwrap();
        for (a, e) in x.iter().zip(&x_true) {
            assert!((a - e).abs() < 1e-14, "{a} vs {e}");
        }
    }

    #[test]
    fn the_dense_solve_pivots() {
        // 先頭の対角が 0 でも、行を入れ替えて解ける
        let mut g = vec![0.0, 1.0, 1.0, 0.0];
        let mut b = vec![2.0, 3.0];
        let x = solve_dense(&mut g, &mut b, 2).unwrap();
        assert_eq!(x, vec![3.0, 2.0]);
    }

    #[test]
    fn a_singular_conductance_matrix_is_reported() {
        let mut g = vec![1.0, 1.0, 1.0, 1.0];
        let mut b = vec![1.0, 1.0];
        assert!(solve_dense(&mut g, &mut b, 2).is_err());
    }
}
