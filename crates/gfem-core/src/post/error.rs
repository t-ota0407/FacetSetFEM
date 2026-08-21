//! 解析解との誤差評価と収束次数の推定（SPEC §8 M2）。
//!
//! 「解析解と照合できること」「メッシュ収束性が理論通りであること」は
//! この道具立てが無いと確かめられない（SPEC §1.2）。
//!
//! 1 次四面体では理論上
//!
//! ```text
//! ‖φ_h − φ‖_L2 = O(h²)      ‖E_h − E‖_L2 = O(h)
//! ```
//!
//! となる。実際に細分化列で誤差を測り、[`convergence_order`] で次数を出して
//! 理論値からの逸脱を CI で検出する。
//!
//! # 電位のゲージについて
//!
//! 全 Neumann 境界の問題では φ は定数分だけ不定なので（SPEC §2.2）、
//! 解と解析解を素朴に引き算しても意味が無い。[`l2_error_potential`] は
//! **差の体積重み平均を引いてから**ノルムを取る。これは
//! `∫(φ_h − φ − c)² dΩ` を最小にする c であり、ゲージの選び方に依らない
//! 「形の誤差」だけを測ることになる。

use crate::element::{ElementKind, tet4};
use crate::error::{GfemError, Result};
use crate::mesh::Mesh;
use crate::solver::reduce;

/// 四面体の 4 点ガウス求積（代数的精度 2 次）の体積座標。
const TET4_GAUSS: [[f64; 4]; 4] = [
    [
        0.585_410_196_624_968_5,
        0.138_196_601_125_010_5,
        0.138_196_601_125_010_5,
        0.138_196_601_125_010_5,
    ],
    [
        0.138_196_601_125_010_5,
        0.585_410_196_624_968_5,
        0.138_196_601_125_010_5,
        0.138_196_601_125_010_5,
    ],
    [
        0.138_196_601_125_010_5,
        0.138_196_601_125_010_5,
        0.585_410_196_624_968_5,
        0.138_196_601_125_010_5,
    ],
    [
        0.138_196_601_125_010_5,
        0.138_196_601_125_010_5,
        0.138_196_601_125_010_5,
        0.585_410_196_624_968_5,
    ],
];

/// 各求積点の重み（体積に対する割合）。
const TET4_GAUSS_WEIGHT: f64 = 0.25;

/// L2 ノルムで測った誤差。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorNorms {
    /// `‖u_h − u‖_L2`
    pub absolute: f64,
    /// `‖u‖_L2`。相対誤差の分母
    pub reference: f64,
    /// `‖u_h − u‖_L2 / ‖u‖_L2`
    pub relative: f64,
    /// 電位の場合、差から取り除いた定数（ゲージのずれ）。電界では 0
    pub gauge_shift: f64,
}

/// 節点電位と解析解の L2 誤差。
///
/// `exact` は座標（m）を受け取って電位 [V] を返す関数。
/// ゲージのずれは自動的に取り除かれる（モジュール冒頭の説明を参照）。
pub fn l2_error_potential<F>(mesh: &Mesh, phi: &[f64], exact: F) -> Result<ErrorNorms>
where
    F: Fn([f64; 3]) -> f64,
{
    let block = require_tet4(mesh)?;
    let n = block.len();

    // 1 巡目: 差の体積重み平均（＝最適な定数シフト）を求める
    let mut diff_integral = Vec::with_capacity(n * 4);
    let mut volumes = Vec::with_capacity(n * 4);
    let mut samples = Vec::with_capacity(n * 4);
    for e in 0..n {
        let conn = block.conn_of(e);
        let p = mesh.tet4_coords(conn);
        let volume = tet4::tet4_shape(&p).volume;
        let nodal = [
            phi[conn[0] as usize],
            phi[conn[1] as usize],
            phi[conn[2] as usize],
            phi[conn[3] as usize],
        ];
        for point in &TET4_GAUSS {
            let x = interpolate_position(&p, point);
            let approx = interpolate_scalar(&nodal, point);
            let truth = exact(x);
            let w = TET4_GAUSS_WEIGHT * volume;
            diff_integral.push(w * (approx - truth));
            volumes.push(w);
            samples.push((w, approx - truth, truth));
        }
    }
    let total_volume = reduce::sum(&volumes);
    let gauge_shift = if total_volume > 0.0 {
        reduce::sum(&diff_integral) / total_volume
    } else {
        0.0
    };

    // 2 巡目: シフトを除いてノルムを取る
    let mut squared_error = Vec::with_capacity(samples.len());
    let mut squared_reference = Vec::with_capacity(samples.len());
    for &(w, diff, truth) in &samples {
        let d = diff - gauge_shift;
        squared_error.push(w * d * d);
        squared_reference.push(w * truth * truth);
    }
    Ok(finish(
        reduce::sum(&squared_error),
        reduce::sum(&squared_reference),
        gauge_shift,
    ))
}

/// 要素ごとのベクトル場（E や J）と解析解の L2 誤差。
///
/// `field` は要素順に 3 成分ずつ並んだ配列、`exact` は座標から真値を返す関数。
/// Tet4 では場が要素内で一定なので、求積点ごとに解析解だけを評価すればよい。
pub fn l2_error_field<F>(mesh: &Mesh, field: &[f64], exact: F) -> Result<ErrorNorms>
where
    F: Fn([f64; 3]) -> [f64; 3],
{
    let block = require_tet4(mesh)?;
    let n = block.len();
    if field.len() != 3 * n {
        return Err(GfemError::internal(format!(
            "要素値の長さが合いません: {} 個ですが 3 × {n} であるべきです",
            field.len()
        )));
    }

    let mut squared_error = Vec::with_capacity(n * 4);
    let mut squared_reference = Vec::with_capacity(n * 4);
    for e in 0..n {
        let p = mesh.tet4_coords(block.conn_of(e));
        let volume = tet4::tet4_shape(&p).volume;
        let approx = [field[3 * e], field[3 * e + 1], field[3 * e + 2]];
        for point in &TET4_GAUSS {
            let x = interpolate_position(&p, point);
            let truth = exact(x);
            let w = TET4_GAUSS_WEIGHT * volume;
            let d = [
                approx[0] - truth[0],
                approx[1] - truth[1],
                approx[2] - truth[2],
            ];
            squared_error.push(w * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
            squared_reference
                .push(w * (truth[0] * truth[0] + truth[1] * truth[1] + truth[2] * truth[2]));
        }
    }
    Ok(finish(
        reduce::sum(&squared_error),
        reduce::sum(&squared_reference),
        0.0,
    ))
}

/// 代表的な要素寸法 `h = (V_total / 要素数)^(1/3)`。
///
/// 収束次数を測るときの横軸に使う。要素形状が揃っていれば実際の辺長に比例する。
pub fn mean_element_size(mesh: &Mesh) -> Result<f64> {
    let block = require_tet4(mesh)?;
    if block.is_empty() {
        return Ok(0.0);
    }
    let volumes: Vec<f64> = (0..block.len())
        .map(|e| tet4::tet4_shape(&mesh.tet4_coords(block.conn_of(e))).volume)
        .collect();
    Ok((reduce::sum(&volumes) / block.len() as f64).cbrt())
}

/// 誤差の収束次数を最小二乗で推定する。
///
/// `log(error) = p · log(h) + c` を当てはめて傾き `p` を返す。
/// 理論値は φ が 2、E が 1（SPEC §8 M2）。
pub fn convergence_order(sizes: &[f64], errors: &[f64]) -> Result<f64> {
    if sizes.len() != errors.len() || sizes.len() < 2 {
        return Err(GfemError::internal(
            "収束次数の推定には長さの等しい 2 点以上のデータが必要です",
        ));
    }
    if sizes.iter().chain(errors).any(|v| *v <= 0.0 || v.is_nan()) {
        return Err(GfemError::internal(
            "収束次数の推定には正の要素寸法と誤差が必要です（誤差が 0 だと対数が取れません）",
        ));
    }
    let xs: Vec<f64> = sizes.iter().map(|h| h.ln()).collect();
    let ys: Vec<f64> = errors.iter().map(|e| e.ln()).collect();
    let n = xs.len() as f64;
    let mean_x = reduce::sum(&xs) / n;
    let mean_y = reduce::sum(&ys) / n;
    let cov: Vec<f64> = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .collect();
    let var: Vec<f64> = xs.iter().map(|x| (x - mean_x) * (x - mean_x)).collect();
    let denominator = reduce::sum(&var);
    if denominator <= 0.0 {
        return Err(GfemError::internal(
            "要素寸法が全て同じなので収束次数を推定できません",
        ));
    }
    Ok(reduce::sum(&cov) / denominator)
}

fn finish(squared_error: f64, squared_reference: f64, gauge_shift: f64) -> ErrorNorms {
    let absolute = squared_error.max(0.0).sqrt();
    let reference = squared_reference.max(0.0).sqrt();
    ErrorNorms {
        absolute,
        reference,
        relative: if reference > 0.0 {
            absolute / reference
        } else {
            f64::NAN
        },
        gauge_shift,
    }
}

fn require_tet4(mesh: &Mesh) -> Result<&crate::mesh::ElementBlock> {
    mesh.block(ElementKind::Tet4).ok_or_else(|| {
        GfemError::unsupported("誤差評価は 1 次四面体（CTETRA）のみに対応しています").with_hint(
            "要素種が混ざったメッシュでは、隣り合う要素が共有する四角形面で\
                 同じ対角線を選ぶ四面体分割が要ります。M7 で対応します（SPEC §5.1 / §8）",
        )
    })
}

/// 体積座標から実座標を求める。
fn interpolate_position(p: &[[f64; 3]; 4], bary: &[f64; 4]) -> [f64; 3] {
    let mut x = [0.0f64; 3];
    for (i, n) in bary.iter().enumerate() {
        for (d, xd) in x.iter_mut().enumerate() {
            *xd += n * p[i][d];
        }
    }
    x
}

/// 体積座標から節点値を補間する。
fn interpolate_scalar(values: &[f64; 4], bary: &[f64; 4]) -> f64 {
    values.iter().zip(bary).map(|(v, n)| v * n).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::nastran::read_nastran_from;
    use crate::units::LengthUnit;
    use std::path::Path;

    fn unit_tet() -> Mesh {
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

    #[test]
    fn exact_linear_potential_has_no_error() {
        // 1 次要素は線形場を厳密に再現する
        let mesh = unit_tet();
        let f = |x: [f64; 3]| 2.0 * x[0] - 3.0 * x[1] + 0.5 * x[2] + 7.0;
        let phi: Vec<f64> = (0..4).map(|i| f(mesh.nodes.coord(i))).collect();
        let e = l2_error_potential(&mesh, &phi, f).unwrap();
        assert!(e.absolute < 1e-15, "{e:?}");
        assert!(e.gauge_shift.abs() < 1e-15);
    }

    #[test]
    fn a_constant_offset_is_removed() {
        // 電位のゲージ（定数分の不定性）は誤差に含めない（SPEC §2.2）
        let mesh = unit_tet();
        let f = |x: [f64; 3]| 2.0 * x[0] - 3.0 * x[1];
        let phi: Vec<f64> = (0..4).map(|i| f(mesh.nodes.coord(i)) + 100.0).collect();
        let e = l2_error_potential(&mesh, &phi, f).unwrap();
        assert!(e.absolute < 1e-13, "{e:?}");
        assert!((e.gauge_shift - 100.0).abs() < 1e-12);
    }

    #[test]
    fn quadrature_integrates_the_reference_norm() {
        // ∫_T 1 dΩ = V なので、φ = 1 の参照ノルムは √V になる
        let mesh = unit_tet();
        let phi = vec![1.0; 4];
        let e = l2_error_potential(&mesh, &phi, |_| 1.0).unwrap();
        assert!((e.reference - (1.0f64 / 6.0).sqrt()).abs() < 1e-15, "{e:?}");
    }

    #[test]
    fn field_error_is_zero_for_a_matching_constant_field() {
        let mesh = unit_tet();
        let field = vec![1.0, -2.0, 3.0];
        let e = l2_error_field(&mesh, &field, |_| [1.0, -2.0, 3.0]).unwrap();
        assert!(e.absolute < 1e-15);
        assert!((e.reference - (14.0f64 / 6.0).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn field_error_measures_the_difference() {
        let mesh = unit_tet();
        let field = vec![1.0, 0.0, 0.0];
        let e = l2_error_field(&mesh, &field, |_| [0.0, 0.0, 0.0]).unwrap();
        // ∫|E_h|² dΩ = 1 × V
        assert!((e.absolute - (1.0f64 / 6.0).sqrt()).abs() < 1e-15);
        assert!(e.relative.is_nan());
    }

    #[test]
    fn mean_size_of_a_unit_tet() {
        let mesh = unit_tet();
        let h = mean_element_size(&mesh).unwrap();
        assert!((h - (1.0f64 / 6.0).cbrt()).abs() < 1e-15);
    }

    #[test]
    fn estimates_a_clean_second_order_slope() {
        let sizes = [0.4, 0.2, 0.1, 0.05];
        let errors: Vec<f64> = sizes.iter().map(|h| 3.0 * h * h).collect();
        let p = convergence_order(&sizes, &errors).unwrap();
        assert!((p - 2.0).abs() < 1e-12, "{p}");
    }

    #[test]
    fn estimates_a_first_order_slope_with_noise() {
        let sizes = [0.4, 0.2, 0.1, 0.05];
        let noise = [1.02, 0.98, 1.01, 0.99];
        let errors: Vec<f64> = sizes.iter().zip(noise).map(|(h, k)| 5.0 * h * k).collect();
        let p = convergence_order(&sizes, &errors).unwrap();
        assert!((p - 1.0).abs() < 0.05, "{p}");
    }

    #[test]
    fn rejects_degenerate_input() {
        assert!(convergence_order(&[0.1], &[1.0]).is_err());
        assert!(convergence_order(&[0.1, 0.2], &[1.0]).is_err());
        assert!(convergence_order(&[0.1, 0.1], &[1.0, 2.0]).is_err());
        assert!(convergence_order(&[0.1, 0.2], &[0.0, 1.0]).is_err());
    }
}
