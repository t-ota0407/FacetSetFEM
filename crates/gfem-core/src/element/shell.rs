//! 面要素（Tria3 / Quad4）の面積と形状関数の面積分（SPEC §2.4）。
//!
//! 電極荷重ベクトルは一様電流密度 `J_a = I / |Γa|` を面積按分で節点に配る:
//!
//! ```text
//! f_i += J_a ∫_F N_i dS         (SPEC §2.4)
//! ```
//!
//! **節点均等分配にはしない。** メッシュが不均一な場合、節点均等では細かく切られた
//! 部分に過剰な電流が集まり、電流密度分布がメッシュ依存に歪むため。

use crate::element::ElementKind;
use crate::geom::{cross, norm, sub};

/// 面 1 枚の積分量。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceIntegral {
    /// 面積 [m²]
    pub area: f64,
    /// `∫_F N_i dS` [m²]。先頭 `n_nodes` 個が有効。総和は `area` に等しい
    pub shape_integral: [f64; 4],
    /// 面の節点数（3 または 4）
    pub n_nodes: usize,
}

/// 2 点ガウス求積の座標と重み（区間 [−1, 1]）。
const GAUSS2: [f64; 2] = [
    -0.577_350_269_189_625_8, // −1/√3
    0.577_350_269_189_625_8,
];

/// 面の面積と `∫N_i dS` を求める。
///
/// - Tria3（線形三角形）: `∫N_i dS = A/3`。解析的に厳密
/// - Quad4（双線形四辺形）: 2×2 ガウス積分。ヤコビアンから積分するので、
///   非平面の四辺形でも面積が正しく出る
pub fn face_integral(kind: ElementKind, p: &[[f64; 3]]) -> FaceIntegral {
    match kind {
        ElementKind::Tria3 => {
            let u = sub(p[1], p[0]);
            let v = sub(p[2], p[0]);
            let area = 0.5 * norm(cross(u, v));
            let third = area / 3.0;
            FaceIntegral {
                area,
                shape_integral: [third, third, third, 0.0],
                n_nodes: 3,
            }
        }
        ElementKind::Quad4 => {
            let mut area = 0.0;
            let mut integral = [0.0f64; 4];
            for &eta in &GAUSS2 {
                for &xi in &GAUSS2 {
                    let n = quad4_shape(xi, eta);
                    let (dxi, deta) = quad4_jacobian(p, xi, eta);
                    // 重みは 1 × 1
                    let detj = norm(cross(dxi, deta));
                    area += detj;
                    for i in 0..4 {
                        integral[i] += n[i] * detj;
                    }
                }
            }
            FaceIntegral {
                area,
                shape_integral: integral,
                n_nodes: 4,
            }
        }
        other => panic!("面要素ではありません: {other:?}"),
    }
}

/// 面の単位法線。節点の並びに対する右ねじの向き。
///
/// 四辺形は 2 本の対角線の外積（＝平均的な法線）を使う。
pub fn face_normal(kind: ElementKind, p: &[[f64; 3]]) -> [f64; 3] {
    let n = match kind {
        ElementKind::Tria3 => cross(sub(p[1], p[0]), sub(p[2], p[0])),
        ElementKind::Quad4 => cross(sub(p[2], p[0]), sub(p[3], p[1])),
        other => panic!("面要素ではありません: {other:?}"),
    };
    let len = norm(n);
    if len > 0.0 {
        [n[0] / len, n[1] / len, n[2] / len]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// 双線形四辺形の形状関数。
fn quad4_shape(xi: f64, eta: f64) -> [f64; 4] {
    [
        0.25 * (1.0 - xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 + eta),
        0.25 * (1.0 - xi) * (1.0 + eta),
    ]
}

/// 双線形四辺形の接ベクトル `(∂x/∂ξ, ∂x/∂η)`。
fn quad4_jacobian(p: &[[f64; 3]], xi: f64, eta: f64) -> ([f64; 3], [f64; 3]) {
    let dn_dxi = [
        -0.25 * (1.0 - eta),
        0.25 * (1.0 - eta),
        0.25 * (1.0 + eta),
        -0.25 * (1.0 + eta),
    ];
    let dn_deta = [
        -0.25 * (1.0 - xi),
        -0.25 * (1.0 + xi),
        0.25 * (1.0 + xi),
        0.25 * (1.0 - xi),
    ];
    let mut dxi = [0.0f64; 3];
    let mut deta = [0.0f64; 3];
    for i in 0..4 {
        for d in 0..3 {
            dxi[d] += dn_dxi[i] * p[i][d];
            deta[d] += dn_deta[i] * p[i][d];
        }
    }
    (dxi, deta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangle_area_and_split() {
        let p = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]];
        let f = face_integral(ElementKind::Tria3, &p);
        assert!((f.area - 6.0).abs() < 1e-15);
        for i in 0..3 {
            assert!((f.shape_integral[i] - 2.0).abs() < 1e-15);
        }
    }

    #[test]
    fn planar_quad_splits_evenly() {
        // SPEC §2.4: 平面四辺形なら各節点に A/4
        let p = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 3.0, 0.0],
            [0.0, 3.0, 0.0],
        ];
        let f = face_integral(ElementKind::Quad4, &p);
        assert!((f.area - 6.0).abs() < 1e-14);
        for i in 0..4 {
            assert!(
                (f.shape_integral[i] - 1.5).abs() < 1e-14,
                "{:?}",
                f.shape_integral
            );
        }
    }

    #[test]
    fn shape_integrals_sum_to_the_area() {
        // Σ N_i ≡ 1 なので、どんな形でも総和は面積に一致する
        let quad = [
            [0.0, 0.0, 0.0],
            [2.0, 0.3, 0.1],
            [2.4, 3.1, -0.2],
            [-0.2, 2.7, 0.4],
        ];
        let f = face_integral(ElementKind::Quad4, &quad);
        let sum: f64 = f.shape_integral.iter().sum();
        assert!((sum - f.area).abs() < 1e-14);

        let tri = [[0.1, 0.2, 0.3], [1.7, -0.4, 0.9], [0.5, 2.2, -0.6]];
        let g = face_integral(ElementKind::Tria3, &tri);
        let sum: f64 = g.shape_integral.iter().sum();
        assert!((sum - g.area).abs() < 1e-15);
    }

    #[test]
    fn quad_area_matches_two_triangles_when_planar() {
        let p = [
            [0.0, 0.0, 1.0],
            [1.5, 0.2, 1.0],
            [1.9, 2.4, 1.0],
            [-0.3, 1.8, 1.0],
        ];
        let quad = face_integral(ElementKind::Quad4, &p).area;
        let t1 = face_integral(ElementKind::Tria3, &[p[0], p[1], p[2]]).area;
        let t2 = face_integral(ElementKind::Tria3, &[p[0], p[2], p[3]]).area;
        assert!((quad - (t1 + t2)).abs() < 1e-13, "{quad} vs {}", t1 + t2);
    }

    #[test]
    fn triangle_normal_follows_the_right_hand_rule() {
        let p = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert_eq!(face_normal(ElementKind::Tria3, &p), [0.0, 0.0, 1.0]);
    }
}
