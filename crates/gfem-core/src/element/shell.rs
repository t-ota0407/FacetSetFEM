//! 面要素（Tria3 / Quad4 / Tria6）の面積と形状関数の面積分（SPEC §2.4）。
//!
//! 電極荷重ベクトルは一様電流密度 `J_a = I / |Γa|` を面積按分で節点に配る:
//!
//! ```text
//! f_i += J_a ∫_F N_i dS         (SPEC §2.4)
//! ```
//!
//! **節点均等分配にはしない。** メッシュが不均一な場合、節点均等では細かく切られた
//! 部分に過剰な電流が集まり、電流密度分布がメッシュ依存に歪むため。
//!
//! 2 次四面体（Tet10）の面は 6 節点の三角形（Tria6）になる。平らな Tria6 では
//! **頂点の ∫N_i dS が 0、辺の中点が A/3 ずつ**になる（頂点の形状関数 `L(2L − 1)` は
//! 面の上で正負の両方の値を取り、ちょうど打ち消し合う）。直感に反するが、
//! 一様電流密度を 2 次の形状関数で正しく配るとこうなる（SPEC §2.4）。

use crate::element::ElementKind;
use crate::geom::{cross, norm, sub};

/// 面の最大節点数（Tria6 の 6）。
pub const MAX_FACE_NODES: usize = 6;

/// 面 1 枚の積分量。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceIntegral {
    /// 面積 [m²]
    pub area: f64,
    /// `∫_F N_i dS` [m²]。先頭 `n_nodes` 個が有効。総和は `area` に等しい
    pub shape_integral: [f64; MAX_FACE_NODES],
    /// 面の節点数（3・4・6）
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
/// - Tria6（2 次三角形）: 6 点則（代数的精度 4 次）。平らなら厳密で、
///   辺が曲がっていてもヤコビアンから積分する
pub fn face_integral(kind: ElementKind, p: &[[f64; 3]]) -> FaceIntegral {
    match kind {
        ElementKind::Tria3 => {
            let u = sub(p[1], p[0]);
            let v = sub(p[2], p[0]);
            let area = 0.5 * norm(cross(u, v));
            let third = area / 3.0;
            FaceIntegral {
                area,
                shape_integral: [third, third, third, 0.0, 0.0, 0.0],
                n_nodes: 3,
            }
        }
        ElementKind::Tria6 => {
            let mut area = 0.0;
            let mut integral = [0.0f64; MAX_FACE_NODES];
            for &(r, s, w) in &TRIANGLE6 {
                let n = tria6_shape(r, s);
                let detj = norm(tria6_normal(p, r, s));
                area += w * detj;
                for i in 0..6 {
                    integral[i] += w * n[i] * detj;
                }
            }
            FaceIntegral {
                area,
                shape_integral: integral,
                n_nodes: 6,
            }
        }
        ElementKind::Quad4 => {
            let mut area = 0.0;
            let mut integral = [0.0f64; MAX_FACE_NODES];
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

/// 面の質量行列 `M_ij = ∫_F N_i N_j dS` [m²]（SPEC §2.5 の完全電極モデル）。
///
/// 先頭 `n × n`（n は面の節点数）が有効で、残りは 0。全成分の和は面積に、
/// 行ごとの和は `∫N_i dS`（[`face_integral`] の `shape_integral`）に等しい。
///
/// - Tria3: `M_ij = A (1 + δ_ij) / 12`。解析的に厳密
/// - Quad4: 2×2 ガウス積分。`N_i N_j |det J|` は ξ・η それぞれについて高々 3 次なので、
///   非平面の四辺形でも厳密
/// - Tria6: 6 点則（代数的精度 4 次）。`N_i N_j` が 4 次なので、平らなら厳密
pub fn face_mass(kind: ElementKind, p: &[[f64; 3]]) -> [[f64; MAX_FACE_NODES]; MAX_FACE_NODES] {
    let mut m = [[0.0f64; MAX_FACE_NODES]; MAX_FACE_NODES];
    match kind {
        ElementKind::Tria3 => {
            let area = 0.5 * norm(cross(sub(p[1], p[0]), sub(p[2], p[0])));
            let off = area / 12.0;
            for (i, row) in m.iter_mut().take(3).enumerate() {
                for (j, v) in row.iter_mut().take(3).enumerate() {
                    *v = if i == j { 2.0 * off } else { off };
                }
            }
        }
        ElementKind::Quad4 => {
            for &eta in &GAUSS2 {
                for &xi in &GAUSS2 {
                    let n = quad4_shape(xi, eta);
                    let (dxi, deta) = quad4_jacobian(p, xi, eta);
                    // 重みは 1 × 1
                    let detj = norm(cross(dxi, deta));
                    for i in 0..4 {
                        for j in 0..4 {
                            m[i][j] += n[i] * n[j] * detj;
                        }
                    }
                }
            }
        }
        ElementKind::Tria6 => {
            for &(r, s, w) in &TRIANGLE6 {
                let n = tria6_shape(r, s);
                let detj = norm(tria6_normal(p, r, s));
                for i in 0..6 {
                    for j in 0..6 {
                        m[i][j] += w * n[i] * n[j] * detj;
                    }
                }
            }
        }
        other => panic!("面要素ではありません: {other:?}"),
    }
    m
}

/// 面の単位法線。節点の並びに対する右ねじの向き。
///
/// 四辺形は 2 本の対角線の外積（＝平均的な法線）を使う。
/// 2 次三角形は面の中心での接ベクトルの外積（平らなら頂点の三角形の法線と同じ向き）。
pub fn face_normal(kind: ElementKind, p: &[[f64; 3]]) -> [f64; 3] {
    let n = match kind {
        ElementKind::Tria3 => cross(sub(p[1], p[0]), sub(p[2], p[0])),
        ElementKind::Quad4 => cross(sub(p[2], p[0]), sub(p[3], p[1])),
        ElementKind::Tria6 => tria6_normal(p, 1.0 / 3.0, 1.0 / 3.0),
        other => panic!("面要素ではありません: {other:?}"),
    };
    let len = norm(n);
    if len > 0.0 {
        [n[0] / len, n[1] / len, n[2] / len]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// 三角形の 6 点則（Dunavant、代数的精度 4 次）。`(r, s, 重み)`。
///
/// 重みは参照三角形の面積 1/2 を掛けた値で、総和は 1/2。
const TRIANGLE6: [(f64, f64, f64); 6] = {
    const A1: f64 = 0.445_948_490_915_964_9;
    const B1: f64 = 0.108_103_018_168_070_2;
    const W1: f64 = 0.5 * 0.223_381_589_678_011_5;
    const A2: f64 = 0.091_576_213_509_770_74;
    const B2: f64 = 0.816_847_572_980_458_5;
    const W2: f64 = 0.5 * 0.109_951_743_655_321_9;
    [
        (A1, A1, W1),
        (A1, B1, W1),
        (B1, A1, W1),
        (A2, A2, W2),
        (A2, B2, W2),
        (B2, A2, W2),
    ]
};

/// 2 次三角形の形状関数。節点の並びは CTRIA6 の G1..G6
/// （頂点 3 つのあとに辺 1-2, 2-3, 3-1 の中点）。
fn tria6_shape(r: f64, s: f64) -> [f64; 6] {
    let l = [1.0 - r - s, r, s];
    [
        l[0] * (2.0 * l[0] - 1.0),
        l[1] * (2.0 * l[1] - 1.0),
        l[2] * (2.0 * l[2] - 1.0),
        4.0 * l[0] * l[1],
        4.0 * l[1] * l[2],
        4.0 * l[2] * l[0],
    ]
}

/// 2 次三角形の `∂x/∂r × ∂x/∂s`。大きさが面積要素、向きが法線。
fn tria6_normal(p: &[[f64; 3]], r: f64, s: f64) -> [f64; 3] {
    let l = [1.0 - r - s, r, s];
    // ∂N_i/∂r, ∂N_i/∂s
    let dr = [
        -(4.0 * l[0] - 1.0),
        4.0 * l[1] - 1.0,
        0.0,
        4.0 * (l[0] - l[1]),
        4.0 * l[2],
        -4.0 * l[2],
    ];
    let ds = [
        -(4.0 * l[0] - 1.0),
        0.0,
        4.0 * l[2] - 1.0,
        -4.0 * l[1],
        4.0 * l[1],
        4.0 * (l[0] - l[2]),
    ];
    let mut tr = [0.0f64; 3];
    let mut ts = [0.0f64; 3];
    for i in 0..6 {
        for d in 0..3 {
            tr[d] += dr[i] * p[i][d];
            ts[d] += ds[i] * p[i][d];
        }
    }
    cross(tr, ts)
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

    #[test]
    #[allow(clippy::needless_range_loop)] // 行列の添字で書くほうが式と対応が取れる
    fn face_mass_rows_sum_to_the_shape_integrals() {
        // Σ_j N_j ≡ 1 なので、質量行列の行和は ∫N_i dS、総和は面積（SPEC §2.5）
        let quad = [
            [0.0, 0.0, 0.0],
            [2.0, 0.3, 0.1],
            [2.4, 3.1, -0.2],
            [-0.2, 2.7, 0.4],
        ];
        let tri = [[0.1, 0.2, 0.3], [1.7, -0.4, 0.9], [0.5, 2.2, -0.6]];
        for (kind, p) in [
            (ElementKind::Quad4, &quad[..]),
            (ElementKind::Tria3, &tri[..]),
        ] {
            let m = face_mass(kind, p);
            let f = face_integral(kind, p);
            for i in 0..f.n_nodes {
                let row: f64 = m[i].iter().sum();
                assert!((row - f.shape_integral[i]).abs() < 1e-14, "{kind:?} 行 {i}");
                for j in 0..4 {
                    assert_eq!(m[i][j], m[j][i], "{kind:?} が対称でない");
                }
            }
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // 行列の添字で書くほうが式と対応が取れる
    fn triangle_mass_is_the_textbook_matrix() {
        // 面積 6 の三角形: 対角 A/6 = 1、非対角 A/12 = 0.5
        let p = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]];
        let m = face_mass(ElementKind::Tria3, &p);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.5 };
                assert!((m[i][j] - expected).abs() < 1e-15, "({i}, {j})");
            }
        }
        assert_eq!(m[3], [0.0; MAX_FACE_NODES]);
    }

    #[test]
    fn quad_mass_integrates_a_product_of_linear_functions_exactly() {
        // 平行四辺形でない平面四辺形で ∫ x y dS を、M を使った xᵀ M y と比べる。
        // x・y は双線形補間で厳密に表せるので、M が厳密なら両者は一致する
        let p = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.5, 1.5, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let m = face_mass(ElementKind::Quad4, &p);
        let x: Vec<f64> = p.iter().map(|q| q[0]).collect();
        let y: Vec<f64> = p.iter().map(|q| q[1]).collect();
        let mut xmy = 0.0;
        for i in 0..4 {
            for j in 0..4 {
                xmy += x[i] * m[i][j] * y[j];
            }
        }
        // 同じ積分を細かい 3 角形分割の数値積分（重心則を 400×400 回）で求める
        let n = 400;
        let mut reference = 0.0;
        for a in 0..n {
            for b in 0..n {
                let (xi, eta) = (
                    -1.0 + (2 * a + 1) as f64 / n as f64,
                    -1.0 + (2 * b + 1) as f64 / n as f64,
                );
                let nn = quad4_shape(xi, eta);
                let (dxi, deta) = quad4_jacobian(&p, xi, eta);
                let detj = norm(cross(dxi, deta));
                let px: f64 = (0..4).map(|k| nn[k] * p[k][0]).sum();
                let py: f64 = (0..4).map(|k| nn[k] * p[k][1]).sum();
                reference += px * py * detj * (2.0 / n as f64).powi(2);
            }
        }
        assert!(
            (xmy - reference).abs() < 1e-5 * reference.abs(),
            "{xmy} vs {reference}"
        );
    }

    /// 頂点 3 つから、辺が真っ直ぐな 6 節点（CTRIA6 の並び）を作る。
    fn straight6(c: [[f64; 3]; 3]) -> [[f64; 3]; 6] {
        let mid = |a: [f64; 3], b: [f64; 3]| [0, 1, 2].map(|d| 0.5 * (a[d] + b[d]));
        [
            c[0],
            c[1],
            c[2],
            mid(c[0], c[1]),
            mid(c[1], c[2]),
            mid(c[2], c[0]),
        ]
    }

    #[test]
    fn tria6_rule_integrates_quartics_exactly() {
        // 参照三角形で ∫ r^a s^b = a! b! / (a + b + 2)!
        let fact = |n: u32| (1..=n).map(f64::from).product::<f64>();
        let weight: f64 = TRIANGLE6.iter().map(|q| q.2).sum();
        assert!((weight - 0.5).abs() < 1e-15);
        for a in 0..=4u32 {
            for b in 0..=(4 - a) {
                let exact = fact(a) * fact(b) / fact(a + b + 2);
                let rule: f64 = TRIANGLE6
                    .iter()
                    .map(|&(r, s, w)| w * r.powi(a as i32) * s.powi(b as i32))
                    .sum();
                assert!(
                    (rule - exact).abs() < 1e-15,
                    "r^{a} s^{b}: {rule} vs {exact}"
                );
            }
        }
    }

    #[test]
    fn a_flat_tria6_puts_the_current_on_its_mid_side_nodes() {
        // SPEC §2.4: 平らな 2 次三角形では、頂点に 0、辺の中点に A/3 ずつ
        let p = straight6([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.5, 4.0, 0.0]]);
        let f = face_integral(ElementKind::Tria6, &p);
        assert_eq!(f.n_nodes, 6);
        assert!((f.area - 6.0).abs() < 1e-14);
        for i in 0..3 {
            assert!(
                f.shape_integral[i].abs() < 1e-14,
                "頂点 {i}: {}",
                f.shape_integral[i]
            );
        }
        for i in 3..6 {
            assert!((f.shape_integral[i] - 2.0).abs() < 1e-14, "中点 {i}");
        }
        // 法線は頂点の三角形と同じ
        let corners = [p[0], p[1], p[2]];
        let (a, b) = (
            face_normal(ElementKind::Tria6, &p),
            face_normal(ElementKind::Tria3, &corners),
        );
        for d in 0..3 {
            assert!((a[d] - b[d]).abs() < 1e-15, "{a:?} vs {b:?}");
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // 行列の添字で書くほうが式と対応が取れる
    fn tria6_mass_integrates_products_of_quadratics() {
        // 2 次式 u, v を節点値で与えると、uᵀ M v = ∫ u v dS が厳密に出る（平らな面）
        let p = straight6([[0.2, -0.1, 1.0], [2.3, 0.4, 1.0], [0.6, 1.9, 1.0]]);
        let m = face_mass(ElementKind::Tria6, &p);
        let u_of = |x: [f64; 3]| 1.0 + x[0] - 0.5 * x[1] + 0.3 * x[0] * x[1];
        let v_of = |x: [f64; 3]| 0.7 * x[0] * x[0] - x[1] + 2.0;
        let (u, v): (Vec<f64>, Vec<f64>) = p.iter().map(|&x| (u_of(x), v_of(x))).unzip();
        let mut umv = 0.0;
        for i in 0..6 {
            for j in 0..6 {
                umv += u[i] * m[i][j] * v[j];
                assert!((m[i][j] - m[j][i]).abs() < 1e-16, "非対称 ({i}, {j})");
            }
        }
        // 同じ積分を、頂点の三角形の上で直接求める（被積分関数は 4 次なので 6 点則で厳密）
        let area = 0.5 * norm(cross(sub(p[1], p[0]), sub(p[2], p[0])));
        let mut reference = 0.0;
        for &(r, s, w) in &TRIANGLE6 {
            let x = [0, 1, 2].map(|d| p[0][d] + r * (p[1][d] - p[0][d]) + s * (p[2][d] - p[0][d]));
            reference += 2.0 * area * w * u_of(x) * v_of(x);
        }
        assert!(
            (umv - reference).abs() < 1e-13 * reference.abs(),
            "{umv} vs {reference}"
        );
        // 行和は ∫N_i dS
        let f = face_integral(ElementKind::Tria6, &p);
        for i in 0..6 {
            let row: f64 = m[i].iter().sum();
            assert!((row - f.shape_integral[i]).abs() < 1e-14, "行 {i}");
        }
    }

    #[test]
    fn a_curved_tria6_has_a_larger_area_than_its_corners() {
        // 中点を面の外へ持ち上げると、頂点の三角形より面積が大きくなる。∫N_i dS の総和は面積
        let mut p = straight6([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        for m in &mut p[3..] {
            m[2] = 0.1;
        }
        let f = face_integral(ElementKind::Tria6, &p);
        assert!(f.area > 0.5, "{}", f.area);
        let sum: f64 = f.shape_integral.iter().sum();
        assert!((sum - f.area).abs() < 1e-15);
    }
}
