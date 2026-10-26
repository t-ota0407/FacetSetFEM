//! 1 次四面体（Tet4）の形状関数と要素剛性（SPEC §2.3）。
//!
//! 形状関数 N_i は体積座標そのものなので ∇N_i は要素内で一定になり、
//! 要素剛性は数値積分なしに解析的に書ける:
//!
//! ```text
//! K^e_ij = V_e · (∇N_i · σ ∇N_j)      (SPEC §2.3)
//! ```
//!
//! この結果として **E は要素内で完全に一定**（区分定数）になる。

use crate::element::shape::{MAX_ELEMENT_NODES, QuadraturePoint, ShapeFunctions};
use crate::geom::{cross, dot, scale, sub};
use crate::material::Conductivity;

/// Tet4 の形状関数勾配と体積。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tet4Shape {
    /// ∇N_i [1/m]。要素内で一定
    pub grad: [[f64; 3]; 4],
    /// 体積 |V| [m³]。要素剛性にはこちらを使う
    pub volume: f64,
    /// 符号付き体積 [m³]。負なら節点順序が反転している
    pub signed_volume: f64,
}

/// 節点座標（m）から形状関数勾配と体積を求める。
///
/// 局所座標を (ξ, η, ζ)、`N1 = ξ, N2 = η, N3 = ζ, N0 = 1 − ξ − η − ζ` とすると、
/// ヤコビ行列 `J = [a b c]`（a = p1−p0, b = p2−p0, c = p3−p0）に対して
/// ∇N_1..3 は `J⁻¹` の各行、∇N_0 はその総和の符号反転になる。
/// `J⁻¹` の行はそのまま外積で書けるので、行列反転を明示的に組む必要はない。
pub fn tet4_shape(p: &[[f64; 3]; 4]) -> Tet4Shape {
    let a = sub(p[1], p[0]);
    let b = sub(p[2], p[0]);
    let c = sub(p[3], p[0]);

    // J⁻¹ の各行 = 余因子ベクトル / det
    let bc = cross(b, c);
    let ca = cross(c, a);
    let ab = cross(a, b);
    let det = dot(a, bc); // = 6V

    let inv = if det != 0.0 { 1.0 / det } else { 0.0 };
    let g1 = scale(bc, inv);
    let g2 = scale(ca, inv);
    let g3 = scale(ab, inv);
    // Σ_i N_i ≡ 1 なので Σ_i ∇N_i = 0
    let g0 = [
        -(g1[0] + g2[0] + g3[0]),
        -(g1[1] + g2[1] + g3[1]),
        -(g1[2] + g2[2] + g3[2]),
    ];

    let signed_volume = det / 6.0;
    Tet4Shape {
        grad: [g0, g1, g2, g3],
        volume: signed_volume.abs(),
        signed_volume,
    }
}

/// 要素剛性行列 `K^e`（SPEC §2.3）。対称 4×4。
///
/// σ をスカラー倍ではなく [`Conductivity::apply`] 経由で作用させているので、
/// 異方性導電率（SPEC §9）を入れてもこの関数は変更不要。
pub fn tet4_stiffness(shape: &Tet4Shape, sigma: &Conductivity) -> [[f64; 4]; 4] {
    let mut k = [[0.0f64; 4]; 4];
    let v = shape.volume;
    for j in 0..4 {
        let sg = sigma.apply(shape.grad[j]);
        for (i, ki) in k.iter_mut().enumerate() {
            ki[j] = v * dot(shape.grad[i], sg);
        }
    }
    k
}

/// [`tet4_stiffness`] の `row` 行目だけ（並列アセンブリ用）。
///
/// 全体を作るときと同じ式を同じ順で計算するので、値は全体の同じ行とビット単位で同じ。
pub fn tet4_stiffness_row(shape: &Tet4Shape, sigma: &Conductivity, row: usize) -> [f64; 4] {
    let mut k = [0.0f64; 4];
    let v = shape.volume;
    for (j, kj) in k.iter_mut().enumerate() {
        let sg = sigma.apply(shape.grad[j]);
        *kj = v * dot(shape.grad[row], sg);
    }
    k
}

/// 要素内の電界 `E = −∇φ`（SPEC §5.1）。Tet4 では要素内で一定。
pub fn tet4_electric_field(shape: &Tet4Shape, phi: &[f64; 4]) -> [f64; 3] {
    let mut e = [0.0f64; 3];
    for (grad, &p) in shape.grad.iter().zip(phi) {
        for (d, ed) in e.iter_mut().enumerate() {
            *ed -= p * grad[d];
        }
    }
    e
}

/// Tet4 の 1 点積分則。∇N が要素内で一定なので 1 点で厳密になる。
static QUADRATURE: [QuadraturePoint; 1] = [QuadraturePoint {
    local: [0.25, 0.25, 0.25],
    // 参照四面体の体積
    weight: 1.0 / 6.0,
}];

impl ShapeFunctions for Tet4 {
    const NODES: usize = 4;
    const CENTRE: [f64; 3] = [0.25, 0.25, 0.25];

    fn quadrature() -> &'static [QuadraturePoint] {
        &QUADRATURE
    }

    fn values(local: [f64; 3]) -> [f64; MAX_ELEMENT_NODES] {
        let mut out = [0.0f64; MAX_ELEMENT_NODES];
        out[0] = 1.0 - local[0] - local[1] - local[2];
        out[1] = local[0];
        out[2] = local[1];
        out[3] = local[2];
        out
    }

    fn local_gradients(_local: [f64; 3]) -> [[f64; 3]; MAX_ELEMENT_NODES] {
        let mut out = [[0.0f64; 3]; MAX_ELEMENT_NODES];
        out[0] = [-1.0, -1.0, -1.0];
        out[1] = [1.0, 0.0, 0.0];
        out[2] = [0.0, 1.0, 0.0];
        out[3] = [0.0, 0.0, 1.0];
        out
    }
}

/// Tet4 の形状関数（[`ShapeFunctions`] 用のマーカー）。
///
/// Tet4 は ∇N が要素内で一定なので、共通のアイソパラメトリック経路を通さず
/// [`tet4_shape`] で直接計算する方が速い。この実装は
/// 「共通経路と結果が一致すること」をテストで確かめるために置いてある。
#[derive(Debug, Clone, Copy)]
pub struct Tet4;

#[cfg(test)]
mod tests {
    use super::*;

    const REF: [[f64; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];

    #[test]
    fn reference_tet_gradients() {
        let s = tet4_shape(&REF);
        assert!((s.volume - 1.0 / 6.0).abs() < 1e-15);
        // N1 = x, N2 = y, N3 = z, N0 = 1 − x − y − z
        assert_eq!(s.grad[1], [1.0, 0.0, 0.0]);
        assert_eq!(s.grad[2], [0.0, 1.0, 0.0]);
        assert_eq!(s.grad[3], [0.0, 0.0, 1.0]);
        assert_eq!(s.grad[0], [-1.0, -1.0, -1.0]);
    }

    #[test]
    fn gradients_sum_to_zero() {
        // 一般の四面体でも Σ ∇N_i = 0（Σ N_i ≡ 1 の帰結）
        let p = [
            [0.13, -0.4, 0.7],
            [1.9, 0.2, -0.3],
            [-0.5, 2.1, 0.4],
            [0.3, 0.1, 3.3],
        ];
        let s = tet4_shape(&p);
        for d in 0..3 {
            let sum: f64 = (0..4).map(|i| s.grad[i][d]).sum();
            assert!(sum.abs() < 1e-14, "{sum}");
        }
    }

    #[test]
    fn gradients_reproduce_a_linear_field_exactly() {
        // φ(x) = 2x − 3y + 0.5z + 7 を節点に与えると、∇φ が厳密に再現される
        let p = [
            [0.1, 0.2, 0.3],
            [1.1, 0.0, 0.4],
            [0.2, 1.3, -0.1],
            [-0.3, 0.4, 1.2],
        ];
        let s = tet4_shape(&p);
        let f = |q: [f64; 3]| 2.0 * q[0] - 3.0 * q[1] + 0.5 * q[2] + 7.0;
        let phi = [f(p[0]), f(p[1]), f(p[2]), f(p[3])];
        let e = tet4_electric_field(&s, &phi);
        // E = −∇φ
        assert!((e[0] + 2.0).abs() < 1e-13, "{e:?}");
        assert!((e[1] - 3.0).abs() < 1e-13, "{e:?}");
        assert!((e[2] + 0.5).abs() < 1e-13, "{e:?}");
    }

    #[test]
    fn a_single_row_is_bitwise_the_row_of_the_full_matrix() {
        // 並列アセンブリは 1 行ずつ作る。全体を作った行とビット単位で同じでなければならない
        let p = [
            [0.1, 0.0, 0.2],
            [2.0, 0.3, 0.0],
            [0.3, 1.7, 0.1],
            [0.4, 0.6, 2.5],
        ];
        let s = tet4_shape(&p);
        for sigma in [
            Conductivity::isotropic(0.35).unwrap(),
            Conductivity::isotropic(1.0 / 3.0).unwrap(),
        ] {
            let full = tet4_stiffness(&s, &sigma);
            for (row, expected) in full.iter().enumerate() {
                let got = tet4_stiffness_row(&s, &sigma, row);
                assert_eq!(
                    got.map(f64::to_bits),
                    expected.map(f64::to_bits),
                    "行 {row}"
                );
            }
        }
    }

    #[test]
    fn stiffness_is_symmetric_and_singular_for_constants() {
        let p = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.3, 1.7, 0.0],
            [0.4, 0.6, 2.5],
        ];
        let s = tet4_shape(&p);
        let k = tet4_stiffness(&s, &Conductivity::isotropic(0.35).unwrap());
        for (i, row) in k.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                assert!((v - k[j][i]).abs() < 1e-18, "非対称: {i},{j}");
            }
        }
        // 定数ベクトルは零空間に入る（SPEC §2.2）
        for row in &k {
            let sum: f64 = row.iter().sum();
            assert!(sum.abs() < 1e-16, "行和が 0 でない: {sum}");
        }
        // 対角は正
        assert!((0..4).all(|i| k[i][i] > 0.0));
    }

    #[test]
    fn inverted_node_order_gives_the_same_stiffness() {
        // 節点順序が反転していても |V| を使うので剛性は変わらない（SPEC §3.1 の警告方針）
        let p = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.3, 1.7, 0.0],
            [0.4, 0.6, 2.5],
        ];
        let q = [p[0], p[1], p[3], p[2]];
        let sigma = Conductivity::isotropic(0.35).unwrap();
        let kp = tet4_stiffness(&tet4_shape(&p), &sigma);
        let kq = tet4_stiffness(&tet4_shape(&q), &sigma);
        assert!(tet4_shape(&q).signed_volume < 0.0);
        // 節点 2 と 3 を入れ替えた並びで比較する
        let perm = [0, 1, 3, 2];
        for (i, row) in kp.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                assert!((v - kq[perm[i]][perm[j]]).abs() < 1e-18);
            }
        }
    }

    #[test]
    fn matches_the_common_isoparametric_path() {
        // 専用の解析的な計算と、共通経路（1 点ガウス積分）が一致することを確かめる。
        // どちらかを直したときにもう一方とずれたら、ここで落ちる
        use crate::element::shape::{self, MAX_ELEMENT_NODES};
        let p = [
            [0.13, -0.4, 0.7],
            [1.9, 0.2, -0.3],
            [-0.5, 2.1, 0.4],
            [0.3, 0.1, 3.3],
        ];
        let sigma = Conductivity::isotropic(0.35).unwrap();

        let direct = tet4_stiffness(&tet4_shape(&p), &sigma);
        let mut common = [[0.0f64; MAX_ELEMENT_NODES]; MAX_ELEMENT_NODES];
        let volume = shape::stiffness::<Tet4>(&p, &sigma, &mut common).unwrap();

        assert!((volume - tet4_shape(&p).volume).abs() < 1e-15);
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (direct[i][j] - common[i][j]).abs() < 1e-15 * direct[i][i].abs().max(1.0),
                    "({i},{j}): {} vs {}",
                    direct[i][j],
                    common[i][j]
                );
            }
        }

        // 電界も一致する
        let phi = [1.0, -2.0, 0.5, 3.0];
        let a = tet4_electric_field(&tet4_shape(&p), &phi);
        let b = shape::electric_field::<Tet4>(&p, &phi, Tet4::CENTRE);
        for d in 0..3 {
            assert!((a[d] - b[d]).abs() < 1e-13, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn satisfies_the_common_element_checks() {
        use crate::element::shape::testing;
        let p = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.3, 1.7, 0.0],
            [0.4, 0.6, 2.5],
        ];
        testing::assert_quadrature_weight(1.0 / 6.0, Tet4::quadrature());
        testing::assert_partition_of_unity::<Tet4>(&p);
        testing::assert_reproduces_linear_field::<Tet4>(&p, [1.5, -2.0, 0.25], 7.0);
        testing::assert_stiffness_properties::<Tet4>(&p, 0.35);
    }

    #[test]
    fn stiffness_scales_linearly_with_sigma() {
        let s = tet4_shape(&REF);
        let a = tet4_stiffness(&s, &Conductivity::isotropic(1.0).unwrap());
        let b = tet4_stiffness(&s, &Conductivity::isotropic(2.5).unwrap());
        for (ra, rb) in a.iter().zip(&b) {
            for (x, y) in ra.iter().zip(rb) {
                assert!((2.5 * x - y).abs() < 1e-18);
            }
        }
    }
}
