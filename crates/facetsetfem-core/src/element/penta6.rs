//! 1 次五面体（Penta6、三角柱）の形状関数（SPEC §2.3）。
//!
//! 局所座標は三角形の面積座標 `(r, s)` と高さ方向 `ζ ∈ [−1, 1]` の組。
//! 節点の並びは Nastran の CPENTA G1..G6（G1..G3 が底面 ζ = −1、G4..G6 が上面）。
//!
//! ```text
//! N_1 = (1−r−s)(1−ζ)/2   N_4 = (1−r−s)(1+ζ)/2
//! N_2 = r(1−ζ)/2         N_5 = r(1+ζ)/2
//! N_3 = s(1−ζ)/2         N_6 = s(1+ζ)/2
//! ```
//!
//! 積分は三角形 3 点 × 高さ方向 2 点の 6 点（SPEC §2.3）。

use crate::element::shape::{MAX_ELEMENT_NODES, QuadraturePoint, ShapeFunctions};

/// Penta6 の形状関数。
#[derive(Debug, Clone, Copy)]
pub struct Penta6;

/// 三角形の 3 点則（代数的精度 2 次）の面積座標。
const TRIANGLE: [[f64; 2]; 3] = [
    [1.0 / 6.0, 1.0 / 6.0],
    [2.0 / 3.0, 1.0 / 6.0],
    [1.0 / 6.0, 2.0 / 3.0],
];

/// 三角形 3 点則の重み。参照三角形の面積 1/2 を 3 等分している。
const TRIANGLE_WEIGHT: f64 = 1.0 / 6.0;

/// 2 点ガウス求積の座標（±1/√3）。
const GAUSS: [f64; 2] = [-0.577_350_269_189_625_8, 0.577_350_269_189_625_8];

/// 6 点の積分点。重みの総和は参照三角柱の体積 1（= 1/2 × 2）になる。
static QUADRATURE: [QuadraturePoint; 6] = [
    QuadraturePoint {
        local: [TRIANGLE[0][0], TRIANGLE[0][1], GAUSS[0]],
        weight: TRIANGLE_WEIGHT,
    },
    QuadraturePoint {
        local: [TRIANGLE[1][0], TRIANGLE[1][1], GAUSS[0]],
        weight: TRIANGLE_WEIGHT,
    },
    QuadraturePoint {
        local: [TRIANGLE[2][0], TRIANGLE[2][1], GAUSS[0]],
        weight: TRIANGLE_WEIGHT,
    },
    QuadraturePoint {
        local: [TRIANGLE[0][0], TRIANGLE[0][1], GAUSS[1]],
        weight: TRIANGLE_WEIGHT,
    },
    QuadraturePoint {
        local: [TRIANGLE[1][0], TRIANGLE[1][1], GAUSS[1]],
        weight: TRIANGLE_WEIGHT,
    },
    QuadraturePoint {
        local: [TRIANGLE[2][0], TRIANGLE[2][1], GAUSS[1]],
        weight: TRIANGLE_WEIGHT,
    },
];

impl ShapeFunctions for Penta6 {
    const NODES: usize = 6;
    const CENTRE: [f64; 3] = [1.0 / 3.0, 1.0 / 3.0, 0.0];

    fn quadrature() -> &'static [QuadraturePoint] {
        &QUADRATURE
    }

    fn values(local: [f64; 3]) -> [f64; MAX_ELEMENT_NODES] {
        let (r, s, zeta) = (local[0], local[1], local[2]);
        let l = [1.0 - r - s, r, s];
        let lower = 0.5 * (1.0 - zeta);
        let upper = 0.5 * (1.0 + zeta);
        let mut out = [0.0f64; MAX_ELEMENT_NODES];
        for i in 0..3 {
            out[i] = l[i] * lower;
            out[i + 3] = l[i] * upper;
        }
        out
    }

    fn local_gradients(local: [f64; 3]) -> [[f64; 3]; MAX_ELEMENT_NODES] {
        let (r, s, zeta) = (local[0], local[1], local[2]);
        let l = [1.0 - r - s, r, s];
        // ∂L_i/∂r, ∂L_i/∂s
        let dl = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]];
        let lower = 0.5 * (1.0 - zeta);
        let upper = 0.5 * (1.0 + zeta);
        let mut out = [[0.0f64; 3]; MAX_ELEMENT_NODES];
        for i in 0..3 {
            out[i] = [dl[i][0] * lower, dl[i][1] * lower, -0.5 * l[i]];
            out[i + 3] = [dl[i][0] * upper, dl[i][1] * upper, 0.5 * l[i]];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::shape::{electric_field, testing, volume};

    /// 節点の局所座標（CPENTA G1..G6 の並び）。
    const LOCAL_NODES: [[f64; 3]; 6] = [
        [0.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
        [0.0, 1.0, -1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ];

    /// 底面が直角三角形（脚 `a`, `b`）、高さ `h` の三角柱。
    fn prism(a: f64, b: f64, h: f64) -> Vec<[f64; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [a, 0.0, 0.0],
            [0.0, b, 0.0],
            [0.0, 0.0, h],
            [a, 0.0, h],
            [0.0, b, h],
        ]
    }

    #[test]
    fn quadrature_weights_sum_to_the_reference_volume() {
        testing::assert_quadrature_weight(1.0, Penta6::quadrature());
    }

    #[test]
    fn shape_functions_are_nodal() {
        for (i, node) in LOCAL_NODES.iter().enumerate() {
            let values = Penta6::values(*node);
            for (j, &value) in values[..6].iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (value - expected).abs() < 1e-15,
                    "N_{j}(節点 {i}) = {value}"
                );
            }
        }
    }

    #[test]
    fn partition_of_unity() {
        testing::assert_partition_of_unity::<Penta6>(&prism(2.0, 3.0, 1.5));
    }

    #[test]
    fn reproduces_a_linear_field_exactly() {
        testing::assert_reproduces_linear_field::<Penta6>(
            &prism(2.0, 3.0, 1.5),
            [0.5, 1.25, -3.0],
            -2.0,
        );
    }

    #[test]
    fn volume_of_a_prism() {
        // 直角三角形の面積 a·b/2 × 高さ h
        let v = volume::<Penta6>(&prism(2.0, 3.0, 1.5)).unwrap();
        assert!((v - 0.5 * 2.0 * 3.0 * 1.5).abs() < 1e-14, "{v}");
    }

    #[test]
    fn volume_of_a_sheared_prism() {
        let mut points = prism(2.0, 3.0, 1.5);
        for p in &mut points {
            p[0] += 0.4 * p[2];
            p[1] -= 0.2 * p[2];
        }
        let v = volume::<Penta6>(&points).unwrap();
        assert!((v - 4.5).abs() < 1e-14, "{v}");
    }

    #[test]
    fn stiffness_is_symmetric_with_a_constant_nullspace() {
        testing::assert_stiffness_properties::<Penta6>(&prism(2.0, 3.0, 1.5), 0.35);
    }

    #[test]
    fn inverted_node_order_is_rejected() {
        let points = prism(1.0, 1.0, 1.0);
        let flipped: Vec<[f64; 3]> = points[3..]
            .iter()
            .chain(points[..3].iter())
            .copied()
            .collect();
        let e = volume::<Penta6>(&flipped).unwrap_err();
        assert!(e.to_string().contains("ヤコビアン"), "{e}");
    }

    #[test]
    fn field_at_the_centre_of_a_linear_solution() {
        let points = prism(2.0, 3.0, 1.5);
        let a = [1.0, -0.5, 2.0];
        let phi: Vec<f64> = points
            .iter()
            .map(|p| a[0] * p[0] + a[1] * p[1] + a[2] * p[2])
            .collect();
        let field = electric_field::<Penta6>(&points, &phi, Penta6::CENTRE);
        for d in 0..3 {
            assert!((field[d] + a[d]).abs() < 1e-13);
        }
    }
}
