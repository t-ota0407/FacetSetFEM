//! 1 次六面体（Hex8）の形状関数（SPEC §2.3）。
//!
//! 局所座標は `ξ, η, ζ ∈ [−1, 1]`、節点の並びは Nastran の CHEXA G1..G8
//! （G1..G4 が底面 ζ = −1、G5..G8 が上面 ζ = +1、いずれも反時計回り）。
//!
//! ```text
//! N_i = (1/8)(1 + ξξ_i)(1 + ηη_i)(1 + ζζ_i)
//! ```
//!
//! 積分は 2×2×2 の 8 点ガウス求積。

use crate::element::shape::{MAX_ELEMENT_NODES, QuadraturePoint, ShapeFunctions};

/// Hex8 の形状関数。
#[derive(Debug, Clone, Copy)]
pub struct Hex8;

/// 節点の局所座標（CHEXA G1..G8 の並び）。
const NODES: [[f64; 3]; 8] = [
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
];

/// 2 点ガウス求積の座標（±1/√3）。
const GAUSS: [f64; 2] = [-0.577_350_269_189_625_8, 0.577_350_269_189_625_8];

/// 2×2×2 の積分点。重みは全て 1 で、総和は参照立方体の体積 8 になる。
static QUADRATURE: [QuadraturePoint; 8] = build_quadrature();

const fn build_quadrature() -> [QuadraturePoint; 8] {
    let mut out = [QuadraturePoint {
        local: [0.0; 3],
        weight: 1.0,
    }; 8];
    let mut n = 0;
    let mut k = 0;
    while k < 2 {
        let mut j = 0;
        while j < 2 {
            let mut i = 0;
            while i < 2 {
                out[n] = QuadraturePoint {
                    local: [GAUSS[i], GAUSS[j], GAUSS[k]],
                    weight: 1.0,
                };
                n += 1;
                i += 1;
            }
            j += 1;
        }
        k += 1;
    }
    out
}

impl ShapeFunctions for Hex8 {
    const NODES: usize = 8;
    const CENTRE: [f64; 3] = [0.0, 0.0, 0.0];

    fn quadrature() -> &'static [QuadraturePoint] {
        &QUADRATURE
    }

    fn values(local: [f64; 3]) -> [f64; MAX_ELEMENT_NODES] {
        let mut out = [0.0f64; MAX_ELEMENT_NODES];
        for (i, n) in NODES.iter().enumerate() {
            out[i] =
                0.125 * (1.0 + local[0] * n[0]) * (1.0 + local[1] * n[1]) * (1.0 + local[2] * n[2]);
        }
        out
    }

    fn local_gradients(local: [f64; 3]) -> [[f64; 3]; MAX_ELEMENT_NODES] {
        let mut out = [[0.0f64; 3]; MAX_ELEMENT_NODES];
        for (i, n) in NODES.iter().enumerate() {
            let f = [
                1.0 + local[0] * n[0],
                1.0 + local[1] * n[1],
                1.0 + local[2] * n[2],
            ];
            out[i] = [
                0.125 * n[0] * f[1] * f[2],
                0.125 * n[1] * f[0] * f[2],
                0.125 * n[2] * f[0] * f[1],
            ];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::shape::{electric_field, testing, volume};

    /// 一辺 `a` の立方体を原点から伸ばしたもの。
    fn cube(a: f64) -> Vec<[f64; 3]> {
        NODES
            .iter()
            .map(|n| {
                [
                    0.5 * a * (n[0] + 1.0),
                    0.5 * a * (n[1] + 1.0),
                    0.5 * a * (n[2] + 1.0),
                ]
            })
            .collect()
    }

    /// 各方向に違う長さの直方体。
    fn brick(lx: f64, ly: f64, lz: f64) -> Vec<[f64; 3]> {
        NODES
            .iter()
            .map(|n| {
                [
                    0.5 * lx * (n[0] + 1.0),
                    0.5 * ly * (n[1] + 1.0),
                    0.5 * lz * (n[2] + 1.0),
                ]
            })
            .collect()
    }

    #[test]
    fn quadrature_weights_sum_to_the_reference_volume() {
        testing::assert_quadrature_weight(8.0, Hex8::quadrature());
    }

    #[test]
    fn shape_functions_are_nodal() {
        // N_i は自分の節点で 1、他の節点で 0
        for (i, node) in NODES.iter().enumerate() {
            let values = Hex8::values(*node);
            for (j, &value) in values[..8].iter().enumerate() {
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
        testing::assert_partition_of_unity::<Hex8>(&brick(2.0, 3.0, 0.5));
    }

    #[test]
    fn reproduces_a_linear_field_exactly() {
        // 1 次要素のパッチテスト
        testing::assert_reproduces_linear_field::<Hex8>(
            &brick(2.0, 3.0, 0.5),
            [1.5, -2.0, 0.25],
            7.0,
        );
    }

    #[test]
    fn volume_of_a_brick() {
        assert!((volume::<Hex8>(&cube(2.0)).unwrap() - 8.0).abs() < 1e-14);
        assert!((volume::<Hex8>(&brick(2.0, 3.0, 0.5)).unwrap() - 3.0).abs() < 1e-14);
    }

    #[test]
    fn volume_of_a_sheared_brick() {
        // せん断しても体積は変わらない（平行六面体）
        let mut points = brick(2.0, 3.0, 0.5);
        for p in &mut points {
            p[0] += 0.7 * p[2];
        }
        assert!((volume::<Hex8>(&points).unwrap() - 3.0).abs() < 1e-14);
    }

    #[test]
    fn stiffness_is_symmetric_with_a_constant_nullspace() {
        testing::assert_stiffness_properties::<Hex8>(&brick(2.0, 3.0, 0.5), 0.35);
    }

    #[test]
    fn inverted_node_order_is_rejected() {
        // 上面と底面を入れ替えると det J < 0 になる
        let points = cube(1.0);
        let flipped: Vec<[f64; 3]> = points[4..]
            .iter()
            .chain(points[..4].iter())
            .copied()
            .collect();
        let e = volume::<Hex8>(&flipped).unwrap_err();
        assert!(e.to_string().contains("ヤコビアン"), "{e}");
        assert!(e.to_string().contains("節点の並び順"), "{e}");
    }

    #[test]
    fn field_at_the_centre_of_a_linear_solution() {
        let points = brick(2.0, 3.0, 0.5);
        let a = [1.0, -0.5, 2.0];
        let phi: Vec<f64> = points
            .iter()
            .map(|p| a[0] * p[0] + a[1] * p[1] + a[2] * p[2])
            .collect();
        let field = electric_field::<Hex8>(&points, &phi, Hex8::CENTRE);
        for d in 0..3 {
            assert!((field[d] + a[d]).abs() < 1e-13);
        }
    }
}
