//! アイソパラメトリック要素の共通処理（SPEC §2.3 / §7）。
//!
//! 要素種ごとに違うのは「節点数」「形状関数」「積分則」の 3 つだけなので、
//! それを [`ShapeFunctions`] に切り出し、ヤコビ行列・要素剛性・電界の計算は
//! ここに 1 度だけ書く。新しい要素種を足すときは trait を実装したファイルを
//! 1 つ増やすだけで済む（SPEC §7 の設計原則）。
//!
//! ```text
//! K^e_ij = Σ_q w_q · |det J(ξ_q)| · (∇N_i(ξ_q) · σ ∇N_j(ξ_q))      (SPEC §2.3)
//! ```
//!
//! σ はスカラー倍ではなく [`Conductivity::apply`] 経由で作用させるので、
//! 異方性導電率（SPEC §9）を入れてもここは変更不要。
//!
//! # ヤコビアンの符号
//!
//! `det J ≤ 0` は節点順序の反転か要素の潰れを意味する。**要素 ID を挙げてエラー**
//! にする（SPEC §2.3）。Tet4 だけは |V| を使えば計算できるので読み込み時は警告に
//! 留めているが（SPEC §3.1）、Hex8 / Penta6 は積分点ごとに符号が変わりうるため
//! 救済せず止める。

use crate::error::{GfemError, Result};
use crate::material::Conductivity;

/// 扱える要素の最大節点数。Tet4 = 4、Penta6 = 6、Hex8 = 8。
///
/// 2 次要素（Tet10 など、SPEC §9）を入れるときはここを増やす。
pub const MAX_ELEMENT_NODES: usize = 8;

/// 積分点。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadraturePoint {
    /// 局所座標
    pub local: [f64; 3],
    /// 重み（参照要素の体積要素を含む）
    pub weight: f64,
}

/// 形状関数と積分則。
///
/// 節点の並びは Nastran のカードと同じ（CTETRA G1..G4、CPENTA G1..G6、CHEXA G1..G8）。
pub trait ShapeFunctions {
    /// 節点数
    const NODES: usize;
    /// 要素の中心を表す局所座標。要素代表値（E, J）の評価点に使う
    const CENTRE: [f64; 3];

    /// 積分則。重みの総和は参照要素の体積になる
    fn quadrature() -> &'static [QuadraturePoint];

    /// 局所座標での形状関数の値。先頭 `NODES` 個が有効
    fn values(local: [f64; 3]) -> [f64; MAX_ELEMENT_NODES];

    /// 局所座標での `∂N_i/∂ξ_j`。先頭 `NODES` 個が有効
    fn local_gradients(local: [f64; 3]) -> [[f64; 3]; MAX_ELEMENT_NODES];
}

/// 積分点における大域座標での形状関数勾配とヤコビアン。
#[derive(Debug, Clone, Copy)]
pub struct PointGeometry {
    /// `∂N_i/∂x_j`。先頭 `NODES` 個が有効
    pub grad: [[f64; 3]; MAX_ELEMENT_NODES],
    /// `det J`（符号つき）
    pub det_jacobian: f64,
}

/// 積分点での幾何量を求める。
///
/// `det J` の符号は判定しない（呼び出し側で扱いを決める）。
pub fn point_geometry<E: ShapeFunctions>(points: &[[f64; 3]], local: [f64; 3]) -> PointGeometry {
    debug_assert_eq!(points.len(), E::NODES);
    let local_grad = E::local_gradients(local);

    // J[j][k] = ∂x_k/∂ξ_j
    let mut jacobian = [[0.0f64; 3]; 3];
    for (i, p) in points.iter().enumerate() {
        for j in 0..3 {
            for k in 0..3 {
                jacobian[j][k] += local_grad[i][j] * p[k];
            }
        }
    }

    let det = jacobian[0][0] * (jacobian[1][1] * jacobian[2][2] - jacobian[1][2] * jacobian[2][1])
        - jacobian[0][1] * (jacobian[1][0] * jacobian[2][2] - jacobian[1][2] * jacobian[2][0])
        + jacobian[0][2] * (jacobian[1][0] * jacobian[2][1] - jacobian[1][1] * jacobian[2][0]);

    // ∇N_i(大域) = J⁻¹ ∇N_i(局所)
    let inv_det = if det != 0.0 { 1.0 / det } else { 0.0 };
    let inverse = [
        [
            (jacobian[1][1] * jacobian[2][2] - jacobian[1][2] * jacobian[2][1]) * inv_det,
            (jacobian[0][2] * jacobian[2][1] - jacobian[0][1] * jacobian[2][2]) * inv_det,
            (jacobian[0][1] * jacobian[1][2] - jacobian[0][2] * jacobian[1][1]) * inv_det,
        ],
        [
            (jacobian[1][2] * jacobian[2][0] - jacobian[1][0] * jacobian[2][2]) * inv_det,
            (jacobian[0][0] * jacobian[2][2] - jacobian[0][2] * jacobian[2][0]) * inv_det,
            (jacobian[0][2] * jacobian[1][0] - jacobian[0][0] * jacobian[1][2]) * inv_det,
        ],
        [
            (jacobian[1][0] * jacobian[2][1] - jacobian[1][1] * jacobian[2][0]) * inv_det,
            (jacobian[0][1] * jacobian[2][0] - jacobian[0][0] * jacobian[2][1]) * inv_det,
            (jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]) * inv_det,
        ],
    ];

    let mut grad = [[0.0f64; 3]; MAX_ELEMENT_NODES];
    for i in 0..E::NODES {
        for k in 0..3 {
            grad[i][k] = inverse[k][0] * local_grad[i][0]
                + inverse[k][1] * local_grad[i][1]
                + inverse[k][2] * local_grad[i][2];
        }
    }
    PointGeometry {
        grad,
        det_jacobian: det,
    }
}

/// 要素剛性行列と体積（SPEC §2.3）。
///
/// `k` は `MAX_ELEMENT_NODES × MAX_ELEMENT_NODES` の作業領域。
/// 先頭 `NODES × NODES` に結果が入る。戻り値は要素体積 [m³]。
pub fn stiffness<E: ShapeFunctions>(
    points: &[[f64; 3]],
    sigma: &Conductivity,
    k: &mut [[f64; MAX_ELEMENT_NODES]; MAX_ELEMENT_NODES],
) -> Result<f64> {
    for row in k.iter_mut() {
        row.fill(0.0);
    }
    let mut volume = 0.0;
    for point in E::quadrature() {
        let geometry = point_geometry::<E>(points, point.local);
        if geometry.det_jacobian <= 0.0 || geometry.det_jacobian.is_nan() {
            return Err(negative_jacobian(geometry.det_jacobian));
        }
        let scale = point.weight * geometry.det_jacobian;
        volume += scale;
        for j in 0..E::NODES {
            let sg = sigma.apply(geometry.grad[j]);
            for (i, row) in k.iter_mut().enumerate().take(E::NODES) {
                let g = geometry.grad[i];
                row[j] += scale * (g[0] * sg[0] + g[1] * sg[1] + g[2] * sg[2]);
            }
        }
    }
    Ok(volume)
}

/// 要素の体積 [m³]。
pub fn volume<E: ShapeFunctions>(points: &[[f64; 3]]) -> Result<f64> {
    let mut total = 0.0;
    for point in E::quadrature() {
        let det = point_geometry::<E>(points, point.local).det_jacobian;
        if det <= 0.0 || det.is_nan() {
            return Err(negative_jacobian(det));
        }
        total += point.weight * det;
    }
    Ok(total)
}

/// 指定した局所座標での電界 `E = −∇φ`（SPEC §5.1）。
pub fn electric_field<E: ShapeFunctions>(
    points: &[[f64; 3]],
    phi: &[f64],
    local: [f64; 3],
) -> [f64; 3] {
    debug_assert_eq!(phi.len(), E::NODES);
    let geometry = point_geometry::<E>(points, local);
    let mut field = [0.0f64; 3];
    for (i, &value) in phi.iter().enumerate() {
        for (d, f) in field.iter_mut().enumerate() {
            *f -= value * geometry.grad[i][d];
        }
    }
    field
}

fn negative_jacobian(det: f64) -> GfemError {
    GfemError::model(crate::tr!(
        "The Jacobian is not positive: det J = {det:.6e}",
        "ヤコビアンが正ではありません: det J = {det:.6e}"
    ))
    .with_hint(crate::tr!(
        "The node order is reversed or the element is collapsed. Check the Nastran node order \
         (for CHEXA, G1..G4 are the bottom face and G5..G8 the top face)",
        "節点の並び順が反転しているか、要素が潰れています。\
         Nastran の節点順序（CHEXA は G1..G4 が底面、G5..G8 が上面）を確認してください"
    ))
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    /// 線形場 `φ(x) = a·x + b` を節点に与えたとき、`E = −a` が厳密に出ることを確かめる。
    ///
    /// 1 次要素が満たすべき最低条件（パッチテスト）。これが通らない形状関数は間違っている。
    pub fn assert_reproduces_linear_field<E: ShapeFunctions>(
        points: &[[f64; 3]],
        a: [f64; 3],
        b: f64,
    ) {
        let phi: Vec<f64> = points
            .iter()
            .map(|p| a[0] * p[0] + a[1] * p[1] + a[2] * p[2] + b)
            .collect();
        for point in E::quadrature() {
            let field = electric_field::<E>(points, &phi, point.local);
            for d in 0..3 {
                assert!(
                    (field[d] + a[d]).abs() < 1e-12,
                    "積分点 {:?}: E[{d}] = {} （期待 {}）",
                    point.local,
                    field[d],
                    -a[d]
                );
            }
        }
    }

    /// 形状関数の総和が 1、勾配の総和が 0 であることを確かめる（分割の一致）。
    pub fn assert_partition_of_unity<E: ShapeFunctions>(points: &[[f64; 3]]) {
        for point in E::quadrature() {
            let values = E::values(point.local);
            let sum: f64 = values[..E::NODES].iter().sum();
            assert!((sum - 1.0).abs() < 1e-14, "ΣN = {sum}");

            let geometry = point_geometry::<E>(points, point.local);
            for d in 0..3 {
                let sum: f64 = (0..E::NODES).map(|i| geometry.grad[i][d]).sum();
                assert!(sum.abs() < 1e-12, "Σ∂N/∂x[{d}] = {sum}");
            }
        }
    }

    /// 積分則の重みの総和が参照要素の体積に一致することを確かめる。
    pub fn assert_quadrature_weight(expected: f64, quadrature: &[QuadraturePoint]) {
        let sum: f64 = quadrature.iter().map(|q| q.weight).sum();
        assert!(
            (sum - expected).abs() < 1e-14,
            "重みの総和 {sum}（期待 {expected}）"
        );
    }

    /// 要素剛性が対称で、定数ベクトルを零空間に持つことを確かめる（SPEC §2.2）。
    pub fn assert_stiffness_properties<E: ShapeFunctions>(points: &[[f64; 3]], sigma: f64) {
        let mut k = [[0.0f64; MAX_ELEMENT_NODES]; MAX_ELEMENT_NODES];
        let volume =
            stiffness::<E>(points, &Conductivity::isotropic(sigma).unwrap(), &mut k).unwrap();
        assert!(volume > 0.0);
        for (i, row) in k.iter().enumerate().take(E::NODES) {
            for (j, &value) in row[..E::NODES].iter().enumerate() {
                assert!(
                    (value - k[j][i]).abs() < 1e-16 * row[i].abs().max(1.0),
                    "非対称: ({i},{j})"
                );
            }
            assert!(row[i] > 0.0, "対角 {i} が正でない");
            let row_sum: f64 = row[..E::NODES].iter().sum();
            assert!(
                row_sum.abs() < 1e-14 * row[i].abs().max(1.0),
                "行 {i} の和が 0 でない: {row_sum}"
            );
        }
    }
}
