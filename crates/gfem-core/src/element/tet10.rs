//! 2 次四面体（Tet10）の形状関数（SPEC §2.3）。
//!
//! 頂点 4 つに加えて、6 本の辺の中点に節点を置く。電位を要素の中で 2 次式で表すので、
//! **E は要素内で 1 次式で変化する**（Tet4 のように要素内一定ではない）。
//!
//! 節点の並びは Nastran の CTETRA G1..G10 と同じ（VTK_QUADRATIC_TETRA とも同じ）:
//! G1..G4 が頂点、G5..G10 が辺 1-2, 2-3, 3-1, 1-4, 2-4, 3-4 の中点。
//! 0 始まりの局所番号では [`EDGES`] の順になる。
//!
//! 体積座標 `L0 = 1 − ξ − η − ζ, L1 = ξ, L2 = η, L3 = ζ` を使って
//!
//! ```text
//! 頂点 i:            N_i = L_i (2 L_i − 1)
//! 辺 (a, b) の中点:  N   = 4 L_a L_b
//! ```
//!
//! 辺が真っ直ぐな要素（1 次のメッシュから上げたもの）では ∇N が 1 次式なので、
//! 要素剛性の被積分関数は 2 次式になり、4 点則（代数的精度 2 次）で厳密に積分できる。
//! 辺が曲がった要素ではヤコビ行列が要素内で変わるため厳密ではないが、
//! 2 次四面体の標準的な積分則として同じ 4 点を使う。

use crate::element::shape::{self, MAX_ELEMENT_NODES, QuadraturePoint, ShapeFunctions};
use crate::error::{GfemError, Result};
use crate::material::Conductivity;

/// Tet10 の形状関数。
#[derive(Debug, Clone, Copy)]
pub struct Tet10;

/// 中点の節点（局所番号 4..9）が乗る辺の、両端の頂点（局所番号）。
///
/// Nastran の CTETRA G5..G10（辺 1-2, 2-3, 3-1, 1-4, 2-4, 3-4）と同じ並び。
pub const EDGES: [[usize; 2]; 6] = [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]];

/// 10 節点の局所座標 `(ξ, η, ζ)`。頂点 4 つと、辺の中点 6 つ。
pub const LOCAL_NODES: [[f64; 3]; 10] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.5, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 0.5],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
];

/// 4 点則の、1 つの頂点に寄った体積座標 `(5 + 3√5) / 20`。
const A: f64 = 0.585_410_196_624_968_5;
/// 4 点則の、残りの 3 頂点の体積座標 `(5 − √5) / 20`。
const B: f64 = 0.138_196_601_125_010_5;

/// 4 点の積分点（代数的精度 2 次）。重みの総和は参照四面体の体積 1/6。
static QUADRATURE: [QuadraturePoint; 4] = [
    QuadraturePoint {
        local: [B, B, B],
        weight: 1.0 / 24.0,
    },
    QuadraturePoint {
        local: [A, B, B],
        weight: 1.0 / 24.0,
    },
    QuadraturePoint {
        local: [B, A, B],
        weight: 1.0 / 24.0,
    },
    QuadraturePoint {
        local: [B, B, A],
        weight: 1.0 / 24.0,
    },
];

/// 局所座標から体積座標 `(L0, L1, L2, L3)` を求める。
fn barycentric(local: [f64; 3]) -> [f64; 4] {
    [
        1.0 - local[0] - local[1] - local[2],
        local[0],
        local[1],
        local[2],
    ]
}

/// `∂L_i/∂(ξ, η, ζ)`。
const DL: [[f64; 3]; 4] = [
    [-1.0, -1.0, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

impl ShapeFunctions for Tet10 {
    const NODES: usize = 10;
    const CENTRE: [f64; 3] = [0.25, 0.25, 0.25];

    fn quadrature() -> &'static [QuadraturePoint] {
        &QUADRATURE
    }

    fn values(local: [f64; 3]) -> [f64; MAX_ELEMENT_NODES] {
        let l = barycentric(local);
        let mut out = [0.0f64; MAX_ELEMENT_NODES];
        for i in 0..4 {
            out[i] = l[i] * (2.0 * l[i] - 1.0);
        }
        for (k, [a, b]) in EDGES.into_iter().enumerate() {
            out[4 + k] = 4.0 * l[a] * l[b];
        }
        out
    }

    fn local_gradients(local: [f64; 3]) -> [[f64; 3]; MAX_ELEMENT_NODES] {
        let l = barycentric(local);
        let mut out = [[0.0f64; 3]; MAX_ELEMENT_NODES];
        for i in 0..4 {
            let c = 4.0 * l[i] - 1.0;
            out[i] = [c * DL[i][0], c * DL[i][1], c * DL[i][2]];
        }
        for (k, [a, b]) in EDGES.into_iter().enumerate() {
            for d in 0..3 {
                out[4 + k][d] = 4.0 * (l[a] * DL[b][d] + l[b] * DL[a][d]);
            }
        }
        out
    }
}

/// 要素の向き（ヤコビアンの符号）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    /// 調べた全ての点で `det J > 0`
    Positive,
    /// 全ての点で `det J < 0`。節点順序が反転しているだけで、計算はできる
    Inverted,
    /// 符号が混じるか、ゼロ・NaN の点がある。要素が潰れているかねじれている
    Invalid,
}

impl Orientation {
    /// 要素剛性・体積に掛ける符号（`Invalid` は `None`）。
    pub fn sign(self) -> Option<f64> {
        match self {
            Orientation::Positive => Some(1.0),
            Orientation::Inverted => Some(-1.0),
            Orientation::Invalid => None,
        }
    }
}

/// ヤコビアンの符号を、10 節点と積分点の全てで調べる（SPEC §3.1）。
///
/// 辺が曲がった要素では `det J` が要素内で変わり、積分点では正でも頂点の近くで負になる
/// ことがある（要素が裏返っている）。積分点だけでなく節点でも見るのはそのため。
/// 辺が真っ直ぐなら `det J` は要素内で一定（= 6V）なので、どの点でも同じ符号になる。
pub fn orientation(points: &[[f64; 3]]) -> Orientation {
    debug_assert_eq!(points.len(), Tet10::NODES);
    let (mut positive, mut negative) = (false, false);
    let checks = LOCAL_NODES
        .iter()
        .copied()
        .chain(QUADRATURE.iter().map(|q| q.local));
    for local in checks {
        let det = shape::point_geometry::<Tet10>(points, local).det_jacobian;
        if det > 0.0 {
            positive = true;
        } else if det < 0.0 {
            negative = true;
        } else {
            // ゼロと NaN
            return Orientation::Invalid;
        }
    }
    match (positive, negative) {
        (true, false) => Orientation::Positive,
        (false, true) => Orientation::Inverted,
        _ => Orientation::Invalid,
    }
}

/// 要素剛性行列と体積（SPEC §2.3）。
///
/// 節点順序が反転しているだけの要素（全ての点で `det J < 0`）は `|det J|` で計算する。
/// Tet4 が |V| を使うのと同じ扱い（SPEC §3.1）で、1 次の反転した四面体を
/// 2 次に上げたものもそのまま解ける。符号が混じる要素はエラーにする。
pub fn stiffness(
    points: &[[f64; 3]],
    sigma: &Conductivity,
    k: &mut [[f64; MAX_ELEMENT_NODES]; MAX_ELEMENT_NODES],
) -> Result<f64> {
    let sign = orientation(points).sign().ok_or_else(twisted)?;
    shape::stiffness_with_sign::<Tet10>(points, sigma, k, sign)
}

/// [`stiffness`] の `row` 行目だけ（並列アセンブリ用）。全体の同じ行とビット単位で同じ。
pub fn stiffness_row(
    points: &[[f64; 3]],
    sigma: &Conductivity,
    row: usize,
) -> Result<[f64; MAX_ELEMENT_NODES]> {
    let sign = orientation(points).sign().ok_or_else(twisted)?;
    shape::stiffness_row_with_sign::<Tet10>(points, sigma, row, sign)
}

/// 要素の体積 [m³]（常に正）。
pub fn volume(points: &[[f64; 3]]) -> Result<f64> {
    let sign = orientation(points).sign().ok_or_else(twisted)?;
    shape::volume_with_sign::<Tet10>(points, sign)
}

fn twisted() -> GfemError {
    GfemError::model(crate::tr!(
        "The Jacobian of a 10-node CTETRA changes sign or vanishes inside the element",
        "10 節点の CTETRA のヤコビアンが、要素の中で符号を変えるかゼロになっています"
    ))
    .with_hint(crate::tr!(
        "A mid-side node is too far from the middle of its edge, or the element is collapsed. \
         Fix the element in the mesh generator",
        "中間節点が辺の中央から離れすぎているか、要素が潰れています。\
         メッシュ生成側で該当要素を修正してください"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::shape::{electric_field, point_geometry, testing};

    /// 頂点 4 つから、辺が真っ直ぐな 10 節点を作る。
    fn straight(corners: [[f64; 3]; 4]) -> [[f64; 3]; 10] {
        let mut p = [[0.0; 3]; 10];
        p[..4].copy_from_slice(&corners);
        for (k, [a, b]) in EDGES.into_iter().enumerate() {
            for d in 0..3 {
                p[4 + k][d] = 0.5 * (corners[a][d] + corners[b][d]);
            }
        }
        p
    }

    const CORNERS: [[f64; 3]; 4] = [
        [0.13, -0.4, 0.7],
        [1.9, 0.2, -0.3],
        [-0.5, 2.1, 0.4],
        [0.3, 0.1, 3.3],
    ];

    #[test]
    fn shape_functions_are_nodal() {
        // N_i(節点 j) = δ_ij
        for (j, &local) in LOCAL_NODES.iter().enumerate() {
            let n = Tet10::values(local);
            for (i, &v) in n[..10].iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((v - expected).abs() < 1e-15, "N_{i}(節点 {j}) = {v}");
            }
        }
    }

    #[test]
    fn local_nodes_are_the_edge_midpoints() {
        for (k, [a, b]) in EDGES.into_iter().enumerate() {
            let mid = [0, 1, 2].map(|d| 0.5 * (LOCAL_NODES[a][d] + LOCAL_NODES[b][d]));
            assert_eq!(LOCAL_NODES[4 + k], mid, "辺 {k}");
        }
    }

    #[test]
    fn gradients_match_finite_differences() {
        let h = 1e-6;
        for q in QUADRATURE.iter().map(|q| q.local).chain([[0.1, 0.2, 0.3]]) {
            let g = Tet10::local_gradients(q);
            for d in 0..3 {
                let (mut up, mut down) = (q, q);
                up[d] += h;
                down[d] -= h;
                let (nu, nd) = (Tet10::values(up), Tet10::values(down));
                for i in 0..10 {
                    let fd = (nu[i] - nd[i]) / (2.0 * h);
                    assert!((fd - g[i][d]).abs() < 1e-8, "N_{i} の ∂/∂ξ{d}");
                }
            }
        }
    }

    #[test]
    fn satisfies_the_common_element_checks() {
        let p = straight(CORNERS);
        testing::assert_quadrature_weight(1.0 / 6.0, Tet10::quadrature());
        testing::assert_partition_of_unity::<Tet10>(&p);
        testing::assert_reproduces_linear_field::<Tet10>(&p, [1.5, -2.0, 0.25], 7.0);
        testing::assert_stiffness_properties::<Tet10>(&p, 0.35);
    }

    #[test]
    fn reproduces_a_quadratic_field_exactly() {
        // 2 次要素の最低条件: φ が 2 次式なら E = −∇φ が要素内のどこでも厳密に出る
        let p = straight(CORNERS);
        let phi_of = |x: [f64; 3]| {
            1.0 + 2.0 * x[0] - x[1] + 0.5 * x[2] + 0.7 * x[0] * x[0] - 1.3 * x[1] * x[2]
                + 0.4 * x[0] * x[1]
                + 0.9 * x[2] * x[2]
        };
        let grad_of = |x: [f64; 3]| {
            [
                2.0 + 1.4 * x[0] + 0.4 * x[1],
                -1.0 - 1.3 * x[2] + 0.4 * x[0],
                0.5 - 1.3 * x[1] + 1.8 * x[2],
            ]
        };
        let phi: Vec<f64> = p.iter().map(|&x| phi_of(x)).collect();
        for local in [
            [0.25, 0.25, 0.25],
            [0.1, 0.6, 0.2],
            [0.0, 0.0, 0.0],
            [A, B, B],
        ] {
            let n = Tet10::values(local);
            let mut x = [0.0; 3];
            for (i, point) in p.iter().enumerate() {
                for d in 0..3 {
                    x[d] += n[i] * point[d];
                }
            }
            let e = electric_field::<Tet10>(&p, &phi, local);
            let g = grad_of(x);
            for d in 0..3 {
                assert!(
                    (e[d] + g[d]).abs() < 1e-12,
                    "{local:?}: E = {e:?}, ∇φ = {g:?}"
                );
            }
        }
    }

    #[test]
    fn a_straight_element_has_the_tet4_volume_and_a_constant_jacobian() {
        let p = straight(CORNERS);
        let v4 = crate::element::tet4_signed_volume(&CORNERS);
        assert!(v4 > 0.0);
        let v10 = volume(&p).unwrap();
        assert!((v10 - v4).abs() < 1e-14 * v4, "{v10} vs {v4}");
        for local in LOCAL_NODES {
            let det = point_geometry::<Tet10>(&p, local).det_jacobian;
            assert!((det - 6.0 * v4).abs() < 1e-12 * v4, "{local:?}: {det}");
        }
        assert_eq!(orientation(&p), Orientation::Positive);
    }

    #[test]
    fn stiffness_integrates_the_straight_element_exactly() {
        // 辺が真っ直ぐなら 4 点則で厳密。より細かい積分（参照四面体の 8 分割 × 4 点）と一致する
        let p = straight(CORNERS);
        let sigma = Conductivity::isotropic(0.35).unwrap();
        let mut k = [[0.0; MAX_ELEMENT_NODES]; MAX_ELEMENT_NODES];
        stiffness(&p, &sigma, &mut k).unwrap();

        let mut fine = [[0.0f64; 10]; 10];
        for q in refined_rule() {
            let g = point_geometry::<Tet10>(&p, q.local);
            let scale = q.weight * g.det_jacobian;
            for (row, a) in fine.iter_mut().zip(&g.grad) {
                for (value, b) in row.iter_mut().zip(&g.grad) {
                    *value += scale * 0.35 * (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
                }
            }
        }
        for i in 0..10 {
            for j in 0..10 {
                assert!(
                    (k[i][j] - fine[i][j]).abs() < 1e-12 * k[i][i].abs(),
                    "({i},{j}): {} vs {}",
                    k[i][j],
                    fine[i][j]
                );
            }
        }
    }

    /// 参照四面体を中点で 8 つに分け、それぞれに 4 点則を当てた 32 点則（検算用）。
    fn refined_rule() -> Vec<QuadraturePoint> {
        let v = LOCAL_NODES;
        // 頂点に接する 4 つと、中の八面体を対角 (4, 9) で割った 4 つ
        let children: [[usize; 4]; 8] = [
            [0, 4, 6, 7],
            [4, 1, 5, 8],
            [6, 5, 2, 9],
            [7, 8, 9, 3],
            [4, 9, 5, 6],
            [4, 9, 6, 7],
            [4, 9, 7, 8],
            [4, 9, 8, 5],
        ];
        let mut rule = Vec::new();
        for c in children {
            let (a, b, cc, d) = (v[c[0]], v[c[1]], v[c[2]], v[c[3]]);
            let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let e2 = [cc[0] - a[0], cc[1] - a[1], cc[2] - a[2]];
            let e3 = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
            let det = e1[0] * (e2[1] * e3[2] - e2[2] * e3[1])
                - e1[1] * (e2[0] * e3[2] - e2[2] * e3[0])
                + e1[2] * (e2[0] * e3[1] - e2[1] * e3[0]);
            for q in &QUADRATURE {
                let [s, t, u] = q.local;
                let local = [0, 1, 2].map(|i| a[i] + s * e1[i] + t * e2[i] + u * e3[i]);
                rule.push(QuadraturePoint {
                    local,
                    weight: q.weight * det.abs(),
                });
            }
        }
        let total: f64 = rule.iter().map(|q| q.weight).sum();
        assert!((total - 1.0 / 6.0).abs() < 1e-15);
        rule
    }

    #[test]
    fn a_single_row_is_bitwise_the_row_of_the_full_matrix() {
        // 並列アセンブリは 1 行ずつ作る。全体を作った行とビット単位で同じでなければならない
        let mut p = straight(CORNERS);
        p[7][1] += 0.03; // 辺が曲がった要素でも
        for sigma in [
            Conductivity::isotropic(0.35).unwrap(),
            Conductivity::isotropic(1.0 / 3.0).unwrap(),
        ] {
            let mut full = [[0.0; MAX_ELEMENT_NODES]; MAX_ELEMENT_NODES];
            stiffness(&p, &sigma, &mut full).unwrap();
            for (row, expected) in full.iter().enumerate().take(10) {
                let got = stiffness_row(&p, &sigma, row).unwrap();
                assert_eq!(
                    got.map(f64::to_bits),
                    expected.map(f64::to_bits),
                    "行 {row}"
                );
            }
        }
    }

    #[test]
    fn an_inverted_element_gives_the_same_stiffness() {
        // 頂点 1 と 2 を入れ替えると向きが反転する。中点も対応して入れ替えると
        // （辺 0-1 ↔ 0-2、辺 1-3 ↔ 2-3）、同じ要素を逆向きに並べたものになる
        let p = straight(CORNERS);
        let perm = [0, 2, 1, 3, 6, 5, 4, 7, 9, 8];
        let q: Vec<[f64; 3]> = perm.iter().map(|&i| p[i]).collect();
        assert_eq!(orientation(&q), Orientation::Inverted);

        let sigma = Conductivity::isotropic(0.35).unwrap();
        let mut kp = [[0.0; MAX_ELEMENT_NODES]; MAX_ELEMENT_NODES];
        let mut kq = [[0.0; MAX_ELEMENT_NODES]; MAX_ELEMENT_NODES];
        let vp = stiffness(&p, &sigma, &mut kp).unwrap();
        let vq = stiffness(&q, &sigma, &mut kq).unwrap();
        assert!((vp - vq).abs() < 1e-15 * vp);
        for i in 0..10 {
            for j in 0..10 {
                assert!(
                    (kp[perm[i]][perm[j]] - kq[i][j]).abs() < 1e-13 * kp[i][i].abs(),
                    "({i},{j})"
                );
            }
        }
    }

    #[test]
    fn a_curved_element_is_still_consistent() {
        // 中点を辺の中央から少しずらした（辺が曲がった）要素でも、1 次の場は厳密に再現し、
        // 剛性は対称・行和ゼロ
        let mut p = straight(CORNERS);
        p[5][0] += 0.08;
        p[9][2] -= 0.05;
        assert_eq!(orientation(&p), Orientation::Positive);
        testing::assert_partition_of_unity::<Tet10>(&p);
        testing::assert_reproduces_linear_field::<Tet10>(&p, [0.3, 1.1, -0.7], -2.0);
        testing::assert_stiffness_properties::<Tet10>(&p, 1.7);
    }

    #[test]
    fn a_folded_element_is_rejected() {
        // 中点を反対側の頂点を越えるところまで動かすと、要素の中でヤコビアンの符号が変わる
        let mut p = straight(CORNERS);
        p[4] = [
            CORNERS[2][0] * 1.5 - 0.25 * (CORNERS[0][0] + CORNERS[1][0]),
            CORNERS[2][1] * 1.5 - 0.25 * (CORNERS[0][1] + CORNERS[1][1]),
            CORNERS[2][2] * 1.5 - 0.25 * (CORNERS[0][2] + CORNERS[1][2]),
        ];
        assert_eq!(orientation(&p), Orientation::Invalid);
        let mut k = [[0.0; MAX_ELEMENT_NODES]; MAX_ELEMENT_NODES];
        let e = stiffness(&p, &Conductivity::isotropic(1.0).unwrap(), &mut k).unwrap_err();
        assert!(e.to_string().contains("ヤコビアン"), "{e}");
    }
}
