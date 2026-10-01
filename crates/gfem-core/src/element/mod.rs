//! 要素の種類と、種類ごとの幾何量。
//!
//! ここに置くのは「要素の識別」（節点数・面・VTK のセル型）と、幾何チェックに
//! 使う最小限の量（Tet4 の符号付き体積・最大辺長）。
//! 要素剛性（SPEC §2.3）は要素種ごとのサブモジュールに置く:
//! [`tet4`]・[`hex8`]・[`penta6`]・[`tet10`]。共通のアイソパラメトリック処理は [`shape`] にあり、
//! 新しい要素種は [`ShapeFunctions`] を実装したファイルを 1 つ足すだけで入る（SPEC §7）。
//! 面要素（電極荷重に使う）は [`shell`]。

pub mod hex8;
pub mod penta6;
pub mod shape;
pub mod shell;
pub mod tet10;
pub mod tet4;

pub use hex8::Hex8;
pub use penta6::Penta6;
pub use shape::{MAX_ELEMENT_NODES, QuadraturePoint, ShapeFunctions};
pub use tet10::Tet10;

/// 対応する要素の種類（SPEC §1.3 / §3.1）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ElementKind {
    /// 1 次四面体（Nastran: CTETRA / 4 節点）
    Tet4,
    /// 1 次五面体（Nastran: CPENTA / 6 節点）
    Penta6,
    /// 1 次六面体（Nastran: CHEXA / 8 節点）
    Hex8,
    /// 2 次四面体（Nastran: CTETRA / 10 節点。SPEC §2.3）
    Tet10,
    /// 1 次三角形シェル（Nastran: CTRIA3）。電極面の指定に使う
    Tria3,
    /// 1 次四辺形シェル（Nastran: CQUAD4）。電極面の指定に使う
    Quad4,
    /// 2 次三角形シェル（Nastran: CTRIA6）。2 次四面体のメッシュで電極面の指定に使う。
    /// 2 次四面体の面（体表面の面を積分するとき）もこの種類として扱う
    Tria6,
}

impl ElementKind {
    /// 要素を構成する節点数。
    pub const fn node_count(self) -> usize {
        match self {
            ElementKind::Tet4 => 4,
            ElementKind::Penta6 => 6,
            ElementKind::Hex8 => 8,
            ElementKind::Tet10 => 10,
            ElementKind::Tria3 => 3,
            ElementKind::Quad4 => 4,
            ElementKind::Tria6 => 6,
        }
    }

    /// 3 次元のソリッド要素か（false ならシェル要素）。
    pub const fn is_solid(self) -> bool {
        matches!(
            self,
            ElementKind::Tet4 | ElementKind::Penta6 | ElementKind::Hex8 | ElementKind::Tet10
        )
    }

    /// 2 次要素か（辺の中点にも節点を持つ）。
    pub const fn is_quadratic(self) -> bool {
        matches!(self, ElementKind::Tet10 | ElementKind::Tria6)
    }

    /// 頂点の節点数。2 次要素では先頭のこの数だけが頂点で、残りは辺の中点。
    pub const fn corner_count(self) -> usize {
        match self {
            ElementKind::Tet10 => 4,
            ElementKind::Tria6 => 3,
            other => other.node_count(),
        }
    }

    /// 対応する Nastran のカード名。
    pub const fn card_name(self) -> &'static str {
        match self {
            ElementKind::Tet4 | ElementKind::Tet10 => "CTETRA",
            ElementKind::Penta6 => "CPENTA",
            ElementKind::Hex8 => "CHEXA",
            ElementKind::Tria3 => "CTRIA3",
            ElementKind::Quad4 => "CQUAD4",
            ElementKind::Tria6 => "CTRIA6",
        }
    }

    /// 要素数の一覧などに出す名前。カード名が同じ CTETRA の 4 節点と 10 節点を区別する。
    pub const fn label(self) -> &'static str {
        match self {
            ElementKind::Tet10 => "CTETRA(10)",
            other => other.card_name(),
        }
    }

    /// VTK のセルタイプ番号（VTU 出力に使う）。
    ///
    /// CTETRA → VTK_TETRA(10), CPENTA → VTK_WEDGE(13), CHEXA → VTK_HEXAHEDRON(12),
    /// 10 節点の CTETRA → VTK_QUADRATIC_TETRA(24),
    /// CTRIA3 → VTK_TRIANGLE(5), CQUAD4 → VTK_QUAD(9), CTRIA6 → VTK_QUADRATIC_TRIANGLE(22)。
    /// 節点の並びは CPENTA だけ VTK と逆向きなので、書き出すときは
    /// [`vtk_node_order`](Self::vtk_node_order) で並べ替えること。
    pub const fn vtk_cell_type(self) -> u8 {
        match self {
            ElementKind::Tet4 => 10,
            ElementKind::Penta6 => 13,
            ElementKind::Hex8 => 12,
            ElementKind::Tet10 => 24,
            ElementKind::Tria3 => 5,
            ElementKind::Quad4 => 9,
            ElementKind::Tria6 => 22,
        }
    }

    /// VTK のセルに書くときの節点の並び替え（局所節点番号）。
    ///
    /// CTETRA（4 節点・10 節点とも）/ CHEXA / CTRIA3 / CQUAD4 は VTK と同じ並びだが、
    /// **CPENTA だけは向きが逆**。VTK_WEDGE は 1 枚目の三角形の法線が
    /// 2 枚目の三角形から**遠ざかる**向きを要求するのに対し、Nastran の CPENTA は
    /// G1〜G3 の法線が G4〜G6 の側を向く。そのまま書くと ParaView 側で
    /// 体積が負の反転セルになるため、三角形の 2 番目と 3 番目を入れ替える。
    pub const fn vtk_node_order(self) -> &'static [usize] {
        match self {
            ElementKind::Tet4 => &[0, 1, 2, 3],
            ElementKind::Penta6 => &[0, 2, 1, 3, 5, 4],
            ElementKind::Hex8 => &[0, 1, 2, 3, 4, 5, 6, 7],
            ElementKind::Tet10 => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ElementKind::Tria3 => &[0, 1, 2],
            ElementKind::Quad4 => &[0, 1, 2, 3],
            ElementKind::Tria6 => &[0, 1, 2, 3, 4, 5],
        }
    }

    /// ソリッド要素の面（局所節点番号）。並びは**外向き法線**になる向き。
    ///
    /// 外表面抽出（SPEC §4.1）と、電極面が体表面に載っているかの照合（SPEC §11.1）に使う。
    /// シェル要素はそれ自体が面なので空を返す。
    ///
    /// **2 次要素の面は頂点だけで表す**（Tet10 は Tet4 と同じ表）。面の照合は頂点で
    /// 決まり、面番号も Tet4 と揃うので、ケースファイルの `[EID, 面番号]` がそのまま使える。
    pub const fn faces(self) -> &'static [&'static [usize]] {
        match self {
            ElementKind::Tet4 | ElementKind::Tet10 => &TET4_FACES,
            ElementKind::Penta6 => &PENTA6_FACES,
            ElementKind::Hex8 => &HEX8_FACES,
            ElementKind::Tria3 | ElementKind::Quad4 | ElementKind::Tria6 => &[],
        }
    }

    /// 2 次要素の面の、辺の中点の節点（局所番号）。[`faces`](Self::faces) と同じ並びで、
    /// 面の頂点 `[c0, c1, c2]` に対して辺 c0-c1, c1-c2, c2-c0 の中点を並べる
    /// （CTRIA6 の G4..G6 と同じ順）。1 次要素では空。
    ///
    /// 頂点のあとにこれを続けたものが、面積分に使う 6 節点の三角形（Tria6）になる。
    pub const fn face_mid_nodes(self) -> &'static [&'static [usize]] {
        match self {
            ElementKind::Tet10 => &TET10_FACE_MIDS,
            _ => &[],
        }
    }

    /// この面（節点数）に対応するシェル要素の種類。外表面を三角形／四辺形として扱うときに使う。
    pub const fn face_kind(n_nodes: usize) -> Option<ElementKind> {
        match n_nodes {
            3 => Some(ElementKind::Tria3),
            4 => Some(ElementKind::Quad4),
            _ => None,
        }
    }
}

/// Tet4 の 4 面。節点順序は外向き法線になるように選んである。
const TET4_FACES: [&[usize]; 4] = [&[1, 2, 3], &[0, 3, 2], &[0, 1, 3], &[0, 2, 1]];

/// Tet10 の 4 面の辺の中点（[`TET4_FACES`] の各面の辺 c0-c1, c1-c2, c2-c0 の順）。
///
/// 中点の局所番号は 4 = 辺 0-1、5 = 1-2、6 = 2-0、7 = 0-3、8 = 1-3、9 = 2-3（[`tet10::EDGES`]）。
const TET10_FACE_MIDS: [&[usize]; 4] = [&[5, 9, 8], &[7, 9, 6], &[4, 8, 7], &[6, 5, 4]];

/// Penta6 の 5 面（三角 2 + 四辺形 3）。
const PENTA6_FACES: [&[usize]; 5] = [
    &[0, 2, 1],
    &[3, 4, 5],
    &[0, 1, 4, 3],
    &[1, 2, 5, 4],
    &[2, 0, 3, 5],
];

/// Hex8 の 6 面。
const HEX8_FACES: [&[usize]; 6] = [
    &[0, 3, 2, 1],
    &[4, 5, 6, 7],
    &[0, 1, 5, 4],
    &[1, 2, 6, 5],
    &[2, 3, 7, 6],
    &[3, 0, 4, 7],
];

/// Tet4 の符号付き体積（SPEC §2.3）。
///
/// `V = ((p1-p0) × (p2-p0)) · (p3-p0) / 6`。
/// Nastran の CTETRA は正の体積になる節点順序を要求するため、負の値は
/// 節点順序の反転を意味する（読み込み時に検出して警告する）。
pub fn tet4_signed_volume(p: &[[f64; 3]; 4]) -> f64 {
    use crate::geom::{cross, dot, sub};
    let a = sub(p[1], p[0]);
    let b = sub(p[2], p[0]);
    let c = sub(p[3], p[0]);
    dot(cross(a, b), c) / 6.0
}

/// 節点集合の最大辺長。退化判定を無次元化するために使う。
pub fn max_edge_length(p: &[[f64; 3]]) -> f64 {
    let mut h: f64 = 0.0;
    for i in 0..p.len() {
        for j in (i + 1)..p.len() {
            let d = [p[j][0] - p[i][0], p[j][1] - p[i][1], p[j][2] - p[i][2]];
            let len2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if len2 > h {
                h = len2;
            }
        }
    }
    h.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_tet_volume() {
        let p = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        assert!((tet4_signed_volume(&p) - 1.0 / 6.0).abs() < 1e-15);
    }

    #[test]
    fn swapping_two_nodes_flips_sign() {
        let p = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let q = [p[0], p[1], p[3], p[2]];
        assert!((tet4_signed_volume(&p) + tet4_signed_volume(&q)).abs() < 1e-15);
    }

    #[test]
    fn node_counts_match_cards() {
        assert_eq!(ElementKind::Tet4.node_count(), 4);
        assert_eq!(ElementKind::Tet10.node_count(), 10);
        assert_eq!(ElementKind::Tet10.corner_count(), 4);
        assert!(ElementKind::Tet10.is_solid() && ElementKind::Tet10.is_quadratic());
        assert!(!ElementKind::Tet4.is_quadratic());
        assert_eq!(ElementKind::Tet10.card_name(), "CTETRA");
        assert_ne!(ElementKind::Tet10.label(), ElementKind::Tet4.label());
        assert_eq!(ElementKind::Hex8.node_count(), 8);
        assert_eq!(ElementKind::Penta6.node_count(), 6);
        assert!(ElementKind::Tet4.is_solid());
        assert!(!ElementKind::Tria3.is_solid());
    }

    #[test]
    fn tet4_faces_point_outward() {
        // 正の体積を持つ基準四面体で、全ての面の法線が重心から外を向くことを確認する
        let p = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        assert!(tet4_signed_volume(&p) > 0.0);
        let centre = [0.25, 0.25, 0.25];
        for face in ElementKind::Tet4.faces() {
            let (a, b, c) = (p[face[0]], p[face[1]], p[face[2]]);
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let n = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let d = [a[0] - centre[0], a[1] - centre[1], a[2] - centre[2]];
            assert!(
                n[0] * d[0] + n[1] * d[1] + n[2] * d[2] > 0.0,
                "面 {face:?} の法線が内側を向いています"
            );
        }
    }

    #[test]
    fn every_solid_face_uses_valid_local_nodes() {
        for kind in [
            ElementKind::Tet4,
            ElementKind::Penta6,
            ElementKind::Hex8,
            ElementKind::Tet10,
        ] {
            let mut seen = vec![0usize; kind.corner_count()];
            for face in kind.faces() {
                assert!(face.len() == 3 || face.len() == 4);
                for &n in *face {
                    assert!(n < kind.corner_count());
                    seen[n] += 1;
                }
            }
            // どの節点も 3 面以上に現れる（＝面の定義に抜けが無い）
            assert!(seen.iter().all(|&c| c >= 3), "{kind:?}: {seen:?}");
        }
    }

    #[test]
    fn tet10_face_mid_nodes_lie_on_the_face_edges() {
        // 面の頂点 [c0, c1, c2] の辺 c0-c1, c1-c2, c2-c0 の中点が、その順に並んでいること
        let corners = ElementKind::Tet10.faces();
        let mids = ElementKind::Tet10.face_mid_nodes();
        assert_eq!(corners.len(), mids.len());
        for (c, m) in corners.iter().zip(mids) {
            for k in 0..3 {
                let (a, b) = (c[k], c[(k + 1) % 3]);
                let [p, q] = tet10::EDGES[m[k] - 4];
                assert!(
                    (p, q) == (a, b) || (p, q) == (b, a),
                    "面 {c:?} の辺 {a}-{b} に中点 {} が乗っていない",
                    m[k]
                );
            }
        }
        assert!(ElementKind::Tet4.face_mid_nodes().is_empty());
    }

    #[test]
    fn max_edge_of_unit_cube_diagonal() {
        let p = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        assert!((max_edge_length(&p) - 3.0f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn vtk_node_order_is_a_permutation() {
        for kind in [
            ElementKind::Tet4,
            ElementKind::Penta6,
            ElementKind::Hex8,
            ElementKind::Tet10,
            ElementKind::Tria3,
            ElementKind::Quad4,
            ElementKind::Tria6,
        ] {
            let order = kind.vtk_node_order();
            assert_eq!(order.len(), kind.node_count(), "{kind:?}");
            let mut sorted = order.to_vec();
            sorted.sort_unstable();
            assert!(
                sorted.iter().copied().eq(0..kind.node_count()),
                "{kind:?}: {order:?}"
            );
        }
    }

    #[test]
    fn wedge_is_flipped_for_vtk() {
        // VTK_WEDGE は 1 枚目の三角形の法線が 2 枚目の三角形から**遠ざかる**向きを
        // 要求する。Nastran の CPENTA は逆（G1〜G3 の法線が G4〜G6 の側を向く）なので、
        // そのまま書くと ParaView で体積が負の反転セルになる
        let cpenta = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        // 1 枚目の三角形の法線と、2 枚目へ向かうベクトルの内積
        let facing = |p: &[[f64; 3]; 6]| {
            let u = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
            let v = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
            let n = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let w = [p[3][0] - p[0][0], p[3][1] - p[0][1], p[3][2] - p[0][2]];
            n[0] * w[0] + n[1] * w[1] + n[2] * w[2]
        };
        assert!(facing(&cpenta) > 0.0, "CPENTA の前提が変わっています");

        let order = ElementKind::Penta6.vtk_node_order();
        let mut vtk = [[0.0f64; 3]; 6];
        for (i, &local) in order.iter().enumerate() {
            vtk[i] = cpenta[local];
        }
        assert!(facing(&vtk) < 0.0, "VTK_WEDGE の向きになっていません");

        // 他の要素は並び替え不要
        assert_eq!(ElementKind::Tet4.vtk_node_order(), &[0, 1, 2, 3]);
        assert_eq!(
            ElementKind::Hex8.vtk_node_order(),
            &[0, 1, 2, 3, 4, 5, 6, 7]
        );
    }
}
