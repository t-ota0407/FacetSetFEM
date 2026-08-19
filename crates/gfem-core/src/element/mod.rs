//! 要素の種類と、種類ごとの幾何量。
//!
//! M0 の時点では「要素の識別」と「幾何チェックに必要な最小限の量」だけを持つ。
//! 要素剛性（SPEC §2.3）は M1（Tet4）・M4（Hex8 / Penta6）でこのモジュールに足す。

/// 対応する要素の種類（SPEC §1.3 / §3.1）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ElementKind {
    /// 1 次四面体（Nastran: CTETRA / 4 節点）
    Tet4,
    /// 1 次五面体（Nastran: CPENTA / 6 節点）
    Penta6,
    /// 1 次六面体（Nastran: CHEXA / 8 節点）
    Hex8,
    /// 1 次三角形シェル（Nastran: CTRIA3）。電極面の指定に使う
    Tria3,
    /// 1 次四辺形シェル（Nastran: CQUAD4）。電極面の指定に使う
    Quad4,
}

impl ElementKind {
    /// 要素を構成する節点数。
    pub const fn node_count(self) -> usize {
        match self {
            ElementKind::Tet4 => 4,
            ElementKind::Penta6 => 6,
            ElementKind::Hex8 => 8,
            ElementKind::Tria3 => 3,
            ElementKind::Quad4 => 4,
        }
    }

    /// 3 次元のソリッド要素か（false ならシェル要素）。
    pub const fn is_solid(self) -> bool {
        matches!(
            self,
            ElementKind::Tet4 | ElementKind::Penta6 | ElementKind::Hex8
        )
    }

    /// 対応する Nastran のカード名。
    pub const fn card_name(self) -> &'static str {
        match self {
            ElementKind::Tet4 => "CTETRA",
            ElementKind::Penta6 => "CPENTA",
            ElementKind::Hex8 => "CHEXA",
            ElementKind::Tria3 => "CTRIA3",
            ElementKind::Quad4 => "CQUAD4",
        }
    }

    /// VTK のセルタイプ番号（VTU 出力に使う）。
    ///
    /// Nastran の節点順序と VTK の節点順序は、ここで扱う 1 次要素については一致する:
    /// CTETRA(G1..G4) → VTK_TETRA(10), CPENTA(G1..G6) → VTK_WEDGE(13),
    /// CHEXA(G1..G8) → VTK_HEXAHEDRON(12), CTRIA3 → VTK_TRIANGLE(5), CQUAD4 → VTK_QUAD(9)。
    pub const fn vtk_cell_type(self) -> u8 {
        match self {
            ElementKind::Tet4 => 10,
            ElementKind::Penta6 => 13,
            ElementKind::Hex8 => 12,
            ElementKind::Tria3 => 5,
            ElementKind::Quad4 => 9,
        }
    }
}

/// Tet4 の符号付き体積（SPEC §2.3）。
///
/// `V = ((p1-p0) × (p2-p0)) · (p3-p0) / 6`。
/// Nastran の CTETRA は正の体積になる節点順序を要求するため、負の値は
/// 節点順序の反転を意味する（読み込み時に検出して警告する）。
pub fn tet4_signed_volume(p: &[[f64; 3]; 4]) -> f64 {
    let a = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
    let b = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
    let c = [p[3][0] - p[0][0], p[3][1] - p[0][1], p[3][2] - p[0][2]];
    let cross = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    (cross[0] * c[0] + cross[1] * c[1] + cross[2] * c[2]) / 6.0
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
        assert_eq!(ElementKind::Hex8.node_count(), 8);
        assert_eq!(ElementKind::Penta6.node_count(), 6);
        assert!(ElementKind::Tet4.is_solid());
        assert!(!ElementKind::Tria3.is_solid());
    }

    #[test]
    fn max_edge_of_unit_cube_diagonal() {
        let p = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        assert!((max_edge_length(&p) - 3.0f64.sqrt()).abs() < 1e-15);
    }
}
