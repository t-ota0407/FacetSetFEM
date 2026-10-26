//! 面共役な四面体分割（SPEC §5.1 / §8 M7）。
//!
//! 等値面・流線・面隣接・L2 誤差評価はいずれも「四面体を線形補間する」前提の
//! アルゴリズムで、M6 までは Tet4 のメッシュにしか使えなかった。ここでは
//! Hex8 / Penta6 を含むメッシュを**四面体の集まりとして見る**ための派生表現を作る。
//!
//! # 面共役でなければならない理由
//!
//! 六面体や五面体を四面体に切ると、四角形の面には対角線を 1 本引くことになる。
//! 隣り合う 2 つの要素が**同じ四角形面に別々の対角線**を引くと、
//!
//! - 等値面: 面の両側で三角形の切り方が違うため、等値面に隙間や重なりができる
//! - 流線: 面を挟んで四面体どうしが対応せず、隣を辿れなくなる
//!
//! という壊れ方をする。だから対角線の選び方を**両側から見て必ず一致する規則**にする。
//!
//! # 規則
//!
//! 1. 四角形面の対角線は、**その面の 4 節点のうち節点番号が最小のもの**を通す
//! 2. 要素は、**節点番号が最小の頂点 v₀** から「v₀ を含まない面」へ張った錐に分ける
//!
//! 規則 1 は面の 4 節点だけで決まるので、隣り合う要素が同じ答えを出す。
//! 規則 2 は v₀ を含む面（近い面）にも自動的に「v₀ を通る対角線」を誘導するが、
//! v₀ は要素全体の最小なのでその面の最小でもあり、規則 1 と一致する。
//! 結果として**すべての四角形面が両側から同じ対角線で割られる**。
//!
//! 分割数は Tet4 が 1、Penta6 が 3、Hex8 が 6、2 次の Tet10 が 8（辺の中点で割る。
//! [`split_tet10`]）。新しい節点は作らないので、
//! 節点値はそのまま使える。
//!
//! # Tet4 だけのメッシュは素通しする
//!
//! 四面体には四角形の面が無いので分割は不要。そのままの接続と順序で通す
//! （[`is_identity`](TetView::is_identity)）。M6 までの結果がビット単位で変わらないための配慮でもある。
//!
//! # 要素の代表値との関係
//!
//! E・J は**元の要素**ごとの値で（Hex8 / Penta6 では要素中心の値。SPEC §5.1）、
//! 分割してもそこは変わらない。同じ元要素から出た四面体は同じ値を持つので、
//! 流線は元要素の中では直線のまま進む。

use crate::element::{ElementKind, tet4_signed_volume};
use crate::error::{FacetSetFemError, Result};
use crate::mesh::{Mesh, NodeIndex};

/// 面共役な四面体分割。
///
/// 節点は元のメッシュのものをそのまま使う（新しい節点は作らない）。
#[derive(Debug, Clone)]
pub struct TetView {
    /// 4 節点ずつ並べた接続
    conn: Vec<NodeIndex>,
    /// 四面体 → 元のソリッド要素の通し番号（[`Mesh::iter_solids`] の順）
    parent: Vec<u32>,
    /// 元のソリッド要素 → 四面体の範囲。長さは要素数 + 1
    offsets: Vec<u32>,
    /// 元のメッシュが Tet4 だけで、分割が恒等だったか
    identity: bool,
    /// 元のメッシュの節点数（接続の節点索引はこれ未満）
    n_nodes: usize,
}

impl TetView {
    /// メッシュを四面体に切り分ける。
    pub fn build(mesh: &Mesh) -> Result<Self> {
        let n_elements = mesh.n_solid_elements();
        let identity = mesh.solids.iter().all(|b| b.kind() == ElementKind::Tet4);

        let mut conn: Vec<NodeIndex> = Vec::with_capacity(n_elements * 4);
        let mut parent: Vec<u32> = Vec::with_capacity(n_elements);
        let mut offsets: Vec<u32> = Vec::with_capacity(n_elements + 1);
        offsets.push(0);

        for e in mesh.iter_solids() {
            match e.kind {
                // 四面体はそのまま。節点順序も変えない
                ElementKind::Tet4 => {
                    conn.extend_from_slice(e.conn);
                    parent.push(e.index as u32);
                }
                ElementKind::Hex8 | ElementKind::Penta6 => {
                    split_solid(e.kind, e.conn, e.index as u32, &mut conn, &mut parent);
                }
                ElementKind::Tet10 => {
                    split_tet10(mesh, e.conn, e.index as u32, &mut conn, &mut parent);
                }
                other => {
                    return Err(FacetSetFemError::unsupported(crate::tr!(
                        "{} cannot be treated as a solid element, so it cannot be split into tetrahedra",
                        "{} はソリッド要素として扱えないため四面体に分割できません",
                        other.card_name()
                    ))
                    .with_hint(crate::tr!(
                        "Use CTETRA, CHEXA or CPENTA",
                        "CTETRA / CHEXA / CPENTA のいずれかを使ってください"
                    )));
                }
            }
            offsets.push(parent.len() as u32);
        }

        Ok(Self {
            conn,
            parent,
            offsets,
            identity,
            n_nodes: mesh.nodes.len(),
        })
    }

    /// 四面体の個数。
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// 四面体が 1 つも無いか。
    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    /// 元のメッシュの節点数（接続の節点索引はこれ未満）。
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// 元のメッシュが Tet4 だけで、分割が恒等だったか。
    ///
    /// `true` なら四面体の通し番号は元の要素の通し番号と一致する。
    pub fn is_identity(&self) -> bool {
        self.identity
    }

    /// 四面体 `t` の接続。
    #[inline]
    pub fn conn_of(&self, t: usize) -> &[NodeIndex] {
        &self.conn[t * 4..t * 4 + 4]
    }

    /// 四面体 `t` を生んだ元のソリッド要素の通し番号。
    #[inline]
    pub fn parent(&self, t: usize) -> u32 {
        self.parent[t]
    }

    /// 四面体 → 元要素の対応表。
    pub fn parents(&self) -> &[u32] {
        &self.parent
    }

    /// 元のソリッド要素 `e` が生んだ四面体の通し番号の範囲。
    pub fn children(&self, e: usize) -> std::ops::Range<usize> {
        self.offsets[e] as usize..self.offsets[e + 1] as usize
    }

    /// 四面体 `t` の節点座標。
    #[inline]
    pub fn coords(&self, mesh: &Mesh, t: usize) -> [[f64; 3]; 4] {
        let c = self.conn_of(t);
        [
            mesh.nodes.coord(c[0]),
            mesh.nodes.coord(c[1]),
            mesh.nodes.coord(c[2]),
            mesh.nodes.coord(c[3]),
        ]
    }

    /// 要素ごとの値（3 成分）を四面体ごとに配り直す。
    ///
    /// E や J をそのまま流線に渡すために使う。恒等分割ならコピーせず借りる。
    pub fn spread_vectors<'a>(&self, values: &'a [f64]) -> std::borrow::Cow<'a, [f64]> {
        if self.identity {
            return std::borrow::Cow::Borrowed(values);
        }
        let mut out = Vec::with_capacity(self.len() * 3);
        for &p in &self.parent {
            let k = p as usize * 3;
            out.extend_from_slice(&values[k..k + 3]);
        }
        std::borrow::Cow::Owned(out)
    }

    /// 分割が幾何的に妥当か（すべての四面体の体積が正か）を調べる。
    ///
    /// 反転した四面体があれば、その元要素の通し番号を返す。
    /// 極端に歪んだ／非凸な六面体では起こりうる。
    pub fn find_inverted(&self, mesh: &Mesh) -> Option<u32> {
        (0..self.len())
            .find(|&t| tet4_signed_volume(&self.coords(mesh, t)) <= 0.0)
            .map(|t| self.parent[t])
    }

    /// 分割後の総体積 [m³]。
    pub fn total_volume(&self, mesh: &Mesh) -> f64 {
        crate::solver::reduce::sum(
            &(0..self.len())
                .map(|t| tet4_signed_volume(&self.coords(mesh, t)))
                .collect::<Vec<f64>>(),
        )
    }
}

/// 2 次四面体の頂点に接する 4 つの小さな四面体（局所番号。向きは元の要素と同じ）。
///
/// 中点の局所番号は 4 = 辺 0-1、5 = 1-2、6 = 2-0、7 = 0-3、8 = 1-3、9 = 2-3。
const TET10_CORNER_CHILDREN: [[usize; 4]; 4] =
    [[0, 4, 6, 7], [4, 1, 5, 8], [6, 5, 2, 9], [7, 8, 9, 3]];

/// 中の八面体の 3 本の対角線と、それぞれを囲む 4 頂点（巡回の順）。
///
/// 対角線 `[a, b]` と巡回の隣どうし `c_k, c_{k+1}` で四面体 `[a, b, c_k, c_{k+1}]` を作ると、
/// 元の要素と同じ向きになるように並べてある（テストで参照要素の座標から確かめている）。
const TET10_DIAGONALS: [([usize; 2], [usize; 4]); 3] = [
    ([4, 9], [5, 6, 7, 8]),
    ([5, 7], [4, 8, 9, 6]),
    ([6, 8], [4, 5, 9, 7]),
];

/// 2 次四面体を、辺の中点で 8 個の四面体に分ける（新しい節点は作らない）。
///
/// 頂点に接する 4 個と、中に残る八面体を対角線 1 本で割った 4 個。どの面も中点で 4 枚の
/// 三角形に割られ、その割り方は面の 6 節点だけで決まるので、隣の要素とも必ず一致する
/// （面共役。外表面の [`Surface::triangulate`](crate::mesh::Surface::triangulate) とも同じ割り方）。
/// 八面体の対角線は面に出ないので、どれを選んでも適合性には関わらない。いちばん短いものを
/// 選ぶ（小さな四面体が潰れにくい。同じ長さなら表の先のもの）。
///
/// 節点値を 8 個の四面体で 1 次補間したものは、2 次の電位を区分 1 次で近似したものになる
/// （等値面・断面の描画用。VTK が 2 次のセルを描くのと同じ考え方）。
fn split_tet10(
    mesh: &Mesh,
    conn: &[NodeIndex],
    element: u32,
    out_conn: &mut Vec<NodeIndex>,
    out_parent: &mut Vec<u32>,
) {
    for child in TET10_CORNER_CHILDREN {
        out_conn.extend(child.iter().map(|&l| conn[l]));
        out_parent.push(element);
    }
    let length2 = |[a, b]: [usize; 2]| {
        let (p, q) = (mesh.nodes.coord(conn[a]), mesh.nodes.coord(conn[b]));
        (0..3).map(|d| (p[d] - q[d]) * (p[d] - q[d])).sum::<f64>()
    };
    let (diagonal, ring) = TET10_DIAGONALS
        .iter()
        .copied()
        .reduce(|best, next| {
            if length2(next.0) < length2(best.0) {
                next
            } else {
                best
            }
        })
        .expect("対角線は 3 本ある");
    for k in 0..4 {
        out_conn.extend([diagonal[0], diagonal[1], ring[k], ring[(k + 1) % 4]].map(|l| conn[l]));
        out_parent.push(element);
    }
}

/// v₀ から「v₀ を含まない面」へ錐を張って四面体に分ける。
fn split_solid(
    kind: ElementKind,
    conn: &[NodeIndex],
    element: u32,
    out_conn: &mut Vec<NodeIndex>,
    out_parent: &mut Vec<u32>,
) {
    // 節点番号が最小の頂点を錐の頂点にする
    let apex = (0..conn.len())
        .min_by_key(|&k| conn[k])
        .expect("ソリッド要素に節点がある");

    for face in kind.faces() {
        if face.contains(&apex) {
            continue; // v₀ を含む面は錐の側面になる
        }
        // 面の節点番号（外向き法線の順）
        let nodes: Vec<NodeIndex> = face.iter().map(|&l| conn[l]).collect();
        for tri in triangulate_face(&nodes) {
            out_conn.push(conn[apex]);
            out_conn.extend_from_slice(&tri);
            out_parent.push(element);
        }
    }
}

/// 面を三角形に割る。四角形は**節点番号が最小の頂点を通る対角線**で 2 枚に。
///
/// 外向き法線の向きは保たれる。四角形の 4 節点だけで決まる規則なので、
/// 隣り合う要素から見ても同じ対角線になる（モジュール冒頭の規則 1）。
fn triangulate_face(nodes: &[NodeIndex]) -> Vec<[NodeIndex; 3]> {
    match nodes.len() {
        3 => vec![[nodes[0], nodes[1], nodes[2]]],
        4 => {
            let k = (0..4).min_by_key(|&i| nodes[i]).expect("4 節点ある");
            if k % 2 == 0 {
                // 対角線 0–2
                vec![
                    [nodes[0], nodes[1], nodes[2]],
                    [nodes[0], nodes[2], nodes[3]],
                ]
            } else {
                // 対角線 1–3
                vec![
                    [nodes[1], nodes[2], nodes[3]],
                    [nodes[1], nodes[3], nodes[0]],
                ]
            }
        }
        n => unreachable!("ソリッド要素の面が {n} 節点になることはない"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::ElementKind;
    use crate::mesh::{ElementBlock, Nodes, Property};
    use crate::units::LengthUnit;
    use std::collections::HashMap;

    /// 単位立方体の 8 頂点を、指定した節点 ID の並びで作る。
    fn cube_mesh(kind: ElementKind, conn: Vec<NodeIndex>, n_nodes: usize) -> Mesh {
        let corners = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let coords: Vec<f64> = corners[..n_nodes].iter().flatten().copied().collect();
        Mesh {
            nodes: Nodes::new((1..=n_nodes as u32).collect(), coords),
            solids: vec![ElementBlock::new(kind, vec![10], vec![1], conn)],
            shells: Vec::new(),
            solid_properties: vec![Property {
                pid: 1,
                mid: None,
                card: "PSOLID",
            }],
            shell_properties: Vec::new(),
            length_unit: LengthUnit::Meter,
            source: None,
        }
    }

    #[test]
    fn a_hexahedron_becomes_six_tetrahedra_that_fill_it() {
        let mesh = cube_mesh(ElementKind::Hex8, (0..8).collect(), 8);
        let view = TetView::build(&mesh).unwrap();
        assert_eq!(view.len(), 6);
        assert!(!view.is_identity());
        assert_eq!(view.children(0), 0..6);
        assert!(view.find_inverted(&mesh).is_none(), "反転した四面体がある");
        // 体積が立方体と一致すること（分割が隙間も重なりも作っていない証拠）
        assert!(
            (view.total_volume(&mesh) - 1.0).abs() < 1e-14,
            "体積 {}",
            view.total_volume(&mesh)
        );
    }

    #[test]
    fn a_wedge_becomes_three_tetrahedra_that_fill_it() {
        // 底面 (0,0,0),(1,0,0),(0,1,0) を z=1 まで押し出した三角柱
        let coords = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0,
        ];
        let mesh = Mesh {
            nodes: Nodes::new((1..=6).collect(), coords),
            solids: vec![ElementBlock::new(
                ElementKind::Penta6,
                vec![10],
                vec![1],
                (0..6).collect(),
            )],
            shells: Vec::new(),
            solid_properties: vec![Property {
                pid: 1,
                mid: None,
                card: "PSOLID",
            }],
            shell_properties: Vec::new(),
            length_unit: LengthUnit::Meter,
            source: None,
        };
        let view = TetView::build(&mesh).unwrap();
        assert_eq!(view.len(), 3);
        assert!(view.find_inverted(&mesh).is_none());
        assert!((view.total_volume(&mesh) - 0.5).abs() < 1e-14);
    }

    #[test]
    fn a_tetrahedral_mesh_passes_through_unchanged() {
        // M6 までの結果を 1 ビットも変えないための素通し
        let coords = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mesh = Mesh {
            nodes: Nodes::new(vec![1, 2, 3, 4], coords),
            solids: vec![ElementBlock::new(
                ElementKind::Tet4,
                vec![10],
                vec![1],
                vec![0, 1, 2, 3],
            )],
            shells: Vec::new(),
            solid_properties: vec![Property {
                pid: 1,
                mid: None,
                card: "PSOLID",
            }],
            shell_properties: Vec::new(),
            length_unit: LengthUnit::Meter,
            source: None,
        };
        let view = TetView::build(&mesh).unwrap();
        assert!(view.is_identity());
        assert_eq!(view.len(), 1);
        assert_eq!(view.conn_of(0), &[0, 1, 2, 3]);
        assert_eq!(view.parent(0), 0);
    }

    /// 四角形面の対角線を「面の 4 節点の集合 → 対角線の節点対」で集める。
    fn quad_diagonals(mesh: &Mesh) -> HashMap<[NodeIndex; 4], [NodeIndex; 2]> {
        let mut out = HashMap::new();
        for e in mesh.iter_solids() {
            for face in e.kind.faces() {
                if face.len() != 4 {
                    continue;
                }
                let nodes: Vec<NodeIndex> = face.iter().map(|&l| e.conn[l]).collect();
                let mut key: [NodeIndex; 4] = [nodes[0], nodes[1], nodes[2], nodes[3]];
                key.sort_unstable();
                let tris = triangulate_face(&nodes);
                // 2 枚の三角形が共有する辺が対角線
                let mut shared: Vec<NodeIndex> = tris[0]
                    .iter()
                    .filter(|n| tris[1].contains(n))
                    .copied()
                    .collect();
                shared.sort_unstable();
                let diagonal = [shared[0], shared[1]];
                if let Some(previous) = out.insert(key, diagonal) {
                    assert_eq!(previous, diagonal, "面 {key:?} の対角線が両側で食い違う");
                }
            }
        }
        out
    }

    #[test]
    fn neighbouring_elements_choose_the_same_diagonal() {
        // 節点番号の付き方が違う 2 つの六面体が 1 枚の四角形面を共有する状況。
        // 対角線の規則が面の節点だけで決まっていれば、両側から見て一致する
        //
        //   下の立方体: 節点 0..7、上の立方体: 節点 4..7 を共有して 8..11 へ
        let coords: Vec<f64> = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, // 0..3
            0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, // 4..7
            0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 1.0, 2.0, 0.0, 1.0, 2.0, // 8..11
        ];
        // 上の立方体は共有面の節点を巡回させて渡す（向きも順序も違う書き方）
        let mesh = Mesh {
            nodes: Nodes::new((1..=12).collect(), coords),
            solids: vec![ElementBlock::new(
                ElementKind::Hex8,
                vec![10, 11],
                vec![1, 1],
                vec![
                    0, 1, 2, 3, 4, 5, 6, 7, //
                    5, 6, 7, 4, 9, 10, 11, 8,
                ],
            )],
            shells: Vec::new(),
            solid_properties: vec![Property {
                pid: 1,
                mid: None,
                card: "PSOLID",
            }],
            shell_properties: Vec::new(),
            length_unit: LengthUnit::Meter,
            source: None,
        };
        // 食い違えば quad_diagonals の中で落ちる
        let diagonals = quad_diagonals(&mesh);
        assert!(!diagonals.is_empty());

        let view = TetView::build(&mesh).unwrap();
        assert_eq!(view.len(), 12);
        assert!(view.find_inverted(&mesh).is_none());
        assert!((view.total_volume(&mesh) - 2.0).abs() < 1e-14);
    }

    #[test]
    fn the_shared_face_is_triangulated_the_same_way_from_both_sides() {
        // 共有面に現れる三角形を数える。両側が同じ切り方なら、
        // 面上の三角形は「表と裏」でちょうど 2 回ずつ現れる
        let coords: Vec<f64> = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, //
            0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 1.0, 2.0, 0.0, 1.0, 2.0,
        ];
        let mesh = Mesh {
            nodes: Nodes::new((1..=12).collect(), coords),
            solids: vec![ElementBlock::new(
                ElementKind::Hex8,
                vec![10, 11],
                vec![1, 1],
                vec![
                    0, 1, 2, 3, 4, 5, 6, 7, //
                    5, 6, 7, 4, 9, 10, 11, 8,
                ],
            )],
            shells: Vec::new(),
            solid_properties: vec![Property {
                pid: 1,
                mid: None,
                card: "PSOLID",
            }],
            shell_properties: Vec::new(),
            length_unit: LengthUnit::Meter,
            source: None,
        };
        let view = TetView::build(&mesh).unwrap();

        // 全四面体の面を集め、内部の面がちょうど 2 回現れることを確かめる。
        // 対角線が食い違うと、共有面の上に 1 回しか現れない三角形ができる
        let mut counts: HashMap<[NodeIndex; 3], usize> = HashMap::new();
        for t in 0..view.len() {
            let c = view.conn_of(t);
            for f in ElementKind::Tet4.faces() {
                let mut key = [c[f[0]], c[f[1]], c[f[2]]];
                key.sort_unstable();
                *counts.entry(key).or_default() += 1;
            }
        }
        // 共有面 z=1 の上にある三角形（4 節点すべてが z=1）を数える
        let on_interface = |k: &[NodeIndex; 3]| k.iter().all(|&n| (4..8).contains(&n));
        let interface: Vec<usize> = counts
            .iter()
            .filter(|(k, _)| on_interface(k))
            .map(|(_, &v)| v)
            .collect();
        assert_eq!(interface.len(), 2, "共有面は三角形 2 枚に割れるはず");
        assert!(
            interface.iter().all(|&c| c == 2),
            "共有面の三角形が両側から現れていない: {interface:?}"
        );
    }

    #[test]
    fn the_split_is_reproducible() {
        let mesh = cube_mesh(ElementKind::Hex8, (0..8).collect(), 8);
        let first = TetView::build(&mesh).unwrap();
        for _ in 0..3 {
            let again = TetView::build(&mesh).unwrap();
            assert_eq!(first.conn, again.conn);
            assert_eq!(first.parent, again.parent);
        }
    }

    #[test]
    fn element_values_are_spread_to_the_children() {
        let mesh = cube_mesh(ElementKind::Hex8, (0..8).collect(), 8);
        let view = TetView::build(&mesh).unwrap();
        let field = vec![1.0, 2.0, 3.0];
        let spread = view.spread_vectors(&field);
        assert_eq!(spread.len(), 6 * 3);
        assert!(spread.chunks(3).all(|c| c == [1.0, 2.0, 3.0]));
    }

    #[test]
    fn the_tet10_children_keep_the_orientation_of_the_element() {
        // 参照要素の座標で、8 個とも正の体積になり、合わせて参照四面体（1/6）を埋める
        use crate::element::tet10::LOCAL_NODES;
        let volume = |c: [usize; 4]| tet4_signed_volume(&c.map(|l| LOCAL_NODES[l]));
        let corners: f64 = TET10_CORNER_CHILDREN.iter().map(|&c| volume(c)).sum();
        for c in TET10_CORNER_CHILDREN {
            assert!(volume(c) > 0.0, "{c:?}");
        }
        for (diagonal, ring) in TET10_DIAGONALS {
            let mut inner = 0.0;
            for k in 0..4 {
                let c = [diagonal[0], diagonal[1], ring[k], ring[(k + 1) % 4]];
                assert!(volume(c) > 0.0, "対角線 {diagonal:?}: {c:?}");
                inner += volume(c);
            }
            assert!(
                (corners + inner - 1.0 / 6.0).abs() < 1e-15,
                "対角線 {diagonal:?}"
            );
        }
    }

    #[test]
    fn a_raised_mesh_is_split_into_eight_tetrahedra_per_element() {
        let linear = crate::testing::kuhn_box([3, 2, 2], [1.5, 1.0, 0.8]);
        let mesh = crate::mesh::elevate(&linear).unwrap();
        let view = TetView::build(&mesh).unwrap();
        assert!(!view.is_identity());
        assert_eq!(view.len(), 8 * mesh.n_solid_elements());
        assert!(view.find_inverted(&mesh).is_none());
        assert!((view.total_volume(&mesh) - 1.5 * 1.0 * 0.8).abs() < 1e-12);
        for e in 0..mesh.n_solid_elements() {
            assert_eq!(view.children(e).len(), 8);
        }
        // 小さな四面体の面は、隣どうしでぴったり重なる（面の出現が 1 回か 2 回）
        let mut count: HashMap<[u32; 3], usize> = HashMap::new();
        for t in 0..view.len() {
            let c = view.conn_of(t);
            for skip in 0..4 {
                let mut f: Vec<u32> = (0..4).filter(|&k| k != skip).map(|k| c[k]).collect();
                f.sort_unstable();
                *count.entry([f[0], f[1], f[2]]).or_insert(0) += 1;
            }
        }
        assert!(
            count.values().all(|&n| n == 1 || n == 2),
            "非多様体の面がある"
        );
        // 外に出る面（1 回だけの面）は、外表面を 4 分割した三角形と一致する
        let surface = crate::mesh::topology::extract_surface(&mesh).unwrap();
        let (tris, _) = surface.triangulate(&mesh);
        let mut outer: Vec<[u32; 3]> = count
            .iter()
            .filter(|&(_, &n)| n == 1)
            .map(|(&f, _)| f)
            .collect();
        let mut from_surface: Vec<[u32; 3]> = tris
            .iter()
            .map(|t| {
                let mut f = *t;
                f.sort_unstable();
                f
            })
            .collect();
        outer.sort_unstable();
        from_surface.sort_unstable();
        assert_eq!(outer, from_surface);
    }
}
