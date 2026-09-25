//! トポロジ: 外表面の抽出（SPEC §4.1）。
//!
//! ソリッド要素の面を全て列挙し、**出現回数が 1 の面**を外表面とする。
//! 外表面は
//!
//! - 電極面が体表面に載っているかの検証（SPEC §11.1）
//! - GUI の表示用データ抽出（SPEC §6.3）
//!
//! の両方で使う。
//!
//! 結果は面キー（節点索引を昇順に並べたもの）の順に並べる。ハッシュ順や
//! 要素順ではなくキー順にすることで、入力が同じなら常に同じ並びになる
//! （CLAUDE.md 設計原則 1）。

use crate::element::ElementKind;
use crate::error::{GfemError, Result, format_sample_list};
use crate::mesh::tetview::TetView;
use crate::mesh::{Mesh, NodeIndex};

/// 面キーの空き枠を表す番兵。
pub const NO_NODE: NodeIndex = u32::MAX;

/// 面を向きに依らず識別する鍵。節点索引を昇順に詰め、余りは [`NO_NODE`]。
pub type FaceKey = [NodeIndex; 4];

/// 節点の並びから面キーを作る。
pub fn face_key(nodes: &[NodeIndex]) -> FaceKey {
    debug_assert!(nodes.len() <= 4);
    let mut key = [NO_NODE; 4];
    key[..nodes.len()].copy_from_slice(nodes);
    key[..nodes.len()].sort_unstable();
    key
}

/// 外表面の面 1 枚。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceFace {
    /// 面の節点索引。**外向き法線**になる順序。先頭 `n_nodes` 個が有効
    pub nodes: [NodeIndex; 4],
    /// 面の節点数（3 または 4）
    pub n_nodes: u8,
    /// 元のソリッド要素の通し番号（`Mesh::iter_solids` の `index`）
    pub element: u32,
    /// 元要素内での局所面番号
    pub local_face: u8,
}

impl SurfaceFace {
    /// 有効な節点索引。
    pub fn nodes(&self) -> &[NodeIndex] {
        &self.nodes[..self.n_nodes as usize]
    }

    /// 面要素としての種類。
    pub fn kind(&self) -> ElementKind {
        match self.n_nodes {
            3 => ElementKind::Tria3,
            _ => ElementKind::Quad4,
        }
    }
}

/// メッシュの外表面。
#[derive(Debug, Clone, Default)]
pub struct Surface {
    faces: Vec<SurfaceFace>,
    /// `faces` と同じ並びの面キー。昇順にソートされているので二分探索できる
    keys: Vec<FaceKey>,
}

impl Surface {
    /// 外表面の面。
    pub fn faces(&self) -> &[SurfaceFace] {
        &self.faces
    }

    /// 面の枚数。
    pub fn len(&self) -> usize {
        self.faces.len()
    }

    /// 面が 1 枚も無いか。
    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }

    /// 指定した節点の並びが外表面の面かどうか（向きは問わない）。
    pub fn contains(&self, nodes: &[NodeIndex]) -> bool {
        self.position(nodes).is_some()
    }

    /// 指定した節点の並びに対応する外表面の面。
    pub fn find(&self, nodes: &[NodeIndex]) -> Option<&SurfaceFace> {
        self.position(nodes).map(|i| &self.faces[i])
    }

    fn position(&self, nodes: &[NodeIndex]) -> Option<usize> {
        if nodes.len() > 4 {
            return None;
        }
        self.keys.binary_search(&face_key(nodes)).ok()
    }

    /// 外表面の総面積 [m²]。
    pub fn total_area(&self, mesh: &Mesh) -> f64 {
        let mut points = [[0.0f64; 3]; 4];
        self.faces
            .iter()
            .map(|f| {
                let n = f.n_nodes as usize;
                mesh.gather_coords(f.nodes(), &mut points[..n]);
                crate::element::shell::face_integral(f.kind(), &points[..n]).area
            })
            .sum()
    }

    /// 表示用に三角形へ分割した外表面。
    ///
    /// GUI には四面体メッシュそのものではなく、この軽い三角形群だけを渡す（SPEC §6.3）。
    /// 戻り値は `(三角形の節点索引, 各三角形の元ソリッド要素の通し番号)`。
    pub fn triangulate(&self) -> (Vec<[NodeIndex; 3]>, Vec<u32>) {
        let mut tris = Vec::with_capacity(self.faces.len());
        let mut owners = Vec::with_capacity(self.faces.len());
        for f in &self.faces {
            let n = f.nodes();
            tris.push([n[0], n[1], n[2]]);
            owners.push(f.element);
            if n.len() == 4 {
                tris.push([n[0], n[2], n[3]]);
                owners.push(f.element);
            }
        }
        (tris, owners)
    }

    /// 各節点が外表面に載っているかのフラグ。
    pub fn node_flags(&self, n_nodes: usize) -> Vec<bool> {
        let mut flags = vec![false; n_nodes];
        for f in &self.faces {
            for &n in f.nodes() {
                flags[n as usize] = true;
            }
        }
        flags
    }
}

/// 外表面を抽出する。
///
/// 3 個以上のソリッド要素に共有される面（非多様体）があればエラーにする。
/// メッシュ生成側の破綻を示しており、そのまま解いても意味のある結果にならないため。
pub fn extract_surface(mesh: &Mesh) -> Result<Surface> {
    // (面キー, 要素通し番号, 局所面番号)
    let n_faces: usize = mesh
        .solids
        .iter()
        .map(|b| b.len() * b.kind().faces().len())
        .sum();
    let mut entries: Vec<(FaceKey, u32, u8)> = Vec::with_capacity(n_faces);
    for e in mesh.iter_solids() {
        for (lf, face) in e.kind.faces().iter().enumerate() {
            let mut nodes = [0u32; 4];
            for (k, &local) in face.iter().enumerate() {
                nodes[k] = e.conn[local];
            }
            entries.push((face_key(&nodes[..face.len()]), e.index as u32, lf as u8));
        }
    }
    sort_face_entries(&mut entries);

    let mut faces: Vec<SurfaceFace> = Vec::new();
    let mut keys: Vec<FaceKey> = Vec::new();
    let mut non_manifold: Vec<String> = Vec::new();

    let mut i = 0;
    while i < entries.len() {
        let mut j = i + 1;
        while j < entries.len() && entries[j].0 == entries[i].0 {
            j += 1;
        }
        let count = j - i;
        if count == 1 {
            let (key, elem, local_face) = entries[i];
            let e = mesh.solid(elem as usize);
            let local = e.kind.faces()[local_face as usize];
            let mut nodes = [NO_NODE; 4];
            for (k, &l) in local.iter().enumerate() {
                nodes[k] = e.conn[l];
            }
            faces.push(SurfaceFace {
                nodes,
                n_nodes: local.len() as u8,
                element: elem,
                local_face,
            });
            keys.push(key);
        } else if count > 2 {
            let ids: Vec<String> = entries[i..j]
                .iter()
                .map(|&(_, elem, _)| mesh.solid(elem as usize).id.to_string())
                .collect();
            non_manifold.push(format!("要素 EID={}", ids.join("/")));
        }
        i = j;
    }

    if !non_manifold.is_empty() {
        return Err(GfemError::model(format!(
            "3 個以上の要素に共有される面があります（非多様体メッシュ、{} 箇所）: {}",
            non_manifold.len(),
            format_sample_list(&non_manifold, 20)
        ))
        .with_hint(
            "要素が重複して定義されているか、メッシュが自己交差しています。\
             メッシュ生成側で修正してください",
        ));
    }

    Ok(Surface { faces, keys })
}

/// `(面キー, 要素（四面体）の通し番号, 局所面番号)` の列を並べる。
///
/// 並列に並べても結果は逐次と同じ（CLAUDE.md 設計原則 1）。要素番号と局所面番号の組は
/// どの 2 つも異なるので、並べ終わった列は入力によって一意に決まり、不安定ソートでも
/// 並び方がぶれない。外表面の抽出は大規模モデルの前処理でいちばん重い段階
/// （1000 万節点で 2.4 億件）なので、ここを並列にする。
fn sort_face_entries(entries: &mut [(FaceKey, u32, u8)]) {
    use rayon::prelude::*;
    entries.par_sort_unstable();
}

/// 隣接要素が無いことを表す番兵。
pub const NO_NEIGHBOUR: u32 = u32::MAX;

/// 四面体の面ごとの隣接関係。
///
/// ストリームラインの追跡（SPEC §6.3）で「面を越えて隣の四面体へ移る」ために使う。
/// 番号は[面共役な四面体分割](crate::mesh::TetView)の通し番号。
#[derive(Debug, Clone, Default)]
pub struct FaceAdjacency {
    /// `tet * 4 + local_face` の位置に隣接四面体の通し番号
    neighbours: Vec<u32>,
    /// 同じ位置に、隣接要素側から見たときの局所面番号
    neighbour_faces: Vec<u8>,
}

impl FaceAdjacency {
    /// 面 `local_face` の向こう側の四面体と、そちらから見た局所面番号。
    ///
    /// 外表面なら `None`。
    pub fn neighbour(&self, tet: usize, local_face: usize) -> Option<(u32, u8)> {
        let k = tet * 4 + local_face;
        match self.neighbours.get(k) {
            Some(&n) if n != NO_NEIGHBOUR => Some((n, self.neighbour_faces[k])),
            _ => None,
        }
    }

    /// 四面体の個数。
    pub fn len(&self) -> usize {
        self.neighbours.len() / 4
    }

    /// 空か。
    pub fn is_empty(&self) -> bool {
        self.neighbours.is_empty()
    }
}

/// 四面体の面の隣接関係を作る（SPEC §5.1 / §8 M7）。
///
/// 入力は[面共役な四面体分割](crate::mesh::TetView)。Hex8 / Penta6 を含むメッシュでも、
/// 四角形面が両側から同じ対角線で割られているので、面はちょうど 2 つの四面体に
/// 共有されるか外表面かのどちらかになる。
pub fn build_face_adjacency(tets: &TetView) -> Result<FaceAdjacency> {
    let n = tets.len();
    let mut entries: Vec<(FaceKey, u32, u8)> = Vec::with_capacity(n * 4);
    for t in 0..n {
        let conn = tets.conn_of(t);
        for (lf, face) in ElementKind::Tet4.faces().iter().enumerate() {
            let nodes: [NodeIndex; 3] = [conn[face[0]], conn[face[1]], conn[face[2]]];
            entries.push((face_key(&nodes), t as u32, lf as u8));
        }
    }
    sort_face_entries(&mut entries);

    let mut adjacency = FaceAdjacency {
        neighbours: vec![NO_NEIGHBOUR; n * 4],
        neighbour_faces: vec![0u8; n * 4],
    };
    let mut i = 0;
    while i < entries.len() {
        let mut j = i + 1;
        while j < entries.len() && entries[j].0 == entries[i].0 {
            j += 1;
        }
        if j - i == 2 {
            let (_, ea, fa) = entries[i];
            let (_, eb, fb) = entries[i + 1];
            let ka = ea as usize * 4 + fa as usize;
            let kb = eb as usize * 4 + fb as usize;
            adjacency.neighbours[ka] = eb;
            adjacency.neighbour_faces[ka] = fb;
            adjacency.neighbours[kb] = ea;
            adjacency.neighbour_faces[kb] = fa;
        }
        i = j;
    }
    Ok(adjacency)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::nastran::read_nastran_from;
    use crate::mesh::tetview::TetView;
    use crate::units::LengthUnit;
    use std::path::Path;

    fn read(text: &str) -> Mesh {
        read_nastran_from(text.as_bytes(), Path::new("<test>"), LengthUnit::Meter)
            .unwrap()
            .0
    }

    #[test]
    fn single_tet_has_four_boundary_faces() {
        let mesh = read(
            "GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
GRID,4,,0.0,0.0,1.0
PSOLID,1,1
CTETRA,10,1,1,2,3,4
",
        );
        let s = extract_surface(&mesh).unwrap();
        assert_eq!(s.len(), 4);
        // 表面積 = 3 つの直角三角形 (1/2 ずつ) + 斜面 (√3/2)
        let expected = 1.5 + 3.0f64.sqrt() / 2.0;
        assert!((s.total_area(&mesh) - expected).abs() < 1e-14);
        assert!(s.node_flags(4).iter().all(|&f| f));
    }

    #[test]
    fn shared_face_is_interior() {
        // 面 (2,3,4) を共有する 2 個の四面体
        let mesh = read(
            "GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
GRID,4,,0.0,0.0,1.0
GRID,5,,1.0,1.0,1.0
PSOLID,1,1
CTETRA,10,1,1,2,3,4
CTETRA,11,1,5,4,3,2
",
        );
        let s = extract_surface(&mesh).unwrap();
        // 4 + 4 面のうち 1 面が共有 → 境界は 6 面
        assert_eq!(s.len(), 6);
        assert!(!s.contains(&[1, 2, 3]));
        assert!(s.contains(&[0, 1, 2]));
        // 向きに依らず引ける
        assert!(s.contains(&[2, 1, 0]));
    }

    #[test]
    fn non_manifold_mesh_is_rejected() {
        // 同じ面を 3 個の四面体が共有する
        let mesh = read(
            "GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
GRID,4,,0.0,0.0,1.0
GRID,5,,1.0,1.0,1.0
GRID,6,,-1.0,-1.0,-1.0
PSOLID,1,1
CTETRA,10,1,1,2,3,4
CTETRA,11,1,5,4,3,2
CTETRA,12,1,6,2,3,4
",
        );
        let e = extract_surface(&mesh).unwrap_err();
        assert!(e.to_string().contains("非多様体"), "{e}");
    }

    #[test]
    fn triangulation_covers_the_whole_surface() {
        let mesh = read(
            "GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
GRID,4,,0.0,0.0,1.0
PSOLID,1,1
CTETRA,10,1,1,2,3,4
",
        );
        let s = extract_surface(&mesh).unwrap();
        let (tris, owners) = s.triangulate();
        // 四面体 1 個の外表面は三角形 4 枚
        assert_eq!(tris.len(), 4);
        assert!(owners.iter().all(|&o| o == 0));
    }

    #[test]
    fn face_adjacency_pairs_shared_faces() {
        let mesh = read(
            "GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
GRID,4,,0.0,0.0,1.0
GRID,5,,1.0,1.0,1.0
PSOLID,1,1
CTETRA,10,1,1,2,3,4
CTETRA,11,1,5,4,3,2
",
        );
        let adjacency = build_face_adjacency(&TetView::build(&mesh).unwrap()).unwrap();
        assert_eq!(adjacency.len(), 2);

        // 共有面はちょうど 1 組
        let shared: Vec<(usize, usize)> = (0..2)
            .flat_map(|e| (0..4).map(move |f| (e, f)))
            .filter(|&(e, f)| adjacency.neighbour(e, f).is_some())
            .collect();
        assert_eq!(shared.len(), 2, "{shared:?}");

        // 相互に指し合っている
        let (e, f) = shared[0];
        let (other, other_face) = adjacency.neighbour(e, f).unwrap();
        assert_eq!(
            adjacency.neighbour(other as usize, other_face as usize),
            Some((e as u32, f as u8))
        );
    }

    #[test]
    fn face_key_ignores_orientation() {
        assert_eq!(face_key(&[3, 1, 2]), face_key(&[1, 2, 3]));
        assert_eq!(face_key(&[3, 1, 2]), [1, 2, 3, NO_NODE]);
        assert_ne!(face_key(&[1, 2, 3]), face_key(&[1, 2, 3, 4]));
    }
}
