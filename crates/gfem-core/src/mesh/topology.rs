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
use crate::mesh::{Mesh, NodeIndex, PropId};
use crate::parallel::{BucketKey, bucket_sort, drop_in_background};
use rayon::prelude::*;

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
///
/// 面の列挙・並べ替え・走査の 3 段とも並列に行うが、どの段も逐次と同じ並びを作るので、
/// 結果は逐次とビット単位で同じ（CLAUDE.md 設計原則 1）。1000 万節点（面 2.4 億件）で
/// 以前は 7.8 秒（逐次の面の列挙 4.0 秒 + 比較ソート 2.8 秒）、いまは 1.8 秒
/// （並べ替えは振り分けソート `parallel::bucket_sort`。SPEC §8 の「2026-09」）。
pub fn extract_surface(mesh: &Mesh) -> Result<Surface> {
    // ブロックごとの面の通し番号の始まり。ブロックの中では要素あたりの面の数が一定なので、
    // 面の通し番号から要素と局所面番号が決まる
    let mut blocks = Vec::with_capacity(mesh.solids.len());
    let (mut first_face, mut first_element) = (0usize, 0usize);
    for block in &mesh.solids {
        blocks.push((first_face, first_element, block, block.kind().faces()));
        first_face += block.len() * block.kind().faces().len();
        first_element += block.len();
    }
    // 面の通し番号 k → (面の節点, 節点数, 要素の通し番号, 局所面番号)
    let face = |k: usize| {
        let at = blocks.partition_point(|&(start, ..)| start <= k) - 1;
        let (start, first_element, block, faces) = blocks[at];
        let (i, lf) = ((k - start) / faces.len(), (k - start) % faces.len());
        let conn = block.conn_of(i);
        let local = faces[lf];
        let mut nodes = [0u32; 4];
        for (m, &l) in local.iter().enumerate() {
            nodes[m] = conn[l];
        }
        (nodes, local.len(), (first_element + i) as u32, lf as u8)
    };
    let entries = bucket_sort(
        first_face,
        mesh.nodes.len(),
        |k| {
            let (nodes, len, ..) = face(k);
            nodes[..len]
                .iter()
                .copied()
                .min()
                .expect("面には節点がある")
        },
        |k| {
            let (nodes, len, element, lf) = face(k);
            (face_key(&nodes[..len]), element, lf)
        },
    );

    surface_from_sorted(mesh, entries)
}

/// 指定した領域（PSOLID の PID）の要素の集まりの境界面。
///
/// 外表面のうちその領域が持つ面と、ほかの領域との境目の面からなり、向きは領域の外向き。
/// GUI で**体表面に出ていない領域**（頭部モデルの大半がそう）をハイライトするときに使う。
/// 以前は体積メッシュ全体を VTK の形式にしてから（頭部クラスで 0.36 秒・468 MB）、その領域の
/// 要素を切り出して VTK に外側の面を取らせていた（領域を選ぶたびに 0.2〜0.6 秒）。
/// 外表面の抽出と同じく、面キーの出現回数が 1 の面を拾う（対象をその領域の要素に絞るだけ）。
/// 領域に要素が無ければ空。
pub fn extract_region_surface(mesh: &Mesh, pid: PropId) -> Result<Surface> {
    // ブロックごとに、その領域の要素（ブロック内の番号、昇順）と、面の通し番号の始まり
    let mut segments = Vec::with_capacity(mesh.solids.len());
    let (mut first_face, mut first_element) = (0usize, 0usize);
    for block in &mesh.solids {
        let members: Vec<u32> = (0..block.len() as u32)
            .into_par_iter()
            .filter(|&i| block.pids()[i as usize] == pid)
            .collect();
        let faces = block.kind().faces();
        let n_faces = members.len() * faces.len();
        segments.push((first_face, first_element, block, faces, members));
        first_face += n_faces;
        first_element += block.len();
    }
    // 面の通し番号 k → (面の節点, 節点数, 要素の通し番号, 局所面番号)。外表面の抽出と同じく
    // 振り分けソートで並べる（要素 216 万個・面 864 万枚で、比較ソートより速い）
    let face = |k: usize| {
        let at = segments.partition_point(|&(start, ..)| start <= k) - 1;
        let (start, first_element, block, faces, members) = &segments[at];
        let (m, lf) = ((k - start) / faces.len(), (k - start) % faces.len());
        let i = members[m] as usize;
        let conn = block.conn_of(i);
        let local = faces[lf];
        let mut nodes = [0u32; 4];
        for (q, &l) in local.iter().enumerate() {
            nodes[q] = conn[l];
        }
        (nodes, local.len(), (first_element + i) as u32, lf as u8)
    };
    let entries = bucket_sort(
        first_face,
        mesh.nodes.len(),
        |k| {
            let (nodes, len, ..) = face(k);
            nodes[..len]
                .iter()
                .copied()
                .min()
                .expect("面には節点がある")
        },
        |k| {
            let (nodes, len, element, lf) = face(k);
            (face_key(&nodes[..len]), element, lf)
        },
    );
    surface_from_sorted(mesh, entries)
}

/// 面キーの順に並べた面の列から、出現回数 1 の面（境界）を拾って [`Surface`] にする。
///
/// 3 回以上現れる面（非多様体）があればエラー。
fn surface_from_sorted(mesh: &Mesh, entries: Vec<FaceEntry>) -> Result<Surface> {
    // 同じ面キーの並びが途中で切れないように区切って、区間ごとに並列に走査する。
    // 区間の結果を区間の順につなぐので、逐次に走査したのと同じ並びになる
    let parts: Vec<Scan> = group_ranges(&entries, rayon::current_num_threads() * 4)
        .into_par_iter()
        .map(|range| scan_surface(mesh, &entries[range]))
        .collect();
    drop_in_background(entries);
    let mut faces: Vec<SurfaceFace> = Vec::with_capacity(parts.iter().map(|p| p.faces.len()).sum());
    let mut keys: Vec<FaceKey> = Vec::with_capacity(faces.capacity());
    let mut non_manifold: Vec<String> = Vec::new();
    for part in parts {
        faces.extend(part.faces);
        keys.extend(part.keys);
        non_manifold.extend(part.non_manifold);
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

/// `(面キー, 要素の通し番号, 局所面番号)`。
type FaceEntry = (FaceKey, u32, u8);

/// 面の列は面キーの最小の節点（先頭）で振り分けて並べる（[`crate::parallel::bucket_sort`]）。
impl BucketKey for FaceEntry {
    fn bucket_key(&self) -> u32 {
        self.0[0]
    }
}

/// 並べた列を、同じ面キーの並びが途中で切れないように `parts` 個ほどの区間に分ける。
fn group_ranges(entries: &[FaceEntry], parts: usize) -> Vec<std::ops::Range<usize>> {
    let n = entries.len();
    let parts = parts.clamp(1, n.max(1));
    let mut starts = vec![0usize];
    for p in 1..parts {
        // 等分点から、面キーの変わり目まで進める
        let mut at = (n * p / parts).max(*starts.last().unwrap());
        while at > 0 && at < n && entries[at].0 == entries[at - 1].0 {
            at += 1;
        }
        if at > *starts.last().unwrap() && at < n {
            starts.push(at);
        }
    }
    starts.push(n);
    starts.windows(2).map(|w| w[0]..w[1]).collect()
}

/// 1 区間の走査結果。
struct Scan {
    faces: Vec<SurfaceFace>,
    keys: Vec<FaceKey>,
    non_manifold: Vec<String>,
}

/// 並べた列の 1 区間から、出現回数 1 の面（外表面）と、3 回以上の面（非多様体）を拾う。
fn scan_surface(mesh: &Mesh, entries: &[FaceEntry]) -> Scan {
    let mut out = Scan {
        faces: Vec::new(),
        keys: Vec::new(),
        non_manifold: Vec::new(),
    };
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
            out.faces.push(SurfaceFace {
                nodes,
                n_nodes: local.len() as u8,
                element: elem,
                local_face,
            });
            out.keys.push(key);
        } else if count > 2 {
            let ids: Vec<String> = entries[i..j]
                .iter()
                .map(|&(_, elem, _)| mesh.solid(elem as usize).id.to_string())
                .collect();
            out.non_manifold.push(format!("要素 EID={}", ids.join("/")));
        }
        i = j;
    }
    out
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
pub fn build_face_adjacency(tets: &TetView) -> FaceAdjacency {
    let n = tets.len();
    let faces = ElementKind::Tet4.faces();
    let nodes = |k: usize| -> [NodeIndex; 3] {
        let conn = tets.conn_of(k / 4);
        let face = faces[k % 4];
        [conn[face[0]], conn[face[1]], conn[face[2]]]
    };
    let entries = bucket_sort(
        n * 4,
        tets.n_nodes(),
        |k| nodes(k).into_iter().min().expect("面には節点がある"),
        |k| (face_key(&nodes(k)), (k / 4) as u32, (k % 4) as u8),
    );

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
    drop_in_background(entries);
    adjacency
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
        let adjacency = build_face_adjacency(&TetView::build(&mesh).unwrap());
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

    /// 以前の逐次の手順そのまま（列挙・並べ替え・走査をすべて逐次で行う）。
    fn serial_surface(mesh: &Mesh) -> (Vec<SurfaceFace>, Vec<FaceKey>) {
        let mut entries: Vec<FaceEntry> = Vec::new();
        for e in mesh.iter_solids() {
            for (lf, face) in e.kind.faces().iter().enumerate() {
                let nodes: Vec<NodeIndex> = face.iter().map(|&l| e.conn[l]).collect();
                entries.push((face_key(&nodes), e.index as u32, lf as u8));
            }
        }
        entries.sort_unstable();
        let scan = scan_surface(mesh, &entries);
        (scan.faces, scan.keys)
    }

    /// 六面体・三角柱・四面体のブロックが並んだメッシュ（要素の通し番号がブロックをまたぐ）。
    fn mixed_blocks() -> Mesh {
        let mut deck = String::from("PSOLID,1,1\n");
        // 3 × 2 × 2 の格子点
        let node = |i: usize, j: usize, k: usize| 1 + i + 4 * (j + 3 * k);
        for k in 0..3 {
            for j in 0..3 {
                for i in 0..4 {
                    deck.push_str(&format!("GRID,{},,{i}.0,{j}.0,{k}.0\n", node(i, j, k)));
                }
            }
        }
        let mut eid = 1;
        for k in 0..2 {
            for j in 0..2 {
                let c = |i: usize, dj: usize, dk: usize| node(i, j + dj, k + dk);
                // x = 0〜1 は六面体
                deck.push_str(&format!(
                    "CHEXA,{eid},1,{},{},{},{},{},{},\n,{},{}\n",
                    c(0, 0, 0),
                    c(1, 0, 0),
                    c(1, 1, 0),
                    c(0, 1, 0),
                    c(0, 0, 1),
                    c(1, 0, 1),
                    c(1, 1, 1),
                    c(0, 1, 1)
                ));
                eid += 1;
                // x = 1〜2 は三角柱 2 つ（軸は x 方向）
                for tri in [[(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 1), (0, 1)]] {
                    let [a, b, d] = tri;
                    deck.push_str(&format!(
                        "CPENTA,{eid},1,{},{},{},{},{},{}\n",
                        c(1, a.0, a.1),
                        c(1, b.0, b.1),
                        c(1, d.0, d.1),
                        c(2, a.0, a.1),
                        c(2, b.0, b.1),
                        c(2, d.0, d.1)
                    ));
                    eid += 1;
                }
                // x = 2〜3 は四面体（Kuhn の 6 分割）
                let corner = |q: usize| c(2 + (q & 1), (q >> 1) & 1, (q >> 2) & 1);
                for t in [
                    [0, 1, 3, 7],
                    [0, 1, 7, 5],
                    [0, 5, 7, 4],
                    [0, 3, 2, 7],
                    [0, 6, 4, 7],
                    [0, 2, 6, 7],
                ] {
                    deck.push_str(&format!(
                        "CTETRA,{eid},1,{},{},{},{}\n",
                        corner(t[0]),
                        corner(t[1]),
                        corner(t[2]),
                        corner(t[3])
                    ));
                    eid += 1;
                }
            }
        }
        read(&deck)
    }

    #[test]
    fn parallel_extraction_matches_the_serial_procedure() {
        // 面の列挙・並べ替え・走査を並列にしても、並びまで逐次と同じ（CLAUDE.md 設計原則 1）
        let mixed = mixed_blocks();
        assert_eq!(
            mixed.solids.len(),
            3,
            "六面体・三角柱・四面体の 3 ブロックのはず"
        );
        assert_eq!(mixed.n_solid_elements(), 4 + 8 + 24);
        // 30 × 20 × 10 の箱は面が 14 万枚で、振り分けの組が数十個になる
        for mesh in [
            crate::testing::kuhn_box([9, 7, 5], [1.0; 3]),
            crate::testing::kuhn_box([30, 20, 10], [1.0; 3]),
            mixed,
        ] {
            let surface = extract_surface(&mesh).unwrap();
            let (faces, keys) = serial_surface(&mesh);
            assert!(!faces.is_empty());
            assert_eq!(surface.faces, faces);
            assert_eq!(surface.keys, keys);
        }
    }

    #[test]
    fn bucket_sorting_gives_exactly_the_full_sort() {
        // 振り分けてから組ごとに並べても、全体を比較ソートしたのと 1 件もずれない。
        // 組を極端に小さくした場合（1 組 1 件の目安）から、組が 1 つの場合まで確かめる
        let n_nodes = 5_000usize;
        let entry = |k: usize| -> FaceEntry {
            // 同じ面キーが 2 回ずつ現れる、節点の散らばった面（内部の面を模す）
            let face = (k / 2) as u64;
            let mut x = face
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let mut nodes = [0u32; 3];
            for node in &mut nodes {
                x ^= x >> 29;
                x = x.wrapping_mul(0xbf58476d1ce4e5b9);
                *node = (x % n_nodes as u64) as u32;
            }
            (face_key(&nodes), (k % 977) as u32, (k % 4) as u8)
        };
        let n = 40_000;
        let mut expected: Vec<FaceEntry> = (0..n).map(entry).collect();
        expected.sort_unstable();
        for faces_per_bucket in [1, 3, 64, 4096, usize::MAX] {
            let got = crate::parallel::bucket_sort_with(
                n,
                n_nodes,
                faces_per_bucket,
                |k| entry(k).0[0],
                entry,
            );
            assert!(got == expected, "組の目安 {faces_per_bucket} 件でずれた");
        }
        assert!(bucket_sort(0, n_nodes, |k| entry(k).0[0], entry).is_empty());
    }

    #[test]
    fn face_adjacency_pairs_every_interior_face() {
        // 面の隣接も同じ振り分けの並べ替えを使う。組が十数個になる大きさ（面 7.2 万枚）で、
        // 内部の面がすべて正しく組になり、残りがちょうど外表面になる
        let mesh = crate::testing::kuhn_box([20, 15, 10], [1.0; 3]);
        let tets = TetView::build(&mesh).unwrap();
        let adjacency = build_face_adjacency(&tets);
        for t in 0..tets.len() {
            for f in 0..4 {
                if let Some((other, of)) = adjacency.neighbour(t, f) {
                    // 相互に指し合い、同じ面キーを持つ
                    assert_eq!(
                        adjacency.neighbour(other as usize, of as usize),
                        Some((t as u32, f as u8))
                    );
                    let key = |t: usize, f: usize| {
                        let face = ElementKind::Tet4.faces()[f];
                        let conn = tets.conn_of(t);
                        face_key(&[conn[face[0]], conn[face[1]], conn[face[2]]])
                    };
                    assert_eq!(key(t, f), key(other as usize, of as usize));
                }
            }
        }
        // 外表面の面の数と、隣接の無い面の数が一致する
        let open = (0..tets.len())
            .flat_map(|t| (0..4).map(move |f| (t, f)))
            .filter(|&(t, f)| adjacency.neighbour(t, f).is_none())
            .count();
        assert_eq!(open, extract_surface(&mesh).unwrap().len());
    }

    #[test]
    fn a_buried_region_is_bounded_by_its_interface() {
        // 4 × 4 × 4 の箱の中央 2 × 2 × 2 だけが PID 2。その境界は 1 辺 2 の立方体の表面
        let mesh = crate::testing::kuhn_box_with([4, 4, 4], [4.0; 3], |[i, j, k]| {
            if (1..3).contains(&i) && (1..3).contains(&j) && (1..3).contains(&k) {
                2
            } else {
                1
            }
        });
        let inner = extract_region_surface(&mesh, 2).unwrap();
        assert!((inner.total_area(&mesh) - 24.0).abs() < 1e-12);
        // 境界の面はどれも PID 2 の要素のもので、向きは領域の外向き（中心から離れる向き）
        for face in inner.faces() {
            assert_eq!(mesh.solid(face.element as usize).pid, 2);
            let p: Vec<[f64; 3]> = face.nodes().iter().map(|&n| mesh.nodes.coord(n)).collect();
            let normal =
                crate::geom::cross(crate::geom::sub(p[1], p[0]), crate::geom::sub(p[2], p[0]));
            let centre = crate::geom::sub(p[0], [2.0; 3]);
            assert!(crate::geom::dot(normal, centre) > 0.0, "内向きの面がある");
        }
        // 外側の領域の境界は、箱の外表面と中の空洞の面の両方
        let outer = extract_region_surface(&mesh, 1).unwrap();
        assert!((outer.total_area(&mesh) - (6.0 * 16.0 + 24.0)).abs() < 1e-12);
        // 外側の領域の境界から空洞の面（＝内側の領域の境界）を除けば、箱の外表面そのもの
        assert_eq!(
            outer.len() - inner.len(),
            extract_surface(&mesh).unwrap().len()
        );
        // 無い PID は空
        assert!(extract_region_surface(&mesh, 99).unwrap().is_empty());
    }

    #[test]
    fn group_ranges_never_split_a_run_of_equal_keys() {
        // 同じ面キーが 1〜3 回ずつ続く列。どの切り方でも、区間は隙間なく並び、
        // 同じ面キーの並びを途中で切らない
        let mut entries: Vec<FaceEntry> = Vec::new();
        for k in 0..500u32 {
            for r in 0..(1 + k % 3) {
                entries.push(([k, k + 1, k + 2, NO_NODE], r, 0));
            }
        }
        for parts in [1, 2, 3, 7, 64, 10_000] {
            let ranges = group_ranges(&entries, parts);
            assert_eq!(ranges.first().unwrap().start, 0);
            assert_eq!(ranges.last().unwrap().end, entries.len());
            for w in ranges.windows(2) {
                assert_eq!(w[0].end, w[1].start);
                assert!(w[0].start < w[0].end, "空の区間がある");
                assert_ne!(
                    entries[w[1].start].0,
                    entries[w[1].start - 1].0,
                    "{parts}: 並びを切った"
                );
            }
        }
        assert_eq!(group_ranges(&[], 8), vec![0..0]);
    }
}
