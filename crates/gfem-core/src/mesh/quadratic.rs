//! 2 次四面体（Tet10）のメッシュの準備と検査（SPEC §2.3 / §3.1）。
//!
//! - [`elevate`]: 1 次の四面体メッシュの各辺の中点に節点を足し、Tet10 のメッシュにする。
//!   COMSOL が 1 次のメッシュのまま 2 次の要素で解くのと同じ考え方で、手持ちの
//!   メッシュと、そこで選んだ電極（`[EID, 面番号]`）がそのまま使える
//! - [`check_orders`]: 2 次要素の隣どうしで中点の節点が食い違っていないか、
//!   1 次要素と混ざっていないかを調べる
//!
//! # 1 次と 2 次を混ぜない理由
//!
//! 1 次要素と 2 次要素が面で接すると、2 次側の面にある中点の節点に相手がいない。
//! 電位がその面でつながらない（不連続になる）まま、行列は対称・半正定値で定数を
//! 零空間に持つので、**CG は何事もなく収束して、境界の近くだけ黙って間違った答えを返す**。
//! 気づけない壊れ方なので、混ざっていたら止める。

use crate::assemble::adjacency::build_node_elements;
use crate::element::ElementKind;
use crate::element::tet10::EDGES;
use crate::error::{GfemError, Result, format_sample_list};
use crate::mesh::{ElementBlock, Mesh, NodeIndex, Nodes};
use rayon::prelude::*;

/// エラー一覧に並べる最大件数。
const SAMPLE_LIMIT: usize = 20;

/// 1 タスクが受け持つ節点の数。
const NODES_PER_TASK: usize = 4096;

/// 1 次の四面体メッシュを 2 次（Tet10）に上げる。
///
/// 各辺の中点に節点を 1 つずつ足す。辺は真っ直ぐなまま（中点はちょうど 2 頂点の中央）。
/// 要素の ID・PID・頂点の並びは変えないので、面番号とケースファイルの電極の指定は
/// そのまま通じる。シェル要素（電極面）も頂点だけの CTRIA3 / CQUAD4 のまま残す。
///
/// # 足す節点の番号
///
/// 辺を両端の節点索引 `(a, b)`（`a < b`）の辞書順に並べ、その順に元の節点の後ろへ置く。
/// 節点 ID は元の最大 ID の次から振る。スレッドの数によらず同じ番号になる
/// （CLAUDE.md 設計原則 1）。
pub fn elevate(mesh: &Mesh) -> Result<Mesh> {
    if let Some(block) = mesh.solids.iter().find(|b| b.kind() != ElementKind::Tet4) {
        let kind = block.kind();
        let message = if kind == ElementKind::Tet10 {
            crate::tr!(
                "The mesh already consists of 10-node CTETRA (second-order) elements",
                "メッシュはすでに 10 節点の CTETRA（2 次要素）です"
            )
        } else {
            crate::tr!(
                "Only tetrahedral meshes can be raised to second order ({} is included)",
                "2 次に上げられるのは四面体のメッシュだけです（{} が含まれています）",
                kind.label()
            )
        };
        return Err(GfemError::unsupported(message).with_hint(crate::tr!(
            "Second-order elements are supported for tetrahedra only. \
             Use a mesh made of 4-node CTETRA elements",
            "2 次要素は四面体だけに対応しています。4 節点の CTETRA だけのメッシュを使ってください"
        )));
    }
    let Some(tets) = mesh.block(ElementKind::Tet4) else {
        return Err(GfemError::model(crate::tr!(
            "There are no solid elements, so the mesh cannot be raised to second order",
            "ソリッド要素が 1 つもないため、2 次に上げられません"
        )));
    };

    let n = mesh.nodes.len();
    let edges = EdgeTable::build(mesh);
    let n_edges = edges.len();

    // 節点索引と節点 ID が u32 に収まるか
    let max_id = mesh.nodes.ids().iter().copied().max().unwrap_or(0) as u64;
    if (n + n_edges) as u64 > u32::MAX as u64 || max_id + n_edges as u64 > u32::MAX as u64 {
        return Err(GfemError::unsupported(crate::tr!(
            "Raising the mesh to second order would need {} nodes, more than can be numbered",
            "2 次に上げると節点が {} 個になり、番号を振りきれません",
            n + n_edges
        ))
        .with_hint(crate::tr!(
            "Use a coarser mesh",
            "もっと粗いメッシュを使ってください"
        )));
    }

    // --- 節点: 元の節点の後ろに、辺の順で中点を置く ---
    let mut ids = mesh.nodes.ids().to_vec();
    ids.extend((1..=n_edges as u32).map(|k| max_id as u32 + k));
    let mut coords = mesh.nodes.coords().to_vec();
    coords.resize(3 * (n + n_edges), 0.0);
    {
        let midpoints = &mut coords[3 * n..];
        let mut pieces: Vec<(usize, &mut [f64])> = Vec::new();
        let mut rest = midpoints;
        for start in (0..n).step_by(NODES_PER_TASK) {
            let end = (start + NODES_PER_TASK).min(n);
            let len = edges.offsets[end] - edges.offsets[start];
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(3 * len);
            pieces.push((start, head));
            rest = tail;
        }
        pieces.into_par_iter().for_each(|(start, out)| {
            let end = (start + NODES_PER_TASK).min(n);
            let mut at = 0;
            for a in start..end {
                let pa = mesh.nodes.coord(a as NodeIndex);
                for &b in edges.of(a) {
                    let pb = mesh.nodes.coord(b);
                    for d in 0..3 {
                        out[at + d] = 0.5 * (pa[d] + pb[d]);
                    }
                    at += 3;
                }
            }
        });
    }

    // --- 要素: 頂点 4 つの後ろに、辺 (G5..G10 の順) の中点を並べる ---
    let mut conn = vec![0 as NodeIndex; tets.len() * 10];
    conn.par_chunks_mut(10).enumerate().for_each(|(e, out)| {
        let c = tets.conn_of(e);
        out[..4].copy_from_slice(c);
        for (k, [i, j]) in EDGES.into_iter().enumerate() {
            out[4 + k] = (n + edges.index(c[i], c[j])) as NodeIndex;
        }
    });

    Ok(Mesh {
        nodes: Nodes::new(ids, coords),
        solids: vec![ElementBlock::new(
            ElementKind::Tet10,
            tets.ids().to_vec(),
            tets.pids().to_vec(),
            conn,
        )],
        shells: mesh.shells.clone(),
        solid_properties: mesh.solid_properties.clone(),
        shell_properties: mesh.shell_properties.clone(),
        length_unit: mesh.length_unit,
        source: mesh.source.clone(),
    })
}

/// 四面体メッシュの辺の一覧。節点 `a` から、`a` より大きい索引の隣の節点を昇順に持つ。
///
/// 辺 `(a, b)`（`a < b`）の通し番号は `offsets[a] + (b が a の並びの何番目か)`。
struct EdgeTable {
    /// 長さ `節点数 + 1`
    offsets: Vec<usize>,
    targets: Vec<NodeIndex>,
}

impl EdgeTable {
    fn build(mesh: &Mesh) -> Self {
        let n = mesh.nodes.len();
        let node_elements = build_node_elements(mesh);
        // 節点の区間ごとに「その区間の節点から出る辺」を並べ、最後につなぐ。
        // 区間の中も区間どうしも節点の順なので、つないだ結果はスレッドの数によらない
        let chunks: Vec<(Vec<u32>, Vec<NodeIndex>)> = (0..n)
            .step_by(NODES_PER_TASK)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|start| {
                let end = (start + NODES_PER_TASK).min(n);
                let mut counts = Vec::with_capacity(end - start);
                let mut targets = Vec::new();
                let mut neighbours: Vec<NodeIndex> = Vec::new();
                for a in start..end {
                    neighbours.clear();
                    for &e in node_elements.of(a) {
                        let element = mesh.solid(e as usize);
                        neighbours.extend(
                            element.conn[..4]
                                .iter()
                                .copied()
                                .filter(|&b| b as usize > a),
                        );
                    }
                    neighbours.sort_unstable();
                    neighbours.dedup();
                    counts.push(neighbours.len() as u32);
                    targets.extend_from_slice(&neighbours);
                }
                (counts, targets)
            })
            .collect();

        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0usize);
        let total: usize = chunks.iter().map(|(_, t)| t.len()).sum();
        let mut targets = Vec::with_capacity(total);
        for (counts, t) in chunks {
            for c in counts {
                offsets.push(offsets.last().unwrap() + c as usize);
            }
            targets.extend_from_slice(&t);
        }
        Self { offsets, targets }
    }

    fn len(&self) -> usize {
        self.targets.len()
    }

    fn of(&self, a: usize) -> &[NodeIndex] {
        &self.targets[self.offsets[a]..self.offsets[a + 1]]
    }

    /// 辺 `(p, q)` の通し番号（向きは問わない）。
    fn index(&self, p: NodeIndex, q: NodeIndex) -> usize {
        let (a, b) = if p < q { (p, q) } else { (q, p) };
        let at = self
            .of(a as usize)
            .binary_search(&b)
            .expect("要素の辺は辺の一覧に必ず載っている");
        self.offsets[a as usize] + at
    }
}

/// 要素の次数が揃っているか、2 次要素の中点の節点が隣と食い違っていないかを調べる。
///
/// 解く前に必ず通す（SPEC §2.3）。読み込み時は、表示だけしたい場合があるので警告に留める
/// （孤立節点と同じ扱い。SPEC §2.2 (c)）。
///
/// 2 次要素（Tet10）だけのメッシュでは、次の 3 つを確かめる。どれも破れると、
/// 隣り合う要素で電位がつながらない（非適合な）メッシュになる。
///
/// 1. どの節点も「頂点」か「中点」のどちらか一方としてだけ使われている
/// 2. 中点の節点は、どの要素から見ても同じ辺（同じ 2 頂点）の中点である
/// 3. 同じ辺を持つ要素は、その辺の中点に同じ節点を使っている
///    （破れるのは、同じ位置に別の GRID を重ねて置いた場合が多い）
pub fn check_orders(mesh: &Mesh) -> Result<()> {
    let has_quadratic = mesh.solids.iter().any(|b| b.kind().is_quadratic());
    if !has_quadratic {
        return Ok(());
    }
    let linear: Vec<&ElementBlock> = mesh
        .solids
        .iter()
        .filter(|b| !b.kind().is_quadratic())
        .collect();
    if !linear.is_empty() {
        let counts: Vec<String> = mesh
            .solids
            .iter()
            .map(|b| format!("{} {}", b.kind().label(), b.len()))
            .collect();
        return Err(GfemError::model(crate::tr!(
            "First-order and second-order elements are mixed ({})",
            "1 次要素と 2 次要素が混ざっています（{}）",
            counts.join(", ")
        ))
        .with_hint(crate::tr!(
            "Where a first-order element touches a second-order one, the mid-side nodes of the \
             second-order element have no counterpart and the potential is not continuous there. \
             Make every element a 10-node CTETRA, or make every element first-order",
            "1 次要素と 2 次要素が接するところでは、2 次要素の辺の中点の節点に相手がおらず、\
             電位がつながりません。すべてを 10 節点の CTETRA にするか、すべてを 1 次要素にしてください"
        )));
    }

    let n = mesh.nodes.len();
    // 1. 頂点として使われているか・中点として使われているか
    let mut as_corner = vec![false; n];
    let mut as_mid = vec![false; n];
    for e in mesh.iter_solids() {
        for &c in &e.conn[..4] {
            as_corner[c as usize] = true;
        }
        for &m in &e.conn[4..] {
            as_mid[m as usize] = true;
        }
    }
    let both: Vec<String> = (0..n)
        .filter(|&i| as_corner[i] && as_mid[i])
        .map(|i| mesh.nodes.id(i as NodeIndex).to_string())
        .collect();
    if !both.is_empty() {
        return Err(GfemError::model(crate::tr!(
            "Some nodes are a corner of one 10-node CTETRA and a mid-side node of another ({}): ID={}",
            "ある 10 節点の CTETRA では頂点、別の要素では辺の中点として使われている節点があります（{} 件）: ID={}",
            both.len(),
            format_sample_list(&both, SAMPLE_LIMIT)
        ))
        .with_hint(order_hint()));
    }

    let node_elements = build_node_elements(mesh);
    // 要素 e の中で節点 m が乗る辺の両端（索引の小さい順）
    let edge_of = |e: u32, m: NodeIndex| -> Option<(NodeIndex, NodeIndex)> {
        let conn = mesh.solid(e as usize).conn;
        let k = conn[4..].iter().position(|&x| x == m)?;
        let [i, j] = EDGES[k];
        let (a, b) = (conn[i], conn[j]);
        Some(if a < b { (a, b) } else { (b, a) })
    };

    // 2. 中点の節点が、要素によって別の辺に乗っていないか
    let ambiguous: Vec<String> = (0..n)
        .into_par_iter()
        .filter(|&m| as_mid[m])
        .filter(|&m| {
            let mut seen = node_elements
                .of(m)
                .iter()
                .filter_map(|&e| edge_of(e, m as NodeIndex));
            let first = seen.next();
            seen.any(|edge| Some(edge) != first)
        })
        .map(|m| mesh.nodes.id(m as NodeIndex).to_string())
        .collect();
    if !ambiguous.is_empty() {
        return Err(GfemError::model(crate::tr!(
            "Some mid-side nodes lie on different edges depending on the element ({}): ID={}",
            "要素によって別の辺の中点になっている節点があります（{} 件）: ID={}",
            ambiguous.len(),
            format_sample_list(&ambiguous, SAMPLE_LIMIT)
        ))
        .with_hint(order_hint()));
    }

    // 3. 同じ辺の中点が、要素によって別の節点になっていないか
    let split: Vec<String> = (0..n)
        .into_par_iter()
        .filter(|&a| as_corner[a])
        .flat_map_iter(|a| {
            // 節点 a から出る辺（相手の索引が大きいもの）と、その中点
            let mut pairs: Vec<(NodeIndex, NodeIndex)> = Vec::new();
            for &e in node_elements.of(a) {
                let conn = mesh.solid(e as usize).conn;
                for (k, [i, j]) in EDGES.into_iter().enumerate() {
                    let (p, q) = (conn[i], conn[j]);
                    let other = if p as usize == a {
                        q
                    } else if q as usize == a {
                        p
                    } else {
                        continue;
                    };
                    if other as usize > a {
                        pairs.push((other, conn[4 + k]));
                    }
                }
            }
            pairs.sort_unstable();
            pairs.dedup();
            let ids = &mesh.nodes;
            pairs
                .windows(2)
                .filter(|w| w[0].0 == w[1].0)
                .map(|w| {
                    format!(
                        "{}-{} ({} / {})",
                        ids.id(a as NodeIndex),
                        ids.id(w[0].0),
                        ids.id(w[0].1),
                        ids.id(w[1].1)
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect();
    if !split.is_empty() {
        return Err(GfemError::model(crate::tr!(
            "Elements that share an edge use different mid-side nodes on it ({}): edge (mid-side nodes) = {}",
            "同じ辺を持つ要素どうしで、その辺の中点の節点が違います（{} 件）: 辺（中点の節点） = {}",
            split.len(),
            format_sample_list(&split, SAMPLE_LIMIT)
        ))
        .with_hint(crate::tr!(
            "The neighbouring elements are not connected at the mid-side node, so the potential \
             is not continuous there. Often a second GRID was placed at the same position; \
             merge the coincident nodes in the mesh generator",
            "隣の要素と中点の節点でつながっていないため、電位がつながりません。\
             同じ位置に別の GRID を重ねて置いている場合が多いので、メッシュ生成側で\
             重なった節点をまとめてください"
        )));
    }
    Ok(())
}

fn order_hint() -> String {
    crate::tr!(
        "The mid-side nodes G5–G10 of a 10-node CTETRA must lie on the edges G1-G2, G2-G3, \
         G3-G1, G1-G4, G2-G4 and G3-G4 in that order. Check the node order in the mesh generator",
        "10 節点の CTETRA の G5〜G10 は、辺 G1-G2, G2-G3, G3-G1, G1-G4, G2-G4, G3-G4 の中点を\
         この順に並べたものでなければなりません。メッシュ生成側で節点の並びを確認してください"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::shape;
    use crate::element::tet10;
    use crate::testing::kuhn_box;

    #[test]
    fn every_edge_gets_one_mid_side_node() {
        let mesh = kuhn_box([2, 2, 1], [2.0, 2.0, 1.0]);
        let q = elevate(&mesh).unwrap();
        // 足した節点の数は、要素の辺を重複なく数えた数に等しい
        let mut edges = std::collections::BTreeSet::new();
        for e in mesh.iter_solids() {
            for [i, j] in EDGES {
                let (a, b) = (e.conn[i], e.conn[j]);
                edges.insert((a.min(b), a.max(b)));
            }
        }
        assert_eq!(q.nodes.len(), mesh.nodes.len() + edges.len());
        assert_eq!(q.n_solid_elements(), mesh.n_solid_elements());
        assert_eq!(q.solids[0].kind(), ElementKind::Tet10);
        // 要素の ID・PID・頂点は変わらない
        let (a, b) = (mesh.block(ElementKind::Tet4).unwrap(), &q.solids[0]);
        assert_eq!(a.ids(), b.ids());
        assert_eq!(a.pids(), b.pids());
        for e in 0..a.len() {
            assert_eq!(a.conn_of(e), &b.conn_of(e)[..4]);
        }
        check_orders(&q).unwrap();
    }

    #[test]
    fn mid_side_nodes_sit_at_the_middle_of_their_edge() {
        let mesh = kuhn_box([2, 1, 1], [1.0, 0.5, 0.5]);
        let q = elevate(&mesh).unwrap();
        for e in q.iter_solids() {
            for (k, [i, j]) in EDGES.into_iter().enumerate() {
                let (pa, pb) = (q.nodes.coord(e.conn[i]), q.nodes.coord(e.conn[j]));
                let pm = q.nodes.coord(e.conn[4 + k]);
                for d in 0..3 {
                    assert_eq!(pm[d], 0.5 * (pa[d] + pb[d]));
                }
            }
        }
    }

    #[test]
    fn the_raised_elements_fill_the_same_volume() {
        let mesh = kuhn_box([2, 2, 2], [1.0, 2.0, 3.0]);
        let q = elevate(&mesh).unwrap();
        let mut total = 0.0;
        for e in q.iter_solids() {
            let mut p = [[0.0; 3]; 10];
            q.gather_coords(e.conn, &mut p);
            assert_eq!(tet10::orientation(&p), tet10::Orientation::Positive);
            total += tet10::volume(&p).unwrap();
        }
        assert!((total - 6.0).abs() < 1e-12, "{total}");
    }

    #[test]
    fn new_node_ids_follow_the_largest_id_and_are_reproducible() {
        let mesh = kuhn_box([3, 2, 2], [1.0, 1.0, 1.0]);
        let q = elevate(&mesh).unwrap();
        let n = mesh.nodes.len();
        let max_id = *mesh.nodes.ids().iter().max().unwrap();
        assert_eq!(&q.nodes.ids()[..n], mesh.nodes.ids());
        for (k, &id) in q.nodes.ids()[n..].iter().enumerate() {
            assert_eq!(id, max_id + 1 + k as u32);
        }
        // スレッドの数を変えても同じ番号・同じ座標になる
        let single = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| elevate(&mesh).unwrap());
        assert_eq!(single.nodes.ids(), q.nodes.ids());
        assert_eq!(
            single
                .nodes
                .coords()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            q.nodes
                .coords()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(single.solids[0].conn(), q.solids[0].conn());
    }

    #[test]
    fn many_tasks_give_the_same_edges_as_one() {
        // 節点の区間をまたぐ継ぎ目で辺が抜けたり重なったりしないこと。
        // NODES_PER_TASK より多い節点のメッシュで、辺の数を直接数えたものと比べる
        let mesh = kuhn_box([20, 15, 15], [1.0, 1.0, 1.0]);
        assert!(mesh.nodes.len() > NODES_PER_TASK);
        let table = EdgeTable::build(&mesh);
        let mut edges = std::collections::BTreeSet::new();
        for e in mesh.iter_solids() {
            for [i, j] in EDGES {
                let (a, b) = (e.conn[i], e.conn[j]);
                edges.insert((a.min(b), a.max(b)));
            }
        }
        assert_eq!(table.len(), edges.len());
        for (k, &(a, b)) in edges.iter().enumerate() {
            assert_eq!(table.index(a, b), k, "辺 ({a}, {b})");
        }
    }

    #[test]
    fn only_tetrahedral_first_order_meshes_can_be_raised() {
        let mesh = kuhn_box([1, 1, 1], [1.0, 1.0, 1.0]);
        let q = elevate(&mesh).unwrap();
        let e = elevate(&q).unwrap_err();
        assert!(e.to_string().contains("すでに"), "{e}");
    }

    /// 1 次のメッシュを上げたものの要素 `e` の節点を書き換えたメッシュ。
    fn with_changed_node(e: usize, local: usize, node: NodeIndex) -> Mesh {
        let mesh = elevate(&kuhn_box([2, 1, 1], [2.0, 1.0, 1.0])).unwrap();
        let block = &mesh.solids[0];
        let mut conn = block.conn().to_vec();
        conn[10 * e + local] = node;
        Mesh {
            solids: vec![ElementBlock::new(
                ElementKind::Tet10,
                block.ids().to_vec(),
                block.pids().to_vec(),
                conn,
            )],
            ..mesh
        }
    }

    #[test]
    fn a_duplicated_mid_side_node_is_found() {
        // 要素 0 の辺 0-1 の中点を、同じ位置に置いた別の節点に差し替える
        let base = elevate(&kuhn_box([2, 1, 1], [2.0, 1.0, 1.0])).unwrap();
        let mid = base.solids[0].conn_of(0)[4];
        let mut ids = base.nodes.ids().to_vec();
        let mut coords = base.nodes.coords().to_vec();
        ids.push(99_999);
        coords.extend_from_slice(&base.nodes.coord(mid));
        let changed = with_changed_node(0, 4, ids.len() as NodeIndex - 1);
        let mesh = Mesh {
            nodes: Nodes::new(ids, coords),
            ..changed
        };
        let e = check_orders(&mesh).unwrap_err();
        assert!(e.to_string().contains("中点の節点が違います"), "{e}");
    }

    #[test]
    fn a_mid_side_node_on_the_wrong_edge_is_found() {
        // 要素 0 の G5 と G6 を入れ替える（別の辺の中点を指す）
        let base = elevate(&kuhn_box([2, 1, 1], [2.0, 1.0, 1.0])).unwrap();
        let c = base.solids[0].conn_of(0).to_vec();
        let mesh = with_changed_node(0, 4, c[5]);
        let mesh = {
            let block = &mesh.solids[0];
            let mut conn = block.conn().to_vec();
            conn[5] = c[4];
            Mesh {
                solids: vec![ElementBlock::new(
                    ElementKind::Tet10,
                    block.ids().to_vec(),
                    block.pids().to_vec(),
                    conn,
                )],
                ..mesh
            }
        };
        let e = check_orders(&mesh).unwrap_err();
        assert!(e.to_string().contains("別の辺の中点"), "{e}");
    }

    #[test]
    fn a_corner_used_as_a_mid_side_node_is_found() {
        let base = elevate(&kuhn_box([2, 1, 1], [2.0, 1.0, 1.0])).unwrap();
        let corner = base.solids[0].conn_of(1)[0];
        let mesh = with_changed_node(0, 9, corner);
        let e = check_orders(&mesh).unwrap_err();
        assert!(e.to_string().contains("頂点"), "{e}");
    }

    #[test]
    fn mixing_orders_is_an_error() {
        let linear = kuhn_box([1, 1, 1], [1.0, 1.0, 1.0]);
        let quadratic = elevate(&linear).unwrap();
        let mixed = Mesh {
            solids: vec![
                ElementBlock::new(
                    ElementKind::Tet4,
                    vec![100],
                    vec![1],
                    linear.solids[0].conn_of(0).to_vec(),
                ),
                quadratic.solids[0].clone(),
            ],
            ..quadratic
        };
        let e = check_orders(&mixed).unwrap_err();
        assert!(e.to_string().contains("混ざっています"), "{e}");
        // 1 次だけなら何も言わない
        check_orders(&linear).unwrap();
    }

    #[test]
    fn the_body_surface_is_the_same_as_the_first_order_one() {
        // 面は頂点だけで照合するので、上げる前と同じ面が同じ要素・面番号で出る
        use crate::mesh::topology::extract_surface;
        let mesh = kuhn_box([2, 2, 2], [1.0, 1.0, 1.0]);
        let q = elevate(&mesh).unwrap();
        let (a, b) = (
            extract_surface(&mesh).unwrap(),
            extract_surface(&q).unwrap(),
        );
        assert_eq!(a.faces().len(), b.faces().len());
        for (fa, fb) in a.faces().iter().zip(b.faces()) {
            assert_eq!(fa.nodes(), fb.nodes());
            assert_eq!((fa.element, fa.local_face), (fb.element, fb.local_face));
        }
    }

    #[test]
    fn solving_with_second_order_elements_is_stopped_for_now() {
        // 電極荷重と後処理が 2 次に対応するまでは、黙って違う答えを出さないよう入口で止める
        use crate::case::Case;
        let mut case = Case::new();
        case.set_conductivity(1, 0.35).unwrap();
        let q = elevate(&kuhn_box([1, 1, 1], [1.0, 1.0, 1.0])).unwrap();
        let e = crate::analysis::solve(&q, &case, None).unwrap_err();
        assert!(e.to_string().contains("まだ使えません"), "{e}");
    }

    #[test]
    fn a_raised_inverted_tetrahedron_is_only_inverted() {
        // 節点順序の反転した Tet4 を上げると、全ての点でヤコビアンが負の Tet10 になる。
        // Tet4 と同じく |det J| で計算できる
        let text = "\
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,0.0,1.0
GRID,4,,0.0,1.0,0.0
CTETRA,10,1,1,2,3,4
";
        let (mesh, _) = crate::io::nastran::read_nastran_from(
            text.as_bytes(),
            std::path::Path::new("<test>"),
            crate::units::LengthUnit::Meter,
        )
        .unwrap();
        let q = elevate(&mesh).unwrap();
        let mut p = [[0.0; 3]; 10];
        q.gather_coords(q.solids[0].conn_of(0), &mut p);
        assert_eq!(tet10::orientation(&p), tet10::Orientation::Inverted);
        assert!((tet10::volume(&p).unwrap() - 1.0 / 6.0).abs() < 1e-15);
        // 符号を見ない共通の経路では負のヤコビアンとして弾かれる
        assert!(shape::volume::<tet10::Tet10>(&p).is_err());
    }
}
