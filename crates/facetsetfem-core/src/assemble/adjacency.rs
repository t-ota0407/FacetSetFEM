//! 節点 → 要素の隣接と、それを使った並列アセンブリ（SPEC §4.2 / §8 M4）。
//!
//! # なぜ「行ごと」に並列化するのか
//!
//! 要素ループを並列化して `atomicAdd` で足し込むと、浮動小数点の加算順序が
//! 実行ごとに変わり再現性が壊れる（CLAUDE.md 設計原則 1 / SPEC §4.2）。
//! そこで**行（節点）ごとにスレッドを割り当て**、その行に触る要素を
//! **通し番号の昇順**にたどって足す。逐次の要素ループでも行 i への加算は
//! 要素の昇順に来るので、**両者はビット単位で一致する**。
//!
//! # 代償
//!
//! 1 つの要素はその節点数（4〜10）ぶん重複して計算される（四面体は 1 行ずつ作るので、
//! 重複するのは要素剛性の 1 行ぶん）。要素剛性の再計算は
//! キャッシュに載る小さな計算なので、20 コアなら差し引きで十分速くなる。
//! 逆に要素剛性を全部保存しておく方式は、Tet4 で CSR の 2.5 倍のメモリを食う
//! （SPEC §4.2 の見積り）ので採らない。

use crate::assemble::csr::CsrMatrix;
use crate::assemble::stiffness::{element_stiffness, tet4_row, tet10_row};
use crate::element::ElementKind;
use crate::element::shape::MAX_ELEMENT_NODES;
use crate::error::{ErrorKind, FacetSetFemError, Result};
use crate::material::Conductivity;
use crate::mesh::{Mesh, PropId};
use crate::parallel::{ITEMS_PER_BUCKET, bucket_scatter, drop_in_background};
use rayon::prelude::*;
use std::collections::BTreeMap;

/// 1 タスクが担当する行数。
const ROWS_PER_TASK: usize = 1024;

/// 節点ごとに、それに接するソリッド要素の通し番号を持つ表。
#[derive(Debug, Clone, Default)]
pub struct NodeElements {
    /// 長さ `節点数 + 1`
    offsets: Vec<usize>,
    /// 要素の通し番号。節点ごとに**昇順**
    elements: Vec<u32>,
}

impl NodeElements {
    /// 節点 `node` に接する要素（通し番号の昇順）。
    pub fn of(&self, node: usize) -> &[u32] {
        &self.elements[self.offsets[node]..self.offsets[node + 1]]
    }

    /// 節点数。
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// 空か。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 総参照数（＝Σ 要素の節点数）。
    pub fn total_references(&self) -> usize {
        self.elements.len()
    }
}

/// 節点 → 要素の表を作る。
///
/// 要素の節点参照をすべて `(節点, 要素)` の組にして、振り分けソート
/// （`parallel::bucket_scatter`）で並べる。組の順はそのまま「節点ごとに、要素の通し番号の
/// 昇順」なので、要素を順にたどって逐次に積んだ表と 1 件もずれない。以前は逐次で、
/// 1000 万節点（参照 2.4 億件）で 0.8 秒かかっていた。
pub fn build_node_elements(mesh: &Mesh) -> NodeElements {
    node_elements_with(mesh, ITEMS_PER_BUCKET)
}

/// [`build_node_elements`] の本体。振り分けの 1 組あたりの枚数の目安を外から変えられる
/// （テストで組を極端に小さくして、振り分けの経路を確かめる）。
fn node_elements_with(mesh: &Mesh, per_bucket: usize) -> NodeElements {
    let n = mesh.nodes.len();
    // ブロックごとの参照の通し番号の始まり。ブロックの中では要素あたりの節点数が一定なので、
    // 参照の通し番号から要素と局所節点番号が決まる
    let mut blocks = Vec::with_capacity(mesh.solids.len());
    let (mut first_ref, mut first_element) = (0usize, 0usize);
    for block in &mesh.solids {
        let per_element = block.kind().node_count();
        blocks.push((first_ref, first_element, block, per_element));
        first_ref += block.len() * per_element;
        first_element += block.len();
    }
    let reference = |k: usize| -> (u32, u32) {
        let at = blocks.partition_point(|&(start, ..)| start <= k) - 1;
        let (start, first_element, block, per_element) = blocks[at];
        let (i, local) = ((k - start) / per_element, (k - start) % per_element);
        (block.conn_of(i)[local], (first_element + i) as u32)
    };
    // 節点で粗く振り分ける。組の中は参照の通し番号の順（＝要素の通し番号の順）に並ぶ
    let mut scattered = bucket_scatter(first_ref, n, per_bucket, |k| reference(k).0, reference);
    let shift = scattered.shift;

    // 組ごとに、節点ごとの計数ソートで仕上げる。組の中はすでに要素の順なので、節点ごとに
    // 数えて並べ直すだけで「節点ごとに要素の昇順」になる（比較ソートが要らない）。
    // 組は節点の範囲 [b << shift, (b + 1) << shift) を受け持つので、書き込む要素の区間も
    // 区間の始まり（offsets）の範囲も組どうしで重ならない
    let mut offsets = vec![0usize; n + 1];
    offsets[n] = first_ref;
    let mut elements = vec![0u32; first_ref];
    let mut element_pieces: Vec<&mut [u32]> = Vec::with_capacity(scattered.sizes.len());
    let mut starts = Vec::with_capacity(scattered.sizes.len());
    let mut rest: &mut [u32] = &mut elements;
    let mut start = 0usize;
    for &size in &scattered.sizes {
        let (head, tail) = std::mem::take(&mut rest).split_at_mut(size);
        element_pieces.push(head);
        starts.push(start);
        rest = tail;
        start += size;
    }
    let buckets = scattered.buckets_mut();
    buckets
        .into_par_iter()
        .zip(element_pieces)
        .zip(offsets[..n].par_chunks_mut(1 << shift))
        .zip(starts)
        .enumerate()
        .for_each(|(b, (((pairs, out), offsets), start))| {
            let first_node = b << shift;
            let mut cursor = vec![0usize; offsets.len()];
            for &(node, _) in pairs.iter() {
                cursor[node as usize - first_node] += 1;
            }
            let mut at = 0usize;
            for (offset, count) in offsets.iter_mut().zip(cursor.iter_mut()) {
                *offset = start + at;
                let c = *count;
                *count = at;
                at += c;
            }
            for &(node, element) in pairs.iter() {
                let slot = &mut cursor[node as usize - first_node];
                out[*slot] = element;
                *slot += 1;
            }
        });
    drop_in_background(scattered.items);
    NodeElements { offsets, elements }
}

/// 剛性行列を並列に組み立てる。
///
/// [`crate::assemble::assemble_stiffness`] と**ビット単位で同じ行列**になる
/// （モジュール冒頭の説明を参照）。
pub fn assemble_stiffness_parallel(
    mesh: &Mesh,
    conductivity: &BTreeMap<PropId, Conductivity>,
    adjacency: &NodeElements,
    matrix: &mut CsrMatrix,
) -> Result<()> {
    let n = matrix.n_rows();
    if adjacency.len() != n {
        return Err(FacetSetFemError::internal(crate::tr!(
            "The node-to-element table does not match the mesh",
            "節点 → 要素の表がメッシュと合っていません"
        )));
    }
    let (row_ptr, col_idx, values) = matrix.parts_mut();

    // 行ブロックごとに可変スライスへ切り分ける（互いに重ならないので並列に書ける）
    let mut tasks: Vec<(usize, usize, &mut [f64])> = Vec::new();
    let mut rest = values;
    let mut start = 0;
    while start < n {
        let end = (start + ROWS_PER_TASK).min(n);
        let (head, tail) = rest.split_at_mut(row_ptr[end] - row_ptr[start]);
        tasks.push((start, end, head));
        rest = tail;
        start = end;
    }

    tasks
        .into_par_iter()
        .try_for_each(|(start, end, slice)| -> Result<()> {
            let base = row_ptr[start];
            let mut local = [[0.0f64; MAX_ELEMENT_NODES]; MAX_ELEMENT_NODES];
            let mut row = [0.0f64; MAX_ELEMENT_NODES];
            // 直前に引いた PID と σ。隣り合う要素はたいてい同じ領域なので、表を引き直さない
            let mut last: Option<(PropId, &Conductivity)> = None;
            for i in start..end {
                let (s, e) = (row_ptr[i] - base, row_ptr[i + 1] - base);
                slice[s..e].fill(0.0);
                let columns = &col_idx[row_ptr[i]..row_ptr[i + 1]];

                for &element in adjacency.of(i) {
                    let reference = mesh.solid(element as usize);
                    let sigma = match last {
                        Some((pid, sigma)) if pid == reference.pid => sigma,
                        _ => {
                            let Some(sigma) = conductivity.get(&reference.pid) else {
                                return Err(FacetSetFemError::new(
                                    ErrorKind::Case,
                                    crate::tr!(
                                        "No conductivity is set for PID={}",
                                        "PID={} に導電率が設定されていません",
                                        reference.pid
                                    ),
                                )
                                .with_entity(crate::tr!(
                                    "element EID={}",
                                    "要素 EID={}",
                                    reference.id
                                ))
                                .with_hint(crate::tr!(
                                    "Add σ [S/m] for this PID to the case's materials",
                                    "ケースの materials に該当 PID の σ [S/m] を追加してください"
                                )));
                            };
                            last = Some((reference.pid, sigma));
                            sigma
                        }
                    };

                    let row_in_element = reference
                        .conn
                        .iter()
                        .position(|&node| node as usize == i)
                        .expect("隣接表に載っている以上、この要素は節点 i を含む");
                    // 要るのは要素剛性の row_in_element 行だけ。四面体（1 次・2 次）は
                    // その行だけを作る（全体を作ったときの同じ行と値はビット単位で同じ）
                    let nn = reference.conn.len();
                    match reference.kind {
                        ElementKind::Tet4 => row[..4].copy_from_slice(&tet4_row(
                            mesh,
                            &reference,
                            sigma,
                            row_in_element,
                        )?),
                        ElementKind::Tet10 => {
                            row = tet10_row(mesh, &reference, sigma, row_in_element)?;
                        }
                        _ => {
                            element_stiffness(mesh, &reference, sigma, &mut local)?;
                            row[..nn].copy_from_slice(&local[row_in_element][..nn]);
                        }
                    }
                    for (j, &column) in reference.conn.iter().enumerate() {
                        let Ok(k) = columns.binary_search(&column) else {
                            return Err(FacetSetFemError::internal(crate::tr!(
                                "The CSR structure has no entry ({i}, {column})",
                                "CSR 構造に成分 ({i}, {column}) がありません"
                            )));
                        };
                        slice[s + k] += row[j];
                    }
                }
            }
            Ok(())
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assemble::sparsity::build_sparsity;
    use crate::assemble::stiffness::assemble_stiffness;

    fn sigma_map(pid: PropId, value: f64) -> BTreeMap<PropId, Conductivity> {
        let mut m = BTreeMap::new();
        m.insert(pid, Conductivity::isotropic(value).unwrap());
        m
    }

    /// nx × ny × nz の格子（間隔 1）を Kuhn 分割した直方体。
    fn box_mesh(nx: usize, ny: usize, nz: usize) -> Mesh {
        crate::testing::kuhn_box([nx, ny, nz], [nx as f64, ny as f64, nz as f64])
    }

    #[test]
    fn the_parallel_table_matches_the_serial_procedure() {
        // 要素を通し番号の順にたどって逐次に積んだ表と 1 件もずれない。組を極端に小さく
        // した場合（振り分けの組が数千個）から既定の大きさまで
        let mesh = box_mesh(14, 9, 7);
        let n = mesh.nodes.len();
        let mut expected: Vec<Vec<u32>> = vec![Vec::new(); n];
        for e in mesh.iter_solids() {
            for &node in e.conn {
                expected[node as usize].push(e.index as u32);
            }
        }
        for per_bucket in [1, 7, 64, ITEMS_PER_BUCKET] {
            let table = node_elements_with(&mesh, per_bucket);
            assert_eq!(table.len(), n);
            for (node, list) in expected.iter().enumerate() {
                assert_eq!(
                    table.of(node),
                    &list[..],
                    "組の目安 {per_bucket}: 節点 {node}"
                );
            }
        }
    }

    #[test]
    fn node_element_lists_are_sorted_and_complete() {
        let mesh = box_mesh(2, 2, 2);
        let adjacency = build_node_elements(&mesh);
        assert_eq!(adjacency.len(), mesh.nodes.len());
        assert_eq!(adjacency.total_references(), mesh.n_solid_elements() * 4);
        for node in 0..adjacency.len() {
            let list = adjacency.of(node);
            assert!(!list.is_empty(), "節点 {node} がどの要素にも属していません");
            assert!(list.windows(2).all(|w| w[0] < w[1]), "昇順でない: {list:?}");
        }
        // 逆向きにも整合している
        for e in mesh.iter_solids() {
            for &node in e.conn {
                assert!(adjacency.of(node as usize).contains(&(e.index as u32)));
            }
        }
    }

    #[test]
    fn parallel_assembly_matches_serial_bitwise() {
        // CLAUDE.md 設計原則 1: 並列化しても行列が変わってはいけない
        let mesh = box_mesh(4, 3, 3);
        let sigma = sigma_map(1, 0.35);

        let mut serial = build_sparsity(&mesh).unwrap();
        assemble_stiffness(&mesh, &sigma, &mut serial).unwrap();

        let adjacency = build_node_elements(&mesh);
        let mut parallel = build_sparsity(&mesh).unwrap();
        assemble_stiffness_parallel(&mesh, &sigma, &adjacency, &mut parallel).unwrap();

        assert_eq!(serial.values(), parallel.values());
    }

    #[test]
    fn parallel_assembly_of_second_order_tetrahedra_matches_serial_bitwise() {
        // 2 次四面体は要素剛性の 1 行だけを作る経路を通る。領域を 2 つに分けて σ を変え、
        // 行の塊（ROWS_PER_TASK）をまたぐ大きさにする
        let linear =
            crate::testing::kuhn_box_with(
                [7, 5, 4],
                [1.0, 0.8, 0.6],
                |[i, ..]| {
                    if i < 3 { 1 } else { 2 }
                },
            );
        let mesh = crate::mesh::elevate(&linear).unwrap();
        assert!(mesh.nodes.len() > ROWS_PER_TASK);
        let mut sigma = sigma_map(1, 0.35);
        sigma.insert(2, Conductivity::isotropic(0.04).unwrap());

        let mut serial = build_sparsity(&mesh).unwrap();
        assemble_stiffness(&mesh, &sigma, &mut serial).unwrap();
        let adjacency = build_node_elements(&mesh);
        let mut parallel = build_sparsity(&mesh).unwrap();
        assemble_stiffness_parallel(&mesh, &sigma, &adjacency, &mut parallel).unwrap();
        let bits = |m: &CsrMatrix| m.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&serial), bits(&parallel));

        // 行和はゼロ（定数ベクトルが零空間。SPEC §2.2）で、対角は正
        let (row_ptr, col_idx, values) = (serial.row_ptr(), serial.col_idx(), serial.values());
        for i in 0..serial.n_rows() {
            let row = &values[row_ptr[i]..row_ptr[i + 1]];
            let diag = col_idx[row_ptr[i]..row_ptr[i + 1]]
                .iter()
                .position(|&c| c as usize == i)
                .map(|k| row[k])
                .unwrap();
            assert!(diag > 0.0, "行 {i}");
            let sum: f64 = row.iter().sum();
            assert!(sum.abs() < 1e-13 * diag, "行 {i} の和 {sum}");
        }
    }

    #[test]
    fn parallel_assembly_works_for_meshes_smaller_than_one_task() {
        let mesh = box_mesh(1, 1, 1);
        assert!(mesh.nodes.len() < ROWS_PER_TASK);
        let sigma = sigma_map(1, 1.0);

        let mut serial = build_sparsity(&mesh).unwrap();
        assemble_stiffness(&mesh, &sigma, &mut serial).unwrap();
        let adjacency = build_node_elements(&mesh);
        let mut parallel = build_sparsity(&mesh).unwrap();
        assemble_stiffness_parallel(&mesh, &sigma, &adjacency, &mut parallel).unwrap();
        assert_eq!(serial.values(), parallel.values());
    }

    #[test]
    fn missing_conductivity_is_reported() {
        let mesh = box_mesh(2, 2, 2);
        let adjacency = build_node_elements(&mesh);
        let mut matrix = build_sparsity(&mesh).unwrap();
        let e = assemble_stiffness_parallel(&mesh, &BTreeMap::new(), &adjacency, &mut matrix)
            .unwrap_err();
        assert!(e.to_string().contains("導電率が設定されていません"), "{e}");
    }
}
