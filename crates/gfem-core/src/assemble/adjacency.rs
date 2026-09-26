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
//! 1 つの要素はその節点数（4〜8）ぶん重複して計算される。要素剛性の再計算は
//! キャッシュに載る小さな計算なので、20 コアなら差し引きで十分速くなる。
//! 逆に要素剛性を全部保存しておく方式は、Tet4 で CSR の 2.5 倍のメモリを食う
//! （SPEC §4.2 の見積り）ので採らない。

use crate::assemble::csr::CsrMatrix;
use crate::assemble::stiffness::element_stiffness;
use crate::element::shape::MAX_ELEMENT_NODES;
use crate::error::{ErrorKind, GfemError, Result};
use crate::material::Conductivity;
use crate::mesh::{Mesh, PropId};
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
pub fn build_node_elements(mesh: &Mesh) -> NodeElements {
    let n = mesh.nodes.len();
    let mut offsets = vec![0usize; n + 1];
    for e in mesh.iter_solids() {
        for &node in e.conn {
            offsets[node as usize + 1] += 1;
        }
    }
    for i in 0..n {
        offsets[i + 1] += offsets[i];
    }
    let mut elements = vec![0u32; offsets[n]];
    let mut cursor = offsets.clone();
    // 要素を通し番号の昇順にたどるので、各節点のリストも自然に昇順になる
    for e in mesh.iter_solids() {
        for &node in e.conn {
            let c = &mut cursor[node as usize];
            elements[*c] = e.index as u32;
            *c += 1;
        }
    }
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
        return Err(GfemError::internal(
            "節点 → 要素の表がメッシュと合っていません",
        ));
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
            for i in start..end {
                let (s, e) = (row_ptr[i] - base, row_ptr[i + 1] - base);
                slice[s..e].fill(0.0);
                let columns = &col_idx[row_ptr[i]..row_ptr[i + 1]];

                for &element in adjacency.of(i) {
                    let reference = mesh.solid(element as usize);
                    let Some(sigma) = conductivity.get(&reference.pid) else {
                        return Err(GfemError::new(
                            ErrorKind::Case,
                            format!("PID={} に導電率が設定されていません", reference.pid),
                        )
                        .with_entity(format!("要素 EID={}", reference.id))
                        .with_hint("ケースの materials に該当 PID の σ [S/m] を追加してください"));
                    };
                    element_stiffness(mesh, &reference, sigma, &mut local)?;

                    let row_in_element = reference
                        .conn
                        .iter()
                        .position(|&node| node as usize == i)
                        .expect("隣接表に載っている以上、この要素は節点 i を含む");
                    for (j, &column) in reference.conn.iter().enumerate() {
                        let Ok(k) = columns.binary_search(&column) else {
                            return Err(GfemError::internal(format!(
                                "CSR 構造に成分 ({i}, {column}) がありません"
                            )));
                        };
                        slice[s + k] += local[row_in_element][j];
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
