//! CSR の非ゼロ構造の構築（SPEC §4.1 / §4.2）。
//!
//! 節点 i の行に入る列は「i に接する要素の全節点」。節点 → 要素の表から行ごとに候補を集め、
//! ソートして重複を詰める。ハッシュを使わないので、同じメッシュからは常に同じ構造
//! （同じ列の並び）になる。行ごとに独立なので並列に作っても逐次と同じ。
//!
//! 以前は要素ごとに候補を各節点へ撒く作業配列（1000 万節点で 3.8 GB）を逐次で埋めていて、
//! 2.5 秒かかった。節点 → 要素の表は並列アセンブリでも使うので、作るのは 1 回で済む。

use crate::assemble::adjacency::{NodeElements, build_node_elements};
use crate::assemble::csr::CsrMatrix;
use crate::error::{GfemError, Result, format_sample_list};
use crate::mesh::Mesh;
use crate::solver::amg::rowwise::ROWS_PER_TASK;
use rayon::prelude::*;

/// エラーに並べる最大件数。
const SAMPLE_LIMIT: usize = 20;

/// メッシュから CSR の非ゼロ構造を作る（値は 0）。
///
/// どの要素からも参照されていない節点があるとその行が空になり、剛性行列の零空間が
/// span{1} より広がってしまう（＝射影 CG の前提が崩れる）ので、ここでエラーにする。
/// 読み込み時（SPEC §3.1）には警告に留めているのは、可視化だけしたい場合があるため。
pub fn build_sparsity(mesh: &Mesh) -> Result<CsrMatrix> {
    build_sparsity_with(mesh, &build_node_elements(mesh))
}

/// 節点 → 要素の表から非ゼロ構造を作る（並列アセンブリで同じ表を使い回すため）。
pub fn build_sparsity_with(mesh: &Mesh, adjacency: &NodeElements) -> Result<CsrMatrix> {
    let n = mesh.nodes.len();
    if n == 0 {
        return Err(GfemError::model(crate::tr!(
            "There are no nodes",
            "節点がありません"
        )));
    }
    if adjacency.len() != n {
        return Err(GfemError::internal(crate::tr!(
            "The node-to-element table does not match the mesh",
            "節点 → 要素の表がメッシュと合っていません"
        )));
    }

    // 孤立節点の検出
    let orphans: Vec<String> = (0..n)
        .filter(|&i| adjacency.of(i).is_empty())
        .map(|i| mesh.nodes.id(i as u32).to_string())
        .collect();
    if !orphans.is_empty() {
        return Err(GfemError::model(crate::tr!(
            "Some nodes are not referenced by any element ({}): ID={}",
            "どの要素からも参照されていない節点があります（{} 件）: ID={}",
            orphans.len(),
            format_sample_list(&orphans, SAMPLE_LIMIT)
        ))
        .with_hint(crate::tr!(
            "The potential at an isolated node is not determined by the equations, so the \
             solution is not unique. Remove such nodes from the mesh before solving",
            "孤立節点の電位は方程式から決まらないため、解が一意になりません。\
             メッシュから取り除いてから解いてください"
        )));
    }

    // 1. 行の塊ごとに、接する要素の全節点を集めて並べ、重複を詰める。塊の手元に溜めておき、
    //    行の長さが揃ってから所定の位置へ写す（候補を 2 回集めずに済む）
    let n_blocks = n.div_ceil(ROWS_PER_TASK);
    let blocks: Vec<(Vec<u32>, Vec<u32>)> = (0..n_blocks)
        .into_par_iter()
        .map_init(
            || Marks::new(n),
            |marks, b| {
                let rows = b * ROWS_PER_TASK..((b + 1) * ROWS_PER_TASK).min(n);
                let mut lengths = Vec::with_capacity(rows.len());
                let mut cols = Vec::new();
                for i in rows {
                    // 候補（接する要素の全節点。四面体なら 1 行 100 個ほど）の重複を目印で先に
                    // 除き、残った 15 個ほどだけを並べる。全部を並べてから詰めるより軽い
                    marks.start_row();
                    for &element in adjacency.of(i) {
                        for &node in mesh.solid(element as usize).conn {
                            marks.insert(node);
                        }
                    }
                    let row = marks.sorted();
                    lengths.push(row.len() as u32);
                    cols.extend_from_slice(row);
                }
                (lengths, cols)
            },
        )
        .collect();

    // 2. 行ポインタ
    let mut row_ptr = Vec::with_capacity(n + 1);
    row_ptr.push(0usize);
    let mut total = 0usize;
    for (lengths, _) in &blocks {
        for &len in lengths {
            total += len as usize;
            row_ptr.push(total);
        }
    }

    // 3. 塊ごとに写す（塊どうしで書く場所は重ならない）
    let mut col_idx = vec![0u32; total];
    let mut pieces: Vec<&mut [u32]> = Vec::with_capacity(n_blocks);
    let mut rest: &mut [u32] = &mut col_idx;
    for (_, cols) in &blocks {
        let (head, tail) = std::mem::take(&mut rest).split_at_mut(cols.len());
        pieces.push(head);
        rest = tail;
    }
    pieces
        .into_par_iter()
        .zip(blocks.par_iter())
        .for_each(|(out, (_, cols))| out.copy_from_slice(cols));

    Ok(CsrMatrix::from_structure(n, row_ptr, col_idx))
}

/// 行の候補の重複を除くための目印（行ごとに世代番号を進めるので、クリアが要らない）。
struct Marks {
    /// 節点ごとの「最後に入れた行の世代番号」
    stamp: Vec<u32>,
    generation: u32,
    row: Vec<u32>,
}

impl Marks {
    fn new(n_nodes: usize) -> Self {
        Self {
            // ゼロで確保するので、触った節点のページだけが実際に割り当てられる
            stamp: vec![0; n_nodes],
            generation: 0,
            row: Vec::new(),
        }
    }

    fn start_row(&mut self) {
        self.generation += 1;
        self.row.clear();
    }

    fn insert(&mut self, node: u32) {
        let stamp = &mut self.stamp[node as usize];
        if *stamp != self.generation {
            *stamp = self.generation;
            self.row.push(node);
        }
    }

    /// この行の列（昇順、重複なし）。
    fn sorted(&mut self) -> &[u32] {
        self.row.sort_unstable();
        &self.row
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::nastran::read_nastran_from;
    use crate::units::LengthUnit;
    use std::path::Path;

    fn read(text: &str) -> Mesh {
        read_nastran_from(text.as_bytes(), Path::new("<test>"), LengthUnit::Meter)
            .unwrap()
            .0
    }

    /// 以前の逐次の手順（要素ごとに候補を各節点へ撒き、行ごとに並べて詰める）。
    fn serial_structure(mesh: &Mesh) -> (Vec<usize>, Vec<u32>) {
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); mesh.nodes.len()];
        for e in mesh.iter_solids() {
            for &node in e.conn {
                rows[node as usize].extend_from_slice(e.conn);
            }
        }
        let mut row_ptr = vec![0usize];
        let mut col_idx = Vec::new();
        for mut row in rows {
            row.sort_unstable();
            row.dedup();
            col_idx.extend(row);
            row_ptr.push(col_idx.len());
        }
        (row_ptr, col_idx)
    }

    #[test]
    fn the_parallel_structure_matches_the_serial_procedure() {
        // 行の塊が複数になる大きさ（3,696 節点）で、以前の逐次の手順と同じ構造になる
        let mesh = crate::testing::kuhn_box([20, 15, 10], [1.0; 3]);
        assert!(mesh.nodes.len() > ROWS_PER_TASK);
        let m = build_sparsity(&mesh).unwrap();
        let (row_ptr, col_idx) = serial_structure(&mesh);
        assert_eq!(m.row_ptr(), &row_ptr[..]);
        assert_eq!(m.col_idx(), &col_idx[..]);
    }

    #[test]
    fn single_tet_is_dense() {
        let mesh = read(
            "GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
GRID,4,,0.0,0.0,1.0
PSOLID,1,1
CTETRA,10,1,1,2,3,4
",
        );
        let m = build_sparsity(&mesh).unwrap();
        assert_eq!(m.n_rows(), 4);
        // 四面体 1 個なら全節点が互いに繋がる = 4x4 の密行列
        assert_eq!(m.nnz(), 16);
        for i in 0..4 {
            let (cols, _) = m.row(i);
            assert_eq!(cols, &[0, 1, 2, 3]);
        }
    }

    #[test]
    fn columns_are_sorted_and_unique() {
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
        let m = build_sparsity(&mesh).unwrap();
        for i in 0..m.n_rows() {
            let (cols, _) = m.row(i);
            assert!(cols.windows(2).all(|w| w[0] < w[1]), "行 {i}: {cols:?}");
            // 対角成分が必ずある
            assert!(cols.contains(&(i as u32)));
        }
        assert!(m.is_symmetric(0.0));
    }

    #[test]
    fn orphan_nodes_are_rejected() {
        let mesh = read(
            "GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
GRID,4,,0.0,0.0,1.0
GRID,99,,5.0,5.0,5.0
PSOLID,1,1
CTETRA,10,1,1,2,3,4
",
        );
        let e = build_sparsity(&mesh).unwrap_err();
        assert!(e.to_string().contains("ID=99"), "{e}");
        assert!(e.to_string().contains("一意"), "{e}");
    }
}
