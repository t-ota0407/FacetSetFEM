//! CSR の非ゼロ構造の構築（SPEC §4.1 / §4.2）。
//!
//! 節点 i の行に入る列は「i に接する要素の全節点」。要素ごとに候補を撒いてから
//! 行単位でソート・重複除去する、いわゆるカウンティング方式で作る。
//! ハッシュを使わないので、同じメッシュからは常に同じ構造（同じ列の並び）になる。

use crate::assemble::csr::CsrMatrix;
use crate::error::{GfemError, Result, format_sample_list};
use crate::mesh::Mesh;

/// エラーに並べる最大件数。
const SAMPLE_LIMIT: usize = 20;

/// メッシュから CSR の非ゼロ構造を作る（値は 0）。
///
/// どの要素からも参照されていない節点があるとその行が空になり、剛性行列の零空間が
/// span{1} より広がってしまう（＝射影 CG の前提が崩れる）ので、ここでエラーにする。
/// 読み込み時（SPEC §3.1）には警告に留めているのは、可視化だけしたい場合があるため。
pub fn build_sparsity(mesh: &Mesh) -> Result<CsrMatrix> {
    let n = mesh.nodes.len();
    if n == 0 {
        return Err(GfemError::model("節点がありません"));
    }

    // 1. 各節点に撒かれる候補の個数を数える
    let mut offsets = vec![0usize; n + 1];
    for e in mesh.iter_solids() {
        let nn = e.conn.len();
        for &node in e.conn {
            offsets[node as usize + 1] += nn;
        }
    }

    // 孤立節点の検出
    let orphans: Vec<String> = (0..n)
        .filter(|&i| offsets[i + 1] == 0)
        .map(|i| mesh.nodes.id(i as u32).to_string())
        .collect();
    if !orphans.is_empty() {
        return Err(GfemError::model(format!(
            "どの要素からも参照されていない節点があります（{} 件）: ID={}",
            orphans.len(),
            format_sample_list(&orphans, SAMPLE_LIMIT)
        ))
        .with_hint(
            "孤立節点の電位は方程式から決まらないため、解が一意になりません。\
             メッシュから取り除いてから解いてください",
        ));
    }

    for i in 0..n {
        offsets[i + 1] += offsets[i];
    }
    let total = offsets[n];

    // 2. 候補を撒く
    let mut scratch = vec![0u32; total];
    let mut cursor = offsets.clone();
    for e in mesh.iter_solids() {
        let nn = e.conn.len();
        for &node in e.conn {
            let c = &mut cursor[node as usize];
            scratch[*c..*c + nn].copy_from_slice(e.conn);
            *c += nn;
        }
    }

    // 3. 行ごとにソートして重複を詰める。行ごとに独立で、各行の結果は中身だけで
    //    決まる（整数のソート）ので、並列にしても逐次と同じ構造になる
    use crate::solver::amg::rowwise::split_rows;
    use rayon::prelude::*;
    let mut rows = split_rows(&mut scratch, &offsets);
    let lengths: Vec<usize> = rows
        .par_iter_mut()
        .map(|row| {
            row.sort_unstable();
            dedup_sorted(row)
        })
        .collect();

    // 4. 詰めた行を 1 本の列番号配列にまとめる
    let mut row_ptr = Vec::with_capacity(n + 1);
    row_ptr.push(0usize);
    for &len in &lengths {
        row_ptr.push(row_ptr.last().copied().unwrap_or(0) + len);
    }
    let mut col_idx = vec![0u32; row_ptr[n]];
    split_rows(&mut col_idx, &row_ptr)
        .into_par_iter()
        .zip(rows.into_par_iter())
        .for_each(|(out, row)| out.copy_from_slice(&row[..out.len()]));

    Ok(CsrMatrix::from_structure(n, row_ptr, col_idx))
}

/// 昇順に並んだ行から重複を取り除き、残った列を行の先頭に詰める。戻り値は残った個数。
fn dedup_sorted(row: &mut [u32]) -> usize {
    let mut kept = 0;
    for k in 0..row.len() {
        if kept == 0 || row[kept - 1] != row[k] {
            row[kept] = row[k];
            kept += 1;
        }
    }
    kept
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
