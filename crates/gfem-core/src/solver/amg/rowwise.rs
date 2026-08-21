//! 行ごとに独立な CSR 構築（SPEC §4.5）。
//!
//! AMG の階層構築は、凝集を除けば**行ごとに完全に独立**な処理の集まり
//! （強連結グラフ・フィルタ・平滑化行列・疎行列積）。行の担当をスレッドに
//! 配っても加算順序は 1 つも変わらないので、**逐次と並列がビット単位で一致する**
//! （CLAUDE.md 設計原則 1）。
//!
//! 実測では 150 万節点の階層構築が 1.4 秒かかり、GPU の V サイクルより重かった。
//! ここを並列化しないと「反復は 6.8 倍速いのに全体は 1.5 倍」で終わってしまう。
//!
//! 構築は 2 パス。行の長さを数えてから場所を確定し、値を埋める。

/// 行を rayon に配りはじめる行数。これより小さいとタスク投入のほうが高くつく。
pub const PARALLEL_MIN_ROWS: usize = 4_096;

/// 行ごとの処理。逐次でも並列でも結果は同じ。
pub fn for_each_row<T: Send>(out: &mut [T], body: impl Fn(usize, &mut T) + Sync + Send) {
    if out.len() < PARALLEL_MIN_ROWS {
        for (i, slot) in out.iter_mut().enumerate() {
            body(i, slot);
        }
        return;
    }
    use rayon::prelude::*;
    let chunk = out
        .len()
        .div_ceil(rayon::current_num_threads().max(1))
        .max(1);
    out.par_chunks_mut(chunk).enumerate().for_each(|(b, part)| {
        let base = b * chunk;
        for (k, slot) in part.iter_mut().enumerate() {
            body(base + k, slot);
        }
    });
}

/// 可変スライスを `row_ptr` の区切りで行ごとに分ける。
///
/// `par_chunks_mut` は等分割しかできないので、長さの違う行に配るための下ごしらえ。
/// 分けるところは逐次だが O(行数) で、中身の計算に比べれば無視できる。
pub fn split_rows<'a, T>(data: &'a mut [T], row_ptr: &[usize]) -> Vec<&'a mut [T]> {
    let mut out = Vec::with_capacity(row_ptr.len().saturating_sub(1));
    let mut rest = data;
    for w in row_ptr.windows(2) {
        let (head, tail) = rest.split_at_mut(w[1] - w[0]);
        out.push(head);
        rest = tail;
    }
    out
}

/// [`split_rows`] の共有参照版。
pub fn split_rows_ref<'a, T>(data: &'a [T], row_ptr: &[usize]) -> Vec<&'a [T]> {
    row_ptr.windows(2).map(|w| &data[w[0]..w[1]]).collect()
}

/// 行ごとに独立な CSR 構造の構築（値は持たない）。
///
/// `count(i)` が行 `i` の非ゼロ数、`fill(i, cols)` がその行の列番号を**昇順で**書く。
pub fn build_structure(
    n_rows: usize,
    count: impl Fn(usize) -> usize + Sync + Send,
    fill: impl Fn(usize, &mut [u32]) + Sync + Send,
) -> (Vec<usize>, Vec<u32>) {
    let mut counts = vec![0usize; n_rows];
    for_each_row(&mut counts, |i, slot| *slot = count(i));

    let mut row_ptr = vec![0usize; n_rows + 1];
    for i in 0..n_rows {
        row_ptr[i + 1] = row_ptr[i] + counts[i];
    }

    let mut col_idx = vec![0u32; row_ptr[n_rows]];
    let mut rows = split_rows(&mut col_idx, &row_ptr);
    for_each_row(&mut rows, |i, cols| fill(i, cols));
    drop(rows);
    (row_ptr, col_idx)
}

/// 行ごとに独立な CSR の構築（列番号と値）。
pub fn build(
    n_rows: usize,
    n_cols: usize,
    count: impl Fn(usize) -> usize + Sync + Send,
    fill: impl Fn(usize, &mut [u32], &mut [f64]) + Sync + Send,
) -> crate::assemble::csr::CsrMatrix {
    let mut counts = vec![0usize; n_rows];
    for_each_row(&mut counts, |i, slot| *slot = count(i));

    let mut row_ptr = vec![0usize; n_rows + 1];
    for i in 0..n_rows {
        row_ptr[i + 1] = row_ptr[i] + counts[i];
    }
    let nnz = row_ptr[n_rows];

    let mut col_idx = vec![0u32; nnz];
    let mut values = vec![0.0f64; nnz];
    {
        let cols = split_rows(&mut col_idx, &row_ptr);
        let vals = split_rows(&mut values, &row_ptr);
        let mut rows: Vec<(&mut [u32], &mut [f64])> = cols.into_iter().zip(vals).collect();
        for_each_row(&mut rows, |i, row| {
            let (cols, vals) = row;
            fill(i, cols, vals);
        });
    }

    let mut m =
        crate::assemble::csr::CsrMatrix::from_structure_rect(n_rows, n_cols, row_ptr, col_idx);
    m.values_mut().copy_from_slice(&values);
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_variable_length_rows() {
        let mut data = [0u32; 6];
        let row_ptr = vec![0, 1, 1, 4, 6];
        let rows = split_rows(&mut data, &row_ptr);
        assert_eq!(rows.len(), 4);
        assert_eq!(
            rows.iter().map(|r| r.len()).collect::<Vec<_>>(),
            [1, 0, 3, 2]
        );
    }

    #[test]
    fn parallel_and_serial_build_the_same_matrix() {
        // 行ごとに独立なので、担当の配り方で結果が変わってはいけない
        let build_with = |n: usize| {
            build(
                n,
                n,
                |i| if i % 3 == 0 { 2 } else { 1 },
                |i, cols, vals| {
                    cols[0] = i as u32;
                    vals[0] = (i as f64).sin();
                    if cols.len() > 1 {
                        cols[1] = ((i + 1) % 100) as u32;
                        vals[1] = (i as f64).cos();
                    }
                },
            )
        };
        // 閾値をまたぐ 2 つの大きさで、同じ規則から作った行を突き合わせる
        let small = build_with(PARALLEL_MIN_ROWS - 1);
        let large = build_with(PARALLEL_MIN_ROWS * 2);
        for i in 0..small.n_rows() {
            assert_eq!(small.row(i).0, large.row(i).0, "行 {i} の列");
            assert_eq!(small.row(i).1, large.row(i).1, "行 {i} の値");
        }
    }

    #[test]
    fn empty_rows_are_allowed() {
        let m = build(
            3,
            3,
            |i| usize::from(i == 1),
            |_, cols, vals| {
                if let (Some(c), Some(v)) = (cols.first_mut(), vals.first_mut()) {
                    *c = 2;
                    *v = 7.0;
                }
            },
        );
        assert_eq!(m.nnz(), 1);
        assert_eq!(m.row(0).0.len(), 0);
        assert_eq!(m.row(1).0, &[2]);
    }
}
