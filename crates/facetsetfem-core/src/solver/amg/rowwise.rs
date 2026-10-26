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

/// 1 タスクが受け持つ行数。
///
/// 以前は行をスレッド数ちょうどの塊に切っていたが、それだと重い行が偏った塊の終わりを
/// 全員が待つ。小さめの塊にして rayon に配らせる（作業を盗み合うので偏りがならされる）。
pub const ROWS_PER_TASK: usize = 2_048;

/// 行ごとの処理。逐次でも並列でも結果は同じ。
pub fn for_each_row<T: Send>(out: &mut [T], body: impl Fn(usize, &mut T) + Sync + Send) {
    for_each_row_init(out, || (), |_, i, slot| body(i, slot));
}

/// [`for_each_row`] の、タスクごとの作業領域つき版（疎行列積の作業配列など）。
///
/// `init` はタスクを受け持つたびに呼ばれうる（rayon が作業を分けた数だけ）。
pub fn for_each_row_init<T: Send, S>(
    out: &mut [T],
    init: impl Fn() -> S + Sync + Send,
    body: impl Fn(&mut S, usize, &mut T) + Sync + Send,
) {
    if out.len() < PARALLEL_MIN_ROWS {
        let mut state = init();
        for (i, slot) in out.iter_mut().enumerate() {
            body(&mut state, i, slot);
        }
        return;
    }
    use rayon::prelude::*;
    out.par_chunks_mut(ROWS_PER_TASK)
        .enumerate()
        .for_each_init(init, |state, (b, part)| {
            let base = b * ROWS_PER_TASK;
            for (k, slot) in part.iter_mut().enumerate() {
                body(state, base + k, slot);
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

/// 行の塊。`rows` の行 `i` の列番号・値は、`cols` / `vals` の中の
/// `row_ptr[i] - row_ptr[rows.start]` から始まる。
struct Block<'a, C> {
    rows: std::ops::Range<usize>,
    cols: C,
    vals: &'a mut [f64],
}

/// 列番号と値を、行の塊ごとの区間に切り分ける。値が空（構造だけを作るとき）なら値は切らない。
fn blocks<'a, C: Splittable<'a>>(
    row_ptr: &[usize],
    mut cols: C,
    mut vals: &'a mut [f64],
) -> Vec<Block<'a, C>> {
    let n = row_ptr.len() - 1;
    let has_values = !vals.is_empty();
    let mut out = Vec::with_capacity(n.div_ceil(ROWS_PER_TASK));
    let mut start = 0;
    while start < n {
        let end = (start + ROWS_PER_TASK).min(n);
        let len = row_ptr[end] - row_ptr[start];
        let (ch, ct) = cols.split(len);
        let (vh, vt) = std::mem::take(&mut vals).split_at_mut(if has_values { len } else { 0 });
        out.push(Block {
            rows: start..end,
            cols: ch,
            vals: vh,
        });
        cols = ct;
        vals = vt;
        start = end;
    }
    out
}

/// 列番号の区間を切り分けられるもの（書く `&mut [u32]` と読むだけの `&[u32]`）。
trait Splittable<'a>: Sized + Send {
    fn split(self, at: usize) -> (Self, Self);
}

impl<'a> Splittable<'a> for &'a mut [u32] {
    fn split(self, at: usize) -> (Self, Self) {
        self.split_at_mut(at)
    }
}

impl<'a> Splittable<'a> for &'a [u32] {
    fn split(self, at: usize) -> (Self, Self) {
        self.split_at(at)
    }
}

/// CSR の各行に、その行の列番号と値の区間を渡して埋めさせる（行ごとに独立な処理）。
///
/// 以前は全行ぶんの切れ端 `(&mut [u32], &mut [f64])` を先に逐次で作っていた（1000 万行で
/// 320 MB）。いまは行の塊（[`ROWS_PER_TASK`] 行）ごとに切り分け、塊の中では `row_ptr` から
/// その場で区間を求める。どの行も自分の区間しか触らないので、結果は逐次と同じ。
/// 構造だけを作るときは `values` に空を渡す（行には空の値の区間が渡る）。
pub fn fill_rows_init<S>(
    row_ptr: &[usize],
    col_idx: &mut [u32],
    values: &mut [f64],
    init: impl Fn() -> S + Sync + Send,
    fill: impl Fn(&mut S, usize, &mut [u32], &mut [f64]) + Sync + Send,
) {
    let run = |state: &mut S, block: Block<'_, &mut [u32]>| {
        let base = row_ptr[block.rows.start];
        let has_values = !block.vals.is_empty();
        for i in block.rows {
            let (s, e) = (row_ptr[i] - base, row_ptr[i + 1] - base);
            let vals: &mut [f64] = if has_values {
                &mut block.vals[s..e]
            } else {
                &mut []
            };
            fill(state, i, &mut block.cols[s..e], vals);
        }
    };
    run_blocks(row_ptr, blocks(row_ptr, col_idx, values), init, run);
}

/// [`fill_rows_init`] の作業領域なし版。
pub fn fill_rows(
    row_ptr: &[usize],
    col_idx: &mut [u32],
    values: &mut [f64],
    fill: impl Fn(usize, &mut [u32], &mut [f64]) + Sync + Send,
) {
    fill_rows_init(row_ptr, col_idx, values, || (), |_, i, c, v| fill(i, c, v));
}

/// 構造（列番号）は読むだけで、値だけを行ごとに書き換える。
pub fn update_values(
    row_ptr: &[usize],
    col_idx: &[u32],
    values: &mut [f64],
    update: impl Fn(usize, &[u32], &mut [f64]) + Sync + Send,
) {
    let run = |_: &mut (), block: Block<'_, &[u32]>| {
        let base = row_ptr[block.rows.start];
        for i in block.rows {
            let (s, e) = (row_ptr[i] - base, row_ptr[i + 1] - base);
            update(i, &block.cols[s..e], &mut block.vals[s..e]);
        }
    };
    run_blocks(row_ptr, blocks(row_ptr, col_idx, values), || (), run);
}

/// 塊を逐次か並列で処理する。
fn run_blocks<B: Send, S>(
    row_ptr: &[usize],
    blocks: Vec<B>,
    init: impl Fn() -> S + Sync + Send,
    run: impl Fn(&mut S, B) + Sync + Send,
) {
    if row_ptr.len() - 1 < PARALLEL_MIN_ROWS {
        let mut state = init();
        for block in blocks {
            run(&mut state, block);
        }
        return;
    }
    use rayon::prelude::*;
    blocks.into_par_iter().for_each_init(init, run);
}

/// 行ごとの非ゼロ数から行ポインタを作る。
pub fn row_pointers(counts: &[usize]) -> Vec<usize> {
    let mut row_ptr = Vec::with_capacity(counts.len() + 1);
    row_ptr.push(0usize);
    let mut total = 0usize;
    for &c in counts {
        total += c;
        row_ptr.push(total);
    }
    row_ptr
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
    let row_ptr = row_pointers(&counts);
    let mut col_idx = vec![0u32; row_ptr[n_rows]];
    fill_rows(&row_ptr, &mut col_idx, &mut [], |i, cols, _| fill(i, cols));
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
    let row_ptr = row_pointers(&counts);
    let nnz = row_ptr[n_rows];
    // ゼロで確保するので、ページは書き込むときに（並列に）初めて触る
    let mut col_idx = vec![0u32; nnz];
    let mut values = vec![0.0f64; nnz];
    fill_rows(&row_ptr, &mut col_idx, &mut values, fill);
    crate::assemble::csr::CsrMatrix::from_parts(n_rows, n_cols, row_ptr, col_idx, values)
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

    /// 行 i の長さ（空の行や、塊をまたいで長さが変わる行を混ぜる）。
    fn ragged(i: usize) -> usize {
        (i * 7 + i / 5) % 6
    }

    #[test]
    fn blocks_hand_every_row_its_own_slice() {
        // 並列に切り替わる大きさで、塊（ROWS_PER_TASK 行）の境目をまたぐ。1 行ずつ逐次に
        // 作ったものと CSR 全体が一致し、構造だけの組み立て・値だけの書き換えも同じ区間を渡す
        let n = PARALLEL_MIN_ROWS * 2 + ROWS_PER_TASK / 3 + 5;
        let fill = |i: usize, cols: &mut [u32], vals: &mut [f64]| {
            for (k, c) in cols.iter_mut().enumerate() {
                *c = (i + k) as u32;
            }
            for (k, v) in vals.iter_mut().enumerate() {
                *v = (i * 10 + k) as f64 + 0.5;
            }
        };
        let m = build(n, n + 8, ragged, fill);

        let mut row_ptr = vec![0usize];
        let (mut cols, mut vals) = (Vec::new(), Vec::new());
        for i in 0..n {
            let (mut c, mut v) = (vec![0u32; ragged(i)], vec![0.0; ragged(i)]);
            fill(i, &mut c, &mut v);
            cols.extend(c);
            vals.extend(v);
            row_ptr.push(cols.len());
        }
        assert_eq!(m.row_ptr(), &row_ptr[..]);
        assert_eq!(m.col_idx(), &cols[..]);
        assert_eq!(m.values(), &vals[..]);

        let (sp, sc) = build_structure(n, ragged, |i, c| fill(i, c, &mut []));
        assert_eq!(sp, row_ptr);
        assert_eq!(sc, cols);

        // 値だけの書き換え: 各行は自分の列番号を読み、自分の値だけを書く
        let mut values = vals.clone();
        update_values(&row_ptr, &cols, &mut values, |i, c, v| {
            for (x, &col) in v.iter_mut().zip(c) {
                *x = -(*x) + (col as usize - i) as f64;
            }
        });
        for i in 0..n {
            for k in row_ptr[i]..row_ptr[i + 1] {
                assert_eq!(values[k], -vals[k] + (k - row_ptr[i]) as f64);
            }
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
