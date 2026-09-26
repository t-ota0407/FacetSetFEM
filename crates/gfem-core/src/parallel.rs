//! 大規模データを並列に、かつ決定的に扱うための部品（SPEC §8 の「2026-09」）。
//!
//! - 振り分け（計数ソート）＋組ごとの並べ替え（[`bucket_sort`]）。外表面の抽出（面 2.4 億件）と
//!   節点 → 要素の表（参照 2.4 億件）で使う
//! - 大きな列の解放を別スレッドに任せる（[`drop_in_background`]）

use rayon::prelude::*;

/// [`bucket_sort`] で振り分ける鍵を持つ項目。
///
/// 鍵は**項目の順序の最上位**でなければならない（`a.bucket_key() < b.bucket_key()` なら
/// `a < b`）。組の順に並べたものが全体の順になるのは、これが成り立つから。
pub(crate) trait BucketKey {
    /// 振り分けの鍵（`n_keys` 未満）。
    fn bucket_key(&self) -> u32;
}

/// 節点 → 要素の表の `(節点, 要素)`。
impl BucketKey for (u32, u32) {
    fn bucket_key(&self) -> u32 {
        self.0
    }
}

/// `n` 件の項目を昇順に並べた列を作る。
///
/// `entry(k)` は k 番目（0 ≤ k < `n`）の項目、`key(k)` はその鍵（`entry(k).bucket_key()` と
/// 同じもの。項目を作るより安く求まるなら、枚数を数えるときにそちらを使う）、`n_keys` は鍵の
/// 上限。比較ソートの代わりに、鍵で粗く振り分けてから（計数ソート）、小さな組ごとに並べる:
///
/// 1. 項目の通し番号を区間に分け、区間ごとに、組ごとの枚数を数える
/// 2. 出力を「組の順、組の中は区間の順」に切り分け、区間ごとに自分の切れ端を受け取る
/// 3. 区間ごとに、自分の切れ端へ項目を書き込む（区間どうしで書く場所は重ならない）
/// 4. 組ごとに並べる（1 組は数千件なのでキャッシュに載る）
///
/// 鍵は順序の最上位なので、組の順に並べれば全体も昇順になる。等しい項目どうしは区別
/// できないので、並べ終わった列は入力だけで決まり、全体を比較ソートしたものと 1 件も
/// ずれない（CLAUDE.md 設計原則 1）。`unsafe` も原子操作も使わない。
///
/// 1000 万節点の面 2.4 億件で、比較ソートの 2.8 秒が 1.6 秒（列挙を含む）になった。
pub(crate) fn bucket_sort<T, K, E>(n: usize, n_keys: usize, key: K, entry: E) -> Vec<T>
where
    T: BucketKey + Copy + Ord + Default + Send + Sync,
    K: Fn(usize) -> u32 + Sync,
    E: Fn(usize) -> T + Sync,
{
    bucket_sort_with(n, n_keys, ITEMS_PER_BUCKET, key, entry)
}

/// [`bucket_sort`] の本体。1 組あたりの枚数の目安 `per_bucket` を外から変えられる
/// （テストで組を極端に小さくして、振り分けの経路を確かめる）。
pub(crate) fn bucket_sort_with<T, K, E>(
    n: usize,
    n_keys: usize,
    per_bucket: usize,
    key: K,
    entry: E,
) -> Vec<T>
where
    T: BucketKey + Copy + Ord + Default + Send + Sync,
    K: Fn(usize) -> u32 + Sync,
    E: Fn(usize) -> T + Sync,
{
    let mut scattered = bucket_scatter(n, n_keys, per_bucket, key, entry);
    // 4. 組ごとに並べる
    scattered
        .buckets_mut()
        .into_par_iter()
        .for_each(|g| g.sort_unstable());
    scattered.items
}

/// 振り分けた列（[`bucket_scatter`] の結果）。
pub(crate) struct Scattered<T> {
    /// 組の順に並んだ項目。組の中は**項目の通し番号の順**（安定）
    pub items: Vec<T>,
    /// 組ごとの件数
    pub sizes: Vec<usize>,
    /// 組 b には鍵が `[b << shift, (b + 1) << shift)` の項目が入る
    pub shift: u32,
}

impl<T> Scattered<T> {
    /// 組ごとの可変スライス。
    pub fn buckets_mut(&mut self) -> Vec<&mut [T]> {
        let mut out = Vec::with_capacity(self.sizes.len());
        let mut rest: &mut [T] = &mut self.items;
        for &size in &self.sizes {
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(size);
            out.push(head);
            rest = tail;
        }
        out
    }
}

/// [`bucket_sort`] の手順 1〜3（振り分けだけ）。組の中は並べず、項目の通し番号の順のまま。
///
/// 組の中をどう仕上げるかを呼び出し側で選べる。節点 → 要素の表は、組の中を節点ごとの
/// 計数ソートで仕上げる（組の中がすでに要素の順に並んでいるので、比較ソートが要らない）。
pub(crate) fn bucket_scatter<T, K, E>(
    n: usize,
    n_keys: usize,
    per_bucket: usize,
    key: K,
    entry: E,
) -> Scattered<T>
where
    T: BucketKey + Copy + Default + Send + Sync,
    K: Fn(usize) -> u32 + Sync,
    E: Fn(usize) -> T + Sync,
{
    // 1 組が数千件になるよう、鍵を 2 の冪で粗くしたものを組にする
    let target = (n / per_bucket.max(1)).clamp(1, MAX_BUCKETS);
    let mut shift = 0u32;
    while (n_keys >> shift) > target {
        shift += 1;
    }
    if n == 0 {
        return Scattered {
            items: Vec::new(),
            sizes: Vec::new(),
            shift,
        };
    }
    let n_buckets = (n_keys.saturating_sub(1) >> shift) + 1;

    let n_parts = (rayon::current_num_threads() * 4).clamp(1, n);
    let part = |p: usize| (n * p / n_parts)..(n * (p + 1) / n_parts);

    // 1. 区間ごとの、組ごとの枚数
    let counts: Vec<Vec<u32>> = (0..n_parts)
        .into_par_iter()
        .map(|p| {
            let mut count = vec![0u32; n_buckets];
            for k in part(p) {
                count[key(k) as usize >> shift] += 1;
            }
            count
        })
        .collect();

    // 2. 出力を切り分ける。既定値（ゼロ）で確保するので、OS からゼロ済みのページが遅延で
    //    渡り、実際に触るのは 3 の並列な書き込みのとき（ゼロ以外で埋めると逐次に 1 秒かかる）
    let mut items: Vec<T> = vec![T::default(); n];
    let mut pieces: Vec<Vec<&mut [T]>> = (0..n_parts)
        .map(|_| Vec::with_capacity(n_buckets))
        .collect();
    let mut sizes: Vec<usize> = Vec::with_capacity(n_buckets);
    let mut rest: &mut [T] = &mut items;
    for b in 0..n_buckets {
        let mut size = 0;
        for (p, count) in counts.iter().enumerate() {
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(count[b] as usize);
            pieces[p].push(head);
            rest = tail;
            size += count[b] as usize;
        }
        sizes.push(size);
    }

    // 3. 区間ごとに、自分の切れ端へ書き込む。区間は通し番号の順に並び、区間の中も
    //    通し番号の順に書くので、組の中は通し番号の順になる
    pieces
        .into_par_iter()
        .enumerate()
        .for_each(|(p, mut mine)| {
            let mut cursor = vec![0usize; n_buckets];
            for k in part(p) {
                let e = entry(k);
                let b = e.bucket_key() as usize >> shift;
                mine[b][cursor[b]] = e;
                cursor[b] += 1;
            }
        });

    Scattered {
        items,
        sizes,
        shift,
    }
}

/// 振り分けの 1 組あたりの枚数の目安。組がキャッシュに載る大きさにする。
///
/// 1,000 件ほどに小さくすると、書き込み先が散らばって全体ではかえって遅くなった。
pub(crate) const ITEMS_PER_BUCKET: usize = 4096;

/// 振り分けの組の数の上限（区間ごとに組の数だけ数える欄を持つので、増やしすぎない）。
const MAX_BUCKETS: usize = 1 << 16;

/// 大きな列の解放を別のスレッドに任せる。
///
/// 1000 万節点の面の列（2.4 億件、5.7 GB）は、解放するだけで 0.5 秒かかる（OS にページを
/// 返す）。解放がいつ終わっても結果には関係しないので、呼び出し側を待たせない。
/// 小さな列はスレッドを立てるほうが高くつくので、その場で解放する。スレッドを立てられ
/// なかったときも、渡した列はその場で解放される。
pub(crate) fn drop_in_background<T: Send + 'static>(v: Vec<T>) {
    if v.capacity() * std::mem::size_of::<T>() < BACKGROUND_DROP_BYTES {
        return;
    }
    let _ = std::thread::Builder::new()
        .name("gfem-drop".into())
        .spawn(move || drop(v));
}

/// これより大きな列の解放は別のスレッドに任せる（[`drop_in_background`]）。
const BACKGROUND_DROP_BYTES: usize = 64 << 20;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_sort_gives_exactly_the_full_sort() {
        // 鍵（節点）が散らばり、同じ鍵に複数の要素がつく (節点, 要素) の列。組を極端に
        // 小さくした場合から組が 1 つの場合まで、全体を比較ソートしたものと 1 件もずれない
        let n_keys = 3_000usize;
        let entry = |k: usize| -> (u32, u32) {
            let x = (k as u64).wrapping_mul(0x9e3779b97f4a7c15) >> 17;
            ((x % n_keys as u64) as u32, (k / 3) as u32)
        };
        let n = 50_000;
        let mut expected: Vec<(u32, u32)> = (0..n).map(entry).collect();
        expected.sort_unstable();
        for per_bucket in [1, 5, 256, ITEMS_PER_BUCKET, usize::MAX] {
            let got = bucket_sort_with(n, n_keys, per_bucket, |k| entry(k).0, entry);
            assert!(got == expected, "組の目安 {per_bucket} 件でずれた");
        }
        assert!(bucket_sort(0, n_keys, |k| entry(k).0, entry).is_empty());
    }
}
