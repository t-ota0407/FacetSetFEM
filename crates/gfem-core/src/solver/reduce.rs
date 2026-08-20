//! 決定的な総和（SPEC §4.4）。
//!
//! 並列内積は加算順序が非決定的になり、実行のたびに最下位ビットが変わる。
//! それでは「同一入力からビット単位で同じ結果」（CLAUDE.md 設計原則 1）が守れない。
//!
//! そこで**固定長ブロックに分けた階層リダクション**で足す:
//!
//! 1. 長さ [`BLOCK`] のブロックに切り、各ブロック内は逐次加算
//! 2. ブロック和の列に対して同じ規則を再帰的に適用
//!
//! 分割位置は配列長だけで決まるので、
//!
//! - CPU 逐次: そのまま計算する
//! - CPU 並列: ブロックを別スレッドに割り当てても、木の形が同じなので結果が変わらない
//! - GPU: 固定ブロック分割 + 階層リダクションで同じ木をたどる
//!
//! のいずれでもビット単位で一致する。
//!
//! 副次的な利点として、逐次の素朴な総和より丸め誤差が小さい
//! （誤差が O(n) ではなく O(log n) で増える）。

/// リダクションのブロック長。全バックエンドで共有する定数。
///
/// 変えると結果のビット列が変わるため、**変更は再現性の破壊にあたる**。
/// 値そのものに深い意味はないが、L1 キャッシュに収まり、GPU の典型的な
/// ブロックサイズの倍数でもある 512 にしてある。
pub const BLOCK: usize = 512;

/// ブロック内の逐次総和。
#[inline]
fn sum_block(v: &[f64]) -> f64 {
    let mut acc = 0.0;
    for &x in v {
        acc += x;
    }
    acc
}

/// 決定的な総和。
pub fn sum(v: &[f64]) -> f64 {
    if v.len() <= BLOCK {
        return sum_block(v);
    }
    let mut partial: Vec<f64> = Vec::with_capacity(v.len().div_ceil(BLOCK));
    for chunk in v.chunks(BLOCK) {
        partial.push(sum_block(chunk));
    }
    sum(&partial)
}

/// 決定的な内積。
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    if a.len() <= BLOCK {
        return dot_block(a, b);
    }
    let mut partial: Vec<f64> = Vec::with_capacity(a.len().div_ceil(BLOCK));
    for (ca, cb) in a.chunks(BLOCK).zip(b.chunks(BLOCK)) {
        partial.push(dot_block(ca, cb));
    }
    sum(&partial)
}

/// ブロック内の逐次内積。
#[inline]
fn dot_block(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        acc += x * y;
    }
    acc
}

/// 決定的な 2 ノルム。
pub fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// 決定的な平均。
pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    sum(v) / v.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_exact_sum_for_small_arrays() {
        let v: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        assert_eq!(sum(&v), 5050.0);
    }

    #[test]
    fn is_more_accurate_than_naive_summation() {
        // 1.0 に微小値を大量に足す古典的な例
        let mut v = vec![1.0];
        v.extend(std::iter::repeat_n(1.0e-16, 100_000));
        let naive = v.iter().fold(0.0f64, |a, b| a + b);
        let exact = 1.0 + 100_000.0 * 1.0e-16;
        assert!(
            (sum(&v) - exact).abs() <= (naive - exact).abs(),
            "ペアワイズ {} / 素朴 {} / 厳密 {}",
            sum(&v),
            naive,
            exact
        );
    }

    #[test]
    fn block_partition_is_reproducible() {
        // 「並列実行時にブロックを別々に計算しても同じ木になる」ことの確認。
        // 手でブロック和を作ってから畳んでも結果が一致する
        let v: Vec<f64> = (0..10_000).map(|i| (i as f64).sin()).collect();
        let manual: Vec<f64> = v.chunks(BLOCK).map(sum_block).collect();
        assert_eq!(sum(&v), sum(&manual));
    }

    #[test]
    fn dot_agrees_with_sum_of_products() {
        let a: Vec<f64> = (0..5000).map(|i| (i as f64).cos()).collect();
        let b: Vec<f64> = (0..5000).map(|i| (i as f64).sin()).collect();
        let products: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
        assert_eq!(dot(&a, &b), sum(&products));
    }

    #[test]
    fn repeated_calls_are_bitwise_identical() {
        let v: Vec<f64> = (0..12_345).map(|i| 1.0 / (i as f64 + 1.0)).collect();
        let first = sum(&v);
        for _ in 0..5 {
            assert_eq!(sum(&v), first);
        }
    }

    #[test]
    fn mean_of_constant_vector() {
        let v = vec![2.5; 3000];
        assert!((mean(&v) - 2.5).abs() < 1e-15);
        assert!((norm2(&[3.0, 4.0]) - 5.0).abs() < 1e-15);
    }

    #[test]
    fn empty_inputs() {
        assert_eq!(sum(&[]), 0.0);
        assert_eq!(mean(&[]), 0.0);
        assert_eq!(dot(&[], &[]), 0.0);
    }
}
