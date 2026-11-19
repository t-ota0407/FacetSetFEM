//! 強連結グラフと決定的な凝集（SPEC §4.5）。
//!
//! 平滑化凝集（smoothed aggregation）の粗化は、節点を小さな塊（凝集）に
//! まとめることで行う。塊 1 つが粗レベルの 1 節点になる。
//!
//! # 順序を固定する
//!
//! 凝集は本質的に貪欲法で、**節点をどの順に見るかで結果が変わる**。
//! 並列マルチグリッドの実装では乱数重み付きの極大独立集合を使うことが多いが、
//! それでは実行のたびに階層が変わり、CLAUDE.md 設計原則 1（ビット単位の再現性）が
//! 壊れる。ここでは**節点番号の昇順に逐次で**貪欲に進める。
//!
//! 凝集は O(nnz) で、CG 本体に比べれば軽い。並列化して再現性を失う取引はしない。

use crate::assemble::csr::CsrMatrix;
use crate::solver::amg::rowwise;

/// 凝集の結果。
#[derive(Debug, Clone)]
pub struct Aggregation {
    /// 節点 → 凝集番号
    pub of_node: Vec<u32>,
    /// 凝集の個数（＝粗レベルの節点数）
    pub count: usize,
}

/// 強連結だけを残した近傍グラフ（対角は含まない）。
#[derive(Debug, Clone)]
pub struct Strength {
    /// 長さ `n + 1`
    pub row_ptr: Vec<usize>,
    /// 行ごとに昇順
    pub col_idx: Vec<u32>,
}

impl Strength {
    /// 節点 `i` の強連結近傍（昇順）。
    #[inline]
    pub fn neighbors(&self, i: usize) -> &[u32] {
        &self.col_idx[self.row_ptr[i]..self.row_ptr[i + 1]]
    }
}

/// 強連結の判定（SPEC §4.5）。
///
/// ```text
/// j が i に強く連結  ⟺  |A_ij| ≥ θ √(|A_ii| |A_jj|)
/// ```
///
/// A は対称なのでこの判定も対称になり、得られるグラフも対称。
/// θ は平滑化凝集の標準的な 0.08 前後を使う（古典 AMG の 0.25 より小さい）。
pub fn strength_graph(a: &CsrMatrix, theta: f64) -> Strength {
    let n = a.n_rows();
    let diag = a.diagonal();
    // 行ごとに独立なので並列に作ってよい（[`rowwise`] の説明を参照）
    let is_strong = |i: usize, j: u32, v: f64| {
        j as usize != i && v.abs() >= theta * (diag[i] * diag[j as usize]).abs().sqrt()
    };
    let (row_ptr, col_idx) = rowwise::build_structure(
        n,
        |i| {
            let (cols, vals) = a.row(i);
            cols.iter()
                .zip(vals)
                .filter(|&(&j, &v)| is_strong(i, j, v))
                .count()
        },
        |i, out| {
            let (cols, vals) = a.row(i);
            let mut k = 0;
            for (&j, &v) in cols.iter().zip(vals) {
                if is_strong(i, j, v) {
                    out[k] = j;
                    k += 1;
                }
            }
        },
    );
    Strength { row_ptr, col_idx }
}

/// 割り当て前を表す番兵。
const UNASSIGNED: u32 = u32::MAX;

/// 標準的な 3 パスの凝集（Vaněk）。節点番号の昇順に進めるので決定的。
pub fn aggregate(strength: &Strength, n: usize) -> Aggregation {
    let mut of_node = vec![UNASSIGNED; n];
    let mut count = 0u32;

    // パス 1: 「自分も近傍も全て未割当」な節点を種にして、近傍ごと 1 つの凝集にする。
    // これが凝集の骨格になる
    for i in 0..n {
        if of_node[i] != UNASSIGNED {
            continue;
        }
        let nb = strength.neighbors(i);
        if nb.is_empty() {
            continue; // 孤立点はパス 3 で拾う
        }
        if nb.iter().any(|&j| of_node[j as usize] != UNASSIGNED) {
            continue;
        }
        of_node[i] = count;
        for &j in nb {
            of_node[j as usize] = count;
        }
        count += 1;
    }

    // パス 2: 残った節点を、強連結先の既存の凝集に入れる。
    // このパスで作った割り当てが同じパスの後続の判断に影響しないよう、
    // 読み取りはパス 1 直後のスナップショットに対して行う（順序依存を減らす）
    let snapshot = of_node.clone();
    for (i, slot) in of_node.iter_mut().enumerate() {
        if *slot != UNASSIGNED {
            continue;
        }
        for &j in strength.neighbors(i) {
            if snapshot[j as usize] != UNASSIGNED {
                *slot = snapshot[j as usize];
                break; // 昇順の最初の 1 つ＝決定的
            }
        }
    }

    // パス 3: どの凝集にも触れなかった節点（孤立点や弱連結だけの塊）で新しい凝集を作る
    for i in 0..n {
        if of_node[i] != UNASSIGNED {
            continue;
        }
        of_node[i] = count;
        for &j in strength.neighbors(i) {
            if of_node[j as usize] == UNASSIGNED {
                of_node[j as usize] = count;
            }
        }
        count += 1;
    }

    debug_assert!(of_node.iter().all(|&a| a != UNASSIGNED));
    Aggregation {
        of_node,
        count: count as usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::laplacian_1d;

    #[test]
    fn every_node_lands_in_exactly_one_aggregate() {
        let a = laplacian_1d(100);
        let s = strength_graph(&a, 0.08);
        let agg = aggregate(&s, 100);
        assert!(agg.of_node.iter().all(|&x| (x as usize) < agg.count));
        let mut sizes = vec![0usize; agg.count];
        for &x in &agg.of_node {
            sizes[x as usize] += 1;
        }
        assert!(sizes.iter().all(|&s| s > 0), "空の凝集がある");
        assert_eq!(sizes.iter().sum::<usize>(), 100);
    }

    #[test]
    fn one_dimensional_grid_coarsens_by_about_three() {
        // 1 次元では「自分 + 左右」で 3 節点の凝集ができる
        let a = laplacian_1d(300);
        let s = strength_graph(&a, 0.08);
        let agg = aggregate(&s, 300);
        assert!(
            (90..=110).contains(&agg.count),
            "凝集数 {} は 100 前後のはず",
            agg.count
        );
    }

    #[test]
    fn aggregation_is_reproducible() {
        // CLAUDE.md 設計原則 1: 同じ入力から常に同じ階層
        let a = laplacian_1d(1000);
        let s = strength_graph(&a, 0.08);
        let first = aggregate(&s, 1000);
        for _ in 0..5 {
            let again = aggregate(&s, 1000);
            assert_eq!(first.of_node, again.of_node);
            assert_eq!(first.count, again.count);
        }
    }

    #[test]
    fn aggregates_are_connected_in_the_strength_graph() {
        let a = laplacian_1d(200);
        let s = strength_graph(&a, 0.08);
        let agg = aggregate(&s, 200);
        // 各凝集について、強連結グラフ上で連結であることを幅優先で確かめる
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); agg.count];
        for (i, &g) in agg.of_node.iter().enumerate() {
            members[g as usize].push(i);
        }
        for (g, group) in members.iter().enumerate() {
            let mut seen = vec![false; group.len()];
            let index: std::collections::HashMap<usize, usize> =
                group.iter().enumerate().map(|(k, &i)| (i, k)).collect();
            let mut stack = vec![0usize];
            seen[0] = true;
            while let Some(k) = stack.pop() {
                for &j in s.neighbors(group[k]) {
                    if let Some(&kj) = index.get(&(j as usize))
                        && !seen[kj]
                    {
                        seen[kj] = true;
                        stack.push(kj);
                    }
                }
            }
            assert!(
                seen.iter().all(|&v| v),
                "凝集 {g} が強連結グラフ上で分断している"
            );
        }
    }

    #[test]
    fn a_weak_connection_is_dropped() {
        // 対角 1、非対角 0.001 → θ = 0.08 では弱い
        let row_ptr = vec![0, 2, 4];
        let col_idx = vec![0, 1, 0, 1];
        let mut m = CsrMatrix::from_structure(2, row_ptr, col_idx);
        m.values_mut().copy_from_slice(&[1.0, -0.001, -0.001, 1.0]);
        let s = strength_graph(&m, 0.08);
        assert!(s.neighbors(0).is_empty());
        // 強連結が無いので、それぞれが独立した凝集になる
        let agg = aggregate(&s, 2);
        assert_eq!(agg.count, 2);
    }
}
