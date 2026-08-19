//! Nastran の節点 ID から内部インデックスへの写像。
//!
//! Nastran のバルクデータは**カードの出現順に依存しない**（要素が先、GRID が後でもよい）。
//! そのため読み込みでは一旦 ID のまま溜め、最後にここでまとめて索引へ解決する。
//!
//! 1000 万節点 × 平均 24 参照＝2 億回以上引くので、素直な木構造では遅い。
//! ID がほぼ連続している実メッシュ（1..N が大半）では直接引きの表を使い、
//! 疎な場合だけソート済み配列の二分探索に落とす。

use crate::mesh::NodeId;

/// 未定義を表す番兵。
const EMPTY: u32 = u32::MAX;

/// 直接引きの表を使う上限。`(max_id - min_id + 1) <= DENSE_FACTOR * n` なら密とみなす。
const DENSE_FACTOR: u64 = 4;

#[derive(Debug)]
enum MapKind {
    /// ID が密。`table[id - min] = index`
    Dense { min: NodeId, table: Vec<u32> },
    /// ID が疎。ID 昇順にソートした (id, index) の配列を二分探索する
    Sparse { sorted: Vec<(NodeId, u32)> },
}

/// ID → インデックスの写像。
#[derive(Debug)]
pub struct IdMap {
    kind: MapKind,
    len: usize,
}

impl IdMap {
    /// `ids[i]` を i 番目とみなす写像を作る。
    ///
    /// 同じ ID が複数回現れた場合は `Err` に**重複した ID の一覧**（昇順・重複なし）を返す。
    pub fn build(ids: &[NodeId]) -> Result<Self, Vec<NodeId>> {
        let len = ids.len();
        if len == 0 {
            return Ok(IdMap {
                kind: MapKind::Sparse { sorted: Vec::new() },
                len,
            });
        }
        let mut min = ids[0];
        let mut max = ids[0];
        for &id in ids {
            min = min.min(id);
            max = max.max(id);
        }
        let span = (max as u64) - (min as u64) + 1;
        if span <= DENSE_FACTOR * (len as u64) && span <= (u32::MAX as u64 - 1) {
            let mut table = vec![EMPTY; span as usize];
            let mut dups: Vec<NodeId> = Vec::new();
            for (i, &id) in ids.iter().enumerate() {
                let slot = &mut table[(id - min) as usize];
                if *slot != EMPTY {
                    dups.push(id);
                } else {
                    *slot = i as u32;
                }
            }
            if !dups.is_empty() {
                dups.sort_unstable();
                dups.dedup();
                return Err(dups);
            }
            Ok(IdMap {
                kind: MapKind::Dense { min, table },
                len,
            })
        } else {
            let mut sorted: Vec<(NodeId, u32)> = ids
                .iter()
                .enumerate()
                .map(|(i, &id)| (id, i as u32))
                .collect();
            sorted.sort_unstable_by_key(|&(id, _)| id);
            let mut dups: Vec<NodeId> = Vec::new();
            for w in sorted.windows(2) {
                if w[0].0 == w[1].0 {
                    dups.push(w[0].0);
                }
            }
            if !dups.is_empty() {
                dups.dedup();
                return Err(dups);
            }
            Ok(IdMap {
                kind: MapKind::Sparse { sorted },
                len,
            })
        }
    }

    /// 登録されている ID の個数。
    pub fn len(&self) -> usize {
        self.len
    }

    /// 空か。
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// ID を索引に変換する。未定義なら `None`。
    #[inline]
    pub fn get(&self, id: NodeId) -> Option<u32> {
        match &self.kind {
            MapKind::Dense { min, table } => {
                if id < *min {
                    return None;
                }
                let off = (id - min) as usize;
                match table.get(off) {
                    Some(&v) if v != EMPTY => Some(v),
                    _ => None,
                }
            }
            MapKind::Sparse { sorted } => sorted
                .binary_search_by_key(&id, |&(i, _)| i)
                .ok()
                .map(|p| sorted[p].1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_ids() {
        let map = IdMap::build(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(map.get(1), Some(0));
        assert_eq!(map.get(5), Some(4));
        assert_eq!(map.get(6), None);
        assert_eq!(map.get(0), None);
    }

    #[test]
    fn sparse_ids() {
        let map = IdMap::build(&[10, 1_000_000, 7, 999]).unwrap();
        assert_eq!(map.get(10), Some(0));
        assert_eq!(map.get(1_000_000), Some(1));
        assert_eq!(map.get(7), Some(2));
        assert_eq!(map.get(999), Some(3));
        assert_eq!(map.get(11), None);
    }

    #[test]
    fn duplicates_are_reported() {
        assert_eq!(IdMap::build(&[1, 2, 2, 3, 1]).unwrap_err(), vec![1, 2]);
        assert_eq!(IdMap::build(&[5, 1_000_000, 5]).unwrap_err(), vec![5]);
    }

    #[test]
    fn empty_map() {
        let map = IdMap::build(&[]).unwrap();
        assert!(map.is_empty());
        assert_eq!(map.get(1), None);
    }
}
