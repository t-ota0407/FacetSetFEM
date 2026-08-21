//! 要素探索の空間分割（SPEC §8 M7）。
//!
//! 流線の始点がどの四面体に入るかを調べるのに、M6 までは全要素を走査していた。
//! 数万要素なら問題にならないが、1000 万要素で 100 本の流線を引くと
//! それだけで数十秒かかる。ここでは**一様格子**で候補を絞る。
//!
//! # なぜ一様格子か
//!
//! kd 木や BVH のほうが不均一なメッシュに強いが、
//!
//! - 有限要素メッシュは要素の大きさがそこまで極端に散らばらない
//! - 構築が O(要素数) の計数ソート 1 回で済み、木の構築より速い
//! - 「点の入る格子を計算する」だけなので探索に分岐が無い
//!
//! ため、この用途では一様格子で十分。四面体はその**軸平行境界箱**が重なる
//! 全ての格子に登録するので、点が四面体の中にあれば必ず候補に挙がる。
//!
//! 格子内の要素は**通し番号の昇順**に並ぶので、同じ点に対して常に同じ
//! 四面体が見つかる（CLAUDE.md 設計原則 1）。

use crate::mesh::Mesh;
use crate::mesh::tetview::TetView;

/// 格子の総数の上限。これを超えないように格子を粗くする。
const MAX_CELLS: usize = 8_000_000;

/// 1 格子あたりに入れたい四面体の数の目安。
const TARGET_PER_CELL: f64 = 2.0;

/// 四面体を軸平行の一様格子に配ったもの。
#[derive(Debug, Clone)]
pub struct CellGrid {
    origin: [f64; 3],
    /// 格子 1 つあたりの大きさの逆数
    inv_cell: [f64; 3],
    dims: [usize; 3],
    /// 長さ `dims の積 + 1`
    starts: Vec<u32>,
    /// 格子ごとの四面体番号（格子の中では昇順）
    items: Vec<u32>,
}

impl CellGrid {
    /// 四面体を格子に配る。
    pub fn build(mesh: &Mesh, tets: &TetView) -> Self {
        let n = tets.len();
        let (lo, hi) = mesh.bounding_box().unwrap_or(([0.0; 3], [1.0; 3]));
        // 境界にちょうど載る点を拾えるよう、ごく僅かに広げる
        let pad = [
            (hi[0] - lo[0]).max(1.0e-12) * 1.0e-9,
            (hi[1] - lo[1]).max(1.0e-12) * 1.0e-9,
            (hi[2] - lo[2]).max(1.0e-12) * 1.0e-9,
        ];
        let origin = [lo[0] - pad[0], lo[1] - pad[1], lo[2] - pad[2]];
        let extent = [
            (hi[0] - lo[0] + 2.0 * pad[0]).max(1.0e-12),
            (hi[1] - lo[1] + 2.0 * pad[1]).max(1.0e-12),
            (hi[2] - lo[2] + 2.0 * pad[2]).max(1.0e-12),
        ];

        let dims = choose_dims(extent, n);
        let inv_cell = [
            dims[0] as f64 / extent[0],
            dims[1] as f64 / extent[1],
            dims[2] as f64 / extent[2],
        ];
        let n_cells = dims[0] * dims[1] * dims[2];

        let cell_of = |x: f64, inv: f64, o: f64, dim: usize| -> usize {
            (((x - o) * inv) as isize).clamp(0, dim as isize - 1) as usize
        };
        // 四面体ごとの格子範囲。2 回計算しないよう覚えておく
        let mut spans: Vec<[u32; 6]> = Vec::with_capacity(n);
        let mut counts = vec![0u32; n_cells + 1];
        for t in 0..n {
            let p = tets.coords(mesh, t);
            let mut span = [0u32; 6];
            for d in 0..3 {
                let mut min = f64::INFINITY;
                let mut max = f64::NEG_INFINITY;
                for corner in &p {
                    min = min.min(corner[d]);
                    max = max.max(corner[d]);
                }
                span[d] = cell_of(min, inv_cell[d], origin[d], dims[d]) as u32;
                span[d + 3] = cell_of(max, inv_cell[d], origin[d], dims[d]) as u32;
            }
            for k in span[2]..=span[5] {
                for j in span[1]..=span[4] {
                    for i in span[0]..=span[3] {
                        let c = (k as usize * dims[1] + j as usize) * dims[0] + i as usize;
                        counts[c + 1] += 1;
                    }
                }
            }
            spans.push(span);
        }
        for c in 0..n_cells {
            counts[c + 1] += counts[c];
        }
        let starts = counts.clone();
        let mut cursor = counts;
        let mut items = vec![0u32; starts[n_cells] as usize];
        // 四面体を昇順にたどるので、格子の中の並びも昇順になる
        for (t, span) in spans.iter().enumerate() {
            for k in span[2]..=span[5] {
                for j in span[1]..=span[4] {
                    for i in span[0]..=span[3] {
                        let c = (k as usize * dims[1] + j as usize) * dims[0] + i as usize;
                        items[cursor[c] as usize] = t as u32;
                        cursor[c] += 1;
                    }
                }
            }
        }

        Self {
            origin,
            inv_cell,
            dims,
            starts,
            items,
        }
    }

    /// 点 `x` の入る格子に登録された四面体（昇順）。
    ///
    /// メッシュの外の点は、いちばん近い格子の中身を返す。呼び出し側は
    /// 体積座標で本当に中に入っているかを確かめること。
    pub fn candidates(&self, x: [f64; 3]) -> &[u32] {
        let mut c = 0usize;
        for d in (0..3).rev() {
            let i = (((x[d] - self.origin[d]) * self.inv_cell[d]) as isize)
                .clamp(0, self.dims[d] as isize - 1) as usize;
            c = c * self.dims[d] + i;
        }
        &self.items[self.starts[c] as usize..self.starts[c + 1] as usize]
    }

    /// 格子の分割数。
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// 登録された `(四面体, 格子)` の組の総数。メモリの目安。
    pub fn n_entries(&self) -> usize {
        self.items.len()
    }
}

/// 格子の分割数を決める。
///
/// 立方体に近い格子になるよう、まず 1 辺の長さを決めてから各方向の分割数を出す。
fn choose_dims(extent: [f64; 3], n: usize) -> [usize; 3] {
    if n == 0 {
        return [1, 1, 1];
    }
    let target = (n as f64 / TARGET_PER_CELL).max(1.0);
    let volume = extent[0] * extent[1] * extent[2];
    let mut side = (volume / target).cbrt();
    if side <= 0.0 || !side.is_finite() {
        // 平板や線状のメッシュでは体積がゼロになる。長い辺から決め直す
        side = extent.iter().cloned().fold(0.0f64, f64::max) / target.cbrt();
    }
    let mut dims = [1usize; 3];
    for d in 0..3 {
        dims[d] = ((extent[d] / side).ceil() as usize).clamp(1, 1024);
    }
    // 上限を超えたら全方向を一律に粗くする
    while dims[0] * dims[1] * dims[2] > MAX_CELLS {
        for d in dims.iter_mut() {
            *d = (*d / 2).max(1);
        }
    }
    dims
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::nastran::read_nastran_from;
    use crate::units::LengthUnit;
    use std::path::Path;

    /// nx × ny × nz の直方体を Kuhn 分割した四面体メッシュ。
    fn box_mesh(nx: usize, ny: usize, nz: usize) -> Mesh {
        let mut deck = String::new();
        let node = |i: usize, j: usize, k: usize| (k * (ny + 1) + j) * (nx + 1) + i + 1;
        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    deck.push_str(&format!(
                        "GRID,{},,{},{},{}\n",
                        node(i, j, k),
                        i as f64 / nx as f64,
                        j as f64 / ny as f64,
                        k as f64 / nz as f64
                    ));
                }
            }
        }
        deck.push_str("PSOLID,1,1\n");
        // Kuhn の 6 分割
        const KUHN: [[usize; 4]; 6] = [
            [0, 1, 3, 7],
            [0, 1, 7, 5],
            [0, 5, 7, 4],
            [0, 3, 2, 7],
            [0, 6, 4, 7],
            [0, 2, 6, 7],
        ];
        let mut eid = 1;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let corner =
                        |c: usize| node(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));
                    for t in KUHN {
                        deck.push_str(&format!(
                            "CTETRA,{eid},1,{},{},{},{}\n",
                            corner(t[0]),
                            corner(t[1]),
                            corner(t[2]),
                            corner(t[3])
                        ));
                        eid += 1;
                    }
                }
            }
        }
        read_nastran_from(deck.as_bytes(), Path::new("<test>"), LengthUnit::Meter)
            .unwrap()
            .0
    }

    #[test]
    fn every_tetrahedron_is_reachable_from_a_point_inside_it() {
        let mesh = box_mesh(4, 4, 4);
        let tets = TetView::build(&mesh).unwrap();
        let grid = CellGrid::build(&mesh, &tets);
        for t in 0..tets.len() {
            let p = tets.coords(&mesh, t);
            // 重心はその四面体の中
            let c = [
                (p[0][0] + p[1][0] + p[2][0] + p[3][0]) / 4.0,
                (p[0][1] + p[1][1] + p[2][1] + p[3][1]) / 4.0,
                (p[0][2] + p[1][2] + p[2][2] + p[3][2]) / 4.0,
            ];
            assert!(
                grid.candidates(c).contains(&(t as u32)),
                "四面体 {t} が重心の格子から見つからない"
            );
        }
    }

    #[test]
    fn the_grid_narrows_the_search_a_lot() {
        let mesh = box_mesh(8, 8, 8);
        let tets = TetView::build(&mesh).unwrap();
        let grid = CellGrid::build(&mesh, &tets);
        let worst = (0..grid.starts.len() - 1)
            .map(|c| (grid.starts[c + 1] - grid.starts[c]) as usize)
            .max()
            .unwrap();
        assert!(
            worst * 10 < tets.len(),
            "最悪の格子に {worst} 個も入っている（全 {} 個）",
            tets.len()
        );
    }

    #[test]
    fn points_outside_the_mesh_are_clamped_not_panicking() {
        let mesh = box_mesh(2, 2, 2);
        let tets = TetView::build(&mesh).unwrap();
        let grid = CellGrid::build(&mesh, &tets);
        for x in [
            [-100.0, 0.5, 0.5],
            [100.0, 0.5, 0.5],
            [0.5, f64::MIN_POSITIVE, 0.5],
            [0.5, 0.5, 1.0],
        ] {
            let _ = grid.candidates(x);
        }
    }

    #[test]
    fn a_flat_mesh_still_gets_a_usable_grid() {
        // z 方向の厚みが極端に薄い板。体積からの見積りが破綻しないこと
        let mut mesh = box_mesh(4, 4, 1);
        let coords_len = mesh.nodes.len();
        let mut coords = mesh.nodes.coords().to_vec();
        for i in 0..coords_len {
            coords[3 * i + 2] *= 1.0e-6;
        }
        mesh.nodes = crate::mesh::Nodes::new(mesh.nodes.ids().to_vec(), coords);
        let tets = TetView::build(&mesh).unwrap();
        let grid = CellGrid::build(&mesh, &tets);
        assert!(grid.dims().iter().all(|&d| d >= 1));
        assert!(grid.n_entries() >= tets.len());
    }
}
