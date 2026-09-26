//! 等値面（Marching Tetrahedra）（SPEC §6.3）。
//!
//! 四面体は「4 節点の値と閾値の大小」だけで切り口が決まるので、
//! Marching Cubes のような曖昧さ（同じ符号配置に複数の解釈がある問題）が起きない。
//! 切り口は 0 枚・1 枚・2 枚の三角形のいずれかになる。
//!
//! 大規模モデルでも GUI に渡すのは切り口の三角形だけで済むよう、
//! 抽出は Rust 側で行う（SPEC §6.3）。
//!
//! Hex8 / Penta6 を含むメッシュは[面共役な四面体分割](crate::mesh::TetView)を
//! 通してから切る。四角形面が両側から同じ対角線で割られるので、
//! 等値面に隙間や重なりができない（SPEC §5.1 / §8 M7）。
//!
//! # 決定性
//!
//! 頂点は「辺」を鍵にして共有し、四面体の通し番号順に生成する。
//! 探索にハッシュ表を使っているが、**出力の並びは生成順で決まる**ので、
//! 同じ入力からは常に同じ三角形列になる（CLAUDE.md 設計原則 1）。

use crate::error::{GfemError, Result};
use crate::geom::{cross, norm, sub};
use crate::mesh::tetview::TetView;
use crate::mesh::{Mesh, NodeIndex};
use std::collections::HashMap;

/// 三角形の集まり（等値面や断面の表示に使う）。
#[derive(Debug, Clone, Default)]
pub struct TriangleMesh {
    /// 頂点座標 [m]。3 成分ずつ
    pub points: Vec<f64>,
    /// 三角形の頂点番号。3 個ずつ
    pub triangles: Vec<u32>,
    /// 頂点に補間された付随スカラー（`extra` に渡した順）
    pub scalars: Vec<Vec<f64>>,
    /// 各三角形の元ソリッド要素の通し番号（分割前の要素。SPEC §5.1）
    pub source_elements: Vec<u32>,
}

impl TriangleMesh {
    /// 頂点数。
    pub fn n_points(&self) -> usize {
        self.points.len() / 3
    }

    /// 三角形の枚数。
    pub fn n_triangles(&self) -> usize {
        self.triangles.len() / 3
    }

    /// 三角形が 1 枚も無いか。
    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    /// 総面積 [m²]。
    pub fn area(&self) -> f64 {
        let mut total = 0.0;
        for t in self.triangles.chunks_exact(3) {
            let p = |k: u32| {
                let i = k as usize * 3;
                [self.points[i], self.points[i + 1], self.points[i + 2]]
            };
            let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
            total += 0.5 * norm(cross(sub(b, a), sub(c, a)));
        }
        total
    }
}

/// 節点値 `values` の等値面（`values = level` の面）を作る。
///
/// `extra` に節点値の配列を渡すと、同じ補間係数で頂点に載せて返す
/// （等値面の上に別の量を色として塗るために使う）。
pub fn iso_surface(
    mesh: &Mesh,
    tets: &TetView,
    values: &[f64],
    level: f64,
    extra: &[&[f64]],
) -> Result<TriangleMesh> {
    if values.len() != mesh.nodes.len() {
        return Err(GfemError::internal(format!(
            "節点値の個数が合いません: {} 個ですが節点は {} 個です",
            values.len(),
            mesh.nodes.len()
        )));
    }
    for (i, e) in extra.iter().enumerate() {
        if e.len() != mesh.nodes.len() {
            return Err(GfemError::internal(format!(
                "extra[{i}] の個数が節点数と合いません"
            )));
        }
    }

    let mut out = TriangleMesh {
        scalars: vec![Vec::new(); extra.len()],
        ..Default::default()
    };
    // 辺 (小さい節点番号, 大きい節点番号) → 生成済み頂点番号
    let mut edge_vertices: HashMap<(NodeIndex, NodeIndex), u32> = HashMap::new();

    for t in 0..tets.len() {
        let conn = tets.conn_of(t);
        let value = [
            values[conn[0] as usize],
            values[conn[1] as usize],
            values[conn[2] as usize],
            values[conn[3] as usize],
        ];
        // 閾値より大きい節点と、それ以外に分ける
        let mut above: Vec<usize> = Vec::with_capacity(4);
        let mut below: Vec<usize> = Vec::with_capacity(4);
        for (i, &v) in value.iter().enumerate() {
            if v > level {
                above.push(i);
            } else {
                below.push(i);
            }
        }
        if above.is_empty() || below.is_empty() {
            continue; // 切り口なし
        }

        let vertex = |out: &mut TriangleMesh,
                      edge_vertices: &mut HashMap<(NodeIndex, NodeIndex), u32>,
                      a: usize,
                      b: usize|
         -> u32 {
            let (na, nb) = (conn[a], conn[b]);
            let key = if na < nb { (na, nb) } else { (nb, na) };
            if let Some(&index) = edge_vertices.get(&key) {
                return index;
            }
            // 辺の向きを鍵に合わせて、補間係数が節点の並び方に依らないようにする
            let (lo, hi) = if na < nb { (a, b) } else { (b, a) };
            let (vlo, vhi) = (value[lo], value[hi]);
            let denominator = vhi - vlo;
            let t = if denominator != 0.0 {
                (level - vlo) / denominator
            } else {
                0.5
            };
            let plo = mesh.nodes.coord(conn[lo]);
            let phi = mesh.nodes.coord(conn[hi]);
            let index = out.n_points() as u32;
            for d in 0..3 {
                out.points.push(plo[d] + t * (phi[d] - plo[d]));
            }
            for (k, source) in extra.iter().enumerate() {
                let a = source[conn[lo] as usize];
                let b = source[conn[hi] as usize];
                out.scalars[k].push(a + t * (b - a));
            }
            edge_vertices.insert(key, index);
            index
        };

        match (above.len(), below.len()) {
            // 1 節点だけが上（または下）→ 三角形 1 枚
            (1, 3) | (3, 1) => {
                let (single, others) = if above.len() == 1 {
                    (above[0], &below)
                } else {
                    (below[0], &above)
                };
                let tri = [
                    vertex(&mut out, &mut edge_vertices, single, others[0]),
                    vertex(&mut out, &mut edge_vertices, single, others[1]),
                    vertex(&mut out, &mut edge_vertices, single, others[2]),
                ];
                out.triangles.extend_from_slice(&tri);
                out.source_elements.push(tets.parent(t));
            }
            // 2 対 2 → 四辺形（三角形 2 枚）
            (2, 2) => {
                // 四辺形の頂点は、四面体の 4 つの面をたどる順に並ぶ
                let quad = [
                    vertex(&mut out, &mut edge_vertices, above[0], below[0]),
                    vertex(&mut out, &mut edge_vertices, above[0], below[1]),
                    vertex(&mut out, &mut edge_vertices, above[1], below[1]),
                    vertex(&mut out, &mut edge_vertices, above[1], below[0]),
                ];
                out.triangles
                    .extend_from_slice(&[quad[0], quad[1], quad[2]]);
                out.triangles
                    .extend_from_slice(&[quad[0], quad[2], quad[3]]);
                out.source_elements.push(tets.parent(t));
                out.source_elements.push(tets.parent(t));
            }
            _ => unreachable!("above + below は必ず 4"),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::tetview::TetView;

    /// 単位立方体を 6 個の四面体に分けたメッシュ（Kuhn 分割）。
    fn unit_cube() -> Mesh {
        crate::testing::kuhn_box([1, 1, 1], [1.0; 3])
    }

    #[test]
    fn plane_through_the_cube_has_the_right_area() {
        let mesh = unit_cube();
        // φ = z の等値面 z = 0.5 は 1×1 の正方形
        let values: Vec<f64> = (0..mesh.nodes.len())
            .map(|i| mesh.nodes.coord(i as u32)[2])
            .collect();
        let surface =
            iso_surface(&mesh, &TetView::build(&mesh).unwrap(), &values, 0.5, &[]).unwrap();
        assert!(!surface.is_empty());
        assert!(
            (surface.area() - 1.0).abs() < 1e-12,
            "面積 {}",
            surface.area()
        );
        // 全ての頂点が z = 0.5 に載っている
        for p in surface.points.chunks_exact(3) {
            assert!((p[2] - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn iso_level_outside_the_range_gives_nothing() {
        let mesh = unit_cube();
        let values: Vec<f64> = (0..mesh.nodes.len())
            .map(|i| mesh.nodes.coord(i as u32)[2])
            .collect();
        assert!(
            iso_surface(&mesh, &TetView::build(&mesh).unwrap(), &values, 2.0, &[])
                .unwrap()
                .is_empty()
        );
        assert!(
            iso_surface(&mesh, &TetView::build(&mesh).unwrap(), &values, -1.0, &[])
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn vertices_are_shared_between_neighbouring_tetrahedra() {
        let mesh = unit_cube();
        let values: Vec<f64> = (0..mesh.nodes.len())
            .map(|i| mesh.nodes.coord(i as u32)[2])
            .collect();
        let surface =
            iso_surface(&mesh, &TetView::build(&mesh).unwrap(), &values, 0.5, &[]).unwrap();
        // 辺ごとに 1 頂点なので、三角形の枚数より頂点数は少ない
        assert!(surface.n_points() < 3 * surface.n_triangles());
        // 同じ座標の頂点が重複していない
        for i in 0..surface.n_points() {
            for j in (i + 1)..surface.n_points() {
                let d: f64 = (0..3)
                    .map(|d| (surface.points[3 * i + d] - surface.points[3 * j + d]).powi(2))
                    .sum();
                assert!(d > 1e-20, "頂点 {i} と {j} が重複しています");
            }
        }
    }

    #[test]
    fn extra_scalars_are_interpolated_with_the_same_weights() {
        let mesh = unit_cube();
        let z: Vec<f64> = (0..mesh.nodes.len())
            .map(|i| mesh.nodes.coord(i as u32)[2])
            .collect();
        // 付随量として 10z を渡すと、等値面上では 10 × 0.5 = 5 になるはず
        let scaled: Vec<f64> = z.iter().map(|v| 10.0 * v).collect();
        let surface =
            iso_surface(&mesh, &TetView::build(&mesh).unwrap(), &z, 0.5, &[&scaled]).unwrap();
        assert_eq!(surface.scalars.len(), 1);
        assert_eq!(surface.scalars[0].len(), surface.n_points());
        for v in &surface.scalars[0] {
            assert!((v - 5.0).abs() < 1e-12, "{v}");
        }
    }

    #[test]
    fn is_reproducible() {
        let mesh = unit_cube();
        let values: Vec<f64> = (0..mesh.nodes.len())
            .map(|i| mesh.nodes.coord(i as u32)[0] + mesh.nodes.coord(i as u32)[1])
            .collect();
        let a = iso_surface(&mesh, &TetView::build(&mesh).unwrap(), &values, 0.7, &[]).unwrap();
        let b = iso_surface(&mesh, &TetView::build(&mesh).unwrap(), &values, 0.7, &[]).unwrap();
        assert_eq!(a.points, b.points);
        assert_eq!(a.triangles, b.triangles);
        assert_eq!(a.source_elements, b.source_elements);
    }

    #[test]
    fn rejects_mismatched_arrays() {
        let mesh = unit_cube();
        let e = iso_surface(
            &mesh,
            &TetView::build(&mesh).unwrap(),
            &[1.0, 2.0],
            0.5,
            &[],
        )
        .unwrap_err();
        assert!(e.to_string().contains("節点値の個数"), "{e}");
    }

    #[test]
    fn diagonal_plane_area() {
        let mesh = unit_cube();
        // x + y = 1 の面は 1 × √2 の長方形
        let values: Vec<f64> = (0..mesh.nodes.len())
            .map(|i| {
                let p = mesh.nodes.coord(i as u32);
                p[0] + p[1]
            })
            .collect();
        let surface =
            iso_surface(&mesh, &TetView::build(&mesh).unwrap(), &values, 1.0, &[]).unwrap();
        assert!(
            (surface.area() - 2.0f64.sqrt()).abs() < 1e-12,
            "面積 {}",
            surface.area()
        );
    }
}
