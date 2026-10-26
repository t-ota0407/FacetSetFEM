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

use crate::error::{FacetSetFemError, Result};
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
    /// 各頂点が載っている辺の両端（節点の通し番号、小さいほうが先）。
    /// 節点そのものにある頂点は両端が同じ
    pub point_edges: Vec<[NodeIndex; 2]>,
    /// 各頂点の、辺の 1 つ目の節点から測った位置（0〜1）。
    ///
    /// 節点値 `v` の頂点での値は `v[a] + t (v[b] - v[a])`（`[a, b]` は [`Self::point_edges`]）。
    /// どの量でも呼び出し側で補間できるように持たせる（[`crate::mesh::section`]）
    pub point_t: Vec<f64>,
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
            let (a, b, c) = (self.point(t[0]), self.point(t[1]), self.point(t[2]));
            total += 0.5 * norm(cross(sub(b, a), sub(c, a)));
        }
        total
    }

    /// 頂点 `k` の座標。
    pub fn point(&self, k: u32) -> [f64; 3] {
        let i = k as usize * 3;
        [self.points[i], self.points[i + 1], self.points[i + 2]]
    }
}

/// 四面体の切り方（[`Builder::march`]）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cutting {
    /// 等値面。値がちょうど閾値の節点は「下」に数え、面積 0 の三角形も残す
    /// （以前からの出力を 1 ビットも変えない）
    IsoSurface,
    /// 平面の切り口（断面・クリップ）。距離がちょうど 0 の節点は「上」（法線の側）に数える。
    /// こうすると平面が要素の面に重なったとき、切り口は**残す側（法線と逆側）の要素**
    /// から作られ、クリップの切り口に切り落とした要素の値が塗られることがない。
    /// 面積が厳密に 0 の三角形は捨てる（[`Builder::triangle`]）
    Plane,
}

impl Cutting {
    /// 値 `v` の節点を閾値 `level` より「上」に数えるか。
    pub(crate) fn is_above(self, v: f64, level: f64) -> bool {
        match self {
            Cutting::IsoSurface => v > level,
            Cutting::Plane => v >= level,
        }
    }
}

/// 頂点を辺ごとに 1 つだけ作りながら三角形を積む（等値面・断面・クリップで共通）。
///
/// 頂点は「辺」（節点そのものなら両端が同じ組）を鍵にして共有する。探索にハッシュ表を
/// 使うが、**出力の並びは生成順で決まる**ので、同じ入力からは常に同じ三角形列になる
/// （CLAUDE.md 設計原則 1）。
pub(crate) struct Builder<'a> {
    mesh: &'a Mesh,
    extra: &'a [&'a [f64]],
    out: TriangleMesh,
    vertices: HashMap<[NodeIndex; 2], u32>,
}

impl<'a> Builder<'a> {
    pub(crate) fn new(mesh: &'a Mesh, extra: &'a [&'a [f64]]) -> Self {
        Self {
            mesh,
            extra,
            out: TriangleMesh {
                scalars: vec![Vec::new(); extra.len()],
                ..Default::default()
            },
            vertices: HashMap::new(),
        }
    }

    /// 節点 `node` にある頂点。
    pub(crate) fn node_vertex(&mut self, node: NodeIndex) -> u32 {
        if let Some(&index) = self.vertices.get(&[node, node]) {
            return index;
        }
        let index = self.out.n_points() as u32;
        self.out
            .points
            .extend_from_slice(&self.mesh.nodes.coord(node));
        for (k, source) in self.extra.iter().enumerate() {
            self.out.scalars[k].push(source[node as usize]);
        }
        self.out.point_edges.push([node, node]);
        self.out.point_t.push(0.0);
        self.vertices.insert([node, node], index);
        index
    }

    /// 辺 `(a, b)` の上で、節点値（`va`, `vb`）が `level` になる点にある頂点。
    pub(crate) fn edge_vertex(
        &mut self,
        a: NodeIndex,
        b: NodeIndex,
        va: f64,
        vb: f64,
        level: f64,
    ) -> u32 {
        // 辺の向きを節点番号に合わせて、補間係数が節点の並び方に依らないようにする
        let (lo, hi, vlo, vhi) = if a < b {
            (a, b, va, vb)
        } else {
            (b, a, vb, va)
        };
        if let Some(&index) = self.vertices.get(&[lo, hi]) {
            return index;
        }
        let denominator = vhi - vlo;
        let t = if denominator != 0.0 {
            (level - vlo) / denominator
        } else {
            0.5
        };
        let plo = self.mesh.nodes.coord(lo);
        let phi = self.mesh.nodes.coord(hi);
        let index = self.out.n_points() as u32;
        for d in 0..3 {
            self.out.points.push(plo[d] + t * (phi[d] - plo[d]));
        }
        for (k, source) in self.extra.iter().enumerate() {
            let a = source[lo as usize];
            let b = source[hi as usize];
            self.out.scalars[k].push(a + t * (b - a));
        }
        self.out.point_edges.push([lo, hi]);
        self.out.point_t.push(t);
        self.vertices.insert([lo, hi], index);
        index
    }

    /// 三角形を 1 枚積む。
    ///
    /// `drop_degenerate` なら、座標が完全に一致する頂点を含む（面積が厳密に 0 の）三角形は
    /// 積まない。平面がちょうど節点を通ると、節点で平面に触れるだけの四面体から
    /// こうした三角形ができる（断面の位置を格子面に合わせたときに日常的に起きる）。
    pub(crate) fn triangle(&mut self, tri: [u32; 3], source: u32, drop_degenerate: bool) {
        if drop_degenerate {
            let [a, b, c] = tri.map(|k| self.out.point(k));
            if a == b || b == c || a == c {
                return;
            }
        }
        self.out.triangles.extend_from_slice(&tri);
        self.out.source_elements.push(source);
    }

    /// 四面体 `t` を、節点値 `values` が `level` の面で切った切り口を積む（Marching Tetrahedra）。
    ///
    /// 切り口は 0 枚・1 枚・2 枚の三角形のいずれか。
    pub(crate) fn march(
        &mut self,
        tets: &TetView,
        t: usize,
        values: &[f64],
        level: f64,
        cutting: Cutting,
    ) {
        let conn = tets.conn_of(t);
        let value: [f64; 4] = std::array::from_fn(|i| values[conn[i] as usize]);
        let drop_degenerate = cutting == Cutting::Plane;
        // 閾値より上の節点と、それ以外に分ける
        let mut above: Vec<usize> = Vec::with_capacity(4);
        let mut below: Vec<usize> = Vec::with_capacity(4);
        for (i, &v) in value.iter().enumerate() {
            if cutting.is_above(v, level) {
                above.push(i);
            } else {
                below.push(i);
            }
        }
        if above.is_empty() || below.is_empty() {
            return; // 切り口なし
        }
        let mut vertex =
            |a: usize, b: usize| self.edge_vertex(conn[a], conn[b], value[a], value[b], level);

        match (above.len(), below.len()) {
            // 1 節点だけが上（または下）→ 三角形 1 枚
            (1, 3) | (3, 1) => {
                let (single, others) = if above.len() == 1 {
                    (above[0], &below)
                } else {
                    (below[0], &above)
                };
                let tri = [
                    vertex(single, others[0]),
                    vertex(single, others[1]),
                    vertex(single, others[2]),
                ];
                self.triangle(tri, tets.parent(t), drop_degenerate);
            }
            // 2 対 2 → 四辺形（三角形 2 枚）
            (2, 2) => {
                // 四辺形の頂点は、四面体の 4 つの面をたどる順に並ぶ
                let quad = [
                    vertex(above[0], below[0]),
                    vertex(above[0], below[1]),
                    vertex(above[1], below[1]),
                    vertex(above[1], below[0]),
                ];
                self.triangle([quad[0], quad[1], quad[2]], tets.parent(t), drop_degenerate);
                self.triangle([quad[0], quad[2], quad[3]], tets.parent(t), drop_degenerate);
            }
            _ => unreachable!("above + below は必ず 4"),
        }
    }

    /// ここまでに積んだ三角形の枚数。
    pub(crate) fn n_triangles(&self) -> usize {
        self.out.n_triangles()
    }

    /// 三角形 `first` 以降の向き（頂点の巡回順）を、法線が `normal` の側を向くようにそろえる。
    ///
    /// 平面の切り口に使う。Marching Tetrahedra の出力は三角形ごとに向きがばらばらで、
    /// 法線を計算して陰影を付けるとまだらになる（SPEC §6.3「法線が無いと陰影が付かない」）。
    pub(crate) fn orient_since(&mut self, first: usize, normal: [f64; 3]) {
        use crate::geom::dot;
        for tri in self.out.triangles[3 * first..].chunks_exact_mut(3) {
            let [a, b, c] = [tri[0], tri[1], tri[2]].map(|k| {
                let i = k as usize * 3;
                [
                    self.out.points[i],
                    self.out.points[i + 1],
                    self.out.points[i + 2],
                ]
            });
            if dot(cross(sub(b, a), sub(c, a)), normal) < 0.0 {
                tri.swap(1, 2);
            }
        }
    }

    pub(crate) fn finish(self) -> TriangleMesh {
        self.out
    }
}

/// 節点値の配列が節点数と合っているか。
pub(crate) fn check_nodal(mesh: &Mesh, values: &[f64], what: &str) -> Result<()> {
    if values.len() != mesh.nodes.len() {
        return Err(FacetSetFemError::internal(crate::tr!(
            "Wrong number of {what}: {} given, but there are {} nodes",
            "{what}の個数が合いません: {} 個ですが節点は {} 個です",
            values.len(),
            mesh.nodes.len()
        )));
    }
    Ok(())
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
    check_nodal(mesh, values, &crate::tr!("nodal values", "節点値"))?;
    for (i, e) in extra.iter().enumerate() {
        check_nodal(mesh, e, &crate::tr!("values in extra[{i}]", "extra[{i}] "))?;
    }
    let mut builder = Builder::new(mesh, extra);
    for t in 0..tets.len() {
        builder.march(tets, t, values, level, Cutting::IsoSurface);
    }
    Ok(builder.finish())
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
