//! 断面とクリップ（SPEC §6.3）。
//!
//! GUI の「断面スライス」「クリップ」は、以前は体積メッシュ全体を VTK の `UnstructuredGrid`
//! にして VTK で切っていた。頭部クラスのモデル（66 万節点・378 万四面体）で、それだけで
//! 468 MB を使い、スライダーを 1 目盛り動かすたびに断面で 1〜2 秒かかった。ここでは
//! **切り口の三角形だけ**を作って渡す。
//!
//! - **断面**は「平面までの符号付き距離」という 1 次関数の等値面（値 0）。1 次関数なので
//!   四面体の中で切り口は厳密に平面になり、Marching Tetrahedra（[`iso_surface`]）の
//!   仕組みがそのまま使える。Hex8 / Penta6 は[面共役な四面体分割](TetView)を通して切る
//! - **クリップ**の見た目は「外表面のうち残す側」と「断面」を合わせたもの。VTK が体積
//!   メッシュを描くときも外側の面しか見えないので、体積メッシュを渡す必要はない。
//!   外表面を切った点と断面の頂点は同じ辺の上なら共有する
//!
//! 頂点ごとに「どの辺のどこにあるか」（[`TriangleMesh::point_edges`] /
//! [`TriangleMesh::point_t`]）を返すので、節点値はどの量でも呼び出し側で補間できる。
//! 要素値は三角形の元要素（[`TriangleMesh::source_elements`]）で引く。
//!
//! [`iso_surface`]: crate::mesh::iso_surface

use crate::error::{ErrorKind, GfemError, Result};
use crate::geom::{dot, norm, scale, sub};
use crate::mesh::isosurface::{Builder, Cutting};
use crate::mesh::tetview::TetView;
use crate::mesh::topology::Surface;
use crate::mesh::{Mesh, TriangleMesh};
use rayon::prelude::*;

/// 切断面。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    /// 平面上の 1 点 [m]
    origin: [f64; 3],
    /// 単位法線
    normal: [f64; 3],
}

impl Plane {
    /// 点 `origin` を通り、法線が `normal` の平面。`normal` は長さ 1 でなくてよい。
    pub fn new(origin: [f64; 3], normal: [f64; 3]) -> Result<Self> {
        if origin.iter().any(|v| !v.is_finite()) {
            return Err(GfemError::new(
                ErrorKind::Case,
                crate::tr!(
                    "The position of the section plane is not a number: {origin:?}",
                    "切断面の位置が数値になっていません: {origin:?}"
                ),
            )
            .with_hint(crate::tr!(
                "Give the plane position as finite coordinates in m",
                "平面の位置は m 単位の有限な座標で指定してください"
            )));
        }
        let length = norm(normal);
        if !length.is_finite() || length == 0.0 {
            return Err(GfemError::new(
                ErrorKind::Case,
                crate::tr!(
                    "Invalid normal of the section plane: {normal:?}",
                    "切断面の法線が不正です: {normal:?}"
                ),
            )
            .with_hint(crate::tr!(
                "Give the normal as a finite vector of nonzero length (e.g. [0, 0, 1])",
                "法線は長さが 0 でない有限なベクトルで指定してください（例: [0, 0, 1]）"
            )));
        }
        Ok(Self {
            origin,
            normal: scale(normal, 1.0 / length),
        })
    }

    /// 平面上の 1 点 [m]。
    pub fn origin(&self) -> [f64; 3] {
        self.origin
    }

    /// 単位法線。
    pub fn normal(&self) -> [f64; 3] {
        self.normal
    }

    /// 節点ごとの符号付き距離 [m]（法線の向きが正）。
    fn distances(&self, mesh: &Mesh) -> Vec<f64> {
        // 節点ごとに独立に計算するので、並列でも結果は逐次と同じ
        (0..mesh.nodes.len() as u32)
            .into_par_iter()
            .map(|i| dot(sub(mesh.nodes.coord(i), self.origin), self.normal))
            .collect()
    }
}

/// 平面で切った切り口を持つ四面体の通し番号（昇順）。
///
/// 平面の「上」（[`Cutting::Plane`] の分け方）と「下」の節点を両方持つもの。全体を
/// 舐めるのは並列で行い、実際に切り口を作る（頂点を共有するので逐次の）処理は
/// この四面体だけに絞る。
fn cut_tets(tets: &TetView, distance: &[f64]) -> Vec<usize> {
    // rayon の collect は順序を保つので、並びは逐次と同じ
    (0..tets.len())
        .into_par_iter()
        .filter(|&t| {
            let conn = tets.conn_of(t);
            let above = conn
                .iter()
                .filter(|&&n| Cutting::Plane.is_above(distance[n as usize], 0.0))
                .count();
            above != 0 && above != 4
        })
        .collect()
}

/// 断面（平面 `plane` で体積メッシュを切った切り口）。
///
/// 三角形の元要素は、切った四面体の分割前のソリッド要素。
pub fn slice(mesh: &Mesh, tets: &TetView, plane: &Plane) -> TriangleMesh {
    let distance = plane.distances(mesh);
    let mut builder = Builder::new(mesh, &[]);
    cut(&mut builder, tets, &distance, plane);
    builder.finish()
}

/// 切り口の三角形を積む。向きは法線の側（クリップで切り落とした側＝外向き）にそろえる。
fn cut(builder: &mut Builder<'_>, tets: &TetView, distance: &[f64], plane: &Plane) {
    let first = builder.n_triangles();
    for t in cut_tets(tets, distance) {
        builder.march(tets, t, distance, 0.0, Cutting::Plane);
    }
    builder.orient_since(first, plane.normal);
}

/// クリップ（平面 `plane` の**法線の向きの側を切り落とした**メッシュ）の見た目。
///
/// 外表面のうち残す側（平面をまたぐ三角形は平面で切る）と、断面を合わせて返す。
/// 残るのは距離が 0 以下の部分で、断面はちょうどその境界になる（隙間ができない）。
/// 外表面の三角形の元要素はその面を持つソリッド要素、断面の三角形は [`slice`] と同じ
/// （平面が要素の面に重なっても、切り口は残す側の要素から作る。[`Cutting::Plane`]）。
pub fn clip(mesh: &Mesh, tets: &TetView, surface: &Surface, plane: &Plane) -> TriangleMesh {
    let distance = plane.distances(mesh);
    let kept = |n: u32| distance[n as usize] <= 0.0;
    let mut builder = Builder::new(mesh, &[]);

    // --- 外表面の残す側 ---------------------------------------------------------
    let (triangles, owners) = surface.triangulate();
    for (tri, owner) in triangles.into_iter().zip(owners) {
        let n_kept = tri.iter().filter(|&&n| kept(n)).count();
        match n_kept {
            0 => {}
            3 => {
                let t = tri.map(|n| builder.node_vertex(n));
                builder.triangle(t, owner, true);
            }
            _ => {
                // 巡回させて向き（外向き）を保ったまま、「仲間外れ」の頂点を先頭に置く
                let odd = (0..3)
                    .find(|&i| kept(tri[i]) == (n_kept == 1))
                    .expect("残す頂点が 1 個か 2 個なら仲間外れが 1 個ある");
                let [p, q, r] = [tri[odd], tri[(odd + 1) % 3], tri[(odd + 2) % 3]];
                let mut cut = |a: u32, b: u32| {
                    builder.edge_vertex(a, b, distance[a as usize], distance[b as usize], 0.0)
                };
                let (pq, rp) = (cut(p, q), cut(r, p));
                if n_kept == 1 {
                    // p だけ残す → 三角形 (p, pq, rp)
                    let p = builder.node_vertex(p);
                    builder.triangle([p, pq, rp], owner, true);
                } else {
                    // p だけ落とす → 四辺形 (pq, q, r, rp)
                    let (q, r) = (builder.node_vertex(q), builder.node_vertex(r));
                    builder.triangle([pq, q, r], owner, true);
                    builder.triangle([pq, r, rp], owner, true);
                }
            }
        }
    }

    // --- 切り口（断面） -----------------------------------------------------------
    cut(&mut builder, tets, &distance, plane);
    builder.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::extract_surface;

    fn cube(n: usize) -> (Mesh, TetView) {
        let mesh = crate::testing::kuhn_box([n, n, n], [1.0; 3]);
        let tets = TetView::build(&mesh).unwrap();
        (mesh, tets)
    }

    #[test]
    fn slicing_a_cube_gives_a_unit_square() {
        let (mesh, tets) = cube(3);
        for (normal, at) in [([0.0, 0.0, 1.0], 0.4), ([1.0, 0.0, 0.0], 0.75)] {
            let plane = Plane::new([at; 3], normal).unwrap();
            let s = slice(&mesh, &tets, &plane);
            assert!((s.area() - 1.0).abs() < 1e-12, "面積 {}", s.area());
            for k in 0..s.n_points() as u32 {
                let d = dot(sub(s.point(k), [at; 3]), normal);
                assert!(d.abs() < 1e-12, "頂点が平面から {d} 離れている");
            }
        }
    }

    #[test]
    fn a_plane_through_grid_nodes_gives_no_duplicate_or_degenerate_triangles() {
        // 格子面にちょうど載る断面（GUI の既定の 50% 位置で起きる）
        let (mesh, tets) = cube(2);
        let s = slice(
            &mesh,
            &tets,
            &Plane::new([0.5; 3], [0.0, 0.0, 1.0]).unwrap(),
        );
        // 片側の四面体の面だけが断面になる（反対側と重ならない）
        assert!((s.area() - 1.0).abs() < 1e-12, "面積 {}", s.area());
        for t in s.triangles.chunks_exact(3) {
            let [a, b, c] = [s.point(t[0]), s.point(t[1]), s.point(t[2])];
            assert!(a != b && b != c && a != c, "面積 0 の三角形がある");
        }
    }

    #[test]
    fn a_cut_on_element_faces_takes_the_elements_on_the_kept_side() {
        // 平面が要素の面に重なるとき、切り口の元要素は法線と逆側（クリップで残す側）。
        // 反対側を取ると、クリップの切り口に切り落とした要素の値が塗られる
        let (mesh, tets) = cube(2);
        let surface = extract_surface(&mesh).unwrap();
        let plane = Plane::new([0.5; 3], [0.0, 0.0, 1.0]).unwrap();
        let centroid_z = |e: u32| {
            let conn = mesh.solid(e as usize).conn;
            conn.iter().map(|&n| mesh.nodes.coord(n)[2]).sum::<f64>() / conn.len() as f64
        };
        let s = slice(&mesh, &tets, &plane);
        assert!(!s.is_empty());
        assert!(s.source_elements.iter().all(|&e| centroid_z(e) < 0.5));
        let c = clip(&mesh, &tets, &surface, &plane);
        assert!(c.source_elements.iter().all(|&e| centroid_z(e) < 0.5));
    }

    #[test]
    fn a_diagonal_slice_has_the_right_area_and_elements() {
        let (mesh, tets) = cube(4);
        // x + y = 1 は 1 × √2 の長方形
        let s = slice(
            &mesh,
            &tets,
            &Plane::new([0.5, 0.5, 0.0], [1.0, 1.0, 0.0]).unwrap(),
        );
        assert!(
            (s.area() - 2.0f64.sqrt()).abs() < 1e-12,
            "面積 {}",
            s.area()
        );
        assert_eq!(s.source_elements.len(), s.n_triangles());
        assert!(
            s.source_elements
                .iter()
                .all(|&e| (e as usize) < mesh.n_solid_elements())
        );
    }

    #[test]
    fn point_edges_reproduce_the_vertex_positions() {
        // 呼び出し側は point_edges / point_t で節点値を補間する。座標で確かめる
        let (mesh, tets) = cube(3);
        let s = slice(
            &mesh,
            &tets,
            &Plane::new([0.3, 0.6, 0.2], [1.0, 2.0, 3.0]).unwrap(),
        );
        assert_eq!(s.point_edges.len(), s.n_points());
        assert_eq!(s.point_t.len(), s.n_points());
        for k in 0..s.n_points() {
            let [a, b] = s.point_edges[k];
            let (pa, pb) = (mesh.nodes.coord(a), mesh.nodes.coord(b));
            let t = s.point_t[k];
            for d in 0..3 {
                let v = pa[d] + t * (pb[d] - pa[d]);
                assert!((v - s.points[3 * k + d]).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn a_plane_outside_the_model_gives_nothing() {
        let (mesh, tets) = cube(2);
        let s = slice(
            &mesh,
            &tets,
            &Plane::new([0.0, 0.0, 2.0], [0.0, 0.0, 1.0]).unwrap(),
        );
        assert!(s.is_empty());
    }

    #[test]
    fn clipping_a_cube_keeps_the_side_against_the_normal() {
        // z > 0.25 を切り落とすと 1 × 1 × 0.25 の箱が残る。見える面積は
        // 底面 1 + 側面 4 × 0.25 + 切り口 1 = 3
        let (mesh, tets) = cube(4);
        let surface = extract_surface(&mesh).unwrap();
        let plane = Plane::new([0.5, 0.5, 0.3], [0.0, 0.0, 1.0]).unwrap();
        let c = clip(&mesh, &tets, &surface, &plane);
        assert!(
            (c.area() - (1.0 + 4.0 * 0.3 + 1.0)).abs() < 1e-12,
            "面積 {}",
            c.area()
        );
        for k in 0..c.n_points() as u32 {
            assert!(c.point(k)[2] <= 0.3 + 1e-12, "切り落とす側に頂点がある");
        }
    }

    #[test]
    fn clipping_shares_vertices_between_the_surface_and_the_cut() {
        // 外表面を切った点と断面の頂点は同じ辺に載るので 1 つにまとまる（隙間ができない）
        let (mesh, tets) = cube(3);
        let surface = extract_surface(&mesh).unwrap();
        let c = clip(
            &mesh,
            &tets,
            &surface,
            &Plane::new([0.5; 3], [1.0, 0.3, 0.2]).unwrap(),
        );
        let mut seen = std::collections::HashSet::new();
        for k in 0..c.n_points() as u32 {
            let p = c.point(k).map(f64::to_bits);
            assert!(
                seen.insert(p),
                "同じ座標の頂点が 2 つある: {:?}",
                c.point(k)
            );
        }
    }

    #[test]
    fn clipping_keeps_the_outward_orientation_of_the_surface() {
        // 残した外表面の三角形は元の外表面と同じ向き（外向き）。底面 z = 0 は -z を向く
        let (mesh, tets) = cube(2);
        let surface = extract_surface(&mesh).unwrap();
        let c = clip(
            &mesh,
            &tets,
            &surface,
            &Plane::new([0.5; 3], [0.0, 0.0, 1.0]).unwrap(),
        );
        let mut bottom = 0;
        for t in c.triangles.chunks_exact(3) {
            let [a, b, p] = [c.point(t[0]), c.point(t[1]), c.point(t[2])];
            if a[2] == 0.0 && b[2] == 0.0 && p[2] == 0.0 {
                let n = crate::geom::cross(sub(b, a), sub(p, a));
                assert!(n[2] < 0.0, "底面の三角形が内向き");
                bottom += 1;
            }
        }
        assert!(bottom > 0);
    }

    #[test]
    fn cut_faces_point_along_the_normal() {
        // 断面の三角形はすべて法線の側を向く（陰影がまだらにならない）
        let (mesh, tets) = cube(3);
        let normal = [0.3, -1.0, 0.6];
        let s = slice(&mesh, &tets, &Plane::new([0.5; 3], normal).unwrap());
        assert!(!s.is_empty());
        for t in s.triangles.chunks_exact(3) {
            let [a, b, c] = [s.point(t[0]), s.point(t[1]), s.point(t[2])];
            assert!(dot(crate::geom::cross(sub(b, a), sub(c, a)), normal) > 0.0);
        }
    }

    #[test]
    fn slicing_is_reproducible() {
        let (mesh, tets) = cube(4);
        let plane = Plane::new([0.37, 0.52, 0.41], [0.2, -1.0, 0.7]).unwrap();
        let (a, b) = (slice(&mesh, &tets, &plane), slice(&mesh, &tets, &plane));
        assert_eq!(a.points, b.points);
        assert_eq!(a.triangles, b.triangles);
        assert_eq!(a.source_elements, b.source_elements);
    }

    #[test]
    fn bad_planes_are_rejected_with_a_reason() {
        let e = Plane::new([0.0; 3], [0.0; 3]).unwrap_err().to_string();
        assert!(e.contains("法線"), "{e}");
        let e = Plane::new([f64::NAN, 0.0, 0.0], [0.0, 0.0, 1.0])
            .unwrap_err()
            .to_string();
        assert!(e.contains("位置"), "{e}");
    }
}
