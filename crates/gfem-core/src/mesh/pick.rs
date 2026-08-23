//! 外表面から電極面を選ぶ（SPEC §2.4 / §6.2 / §11.1）。
//!
//! シェル要素（CTRIA3 / CQUAD4）を持たないモデルでも電極を定義できるようにする。
//! 経皮電気刺激で使う人体モデルは**ソリッド要素だけで配られるのが普通**で、
//! シェルを前提にすると電極をひとつも置けない。外表面はソリッドの面から
//! 抽出できている（[`extract_surface`](super::topology::extract_surface)）ので、
//! そこから直接選べばよい。
//!
//! 選び方は「3D ビューでクリックした点」＋「半径」＋「PID の絞り込み」。
//!
//! **球に入っているかだけで決めてはいけない。** 実際の頭部モデルでは、首の
//! 切断面・耳の裏・折り返した皮膚など、**空間的には近いのに体表面としては
//! つながっていない面**が球に入る。クリックした面から**稜線でつながっている面だけ**を
//! 幅優先でたどることで、皮膚の上を這うように広がる選択になる。
//!
//! 返す面は通し番号の昇順で、入力が同じなら常に同じ集合・同じ並びになる
//! （CLAUDE.md 設計原則 1）。

use crate::mesh::topology::Surface;
use crate::mesh::{Mesh, NodeIndex, PropId};
use std::collections::BTreeMap;

/// 外表面の面を選ぶための下ごしらえ。
///
/// 重心・PID・稜線での隣接をまとめて持つ。メッシュだけで決まるので、
/// 1 度作れば電極を何個置いても使い回せる。
#[derive(Debug, Clone)]
pub struct SurfacePicker {
    /// 面の重心 [m]（3 個ずつ）
    centroids: Vec<f64>,
    /// 面を持つソリッド要素の PID
    pids: Vec<PropId>,
    /// 面の面積 [m²]
    areas: Vec<f64>,
    /// 隣接面の並び（CSR 風）
    offsets: Vec<u32>,
    neighbours: Vec<u32>,
}

impl SurfacePicker {
    /// 外表面から下ごしらえを作る。
    pub fn build(mesh: &Mesh, surface: &Surface) -> Self {
        let faces = surface.faces();
        let mut centroids = Vec::with_capacity(faces.len() * 3);
        let mut pids = Vec::with_capacity(faces.len());
        let mut areas = Vec::with_capacity(faces.len());
        let mut points = [[0.0f64; 3]; 4];

        for f in faces {
            let nodes = f.nodes();
            mesh.gather_coords(nodes, &mut points[..nodes.len()]);
            let mut c = [0.0f64; 3];
            for p in &points[..nodes.len()] {
                for k in 0..3 {
                    c[k] += p[k];
                }
            }
            let inv = 1.0 / nodes.len() as f64;
            centroids.extend(c.iter().map(|v| v * inv));
            pids.push(mesh.solid(f.element as usize).pid);
            areas.push(crate::element::shell::face_integral(f.kind(), &points[..nodes.len()]).area);
        }

        // 稜線 → その稜線を持つ面。外表面は多様体なので、ふつう 1 稜線に 2 面
        let mut edges: BTreeMap<[NodeIndex; 2], Vec<u32>> = BTreeMap::new();
        for (i, f) in faces.iter().enumerate() {
            let nodes = f.nodes();
            for k in 0..nodes.len() {
                let a = nodes[k];
                let b = nodes[(k + 1) % nodes.len()];
                edges
                    .entry([a.min(b), a.max(b)])
                    .or_default()
                    .push(i as u32);
            }
        }
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); faces.len()];
        for sharing in edges.values() {
            for (i, &a) in sharing.iter().enumerate() {
                for &b in sharing.iter().skip(i + 1) {
                    lists[a as usize].push(b);
                    lists[b as usize].push(a);
                }
            }
        }
        let mut offsets = Vec::with_capacity(faces.len() + 1);
        let mut neighbours = Vec::new();
        offsets.push(0);
        for list in &mut lists {
            list.sort_unstable();
            list.dedup();
            neighbours.extend_from_slice(list);
            offsets.push(neighbours.len() as u32);
        }

        Self {
            centroids,
            pids,
            areas,
            offsets,
            neighbours,
        }
    }

    /// 面の枚数。
    pub fn len(&self) -> usize {
        self.pids.len()
    }

    /// 面が 1 枚も無いか。
    pub fn is_empty(&self) -> bool {
        self.pids.is_empty()
    }

    /// 面の重心 [m]。
    pub fn centroid(&self, face: u32) -> [f64; 3] {
        let i = face as usize * 3;
        [
            self.centroids[i],
            self.centroids[i + 1],
            self.centroids[i + 2],
        ]
    }

    /// 面を持つソリッド要素の PID。
    pub fn pid(&self, face: u32) -> PropId {
        self.pids[face as usize]
    }

    /// 面の面積 [m²]。
    pub fn area(&self, face: u32) -> f64 {
        self.areas[face as usize]
    }

    /// 面の集合の総面積 [m²]。
    pub fn total_area(&self, faces: &[u32]) -> f64 {
        faces.iter().map(|&f| self.area(f)).sum()
    }

    /// 稜線を共有する面。
    pub fn neighbours(&self, face: u32) -> &[u32] {
        let lo = self.offsets[face as usize] as usize;
        let hi = self.offsets[face as usize + 1] as usize;
        &self.neighbours[lo..hi]
    }

    /// 指定した点にいちばん近い面。`pid` を与えるとその領域の面だけから探す。
    ///
    /// 同じ距離の面が複数あるときは通し番号の小さいほうを返す（決定性のため）。
    pub fn nearest(&self, point: [f64; 3], pid: Option<PropId>) -> Option<u32> {
        let mut best: Option<(f64, u32)> = None;
        for face in 0..self.len() as u32 {
            if pid.is_some_and(|p| self.pid(face) != p) {
                continue;
            }
            let d = squared_distance(self.centroid(face), point);
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, face));
            }
        }
        best.map(|(_, face)| face)
    }

    /// クリック点を中心に、半径の中で**つながっている**面を選ぶ。
    ///
    /// クリックした面（＝ `point` にいちばん近い面）は半径に関わらず必ず含める。
    /// 半径を小さくしても「1 枚も選ばれない」ことが起きないようにするため。
    pub fn select(&self, point: [f64; 3], radius: f64, pid: Option<PropId>) -> Vec<u32> {
        let Some(seed) = self.nearest(point, pid) else {
            return Vec::new();
        };
        let r2 = radius * radius;
        let mut chosen = vec![false; self.len()];
        let mut queue = std::collections::VecDeque::new();
        chosen[seed as usize] = true;
        queue.push_back(seed);

        while let Some(face) = queue.pop_front() {
            for &next in self.neighbours(face) {
                if chosen[next as usize] {
                    continue;
                }
                if pid.is_some_and(|p| self.pid(next) != p) {
                    continue;
                }
                if squared_distance(self.centroid(next), point) > r2 {
                    continue;
                }
                chosen[next as usize] = true;
                queue.push_back(next);
            }
        }

        (0..self.len() as u32)
            .filter(|&f| chosen[f as usize])
            .collect()
    }
}

fn squared_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).powi(2)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::nastran::read_nastran_from;
    use crate::mesh::topology::extract_surface;
    use crate::units::LengthUnit;
    use std::path::Path;

    /// 1 辺 2 の立方体を 2 分割 × 2 分割 × 2 分割……ではなく、
    /// 面ごとの選択を追える最小の箱（六面体 1 個）。
    fn unit_box() -> Mesh {
        let text = "\
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,1.0,1.0,0.0
GRID,4,,0.0,1.0,0.0
GRID,5,,0.0,0.0,1.0
GRID,6,,1.0,0.0,1.0
GRID,7,,1.0,1.0,1.0
GRID,8,,0.0,1.0,1.0
CHEXA,10,1,1,2,3,4,5,6,7,8
PSOLID,1,1
";
        read_nastran_from(text.as_bytes(), Path::new("box.bdf"), LengthUnit::Meter)
            .unwrap()
            .0
    }

    #[test]
    fn a_click_selects_the_face_it_landed_on() {
        let mesh = unit_box();
        let surface = extract_surface(&mesh).unwrap();
        let picker = SurfacePicker::build(&mesh, &surface);
        assert_eq!(picker.len(), 6, "立方体の外表面は 6 面");

        // 上面の中央をクリック。半径ゼロでも 1 枚は選ばれる
        let faces = picker.select([0.5, 0.5, 1.0], 0.0, None);
        assert_eq!(faces.len(), 1);
        assert!((picker.centroid(faces[0])[2] - 1.0).abs() < 1e-12);
        assert!((picker.total_area(&faces) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_patch_grows_along_the_surface() {
        let mesh = unit_box();
        let surface = extract_surface(&mesh).unwrap();
        let picker = SurfacePicker::build(&mesh, &surface);

        // 半径を上げると、稜線でつながっている隣の面まで広がる
        let faces = picker.select([0.5, 0.5, 1.0], 0.8, None);
        assert!(faces.len() > 1, "隣の面まで広がるはず");
        // 反対側（底面）は重心までの距離が 1.0 を超えるので入らない
        assert!(
            faces
                .iter()
                .all(|&f| picker.centroid(f)[2] > 0.0 || picker.centroid(f)[2] == 0.5),
            "底面が入っている"
        );
    }

    #[test]
    fn the_result_is_sorted_and_deterministic() {
        let mesh = unit_box();
        let surface = extract_surface(&mesh).unwrap();
        let picker = SurfacePicker::build(&mesh, &surface);
        let a = picker.select([0.5, 0.5, 1.0], 0.9, None);
        let b = picker.select([0.5, 0.5, 1.0], 0.9, None);
        assert_eq!(a, b);
        assert!(a.windows(2).all(|w| w[0] < w[1]), "昇順・重複なしのこと");
    }
}
