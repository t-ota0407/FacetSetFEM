//! 外表面から電極面を選ぶ（SPEC §2.4 / §6.2 / §11.1）。
//!
//! シェル要素（CTRIA3 / CQUAD4）を持たないモデルでも電極を定義できるようにする。
//! 経皮電気刺激で使う人体モデルは**ソリッド要素だけで配られるのが普通**で、
//! シェルを前提にすると電極をひとつも置けない。外表面はソリッドの面から
//! 抽出できている（[`extract_surface`](super::topology::extract_surface)）ので、
//! そこから直接選べばよい。
//!
//! 選び方は「3D ビューでクリックした点」＋「形」＋「PID の絞り込み」。形は円（半径）か、
//! 角を丸められる矩形（幅・高さ・角の半径・向き）。
//!
//! **球に入っているかだけで決めてはいけない。** 実際の頭部モデルでは、首の
//! 切断面・耳の裏・折り返した皮膚など、**空間的には近いのに体表面としては
//! つながっていない面**が球に入る。クリックした面から**稜線でつながっている面だけ**を
//! 幅優先でたどることで、皮膚の上を這うように広がる選択になる。
//!
//! 返す面は通し番号の昇順で、入力が同じなら常に同じ集合・同じ並びになる
//! （CLAUDE.md 設計原則 1）。

use crate::element::shell::face_normal;
use crate::error::{ErrorKind, GfemError, Result};
use crate::geom::{cross, dot, norm, scale, sub};
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
    /// 面の外向き単位法線（3 個ずつ）。矩形パッチの接平面を決めるのに使う
    normals: Vec<f64>,
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
        let mut normals = Vec::with_capacity(faces.len() * 3);
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
            // 外表面の面は外向き法線になる順に並んでいる（topology の SurfaceFace）
            normals.extend(face_normal(f.kind(), &points[..nodes.len()]));
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
            normals,
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

    /// 面の外向き単位法線。
    pub fn normal(&self, face: u32) -> [f64; 3] {
        let i = face as usize * 3;
        [self.normals[i], self.normals[i + 1], self.normals[i + 2]]
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

    /// 点の周り（半径 `radius` の中でつながっている面）の外向き法線の、面積重みの平均。
    ///
    /// 足す順は面の通し番号の昇順（[`Self::select`] の戻り値の並び）で、決定的。
    ///
    /// **クリックした面と逆側・真横を向いた面は足さない。** 体の縁（角や稜）の近くを
    /// クリックすると、距離では回り込んだ先の面も入り、平均の法線が傾いて矩形が
    /// 縁の向こうへ回り込む。
    fn mean_normal(&self, point: [f64; 3], radius: f64, pid: Option<PropId>) -> Option<[f64; 3]> {
        let seed = self.nearest(point, pid)?;
        let reference = self.normal(seed);
        let mut sum = [0.0f64; 3];
        for f in self.select(point, radius, pid) {
            if dot(self.normal(f), reference) <= 0.0 {
                continue;
            }
            let n = scale(self.normal(f), self.area(f));
            for k in 0..3 {
                sum[k] += n[k];
            }
        }
        let len = norm(sum);
        (len > 0.0).then(|| scale(sum, 1.0 / len))
    }

    /// 円パッチの接平面の座標系（輪郭を描くためだけに使う。選択には使わない）。
    ///
    /// 円の選択は中心からの 3 次元の距離で決まり、向きを持たない。`u` は法線と
    /// いちばん直交に近い座標軸を接平面に投影したもので、決定的に決まる。
    pub fn circle_frame(
        &self,
        point: [f64; 3],
        radius: f64,
        pid: Option<PropId>,
    ) -> Option<PatchFrame> {
        let n = self.mean_normal(point, 0.5 * radius, pid)?;
        let k = (0..3)
            .min_by(|&a, &b| n[a].abs().total_cmp(&n[b].abs()))
            .unwrap_or(0);
        let mut e = [0.0f64; 3];
        e[k] = 1.0;
        let p = sub(e, scale(n, dot(e, n)));
        let u = scale(p, 1.0 / norm(p));
        Some(PatchFrame {
            origin: point,
            u,
            v: cross(n, u),
            normal: n,
        })
    }

    /// 矩形パッチの接平面の座標系（SPEC §11.1）。
    ///
    /// - 法線はクリック点の周り（短い辺の半分の距離の中でつながっている面）の外向き法線の
    ///   面積重み平均。面 1 枚の法線はメッシュの凹凸で揺れるため
    /// - `u`（幅の辺の向き）は `axis` を接平面に投影し、法線まわりに `angle` 度回したもの。
    ///   外から見て反時計回りが正。`v = n × u`（高さの辺の向き）
    ///
    /// クリック点の近くに面が無ければ `Ok(None)`。
    pub fn rect_frame(
        &self,
        point: [f64; 3],
        rect: &RectShape,
        pid: Option<PropId>,
    ) -> Result<Option<PatchFrame>> {
        rect.check()?;
        let Some(n) = self.mean_normal(point, 0.5 * rect.width.min(rect.height), pid) else {
            return Ok(None);
        };
        let a = rect.axis;
        let p = sub(a, scale(n, dot(a, n)));
        // 投影がほとんど残らない＝向きの軸が面の法線とほぼ平行（3° 以内）。
        // このあたりでは向きがわずかな揺れで大きく変わるので、黙って決めない
        if norm(p) < AXIS_PARALLEL_LIMIT * norm(a) {
            return Err(GfemError::new(
                ErrorKind::Case,
                crate::tr!(
                    "The orientation axis {:?} of the rectangular patch is almost parallel to the \
                     normal [{:.3}, {:.3}, {:.3}] of the face it is placed on, so the edge direction \
                     cannot be determined",
                    "矩形パッチの向きの軸 {:?} が、置こうとした面の法線 [{:.3}, {:.3}, {:.3}] と\
                     ほぼ平行で、辺の向きが決まりません",
                    a,
                    n[0],
                    n[1],
                    n[2]
                ),
            )
            .with_hint(crate::tr!(
                "In the GUI you are looking at the body surface almost edge-on. Rotate the view so \
                 that you face the surface where the electrode goes, then click again (in a case \
                 file, give an axis that runs along the body surface)",
                "GUI では体表面をほぼ真横から見ています。電極を置く面を正面から見るように\
                 回してからクリックしてください（ケースファイルでは axis を体表面に沿う向きに\
                 書いてください）"
            )));
        }
        let u0 = scale(p, 1.0 / norm(p));
        let w0 = cross(n, u0);
        let (sin, cos) = rect.angle.to_radians().sin_cos();
        let u = [
            u0[0] * cos + w0[0] * sin,
            u0[1] * cos + w0[1] * sin,
            u0[2] * cos + w0[2] * sin,
        ];
        Ok(Some(PatchFrame {
            origin: point,
            u,
            v: cross(n, u),
            normal: n,
        }))
    }

    /// クリック点を中心に、矩形（角丸可）に入っていて**つながっている**面を選ぶ（SPEC §11.1）。
    ///
    /// 面の重心を接平面に投影して矩形に入るかで決める。円と同じく、クリックした面は
    /// 大きさに関わらず必ず含め、稜線でつながっている面だけを幅優先でたどる。
    /// さらに**法線が接平面の法線と逆を向く面は入れない**——耳の裏のように、
    /// 回り込んだ先の面が投影すると矩形に入ってしまうのを防ぐため。
    pub fn select_rect(
        &self,
        point: [f64; 3],
        rect: &RectShape,
        pid: Option<PropId>,
    ) -> Result<Vec<u32>> {
        let Some(frame) = self.rect_frame(point, rect, pid)? else {
            return Ok(Vec::new());
        };
        let Some(seed) = self.nearest(point, pid) else {
            return Ok(Vec::new());
        };
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
                if dot(self.normal(next), frame.normal) <= 0.0 {
                    continue;
                }
                let (x, y) = frame.local(self.centroid(next));
                if !rect.contains(x, y) {
                    continue;
                }
                chosen[next as usize] = true;
                queue.push_back(next);
            }
        }

        Ok((0..self.len() as u32)
            .filter(|&f| chosen[f as usize])
            .collect())
    }
}

/// 向きの軸が法線とほぼ平行とみなす割合（接平面への投影の長さ / 軸の長さ。sin 3° ≈ 0.052）。
const AXIS_PARALLEL_LIMIT: f64 = 0.05;

/// 円の輪郭を何分割で描くか。
const CIRCLE_SEGMENTS: usize = 64;

/// 角丸の 1 つの角を何分割で描くか。
const CORNER_SEGMENTS: usize = 8;

/// 矩形パッチの形（SPEC §11.1）。長さは m、角度は度。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RectShape {
    /// 幅 [m]。`axis` の向きの辺の長さ
    pub width: f64,
    /// 高さ [m]
    pub height: f64,
    /// 角の半径 [m]。0 なら尖った角。短い辺の半分まで
    ///
    /// 尖った角には電流が集中し、|J| の最大値が実物より大きく出る。市販の電極は角が
    /// 丸いことが多いので、ホットスポットを評価するなら実物に合わせる。
    pub corner_radius: f64,
    /// 幅の辺の向き（世界座標）。接平面に投影して使うので、長さは問わない
    ///
    /// GUI はクリックしたときの画面の横方向を入れる（「画面に対してまっすぐ」）。
    /// カメラには依存しないので、ケースファイルから同じ向きを再現できる。
    pub axis: [f64; 3],
    /// `axis` から法線まわりに回す角度 [度]。外から見て反時計回りが正
    pub angle: f64,
}

impl RectShape {
    /// 寸法と向きが使える値か確かめる。
    pub fn check(&self) -> Result<()> {
        let bad =
            |what: String, hint: String| Err(GfemError::new(ErrorKind::Case, what).with_hint(hint));
        if !(self.width.is_finite()
            && self.width > 0.0
            && self.height.is_finite()
            && self.height > 0.0)
        {
            return bad(
                crate::tr!(
                    "Invalid rectangular patch size: width {} m × height {} m",
                    "矩形パッチの寸法が不正です: 幅 {} m × 高さ {} m",
                    self.width,
                    self.height
                ),
                crate::tr!(
                    "Give the width and height as positive finite values (in m; 20 mm is 0.02)",
                    "幅と高さは正の有限値で指定してください（単位は m。20 mm なら 0.02）"
                ),
            );
        }
        let limit = 0.5 * self.width.min(self.height);
        if !(self.corner_radius.is_finite()
            && self.corner_radius >= 0.0
            && self.corner_radius <= limit)
        {
            return bad(
                crate::tr!(
                    "The corner radius {} m of the rectangular patch is outside the allowed range \
                     0–{} m (half the shorter side)",
                    "矩形パッチの角の半径 {} m が、使える範囲 0〜{} m（短い辺の半分）を外れています",
                    self.corner_radius,
                    limit
                ),
                crate::tr!(
                    "Give a corner radius between 0 (sharp corners) and half the shorter side",
                    "角の半径は 0（尖った角）から短い辺の半分までで指定してください"
                ),
            );
        }
        if !self.axis.iter().all(|v| v.is_finite()) || norm(self.axis) == 0.0 {
            return bad(
                crate::tr!(
                    "Invalid orientation axis of the rectangular patch: {:?}",
                    "矩形パッチの向きの軸が不正です: {:?}",
                    self.axis
                ),
                crate::tr!(
                    "Give axis as a nonzero 3D vector along the width edge (e.g. [1, 0, 0])",
                    "axis は幅の辺の向きを表す 0 でない 3 次元ベクトルで指定してください（例: [1, 0, 0]）"
                ),
            );
        }
        if !self.angle.is_finite() {
            return bad(
                crate::tr!(
                    "The angle of the rectangular patch is not a number: {}",
                    "矩形パッチの角度が数値になっていません: {}",
                    self.angle
                ),
                crate::tr!(
                    "Give angle in degrees (0 if omitted)",
                    "angle は度で指定してください（省略すると 0）"
                ),
            );
        }
        Ok(())
    }

    /// 指定どおりの面積 [m²]（角丸のぶんを引いた値）。
    pub fn area(&self) -> f64 {
        let r = self.corner_radius;
        self.width * self.height - (4.0 - std::f64::consts::PI) * r * r
    }

    /// 接平面上の点 (x, y)（中心が原点、x が幅の向き）が形に入るか。
    pub fn contains(&self, x: f64, y: f64) -> bool {
        let (a, b, r) = (0.5 * self.width, 0.5 * self.height, self.corner_radius);
        let (ax, ay) = (x.abs(), y.abs());
        if ax > a || ay > b {
            return false;
        }
        let (dx, dy) = (ax - (a - r), ay - (b - r));
        !(dx > 0.0 && dy > 0.0) || dx * dx + dy * dy <= r * r
    }

    /// 接平面上の輪郭（反時計回り。最後の点と最初の点は結ばれていない）。
    fn outline_2d(&self) -> Vec<[f64; 2]> {
        let (a, b, r) = (0.5 * self.width, 0.5 * self.height, self.corner_radius);
        let corners = [
            (a - r, b - r),
            (r - a, b - r),
            (r - a, r - b),
            (a - r, r - b),
        ];
        let mut out = Vec::new();
        for (q, &(cx, cy)) in corners.iter().enumerate() {
            if r == 0.0 {
                out.push([cx, cy]);
                continue;
            }
            for s in 0..=CORNER_SEGMENTS {
                let t =
                    (q as f64 + s as f64 / CORNER_SEGMENTS as f64) * std::f64::consts::FRAC_PI_2;
                out.push([cx + r * t.cos(), cy + r * t.sin()]);
            }
        }
        out
    }
}

/// パッチを置く接平面の座標系。`u`・`v`・`normal` は正規直交で右手系（`v = normal × u`）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatchFrame {
    /// 原点 [m]（クリックした点）
    pub origin: [f64; 3],
    /// 幅の向き
    pub u: [f64; 3],
    /// 高さの向き
    pub v: [f64; 3],
    /// 外向きの法線
    pub normal: [f64; 3],
}

impl PatchFrame {
    /// 点を接平面に投影した座標 (x, y)。
    pub fn local(&self, p: [f64; 3]) -> (f64, f64) {
        let d = sub(p, self.origin);
        (dot(d, self.u), dot(d, self.v))
    }

    /// 接平面上の (x, y) を世界座標に戻す。
    pub fn world(&self, x: f64, y: f64) -> [f64; 3] {
        let (o, u, v) = (self.origin, self.u, self.v);
        [
            o[0] + u[0] * x + v[0] * y,
            o[1] + u[1] * x + v[1] * y,
            o[2] + u[2] * x + v[2] * y,
        ]
    }

    /// 矩形の輪郭（世界座標。最後の点と最初の点は結ばれていない）。表示用。
    pub fn rect_outline(&self, rect: &RectShape) -> Vec<[f64; 3]> {
        rect.outline_2d()
            .into_iter()
            .map(|[x, y]| self.world(x, y))
            .collect()
    }

    /// 円の輪郭（世界座標。最後の点と最初の点は結ばれていない）。表示用。
    pub fn circle_outline(&self, radius: f64) -> Vec<[f64; 3]> {
        (0..CIRCLE_SEGMENTS)
            .map(|s| {
                let t = s as f64 / CIRCLE_SEGMENTS as f64 * std::f64::consts::TAU;
                self.world(radius * t.cos(), radius * t.sin())
            })
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
    fn the_outline_runs_along_the_edge_of_the_shape() {
        // 輪郭の点はどれも形の縁に載る（内側判定の境界上）。角丸なら角ごとに弧を描く
        let rect = RectShape {
            width: 0.02,
            height: 0.01,
            corner_radius: 0.002,
            axis: [1.0, 0.0, 0.0],
            angle: 0.0,
        };
        let outline = rect.outline_2d();
        assert_eq!(outline.len(), 4 * (CORNER_SEGMENTS + 1));
        for [x, y] in outline {
            assert!(
                rect.contains(x * (1.0 - 1e-9), y * (1.0 - 1e-9)),
                "({x}, {y}) が外"
            );
            assert!(
                !rect.contains(x * 1.001, y * 1.001),
                "({x}, {y}) が縁より内側"
            );
        }
        let sharp = RectShape {
            corner_radius: 0.0,
            ..rect
        };
        assert_eq!(sharp.outline_2d().len(), 4);
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
