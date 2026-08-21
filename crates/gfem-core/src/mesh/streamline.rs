//! ストリームライン（SPEC §6.3）。
//!
//! Tet4 では E も J も**要素内で一定**なので、流線は要素ごとに直線になる。
//! したがって Runge–Kutta のような刻み幅つきの積分は要らず、
//! 「いまの要素をどこで出るか」を解くだけで**厳密に**追跡できる。
//!
//! # 出口面の求め方
//!
//! 体積座標 λ_i は位置の 1 次関数で、∇N_i は要素内で一定だから
//!
//! ```text
//! λ_i(x + s v) = λ_i(x) + s (∇N_i · v)
//! ```
//!
//! となる。∇N_i · v < 0 の面について λ_i が 0 になる s を求め、
//! その最小値が出口。対応する面（節点 i の対面）を越えて隣の要素へ移る。
//!
//! 面の上に載った状態から再開すると丸め誤差で進めなくなることがあるので、
//! **入ってきた面は出口の候補から外す**。これで刻み幅の下限を設けずに済む。

use crate::element::{ElementKind, tet4};
use crate::error::{GfemError, Result};
use crate::mesh::Mesh;
use crate::mesh::topology::FaceAdjacency;

/// 追跡する向き。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceDirection {
    /// 場の向き（電流の流れる向き）
    Forward,
    /// 場と逆向き
    Backward,
}

impl TraceDirection {
    fn sign(self) -> f64 {
        match self {
            TraceDirection::Forward => 1.0,
            TraceDirection::Backward => -1.0,
        }
    }
}

/// 追跡を打ち切った理由。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// 外表面に到達した
    Boundary,
    /// 指定した長さに達した
    MaxLength,
    /// 指定した要素数に達した
    MaxSteps,
    /// 場がゼロになった（電流が流れていない領域）
    ZeroField,
}

/// 1 本の流線。
#[derive(Debug, Clone)]
pub struct Streamline {
    /// 折れ線の頂点 [m]。3 成分ずつ
    pub points: Vec<f64>,
    /// 各区間が属する要素の通し番号（頂点数 − 1 個）
    pub elements: Vec<u32>,
    /// 総延長 [m]
    pub length: f64,
    /// 打ち切った理由
    pub stop_reason: StopReason,
}

impl Streamline {
    /// 頂点数。
    pub fn n_points(&self) -> usize {
        self.points.len() / 3
    }
}

/// 追跡の設定。
#[derive(Debug, Clone, Copy)]
pub struct TraceOptions {
    /// 通過できる要素数の上限
    pub max_steps: usize,
    /// 総延長の上限 [m]
    pub max_length: f64,
    /// 向き
    pub direction: TraceDirection,
}

impl Default for TraceOptions {
    fn default() -> Self {
        Self {
            max_steps: 100_000,
            max_length: f64::INFINITY,
            direction: TraceDirection::Forward,
        }
    }
}

/// 点 `x` を含む要素を探す。
///
/// 全要素を走査する素朴な実装。数万〜数十万要素では十分だが、
/// 1000 万要素では空間分割が要る（M7 で最適化する）。
pub fn locate_element(mesh: &Mesh, x: [f64; 3]) -> Result<Option<usize>> {
    let block = require_tet4(mesh)?;
    for e in 0..block.len() {
        let p = mesh.tet4_coords(block.conn_of(e));
        let shape = tet4::tet4_shape(&p);
        let lambda = barycentric(&shape, p[0], x);
        // 面の上に載っている場合も含めたいので、わずかな負を許す
        if lambda.iter().all(|&l| l >= -1.0e-12) {
            return Ok(Some(e));
        }
    }
    Ok(None)
}

/// 1 本の流線を追跡する。
///
/// `field` は要素ごとに 3 成分並んだ配列（E や J）。
/// 始点がメッシュの外なら `None` を返す。
pub fn trace_streamline(
    mesh: &Mesh,
    adjacency: &FaceAdjacency,
    field: &[f64],
    start: [f64; 3],
    options: &TraceOptions,
) -> Result<Option<Streamline>> {
    let block = require_tet4(mesh)?;
    if field.len() != 3 * block.len() {
        return Err(GfemError::internal(format!(
            "要素値の長さが合いません: {} 個ですが 3 × {} であるべきです",
            field.len(),
            block.len()
        )));
    }
    let Some(mut element) = locate_element(mesh, start)? else {
        return Ok(None);
    };

    let sign = options.direction.sign();
    let mut x = start;
    let mut points: Vec<f64> = x.to_vec();
    let mut elements: Vec<u32> = Vec::new();
    let mut length = 0.0;
    let mut entry_face: Option<usize> = None;
    let mut stop_reason = StopReason::MaxSteps;

    for _ in 0..options.max_steps {
        let p = mesh.tet4_coords(block.conn_of(element));
        let shape = tet4::tet4_shape(&p);
        let v = [
            sign * field[3 * element],
            sign * field[3 * element + 1],
            sign * field[3 * element + 2],
        ];
        let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if speed <= 0.0 {
            stop_reason = StopReason::ZeroField;
            break;
        }

        // λ_i(x + s v) = λ_i(x) + s (∇N_i · v) が 0 になる最小の s
        let lambda = barycentric(&shape, p[0], x);
        let mut best_s = f64::INFINITY;
        let mut exit_face = usize::MAX;
        for (i, &l) in lambda.iter().enumerate() {
            if Some(i) == entry_face {
                continue; // 入ってきた面からは出ない
            }
            let g = shape.grad[i];
            let rate = g[0] * v[0] + g[1] * v[1] + g[2] * v[2];
            if rate >= 0.0 {
                continue;
            }
            let s = -l / rate;
            if s >= 0.0 && s < best_s {
                best_s = s;
                exit_face = i;
            }
        }
        if !best_s.is_finite() {
            // 数値的に出口が見つからない（起こらないはずだが、無限ループは避ける）
            stop_reason = StopReason::ZeroField;
            break;
        }

        let step_length = best_s * speed;
        if length + step_length >= options.max_length {
            let s = (options.max_length - length) / speed;
            x = [x[0] + s * v[0], x[1] + s * v[1], x[2] + s * v[2]];
            points.extend_from_slice(&x);
            elements.push(element as u32);
            length = options.max_length;
            stop_reason = StopReason::MaxLength;
            break;
        }

        x = [
            x[0] + best_s * v[0],
            x[1] + best_s * v[1],
            x[2] + best_s * v[2],
        ];
        points.extend_from_slice(&x);
        elements.push(element as u32);
        length += step_length;

        match adjacency.neighbour(element, exit_face) {
            Some((next, next_face)) => {
                element = next as usize;
                entry_face = Some(next_face as usize);
            }
            None => {
                stop_reason = StopReason::Boundary;
                break;
            }
        }
    }

    Ok(Some(Streamline {
        points,
        elements,
        length,
        stop_reason,
    }))
}

/// 複数の始点から流線を引く。始点がメッシュの外のものは飛ばす。
pub fn trace_streamlines(
    mesh: &Mesh,
    adjacency: &FaceAdjacency,
    field: &[f64],
    seeds: &[[f64; 3]],
    options: &TraceOptions,
) -> Result<Vec<Streamline>> {
    let mut out = Vec::with_capacity(seeds.len());
    for &seed in seeds {
        if let Some(line) = trace_streamline(mesh, adjacency, field, seed, options)? {
            out.push(line);
        }
    }
    Ok(out)
}

/// 位置 `x` の体積座標。
///
/// `λ_i(x) = δ_{i0} + ∇N_i · (x − p0)`。∇N_i が要素内で一定なので閉じた式で書ける。
fn barycentric(shape: &tet4::Tet4Shape, p0: [f64; 3], x: [f64; 3]) -> [f64; 4] {
    let d = [x[0] - p0[0], x[1] - p0[1], x[2] - p0[2]];
    let mut lambda = [0.0f64; 4];
    for (i, l) in lambda.iter_mut().enumerate() {
        let g = shape.grad[i];
        *l = g[0] * d[0] + g[1] * d[1] + g[2] * d[2];
        if i == 0 {
            *l += 1.0;
        }
    }
    lambda
}

fn require_tet4(mesh: &Mesh) -> Result<&crate::mesh::ElementBlock> {
    mesh.block(ElementKind::Tet4).ok_or_else(|| {
        GfemError::unsupported("ストリームラインは 1 次四面体（CTETRA）のみに対応しています")
            .with_hint(
                "要素種が混ざったメッシュでは、隣り合う要素が共有する四角形面で\
                 同じ対角線を選ぶ四面体分割が要ります。M7 で対応します（SPEC §5.1 / §8）",
            )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::nastran::read_nastran_from;
    use crate::mesh::topology::build_face_adjacency;
    use crate::units::LengthUnit;
    use std::path::Path;

    /// nx × 1 × 1 の直方体を Kuhn 分割した棒（x 方向に伸びる）。
    fn bar(nx: usize) -> Mesh {
        let mut deck = String::new();
        let node = |i: usize, j: usize, k: usize| i + (nx + 1) * (j + 2 * k) + 1;
        for k in 0..2 {
            for j in 0..2 {
                for i in 0..=nx {
                    deck.push_str(&format!(
                        "GRID,{},,{},{},{}\n",
                        node(i, j, k),
                        i as f64 / nx as f64,
                        j as f64,
                        k as f64
                    ));
                }
            }
        }
        deck.push_str("PSOLID,1,1\n");
        let mut eid = 1;
        for i in 0..nx {
            let corner: Vec<usize> = (0..8)
                .map(|n| node(i + (n & 1), (n >> 1) & 1, (n >> 2) & 1))
                .collect();
            for t in [
                [0, 1, 3, 7],
                [0, 1, 5, 7],
                [0, 2, 3, 7],
                [0, 2, 6, 7],
                [0, 4, 5, 7],
                [0, 4, 6, 7],
            ] {
                deck.push_str(&format!(
                    "CTETRA,{eid},1,{},{},{},{}\n",
                    corner[t[0]], corner[t[1]], corner[t[2]], corner[t[3]]
                ));
                eid += 1;
            }
        }
        read_nastran_from(deck.as_bytes(), Path::new("<test>"), LengthUnit::Meter)
            .unwrap()
            .0
    }

    /// 全要素に同じベクトルを与える場。
    fn uniform_field(mesh: &Mesh, v: [f64; 3]) -> Vec<f64> {
        let n = mesh.n_solid_elements();
        let mut out = Vec::with_capacity(3 * n);
        for _ in 0..n {
            out.extend_from_slice(&v);
        }
        out
    }

    #[test]
    fn locates_points_inside_and_outside() {
        let mesh = bar(2);
        assert!(locate_element(&mesh, [0.5, 0.5, 0.5]).unwrap().is_some());
        assert!(locate_element(&mesh, [5.0, 0.5, 0.5]).unwrap().is_none());
        assert!(locate_element(&mesh, [0.5, -1.0, 0.5]).unwrap().is_none());
    }

    #[test]
    fn uniform_field_gives_a_straight_line_to_the_boundary() {
        let mesh = bar(4);
        let adjacency = build_face_adjacency(&mesh).unwrap();
        let field = uniform_field(&mesh, [1.0, 0.0, 0.0]);
        let line = trace_streamline(
            &mesh,
            &adjacency,
            &field,
            [0.05, 0.5, 0.5],
            &TraceOptions::default(),
        )
        .unwrap()
        .unwrap();

        assert_eq!(line.stop_reason, StopReason::Boundary);
        // x = 1 の面まで進む
        let last = &line.points[line.points.len() - 3..];
        assert!((last[0] - 1.0).abs() < 1e-12, "終点 {last:?}");
        assert!((line.length - 0.95).abs() < 1e-12, "長さ {}", line.length);
        // y, z は変わらない
        for p in line.points.chunks_exact(3) {
            assert!((p[1] - 0.5).abs() < 1e-12);
            assert!((p[2] - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn backward_tracing_goes_the_other_way() {
        let mesh = bar(4);
        let adjacency = build_face_adjacency(&mesh).unwrap();
        let field = uniform_field(&mesh, [1.0, 0.0, 0.0]);
        let line = trace_streamline(
            &mesh,
            &adjacency,
            &field,
            [0.6, 0.5, 0.5],
            &TraceOptions {
                direction: TraceDirection::Backward,
                ..Default::default()
            },
        )
        .unwrap()
        .unwrap();
        let last = &line.points[line.points.len() - 3..];
        assert!((last[0] - 0.0).abs() < 1e-12, "終点 {last:?}");
        assert!((line.length - 0.6).abs() < 1e-12);
    }

    #[test]
    fn respects_the_length_limit() {
        let mesh = bar(4);
        let adjacency = build_face_adjacency(&mesh).unwrap();
        let field = uniform_field(&mesh, [1.0, 0.0, 0.0]);
        let line = trace_streamline(
            &mesh,
            &adjacency,
            &field,
            [0.05, 0.5, 0.5],
            &TraceOptions {
                max_length: 0.3,
                ..Default::default()
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(line.stop_reason, StopReason::MaxLength);
        assert!((line.length - 0.3).abs() < 1e-12);
        let last = &line.points[line.points.len() - 3..];
        assert!((last[0] - 0.35).abs() < 1e-12, "終点 {last:?}");
    }

    #[test]
    fn diagonal_field_follows_the_direction() {
        let mesh = bar(4);
        let adjacency = build_face_adjacency(&mesh).unwrap();
        let field = uniform_field(&mesh, [1.0, 1.0, 0.0]);
        let line = trace_streamline(
            &mesh,
            &adjacency,
            &field,
            [0.1, 0.1, 0.5],
            &TraceOptions::default(),
        )
        .unwrap()
        .unwrap();
        // y = 1 の面に到達するまでに Δx = Δy = 0.9 進む
        let last = &line.points[line.points.len() - 3..];
        assert!((last[1] - 1.0).abs() < 1e-12, "終点 {last:?}");
        assert!((last[0] - 1.0).abs() < 1e-12, "終点 {last:?}");
        // 折れ線の各点は常に x = y の直線上
        for p in line.points.chunks_exact(3) {
            assert!((p[0] - p[1]).abs() < 1e-12, "{p:?}");
        }
    }

    #[test]
    fn zero_field_stops_immediately() {
        let mesh = bar(2);
        let adjacency = build_face_adjacency(&mesh).unwrap();
        let field = uniform_field(&mesh, [0.0, 0.0, 0.0]);
        let line = trace_streamline(
            &mesh,
            &adjacency,
            &field,
            [0.5, 0.5, 0.5],
            &TraceOptions::default(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(line.stop_reason, StopReason::ZeroField);
        assert_eq!(line.n_points(), 1);
    }

    #[test]
    fn seed_outside_the_mesh_is_skipped() {
        let mesh = bar(2);
        let adjacency = build_face_adjacency(&mesh).unwrap();
        let field = uniform_field(&mesh, [1.0, 0.0, 0.0]);
        let lines = trace_streamlines(
            &mesh,
            &adjacency,
            &field,
            &[[0.5, 0.5, 0.5], [9.0, 9.0, 9.0]],
            &TraceOptions::default(),
        )
        .unwrap();
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn is_reproducible() {
        let mesh = bar(6);
        let adjacency = build_face_adjacency(&mesh).unwrap();
        let field = uniform_field(&mesh, [1.0, 0.3, -0.2]);
        let options = TraceOptions::default();
        let a = trace_streamline(&mesh, &adjacency, &field, [0.05, 0.5, 0.5], &options)
            .unwrap()
            .unwrap();
        let b = trace_streamline(&mesh, &adjacency, &field, [0.05, 0.5, 0.5], &options)
            .unwrap()
            .unwrap();
        assert_eq!(a.points, b.points);
        assert_eq!(a.elements, b.elements);
    }
}
