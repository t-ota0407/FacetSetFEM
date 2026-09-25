//! ストリームライン（SPEC §6.3）。
//!
//! E も J も**要素内で一定**（Hex8 / Penta6 では要素中心の代表値。SPEC §5.1）なので、
//! 流線は要素ごとに直線になる。
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
//! 入ってきた面の上から再開すると、丸め誤差でその面を「出口」と判定して
//! 進めなくなることがある。そこで**歩幅がゼロのときだけ**入ってきた面を飛ばす。
//! 候補から無条件に外してはいけない——そこが本当の出口のとき、より遠い面まで
//! 突き抜けてメッシュの外に点が出る（M6 まではこの壊れ方をしていた。SPEC §6.3）。
//! 稜線や頂点に載って歩幅が暴れたときは `MAX_STEP_IN_SIZES` で打ち切る。
//!
//! # 要素種
//!
//! Hex8 / Penta6 を含むメッシュは[面共役な四面体分割](crate::mesh::TetView)を通す
//! （SPEC §5.1 / §8 M7）。四角形面が両側から同じ対角線で割られるので、
//! 面を越えて隣を辿れる。始点の探索は[一様格子](crate::mesh::CellGrid)で候補を絞る。

use crate::element::tet4;
use crate::error::{GfemError, Result};
use crate::mesh::Mesh;
use crate::mesh::spatial::CellGrid;
use crate::mesh::tetview::TetView;
use crate::mesh::topology::{FaceAdjacency, build_face_adjacency};

/// 1 回の前進が四面体の大きさの何倍までなら妥当とみなすか。
///
/// 四面体の中の点から直線で出るまでの距離は、厳密には対角長を超えない。
/// それを大きく超える値が出たら、幾何が退化していて `s = −λ/rate` が
/// 意味を失っている（稜線や頂点にちょうど載ると、λ も rate もほぼゼロになり、
/// 商が暴れる。実測で 10²⁹ m 先へ飛んだ）。
///
/// **平行に近い面を候補から外す**という直し方は誤り。それが本当の出口である
/// ことがあり、外すと別の面まで突き抜けてメッシュの外へ出てしまう
/// （電極の縁で電流が表面に平行に流れるときに実際に起きた）。
/// 候補は減らさず、**歩幅そのものを見張る**。
const MAX_STEP_IN_SIZES: f64 = 4.0;

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
    /// 幾何的に追跡を続けられなくなった
    ///
    /// 流線が要素の稜線や頂点にちょうど載り続けると、出口面が数値的に決まらなくなる。
    /// 途中で打ち切って、そこまでの経路を返す。
    Degenerate,
}

/// 1 本の流線。
#[derive(Debug, Clone)]
pub struct Streamline {
    /// 折れ線の頂点 [m]。3 成分ずつ
    pub points: Vec<f64>,
    /// 各区間が属する**元のソリッド要素**の通し番号（頂点数 − 1 個）
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

/// 四面体分割・面隣接・空間分割をまとめて持つ、流線を引くための下ごしらえ（SPEC §8 M7）。
///
/// この 3 つは**メッシュだけで決まる**ので、流線を引くたびに作り直すのは無駄になる。
/// 1 度作って使い回すこと。
///
/// 場（E や J）は**元のソリッド要素**ごとの値をそのまま渡す。
/// 分割してできた四面体は親の値を引き継ぐので、
/// 流線は元の要素の中では直線のまま進む（SPEC §5.1）。
#[derive(Debug)]
pub struct Navigator {
    tets: TetView,
    adjacency: FaceAdjacency,
    grid: CellGrid,
}

impl Navigator {
    /// メッシュから作る。
    pub fn build(mesh: &Mesh) -> Result<Self> {
        let tets = TetView::build(mesh)?;
        let adjacency = build_face_adjacency(&tets)?;
        let grid = CellGrid::build(mesh, &tets);
        Ok(Self {
            tets,
            adjacency,
            grid,
        })
    }

    /// 面共役な四面体分割。
    pub fn tets(&self) -> &TetView {
        &self.tets
    }

    /// 四面体の面の隣接関係。
    pub fn adjacency(&self) -> &FaceAdjacency {
        &self.adjacency
    }

    /// 点 `x` を含む四面体を探す。メッシュの外なら `None`。
    ///
    /// 空間分割で候補を絞ってから体積座標で判定する。候補は四面体の通し番号の
    /// 昇順に並んでいるので、面の上に載った点でも常に同じ答えになる。
    pub fn locate_tet(&self, mesh: &Mesh, x: [f64; 3]) -> Option<usize> {
        for &t in self.grid.candidates(x) {
            let p = self.tets.coords(mesh, t as usize);
            let shape = tet4::tet4_shape(&p);
            let lambda = barycentric(&shape, p[0], x);
            // 面の上に載っている場合も含めたいので、わずかな負を許す
            if lambda.iter().all(|&l| l >= -1.0e-12) {
                return Some(t as usize);
            }
        }
        None
    }

    /// 点 `x` を含む**元のソリッド要素**の通し番号。
    pub fn locate_element(&self, mesh: &Mesh, x: [f64; 3]) -> Option<usize> {
        self.locate_tet(mesh, x)
            .map(|t| self.tets.parent(t) as usize)
    }

    /// 点 `x` を含む四面体のうち、**場の向きに進んだ先が中に入る**ものを選ぶ。
    ///
    /// 面・稜線・頂点にちょうど載った始点では、その点を含む四面体が複数ある。
    /// 単に最初のものを採ると、場がそこから外向きの四面体を選んでしまい、
    /// 1 歩も進めないまま隣を巡り続けることがある。GUI が始点を格子状に置くと
    /// 節点面にぴったり載るので、これは実際に起きる。
    fn locate_tet_towards(
        &self,
        mesh: &Mesh,
        x: [f64; 3],
        field: &[f64],
        sign: f64,
    ) -> Option<usize> {
        let mut fallback = None;
        for &t in self.grid.candidates(x) {
            let t = t as usize;
            let p = self.tets.coords(mesh, t);
            let shape = tet4::tet4_shape(&p);
            let lambda = barycentric(&shape, p[0], x);
            if !lambda.iter().all(|&l| l >= -1.0e-12) {
                continue;
            }
            fallback.get_or_insert(t);

            // 場の向きにごく僅かだけ進めた点が、まだこの四面体の中にあるか
            let parent = self.tets.parent(t) as usize;
            let v = [
                sign * field[3 * parent],
                sign * field[3 * parent + 1],
                sign * field[3 * parent + 2],
            ];
            let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if speed <= 0.0 {
                continue;
            }
            let step = 1.0e-6 * tet_size(&p) / speed;
            let ahead = [x[0] + step * v[0], x[1] + step * v[1], x[2] + step * v[2]];
            if barycentric(&shape, p[0], ahead)
                .iter()
                .all(|&l| l >= -1.0e-12)
            {
                return Some(t);
            }
        }
        // 場がゼロ、または境界から外へ出ていく向き。含む四面体をそのまま返す
        fallback
    }

    /// 1 本の流線を追跡する。
    ///
    /// `field` は**元のソリッド要素**ごとに 3 成分並んだ配列（E や J）。
    /// 始点がメッシュの外なら `None` を返す。
    pub fn trace(
        &self,
        mesh: &Mesh,
        field: &[f64],
        start: [f64; 3],
        options: &TraceOptions,
    ) -> Result<Option<Streamline>> {
        let n_elements = mesh.n_solid_elements();
        if field.len() != 3 * n_elements {
            return Err(GfemError::internal(format!(
                "要素値の長さが合いません: {} 個ですが 3 × {n_elements} であるべきです",
                field.len()
            )));
        }
        let sign = options.direction.sign();
        // 始点が面や稜線にちょうど載っていると、その点を含む四面体が何個もある。
        // 「場の向きに進んだ先が中に入る」ものを選ばないと、いきなり外へ出てしまう
        let Some(mut tet) = self.locate_tet_towards(mesh, start, field, sign) else {
            return Ok(None);
        };
        let mut x = start;
        let mut points: Vec<f64> = x.to_vec();
        let mut elements: Vec<u32> = Vec::new();
        let mut length = 0.0;
        let mut entry_face: Option<usize> = None;
        let mut stop_reason = StopReason::MaxSteps;

        for _ in 0..options.max_steps {
            let parent = self.tets.parent(tet) as usize;
            let p = self.tets.coords(mesh, tet);
            let shape = tet4::tet4_shape(&p);
            let v = [
                sign * field[3 * parent],
                sign * field[3 * parent + 1],
                sign * field[3 * parent + 2],
            ];
            let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if speed <= 0.0 {
                stop_reason = StopReason::ZeroField;
                break;
            }

            // λ_i(x + s v) = λ_i(x) + s (∇N_i · v) が 0 になる最小の s
            let lambda = barycentric(&shape, p[0], x);
            let size = tet_size(&p);
            let mut best_s = f64::INFINITY;
            let mut exit_face = usize::MAX;
            for (i, &l) in lambda.iter().enumerate() {
                let g = shape.grad[i];
                let rate = g[0] * v[0] + g[1] * v[1] + g[2] * v[2];
                if rate >= 0.0 {
                    continue; // その面からは遠ざかっている
                }
                // 面の上に載っている点は丸めで λ がわずかに負になりうる
                let s = -l.max(0.0) / rate;
                // 入ってきた面の上に居るあいだは、そこを出口にしても進まない。
                // ただし**候補から外してしまってはいけない**——本当にそこが最も近い
                // 出口のときに、より遠い面まで突き抜けてメッシュの外へ出てしまう
                // （電極の縁で電流が表面に平行に流れる場所で実際に起きた）
                if Some(i) == entry_face && s * speed <= 1.0e-9 * size {
                    continue;
                }
                if s >= 0.0 && s < best_s {
                    best_s = s;
                    exit_face = i;
                }
            }
            let step_length = best_s * speed;
            if !best_s.is_finite() || step_length > MAX_STEP_IN_SIZES * size {
                // 稜線や頂点に載り続けて出口が決まらない。そこまでを返す
                stop_reason = StopReason::Degenerate;
                break;
            }

            if length + step_length >= options.max_length {
                let s = (options.max_length - length) / speed;
                x = [x[0] + s * v[0], x[1] + s * v[1], x[2] + s * v[2]];
                points.extend_from_slice(&x);
                elements.push(parent as u32);
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
            elements.push(parent as u32);
            length += step_length;

            match self.adjacency.neighbour(tet, exit_face) {
                Some((next, next_face)) => {
                    tet = next as usize;
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
    pub fn trace_many(
        &self,
        mesh: &Mesh,
        field: &[f64],
        seeds: &[[f64; 3]],
        options: &TraceOptions,
    ) -> Result<Vec<Streamline>> {
        let mut out = Vec::with_capacity(seeds.len());
        for &seed in seeds {
            if let Some(line) = self.trace(mesh, field, seed, options)? {
                out.push(line);
            }
        }
        Ok(out)
    }
}

/// 四面体の代表寸法（最大辺長）[m]。
fn tet_size(p: &[[f64; 3]; 4]) -> f64 {
    let mut worst: f64 = 0.0;
    for i in 0..4 {
        for j in (i + 1)..4 {
            let d = [p[j][0] - p[i][0], p[j][1] - p[i][1], p[j][2] - p[i][2]];
            worst = worst.max((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
        }
    }
    worst
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::nastran::read_nastran_from;
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
        let nav = Navigator::build(&mesh).unwrap();
        assert!(nav.locate_element(&mesh, [0.5, 0.5, 0.5]).is_some());
        assert!(nav.locate_element(&mesh, [5.0, 0.5, 0.5]).is_none());
        assert!(nav.locate_element(&mesh, [0.5, -1.0, 0.5]).is_none());
    }

    #[test]
    fn uniform_field_gives_a_straight_line_to_the_boundary() {
        let mesh = bar(4);
        let nav = Navigator::build(&mesh).unwrap();
        let field = uniform_field(&mesh, [1.0, 0.0, 0.0]);
        let line = nav
            .trace(&mesh, &field, [0.05, 0.5, 0.5], &TraceOptions::default())
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
        let nav = Navigator::build(&mesh).unwrap();
        let field = uniform_field(&mesh, [1.0, 0.0, 0.0]);
        let line = nav
            .trace(
                &mesh,
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
        let nav = Navigator::build(&mesh).unwrap();
        let field = uniform_field(&mesh, [1.0, 0.0, 0.0]);
        let line = nav
            .trace(
                &mesh,
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
        let nav = Navigator::build(&mesh).unwrap();
        let field = uniform_field(&mesh, [1.0, 1.0, 0.0]);
        let line = nav
            .trace(&mesh, &field, [0.1, 0.1, 0.5], &TraceOptions::default())
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
        let nav = Navigator::build(&mesh).unwrap();
        let field = uniform_field(&mesh, [0.0, 0.0, 0.0]);
        let line = nav
            .trace(&mesh, &field, [0.5, 0.5, 0.5], &TraceOptions::default())
            .unwrap()
            .unwrap();
        assert_eq!(line.stop_reason, StopReason::ZeroField);
        assert_eq!(line.n_points(), 1);
    }

    #[test]
    fn seed_outside_the_mesh_is_skipped() {
        let mesh = bar(2);
        let nav = Navigator::build(&mesh).unwrap();
        let field = uniform_field(&mesh, [1.0, 0.0, 0.0]);
        let lines = nav
            .trace_many(
                &mesh,
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
        let nav = Navigator::build(&mesh).unwrap();
        let field = uniform_field(&mesh, [1.0, 0.3, -0.2]);
        let options = TraceOptions::default();
        let a = nav
            .trace(&mesh, &field, [0.05, 0.5, 0.5], &options)
            .unwrap()
            .unwrap();
        let b = nav
            .trace(&mesh, &field, [0.05, 0.5, 0.5], &options)
            .unwrap()
            .unwrap();
        assert_eq!(a.points, b.points);
        assert_eq!(a.elements, b.elements);
    }
}
