//! メッシュの内部表現。
//!
//! 座標は**必ず SI（m）に正規化済み**（CLAUDE.md 設計原則 7）。
//! 要素は種類ごとのブロック（SoA）に分けて持つ。こうすると
//!
//! - 要素種ごとの剛性計算を型で分岐できる（SPEC §7「要素種は 1 ファイルで追加」）
//! - 接続配列が固定長ストライドになり、CSR 構築（SPEC §4.2）が単純になる
//!
//! という利点がある。

pub mod idmap;
pub mod isosurface;
pub mod pick;
pub mod quadratic;
pub mod section;
pub mod spatial;
pub mod streamline;
pub mod tetview;
pub mod topology;

use crate::element::{ElementKind, max_edge_length, tet4_signed_volume};
use crate::units::LengthUnit;
use std::path::PathBuf;

pub use idmap::IdMap;
pub use isosurface::{TriangleMesh, iso_surface};
pub use pick::{PatchFrame, RectShape, SurfacePicker};
pub use quadratic::{check_orders, elevate};
pub use section::{Plane, clip, slice};
pub use spatial::CellGrid;
pub use streamline::{Navigator, Streamline, TraceDirection, TraceOptions};
pub use tetview::TetView;
pub use topology::{
    FaceAdjacency, Surface, SurfaceFace, build_face_adjacency, extract_region_boundaries,
    extract_region_surface, extract_surface,
};

/// Nastran の節点 ID。
pub type NodeId = u32;
/// メッシュ内部の節点索引（0 始まり）。
pub type NodeIndex = u32;
/// Nastran の要素 ID。
pub type ElemId = u32;
/// Nastran のプロパティ ID（PSOLID / PSHELL の PID）。領域の単位（SPEC §3.1）。
pub type PropId = u32;

/// 節点集合。
#[derive(Debug, Default, Clone)]
pub struct Nodes {
    ids: Vec<NodeId>,
    /// `3 * len` 要素。`[x0,y0,z0, x1,y1,z1, ...]`、単位は m
    coords: Vec<f64>,
}

impl Nodes {
    /// ID 配列と座標配列（SI, m）から作る。
    pub fn new(ids: Vec<NodeId>, coords: Vec<f64>) -> Self {
        debug_assert_eq!(ids.len() * 3, coords.len());
        Self { ids, coords }
    }

    /// 節点数。
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// 節点が 1 つも無いか。
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Nastran の節点 ID 配列（内部索引順）。
    pub fn ids(&self) -> &[NodeId] {
        &self.ids
    }

    /// 座標配列（m、3 成分ずつ）。
    pub fn coords(&self) -> &[f64] {
        &self.coords
    }

    /// 索引 `i` の節点 ID。
    pub fn id(&self, i: NodeIndex) -> NodeId {
        self.ids[i as usize]
    }

    /// 索引 `i` の座標（m）。
    pub fn coord(&self, i: NodeIndex) -> [f64; 3] {
        let k = i as usize * 3;
        [self.coords[k], self.coords[k + 1], self.coords[k + 2]]
    }

    /// バウンディングボックス（m）。節点が無ければ `None`。
    ///
    /// SPEC §3.2 の「単位の取り違えに気づかせる表示」に使う。
    pub fn bounding_box(&self) -> Option<([f64; 3], [f64; 3])> {
        if self.is_empty() {
            return None;
        }
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in self.coords.chunks_exact(3) {
            for d in 0..3 {
                lo[d] = lo[d].min(p[d]);
                hi[d] = hi[d].max(p[d]);
            }
        }
        Some((lo, hi))
    }
}

/// 同じ種類の要素をまとめたブロック。
#[derive(Debug, Clone)]
pub struct ElementBlock {
    kind: ElementKind,
    ids: Vec<ElemId>,
    pids: Vec<PropId>,
    /// `node_count()` ごとに 1 要素分。値は節点**索引**（ID ではない）
    conn: Vec<NodeIndex>,
}

impl ElementBlock {
    /// 生成。`conn` の長さは `ids.len() * kind.node_count()` でなければならない。
    pub fn new(
        kind: ElementKind,
        ids: Vec<ElemId>,
        pids: Vec<PropId>,
        conn: Vec<NodeIndex>,
    ) -> Self {
        debug_assert_eq!(ids.len(), pids.len());
        debug_assert_eq!(ids.len() * kind.node_count(), conn.len());
        Self {
            kind,
            ids,
            pids,
            conn,
        }
    }

    /// 要素種。
    pub fn kind(&self) -> ElementKind {
        self.kind
    }

    /// 要素数。
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// 空か。
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// 要素 ID の配列。
    pub fn ids(&self) -> &[ElemId] {
        &self.ids
    }

    /// 要素ごとの PID の配列。
    pub fn pids(&self) -> &[PropId] {
        &self.pids
    }

    /// 接続配列全体（節点索引）。
    pub fn conn(&self) -> &[NodeIndex] {
        &self.conn
    }

    /// ブロック内 `i` 番目の要素の接続（節点索引）。
    pub fn conn_of(&self, i: usize) -> &[NodeIndex] {
        let n = self.kind.node_count();
        &self.conn[i * n..(i + 1) * n]
    }
}

/// プロパティ（PSOLID / PSHELL）。
///
/// PID が「領域」の単位になる（SPEC §3.1）。σ はここではなくケース側で与える
/// ——Nastran ファイルは形状と領域分けだけを担うため（SPEC §3.3）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Property {
    /// プロパティ ID
    pub pid: PropId,
    /// 参照している材料 ID（導電率としては使わない。参照整合性のみ）
    pub mid: Option<u32>,
    /// このプロパティのカード名（`PSOLID` / `PSHELL`）
    pub card: &'static str,
}

/// 全ブロックを通しで走査するときの、要素 1 個分の参照。
#[derive(Debug, Clone, Copy)]
pub struct ElementRef<'a> {
    /// ソリッド（またはシェル）全体での通し番号。CSR や要素値配列の添字になる
    pub index: usize,
    /// 要素種
    pub kind: ElementKind,
    /// Nastran の要素 ID
    pub id: ElemId,
    /// プロパティ ID
    pub pid: PropId,
    /// 節点索引
    pub conn: &'a [NodeIndex],
}

/// ブロックの並びを 1 本の要素列として走査する。
///
/// `index` はブロックをまたいだ通し番号で、この順序が**要素値配列の並びを定義する**
/// （VTU のセル順序、要素ごとの E・J の並びは全てこれに従う）。
fn iter_blocks(blocks: &[ElementBlock]) -> impl Iterator<Item = ElementRef<'_>> {
    blocks
        .iter()
        .flat_map(|b| (0..b.len()).map(move |i| (b, i)))
        .enumerate()
        .map(|(index, (b, i))| ElementRef {
            index,
            kind: b.kind(),
            id: b.ids()[i],
            pid: b.pids()[i],
            conn: b.conn_of(i),
        })
}

/// ブロックの並びの中で、通し番号 `index` の要素を引く。範囲外なら `None`。
///
/// ブロック数は要素種の数だけ（高々 3 個）なので、線形に探しても十分速い。
fn element_at(blocks: &[ElementBlock], index: usize) -> Option<ElementRef<'_>> {
    let mut offset = 0;
    for block in blocks {
        if index < offset + block.len() {
            let i = index - offset;
            return Some(ElementRef {
                index,
                kind: block.kind(),
                id: block.ids()[i],
                pid: block.pids()[i],
                conn: block.conn_of(i),
            });
        }
        offset += block.len();
    }
    None
}

/// 有限要素メッシュ。
#[derive(Debug, Clone)]
pub struct Mesh {
    /// 節点（座標は m）
    pub nodes: Nodes,
    /// ソリッド要素ブロック（Tet4 / Penta6 / Hex8）
    pub solids: Vec<ElementBlock>,
    /// シェル要素ブロック（Tria3 / Quad4）。電極面の指定に使う
    pub shells: Vec<ElementBlock>,
    /// PSOLID 一覧（PID 昇順）
    pub solid_properties: Vec<Property>,
    /// PSHELL 一覧（PID 昇順）
    pub shell_properties: Vec<Property>,
    /// 読み込み時に指定された長さ単位（座標は既に m に換算済み。記録用）
    pub length_unit: LengthUnit,
    /// 読み込み元ファイル
    pub source: Option<PathBuf>,
}

impl Mesh {
    /// ソリッド要素の総数。
    pub fn n_solid_elements(&self) -> usize {
        self.solids.iter().map(|b| b.len()).sum()
    }

    /// シェル要素の総数。
    pub fn n_shell_elements(&self) -> usize {
        self.shells.iter().map(|b| b.len()).sum()
    }

    /// 全要素数（ソリッド + シェル）。
    pub fn n_elements(&self) -> usize {
        self.n_solid_elements() + self.n_shell_elements()
    }

    /// ソリッド → シェルの順に全ブロックを走査する。VTU 出力のセル順序はこれに従う。
    pub fn blocks(&self) -> impl Iterator<Item = &ElementBlock> {
        self.solids.iter().chain(self.shells.iter())
    }

    /// 全ソリッド要素を通し番号つきで走査する。
    pub fn iter_solids(&self) -> impl Iterator<Item = ElementRef<'_>> {
        iter_blocks(&self.solids)
    }

    /// 通し番号でソリッド要素を引く。
    ///
    /// # パニック
    ///
    /// 範囲外の通し番号を渡すとパニックする。
    pub fn solid(&self, index: usize) -> ElementRef<'_> {
        element_at(&self.solids, index)
            .unwrap_or_else(|| panic!("ソリッド要素の通し番号が範囲外です: {index}"))
    }

    /// 全シェル要素（電極候補面）を通し番号つきで走査する。
    pub fn iter_shells(&self) -> impl Iterator<Item = ElementRef<'_>> {
        iter_blocks(&self.shells)
    }

    /// 通し番号でシェル要素を引く（[`Mesh::iter_shells`] の `index`）。
    ///
    /// # パニック
    ///
    /// 範囲外の通し番号を渡すとパニックする。
    pub fn shell(&self, index: usize) -> ElementRef<'_> {
        element_at(&self.shells, index)
            .unwrap_or_else(|| panic!("シェル要素の通し番号が範囲外です: {index}"))
    }

    /// 接続から節点座標（m）を取り出して `out` を埋める。
    ///
    /// 要素ループの内側で `Vec` を確保しないための形。
    pub fn gather_coords(&self, conn: &[NodeIndex], out: &mut [[f64; 3]]) {
        for (o, &n) in out.iter_mut().zip(conn) {
            *o = self.nodes.coord(n);
        }
    }

    /// 接続から要素の重心（m）を求める。
    pub fn element_centroid(&self, conn: &[NodeIndex]) -> [f64; 3] {
        let mut c = [0.0f64; 3];
        let scale = 1.0 / conn.len() as f64;
        for &n in conn {
            let p = self.nodes.coord(n);
            for (d, cd) in c.iter_mut().enumerate() {
                *cd += scale * p[d];
            }
        }
        c
    }

    /// Tet4 要素の 4 節点座標（m）。
    pub fn tet4_coords(&self, conn: &[NodeIndex]) -> [[f64; 3]; 4] {
        debug_assert_eq!(conn.len(), 4);
        [
            self.nodes.coord(conn[0]),
            self.nodes.coord(conn[1]),
            self.nodes.coord(conn[2]),
            self.nodes.coord(conn[3]),
        ]
    }

    /// 指定した種類のブロックを探す。
    pub fn block(&self, kind: ElementKind) -> Option<&ElementBlock> {
        self.blocks().find(|b| b.kind() == kind)
    }

    /// バウンディングボックス（m）。
    pub fn bounding_box(&self) -> Option<([f64; 3], [f64; 3])> {
        self.nodes.bounding_box()
    }

    /// **m として読んだ**メッシュを、座標の単位が `unit` だったものとして読み直した形にする。
    ///
    /// GUI は単位を選ばせる前に寸法を見せる（SPEC §3.2）ため、まず m のつもりで読む。
    /// 選ばれた単位で読み直すとファイルを 2 回パースすることになるので、代わりにこれを使う。
    /// 読み込みは「座標変換 → 最後にまとめて単位換算」の順（SPEC §3.1）なので、
    /// m で読んだ座標に換算係数を掛けたものは、最初から `unit` で読んだ座標と
    /// **ビット単位で同じ**になる。形状のチェックは寸法によらないので、警告も変わらない。
    ///
    /// m 以外の単位で読んだメッシュに使うとエラー（割り戻しで丸めが入るため）。
    pub fn reinterpret_length_unit(&self, unit: LengthUnit) -> crate::error::Result<Mesh> {
        if self.length_unit != LengthUnit::Meter {
            return Err(crate::error::GfemError::internal(crate::tr!(
                "Only a mesh read in m can have its length unit reinterpreted (this mesh is in {})",
                "単位を付け替えられるのは m として読んだメッシュだけです（このメッシュは {}）",
                self.length_unit.symbol()
            ))
            .with_hint(crate::tr!(
                "Read the file again with read_nastran in the chosen unit",
                "選んだ単位で read_nastran し直してください"
            )));
        }
        let mut mesh = self.clone();
        let scale = unit.to_meters();
        // 読み込み（io::nastran::parse）と同じく、係数が 1 のときは掛けない
        if scale != 1.0 {
            let coords: Vec<f64> = self.nodes.coords().iter().map(|v| v * scale).collect();
            mesh.nodes = Nodes::new(self.nodes.ids().to_vec(), coords);
        }
        mesh.length_unit = unit;
        Ok(mesh)
    }

    /// ソリッド要素が実際に使っている PID の一覧（昇順・重複なし）。
    pub fn solid_pids(&self) -> Vec<PropId> {
        let mut v: Vec<PropId> = self
            .solids
            .iter()
            .flat_map(|b| b.pids().iter().copied())
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// シェル要素が実際に使っている PID の一覧（昇順・重複なし）。
    pub fn shell_pids(&self) -> Vec<PropId> {
        let mut v: Vec<PropId> = self
            .shells
            .iter()
            .flat_map(|b| b.pids().iter().copied())
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Tet4 要素の符号付き体積（m³）。ブロック内の要素順。
    ///
    /// SPEC §2.3 の要素剛性は |V| を使うが、符号は節点順序の妥当性チェックに使う。
    pub fn tet4_signed_volumes(&self) -> Vec<f64> {
        let Some(block) = self.block(ElementKind::Tet4) else {
            return Vec::new();
        };
        (0..block.len())
            .map(|e| {
                let c = block.conn_of(e);
                let p = [
                    self.nodes.coord(c[0]),
                    self.nodes.coord(c[1]),
                    self.nodes.coord(c[2]),
                    self.nodes.coord(c[3]),
                ];
                tet4_signed_volume(&p)
            })
            .collect()
    }

    /// 要素の代表寸法（最大辺長, m）。退化判定の無次元化に使う。
    pub(crate) fn element_max_edge(&self, block: &ElementBlock, e: usize) -> f64 {
        // 読み込み時に全要素について呼ぶので、要素ごとに Vec を作らない
        let conn = block.conn_of(e);
        let mut buffer = [[0.0f64; 3]; crate::element::MAX_ELEMENT_NODES];
        self.gather_coords(conn, &mut buffer[..conn.len()]);
        max_edge_length(&buffer[..conn.len()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_tet() -> Mesh {
        Mesh {
            nodes: Nodes::new(
                vec![1, 2, 3, 4],
                vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ),
            solids: vec![ElementBlock::new(
                ElementKind::Tet4,
                vec![10],
                vec![1],
                vec![0, 1, 2, 3],
            )],
            shells: Vec::new(),
            solid_properties: vec![Property {
                pid: 1,
                mid: None,
                card: "PSOLID",
            }],
            shell_properties: Vec::new(),
            length_unit: LengthUnit::Meter,
            source: None,
        }
    }

    #[test]
    fn counts_and_bbox() {
        let m = single_tet();
        assert_eq!(m.n_solid_elements(), 1);
        assert_eq!(m.n_elements(), 1);
        let (lo, hi) = m.bounding_box().unwrap();
        assert_eq!(lo, [0.0, 0.0, 0.0]);
        assert_eq!(hi, [1.0, 1.0, 1.0]);
        assert_eq!(m.solid_pids(), vec![1]);
    }

    #[test]
    fn signed_volume() {
        let m = single_tet();
        let v = m.tet4_signed_volumes();
        assert_eq!(v.len(), 1);
        assert!((v[0] - 1.0 / 6.0).abs() < 1e-15);
    }
}
