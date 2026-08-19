//! VTU（VTK XML UnstructuredGrid）ライタ（SPEC §5.5）。
//!
//! `appended` + `raw` + zlib 圧縮で書く。ParaView で直接開けて、
//! 1000 万節点でも実用的なサイズと速度になる。
//!
//! `appended` 形式では各配列のファイル内オフセットを XML ヘッダに書く必要がある。
//! オフセットは圧縮後のサイズに依存するので、
//!
//! 1. オフセット値を固定幅の空白で埋めて XML を書く
//! 2. 実データを書きながら本当のオフセットを記録する
//! 3. 最後にシークして 1 の場所を埋める
//!
//! という順で処理する。全配列をメモリ上に二重持ちしないためにこうしている。

use crate::error::{GfemError, Result};
use crate::mesh::Mesh;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

/// 圧縮ブロックの大きさ。VTK 自身の既定値に合わせている。
const BLOCK_SIZE: usize = 32768;

/// `offset="..."` に確保する桁数。u64 の最大値がちょうど 20 桁。
const OFFSET_WIDTH: usize = 20;

/// 出力するバイト順。XML ヘッダに書く値と実際の書き出しを一致させる。
const BYTE_ORDER: &str = if cfg!(target_endian = "little") {
    "LittleEndian"
} else {
    "BigEndian"
};

/// VTU 出力の設定。
#[derive(Debug, Clone, Copy)]
pub struct VtuOptions {
    /// zlib 圧縮を使うか
    pub compress: bool,
    /// 圧縮レベル（0〜9）
    pub level: u32,
}

impl Default for VtuOptions {
    fn default() -> Self {
        Self {
            compress: true,
            level: 6,
        }
    }
}

/// バイト列としてそのまま書き出せる型。
///
/// いずれもパディングや不正ビットパターンを持たないため、
/// ビット列を `u8` の並びとして読み出してよい。
trait Raw: Copy {}
impl Raw for f64 {}
impl Raw for i32 {}
impl Raw for i64 {}
impl Raw for u8 {}

/// `&[T]` をネイティブバイト順のバイト列として見る。
fn as_bytes<T: Raw>(v: &[T]) -> &[u8] {
    // SAFETY: `Raw` を実装する型はパディングを持たない POD であり、
    // u8 のアライメント要求は 1 なので常に満たされる。
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// データ配列の中身。
#[derive(Debug, Clone, Copy)]
pub enum ArrayValues<'a> {
    /// 倍精度実数（物理量は全て f64。CLAUDE.md 設計原則 2）
    F64(&'a [f64]),
    /// 32 bit 整数
    I32(&'a [i32]),
    /// 64 bit 整数
    I64(&'a [i64]),
    /// 符号なし 8 bit 整数
    U8(&'a [u8]),
}

impl ArrayValues<'_> {
    const fn vtk_type(&self) -> &'static str {
        match self {
            ArrayValues::F64(_) => "Float64",
            ArrayValues::I32(_) => "Int32",
            ArrayValues::I64(_) => "Int64",
            ArrayValues::U8(_) => "UInt8",
        }
    }

    /// 要素数（成分数で割る前の総数）。
    pub fn len(&self) -> usize {
        match self {
            ArrayValues::F64(v) => v.len(),
            ArrayValues::I32(v) => v.len(),
            ArrayValues::I64(v) => v.len(),
            ArrayValues::U8(v) => v.len(),
        }
    }

    /// 空か。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn bytes(&self) -> &[u8] {
        match self {
            ArrayValues::F64(v) => as_bytes(v),
            ArrayValues::I32(v) => as_bytes(v),
            ArrayValues::I64(v) => as_bytes(v),
            ArrayValues::U8(v) => as_bytes(v),
        }
    }
}

/// 名前付きのデータ配列。
#[derive(Debug, Clone)]
pub struct DataArray<'a> {
    /// ParaView に表示される名前
    pub name: String,
    /// 1 点／1 セルあたりの成分数
    pub n_components: usize,
    /// 値
    pub values: ArrayValues<'a>,
}

impl<'a> DataArray<'a> {
    /// スカラーの f64 配列。
    pub fn scalars_f64(name: &str, values: &'a [f64]) -> Self {
        Self {
            name: name.to_string(),
            n_components: 1,
            values: ArrayValues::F64(values),
        }
    }

    /// 3 成分ベクトルの f64 配列。
    pub fn vectors_f64(name: &str, values: &'a [f64]) -> Self {
        Self {
            name: name.to_string(),
            n_components: 3,
            values: ArrayValues::F64(values),
        }
    }

    /// スカラーの i64 配列（ID の出力に使う）。
    pub fn scalars_i64(name: &str, values: &'a [i64]) -> Self {
        Self {
            name: name.to_string(),
            n_components: 1,
            values: ArrayValues::I64(values),
        }
    }
}

/// 節点値・要素値のまとまり。
#[derive(Debug, Clone, Default)]
pub struct VtuData<'a> {
    /// 節点に載る量（φ など）
    pub point_data: Vec<DataArray<'a>>,
    /// 要素に載る量（E, J, PID など）
    pub cell_data: Vec<DataArray<'a>>,
}

/// メッシュと付随データを VTU として書き出す。
pub fn write_vtu(
    path: impl AsRef<Path>,
    mesh: &Mesh,
    data: &VtuData<'_>,
    opts: &VtuOptions,
) -> Result<()> {
    let path = path.as_ref();
    let n_points = mesh.nodes.len();
    let n_cells = mesh.n_elements();

    for a in &data.point_data {
        check_array(a, n_points, "節点")?;
    }
    for a in &data.cell_data {
        check_array(a, n_cells, "要素")?;
    }

    // --- セル情報を VTK の並びに変換する ---
    let conn_total: usize = mesh.blocks().map(|b| b.len() * b.kind().node_count()).sum();
    let use_i32 = n_points <= i32::MAX as usize && conn_total <= i32::MAX as usize;
    let mut connectivity = IntBuf::with_capacity(use_i32, conn_total);
    let mut offsets = IntBuf::with_capacity(use_i32, n_cells);
    let mut types: Vec<u8> = Vec::with_capacity(n_cells);
    let mut acc: i64 = 0;
    for block in mesh.blocks() {
        let nn = block.kind().node_count();
        let ty = block.kind().vtk_cell_type();
        for &node in block.conn() {
            connectivity.push(node as i64);
        }
        for _ in 0..block.len() {
            acc += nn as i64;
            offsets.push(acc);
            types.push(ty);
        }
    }

    // --- 書き出す配列を順番に並べる（XML の offset とデータの順序を一致させる）---
    let mut entries: Vec<Entry<'_>> =
        Vec::with_capacity(4 + data.point_data.len() + data.cell_data.len());
    entries.push(Entry::new(
        "Points",
        3,
        ArrayValues::F64(mesh.nodes.coords()),
    ));
    entries.push(Entry::new("connectivity", 1, connectivity.values()));
    entries.push(Entry::new("offsets", 1, offsets.values()));
    entries.push(Entry::new("types", 1, ArrayValues::U8(&types)));
    let point_data_start = entries.len();
    for a in &data.point_data {
        entries.push(Entry::new(&a.name, a.n_components, a.values));
    }
    let cell_data_start = entries.len();
    for a in &data.cell_data {
        entries.push(Entry::new(&a.name, a.n_components, a.values));
    }

    let file = File::create(path).map_err(|e| {
        GfemError::io(path, "VTU ファイルを作成できません", e)
            .with_hint("出力先のディレクトリが存在し、書き込み権限があるかを確認してください")
    })?;
    let mut out = Out::new(BufWriter::with_capacity(1 << 20, file));

    let io_err = |e: std::io::Error| {
        GfemError::io(path, "VTU ファイルの書き込みに失敗しました", e)
            .with_hint("ディスクの空き容量を確認してください")
    };

    // --- XML ヘッダ ---
    let mut placeholders: Vec<u64> = Vec::with_capacity(entries.len());
    (|| -> std::io::Result<()> {
        out.str("<?xml version=\"1.0\"?>\n")?;
        out.str("<VTKFile type=\"UnstructuredGrid\" version=\"1.0\"")?;
        out.string(format!(
            " byte_order=\"{BYTE_ORDER}\" header_type=\"UInt64\""
        ))?;
        if opts.compress {
            out.str(" compressor=\"vtkZLibDataCompressor\"")?;
        }
        out.str(">\n  <UnstructuredGrid>\n")?;
        out.string(format!(
            "    <Piece NumberOfPoints=\"{n_points}\" NumberOfCells=\"{n_cells}\">\n"
        ))?;

        out.str("      <Points>\n")?;
        write_data_array(&mut out, &entries[0], &mut placeholders)?;
        out.str("      </Points>\n")?;

        out.str("      <Cells>\n")?;
        for e in &entries[1..4] {
            write_data_array(&mut out, e, &mut placeholders)?;
        }
        out.str("      </Cells>\n")?;

        out.str("      <PointData>\n")?;
        for e in &entries[point_data_start..cell_data_start] {
            write_data_array(&mut out, e, &mut placeholders)?;
        }
        out.str("      </PointData>\n")?;

        out.str("      <CellData>\n")?;
        for e in &entries[cell_data_start..] {
            write_data_array(&mut out, e, &mut placeholders)?;
        }
        out.str("      </CellData>\n")?;

        out.str("    </Piece>\n  </UnstructuredGrid>\n")?;

        // --- 実データ ---
        out.str("  <AppendedData encoding=\"raw\">\n   _")?;
        let data_start = out.pos;
        let mut real_offsets: Vec<u64> = Vec::with_capacity(entries.len());
        for e in &entries {
            real_offsets.push(out.pos - data_start);
            write_block(&mut out, e.values.bytes(), opts)?;
        }
        out.str("\n  </AppendedData>\n</VTKFile>\n")?;

        // --- オフセットを埋め戻す ---
        let mut file = out.into_inner()?;
        for (&pos, &off) in placeholders.iter().zip(real_offsets.iter()) {
            file.seek(SeekFrom::Start(pos))?;
            let s = format!("{off:<OFFSET_WIDTH$}");
            debug_assert_eq!(s.len(), OFFSET_WIDTH);
            file.write_all(s.as_bytes())?;
        }
        file.flush()
    })()
    .map_err(io_err)?;

    Ok(())
}

/// メッシュだけを（節点 ID・PID・要素 ID を付けて）VTU に書き出す。
///
/// M0 でメッシュの読み込み結果を目視確認するために使う。
pub fn write_mesh_vtu(path: impl AsRef<Path>, mesh: &Mesh, opts: &VtuOptions) -> Result<()> {
    let node_id: Vec<i64> = mesh.nodes.ids().iter().map(|&v| v as i64).collect();
    let mut pid: Vec<i64> = Vec::with_capacity(mesh.n_elements());
    let mut eid: Vec<i64> = Vec::with_capacity(mesh.n_elements());
    for block in mesh.blocks() {
        pid.extend(block.pids().iter().map(|&v| v as i64));
        eid.extend(block.ids().iter().map(|&v| v as i64));
    }
    let data = VtuData {
        point_data: vec![DataArray::scalars_i64("NodeID", &node_id)],
        cell_data: vec![
            DataArray::scalars_i64("PID", &pid),
            DataArray::scalars_i64("ElementID", &eid),
        ],
    };
    write_vtu(path, mesh, &data, opts)
}

// --- 内部実装 ---------------------------------------------------------------

/// 書き出し対象の 1 配列。
struct Entry<'a> {
    name: String,
    n_components: usize,
    values: ArrayValues<'a>,
}

impl<'a> Entry<'a> {
    fn new(name: &str, n_components: usize, values: ArrayValues<'a>) -> Self {
        Self {
            name: name.to_string(),
            n_components,
            values,
        }
    }
}

/// 32 bit / 64 bit を実行時に切り替える整数バッファ。
///
/// 節点数と接続数が 2^31 に収まるうちは Int32 で書く（ファイルサイズが半分になる）。
enum IntBuf {
    I32(Vec<i32>),
    I64(Vec<i64>),
}

impl IntBuf {
    fn with_capacity(use_i32: bool, n: usize) -> Self {
        if use_i32 {
            IntBuf::I32(Vec::with_capacity(n))
        } else {
            IntBuf::I64(Vec::with_capacity(n))
        }
    }

    fn push(&mut self, v: i64) {
        match self {
            IntBuf::I32(b) => b.push(v as i32),
            IntBuf::I64(b) => b.push(v),
        }
    }

    fn values(&self) -> ArrayValues<'_> {
        match self {
            IntBuf::I32(b) => ArrayValues::I32(b),
            IntBuf::I64(b) => ArrayValues::I64(b),
        }
    }
}

/// バイト位置を数えながら書く writer。
struct Out<W: Write> {
    inner: W,
    pos: u64,
}

impl<W: Write> Out<W> {
    fn new(inner: W) -> Self {
        Self { inner, pos: 0 }
    }

    fn bytes(&mut self, b: &[u8]) -> std::io::Result<()> {
        self.inner.write_all(b)?;
        self.pos += b.len() as u64;
        Ok(())
    }

    fn str(&mut self, s: &str) -> std::io::Result<()> {
        self.bytes(s.as_bytes())
    }

    fn string(&mut self, s: String) -> std::io::Result<()> {
        self.bytes(s.as_bytes())
    }

    fn u64(&mut self, v: u64) -> std::io::Result<()> {
        self.bytes(&v.to_ne_bytes())
    }
}

impl Out<BufWriter<File>> {
    fn into_inner(self) -> std::io::Result<File> {
        self.inner
            .into_inner()
            .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

/// `<DataArray .../>` を 1 本書き、`offset` の値を埋める位置を記録する。
fn write_data_array<W: Write>(
    out: &mut Out<W>,
    entry: &Entry<'_>,
    placeholders: &mut Vec<u64>,
) -> std::io::Result<()> {
    out.string(format!(
        "        <DataArray type=\"{}\" Name=\"{}\" NumberOfComponents=\"{}\" format=\"appended\" offset=\"",
        entry.values.vtk_type(),
        escape_attr(&entry.name),
        entry.n_components
    ))?;
    placeholders.push(out.pos);
    out.string(" ".repeat(OFFSET_WIDTH))?;
    out.str("\"/>\n")
}

/// XML 属性値としてまずい文字を実体参照に置き換える。
fn escape_attr(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

/// appended セクションに 1 配列分のブロックを書く。
///
/// 非圧縮なら `[全バイト数: u64][データ]`。
/// 圧縮ありなら `[ブロック数][ブロック長][最終ブロック長][各圧縮後長...][圧縮データ...]`。
fn write_block<W: Write>(out: &mut Out<W>, bytes: &[u8], opts: &VtuOptions) -> std::io::Result<()> {
    if !opts.compress {
        out.u64(bytes.len() as u64)?;
        return out.bytes(bytes);
    }
    if bytes.is_empty() {
        out.u64(0)?;
        out.u64(0)?;
        out.u64(0)?;
        return Ok(());
    }
    let n_blocks = bytes.len().div_ceil(BLOCK_SIZE);
    // 最終ブロックがちょうど BLOCK_SIZE のときは 0 を書く（VTK の規約）
    let last = bytes.len() % BLOCK_SIZE;
    let compressed: Vec<Vec<u8>> = bytes
        .chunks(BLOCK_SIZE)
        .map(|c| zlib_compress(c, opts.level))
        .collect();
    out.u64(n_blocks as u64)?;
    out.u64(BLOCK_SIZE as u64)?;
    out.u64(last as u64)?;
    for c in &compressed {
        out.u64(c.len() as u64)?;
    }
    for c in &compressed {
        out.bytes(c)?;
    }
    Ok(())
}

fn zlib_compress(input: &[u8], level: u32) -> Vec<u8> {
    let mut e = flate2::write::ZlibEncoder::new(
        Vec::with_capacity(input.len() / 2 + 64),
        flate2::Compression::new(level.min(9)),
    );
    // Vec への書き込みは失敗しない
    e.write_all(input).expect("Vec への書き込みは失敗しない");
    e.finish().expect("Vec への書き込みは失敗しない")
}

fn check_array(a: &DataArray<'_>, n: usize, what: &str) -> Result<()> {
    let expected = n * a.n_components;
    if a.values.len() != expected {
        return Err(GfemError::internal(format!(
            "VTU に書く配列 \"{}\" の長さが合いません: {} 個ありますが、{}数 {} × 成分数 {} = {} 個であるべきです",
            a.name,
            a.values.len(),
            what,
            n,
            a.n_components,
            expected
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::ElementKind;
    use crate::mesh::{ElementBlock, Nodes, Property};
    use crate::units::LengthUnit;

    fn two_tets() -> Mesh {
        // 立方体の一部。節点 5 個、四面体 2 個
        let coords = vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
            1.0, 1.0, 1.0,
        ];
        Mesh {
            nodes: Nodes::new(vec![1, 2, 3, 4, 5], coords),
            solids: vec![ElementBlock::new(
                ElementKind::Tet4,
                vec![10, 11],
                vec![1, 2],
                vec![0, 1, 2, 3, 1, 2, 3, 4],
            )],
            shells: vec![ElementBlock::new(
                ElementKind::Tria3,
                vec![100],
                vec![101],
                vec![0, 1, 2],
            )],
            solid_properties: vec![
                Property {
                    pid: 1,
                    mid: None,
                    card: "PSOLID",
                },
                Property {
                    pid: 2,
                    mid: None,
                    card: "PSOLID",
                },
            ],
            shell_properties: vec![Property {
                pid: 101,
                mid: None,
                card: "PSHELL",
            }],
            length_unit: LengthUnit::Meter,
            source: None,
        }
    }

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("gfem_test_{}_{name}", std::process::id()));
        p
    }

    #[test]
    fn writes_compressed_and_uncompressed() {
        let mesh = two_tets();
        for (name, compress) in [("c.vtu", true), ("u.vtu", false)] {
            let p = tmp_path(name);
            write_mesh_vtu(&p, &mesh, &VtuOptions { compress, level: 6 }).unwrap();
            let bytes = std::fs::read(&p).unwrap();
            let head = String::from_utf8_lossy(&bytes[..600]).to_string();
            assert!(head.contains("NumberOfPoints=\"5\""), "{head}");
            assert!(head.contains("NumberOfCells=\"3\""), "{head}");
            assert_eq!(head.contains("vtkZLibDataCompressor"), compress);
            // offset が空白のまま残っていないこと
            assert!(!head.contains("offset=\"                    \""), "{head}");
            std::fs::remove_file(&p).ok();
        }
    }

    #[test]
    fn cell_types_follow_block_order() {
        let mesh = two_tets();
        let p = tmp_path("types.vtu");
        write_mesh_vtu(
            &p,
            &mesh,
            &VtuOptions {
                compress: false,
                level: 0,
            },
        )
        .unwrap();
        let bytes = std::fs::read(&p).unwrap();
        // types 配列は Tet4, Tet4, Tria3 の順（VTK: 10, 10, 5）
        let needle = [10u8, 10, 5];
        assert!(
            bytes.windows(3).any(|w| w == needle),
            "types 配列が見つかりません"
        );
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn rejects_wrong_length_arrays() {
        let mesh = two_tets();
        let bad = vec![1.0f64, 2.0];
        let data = VtuData {
            point_data: vec![DataArray::scalars_f64("phi", &bad)],
            cell_data: Vec::new(),
        };
        let p = tmp_path("bad.vtu");
        let e = write_vtu(&p, &mesh, &data, &VtuOptions::default()).unwrap_err();
        assert!(e.to_string().contains("長さが合いません"), "{e}");
        std::fs::remove_file(&p).ok();
    }
}
