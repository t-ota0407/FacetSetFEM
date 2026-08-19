//! 論理カードから [`Mesh`] を組み立てる（SPEC §3.1）。
//!
//! Nastran のバルクデータは**カードの出現順に依存しない**ので、
//! 読み込み中は要素の接続を節点 ID のまま溜め、最後にまとめて索引へ解決する。
//!
//! 方針は SPEC §3.1 の「厳しく、かつ親切に」。
//! 未対応カードは黙って捨てず、カード名・件数・最初の行番号を集計して報告する。

use super::field::{parse_int, parse_real};
use super::reader::{BulkReader, Card};
use crate::element::{ElementKind, tet4_signed_volume};
use crate::error::{ErrorKind, GfemError, Result, format_sample_list};
use crate::mesh::{ElemId, ElementBlock, IdMap, Mesh, NodeId, NodeIndex, Nodes, PropId, Property};
use crate::units::LengthUnit;
use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// 内容は理解できるが、電流場解析には使わないカード。件数だけ数える。
const IGNORED_CARDS: &[&str] = &[
    "MAT1", "MAT2", "MAT3", "MAT4", "MAT5", "MAT8", "MAT9", "PARAM", "ENDDATA",
];

/// エラー一覧に並べる最大件数。
const SAMPLE_LIMIT: usize = 20;

/// 退化要素とみなす無次元体積のしきい値。`|V| / h³` がこれ以下なら潰れている。
///
/// 形の良い四面体では `V / h³ ≈ 0.118`。極端に薄いスライバ要素を誤検出しないよう、
/// 「数値的に完全に潰れている」ものだけを捕まえる値にしてある。
const DEGENERATE_VOLUME_RATIO: f64 = 1.0e-12;

/// 未対応カードの集計。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CardCount {
    /// カード名
    pub name: String,
    /// 出現回数
    pub count: usize,
    /// 最初に現れた行番号
    pub first_line: usize,
}

/// 読み込みレポート（SPEC §6.2 の「読み込みレポート」に出す内容）。
#[derive(Debug, Clone, Default)]
pub struct ReadReport {
    /// 読み込んだファイル
    pub file: PathBuf,
    /// 読んだ物理行数
    pub lines: usize,
    /// 認識したカードごとの件数（カード名の昇順）
    pub card_counts: Vec<(String, usize)>,
    /// 未対応だったカードの集計
    pub unsupported: Vec<CardCount>,
    /// 警告（致命的ではないが利用者が知るべきこと）
    pub warnings: Vec<String>,
}

impl fmt::Display for ReadReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "読み込み: {} （{} 行）", self.file.display(), self.lines)?;
        for (name, count) in &self.card_counts {
            writeln!(f, "  {name:<8} {count} 件")?;
        }
        for u in &self.unsupported {
            writeln!(
                f,
                "  [未対応] {} {} 件（最初の出現: {} 行目）",
                u.name, u.count, u.first_line
            )?;
        }
        for w in &self.warnings {
            writeln!(f, "  [警告] {w}")?;
        }
        Ok(())
    }
}

/// Nastran バルクデータファイルを読み込む。
///
/// `length_unit` はファイル中の座標値の単位。座標は読み込み時に SI（m）へ換算される
/// （CLAUDE.md 設計原則 7 / SPEC §3.2）。
pub fn read_nastran(path: impl AsRef<Path>, length_unit: LengthUnit) -> Result<(Mesh, ReadReport)> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|e| {
        GfemError::io(path, "Nastran ファイルを開けません", e)
            .with_hint("パスが正しいか、読み取り権限があるかを確認してください")
    })?;
    // 1 MiB のバッファ。大規模ファイルでシステムコール回数を抑える
    let reader = BufReader::with_capacity(1 << 20, file);
    read_nastran_from(reader, path, length_unit)
}

/// 任意の [`BufRead`] から読み込む（テストと `INCLUDE` 展開のため）。
pub fn read_nastran_from<R: BufRead>(
    reader: R,
    name: impl AsRef<Path>,
    length_unit: LengthUnit,
) -> Result<(Mesh, ReadReport)> {
    let name = name.as_ref();
    let mut bulk = BulkReader::new(reader);
    let mut card = Card::new();
    let mut builder = Builder::new(length_unit);
    loop {
        match bulk.next_card(&mut card) {
            Ok(true) => builder.add_card(&card).map_err(|e| e.or_file(name))?,
            Ok(false) => break,
            Err(e) => return Err(e.or_file(name)),
        }
    }
    builder
        .finish(name, bulk.lines_read())
        .map_err(|e| e.or_file(name))
}

/// 要素種ごとの、まだ ID 解決していない生の接続。
#[derive(Debug)]
struct RawBlock {
    ids: Vec<ElemId>,
    pids: Vec<PropId>,
    conn_ids: Vec<NodeId>,
    /// 要素ごとの行番号（エラー報告用）
    lines: Vec<u32>,
}

impl RawBlock {
    fn new() -> Self {
        Self {
            ids: Vec::new(),
            pids: Vec::new(),
            conn_ids: Vec::new(),
            lines: Vec::new(),
        }
    }
}

struct Builder {
    unit: LengthUnit,
    node_ids: Vec<NodeId>,
    node_coords: Vec<f64>,
    blocks: BTreeMap<ElementKind, RawBlock>,
    properties: BTreeMap<PropId, Property>,
    card_counts: BTreeMap<String, usize>,
    unsupported: BTreeMap<String, (usize, usize)>,
    warnings: Vec<String>,
}

impl Builder {
    fn new(unit: LengthUnit) -> Self {
        Self {
            unit,
            node_ids: Vec::new(),
            node_coords: Vec::new(),
            blocks: BTreeMap::new(),
            properties: BTreeMap::new(),
            card_counts: BTreeMap::new(),
            unsupported: BTreeMap::new(),
            warnings: Vec::new(),
        }
    }

    fn add_card(&mut self, card: &Card) -> Result<()> {
        let name = card.name();
        match name {
            "GRID" => self.count_and(name, |s| s.grid(card)),
            "CTETRA" => self.count_and(name, |s| s.solid(card, ElementKind::Tet4, 10)),
            "CPENTA" => self.count_and(name, |s| s.solid(card, ElementKind::Penta6, 15)),
            "CHEXA" => self.count_and(name, |s| s.solid(card, ElementKind::Hex8, 20)),
            "CTRIA3" => self.count_and(name, |s| s.shell(card, ElementKind::Tria3)),
            "CQUAD4" => self.count_and(name, |s| s.shell(card, ElementKind::Quad4)),
            "PSOLID" => self.count_and(name, |s| s.property(card, "PSOLID")),
            "PSHELL" => self.count_and(name, |s| s.property(card, "PSHELL")),
            "CORD2R" | "CORD2C" | "CORD2S" => {
                // 座標系カード自体は M4 で対応する。GRID から参照されていれば
                // grid() 側でエラーになるので、ここでは件数を数えるだけでよい
                self.note_unsupported(name, card.line());
                Ok(())
            }
            "INCLUDE" => Err(GfemError::unsupported(
                "INCLUDE による外部ファイルの取り込みは未対応です",
            )
            .with_line(card.line())
            .with_card("INCLUDE")
            .with_hint(
                "取り込み先の内容を 1 つのファイルにまとめてから読み込ませてください\
                 （INCLUDE の対応は SPEC §3.1 の予定項目です）",
            )),
            _ if IGNORED_CARDS.contains(&name) => {
                *self.card_counts.entry(name.to_string()).or_insert(0) += 1;
                Ok(())
            }
            _ => {
                self.note_unsupported(name, card.line());
                Ok(())
            }
        }
    }

    fn count_and<F>(&mut self, name: &str, f: F) -> Result<()>
    where
        F: FnOnce(&mut Self) -> Result<()>,
    {
        *self.card_counts.entry(name.to_string()).or_insert(0) += 1;
        f(self)
    }

    fn note_unsupported(&mut self, name: &str, line: usize) {
        let e = self
            .unsupported
            .entry(name.to_string())
            .or_insert((0, line));
        e.0 += 1;
    }

    /// `GRID ID CP X1 X2 X3 CD PS SEID`
    fn grid(&mut self, card: &Card) -> Result<()> {
        let id = field_id(card, 1, "節点 ID")?;
        let cp = field_int(card, 2)?.unwrap_or(0);
        if cp != 0 {
            return Err(GfemError::unsupported(format!(
                "GRID の定義座標系 CP={cp} は未対応です（デカルト座標の直接指定のみ対応）"
            ))
            .with_line(card.line())
            .with_card("GRID")
            .with_field(2)
            .with_entity(format!("節点 ID={id}"))
            .with_hint(
                "CORD2R / CORD2C / CORD2S による座標変換は M4 で対応します。\
                 それまではデカルト座標に展開したファイルを使ってください",
            ));
        }
        // CD（変位座標系）はスカラー場では意味を持たないので読み飛ばす（SPEC §3.1）
        let f = self.unit.to_meters();
        let x = field_real(card, 3, 0.0)?;
        let y = field_real(card, 4, 0.0)?;
        let z = field_real(card, 5, 0.0)?;
        self.node_ids.push(id);
        self.node_coords.push(x * f);
        self.node_coords.push(y * f);
        self.node_coords.push(z * f);
        Ok(())
    }

    /// `CTETRA / CPENTA / CHEXA EID PID G1 G2 ...`
    fn solid(&mut self, card: &Card, kind: ElementKind, higher_order_nodes: usize) -> Result<()> {
        let n = kind.node_count();
        let eid = field_id(card, 1, "要素 ID")?;
        let pid = field_id(card, 2, "プロパティ ID (PID)")?;
        let last = card.last_non_blank(3);
        if last > 3 + n {
            return Err(GfemError::unsupported(format!(
                "{} の {} 節点版（2 次要素）は未対応です",
                kind.card_name(),
                higher_order_nodes
            ))
            .with_line(card.line())
            .with_card(kind.card_name())
            .with_entity(format!("要素 EID={eid}"))
            .with_hint(format!(
                "v1.0 は 1 次要素のみ対応です（SPEC §1.4）。{} 節点の {} に変換してください",
                n,
                kind.card_name()
            )));
        }
        let block = self.blocks.entry(kind).or_insert_with(RawBlock::new);
        block.ids.push(eid);
        block.pids.push(pid);
        block.lines.push(card.line() as u32);
        for k in 0..n {
            block
                .conn_ids
                .push(field_id(card, 3 + k, &format!("節点 G{}", k + 1))?);
        }
        Ok(())
    }

    /// `CTRIA3 / CQUAD4 EID PID G1 ... [THETA/MCID ZOFFS]`
    fn shell(&mut self, card: &Card, kind: ElementKind) -> Result<()> {
        let n = kind.node_count();
        let eid = field_id(card, 1, "要素 ID")?;
        let pid = field_id(card, 2, "プロパティ ID (PID)")?;
        let block = self.blocks.entry(kind).or_insert_with(RawBlock::new);
        block.ids.push(eid);
        block.pids.push(pid);
        block.lines.push(card.line() as u32);
        for k in 0..n {
            block
                .conn_ids
                .push(field_id(card, 3 + k, &format!("節点 G{}", k + 1))?);
        }
        // THETA/MCID・ZOFFS 以降は構造解析用なので読み飛ばす
        Ok(())
    }

    /// `PSOLID PID MID ...` / `PSHELL PID MID1 T ...`
    fn property(&mut self, card: &Card, kind: &'static str) -> Result<()> {
        let pid = field_id(card, 1, "プロパティ ID (PID)")?;
        let mid = field_int(card, 2)?.and_then(|v| u32::try_from(v).ok());
        let prop = Property {
            pid,
            mid,
            card: kind,
        };
        if let Some(prev) = self.properties.insert(pid, prop) {
            if prev.card == kind {
                self.warnings.push(format!(
                    "{kind} の PID={pid} が複数回定義されています（{} 行目のものを使います）",
                    card.line()
                ));
            } else {
                return Err(GfemError::model(format!(
                    "PID={pid} が {} と {kind} の両方で定義されています",
                    prev.card
                ))
                .with_line(card.line())
                .with_card(kind)
                .with_hint(
                    "PID は領域（PSOLID）と電極候補面（PSHELL）を識別する鍵なので、\
                     全プロパティカードを通して一意でなければなりません。片方の PID を変更してください",
                ));
            }
        }
        Ok(())
    }

    fn finish(self, name: &Path, lines: usize) -> Result<(Mesh, ReadReport)> {
        let Builder {
            unit,
            node_ids,
            node_coords,
            blocks,
            properties,
            card_counts,
            unsupported,
            mut warnings,
        } = self;

        if node_ids.is_empty() {
            return Err(
                GfemError::model("GRID カードが 1 枚もありません").with_hint(
                    "バルクデータセクション（BEGIN BULK 〜 ENDDATA）に GRID があるか、\
                     ファイルが空でないかを確認してください",
                ),
            );
        }

        // --- 節点 ID → 索引 ---
        let idmap = IdMap::build(&node_ids).map_err(|dups| {
            let list: Vec<String> = dups.iter().map(|d| d.to_string()).collect();
            GfemError::model(format!(
                "節点 ID が重複しています（{} 件）: {}",
                dups.len(),
                format_sample_list(&list, SAMPLE_LIMIT)
            ))
            .with_card("GRID")
            .with_hint("同じ ID の GRID が 2 枚以上あります。重複を取り除いてください")
        })?;

        // --- 接続を索引へ解決 ---
        let mut solids: Vec<ElementBlock> = Vec::new();
        let mut shells: Vec<ElementBlock> = Vec::new();
        let mut all_element_ids: Vec<(ElemId, &'static str)> = Vec::new();
        let mut referenced = vec![false; node_ids.len()];

        for (kind, raw) in blocks {
            let n = kind.node_count();
            let mut conn: Vec<NodeIndex> = Vec::with_capacity(raw.conn_ids.len());
            for (e, chunk) in raw.conn_ids.chunks_exact(n).enumerate() {
                for (k, &gid) in chunk.iter().enumerate() {
                    let Some(idx) = idmap.get(gid) else {
                        return Err(GfemError::model(format!(
                            "節点 ID {gid} が GRID として定義されていません"
                        ))
                        .with_line(raw.lines[e] as usize)
                        .with_card(kind.card_name())
                        .with_field(3 + k)
                        .with_entity(format!("要素 EID={}", raw.ids[e]))
                        .with_hint(format!(
                            "GRID カードに ID {gid} を追加するか、{} の節点 ID を修正してください",
                            kind.card_name()
                        )));
                    };
                    referenced[idx as usize] = true;
                    conn.push(idx);
                }
            }
            for &id in &raw.ids {
                all_element_ids.push((id, kind.card_name()));
            }
            let block = ElementBlock::new(kind, raw.ids, raw.pids, conn);
            if kind.is_solid() {
                solids.push(block);
            } else {
                shells.push(block);
            }
        }

        // --- 要素 ID の重複 ---
        all_element_ids.sort_unstable();
        let mut dup_elems: Vec<String> = Vec::new();
        for w in all_element_ids.windows(2) {
            if w[0].0 == w[1].0 {
                dup_elems.push(format!("EID={} ({} / {})", w[0].0, w[0].1, w[1].1));
            }
        }
        if !dup_elems.is_empty() {
            dup_elems.dedup();
            return Err(GfemError::model(format!(
                "要素 ID が重複しています（{} 件）: {}",
                dup_elems.len(),
                format_sample_list(&dup_elems, SAMPLE_LIMIT)
            ))
            .with_hint("要素 ID は全要素カードを通して一意でなければなりません"));
        }

        // --- プロパティの参照整合性 ---
        let mut solid_properties: Vec<Property> = Vec::new();
        let mut shell_properties: Vec<Property> = Vec::new();
        let mut used: BTreeMap<PropId, bool> = BTreeMap::new();
        for block in solids.iter().chain(shells.iter()) {
            for &pid in block.pids() {
                used.insert(pid, block.kind().is_solid());
            }
        }
        for (&pid, &is_solid) in &used {
            match properties.get(&pid) {
                Some(p) => {
                    let expected = if is_solid { "PSOLID" } else { "PSHELL" };
                    if p.card != expected {
                        warnings.push(format!(
                            "PID={pid} は {} で定義されていますが、{} 要素から参照されています",
                            p.card,
                            if is_solid {
                                "ソリッド"
                            } else {
                                "シェル"
                            }
                        ));
                    }
                    if is_solid {
                        solid_properties.push(p.clone());
                    } else {
                        shell_properties.push(p.clone());
                    }
                }
                None => {
                    // 領域一覧が欠けると GUI で σ を割り当てられなくなるので補完する
                    warnings.push(format!(
                        "PID={pid} を参照する要素がありますが、対応する {} カードがありません。\
                         領域として自動的に登録しました",
                        if is_solid { "PSOLID" } else { "PSHELL" }
                    ));
                    let p = Property {
                        pid,
                        mid: None,
                        card: if is_solid { "PSOLID" } else { "PSHELL" },
                    };
                    if is_solid {
                        solid_properties.push(p);
                    } else {
                        shell_properties.push(p);
                    }
                }
            }
        }
        let unused: Vec<String> = properties
            .keys()
            .filter(|pid| !used.contains_key(pid))
            .map(|pid| pid.to_string())
            .collect();
        if !unused.is_empty() {
            warnings.push(format!(
                "どの要素からも参照されていないプロパティがあります（{} 件）: PID={}",
                unused.len(),
                format_sample_list(&unused, SAMPLE_LIMIT)
            ));
        }

        // --- 孤立節点 ---
        let orphans: Vec<String> = referenced
            .iter()
            .enumerate()
            .filter(|&(_, &r)| !r)
            .map(|(i, _)| node_ids[i].to_string())
            .collect();
        if !orphans.is_empty() {
            warnings.push(format!(
                "どの要素からも参照されていない孤立節点があります（{} 件）: ID={}",
                orphans.len(),
                format_sample_list(&orphans, SAMPLE_LIMIT)
            ));
        }

        let mesh = Mesh {
            nodes: Nodes::new(node_ids, node_coords),
            solids,
            shells,
            solid_properties,
            shell_properties,
            length_unit: unit,
            source: Some(name.to_path_buf()),
        };

        if mesh.n_solid_elements() == 0 {
            warnings.push(
                "ソリッド要素（CTETRA / CPENTA / CHEXA）が 1 つもありません。解析はできません"
                    .to_string(),
            );
        }
        check_tet4_geometry(&mesh, &mut warnings)?;

        let report = ReadReport {
            file: name.to_path_buf(),
            lines,
            card_counts: card_counts.into_iter().collect(),
            unsupported: unsupported
                .into_iter()
                .map(|(name, (count, first_line))| CardCount {
                    name,
                    count,
                    first_line,
                })
                .collect(),
            warnings,
        };
        Ok((mesh, report))
    }
}

/// Tet4 の退化・節点順序を検査する（SPEC §3.1）。
///
/// - 体積が数値的にゼロ ＝ 潰れた要素。剛性行列が作れないのでエラー
/// - 体積が負 ＝ 節点順序の反転。要素剛性は |V| を使うので計算は可能。警告に留める
///
/// Hex8 / Penta6 のヤコビアン検査（SPEC §2.3）は、要素の形状関数を実装する M4 で追加する。
fn check_tet4_geometry(mesh: &Mesh, warnings: &mut Vec<String>) -> Result<()> {
    let Some(block) = mesh.block(ElementKind::Tet4) else {
        return Ok(());
    };
    let mut degenerate: Vec<String> = Vec::new();
    let mut inverted: Vec<String> = Vec::new();
    for e in 0..block.len() {
        let c = block.conn_of(e);
        let p = [
            mesh.nodes.coord(c[0]),
            mesh.nodes.coord(c[1]),
            mesh.nodes.coord(c[2]),
            mesh.nodes.coord(c[3]),
        ];
        let v = tet4_signed_volume(&p);
        let h = mesh.element_max_edge(block, e);
        if h <= 0.0 || v.abs() <= DEGENERATE_VOLUME_RATIO * h * h * h {
            degenerate.push(block.ids()[e].to_string());
        } else if v < 0.0 {
            inverted.push(block.ids()[e].to_string());
        }
    }
    if !degenerate.is_empty() {
        return Err(GfemError::model(format!(
            "体積がゼロの退化した CTETRA があります（{} 件）: EID={}",
            degenerate.len(),
            format_sample_list(&degenerate, SAMPLE_LIMIT)
        ))
        .with_card("CTETRA")
        .with_hint(
            "4 節点が同一平面上にあるか、同じ節点を重複して参照しています。\
             メッシュ生成側で該当要素を修正してください",
        ));
    }
    if !inverted.is_empty() {
        warnings.push(format!(
            "節点順序が反転している（符号付き体積が負の）CTETRA があります（{} 件）: EID={}。\
             要素剛性は |V| で計算するため結果には影響しません",
            inverted.len(),
            format_sample_list(&inverted, SAMPLE_LIMIT)
        ));
    }
    Ok(())
}

// --- フィールド読み取りのヘルパ ---------------------------------------------

fn field_int(card: &Card, i: usize) -> Result<Option<i64>> {
    let s = card.field(i);
    if s.is_empty() {
        return Ok(None);
    }
    parse_int(s).map(Some).ok_or_else(|| {
        GfemError::syntax(format!("整数として解釈できません: \"{s}\""))
            .with_line(card.line())
            .with_card(card.name())
            .with_field(i)
            .with_hint("符号付きの十進整数を書いてください")
    })
}

fn field_real(card: &Card, i: usize, default: f64) -> Result<f64> {
    let s = card.field(i);
    if s.is_empty() {
        return Ok(default);
    }
    parse_real(s).ok_or_else(|| {
        GfemError::syntax(format!("実数として解釈できません: \"{s}\""))
            .with_line(card.line())
            .with_card(card.name())
            .with_field(i)
            .with_hint("Nastran の実数形式（例: 1.0, 7., .5, 1.5+3, 2.5E-2）で書いてください")
    })
}

/// ID フィールド（1 以上の整数）を読む。
fn field_id(card: &Card, i: usize, what: &str) -> Result<u32> {
    let Some(v) = field_int(card, i)? else {
        return Err(GfemError::syntax(format!("{what} が空です"))
            .with_line(card.line())
            .with_card(card.name())
            .with_field(i)
            .with_hint("必須フィールドです。1 以上の整数を書いてください"));
    };
    if v < 1 || v > u32::MAX as i64 {
        return Err(
            GfemError::new(ErrorKind::Model, format!("{what} が範囲外です: {v}"))
                .with_line(card.line())
                .with_card(card.name())
                .with_field(i)
                .with_hint(format!(
                    "ID は 1 以上 {} 以下でなければなりません",
                    u32::MAX
                )),
        );
    }
    Ok(v as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY: &str = "\
$ 単位四面体 1 個
BEGIN BULK
GRID    1               0.0     0.0     0.0
GRID    2               1.0     0.0     0.0
GRID    3               0.0     1.0     0.0
GRID    4               0.0     0.0     1.0
PSOLID  1       1
CTETRA  10      1       1       2       3       4
ENDDATA
";

    fn read(text: &str, unit: LengthUnit) -> Result<(Mesh, ReadReport)> {
        read_nastran_from(text.as_bytes(), Path::new("<test>"), unit)
    }

    #[test]
    fn reads_a_single_tet() {
        let (mesh, report) = read(TINY, LengthUnit::Meter).unwrap();
        assert_eq!(mesh.nodes.len(), 4);
        assert_eq!(mesh.n_solid_elements(), 1);
        assert_eq!(mesh.solid_pids(), vec![1]);
        assert_eq!(mesh.solid_properties.len(), 1);
        assert_eq!(mesh.solid_properties[0].mid, Some(1));
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        let v = mesh.tet4_signed_volumes();
        assert!((v[0] - 1.0 / 6.0).abs() < 1e-15);
    }

    #[test]
    fn converts_units_to_si() {
        let (mesh, _) = read(TINY, LengthUnit::Millimeter).unwrap();
        let (lo, hi) = mesh.bounding_box().unwrap();
        assert_eq!(lo, [0.0, 0.0, 0.0]);
        // 1 mm = 1e-3 m
        assert!((hi[0] - 1.0e-3).abs() < 1e-18);
        let v = mesh.tet4_signed_volumes();
        assert!((v[0] - 1.0e-9 / 6.0).abs() < 1e-24);
    }

    #[test]
    fn free_format_is_equivalent() {
        let free = "\
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
GRID,4,,0.0,0.0,1.0
PSOLID,1,1
CTETRA,10,1,1,2,3,4
";
        let (a, _) = read(TINY, LengthUnit::Meter).unwrap();
        let (b, _) = read(free, LengthUnit::Meter).unwrap();
        assert_eq!(a.nodes.coords(), b.nodes.coords());
        assert_eq!(
            a.block(ElementKind::Tet4).unwrap().conn(),
            b.block(ElementKind::Tet4).unwrap().conn()
        );
    }

    #[test]
    fn cards_may_appear_in_any_order() {
        let reordered = "\
CTETRA,10,1,1,2,3,4
PSOLID,1,1
GRID,4,,0.0,0.0,1.0
GRID,3,,0.0,1.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,1,,0.0,0.0,0.0
";
        let (mesh, _) = read(reordered, LengthUnit::Meter).unwrap();
        assert_eq!(mesh.n_solid_elements(), 1);
        let v = mesh.tet4_signed_volumes();
        assert!((v[0] - 1.0 / 6.0).abs() < 1e-15);
    }

    #[test]
    fn missing_node_reports_element_and_line() {
        let text = "GRID,1,,0.0,0.0,0.0\nCTETRA,10,1,1,2,3,4\n";
        let e = read(text, LengthUnit::Meter).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("節点 ID 2"), "{s}");
        assert!(s.contains("EID=10"), "{s}");
        assert!(s.contains("対処"), "{s}");
    }

    #[test]
    fn duplicate_node_ids_are_rejected() {
        let text = "GRID,1,,0.0,0.0,0.0\nGRID,1,,1.0,0.0,0.0\nCTETRA,10,1,1,1,1,1\n";
        let e = read(text, LengthUnit::Meter).unwrap_err();
        assert!(e.to_string().contains("節点 ID が重複"));
    }

    #[test]
    fn duplicate_element_ids_are_rejected() {
        let mut text = TINY.replace(
            "ENDDATA",
            "CTETRA  10      1       1       2       3       4\nENDDATA",
        );
        text.push('\n');
        let e = read(&text, LengthUnit::Meter).unwrap_err();
        assert!(e.to_string().contains("要素 ID が重複"), "{e}");
    }

    #[test]
    fn degenerate_tet_is_rejected() {
        let text = "\
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
GRID,4,,1.0,1.0,0.0
CTETRA,10,1,1,2,3,4
";
        let e = read(text, LengthUnit::Meter).unwrap_err();
        assert!(e.to_string().contains("退化"), "{e}");
    }

    #[test]
    fn inverted_tet_is_only_a_warning() {
        let text = "\
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,0.0,1.0
GRID,4,,0.0,1.0,0.0
CTETRA,10,1,1,2,3,4
";
        let (_, report) = read(text, LengthUnit::Meter).unwrap();
        assert!(
            report.warnings.iter().any(|w| w.contains("反転")),
            "{:?}",
            report.warnings
        );
    }

    #[test]
    fn tet10_gives_a_helpful_error() {
        let text = "\
GRID,1,,0.0,0.0,0.0
CTETRA,10,1,1,2,3,4,5,6,7,8
,9,10
";
        let e = read(text, LengthUnit::Meter).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("2 次要素"), "{s}");
        assert!(s.contains("SPEC §1.4"), "{s}");
    }

    #[test]
    fn cp_field_is_rejected_with_guidance() {
        let text = "GRID,1,7,0.0,0.0,0.0\n";
        let e = read(text, LengthUnit::Meter).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("CP=7"), "{s}");
        assert!(s.contains("M4"), "{s}");
    }

    #[test]
    fn include_is_rejected() {
        let text = "INCLUDE 'other.bdf'\nGRID,1,,0.0,0.0,0.0\n";
        let e = read(text, LengthUnit::Meter).unwrap_err();
        assert!(e.to_string().contains("INCLUDE"));
    }

    #[test]
    fn unsupported_cards_are_reported_not_dropped_silently() {
        let text = format!(
            "{}\nCBAR,99,1,1,2,0.0,0.0,1.0\n",
            TINY.trim_end().replace("ENDDATA", "")
        );
        let (_, report) = read(&text, LengthUnit::Meter).unwrap();
        assert_eq!(report.unsupported.len(), 1);
        assert_eq!(report.unsupported[0].name, "CBAR");
        assert_eq!(report.unsupported[0].count, 1);
    }

    #[test]
    fn orphan_nodes_are_warned() {
        let text = format!("{}GRID,99,,5.0,5.0,5.0\n", TINY.replace("ENDDATA", ""));
        let (_, report) = read(&text, LengthUnit::Meter).unwrap();
        assert!(
            report.warnings.iter().any(|w| w.contains("孤立節点")),
            "{:?}",
            report.warnings
        );
    }

    #[test]
    fn pid_collision_between_psolid_and_pshell_is_an_error() {
        let text = "\
GRID,1,,0.0,0.0,0.0
PSOLID,1,1
PSHELL,1,1,0.1
";
        let e = read(text, LengthUnit::Meter).unwrap_err();
        assert!(e.to_string().contains("両方で定義"), "{e}");
    }

    #[test]
    fn shells_are_read_into_their_own_blocks() {
        let text = format!(
            "{}CTRIA3,100,101,1,2,3\nPSHELL,101,1,0.1\n",
            TINY.replace("ENDDATA", "")
        );
        let (mesh, _) = read(&text, LengthUnit::Meter).unwrap();
        assert_eq!(mesh.n_shell_elements(), 1);
        assert_eq!(mesh.shell_pids(), vec![101]);
        assert_eq!(mesh.n_elements(), 2);
    }

    #[test]
    fn empty_file_is_an_error() {
        let e = read("$ 何も無い\n", LengthUnit::Meter).unwrap_err();
        assert!(e.to_string().contains("GRID"));
    }
}
