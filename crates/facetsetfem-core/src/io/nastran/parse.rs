//! 論理カードから [`Mesh`] を組み立てる（SPEC §3.1）。
//!
//! Nastran のバルクデータは**カードの出現順に依存しない**ので、
//! 読み込み中は要素の接続を節点 ID のまま溜め、最後にまとめて索引へ解決する。
//!
//! 方針は SPEC §3.1 の「厳しく、かつ親切に」。
//! 未対応カードは黙って捨てず、カード名・件数・最初の行番号を集計して報告する。

use super::coord::{self, CoordDefinition, CoordKind};
use super::field::{parse_int, parse_real};
use super::reader::{BulkReader, Card};
use crate::element::hex8::Hex8;
use crate::element::penta6::Penta6;
use crate::element::shape;
use crate::element::tet10::{self, Orientation};
use crate::element::{ElementKind, tet4_signed_volume};
use crate::error::{ErrorKind, FacetSetFemError, Result, format_sample_list};
use crate::mesh::{ElemId, ElementBlock, IdMap, Mesh, NodeId, NodeIndex, Nodes, PropId, Property};
use crate::provenance::HashingReader;
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

/// 節点フィールドのエラーに出す名前（G1〜G10）。
///
/// 要素 1 個につき節点の数だけ引くので、`format!` で作ると 1000 万節点級のモデルで
/// 数億回の文字列確保になる。エラーにならない限り使わない名前なので、固定の表にしておく。
/// 英語と日本語の組で持ち、エラーを作るときに片方を選ぶ（[`FieldName`]）。
///
/// **最も節点の多い要素（Tet10 の 10）以上の長さが要る。** 足りないと `take(n)` が
/// 黙って途中で止まり、接続が 1 要素ずつずれる。
const NODE_FIELD_NAMES: [FieldName; shape::MAX_ELEMENT_NODES] = [
    ("node G1", "節点 G1"),
    ("node G2", "節点 G2"),
    ("node G3", "節点 G3"),
    ("node G4", "節点 G4"),
    ("node G5", "節点 G5"),
    ("node G6", "節点 G6"),
    ("node G7", "節点 G7"),
    ("node G8", "節点 G8"),
    ("node G9", "節点 G9"),
    ("node G10", "節点 G10"),
];

/// エラーに出すフィールド名（英語, 日本語）。読むたびに訳を選ばないよう、
/// エラーを作るときまで組のまま持つ（SPEC §6.6）。
type FieldName = (&'static str, &'static str);

const NODE_ID: FieldName = ("node ID", "節点 ID");
const ELEMENT_ID: FieldName = ("element ID", "要素 ID");
const PROPERTY_ID: FieldName = ("property ID (PID)", "プロパティ ID (PID)");
const COORD_ID: FieldName = ("coordinate system ID (CID)", "座標系 ID (CID)");

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
    /// 読み込んだファイル全体の SHA-256（16 進小文字。`sha256sum` と同じ値）
    ///
    /// **解いたメッシュそのもののハッシュ**として実行サマリに残す（SPEC §5.5）。
    /// ファイルから読んだときだけ入る（[`read_nastran_from`] では `None`）。
    pub sha256: Option<String>,
}

impl fmt::Display for ReadReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ja = crate::i18n::is_ja();
        let file = self.file.display();
        let lines = self.lines;
        if ja {
            writeln!(f, "読み込み: {file} （{lines} 行）")?;
        } else {
            writeln!(f, "Read: {file} ({lines} lines)")?;
        }
        for (name, count) in &self.card_counts {
            if ja {
                writeln!(f, "  {name:<8} {count} 件")?;
            } else {
                writeln!(f, "  {name:<8} {count}")?;
            }
        }
        for u in &self.unsupported {
            let (name, count, first) = (&u.name, u.count, u.first_line);
            if ja {
                writeln!(
                    f,
                    "  [未対応] {name} {count} 件（最初の出現: {first} 行目）"
                )?;
            } else {
                writeln!(
                    f,
                    "  [unsupported] {name} ×{count} (first seen at line {first})"
                )?;
            }
        }
        let head = if ja { "警告" } else { "warning" };
        for w in &self.warnings {
            writeln!(f, "  [{head}] {w}")?;
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
        FacetSetFemError::io(
            path,
            crate::tr!(
                "Cannot open the Nastran file",
                "Nastran ファイルを開けません"
            ),
            e,
        )
        .with_hint(crate::tr!(
            "Check that the path is correct and that you have read permission",
            "パスが正しいか、読み取り権限があるかを確認してください"
        ))
    })?;
    // 読んだバイト列をその場でハッシュに流す（SPEC §5.5）。実行サマリを書く時点で
    // ファイルを読み直すと、その間に書き換えられていても気づけない。
    // 1 MiB のバッファは大規模ファイルでシステムコール回数を抑えるため
    let mut reader = BufReader::with_capacity(1 << 20, HashingReader::new(file));
    let (mesh, mut report) = read_nastran_from(&mut reader, path, length_unit)?;
    // ENDDATA の後ろも含めたファイル全体のハッシュにする（sha256sum と同じ値になる）
    std::io::copy(&mut reader, &mut std::io::sink()).map_err(|e| {
        FacetSetFemError::io(
            path,
            crate::tr!(
                "Cannot read the rest of the file to compute its hash",
                "ハッシュを取るためにファイルの残りを読めません"
            ),
            e,
        )
    })?;
    report.sha256 = Some(reader.into_inner().finish());
    Ok((mesh, report))
}

/// 任意の [`BufRead`] から読み込む（テストや、メモリ上のバイト列を読むため）。
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
    /// ファイルに書かれたままの座標。単位換算と座標変換は `finish()` でまとめて行う
    node_coords: Vec<f64>,
    /// CP が 0 でない節点だけの `(節点索引, CID)`。ほとんどの入力では空
    node_cp: Vec<(u32, u32)>,
    /// CORD2R / CORD2C / CORD2S の定義
    coord_definitions: BTreeMap<u32, CoordDefinition>,
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
            node_cp: Vec::new(),
            coord_definitions: BTreeMap::new(),
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
            "CTETRA" => self.count_and(name, |s| s.tetra(card)),
            "CPENTA" => self.count_and(name, |s| s.solid(card, ElementKind::Penta6, 15)),
            "CHEXA" => self.count_and(name, |s| s.solid(card, ElementKind::Hex8, 20)),
            "CTRIA3" => self.count_and(name, |s| s.shell(card, ElementKind::Tria3)),
            "CQUAD4" => self.count_and(name, |s| s.shell(card, ElementKind::Quad4)),
            "CTRIA6" => self.count_and(name, |s| s.tria6(card)),
            "PSOLID" => self.count_and(name, |s| s.property(card, "PSOLID")),
            "PSHELL" => self.count_and(name, |s| s.property(card, "PSHELL")),
            "CORD2R" => self.count_and(name, |s| s.coord(card, CoordKind::Rectangular)),
            "CORD2C" => self.count_and(name, |s| s.coord(card, CoordKind::Cylindrical)),
            "CORD2S" => self.count_and(name, |s| s.coord(card, CoordKind::Spherical)),
            "INCLUDE" => Err(FacetSetFemError::unsupported(crate::tr!(
                "Including external files with INCLUDE is not supported",
                "INCLUDE による外部ファイルの取り込みは未対応です"
            ))
            .with_line(card.line())
            .with_card("INCLUDE")
            .with_hint(crate::tr!(
                "Merge the included contents into a single file before loading it \
                 (INCLUDE support is planned in SPEC §3.1)",
                "取り込み先の内容を 1 つのファイルにまとめてから読み込ませてください\
                 （INCLUDE の対応は SPEC §3.1 の予定項目です）"
            ))),
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
        let id = field_id(card, 1, NODE_ID)?;
        let cp = field_int(card, 2)?.unwrap_or(0);
        if !(0..=u32::MAX as i64).contains(&cp) {
            return Err(FacetSetFemError::syntax(crate::tr!(
                "The definition coordinate system CP is out of range: {cp}",
                "定義座標系 CP が範囲外です: {cp}"
            ))
            .with_line(card.line())
            .with_card("GRID")
            .with_field(2)
            .with_entity(crate::tr!("node ID={id}", "節点 ID={id}"))
            .with_hint(crate::tr!(
                "Write a coordinate system ID of 0 or more in CP",
                "CP には 0 以上の座標系 ID を書いてください"
            )));
        }
        // CD（変位座標系）はスカラー場では意味を持たないので読み飛ばす（SPEC §3.1）
        let x = field_real(card, 3, 0.0)?;
        let y = field_real(card, 4, 0.0)?;
        let z = field_real(card, 5, 0.0)?;
        if cp != 0 {
            self.node_cp.push((self.node_ids.len() as u32, cp as u32));
        }
        self.node_ids.push(id);
        // 座標変換と単位換算は全カードを読んでからまとめて行う（SPEC §3.1）
        self.node_coords.push(x);
        self.node_coords.push(y);
        self.node_coords.push(z);
        Ok(())
    }

    /// `CORD2R / CORD2C / CORD2S CID RID A1 A2 A3 B1 B2 B3 / C1 C2 C3`
    fn coord(&mut self, card: &Card, kind: CoordKind) -> Result<()> {
        let cid = field_id(card, 1, COORD_ID)?;
        let rid = field_int(card, 2)?.unwrap_or(0);
        if !(0..=u32::MAX as i64).contains(&rid) {
            return Err(FacetSetFemError::syntax(crate::tr!(
                "The reference coordinate system RID is out of range: {rid}",
                "参照座標系 RID が範囲外です: {rid}"
            ))
            .with_line(card.line())
            .with_card(kind.card_name())
            .with_field(2)
            .with_entity(format!("CID={cid}"))
            .with_hint(crate::tr!(
                "Write a coordinate system ID of 0 or more in RID",
                "RID には 0 以上の座標系 ID を書いてください"
            )));
        }
        let point = |start: usize| -> Result<[f64; 3]> {
            Ok([
                field_real(card, start, 0.0)?,
                field_real(card, start + 1, 0.0)?,
                field_real(card, start + 2, 0.0)?,
            ])
        };
        let definition = CoordDefinition {
            cid,
            rid: rid as u32,
            kind,
            a: point(3)?,
            b: point(6)?,
            c: point(9)?,
            line: card.line(),
        };
        if self.coord_definitions.insert(cid, definition).is_some() {
            self.warnings.push(crate::tr!(
                "{} CID={cid} is defined more than once (the one on line {} is used)",
                "{} の CID={cid} が複数回定義されています（{} 行目のものを使います）",
                kind.card_name(),
                card.line()
            ));
        }
        Ok(())
    }

    /// `CTETRA EID PID G1 G2 G3 G4 [G5 ... G10]`
    ///
    /// G5〜G10（辺の中点）が全て空欄なら 1 次の Tet4、全て書かれていれば 2 次の Tet10。
    /// Nastran は中点の一部だけを書くことも許すが、そういう要素は形状関数が別物になるので
    /// 扱わない（空欄の欄を挙げてエラーにする）。
    fn tetra(&mut self, card: &Card) -> Result<()> {
        let last = card.last_non_blank(3);
        if last <= 3 + 4 {
            return self.solid(card, ElementKind::Tet4, 10);
        }
        let eid = field_id(card, 1, ELEMENT_ID)?;
        if last > 3 + 10 {
            return Err(FacetSetFemError::syntax(crate::tr!(
                "CTETRA has more node fields than the 10 it can hold",
                "CTETRA の節点の欄が、書ける上限の 10 個を超えています"
            ))
            .with_line(card.line())
            .with_card("CTETRA")
            .with_field(last - 1)
            .with_entity(crate::tr!("element EID={eid}", "要素 EID={eid}"))
            .with_hint(crate::tr!(
                "CTETRA takes G1–G4 (corners) and optionally G5–G10 (mid-side nodes). \
                 Check the continuation lines",
                "CTETRA に書けるのは G1〜G4（頂点）と、必要なら G5〜G10（辺の中点）だけです。\
                 継続行を確認してください"
            )));
        }
        let blank: Vec<&str> = (4..10)
            .filter(|&k| card.field(3 + k).is_empty())
            .map(|k| {
                let (en, ja) = NODE_FIELD_NAMES[k];
                if crate::i18n::is_ja() { ja } else { en }
            })
            .collect();
        if !blank.is_empty() {
            return Err(FacetSetFemError::unsupported(crate::tr!(
                "A CTETRA with only some of its mid-side nodes is not supported (blank: {})",
                "辺の中点の一部だけを書いた CTETRA には対応していません（空欄: {}）",
                blank.join(", ")
            ))
            .with_line(card.line())
            .with_card("CTETRA")
            .with_entity(crate::tr!("element EID={eid}", "要素 EID={eid}"))
            .with_hint(crate::tr!(
                "Write all six mid-side nodes G5–G10 (second-order element), \
                 or leave all of them blank (first-order element)",
                "辺の中点 G5〜G10 を 6 つとも書く（2 次要素）か、すべて空欄にしてください（1 次要素）"
            )));
        }
        self.solid(card, ElementKind::Tet10, 10)
    }

    /// `CTETRA / CPENTA / CHEXA EID PID G1 G2 ...`
    fn solid(&mut self, card: &Card, kind: ElementKind, higher_order_nodes: usize) -> Result<()> {
        let n = kind.node_count();
        let eid = field_id(card, 1, ELEMENT_ID)?;
        let pid = field_id(card, 2, PROPERTY_ID)?;
        let last = card.last_non_blank(3);
        if last > 3 + n {
            return Err(FacetSetFemError::unsupported(crate::tr!(
                "The {1}-node version of {0} (second-order element) is not supported",
                "{} の {} 節点版（2 次要素）は未対応です",
                kind.card_name(),
                higher_order_nodes
            ))
            .with_line(card.line())
            .with_card(kind.card_name())
            .with_entity(crate::tr!("element EID={eid}", "要素 EID={eid}"))
            .with_hint(crate::tr!(
                "Second-order elements are supported for CTETRA (10 nodes) only (SPEC §2.3). \
                 Convert to the {}-node {}, or use a tetrahedral mesh",
                "2 次要素は CTETRA（10 節点）だけに対応しています（SPEC §2.3）。\
                 {} 節点の {} に変換するか、四面体のメッシュを使ってください",
                n,
                kind.card_name()
            )));
        }
        let block = self.blocks.entry(kind).or_insert_with(RawBlock::new);
        block.ids.push(eid);
        block.pids.push(pid);
        block.lines.push(card.line() as u32);
        for (k, name) in NODE_FIELD_NAMES.iter().take(n).enumerate() {
            block.conn_ids.push(field_id(card, 3 + k, *name)?);
        }
        Ok(())
    }

    /// `CTRIA6 EID PID G1 G2 G3 [G4 G5 G6] [THETA ZOFFS T1 T2 T3 TFLAG]`
    ///
    /// G4〜G6（辺の中点）が全て空欄なら 1 次の Tria3、全て書かれていれば 2 次の Tria6。
    /// 一部だけのものは CTETRA と同じく扱わない。後ろに構造用の欄が続くので、
    /// CTETRA のように「最後の空でない欄」では判定できず、G4〜G6 を 1 つずつ見る。
    fn tria6(&mut self, card: &Card) -> Result<()> {
        let blank: Vec<usize> = (3..6).filter(|&k| card.field(3 + k).is_empty()).collect();
        match blank.len() {
            0 => self.shell(card, ElementKind::Tria6),
            3 => self.shell(card, ElementKind::Tria3),
            _ => {
                let eid = field_id(card, 1, ELEMENT_ID)?;
                let names: Vec<&str> = blank
                    .iter()
                    .map(|&k| {
                        let (en, ja) = NODE_FIELD_NAMES[k];
                        if crate::i18n::is_ja() { ja } else { en }
                    })
                    .collect();
                Err(FacetSetFemError::unsupported(crate::tr!(
                    "A CTRIA6 with only some of its mid-side nodes is not supported (blank: {})",
                    "辺の中点の一部だけを書いた CTRIA6 には対応していません（空欄: {}）",
                    names.join(", ")
                ))
                .with_line(card.line())
                .with_card("CTRIA6")
                .with_entity(crate::tr!("element EID={eid}", "要素 EID={eid}"))
                .with_hint(crate::tr!(
                    "Write all three mid-side nodes G4–G6, or leave all of them blank",
                    "辺の中点 G4〜G6 を 3 つとも書くか、すべて空欄にしてください"
                )))
            }
        }
    }

    /// `CTRIA3 / CQUAD4 EID PID G1 ... [THETA/MCID ZOFFS]`
    fn shell(&mut self, card: &Card, kind: ElementKind) -> Result<()> {
        let n = kind.node_count();
        let eid = field_id(card, 1, ELEMENT_ID)?;
        let pid = field_id(card, 2, PROPERTY_ID)?;
        let block = self.blocks.entry(kind).or_insert_with(RawBlock::new);
        block.ids.push(eid);
        block.pids.push(pid);
        block.lines.push(card.line() as u32);
        for (k, name) in NODE_FIELD_NAMES.iter().take(n).enumerate() {
            block.conn_ids.push(field_id(card, 3 + k, *name)?);
        }
        // THETA/MCID・ZOFFS 以降は構造解析用なので読み飛ばす
        Ok(())
    }

    /// `PSOLID PID MID ...` / `PSHELL PID MID1 T ...`
    fn property(&mut self, card: &Card, kind: &'static str) -> Result<()> {
        let pid = field_id(card, 1, PROPERTY_ID)?;
        let mid = field_int(card, 2)?.and_then(|v| u32::try_from(v).ok());
        let prop = Property {
            pid,
            mid,
            card: kind,
        };
        if let Some(prev) = self.properties.insert(pid, prop) {
            if prev.card == kind {
                self.warnings.push(crate::tr!(
                    "{kind} PID={pid} is defined more than once (the one on line {} is used)",
                    "{kind} の PID={pid} が複数回定義されています（{} 行目のものを使います）",
                    card.line()
                ));
            } else {
                return Err(FacetSetFemError::model(crate::tr!(
                    "PID={pid} is defined by both {} and {kind}",
                    "PID={pid} が {} と {kind} の両方で定義されています",
                    prev.card
                ))
                .with_line(card.line())
                .with_card(kind)
                .with_hint(crate::tr!(
                    "A PID identifies a region (PSOLID) or a candidate electrode surface (PSHELL), \
                     so it must be unique across all property cards. Change one of the PIDs",
                    "PID は領域（PSOLID）と電極候補面（PSHELL）を識別する鍵なので、\
                     全プロパティカードを通して一意でなければなりません。片方の PID を変更してください"
                )));
            }
        }
        Ok(())
    }

    fn finish(self, name: &Path, lines: usize) -> Result<(Mesh, ReadReport)> {
        let Builder {
            unit,
            node_ids,
            mut node_coords,
            node_cp,
            coord_definitions,
            blocks,
            properties,
            card_counts,
            unsupported,
            mut warnings,
        } = self;

        // --- 座標変換と単位換算（SPEC §3.1 / §3.2）---
        //
        // CORD2* の A・B・C はファイルの単位で書かれているので、変換を先に済ませ、
        // SI への換算は最後にまとめて掛ける
        if !node_cp.is_empty() {
            let systems = coord::resolve(&coord_definitions)?;
            for &(index, cid) in &node_cp {
                let Some(system) = systems.get(cid) else {
                    return Err(FacetSetFemError::model(crate::tr!(
                        "The coordinate system CID={cid} referenced by GRID is not defined",
                        "GRID が参照する座標系 CID={cid} が定義されていません"
                    ))
                    .with_card("GRID")
                    .with_entity(crate::tr!(
                        "node ID={}",
                        "節点 ID={}",
                        node_ids[index as usize]
                    ))
                    .with_hint(crate::tr!(
                        "Define that CID with CORD2R / CORD2C / CORD2S",
                        "CORD2R / CORD2C / CORD2S でその CID を定義してください"
                    )));
                };
                let k = index as usize * 3;
                let local = [node_coords[k], node_coords[k + 1], node_coords[k + 2]];
                node_coords[k..k + 3].copy_from_slice(&system.to_basic(local));
            }
        }
        let scale = unit.to_meters();
        if scale != 1.0 {
            for value in &mut node_coords {
                *value *= scale;
            }
        }

        if node_ids.is_empty() {
            return Err(FacetSetFemError::model(crate::tr!(
                "There are no GRID cards",
                "GRID カードが 1 枚もありません"
            ))
            .with_hint(crate::tr!(
                "Check that the bulk data section (BEGIN BULK ... ENDDATA) contains GRID cards \
                 and that the file is not empty",
                "バルクデータセクション（BEGIN BULK 〜 ENDDATA）に GRID があるか、\
                 ファイルが空でないかを確認してください"
            )));
        }

        // --- 節点 ID → 索引 ---
        let idmap = IdMap::build(&node_ids).map_err(|dups| {
            let list: Vec<String> = dups.iter().map(|d| d.to_string()).collect();
            FacetSetFemError::model(crate::tr!(
                "Duplicate node IDs ({}): {}",
                "節点 ID が重複しています（{} 件）: {}",
                dups.len(),
                format_sample_list(&list, SAMPLE_LIMIT)
            ))
            .with_card("GRID")
            .with_hint(crate::tr!(
                "Two or more GRID cards have the same ID. Remove the duplicates",
                "同じ ID の GRID が 2 枚以上あります。重複を取り除いてください"
            ))
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
                        return Err(FacetSetFemError::model(crate::tr!(
                            "Node ID {gid} is not defined as a GRID",
                            "節点 ID {gid} が GRID として定義されていません"
                        ))
                        .with_line(raw.lines[e] as usize)
                        .with_card(kind.card_name())
                        .with_field(3 + k)
                        .with_entity(crate::tr!("element EID={}", "要素 EID={}", raw.ids[e]))
                        .with_hint(crate::tr!(
                            "Add a GRID card with ID {gid}, or fix the node IDs of the {}",
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
            return Err(FacetSetFemError::model(crate::tr!(
                "Duplicate element IDs ({}): {}",
                "要素 ID が重複しています（{} 件）: {}",
                dup_elems.len(),
                format_sample_list(&dup_elems, SAMPLE_LIMIT)
            ))
            .with_hint(crate::tr!(
                "Element IDs must be unique across all element cards",
                "要素 ID は全要素カードを通して一意でなければなりません"
            )));
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
                        warnings.push(crate::tr!(
                            "PID={pid} is defined by {}, but is referenced by {} elements",
                            "PID={pid} は {} で定義されていますが、{} 要素から参照されています",
                            p.card,
                            if is_solid {
                                crate::tr!("solid", "ソリッド")
                            } else {
                                crate::tr!("shell", "シェル")
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
                    warnings.push(crate::tr!(
                        "Some elements reference PID={pid}, but there is no matching {} card. \
                         It was registered as a region automatically",
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
            warnings.push(crate::tr!(
                "Some properties are not referenced by any element ({}): PID={}",
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
            warnings.push(crate::tr!(
                "Some isolated nodes are not referenced by any element ({}): ID={}",
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
            warnings.push(crate::tr!(
                "There are no solid elements (CTETRA / CPENTA / CHEXA). The model cannot be analysed",
                "ソリッド要素（CTETRA / CPENTA / CHEXA）が 1 つもありません。解析はできません"
            ));
        }
        check_tet4_geometry(&mesh, &mut warnings)?;
        check_isoparametric_geometry(&mesh)?;
        check_tet10_geometry(&mesh, &mut warnings)?;
        // 次数の混在・中点の食い違いは解く前にも必ず止める（SPEC §2.3）。
        // 表示だけしたい場合があるので、読み込みでは警告に留める（孤立節点と同じ扱い）
        if let Err(e) = crate::mesh::check_orders(&mesh) {
            warnings.push(match e.hint() {
                Some(hint) => format!("{} ({hint})", e.message()),
                None => e.message().to_string(),
            });
        }

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
            sha256: None,
        };
        Ok((mesh, report))
    }
}

/// Tet4 の退化・節点順序を検査する（SPEC §3.1）。
///
/// - 体積が数値的にゼロ ＝ 潰れた要素。剛性行列が作れないのでエラー
/// - 体積が負 ＝ 節点順序の反転。要素剛性は |V| を使うので計算は可能。警告に留める
///
/// Hex8 / Penta6 は積分点ごとにヤコビアンを見る必要があるので
/// [`check_isoparametric_geometry`] で別に扱う。
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
        return Err(FacetSetFemError::model(crate::tr!(
            "Some CTETRA elements are degenerate with zero volume ({}): EID={}",
            "体積がゼロの退化した CTETRA があります（{} 件）: EID={}",
            degenerate.len(),
            format_sample_list(&degenerate, SAMPLE_LIMIT)
        ))
        .with_card("CTETRA")
        .with_hint(crate::tr!(
            "The 4 nodes lie in one plane, or the same node is referenced more than once. \
             Fix these elements in the mesh generator",
            "4 節点が同一平面上にあるか、同じ節点を重複して参照しています。\
             メッシュ生成側で該当要素を修正してください"
        )));
    }
    if !inverted.is_empty() {
        warnings.push(crate::tr!(
            "Some CTETRA elements have reversed node order (negative signed volume) ({}): EID={}. \
             The element stiffness uses |V|, so the results are not affected",
            "節点順序が反転している（符号付き体積が負の）CTETRA があります（{} 件）: EID={}。\
             要素剛性は |V| で計算するため結果には影響しません",
            inverted.len(),
            format_sample_list(&inverted, SAMPLE_LIMIT)
        ));
    }
    Ok(())
}

/// Hex8 / Penta6 のヤコビアンを検査する（SPEC §2.3 / §3.1）。
///
/// これらの要素は Tet4 と違い `det J` が要素内で一定でないため、
/// **全ての積分点**で正であることを確かめる必要がある。1 点でも `det J ≤ 0` なら
/// 要素剛性が作れないので、EID を挙げてエラーにする。
///
/// Tet4 の「符号付き体積が負なら警告」（節点順序の反転にすぎない）と扱いが違うのは、
/// 六面体・五面体では負のヤコビアンが単なる反転とは限らず、
/// 面がねじれて自己交差している場合があるため。どちらなのかを読み込み時に
/// 判定するより、EID を出して直してもらうほうが確実で親切。
///
/// アセンブリでも同じ検査をしているが（`element::shape::stiffness`）、
/// ここで見ておくと**全ての不正要素を一度に**報告できる。
fn check_isoparametric_geometry(mesh: &Mesh) -> Result<()> {
    for kind in [ElementKind::Hex8, ElementKind::Penta6] {
        let Some(block) = mesh.block(kind) else {
            continue;
        };
        let mut bad: Vec<String> = Vec::new();
        for e in 0..block.len() {
            let conn = block.conn_of(e);
            let mut buffer = [[0.0f64; 3]; shape::MAX_ELEMENT_NODES];
            let points = &mut buffer[..conn.len()];
            mesh.gather_coords(conn, points);
            let volume = match kind {
                ElementKind::Hex8 => shape::volume::<Hex8>(points),
                _ => shape::volume::<Penta6>(points),
            };
            let h = mesh.element_max_edge(block, e);
            let degenerate = match volume {
                Ok(v) => h <= 0.0 || v <= DEGENERATE_VOLUME_RATIO * h * h * h,
                Err(_) => true,
            };
            if degenerate {
                bad.push(block.ids()[e].to_string());
            }
        }
        if !bad.is_empty() {
            let card = kind.card_name();
            let order = match kind {
                ElementKind::Hex8 => crate::tr!(
                    "G1–G4 form the bottom face and G5–G8 lie directly above them (counterclockwise seen from the bottom face)",
                    "G1〜G4 が底面、G5〜G8 がその真上（底面から見て反時計回り）"
                ),
                _ => crate::tr!(
                    "G1–G3 form the bottom face and G4–G6 lie directly above them (counterclockwise seen from the bottom face)",
                    "G1〜G3 が底面、G4〜G6 がその真上（底面から見て反時計回り）"
                ),
            };
            return Err(FacetSetFemError::model(crate::tr!(
                "Some {card} elements have a non-positive Jacobian ({}): EID={}",
                "ヤコビアンが正でない {card} があります（{} 件）: EID={}",
                bad.len(),
                format_sample_list(&bad, SAMPLE_LIMIT)
            ))
            .with_card(card)
            .with_hint(crate::tr!(
                "The node order is reversed, or the element is collapsed or twisted. For {card}, {order}. \
                 Fix these elements in the mesh generator",
                "節点の並びが逆か、要素が潰れている／ねじれています。{card} の並びは {order} です。\
                 メッシュ生成側で該当要素を修正してください"
            )));
        }
    }
    Ok(())
}

/// 10 節点の CTETRA のヤコビアンを検査する（SPEC §2.3 / §3.1）。
///
/// 辺が曲がりうるので、10 節点と積分点の全てで `det J` の符号を見る
/// （[`tet10::orientation`]）。
///
/// - 符号が混じる・ゼロになる ＝ 中点が辺の中央から離れすぎて要素が裏返っている。エラー
/// - 全ての点で負 ＝ 節点順序の反転。Tet4 と同じく |det J| で計算できるので警告に留める
/// - 体積が数値的にゼロ ＝ 潰れた要素。エラー
fn check_tet10_geometry(mesh: &Mesh, warnings: &mut Vec<String>) -> Result<()> {
    let Some(block) = mesh.block(ElementKind::Tet10) else {
        return Ok(());
    };
    let mut bad: Vec<String> = Vec::new();
    let mut inverted: Vec<String> = Vec::new();
    for e in 0..block.len() {
        let mut points = [[0.0f64; 3]; 10];
        mesh.gather_coords(block.conn_of(e), &mut points);
        let h = mesh.element_max_edge(block, e);
        let degenerate = match tet10::volume(&points) {
            Ok(v) => h <= 0.0 || v <= DEGENERATE_VOLUME_RATIO * h * h * h,
            Err(_) => true,
        };
        if degenerate {
            bad.push(block.ids()[e].to_string());
        } else if tet10::orientation(&points) == Orientation::Inverted {
            inverted.push(block.ids()[e].to_string());
        }
    }
    if !bad.is_empty() {
        return Err(FacetSetFemError::model(crate::tr!(
            "Some 10-node CTETRA elements are collapsed, or their Jacobian changes sign inside \
             the element ({}): EID={}",
            "潰れているか、要素の中でヤコビアンの符号が変わる 10 節点の CTETRA があります\
             （{} 件）: EID={}",
            bad.len(),
            format_sample_list(&bad, SAMPLE_LIMIT)
        ))
        .with_card("CTETRA")
        .with_hint(crate::tr!(
            "A mid-side node (G5–G10) is too far from the middle of its edge, or the corners \
             lie in one plane. Fix these elements in the mesh generator",
            "辺の中点（G5〜G10）が辺の中央から離れすぎているか、頂点が同一平面上にあります。\
             メッシュ生成側で該当要素を修正してください"
        )));
    }
    if !inverted.is_empty() {
        warnings.push(crate::tr!(
            "Some 10-node CTETRA elements have reversed node order (negative Jacobian) ({}): \
             EID={}. The element stiffness uses |det J|, so the results are not affected",
            "節点順序が反転している（ヤコビアンが負の）10 節点の CTETRA があります（{} 件）: \
             EID={}。要素剛性は |det J| で計算するため結果には影響しません",
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
        FacetSetFemError::syntax(crate::tr!(
            "Cannot interpret as an integer: \"{s}\"",
            "整数として解釈できません: \"{s}\""
        ))
        .with_line(card.line())
        .with_card(card.name())
        .with_field(i)
        .with_hint(crate::tr!(
            "Write a signed decimal integer",
            "符号付きの十進整数を書いてください"
        ))
    })
}

fn field_real(card: &Card, i: usize, default: f64) -> Result<f64> {
    let s = card.field(i);
    if s.is_empty() {
        return Ok(default);
    }
    parse_real(s).ok_or_else(|| {
        FacetSetFemError::syntax(crate::tr!(
            "Cannot interpret as a real number: \"{s}\"",
            "実数として解釈できません: \"{s}\""
        ))
        .with_line(card.line())
        .with_card(card.name())
        .with_field(i)
        .with_hint(crate::tr!(
            "Write it in a Nastran real format (e.g. 1.0, 7., .5, 1.5+3, 2.5E-2)",
            "Nastran の実数形式（例: 1.0, 7., .5, 1.5+3, 2.5E-2）で書いてください"
        ))
    })
}

/// ID フィールド（1 以上の整数）を読む。
fn field_id(card: &Card, i: usize, what: FieldName) -> Result<u32> {
    // 名前はエラーのときだけ選ぶ（要素の節点ごとに呼ばれるので、正常系で言語を引かない）
    let name = || if crate::i18n::is_ja() { what.1 } else { what.0 };
    let Some(v) = field_int(card, i)? else {
        let what = name();
        return Err(
            FacetSetFemError::syntax(crate::tr!("{what} is empty", "{what} が空です"))
                .with_line(card.line())
                .with_card(card.name())
                .with_field(i)
                .with_hint(crate::tr!(
                    "This field is required. Write an integer of 1 or more",
                    "必須フィールドです。1 以上の整数を書いてください"
                )),
        );
    };
    if v < 1 || v > u32::MAX as i64 {
        let what = name();
        return Err(FacetSetFemError::new(
            ErrorKind::Model,
            crate::tr!("{what} is out of range: {v}", "{what} が範囲外です: {v}"),
        )
        .with_line(card.line())
        .with_card(card.name())
        .with_field(i)
        .with_hint(crate::tr!(
            "An ID must be between 1 and {}",
            "ID は 1 以上 {} 以下でなければなりません",
            u32::MAX
        )));
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

    /// 頂点 (0,0,0), (1,0,0), (0,1,0), (0,0,1) の四面体と、その辺の中点の GRID（ID 1〜10）。
    const TET10_GRIDS: &str = "\
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
GRID,4,,0.0,0.0,1.0
GRID,5,,0.5,0.0,0.0
GRID,6,,0.5,0.5,0.0
GRID,7,,0.0,0.5,0.0
GRID,8,,0.0,0.0,0.5
GRID,9,,0.5,0.0,0.5
GRID,10,,0.0,0.5,0.5
PSOLID,1,1
";

    fn node_ids(mesh: &Mesh, e: usize) -> Vec<u32> {
        mesh.solids[0]
            .conn_of(e)
            .iter()
            .map(|&i| mesh.nodes.id(i))
            .collect()
    }

    #[test]
    fn a_ten_node_ctetra_is_read_as_tet10() {
        // 自由形式: 親の行に G1〜G6、継続行に G7〜G10
        let text = format!("{TET10_GRIDS}CTETRA,10,1,1,2,3,4,5,6\n,7,8,9,10\n");
        let (mesh, report) = read(&text, LengthUnit::Meter).unwrap();
        assert_eq!(mesh.solids.len(), 1);
        assert_eq!(mesh.solids[0].kind(), ElementKind::Tet10);
        assert_eq!(node_ids(&mesh, 0), (1..=10).collect::<Vec<u32>>());
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
    }

    #[test]
    fn a_ten_node_ctetra_in_small_field_format_is_read() {
        // 小フィールド: 1 行に 8 欄（G1〜G6）、継続行に G7〜G10
        let text = format!(
            "{TET10_GRIDS}{:<8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}+\n\
             {:<8}{:>8}{:>8}{:>8}{:>8}\n",
            "CTETRA", 10, 1, 1, 2, 3, 4, 5, 6, "+", 7, 8, 9, 10
        );
        let (mesh, _) = read(&text, LengthUnit::Meter).unwrap();
        assert_eq!(mesh.solids[0].kind(), ElementKind::Tet10);
        assert_eq!(node_ids(&mesh, 0), (1..=10).collect::<Vec<u32>>());
    }

    #[test]
    fn ctetra_with_blank_mid_side_nodes_is_first_order() {
        // G5〜G10 が全て空欄なら 1 次
        let text = format!("{TET10_GRIDS}CTETRA,10,1,1,2,3,4,,\n,,,,\n");
        let (mesh, _) = read(&text, LengthUnit::Meter).unwrap();
        assert_eq!(mesh.solids[0].kind(), ElementKind::Tet4);
    }

    #[test]
    fn ctetra_with_some_mid_side_nodes_gives_a_helpful_error() {
        // 自由形式では行末の空欄が落ちるので、空欄は行の途中に置く
        let text = format!("{TET10_GRIDS}CTETRA,10,1,1,2,3,4,5,,7\n,8,,10\n");
        let e = read(&text, LengthUnit::Meter).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("一部だけ"), "{s}");
        assert!(s.contains("G6") && s.contains("G9"), "{s}");
        assert!(!s.contains("G7"), "{s}");
    }

    #[test]
    fn second_order_pentahedra_are_still_rejected() {
        let text = "\
GRID,1,,0.0,0.0,0.0
CPENTA,10,1,1,2,3,4,5,6
,7,8,9
";
        let e = read(text, LengthUnit::Meter).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("2 次要素"), "{s}");
        assert!(s.contains("CTETRA（10 節点）"), "{s}");
    }

    #[test]
    fn an_inverted_ten_node_ctetra_is_only_a_warning() {
        // 頂点 G2 と G3 を入れ替え、中点もそれに合わせて並べる
        let text = format!("{TET10_GRIDS}CTETRA,10,1,1,3,2,4,7,6\n,5,8,10,9\n");
        let (mesh, report) = read(&text, LengthUnit::Meter).unwrap();
        assert_eq!(mesh.solids[0].kind(), ElementKind::Tet10);
        assert!(
            report.warnings.iter().any(|w| w.contains("反転")),
            "{:?}",
            report.warnings
        );
    }

    #[test]
    fn a_folded_ten_node_ctetra_is_an_error() {
        // G5（辺 G1-G2 の中点）を反対側の頂点の先まで動かすと、要素が裏返る
        let text = TET10_GRIDS.replace("GRID,5,,0.5,0.0,0.0", "GRID,5,,-0.25,1.5,0.0")
            + "CTETRA,10,1,1,2,3,4,5,6\n,7,8,9,10\n";
        let e = read(&text, LengthUnit::Meter).unwrap_err();
        assert!(e.to_string().contains("ヤコビアンの符号"), "{e}");
    }

    #[test]
    fn ctria6_is_read_with_or_without_its_mid_side_nodes() {
        // G4〜G6 があれば 2 次の Tria6、全て空欄なら 1 次の Tria3。後ろの THETA などは読み飛ばす
        let text = format!(
            "{TET10_GRIDS}PSHELL,101,1,0.1\nCTRIA6,100,101,1,3,2,7,6,5\n,0.0\n\
             CTRIA6,101,101,1,2,4,,,\n"
        );
        let (mesh, _) = read(&text, LengthUnit::Meter).unwrap();
        let kinds: Vec<ElementKind> = mesh.shells.iter().map(|b| b.kind()).collect();
        assert_eq!(kinds, [ElementKind::Tria3, ElementKind::Tria6]);
        let tria6 = mesh.block(ElementKind::Tria6).unwrap();
        let ids: Vec<u32> = tria6.conn_of(0).iter().map(|&i| mesh.nodes.id(i)).collect();
        assert_eq!(ids, [1, 3, 2, 7, 6, 5]);
    }

    #[test]
    fn ctria6_with_some_mid_side_nodes_gives_a_helpful_error() {
        let text = format!("{TET10_GRIDS}CTRIA6,100,101,1,3,2,7,,5\n");
        let e = read(&text, LengthUnit::Meter).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("一部だけ") && s.contains("G5"), "{s}");
    }

    #[test]
    fn mixed_orders_are_a_warning_when_reading() {
        // 表示だけならできるので読み込みは通す。解く前には止める（mesh::check_orders）
        let text = format!(
            "{TET10_GRIDS}GRID,11,,1.0,1.0,1.0\n\
             CTETRA,10,1,1,2,3,4,5,6\n,7,8,9,10\n\
             CTETRA,11,1,2,3,4,11\n"
        );
        let (mesh, report) = read(&text, LengthUnit::Meter).unwrap();
        assert_eq!(mesh.solids.len(), 2);
        assert!(
            report.warnings.iter().any(|w| w.contains("混ざって")),
            "{:?}",
            report.warnings
        );
    }

    #[test]
    fn grid_in_a_cylindrical_system_is_converted() {
        // CID 7 は基本座標系に揃えた円筒座標。(r=2, θ=90°, z=3) → (0, 2, 3)
        let text = "\
CORD2C,7,0,0.0,0.0,0.0,0.0,0.0,1.0
,1.0,0.0,0.0
GRID,1,7,2.0,90.0,3.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
GRID,4,,0.0,0.0,1.0
PSOLID,1,1
CTETRA,10,1,1,2,3,4
";
        let (mesh, report) = read(text, LengthUnit::Meter).unwrap();
        let p = mesh.nodes.coord(0);
        assert!(p[0].abs() < 1e-12, "{p:?}");
        assert!((p[1] - 2.0).abs() < 1e-12, "{p:?}");
        assert!((p[2] - 3.0).abs() < 1e-12, "{p:?}");
        // CP を使わない節点はそのまま
        assert_eq!(mesh.nodes.coord(1), [1.0, 0.0, 0.0]);
        assert!(report.unsupported.is_empty(), "{:?}", report.unsupported);
    }

    #[test]
    fn a_metre_read_can_take_on_another_unit_without_rereading() {
        // GUI は寸法を見せるために m として読み、選ばれた単位に付け替える。
        // 最初からその単位で読んだものとビット単位で同じ座標にならなければならない
        // （座標変換を含むメッシュでも）
        let text = "\
CORD2C,7,0,0.1,0.2,0.3,0.1,0.2,1.3
,1.1,0.2,0.3
GRID,1,7,2.3,37.0,3.1
GRID,2,,1.7,0.3,0.1
GRID,3,,0.2,1.9,0.4
GRID,4,,0.3,0.1,1.3
PSOLID,1,1
CTETRA,10,1,1,2,3,4
";
        let (as_metre, metre_report) = read(text, LengthUnit::Meter).unwrap();
        for unit in [
            LengthUnit::Millimeter,
            LengthUnit::Centimeter,
            LengthUnit::Inch,
            LengthUnit::Meter,
        ] {
            let (direct, report) = read(text, unit).unwrap();
            let relabelled = as_metre.reinterpret_length_unit(unit).unwrap();
            assert_eq!(relabelled.nodes.coords(), direct.nodes.coords(), "{unit}");
            assert_eq!(relabelled.length_unit, unit);
            // 形状のチェックは寸法によらないので、警告も同じ
            assert_eq!(report.warnings, metre_report.warnings);
        }
        // m 以外で読んだものからは付け替えない（割り戻しで丸めが入る）
        let (as_mm, _) = read(text, LengthUnit::Millimeter).unwrap();
        assert!(as_mm.reinterpret_length_unit(LengthUnit::Meter).is_err());
    }

    #[test]
    fn coordinate_transform_happens_before_the_unit_conversion() {
        // 円筒座標の r は長さなので単位換算の対象、θ は度なので対象外
        let text = "\
CORD2C,7,0,0.0,0.0,0.0,0.0,0.0,1.0
,1.0,0.0,0.0
GRID,1,7,2.0,90.0,3.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
GRID,4,,0.0,0.0,1.0
PSOLID,1,1
CTETRA,10,1,1,2,3,4
";
        let (mesh, _) = read(text, LengthUnit::Millimeter).unwrap();
        let p = mesh.nodes.coord(0);
        assert!((p[1] - 2.0e-3).abs() < 1e-18, "{p:?}");
        assert!((p[2] - 3.0e-3).abs() < 1e-18, "{p:?}");
    }

    #[test]
    fn grid_referencing_an_undefined_system_is_reported() {
        let text = "GRID,1,7,0.0,0.0,0.0\n";
        let e = read(text, LengthUnit::Meter).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("CID=7"), "{s}");
        assert!(s.contains("節点 ID=1"), "{s}");
    }

    #[test]
    fn rectangular_system_shifts_the_origin() {
        let text = "\
CORD2R,5,0,10.0,20.0,30.0,10.0,20.0,31.0
,11.0,20.0,30.0
GRID,1,5,1.0,2.0,3.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
GRID,4,,0.0,0.0,1.0
PSOLID,1,1
CTETRA,10,1,1,2,3,4
";
        let (mesh, _) = read(text, LengthUnit::Meter).unwrap();
        assert_eq!(mesh.nodes.coord(0), [11.0, 22.0, 33.0]);
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
