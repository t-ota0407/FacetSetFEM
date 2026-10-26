//! Nastran バルクデータの「論理カード」ストリーミング読み出し（SPEC §3.1）。
//!
//! 物理行と論理カードは 1 対 1 ではない。継続行・書式の混在・コメント・
//! Executive / Case Control セクションの読み飛ばしをここで吸収し、
//! 上位（`parse.rs`）には「カード名 + データフィールド列」だけを渡す。
//!
//! 1000 万節点を読むため、カードは呼び出し側が用意した [`Card`] を使い回して
//! 1 カードあたりのアロケーションを増やさない設計にしている。

use super::field::{LineForm, is_continuation, line_form, split_fixed, split_free};
use crate::error::{ErrorKind, FacetSetFemError, Result};

/// `BEGIN BULK` を探しながら退避しておく行数の上限。
///
/// Executive / Case Control セクションがこれより長いことは実務上あり得ない。
/// 上限に達しても `BEGIN BULK` が見つからない場合は「BEGIN BULK の無い
/// バルクデータのみのファイル」とみなし、退避した行をバルクとして読み直す。
const PRE_BULK_LINE_LIMIT: usize = 10_000;

/// 論理カード 1 枚。フィールドは 1 本のバッファに連結して保持する。
#[derive(Debug, Default, Clone)]
pub struct Card {
    line: usize,
    buf: String,
    /// 各フィールドの終端位置（`buf` 内のバイト位置）
    ends: Vec<u32>,
}

impl Card {
    /// 空のカードを作る。
    pub fn new() -> Self {
        Self::default()
    }

    /// このカードが始まる行番号（1 始まり）。
    pub fn line(&self) -> usize {
        self.line
    }

    /// フィールド数（0 番＝カード名を含む）。
    pub fn n_fields(&self) -> usize {
        self.ends.len()
    }

    /// カード名（大文字化済み、`*` を除去済み）。
    pub fn name(&self) -> &str {
        self.field(0)
    }

    /// `i` 番のフィールド。範囲外は空文字列（＝既定値）を返す。
    pub fn field(&self, i: usize) -> &str {
        if i >= self.ends.len() {
            return "";
        }
        let start = if i == 0 { 0 } else { self.ends[i - 1] as usize };
        &self.buf[start..self.ends[i] as usize]
    }

    /// `from` 番以降で、値が入っている最後のフィールド番号 + 1。
    pub fn last_non_blank(&self, from: usize) -> usize {
        let mut last = from;
        for i in from..self.n_fields() {
            if !self.field(i).is_empty() {
                last = i + 1;
            }
        }
        last
    }

    fn reset(&mut self, line: usize) {
        self.line = line;
        self.buf.clear();
        self.ends.clear();
    }

    fn push_field(&mut self, s: &str) {
        self.buf.push_str(s);
        self.ends.push(self.buf.len() as u32);
    }

    fn push_field_upper(&mut self, s: &str) {
        for c in s.chars() {
            self.buf.push(c.to_ascii_uppercase());
        }
        self.ends.push(self.buf.len() as u32);
    }

    /// 末尾の空フィールドを落とす（カード名は必ず残す）。
    fn trim_trailing_blanks(&mut self) {
        while self.ends.len() > 1 {
            let n = self.ends.len();
            if self.ends[n - 2] == self.ends[n - 1] {
                self.ends.pop();
            } else {
                break;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// `BEGIN BULK` を探している
    PreBulk,
    /// バルクデータ本体
    Bulk,
    /// `ENDDATA` またはファイル末尾に到達した
    Done,
}

/// 行の供給元。先読み（push back）と `BEGIN BULK` 探索の退避を担当する。
#[derive(Debug)]
struct LineSource<R> {
    inner: R,
    lineno: usize,
    eof: bool,
    phase: Phase,
    pre_buffer: Vec<(usize, String)>,
    replay: Vec<(usize, String)>,
    replay_idx: usize,
    pushback: Option<(usize, String)>,
    buf: String,
}

impl<R: std::io::BufRead> LineSource<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            lineno: 0,
            eof: false,
            phase: Phase::PreBulk,
            pre_buffer: Vec::new(),
            replay: Vec::new(),
            replay_idx: 0,
            pushback: None,
            buf: String::new(),
        }
    }

    fn raw_line(&mut self) -> Result<Option<(usize, String)>> {
        if self.eof {
            return Ok(None);
        }
        self.buf.clear();
        let n = self.inner.read_line(&mut self.buf).map_err(|e| {
            FacetSetFemError::new(
                ErrorKind::Io,
                crate::tr!(
                    "Failed to read the file",
                    "ファイルの読み取りに失敗しました"
                ),
            )
            .with_line(self.lineno + 1)
            .with_hint(crate::tr!(
                "Check that the file is ASCII / UTF-8 text",
                "ファイルが ASCII / UTF-8 のテキストであることを確認してください"
            ))
            .with_source(e)
        })?;
        if n == 0 {
            self.eof = true;
            return Ok(None);
        }
        self.lineno += 1;
        let s = self.buf.trim_end_matches(['\n', '\r']).to_string();
        Ok(Some((self.lineno, s)))
    }

    /// バルクデータの次の行を返す。
    fn next_line(&mut self) -> Result<Option<(usize, String)>> {
        if let Some(l) = self.pushback.take() {
            return Ok(Some(l));
        }
        loop {
            if self.replay_idx < self.replay.len() {
                let l = std::mem::take(&mut self.replay[self.replay_idx]);
                self.replay_idx += 1;
                return Ok(Some(l));
            }
            match self.phase {
                Phase::Done => return Ok(None),
                Phase::Bulk => return self.raw_line(),
                Phase::PreBulk => match self.raw_line()? {
                    None => {
                        // BEGIN BULK が無いままファイル末尾に達した
                        // → バルクデータのみのファイルとみなして退避分を読み直す
                        self.phase = Phase::Bulk;
                        self.start_replay();
                        if self.replay.is_empty() {
                            self.phase = Phase::Done;
                            return Ok(None);
                        }
                    }
                    Some((n, s)) => {
                        if is_begin_bulk(&s) {
                            self.pre_buffer.clear();
                            self.phase = Phase::Bulk;
                        } else {
                            self.pre_buffer.push((n, s));
                            if self.pre_buffer.len() >= PRE_BULK_LINE_LIMIT {
                                self.phase = Phase::Bulk;
                                self.start_replay();
                            }
                        }
                    }
                },
            }
        }
    }

    fn start_replay(&mut self) {
        self.replay = std::mem::take(&mut self.pre_buffer);
        self.replay_idx = 0;
    }

    fn push_back(&mut self, line: (usize, String)) {
        debug_assert!(self.pushback.is_none());
        self.pushback = Some(line);
    }

    /// `ENDDATA` に到達した。以降は何も返さない。
    fn finish(&mut self) {
        self.phase = Phase::Done;
        self.pushback = None;
        self.replay.clear();
        self.replay_idx = 0;
        self.pre_buffer.clear();
    }
}

/// `BEGIN BULK` 行か。
fn is_begin_bulk(line: &str) -> bool {
    let mut it = line.split_whitespace();
    matches!(
        (it.next(), it.next()),
        (Some(a), Some(b)) if a.eq_ignore_ascii_case("BEGIN") && b.eq_ignore_ascii_case("BULK")
    )
}

/// 行末コメント（`$` 以降）を落とす。
///
/// Nastran のフィールド値に `$` が現れることは無いため、位置によらず切ってよい。
/// これにより「コメント中のカンマで自由形式と誤判定する」事故も防げる。
fn strip_comment(line: &str) -> &str {
    match line.find('$') {
        Some(i) => &line[..i],
        None => line,
    }
}

/// バルクデータを論理カード単位で読み出す。
#[derive(Debug)]
pub struct BulkReader<R> {
    src: LineSource<R>,
}

impl<R: std::io::BufRead> BulkReader<R> {
    /// 生成。
    pub fn new(inner: R) -> Self {
        Self {
            src: LineSource::new(inner),
        }
    }

    /// 読み終えた物理行数。
    pub fn lines_read(&self) -> usize {
        self.src.lineno
    }

    /// 次の論理カードを `card` に読み込む。カードが尽きたら `Ok(false)`。
    pub fn next_card(&mut self, card: &mut Card) -> Result<bool> {
        // --- 親カードの行を探す ---
        let (lineno, line) = loop {
            let Some((n, raw)) = self.src.next_line()? else {
                return Ok(false);
            };
            let body = strip_comment(&raw);
            if body.trim().is_empty() {
                continue;
            }
            if body.trim().eq_ignore_ascii_case("ENDDATA") {
                self.src.finish();
                return Ok(false);
            }
            if is_begin_bulk(body) {
                // ファイル中に 2 度目の BEGIN BULK が現れた場合は無視する
                continue;
            }
            break (n, raw);
        };

        let body = strip_comment(&line);
        check_ascii(body, lineno)?;
        let form = line_form(body);
        if is_continuation(body, form) {
            return Err(FacetSetFemError::syntax(crate::tr!(
                "Continuation line without a parent card",
                "対応する親カードのない継続行です"
            ))
            .with_line(lineno)
            .with_hint(crate::tr!(
                "The preceding card may be missing. Also check how comment lines are written \
                 (a leading $)",
                "直前のカードが欠けているか、コメント行の書き方（先頭の $）を確認してください"
            )));
        }

        card.reset(lineno);
        push_line(card, body, form, false);

        // --- 継続行を取り込む ---
        loop {
            let Some((n, raw)) = self.src.next_line()? else {
                break;
            };
            let body = strip_comment(&raw);
            if body.trim().is_empty() {
                continue;
            }
            if body.trim().eq_ignore_ascii_case("ENDDATA") {
                self.src.push_back((n, raw));
                break;
            }
            check_ascii(body, n)?;
            let f = line_form(body);
            if is_continuation(body, f) {
                push_line(card, body, f, true);
            } else {
                self.src.push_back((n, raw));
                break;
            }
        }

        card.trim_trailing_blanks();
        Ok(true)
    }
}

/// 1 行分のフィールドをカードに積む。
///
/// 固定形式では**位置に意味がある**ため、空フィールドも含めて所定の本数を必ず積む。
/// 自由形式は行ごとに本数が変わるので、`split_free` が返した分だけ積む。
fn push_line(card: &mut Card, line: &str, form: LineForm, continuation: bool) {
    match form {
        LineForm::Free => {
            let fields = split_free(line);
            if !continuation {
                let name = fields.first().copied().unwrap_or("");
                card.push_field_upper(name.trim_end_matches('*'));
            }
            for f in fields.iter().skip(1) {
                card.push_field(f);
            }
        }
        _ => {
            let (name, data) = split_fixed(line, form);
            if !continuation {
                card.push_field_upper(name.trim_end_matches('*'));
            }
            for f in data {
                card.push_field(f);
            }
        }
    }
}

/// データ行が ASCII であることを確認する。
///
/// Nastran バルクデータは ASCII と定められている。多バイト文字が混じると
/// 桁位置がずれて静かに誤読するため、黙って読み飛ばさずエラーにする。
fn check_ascii(line: &str, lineno: usize) -> Result<()> {
    if line.is_ascii() {
        return Ok(());
    }
    Err(FacetSetFemError::syntax(crate::tr!(
        "A bulk data line contains non-ASCII characters",
        "バルクデータ行に ASCII 以外の文字が含まれています"
    ))
    .with_line(lineno)
    .with_hint(crate::tr!(
        "Nastran bulk data must be written in ASCII. To write non-ASCII text such as \
             Japanese, turn the line into a comment by starting it with $",
        "Nastran のバルクデータは ASCII で書かれている必要があります。\
             日本語などを書く場合は行頭を $ にしてコメントにしてください"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cards(text: &str) -> Vec<Vec<String>> {
        let mut r = BulkReader::new(text.as_bytes());
        let mut card = Card::new();
        let mut out = Vec::new();
        while r.next_card(&mut card).unwrap() {
            out.push(
                (0..card.n_fields())
                    .map(|i| card.field(i).to_string())
                    .collect(),
            );
        }
        out
    }

    #[test]
    fn skips_executive_and_case_control() {
        let text = "SOL 101\nCEND\nTITLE = test\nBEGIN BULK\nGRID,1,,0.0,0.0,0.0\nENDDATA\n";
        let c = cards(text);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0][0], "GRID");
    }

    #[test]
    fn bulk_only_file_without_begin_bulk() {
        let text = "$ コメント\nGRID,1,,0.0,0.0,0.0\nGRID,2,,1.0,0.0,0.0\n";
        let c = cards(text);
        assert_eq!(c.len(), 2);
        assert_eq!(c[1][1], "2");
    }

    #[test]
    fn small_field_with_explicit_continuation() {
        let text = concat!(
            "BEGIN BULK\n",
            "CHEXA   1       1       1       2       3       4       5       6\n",
            "+       7       8\n",
            "ENDDATA\n"
        );
        let c = cards(text);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0][0], "CHEXA");
        // EID, PID, G1..G8 = 10 データフィールド + カード名
        assert_eq!(c[0].len(), 11);
        assert_eq!(c[0][9], "7");
        assert_eq!(c[0][10], "8");
    }

    #[test]
    fn small_field_with_implicit_continuation() {
        let text = concat!(
            "CHEXA   1       1       1       2       3       4       5       6\n",
            "        7       8\n"
        );
        let c = cards(text);
        assert_eq!(c[0].len(), 11);
        assert_eq!(c[0][10], "8");
    }

    #[test]
    fn free_field_continuation() {
        let text = "CHEXA,1,1,1,2,3,4,5,6,+C1\n+C1,7,8\n";
        let c = cards(text);
        assert_eq!(c[0].len(), 11);
        assert_eq!(c[0][9], "7");
        assert_eq!(c[0][10], "8");
    }

    #[test]
    fn large_field_card() {
        let text = concat!(
            "GRID*   100                             0.0             1.0\n",
            "*       2.0\n"
        );
        let c = cards(text);
        assert_eq!(c[0][0], "GRID");
        assert_eq!(c[0][1], "100");
        assert_eq!(c[0][2], "");
        assert_eq!(c[0][3], "0.0");
        assert_eq!(c[0][4], "1.0");
        assert_eq!(c[0][5], "2.0");
    }

    #[test]
    fn interior_blank_fields_are_kept_in_fixed_form() {
        let text = "GRID    1               0.0     1.0     2.0\n";
        let c = cards(text);
        assert_eq!(c[0][1], "1");
        assert_eq!(c[0][2], ""); // CP は空
        assert_eq!(c[0][3], "0.0");
    }

    #[test]
    fn trailing_comment_does_not_switch_to_free_form() {
        let text = "GRID    1       0       0.0     1.0     2.0     $ x, y, z\n";
        let c = cards(text);
        assert_eq!(c[0][0], "GRID");
        assert_eq!(c[0][3], "0.0");
        assert_eq!(c[0][5], "2.0");
    }

    #[test]
    fn lowercase_card_names_are_normalised() {
        let c = cards("grid,1,,0.0,0.0,0.0\n");
        assert_eq!(c[0][0], "GRID");
    }

    #[test]
    fn orphan_continuation_is_an_error() {
        let mut r = BulkReader::new("BEGIN BULK\n+       1       2\n".as_bytes());
        let mut card = Card::new();
        let e = r.next_card(&mut card).unwrap_err();
        assert!(e.to_string().contains("継続行"));
    }

    #[test]
    fn non_ascii_data_line_is_rejected_but_comment_is_fine() {
        let c = cards("$ 日本語のコメントは許される\nGRID,1,,0.0,0.0,0.0\n");
        assert_eq!(c.len(), 1);

        let mut r = BulkReader::new("GRID,1,,０.0,0.0,0.0\n".as_bytes());
        let mut card = Card::new();
        assert!(r.next_card(&mut card).is_err());
    }

    #[test]
    fn stops_at_enddata() {
        let c = cards("GRID,1,,0.0,0.0,0.0\nENDDATA\nGRID,2,,1.0,0.0,0.0\n");
        assert_eq!(c.len(), 1);
    }
}
