//! エラー型。
//!
//! CLAUDE.md の設計原則 6（エラーは親切に）に従い、すべてのエラーが
//!
//! - **何が** 起きたのか（`message`）
//! - **どこで** 起きたのか（`location`: ファイル・行・カード・フィールド・要素 ID）
//! - **どうすれば直るか**（`hint`）
//!
//! の 3 点を持てる構造にしている。`Display` はこの 3 点を必ず整形して出す。

use std::fmt;
use std::path::{Path, PathBuf};

/// `gfem-core` 共通の `Result`。
pub type Result<T> = std::result::Result<T, GfemError>;

/// エラーの分類。GUI 側でアイコンや色を変えるために使う。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorKind {
    /// ファイルの入出力に失敗した
    Io,
    /// 入力ファイルの構文が壊れている
    Syntax,
    /// 構文は正しいが、モデルの内容が不正（参照切れ・退化要素など）
    Model,
    /// 仕様として未対応の機能を要求された
    Unsupported,
    /// 解析ケースの設定が不足・矛盾している
    Case,
    /// 反復解法が収束しなかった
    Convergence,
    /// 利用者に中断された
    Cancelled,
    /// 実装の不備（利用者側では直せない）
    Internal,
}

impl ErrorKind {
    /// メッセージ先頭に出すラベル（いまの表示言語で。SPEC §6.6）。
    pub fn label(self) -> &'static str {
        let (en, ja) = match self {
            ErrorKind::Io => ("I/O error", "入出力エラー"),
            ErrorKind::Syntax => ("Syntax error", "構文エラー"),
            ErrorKind::Model => ("Model error", "モデルエラー"),
            ErrorKind::Unsupported => ("Unsupported", "未対応"),
            ErrorKind::Case => ("Setup error", "設定エラー"),
            ErrorKind::Convergence => ("Convergence error", "収束エラー"),
            ErrorKind::Cancelled => ("Cancelled", "中断"),
            ErrorKind::Internal => ("Internal error", "内部エラー"),
        };
        if crate::i18n::is_ja() { ja } else { en }
    }
}

/// エラーの発生位置。分かる範囲だけ埋める。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Location {
    /// 対象ファイル
    pub file: Option<PathBuf>,
    /// 1 始まりの行番号
    pub line: Option<usize>,
    /// Nastran カード名（例: `CTETRA`）
    pub card: Option<String>,
    /// カード内のフィールド番号（0 = カード名）
    pub field: Option<usize>,
    /// 対象エンティティ（例: `要素 EID=101`）
    pub entity: Option<String>,
}

impl Location {
    /// 位置情報が一つも無いか。
    pub fn is_empty(&self) -> bool {
        self.file.is_none()
            && self.line.is_none()
            && self.card.is_none()
            && self.field.is_none()
            && self.entity.is_none()
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts: Vec<String> = Vec::new();
        match (&self.file, self.line) {
            (Some(p), Some(l)) => parts.push(format!("{}:{}", p.display(), l)),
            (Some(p), None) => parts.push(p.display().to_string()),
            (None, Some(l)) => parts.push(crate::tr!("line {l}", "{l} 行目")),
            (None, None) => {}
        }
        if let Some(c) = &self.card {
            parts.push(crate::tr!("{c} card", "{c} カード"));
        }
        if let Some(i) = self.field {
            parts.push(crate::tr!("field {i}", "フィールド {i}"));
        }
        if let Some(e) = &self.entity {
            parts.push(e.clone());
        }
        write!(f, "{}", parts.join(" "))
    }
}

/// GenericFEM の共通エラー。
///
/// 中身は `Box` に入れてある。エラーは例外的な経路にしか現れないので、
/// すべての `Result<T>` を太らせないよう、この型自体はポインタ 1 本分に抑えている。
#[derive(Debug)]
pub struct GfemError(Box<Detail>);

#[derive(Debug)]
struct Detail {
    kind: ErrorKind,
    message: String,
    location: Location,
    hint: Option<String>,
    source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
}

impl GfemError {
    /// 分類とメッセージからエラーを作る。位置とヒントは `with_*` で足す。
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self(Box::new(Detail {
            kind,
            message: message.into(),
            location: Location::default(),
            hint: None,
            source: None,
        }))
    }

    /// 構文エラー。
    pub fn syntax(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Syntax, message)
    }

    /// モデルの内容が不正。
    pub fn model(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Model, message)
    }

    /// 未対応の機能。
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Unsupported, message)
    }

    /// 実装の不備。
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Internal, message)
    }

    /// `std::io::Error` を対象パス付きで包む。
    pub fn io(path: impl AsRef<Path>, message: impl Into<String>, source: std::io::Error) -> Self {
        Self::new(ErrorKind::Io, message)
            .with_file(path)
            .with_source(source)
    }

    /// 対象ファイルを設定する。
    pub fn with_file(mut self, path: impl AsRef<Path>) -> Self {
        self.0.location.file = Some(path.as_ref().to_path_buf());
        self
    }

    /// ファイルがまだ設定されていない場合だけ設定する（下位が付けた位置を上書きしない）。
    pub fn or_file(mut self, path: impl AsRef<Path>) -> Self {
        if self.0.location.file.is_none() {
            self.0.location.file = Some(path.as_ref().to_path_buf());
        }
        self
    }

    /// 行番号（1 始まり）を設定する。
    pub fn with_line(mut self, line: usize) -> Self {
        self.0.location.line = Some(line);
        self
    }

    /// カード名を設定する。
    pub fn with_card(mut self, card: impl Into<String>) -> Self {
        self.0.location.card = Some(card.into());
        self
    }

    /// フィールド番号を設定する。
    pub fn with_field(mut self, field: usize) -> Self {
        self.0.location.field = Some(field);
        self
    }

    /// 対象エンティティ（要素 ID など）を設定する。
    pub fn with_entity(mut self, entity: impl Into<String>) -> Self {
        self.0.location.entity = Some(entity.into());
        self
    }

    /// 対処方法を設定する。
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.0.hint = Some(hint.into());
        self
    }

    /// 原因となったエラーを設定する。
    pub fn with_source(mut self, source: impl std::error::Error + Send + Sync + 'static) -> Self {
        self.0.source = Some(Box::new(source));
        self
    }

    /// 分類。
    pub fn kind(&self) -> ErrorKind {
        self.0.kind
    }

    /// 発生位置。
    pub fn location(&self) -> &Location {
        &self.0.location
    }

    /// 説明本文（「何が」の部分）。
    pub fn message(&self) -> &str {
        &self.0.message
    }

    /// 対処方法（「どうすれば直るか」の部分）。
    pub fn hint(&self) -> Option<&str> {
        self.0.hint.as_deref()
    }
}

impl fmt::Display for GfemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ja = crate::i18n::is_ja();
        write!(f, "[{}] {}", self.0.kind.label(), self.0.message)?;
        if !self.0.location.is_empty() {
            let head = if ja { "場所" } else { "Where" };
            write!(f, "\n  {head}: {}", self.0.location)?;
        }
        if let Some(h) = &self.0.hint {
            let head = if ja { "対処" } else { "Fix" };
            write!(f, "\n  {head}: {h}")?;
        }
        if let Some(s) = &self.0.source {
            let head = if ja { "原因" } else { "Cause" };
            write!(f, "\n  {head}: {s}")?;
        }
        Ok(())
    }
}

impl std::error::Error for GfemError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0
            .source
            .as_ref()
            .map(|b| b.as_ref() as &dyn std::error::Error)
    }
}

/// 多数の同種エラーを列挙するときの共通整形。
///
/// 全部並べるとログが読めなくなるので、先頭 `limit` 件だけ出して残りは件数で示す。
pub fn format_sample_list(items: &[String], limit: usize) -> String {
    if items.len() <= limit {
        items.join(", ")
    } else {
        crate::tr!(
            "{} ... and {} more ({} in total)",
            "{} ... 他 {} 件（全 {} 件）",
            items[..limit].join(", "),
            items.len() - limit,
            items.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_contains_three_parts() {
        let e = GfemError::syntax("実数として解釈できません: \"1.0.0\"")
            .with_file("model.bdf")
            .with_line(12)
            .with_card("GRID")
            .with_field(4)
            .with_hint("Nastran の実数形式（例: 1.0, 1.-3, 2.5E+2）で記述してください");
        let s = e.to_string();
        assert!(s.contains("[構文エラー]"));
        assert!(s.contains("model.bdf:12"));
        assert!(s.contains("GRID カード"));
        assert!(s.contains("フィールド 4"));
        assert!(s.contains("対処:"));
    }

    #[test]
    fn error_stays_pointer_sized() {
        // Result<T> を太らせないための不変条件
        assert_eq!(
            std::mem::size_of::<GfemError>(),
            std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn sample_list_truncates() {
        let v: Vec<String> = (1..=30).map(|i| i.to_string()).collect();
        let s = format_sample_list(&v, 5);
        assert!(s.starts_with("1, 2, 3, 4, 5 ..."));
        assert!(s.contains("全 30 件"));
    }
}
