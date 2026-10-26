//! 表示言語（英語 / 日本語）の切り替え（SPEC §6.6）。
//!
//! エラー・警告・読み込みレポートなど、**人が読む文言はすべて英語と日本語の両方を持ち**、
//! いまの言語で片方を選ぶ。既定は英語。
//!
//! - 言語はプロセス全体で 1 つ（[`set_language`]）。GUI は起動時に設定から決める。
//!   最初に使われるまでに指定されなければ、環境変数 `FACETSETFEM_LANG`（`en` / `ja`）を見て、
//!   それも無ければ英語
//! - スレッドごとに一時的に上書きできる（[`with_language`]）。並列に走るテストの中で、
//!   ほかのテストの言語を変えずに片方の言語の文言を確かめるため
//! - 文言は [`tr!`](crate::tr) で**同じ場所に英語と日本語を並べて書く**。`format!` の書式は
//!   リテラルでなければならないので、英語をキーにした表を引く方式は Rust では取れない。
//!   並べて書けば訳の付け忘れも起きない
//!
//! 文言はエラーを作った時点の言語で組み立てる（後から言語を変えても作り直さない）。

use std::cell::Cell;
use std::sync::atomic::{AtomicU8, Ordering};

/// 表示言語。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    /// 英語（既定）
    En,
    /// 日本語
    Ja,
}

impl Language {
    /// `"en"` / `"ja"` を読む（大文字小文字は問わない。`ja_JP` のような地域付きも可）。
    pub fn parse(code: &str) -> Option<Language> {
        let code = code.trim().to_ascii_lowercase();
        let head = code.split(['_', '-', '.']).next().unwrap_or("");
        match head {
            "en" => Some(Language::En),
            "ja" | "jp" => Some(Language::Ja),
            _ => None,
        }
    }

    /// `"en"` / `"ja"`。
    pub const fn code(self) -> &'static str {
        match self {
            Language::En => "en",
            Language::Ja => "ja",
        }
    }

    const fn to_u8(self) -> u8 {
        match self {
            Language::En => 1,
            Language::Ja => 2,
        }
    }
}

/// 0 = まだ決めていない（最初に使うときに環境変数を見る）
static CURRENT: AtomicU8 = AtomicU8::new(0);

thread_local! {
    static OVERRIDE: Cell<Option<Language>> = const { Cell::new(None) };
}

/// 環境変数で既定の言語を指定するときの名前。
pub const LANGUAGE_ENV: &str = "FACETSETFEM_LANG";

/// いまの表示言語。
pub fn language() -> Language {
    if let Some(lang) = OVERRIDE.with(Cell::get) {
        return lang;
    }
    match CURRENT.load(Ordering::Relaxed) {
        1 => Language::En,
        2 => Language::Ja,
        _ => {
            let lang = std::env::var(LANGUAGE_ENV)
                .ok()
                .and_then(|v| Language::parse(&v))
                .unwrap_or(Language::En);
            // ほかのスレッドが先に決めていればそちらを尊重する
            let _ = CURRENT.compare_exchange(0, lang.to_u8(), Ordering::Relaxed, Ordering::Relaxed);
            language()
        }
    }
}

/// プロセス全体の表示言語を決める。
pub fn set_language(lang: Language) {
    CURRENT.store(lang.to_u8(), Ordering::Relaxed);
}

/// いまのスレッドだけ、`f` の間だけ言語を `lang` にする。
pub fn with_language<R>(lang: Language, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<Language>);
    impl Drop for Restore {
        fn drop(&mut self) {
            OVERRIDE.with(|o| o.set(self.0));
        }
    }
    let _restore = Restore(OVERRIDE.with(|o| o.replace(Some(lang))));
    f()
}

/// いまの言語が日本語か（[`tr!`](crate::tr) から使う）。
#[inline]
pub fn is_ja() -> bool {
    language() == Language::Ja
}

/// 英語と日本語を並べて書き、いまの言語のほうを `String` にする。
///
/// ```
/// # use facetsetfem_core::tr;
/// let n = 3;
/// let text = tr!("{n} faces", "{n} 面");
/// ```
///
/// 書式の後ろに引数を渡すと、両方の書式で共有する（`tr!("{}", "{}", x)`）。
#[macro_export]
macro_rules! tr {
    ($en:literal, $ja:literal $(,)?) => {
        if $crate::i18n::is_ja() { format!($ja) } else { format!($en) }
    };
    ($en:literal, $ja:literal, $($arg:tt)+) => {
        if $crate::i18n::is_ja() { format!($ja, $($arg)+) } else { format!($en, $($arg)+) }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_language_codes() {
        assert_eq!(Language::parse("en"), Some(Language::En));
        assert_eq!(Language::parse("JA"), Some(Language::Ja));
        assert_eq!(Language::parse("ja_JP.UTF-8"), Some(Language::Ja));
        assert_eq!(Language::parse("en-US"), Some(Language::En));
        assert_eq!(Language::parse("fr"), None);
    }

    #[test]
    fn the_thread_override_picks_the_text() {
        let n = 3;
        let en = with_language(Language::En, || tr!("{n} faces", "{n} 面"));
        let ja = with_language(Language::Ja, || tr!("{n} faces", "{n} 面"));
        assert_eq!(en, "3 faces");
        assert_eq!(ja, "3 面");
        let shared = with_language(Language::En, || tr!("{} of {}", "{1} のうち {0}", 1, 2));
        assert_eq!(shared, "1 of 2");
    }
}
