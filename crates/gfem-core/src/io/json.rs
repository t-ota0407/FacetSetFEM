//! 最小限の JSON ライタ（SPEC §5.5 の実行サマリ出力に使う）。
//!
//! 書き出すのは自分で組み立てた構造だけなので、`serde` を入れずに済ませている
//! （`gfem-core` の依存を増やさないため。CLAUDE.md 設計原則 5 の精神）。
//! オブジェクトのキーは**挿入順**を保つので、同じ入力からは同じバイト列が出る。

use std::fmt;

/// JSON の値。
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// `null`
    Null,
    /// `true` / `false`
    Bool(bool),
    /// 整数
    Int(i64),
    /// 実数。NaN と無限大は JSON で表せないので `null` になる
    Number(f64),
    /// 文字列
    String(String),
    /// 配列
    Array(Vec<Json>),
    /// オブジェクト（挿入順を保つ）
    Object(Vec<(String, Json)>),
}

impl Json {
    /// 空のオブジェクト。
    pub fn object() -> Self {
        Json::Object(Vec::new())
    }

    /// オブジェクトに要素を足す。オブジェクト以外に対しては何もしない。
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<Json>) -> &mut Self {
        if let Json::Object(entries) = self {
            entries.push((key.into(), value.into()));
        }
        self
    }

    /// 省略可能な値。`None` なら `null` を入れる。
    pub fn set_option<T: Into<Json>>(
        &mut self,
        key: impl Into<String>,
        value: Option<T>,
    ) -> &mut Self {
        match value {
            Some(v) => self.set(key, v),
            None => self.set(key, Json::Null),
        }
    }

    /// 整形して文字列にする。
    pub fn to_pretty_string(&self) -> String {
        let mut out = String::new();
        write_value(&mut out, self, 0);
        out.push('\n');
        out
    }
}

impl From<bool> for Json {
    fn from(v: bool) -> Self {
        Json::Bool(v)
    }
}
impl From<i64> for Json {
    fn from(v: i64) -> Self {
        Json::Int(v)
    }
}
impl From<usize> for Json {
    fn from(v: usize) -> Self {
        Json::Int(v as i64)
    }
}
impl From<u32> for Json {
    fn from(v: u32) -> Self {
        Json::Int(v as i64)
    }
}
impl From<f64> for Json {
    fn from(v: f64) -> Self {
        Json::Number(v)
    }
}
impl From<&str> for Json {
    fn from(v: &str) -> Self {
        Json::String(v.to_string())
    }
}
impl From<String> for Json {
    fn from(v: String) -> Self {
        Json::String(v)
    }
}
impl<T: Into<Json>> From<Vec<T>> for Json {
    fn from(v: Vec<T>) -> Self {
        Json::Array(v.into_iter().map(Into::into).collect())
    }
}

impl fmt::Display for Json {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_pretty_string())
    }
}

fn write_value(out: &mut String, value: &Json, indent: usize) {
    match value {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Int(i) => out.push_str(&i.to_string()),
        Json::Number(v) => {
            if v.is_finite() {
                // 17 桁あれば f64 は往復できる
                out.push_str(&format!("{v:.17e}"));
            } else {
                // NaN / ±∞ は JSON で表せない
                out.push_str("null");
            }
        }
        Json::String(s) => write_string(out, s),
        Json::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push_str("[\n");
            for (i, item) in items.iter().enumerate() {
                push_indent(out, indent + 1);
                write_value(out, item, indent + 1);
                if i + 1 < items.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            push_indent(out, indent);
            out.push(']');
        }
        Json::Object(entries) => {
            if entries.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push_str("{\n");
            for (i, (key, item)) in entries.iter().enumerate() {
                push_indent(out, indent + 1);
                write_string(out, key);
                out.push_str(": ");
                write_value(out, item, indent + 1);
                if i + 1 < entries.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            push_indent(out, indent);
            out.push('}');
        }
    }
}

fn push_indent(out: &mut String, level: usize) {
    for _ in 0..level {
        out.push_str("  ");
    }
}

fn write_string(out: &mut String, value: &str) {
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            // 制御文字は \u エスケープ
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_a_nested_object() {
        let mut root = Json::object();
        root.set("name", "gfem")
            .set("version", 1i64)
            .set("ok", true)
            .set("values", vec![1.0, 2.5]);
        let mut inner = Json::object();
        inner.set("iterations", 94usize);
        root.set("solver", inner);

        let text = root.to_pretty_string();
        assert!(text.contains("\"name\": \"gfem\""));
        assert!(text.contains("\"iterations\": 94"));
        assert!(text.contains("\"ok\": true"));
    }

    #[test]
    fn escapes_strings() {
        let mut root = Json::object();
        root.set("text", "改行\nと\"引用符\"とタブ\t");
        let text = root.to_pretty_string();
        assert!(text.contains("\\n"));
        assert!(text.contains("\\\"引用符\\\""));
        assert!(text.contains("\\t"));
    }

    #[test]
    fn non_finite_numbers_become_null() {
        let mut root = Json::object();
        root.set("nan", f64::NAN).set("inf", f64::INFINITY);
        let text = root.to_pretty_string();
        assert!(text.contains("\"nan\": null"));
        assert!(text.contains("\"inf\": null"));
    }

    #[test]
    fn keeps_insertion_order() {
        let mut root = Json::object();
        root.set("z", 1i64).set("a", 2i64).set("m", 3i64);
        let text = root.to_pretty_string();
        let z = text.find("\"z\"").unwrap();
        let a = text.find("\"a\"").unwrap();
        let m = text.find("\"m\"").unwrap();
        assert!(z < a && a < m, "挿入順が保たれていません");
    }

    #[test]
    fn empty_containers() {
        let mut root = Json::object();
        root.set("array", Json::Array(Vec::new()))
            .set("object", Json::object());
        let text = root.to_pretty_string();
        assert!(text.contains("\"array\": []"));
        assert!(text.contains("\"object\": {}"));
    }

    #[test]
    fn optional_values() {
        let mut root = Json::object();
        root.set_option("present", Some("yes"))
            .set_option::<&str>("absent", None);
        let text = root.to_pretty_string();
        assert!(text.contains("\"present\": \"yes\""));
        assert!(text.contains("\"absent\": null"));
    }
}
