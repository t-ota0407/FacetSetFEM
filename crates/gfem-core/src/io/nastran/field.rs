//! Nastran バルクデータのフィールド分解と値の解釈（SPEC §3.1）。

/// 1 行の書式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineForm {
    /// 小フィールド形式（8 桁 × 10）
    Small,
    /// 大フィールド形式（8 + 16 × 4 + 8、カード名に `*` が付く）
    Large,
    /// 自由形式（カンマ区切り）
    Free,
}

impl LineForm {
    /// この形式で 1 行に載るデータフィールド数（カード名・継続マーカーを除く）。
    /// 自由形式は可変なので `None`。
    pub const fn data_fields_per_line(self) -> Option<usize> {
        match self {
            LineForm::Small => Some(8),
            LineForm::Large => Some(4),
            LineForm::Free => None,
        }
    }
}

/// 行の書式を判定する。
pub fn line_form(line: &str) -> LineForm {
    if line.contains(',') {
        return LineForm::Free;
    }
    // 継続行は先頭が '*'、親カードは名前フィールドの末尾が '*'
    if line.starts_with('*') {
        return LineForm::Large;
    }
    let head = line.get(..8).unwrap_or(line);
    if head.trim_end().ends_with('*') {
        LineForm::Large
    } else {
        LineForm::Small
    }
}

/// 継続行か（＝先頭フィールドが空、もしくは `+` / `*` で始まる）。
pub fn is_continuation(line: &str, form: LineForm) -> bool {
    match form {
        LineForm::Free => {
            let first = line.split(',').next().unwrap_or("").trim();
            first.is_empty() || first.starts_with('+') || first.starts_with('*')
        }
        _ => {
            let b = line.as_bytes();
            match b.first() {
                Some(&b'+') | Some(&b'*') => true,
                _ => line.get(..8).unwrap_or(line).trim().is_empty(),
            }
        }
    }
}

/// 固定形式の 1 行を、名前フィールドとデータフィールド列に分解する。
///
/// 継続マーカー（最終 8 桁のフィールド）は含めない。
pub fn split_fixed(line: &str, form: LineForm) -> (&str, FixedFields<'_>) {
    let (width, count) = match form {
        LineForm::Large => (16usize, 4usize),
        _ => (8usize, 8usize),
    };
    let name = slice_trim(line, 0, 8);
    (
        name,
        FixedFields {
            line,
            pos: 8,
            width,
            remaining: count,
        },
    )
}

/// `split_fixed` が返すデータフィールドの iterator。
#[derive(Debug)]
pub struct FixedFields<'a> {
    line: &'a str,
    pos: usize,
    width: usize,
    remaining: usize,
}

impl<'a> Iterator for FixedFields<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        if self.remaining == 0 {
            return None;
        }
        let s = slice_trim(self.line, self.pos, self.width);
        self.pos += self.width;
        self.remaining -= 1;
        Some(s)
    }
}

/// `line[start..start+width]` を取り出して trim する。行が短ければ空文字列。
fn slice_trim(line: &str, start: usize, width: usize) -> &str {
    if start >= line.len() {
        return "";
    }
    let end = (start + width).min(line.len());
    line.get(start..end).unwrap_or("").trim()
}

/// 自由形式の 1 行をカンマで分解する。
///
/// 末尾の空フィールドと、末尾の継続マーカー（`+` で始まるフィールド）は取り除く。
/// 途中の空フィールドは既定値を意味するので残す。
pub fn split_free(line: &str) -> Vec<&str> {
    let mut fields: Vec<&str> = line.split(',').map(str::trim).collect();
    while let Some(last) = fields.last() {
        if last.is_empty() || (fields.len() > 1 && last.starts_with('+')) {
            fields.pop();
        } else {
            break;
        }
    }
    fields
}

/// Nastran の整数フィールドを解釈する。空なら `None`。
pub fn parse_int(field: &str) -> Option<i64> {
    let t = field.trim();
    if t.is_empty() {
        return None;
    }
    t.parse::<i64>().ok()
}

/// Nastran の実数フィールドを解釈する。空なら `None`。
///
/// Nastran は Fortran 由来の表記を許すため、通常の `f64::from_str` では読めない:
///
/// - 指数の `E` を省略した形式: `1.5+3` = 1.5e3、`2.-4` = 2.0e-4
/// - 倍精度の `D` 指数: `1.5D+3`
/// - 小数点で終わる形式: `7.`
pub fn parse_real(field: &str) -> Option<f64> {
    let t = field.trim();
    if t.is_empty() {
        return None;
    }
    let bytes = t.as_bytes();
    let mut out = String::with_capacity(t.len() + 2);
    let mut i = 0;
    if bytes[0] == b'+' || bytes[0] == b'-' {
        out.push(bytes[0] as char);
        i = 1;
    }
    let mut seen_exp = false;
    let mut seen_digit = false;
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            b'0'..=b'9' => {
                seen_digit = true;
                out.push(c as char);
            }
            b'.' => {
                if seen_exp {
                    return None;
                }
                out.push('.');
            }
            b'E' | b'e' | b'D' | b'd' => {
                if seen_exp || !seen_digit {
                    return None;
                }
                seen_exp = true;
                // "1.E+3" のような形を避けるため小数点の後に 0 を補う
                if out.ends_with('.') {
                    out.push('0');
                }
                out.push('E');
            }
            b'+' | b'-' => {
                if out.ends_with('E') {
                    // 指数の符号
                    out.push(c as char);
                } else {
                    // 指数記号を省略した Fortran 表記
                    if seen_exp || !seen_digit {
                        return None;
                    }
                    seen_exp = true;
                    if out.ends_with('.') {
                        out.push('0');
                    }
                    out.push('E');
                    out.push(c as char);
                }
            }
            _ => return None,
        }
        i += 1;
    }
    if !seen_digit {
        return None;
    }
    if out.ends_with('.') {
        out.push('0');
    }
    let v = out.parse::<f64>().ok()?;
    if v.is_finite() { Some(v) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_plain_forms() {
        assert_eq!(parse_real("7.0"), Some(7.0));
        assert_eq!(parse_real("  -7. "), Some(-7.0));
        assert_eq!(parse_real(".5"), Some(0.5));
        assert_eq!(parse_real("+.5"), Some(0.5));
        assert_eq!(parse_real("3"), Some(3.0));
    }

    #[test]
    fn real_exponent_forms() {
        assert_eq!(parse_real("1.5E+3"), Some(1500.0));
        assert_eq!(parse_real("1.5e3"), Some(1500.0));
        assert_eq!(parse_real("1.5D+3"), Some(1500.0));
        // 指数記号を省略した Fortran 表記
        assert_eq!(parse_real("1.5+3"), Some(1500.0));
        assert_eq!(parse_real("2.-4"), Some(2.0e-4));
        assert_eq!(parse_real("-2.5-2"), Some(-0.025));
    }

    #[test]
    fn real_rejects_garbage() {
        assert_eq!(parse_real(""), None);
        assert_eq!(parse_real("abc"), None);
        assert_eq!(parse_real("1.0.0"), None);
        assert_eq!(parse_real("1.0E"), None);
        assert_eq!(parse_real("E5"), None);
        assert_eq!(parse_real("1.0E+3-2"), None);
    }

    #[test]
    fn int_forms() {
        assert_eq!(parse_int(" 42 "), Some(42));
        assert_eq!(parse_int("-1"), Some(-1));
        assert_eq!(parse_int("+7"), Some(7));
        assert_eq!(parse_int(""), None);
        assert_eq!(parse_int("1.0"), None);
    }

    #[test]
    fn form_detection() {
        assert_eq!(line_form("GRID    1       0       0.0"), LineForm::Small);
        assert_eq!(line_form("GRID*   1"), LineForm::Large);
        assert_eq!(line_form("*       1.0"), LineForm::Large);
        assert_eq!(line_form("GRID,1,,0.0,0.0,0.0"), LineForm::Free);
    }

    #[test]
    fn continuation_detection() {
        assert!(is_continuation("+       1       2", LineForm::Small));
        assert!(is_continuation("        1       2", LineForm::Small));
        assert!(!is_continuation("CTETRA  1       2", LineForm::Small));
        assert!(is_continuation(",7,8", LineForm::Free));
        assert!(is_continuation("+CONT,7,8", LineForm::Free));
        assert!(!is_continuation("GRID,1", LineForm::Free));
        // 大フィールドの親カードは継続行ではない
        assert!(!is_continuation("GRID*   1", LineForm::Large));
        assert!(is_continuation("*       1.0", LineForm::Large));
    }

    #[test]
    fn fixed_split_small() {
        let line = "GRID    1       0       0.0     1.0     2.0";
        let (name, fields) = split_fixed(line, LineForm::Small);
        assert_eq!(name, "GRID");
        let v: Vec<&str> = fields.collect();
        assert_eq!(v, vec!["1", "0", "0.0", "1.0", "2.0", "", "", ""]);
    }

    #[test]
    fn fixed_split_large() {
        let line = "GRID*   1               0               1.0             2.0";
        let (name, fields) = split_fixed(line, LineForm::Large);
        assert_eq!(name, "GRID*");
        let v: Vec<&str> = fields.collect();
        assert_eq!(v, vec!["1", "0", "1.0", "2.0"]);
    }

    #[test]
    fn free_split_keeps_interior_blanks() {
        assert_eq!(
            split_free("GRID,1,,0.0,0.0,0.0"),
            vec!["GRID", "1", "", "0.0", "0.0", "0.0"]
        );
        assert_eq!(
            split_free("CHEXA,1,1,1,2,3,4,5,6,+C1"),
            vec!["CHEXA", "1", "1", "1", "2", "3", "4", "5", "6"]
        );
        assert_eq!(split_free("GRID,1,"), vec!["GRID", "1"]);
    }
}
