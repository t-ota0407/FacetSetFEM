//! 単位系。
//!
//! CLAUDE.md の設計原則 7: **内部は必ず SI（m, S/m, A）**。
//! Nastran ファイルには単位情報が無いため、読み込み時に長さ単位を受け取って
//! ここで m に換算する（SPEC §3.2）。

use crate::error::{ErrorKind, GfemError, Result};
use std::fmt;

/// 入力メッシュの長さ単位。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LengthUnit {
    /// ミリメートル
    Millimeter,
    /// センチメートル
    Centimeter,
    /// メートル
    Meter,
    /// インチ
    Inch,
}

impl LengthUnit {
    /// GUI の選択肢に出す順序（SPEC §3.2）。
    pub const ALL: [LengthUnit; 4] = [
        LengthUnit::Millimeter,
        LengthUnit::Centimeter,
        LengthUnit::Meter,
        LengthUnit::Inch,
    ];

    /// この単位の 1 が何メートルか。
    pub const fn to_meters(self) -> f64 {
        match self {
            LengthUnit::Millimeter => 1.0e-3,
            LengthUnit::Centimeter => 1.0e-2,
            LengthUnit::Meter => 1.0,
            // 国際インチの定義値。丸め誤差の入らない厳密値
            LengthUnit::Inch => 0.0254,
        }
    }

    /// 表示用の記号。
    pub const fn symbol(self) -> &'static str {
        match self {
            LengthUnit::Millimeter => "mm",
            LengthUnit::Centimeter => "cm",
            LengthUnit::Meter => "m",
            LengthUnit::Inch => "inch",
        }
    }

    /// 文字列から解釈する。ケースファイル（YAML）と Python API の両方で使う。
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "mm" | "millimeter" | "millimetre" => Ok(LengthUnit::Millimeter),
            "cm" | "centimeter" | "centimetre" => Ok(LengthUnit::Centimeter),
            "m" | "meter" | "metre" => Ok(LengthUnit::Meter),
            "in" | "inch" | "inches" => Ok(LengthUnit::Inch),
            other => Err(GfemError::new(
                ErrorKind::Case,
                format!("長さ単位として解釈できません: \"{other}\""),
            )
            .with_hint("mm / cm / m / inch のいずれかを指定してください")),
        }
    }
}

impl fmt::Display for LengthUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.symbol())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion_factors() {
        assert_eq!(LengthUnit::Millimeter.to_meters(), 1.0e-3);
        assert_eq!(LengthUnit::Meter.to_meters(), 1.0);
        assert_eq!(LengthUnit::Inch.to_meters(), 0.0254);
    }

    #[test]
    fn parse_accepts_common_spellings() {
        assert_eq!(LengthUnit::parse("MM").unwrap(), LengthUnit::Millimeter);
        assert_eq!(LengthUnit::parse(" inch ").unwrap(), LengthUnit::Inch);
        assert!(LengthUnit::parse("furlong").is_err());
    }
}
