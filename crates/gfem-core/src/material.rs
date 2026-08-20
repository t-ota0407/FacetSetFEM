//! 導電率。
//!
//! v1.0 は等方性・実数・周波数非依存のみ（SPEC §1.3）。
//! ただし SPEC §9 の「異方性導電率」への備えとして、
//!
//! - 型を enum にして `Tensor` を後から足せるようにする
//! - 要素剛性を `σ` のスカラー乗算ではなく **`σ` をベクトルに作用させる形**で書く
//!
//! の 2 点を最初から満たしておく。こうしておけば異方性の追加が
//! [`Conductivity::apply`] の分岐 1 つで済み、アセンブリ側は変更不要になる。

use crate::error::{ErrorKind, GfemError, Result};
use std::fmt;

/// 導電率 σ [S/m]。
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Conductivity {
    /// 等方性。σ は正のスカラー
    Isotropic(f64),
}

impl Conductivity {
    /// 等方性導電率を作る。正の有限値でなければエラー。
    pub fn isotropic(sigma: f64) -> Result<Self> {
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(GfemError::new(
                ErrorKind::Case,
                format!("導電率は正の有限値でなければなりません: σ = {sigma}"),
            )
            .with_hint(
                "生体組織の導電率は概ね 0.01〜2 S/m の範囲です（骨 0.02、脂肪 0.04、\
                 皮膚 0.1、筋 0.35 など）。単位が S/m になっているか確認してください",
            ));
        }
        Ok(Conductivity::Isotropic(sigma))
    }

    /// `σ v` を計算する。
    ///
    /// 異方性を入れるときはここに `Tensor` の分岐を足すだけでよい。
    #[inline]
    pub fn apply(&self, v: [f64; 3]) -> [f64; 3] {
        match *self {
            Conductivity::Isotropic(s) => [s * v[0], s * v[1], s * v[2]],
        }
    }

    /// 等方性のときのスカラー値。
    ///
    /// 電流密度の大きさ `|J| = σ|E|` のように、等方性を前提とした簡便計算に使う。
    #[inline]
    pub fn scalar(&self) -> f64 {
        match *self {
            Conductivity::Isotropic(s) => s,
        }
    }

    /// 最大固有値。条件数の見積り（SPEC §4.3）に使う。
    #[inline]
    pub fn max_eigenvalue(&self) -> f64 {
        match *self {
            Conductivity::Isotropic(s) => s,
        }
    }
}

impl fmt::Display for Conductivity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Conductivity::Isotropic(s) => write!(f, "{s} S/m"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_isotropic_conductivity() {
        let s = Conductivity::isotropic(0.35).unwrap();
        // 掛け算の結果をそのまま返すこと（丸めを挟まない）を確かめたいので、
        // 期待値も同じ演算で作る
        assert_eq!(
            s.apply([1.0, 2.0, -3.0]),
            [0.35 * 1.0, 0.35 * 2.0, 0.35 * -3.0]
        );
        assert_eq!(s.scalar(), 0.35);
    }

    #[test]
    fn rejects_non_positive_conductivity() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let e = Conductivity::isotropic(bad).unwrap_err();
            assert!(e.to_string().contains("正の有限値"), "{e}");
        }
    }
}
