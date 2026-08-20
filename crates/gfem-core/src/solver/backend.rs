//! 線形代数バックエンドの抽象化（SPEC §4.4）。
//!
//! CG のアルゴリズム本体（[`super::cg`]）はこの trait だけに依存する。
//! 実装を差し替えても**ビット単位で同じ結果**になることが要件なので、
//! 総和の順序は [`super::reduce`] の規則に全実装が従うこと。
//!
//! # 実装状況
//!
//! | バックエンド | 状態 |
//! |---|---|
//! | [`CpuSerial`] | M1 で実装 |
//! | `CpuParallel`（rayon） | M4 |
//! | `Cuda`（cudarc） | M5 |
//!
//! # M5 に向けた注意
//!
//! GPU ではベクトルがデバイスメモリ上に載るため、この trait は
//! 「ベクトル表現」を関連型に持ち上げる必要が出る見込み。CG 本体が
//! 全てのベクトル演算をこの trait 経由で行っている限り、その変更は
//! trait 定義と CG のシグネチャに閉じる。

use crate::assemble::csr::CsrMatrix;
use crate::solver::reduce;

/// 線形代数の基本演算。
pub trait LinearBackend {
    /// ログや出力 JSON に記録する名前（SPEC §5.5 のプロビナンス）。
    fn name(&self) -> &'static str;

    /// `y = A x`
    fn spmv(&self, a: &CsrMatrix, x: &[f64], y: &mut [f64]);

    /// `y += alpha x`
    fn axpy(&self, alpha: f64, x: &[f64], y: &mut [f64]);

    /// `y = x + beta y`
    fn xpby(&self, x: &[f64], beta: f64, y: &mut [f64]);

    /// 内積。**決定的な順序**で計算すること
    fn dot(&self, x: &[f64], y: &[f64]) -> f64;

    /// `x *= alpha`
    fn scale(&self, alpha: f64, x: &mut [f64]);

    /// 成分ごとの積 `y_i = a_i * x_i`（Jacobi 前処理に使う）
    fn multiply(&self, a: &[f64], x: &[f64], y: &mut [f64]);

    /// ゲージ射影 `x ← (I − 11ᵀ/N) x`（SPEC §2.2）
    fn project_mean_zero(&self, x: &mut [f64]);

    /// 2 ノルム
    fn norm2(&self, x: &[f64]) -> f64 {
        self.dot(x, x).sqrt()
    }
}

/// CPU 逐次バックエンド。
///
/// 全ての総和を [`reduce`] のブロック分割規則で行うので、
/// M4 の rayon 版・M5 の CUDA 版と結果が一致する。
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuSerial;

impl LinearBackend for CpuSerial {
    fn name(&self) -> &'static str {
        "cpu_serial"
    }

    fn spmv(&self, a: &CsrMatrix, x: &[f64], y: &mut [f64]) {
        a.spmv(x, y);
    }

    fn axpy(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        debug_assert_eq!(x.len(), y.len());
        for (yi, &xi) in y.iter_mut().zip(x) {
            *yi += alpha * xi;
        }
    }

    fn xpby(&self, x: &[f64], beta: f64, y: &mut [f64]) {
        debug_assert_eq!(x.len(), y.len());
        for (yi, &xi) in y.iter_mut().zip(x) {
            *yi = xi + beta * *yi;
        }
    }

    fn dot(&self, x: &[f64], y: &[f64]) -> f64 {
        reduce::dot(x, y)
    }

    fn scale(&self, alpha: f64, x: &mut [f64]) {
        for xi in x.iter_mut() {
            *xi *= alpha;
        }
    }

    fn multiply(&self, a: &[f64], x: &[f64], y: &mut [f64]) {
        debug_assert_eq!(a.len(), x.len());
        for (yi, (&ai, &xi)) in y.iter_mut().zip(a.iter().zip(x)) {
            *yi = ai * xi;
        }
    }

    fn project_mean_zero(&self, x: &mut [f64]) {
        let m = reduce::mean(x);
        for xi in x.iter_mut() {
            *xi -= m;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_removes_the_constant_component() {
        let b = CpuSerial;
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        b.project_mean_zero(&mut x);
        assert!((reduce::sum(&x)).abs() < 1e-15);
        assert!((x[0] + 1.5).abs() < 1e-15);
    }

    #[test]
    fn projection_is_idempotent() {
        let b = CpuSerial;
        let mut x: Vec<f64> = (0..1000).map(|i| (i as f64).sin() + 3.0).collect();
        b.project_mean_zero(&mut x);
        let once = x.clone();
        b.project_mean_zero(&mut x);
        // 2 回目の射影で動く量は、1 回目の残差平均ぶん（数 ULP）に収まるべき
        let scale = once.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let shift = once
            .iter()
            .zip(&x)
            .map(|(a, c)| (a - c).abs())
            .fold(0.0f64, f64::max);
        assert!(
            shift <= 16.0 * f64::EPSILON * scale,
            "2 回目の射影で {shift} 動きました（許容 {}）",
            16.0 * f64::EPSILON * scale
        );
    }

    #[test]
    fn vector_operations() {
        let b = CpuSerial;
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![10.0, 20.0, 30.0];
        b.axpy(2.0, &x, &mut y);
        assert_eq!(y, vec![12.0, 24.0, 36.0]);

        let mut z = vec![1.0, 1.0, 1.0];
        b.xpby(&x, 3.0, &mut z);
        assert_eq!(z, vec![4.0, 5.0, 6.0]);

        b.scale(0.5, &mut z);
        assert_eq!(z, vec![2.0, 2.5, 3.0]);

        let mut w = vec![0.0; 3];
        b.multiply(&[2.0, 4.0, 8.0], &x, &mut w);
        assert_eq!(w, vec![2.0, 8.0, 24.0]);

        assert_eq!(b.dot(&x, &x), 14.0);
        assert!((b.norm2(&[3.0, 4.0]) - 5.0).abs() < 1e-15);
    }
}
