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
//! | [`CpuParallel`]（rayon） | M4 で実装 |
//! | `Cuda`（cudarc） | M5 |
//!
//! **[`CpuSerial`] と [`CpuParallel`] はビット単位で同じ結果を出す。**
//! 総和は [`reduce`] の固定ブロック分割で行うので、どのスレッドがどのブロックを
//! 担当しても値が変わらない。SpMV も行ごとに独立なので順序の問題が起きない。
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

/// CPU 並列バックエンド（rayon）。
///
/// [`CpuSerial`] と**ビット単位で同じ結果**を出す。総和は固定ブロック分割
/// （[`reduce::par_sum`]）、SpMV は行ごとに独立なので、担当の割り振り方に
/// 結果が依存しない。
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuParallel;

/// 並列処理で 1 タスクが担当する要素数。小さすぎるとタスク投入の方が高くつく。
const CHUNK: usize = 4096;

impl LinearBackend for CpuParallel {
    fn name(&self) -> &'static str {
        "cpu_parallel"
    }

    fn spmv(&self, a: &CsrMatrix, x: &[f64], y: &mut [f64]) {
        use rayon::prelude::*;
        debug_assert_eq!(x.len(), a.n_rows());
        let (row_ptr, col_idx, values) = (a.row_ptr(), a.col_idx(), a.values());
        y.par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(chunk, out)| {
                let offset = chunk * CHUNK;
                for (local, yi) in out.iter_mut().enumerate() {
                    let i = offset + local;
                    let (s, e) = (row_ptr[i], row_ptr[i + 1]);
                    // 行内は列番号の昇順に逐次加算（逐次版と同じ順序）
                    let mut acc = 0.0;
                    for k in s..e {
                        acc += values[k] * x[col_idx[k] as usize];
                    }
                    *yi = acc;
                }
            });
    }

    fn axpy(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        use rayon::prelude::*;
        debug_assert_eq!(x.len(), y.len());
        y.par_chunks_mut(CHUNK)
            .zip(x.par_chunks(CHUNK))
            .for_each(|(out, xs)| {
                for (yi, &xi) in out.iter_mut().zip(xs) {
                    *yi += alpha * xi;
                }
            });
    }

    fn xpby(&self, x: &[f64], beta: f64, y: &mut [f64]) {
        use rayon::prelude::*;
        debug_assert_eq!(x.len(), y.len());
        y.par_chunks_mut(CHUNK)
            .zip(x.par_chunks(CHUNK))
            .for_each(|(out, xs)| {
                for (yi, &xi) in out.iter_mut().zip(xs) {
                    *yi = xi + beta * *yi;
                }
            });
    }

    fn dot(&self, x: &[f64], y: &[f64]) -> f64 {
        reduce::par_dot(x, y)
    }

    fn scale(&self, alpha: f64, x: &mut [f64]) {
        use rayon::prelude::*;
        x.par_chunks_mut(CHUNK).for_each(|chunk| {
            for xi in chunk {
                *xi *= alpha;
            }
        });
    }

    fn multiply(&self, a: &[f64], x: &[f64], y: &mut [f64]) {
        use rayon::prelude::*;
        debug_assert_eq!(a.len(), x.len());
        y.par_chunks_mut(CHUNK)
            .zip(a.par_chunks(CHUNK))
            .zip(x.par_chunks(CHUNK))
            .for_each(|((out, av), xv)| {
                for ((yi, &ai), &xi) in out.iter_mut().zip(av).zip(xv) {
                    *yi = ai * xi;
                }
            });
    }

    fn project_mean_zero(&self, x: &mut [f64]) {
        use rayon::prelude::*;
        let m = reduce::par_mean(x);
        x.par_chunks_mut(CHUNK).for_each(|chunk| {
            for xi in chunk {
                *xi -= m;
            }
        });
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

    /// 3 重対角行列（テスト用）。
    fn tridiagonal(n: usize) -> CsrMatrix {
        let mut row_ptr = vec![0usize];
        let mut col_idx: Vec<u32> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        for i in 0..n {
            for (c, v) in [(i.wrapping_sub(1), -1.0), (i, 2.0), (i + 1, -1.0)] {
                if c < n {
                    col_idx.push(c as u32);
                    values.push(v);
                }
            }
            row_ptr.push(col_idx.len());
        }
        let mut m = CsrMatrix::from_structure(n, row_ptr, col_idx);
        m.values_mut().copy_from_slice(&values);
        m
    }

    #[test]
    fn parallel_backend_matches_serial_bitwise() {
        // CLAUDE.md 設計原則 1: CPU 逐次と CPU 並列でビット単位で同じ結果
        let n = 20_000;
        let matrix = tridiagonal(n);
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let z: Vec<f64> = (0..n).map(|i| 1.0 / (i as f64 + 1.0)).collect();

        let (serial, parallel) = (CpuSerial, CpuParallel);

        let mut ys = vec![0.0; n];
        let mut yp = vec![0.0; n];
        serial.spmv(&matrix, &x, &mut ys);
        parallel.spmv(&matrix, &x, &mut yp);
        assert_eq!(ys, yp, "SpMV");

        assert_eq!(serial.dot(&x, &z), parallel.dot(&x, &z), "内積");
        assert_eq!(serial.norm2(&x), parallel.norm2(&x), "ノルム");

        let mut a = x.clone();
        let mut b = x.clone();
        serial.axpy(1.5, &z, &mut a);
        parallel.axpy(1.5, &z, &mut b);
        assert_eq!(a, b, "axpy");

        serial.xpby(&z, 0.25, &mut a);
        parallel.xpby(&z, 0.25, &mut b);
        assert_eq!(a, b, "xpby");

        serial.scale(0.5, &mut a);
        parallel.scale(0.5, &mut b);
        assert_eq!(a, b, "scale");

        let mut c = vec![0.0; n];
        let mut d = vec![0.0; n];
        serial.multiply(&z, &x, &mut c);
        parallel.multiply(&z, &x, &mut d);
        assert_eq!(c, d, "multiply");

        let mut e = x.clone();
        let mut f = x.clone();
        serial.project_mean_zero(&mut e);
        parallel.project_mean_zero(&mut f);
        assert_eq!(e, f, "射影");
    }

    #[test]
    fn parallel_backend_handles_sizes_that_are_not_multiples_of_the_chunk() {
        for n in [1usize, 7, CHUNK - 1, CHUNK, CHUNK + 1] {
            let matrix = tridiagonal(n);
            let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let mut ys = vec![0.0; n];
            let mut yp = vec![0.0; n];
            CpuSerial.spmv(&matrix, &x, &mut ys);
            CpuParallel.spmv(&matrix, &x, &mut yp);
            assert_eq!(ys, yp, "n = {n}");
        }
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
