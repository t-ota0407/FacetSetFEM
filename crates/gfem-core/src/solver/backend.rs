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
//! | `Cuda`（cudarc） | M5 で実装（`gfem-cuda` クレート） |
//!
//! **どのバックエンドでもビット単位で同じ結果を出す。**
//! 総和は [`reduce`] の固定ブロック分割で行うので、どのスレッドが／どの GPU スレッドが
//! どのブロックを担当しても値が変わらない。SpMV も行ごとに独立なので順序の問題が起きない。
//!
//! # ベクトルはバックエンドが持つ
//!
//! GPU ではベクトルがデバイスメモリ上に載るため、ベクトルと行列の表現は
//! 関連型 [`LinearBackend::Vector`] / [`LinearBackend::Matrix`] にしてある
//! （SPEC §4.4 が M5 で必要になると予告していた変更）。CG 本体は
//! 生成・転送・演算を全てこの trait 経由で行うので、ホストとデバイスの
//! どちらにデータがあるかを知らない。
//!
//! CPU バックエンドではベクトルはただの `Vec<f64>`、行列は元の [`CsrMatrix`] への
//! 参照なので、この抽象化による複製やコピーは発生しない。

use crate::assemble::csr::CsrMatrix;
use crate::error::Result;
use crate::solver::reduce;

/// 線形代数の基本演算。
///
/// スカラー（`alpha` や内積の結果）は常にホスト側の `f64` でやりとりする。
/// CG のスカラー演算はホストで行うため、この境界の置き方だけで
/// 「浮動小数点の演算順序がバックエンドによって変わらない」ことが保証しやすくなる。
pub trait LinearBackend {
    /// ベクトルの表現。CPU では `Vec<f64>`、GPU ではデバイスメモリ上の確保
    type Vector;

    /// 行列の表現。CPU では [`CsrMatrix`] への参照、GPU では VRAM 上の CSR
    type Matrix<'a>;

    /// ログや出力 JSON に記録する名前（SPEC §5.5 のプロビナンス）。
    fn name(&self) -> &'static str;

    // --- 生成と転送 -------------------------------------------------------

    /// 行列をバックエンドが使える形にする（GPU では VRAM へ転送する）。
    fn upload_matrix<'a>(&self, a: &'a CsrMatrix) -> Result<Self::Matrix<'a>>;

    /// 長さ `n` のゼロベクトルを作る。
    fn zeros(&self, n: usize) -> Result<Self::Vector>;

    /// ホストのスライスからベクトルを作る。
    fn upload(&self, host: &[f64]) -> Result<Self::Vector>;

    /// ベクトルをホストへ書き戻す。
    fn download(&self, x: &Self::Vector, host: &mut [f64]) -> Result<()>;

    /// ホストのスライスを既存のベクトルへ書き込む。
    ///
    /// AMG の最粗レベルはホストで直接解く（[`crate::solver::amg`]）ので、
    /// V サイクルのたびに結果を戻す必要がある。[`upload`](Self::upload) だと
    /// 反復ごとに確保が走ってしまうため、書き込み先を渡せる形も用意する。
    fn upload_into(&self, host: &[f64], x: &mut Self::Vector) -> Result<()>;

    /// `dst ← src`（同じ長さであること）。
    fn copy_from(&self, src: &Self::Vector, dst: &mut Self::Vector) -> Result<()>;

    /// ベクトルの長さ。
    fn len(&self, x: &Self::Vector) -> usize;

    // --- 演算 -------------------------------------------------------------

    /// `y = A x`
    fn spmv(&self, a: &Self::Matrix<'_>, x: &Self::Vector, y: &mut Self::Vector) -> Result<()>;

    /// `y += alpha x`
    fn axpy(&self, alpha: f64, x: &Self::Vector, y: &mut Self::Vector) -> Result<()>;

    /// `y = x + beta y`
    fn xpby(&self, x: &Self::Vector, beta: f64, y: &mut Self::Vector) -> Result<()>;

    /// 内積。**決定的な順序**で計算すること（[`reduce`] の規則）
    fn dot(&self, x: &Self::Vector, y: &Self::Vector) -> Result<f64>;

    /// 総和。**決定的な順序**で計算すること（[`reduce`] の規則）
    fn sum(&self, x: &Self::Vector) -> Result<f64>;

    /// `x *= alpha`
    fn scale(&self, alpha: f64, x: &mut Self::Vector) -> Result<()>;

    /// 成分ごとの積 `y_i = a_i * x_i`（Jacobi 前処理に使う）
    fn multiply(&self, a: &Self::Vector, x: &Self::Vector, y: &mut Self::Vector) -> Result<()>;

    /// 全成分に定数を足す `x_i += delta`
    fn shift(&self, delta: f64, x: &mut Self::Vector) -> Result<()>;

    /// 全成分を `value` にする。
    ///
    /// AMG の V サイクル（SPEC §4.5）が、各レベルの補正量を毎回ゼロから始めるために使う。
    /// `scale(0.0)` で代用すると負の成分が `-0.0` になって総和の符号が変わりうるので、
    /// 専用の操作にしてある。
    fn fill(&self, value: f64, x: &mut Self::Vector) -> Result<()>;

    // --- 既定実装 ---------------------------------------------------------

    /// 2 ノルム。
    ///
    /// 平方根はホスト側で取る。IEEE 754 の平方根は正しく丸められるので、
    /// バックエンドによって結果が変わることはない。
    fn norm2(&self, x: &Self::Vector) -> Result<f64> {
        Ok(self.dot(x, x)?.sqrt())
    }

    /// ゲージ射影 `x ← (I − 11ᵀ/N) x`（SPEC §2.2）。
    ///
    /// 平均をホスト側で求めてから引くので、[`sum`](Self::sum) さえ決定的なら
    /// 射影も自動的に決定的になる。`x - m` と `x + (-m)` は IEEE 754 で
    /// 厳密に等しいため、実装ごとに差は出ない。
    fn project_mean_zero(&self, x: &mut Self::Vector) -> Result<()> {
        let n = self.len(x);
        if n == 0 {
            return Ok(());
        }
        let mean = self.sum(x)? / n as f64;
        self.shift(-mean, x)
    }
}

/// CPU 逐次バックエンド。
///
/// 全ての総和を [`reduce`] のブロック分割規則で行うので、
/// [`CpuParallel`] や CUDA 版と結果が一致する。
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuSerial;

impl LinearBackend for CpuSerial {
    type Vector = Vec<f64>;
    type Matrix<'a> = &'a CsrMatrix;

    fn name(&self) -> &'static str {
        "cpu_serial"
    }

    fn upload_matrix<'a>(&self, a: &'a CsrMatrix) -> Result<&'a CsrMatrix> {
        Ok(a)
    }

    fn zeros(&self, n: usize) -> Result<Vec<f64>> {
        Ok(vec![0.0; n])
    }

    fn upload(&self, host: &[f64]) -> Result<Vec<f64>> {
        Ok(host.to_vec())
    }

    fn download(&self, x: &Vec<f64>, host: &mut [f64]) -> Result<()> {
        host.copy_from_slice(x);
        Ok(())
    }

    fn upload_into(&self, host: &[f64], x: &mut Vec<f64>) -> Result<()> {
        x.copy_from_slice(host);
        Ok(())
    }

    fn copy_from(&self, src: &Vec<f64>, dst: &mut Vec<f64>) -> Result<()> {
        dst.copy_from_slice(src);
        Ok(())
    }

    fn len(&self, x: &Vec<f64>) -> usize {
        x.len()
    }

    fn spmv(&self, a: &&CsrMatrix, x: &Vec<f64>, y: &mut Vec<f64>) -> Result<()> {
        a.spmv(x, y);
        Ok(())
    }

    fn axpy(&self, alpha: f64, x: &Vec<f64>, y: &mut Vec<f64>) -> Result<()> {
        debug_assert_eq!(x.len(), y.len());
        for (yi, &xi) in y.iter_mut().zip(x) {
            *yi += alpha * xi;
        }
        Ok(())
    }

    fn xpby(&self, x: &Vec<f64>, beta: f64, y: &mut Vec<f64>) -> Result<()> {
        debug_assert_eq!(x.len(), y.len());
        for (yi, &xi) in y.iter_mut().zip(x) {
            *yi = xi + beta * *yi;
        }
        Ok(())
    }

    fn dot(&self, x: &Vec<f64>, y: &Vec<f64>) -> Result<f64> {
        Ok(reduce::dot(x, y))
    }

    fn sum(&self, x: &Vec<f64>) -> Result<f64> {
        Ok(reduce::sum(x))
    }

    fn scale(&self, alpha: f64, x: &mut Vec<f64>) -> Result<()> {
        for xi in x.iter_mut() {
            *xi *= alpha;
        }
        Ok(())
    }

    fn multiply(&self, a: &Vec<f64>, x: &Vec<f64>, y: &mut Vec<f64>) -> Result<()> {
        debug_assert_eq!(a.len(), x.len());
        for (yi, (&ai, &xi)) in y.iter_mut().zip(a.iter().zip(x)) {
            *yi = ai * xi;
        }
        Ok(())
    }

    fn shift(&self, delta: f64, x: &mut Vec<f64>) -> Result<()> {
        for xi in x.iter_mut() {
            *xi += delta;
        }
        Ok(())
    }

    fn fill(&self, value: f64, x: &mut Vec<f64>) -> Result<()> {
        x.fill(value);
        Ok(())
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
    type Vector = Vec<f64>;
    type Matrix<'a> = &'a CsrMatrix;

    fn name(&self) -> &'static str {
        "cpu_parallel"
    }

    fn upload_matrix<'a>(&self, a: &'a CsrMatrix) -> Result<&'a CsrMatrix> {
        Ok(a)
    }

    fn zeros(&self, n: usize) -> Result<Vec<f64>> {
        Ok(vec![0.0; n])
    }

    fn upload(&self, host: &[f64]) -> Result<Vec<f64>> {
        Ok(host.to_vec())
    }

    fn download(&self, x: &Vec<f64>, host: &mut [f64]) -> Result<()> {
        host.copy_from_slice(x);
        Ok(())
    }

    fn upload_into(&self, host: &[f64], x: &mut Vec<f64>) -> Result<()> {
        x.copy_from_slice(host);
        Ok(())
    }

    fn copy_from(&self, src: &Vec<f64>, dst: &mut Vec<f64>) -> Result<()> {
        dst.copy_from_slice(src);
        Ok(())
    }

    fn len(&self, x: &Vec<f64>) -> usize {
        x.len()
    }

    fn spmv(&self, a: &&CsrMatrix, x: &Vec<f64>, y: &mut Vec<f64>) -> Result<()> {
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
        Ok(())
    }

    fn axpy(&self, alpha: f64, x: &Vec<f64>, y: &mut Vec<f64>) -> Result<()> {
        use rayon::prelude::*;
        debug_assert_eq!(x.len(), y.len());
        y.par_chunks_mut(CHUNK)
            .zip(x.par_chunks(CHUNK))
            .for_each(|(out, xs)| {
                for (yi, &xi) in out.iter_mut().zip(xs) {
                    *yi += alpha * xi;
                }
            });
        Ok(())
    }

    fn xpby(&self, x: &Vec<f64>, beta: f64, y: &mut Vec<f64>) -> Result<()> {
        use rayon::prelude::*;
        debug_assert_eq!(x.len(), y.len());
        y.par_chunks_mut(CHUNK)
            .zip(x.par_chunks(CHUNK))
            .for_each(|(out, xs)| {
                for (yi, &xi) in out.iter_mut().zip(xs) {
                    *yi = xi + beta * *yi;
                }
            });
        Ok(())
    }

    fn dot(&self, x: &Vec<f64>, y: &Vec<f64>) -> Result<f64> {
        Ok(reduce::par_dot(x, y))
    }

    fn sum(&self, x: &Vec<f64>) -> Result<f64> {
        Ok(reduce::par_sum(x))
    }

    fn scale(&self, alpha: f64, x: &mut Vec<f64>) -> Result<()> {
        use rayon::prelude::*;
        x.par_chunks_mut(CHUNK).for_each(|chunk| {
            for xi in chunk {
                *xi *= alpha;
            }
        });
        Ok(())
    }

    fn multiply(&self, a: &Vec<f64>, x: &Vec<f64>, y: &mut Vec<f64>) -> Result<()> {
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
        Ok(())
    }

    fn shift(&self, delta: f64, x: &mut Vec<f64>) -> Result<()> {
        use rayon::prelude::*;
        x.par_chunks_mut(CHUNK).for_each(|chunk| {
            for xi in chunk {
                *xi += delta;
            }
        });
        Ok(())
    }

    fn fill(&self, value: f64, x: &mut Vec<f64>) -> Result<()> {
        use rayon::prelude::*;
        x.par_chunks_mut(CHUNK).for_each(|chunk| chunk.fill(value));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_removes_the_constant_component() {
        let b = CpuSerial;
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        b.project_mean_zero(&mut x).unwrap();
        assert!((reduce::sum(&x)).abs() < 1e-15);
        assert!((x[0] + 1.5).abs() < 1e-15);
    }

    #[test]
    fn projection_is_idempotent() {
        let b = CpuSerial;
        let mut x: Vec<f64> = (0..1000).map(|i| (i as f64).sin() + 3.0).collect();
        b.project_mean_zero(&mut x).unwrap();
        let once = x.clone();
        b.project_mean_zero(&mut x).unwrap();
        // 2 回目の射影で動く量は、1 回目の残差平均ぶん（数 ULP）に収まるべき
        let scale = once.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let shift = once
            .iter()
            .zip(&x)
            .map(|(a, c)| (a - c).abs())
            .fold(0.0f64, f64::max);
        assert!(shift < 1e-14 * scale, "shift = {shift}");
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
        let ms = serial.upload_matrix(&matrix).unwrap();
        let mp = parallel.upload_matrix(&matrix).unwrap();

        let mut ys = vec![0.0; n];
        let mut yp = vec![0.0; n];
        serial.spmv(&ms, &x, &mut ys).unwrap();
        parallel.spmv(&mp, &x, &mut yp).unwrap();
        assert_eq!(ys, yp, "SpMV");

        assert_eq!(
            serial.dot(&x, &z).unwrap(),
            parallel.dot(&x, &z).unwrap(),
            "内積"
        );
        assert_eq!(serial.sum(&x).unwrap(), parallel.sum(&x).unwrap(), "総和");
        assert_eq!(
            serial.norm2(&x).unwrap(),
            parallel.norm2(&x).unwrap(),
            "ノルム"
        );

        let mut a = x.clone();
        let mut b = x.clone();
        serial.axpy(1.5, &z, &mut a).unwrap();
        parallel.axpy(1.5, &z, &mut b).unwrap();
        assert_eq!(a, b, "axpy");

        serial.xpby(&z, 0.25, &mut a).unwrap();
        parallel.xpby(&z, 0.25, &mut b).unwrap();
        assert_eq!(a, b, "xpby");

        serial.scale(0.5, &mut a).unwrap();
        parallel.scale(0.5, &mut b).unwrap();
        assert_eq!(a, b, "scale");

        let mut c = vec![0.0; n];
        let mut d = vec![0.0; n];
        serial.multiply(&z, &x, &mut c).unwrap();
        parallel.multiply(&z, &x, &mut d).unwrap();
        assert_eq!(c, d, "multiply");

        let mut e = x.clone();
        let mut f = x.clone();
        serial.project_mean_zero(&mut e).unwrap();
        parallel.project_mean_zero(&mut f).unwrap();
        assert_eq!(e, f, "射影");
    }

    #[test]
    fn parallel_backend_handles_sizes_that_are_not_multiples_of_the_chunk() {
        for n in [1usize, 7, CHUNK - 1, CHUNK, CHUNK + 1] {
            let matrix = tridiagonal(n);
            let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let mut ys = vec![0.0; n];
            let mut yp = vec![0.0; n];
            let ms = CpuSerial.upload_matrix(&matrix).unwrap();
            let mp = CpuParallel.upload_matrix(&matrix).unwrap();
            CpuSerial.spmv(&ms, &x, &mut ys).unwrap();
            CpuParallel.spmv(&mp, &x, &mut yp).unwrap();
            assert_eq!(ys, yp, "n = {n}");
        }
    }

    #[test]
    fn vector_operations() {
        let b = CpuSerial;
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![10.0, 20.0, 30.0];
        b.axpy(2.0, &x, &mut y).unwrap();
        assert_eq!(y, vec![12.0, 24.0, 36.0]);

        let mut z = vec![1.0, 1.0, 1.0];
        b.xpby(&x, 3.0, &mut z).unwrap();
        assert_eq!(z, vec![4.0, 5.0, 6.0]);

        b.scale(0.5, &mut z).unwrap();
        assert_eq!(z, vec![2.0, 2.5, 3.0]);

        let mut w = vec![0.0; 3];
        b.multiply(&vec![2.0, 4.0, 8.0], &x, &mut w).unwrap();
        assert_eq!(w, vec![2.0, 8.0, 24.0]);

        b.shift(1.0, &mut w).unwrap();
        assert_eq!(w, vec![3.0, 9.0, 25.0]);

        assert_eq!(b.dot(&x, &x).unwrap(), 14.0);
        assert_eq!(b.sum(&x).unwrap(), 6.0);
        assert!((b.norm2(&vec![3.0, 4.0]).unwrap() - 5.0).abs() < 1e-15);
    }

    #[test]
    fn transfers_round_trip() {
        for b in [&CpuSerial as &dyn TestBackend, &CpuParallel] {
            b.check();
        }
    }

    /// 転送まわりを両バックエンドで同じ手順で確かめるための小さな型消去。
    trait TestBackend {
        fn check(&self);
    }

    impl<B: LinearBackend<Vector = Vec<f64>>> TestBackend for B {
        fn check(&self) {
            let host = [1.0, 2.0, 3.0, 4.0];
            let v = self.upload(&host).unwrap();
            assert_eq!(self.len(&v), 4);

            let mut back = [0.0; 4];
            self.download(&v, &mut back).unwrap();
            assert_eq!(back, host);

            let mut z = self.zeros(4).unwrap();
            assert_eq!(z, vec![0.0; 4]);
            self.copy_from(&v, &mut z).unwrap();
            assert_eq!(z, host.to_vec());
        }
    }
}
