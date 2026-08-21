//! 射影付き前処理付き共役勾配法（SPEC §2.2 / §4.3）。
//!
//! 全 Neumann 境界なので K は特異で、零空間は定数ベクトル span{1}。
//! **1 節点ピン留めはしない**——局所的に解を歪め、条件数も悪化させるため。
//! 代わりに射影
//!
//! ```text
//! P = I − 11ᵀ/N          (SPEC §2.2 (b))
//! ```
//!
//! を残差・探索方向・前処理後のベクトルに毎反復かける。こうすると CG は
//! 零空間に直交する部分空間の中だけで動き、そこでは K が正定値なので通常どおり収束する。
//!
//! アルゴリズム本体はここ 1 箇所だけに書く（SPEC §4.4）。CPU 逐次・CPU 並列・GPU の
//! どれで走らせても同じ結果になるよう、ベクトル演算は全て [`LinearBackend`] 経由で行う。
//!
//! # ホストとデバイスの境目
//!
//! 作業ベクトル（r, z, p, Ap と解）はバックエンドが持つ表現で確保し、
//! **反復の間は一度もホストへ戻さない**。GPU では VRAM に常駐することになる（SPEC §8 M5）。
//! ホストへ戻すのは内積・総和の結果というスカラーだけで、α・β・残差の割り算は
//! ホストの `f64` で行う。こうしておくと、スカラー演算の順序がバックエンドによって
//! 変わりようがない。

use crate::assemble::csr::CsrMatrix;
use crate::case::{Preconditioner, SolverOptions};
use crate::error::{GfemError, Result};
use crate::solver::backend::LinearBackend;

/// 停滞とみなすまでの反復数。
const STALL_WINDOW: usize = 500;

/// 「改善した」とみなす残差の減少率。
const STALL_IMPROVEMENT: f64 = 0.99;

/// CG の実行結果。
#[derive(Debug, Clone)]
pub struct CgReport {
    /// 実行した反復回数
    pub iterations: usize,
    /// 最終的な相対残差 ‖r‖/‖f‖
    pub relative_residual: f64,
    /// 収束判定を満たしたか
    pub converged: bool,
    /// 停滞を検出して打ち切ったか
    pub stalled: bool,
    /// 進捗コールバックから中断されたか
    pub cancelled: bool,
    /// 反復ごとの相対残差（GUI の収束曲線用。SPEC §4.3）
    pub history: Vec<f64>,
    /// 使用したバックエンド名
    pub backend: &'static str,
}

impl CgReport {
    /// 収束しなかった理由を説明する文字列。収束していれば `None`。
    pub fn failure_reason(&self) -> Option<String> {
        if self.converged {
            return None;
        }
        Some(if self.cancelled {
            "利用者によって中断されました".to_string()
        } else if self.stalled {
            format!(
                "残差が {} 反復のあいだ改善しませんでした（相対残差 {:.3e} で停滞）",
                STALL_WINDOW, self.relative_residual
            )
        } else {
            format!(
                "最大反復回数 {} に達しました（相対残差 {:.3e}）",
                self.iterations, self.relative_residual
            )
        })
    }
}

/// 進捗の通知先。`false` を返すと計算を中断する（SPEC §6.2 の「キャンセル可能」）。
pub type ProgressFn<'a> = &'a mut dyn FnMut(usize, f64) -> bool;

/// `K φ = f` を射影付き前処理付き CG で解く。
///
/// `phi` は初期値かつ出力。ゼロで渡すのが普通。
/// 解はゲージ固定により**全節点電位の平均がゼロ**の代表元になる。
pub fn solve<B: LinearBackend>(
    backend: &B,
    matrix: &CsrMatrix,
    rhs: &[f64],
    phi: &mut [f64],
    options: &SolverOptions,
    progress: Option<ProgressFn<'_>>,
) -> Result<CgReport> {
    let n = matrix.n_rows();
    if rhs.len() != n || phi.len() != n {
        return Err(GfemError::internal(format!(
            "ベクトルの長さが行列と合いません: 行列 {n}, 右辺 {}, 解 {}",
            rhs.len(),
            phi.len()
        )));
    }
    let mut progress = progress;

    // 前処理（対角の逆数）。ホスト側で作ってから転送する
    let inv_diag = match options.preconditioner {
        Preconditioner::None => None,
        Preconditioner::Jacobi => {
            let diag = matrix.diagonal();
            if let Some((i, d)) = diag
                .iter()
                .enumerate()
                .find(|&(_, &d)| d <= 0.0 || d.is_nan())
            {
                return Err(GfemError::internal(format!(
                    "剛性行列の対角成分が正ではありません: 行 {i} で {d}"
                ))
                .with_hint("退化要素か、導電率がゼロ以下の領域が残っている可能性があります"));
            }
            Some(diag.iter().map(|d| 1.0 / d).collect::<Vec<f64>>())
        }
    };

    let a = backend.upload_matrix(matrix)?;
    let inv_diag = match inv_diag {
        Some(d) => Some(backend.upload(&d)?),
        None => None,
    };

    // 右辺と初期値を零空間に直交させる。
    // f は解析的には 1ᵀf = 0 を満たすが、面積按分の丸めでわずかにずれるため落としておく
    let mut f = backend.upload(rhs)?;
    backend.project_mean_zero(&mut f)?;
    let mut x = backend.upload(phi)?;
    backend.project_mean_zero(&mut x)?;

    let norm_f = backend.norm2(&f)?;
    let mut history = Vec::new();
    if norm_f == 0.0 {
        // 電流が入っていない。解は恒等的にゼロ
        phi.fill(0.0);
        return Ok(CgReport {
            iterations: 0,
            relative_residual: 0.0,
            converged: true,
            stalled: false,
            cancelled: false,
            history,
            backend: backend.name(),
        });
    }

    let mut r = backend.zeros(n)?;
    let mut z = backend.zeros(n)?;
    let mut p = backend.zeros(n)?;
    let mut ap = backend.zeros(n)?;

    // r = P (f − K φ)
    backend.spmv(&a, &x, &mut r)?;
    backend.scale(-1.0, &mut r)?;
    backend.axpy(1.0, &f, &mut r)?;
    backend.project_mean_zero(&mut r)?;

    let mut residual = backend.norm2(&r)? / norm_f;
    history.push(residual);
    if residual < options.tolerance {
        backend.download(&x, phi)?;
        return Ok(CgReport {
            iterations: 0,
            relative_residual: residual,
            converged: true,
            stalled: false,
            cancelled: false,
            history,
            backend: backend.name(),
        });
    }

    apply_preconditioner(backend, inv_diag.as_ref(), &r, &mut z)?;
    backend.project_mean_zero(&mut z)?;
    backend.copy_from(&z, &mut p)?;
    let mut rz = backend.dot(&r, &z)?;

    let mut best = residual;
    let mut best_iteration = 0usize;
    let mut converged = false;
    let mut stalled = false;
    let mut cancelled = false;
    let mut iteration = 0usize;

    while iteration < options.max_iterations {
        iteration += 1;

        backend.spmv(&a, &p, &mut ap)?;
        // K p は零空間に直交するはずだが、丸めで漏れる成分を毎回落とす
        backend.project_mean_zero(&mut ap)?;

        let pap = backend.dot(&p, &ap)?;
        if pap <= 0.0 || pap.is_nan() {
            return Err(GfemError::internal(format!(
                "CG が破綻しました: pᵀKp = {pap} が正ではありません（反復 {iteration}）"
            ))
            .with_hint(
                "剛性行列が正定値でない可能性があります。退化要素や負の導電率、\
                 メッシュの非連結を確認してください",
            ));
        }
        let alpha = rz / pap;

        backend.axpy(alpha, &p, &mut x)?;
        backend.axpy(-alpha, &ap, &mut r)?;
        backend.project_mean_zero(&mut r)?;

        residual = backend.norm2(&r)? / norm_f;
        history.push(residual);

        if let Some(cb) = progress.as_deref_mut()
            && !cb(iteration, residual)
        {
            cancelled = true;
            break;
        }

        if residual < options.tolerance {
            converged = true;
            break;
        }

        // 停滞検出（SPEC §4.3）
        if residual < best * STALL_IMPROVEMENT {
            best = residual;
            best_iteration = iteration;
        } else if iteration - best_iteration >= STALL_WINDOW {
            stalled = true;
            break;
        }

        apply_preconditioner(backend, inv_diag.as_ref(), &r, &mut z)?;
        backend.project_mean_zero(&mut z)?;
        let rz_next = backend.dot(&r, &z)?;
        let beta = rz_next / rz;
        rz = rz_next;
        // p = z + beta p
        backend.xpby(&z, beta, &mut p)?;
    }

    // 代表元を「平均ゼロ」に揃える（丸めで漏れた定数成分を落とす）
    backend.project_mean_zero(&mut x)?;
    backend.download(&x, phi)?;

    Ok(CgReport {
        iterations: iteration,
        relative_residual: residual,
        converged,
        stalled,
        cancelled,
        history,
        backend: backend.name(),
    })
}

fn apply_preconditioner<B: LinearBackend>(
    backend: &B,
    inv_diag: Option<&B::Vector>,
    r: &B::Vector,
    z: &mut B::Vector,
) -> Result<()> {
    match inv_diag {
        Some(d) => backend.multiply(d, r, z),
        None => backend.copy_from(r, z),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::Backend;
    use crate::solver::backend::CpuSerial;

    /// 1 次元の格子ラプラシアン（全 Neumann）。K = tridiag(−1, 2, −1) の端を 1 にしたもの。
    ///
    /// 零空間はちょうど定数ベクトルなので、射影 CG の検証にそのまま使える。
    fn laplacian_1d(n: usize) -> CsrMatrix {
        let mut row_ptr = vec![0usize];
        let mut col_idx: Vec<u32> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        for i in 0..n {
            let mut entries: Vec<(u32, f64)> = Vec::new();
            let mut diag = 0.0;
            if i > 0 {
                entries.push((i as u32 - 1, -1.0));
                diag += 1.0;
            }
            if i + 1 < n {
                entries.push((i as u32 + 1, -1.0));
                diag += 1.0;
            }
            entries.push((i as u32, diag));
            entries.sort_by_key(|&(c, _)| c);
            for (c, v) in entries {
                col_idx.push(c);
                values.push(v);
            }
            row_ptr.push(col_idx.len());
        }
        let mut m = CsrMatrix::from_structure(n, row_ptr, col_idx);
        m.values_mut().copy_from_slice(&values);
        m
    }

    fn options() -> SolverOptions {
        SolverOptions {
            tolerance: 1.0e-12,
            max_iterations: 5000,
            preconditioner: Preconditioner::Jacobi,
            backend: Backend::CpuSerial,
        }
    }

    #[test]
    fn solves_a_singular_neumann_system() {
        let n = 200;
        let k = laplacian_1d(n);
        // 両端に ±1 を入れる（1ᵀf = 0）
        let mut f = vec![0.0; n];
        f[0] = 1.0;
        f[n - 1] = -1.0;

        let mut phi = vec![0.0; n];
        let report = solve(&CpuSerial, &k, &f, &mut phi, &options(), None).unwrap();
        assert!(report.converged, "{:?}", report.failure_reason());

        // 残差を直接確認する
        let mut kx = vec![0.0; n];
        k.spmv(&phi, &mut kx);
        let residual: f64 = kx
            .iter()
            .zip(&f)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        assert!(residual < 1e-10, "残差 {residual}");

        // ゲージ固定: 平均がゼロ
        let mean: f64 = phi.iter().sum::<f64>() / n as f64;
        assert!(mean.abs() < 1e-12, "平均 {mean}");

        // 1 次元定電流なので φ は線形に落ちる
        let slope = phi[1] - phi[0];
        for i in 1..n {
            assert!((phi[i] - phi[i - 1] - slope).abs() < 1e-9, "節点 {i}");
        }
    }

    #[test]
    fn result_is_bitwise_repeatable() {
        let n = 300;
        let k = laplacian_1d(n);
        let mut f = vec![0.0; n];
        f[0] = 2.0e-3;
        f[n - 1] = -2.0e-3;

        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        let ra = solve(&CpuSerial, &k, &f, &mut a, &options(), None).unwrap();
        let rb = solve(&CpuSerial, &k, &f, &mut b, &options(), None).unwrap();
        assert_eq!(a, b);
        assert_eq!(ra.iterations, rb.iterations);
        assert_eq!(ra.history, rb.history);
    }

    #[test]
    fn a_constant_shift_in_the_initial_guess_does_not_change_the_answer() {
        let n = 120;
        let k = laplacian_1d(n);
        let mut f = vec![0.0; n];
        f[0] = 1.0;
        f[n - 1] = -1.0;

        let mut a = vec![0.0; n];
        let mut b = vec![1234.5; n];
        solve(&CpuSerial, &k, &f, &mut a, &options(), None).unwrap();
        solve(&CpuSerial, &k, &f, &mut b, &options(), None).unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn zero_right_hand_side_gives_zero_solution() {
        let k = laplacian_1d(50);
        let f = vec![0.0; 50];
        let mut phi = vec![0.0; 50];
        let report = solve(&CpuSerial, &k, &f, &mut phi, &options(), None).unwrap();
        assert!(report.converged);
        assert_eq!(report.iterations, 0);
        assert!(phi.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn progress_callback_can_cancel() {
        let n = 500;
        let k = laplacian_1d(n);
        let mut f = vec![0.0; n];
        f[0] = 1.0;
        f[n - 1] = -1.0;
        let mut phi = vec![0.0; n];
        let mut seen = Vec::new();
        let mut cb = |it: usize, res: f64| {
            seen.push((it, res));
            it < 5
        };
        let report = solve(&CpuSerial, &k, &f, &mut phi, &options(), Some(&mut cb)).unwrap();
        assert!(report.cancelled);
        assert_eq!(report.iterations, 5);
        assert_eq!(seen.len(), 5);
        assert!(report.failure_reason().unwrap().contains("中断"));
    }

    #[test]
    fn hitting_the_iteration_limit_is_reported() {
        let n = 400;
        let k = laplacian_1d(n);
        let mut f = vec![0.0; n];
        f[0] = 1.0;
        f[n - 1] = -1.0;
        let mut phi = vec![0.0; n];
        let opts = SolverOptions {
            tolerance: 1.0e-14,
            max_iterations: 3,
            preconditioner: Preconditioner::Jacobi,
            backend: Backend::CpuSerial,
        };
        let report = solve(&CpuSerial, &k, &f, &mut phi, &opts, None).unwrap();
        assert!(!report.converged);
        assert_eq!(report.iterations, 3);
        assert!(report.failure_reason().unwrap().contains("最大反復回数"));
    }

    #[test]
    fn works_without_a_preconditioner_too() {
        let n = 100;
        let k = laplacian_1d(n);
        let mut f = vec![0.0; n];
        f[0] = 1.0;
        f[n - 1] = -1.0;
        let mut phi = vec![0.0; n];
        let opts = SolverOptions {
            tolerance: 1.0e-12,
            max_iterations: 5000,
            preconditioner: Preconditioner::None,
            backend: Backend::CpuSerial,
        };
        let report = solve(&CpuSerial, &k, &f, &mut phi, &opts, None).unwrap();
        assert!(report.converged, "{:?}", report.failure_reason());
    }
}
