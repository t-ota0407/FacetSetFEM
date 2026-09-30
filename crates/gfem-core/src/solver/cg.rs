//! 射影付き前処理付き共役勾配法（SPEC §2.2 / §4.3）。
//!
//! 完全電極モデルの電極がある系（SPEC §2.5）は正則なので、射影をかけずに同じ CG で解く
//! （[`Gauge`]）。同じ行列で右辺を替えて何度か解くときは [`solve_many`] が前処理を使い回す。
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
use crate::case::SolverOptions;
use crate::error::{GfemError, Result};
use crate::solver::amg::AmgInfo;
use crate::solver::backend::LinearBackend;
use crate::solver::precond;
use std::time::Instant;

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
    /// 使用した前処理の名前
    pub preconditioner: &'static str,
    /// AMG の階層の要約（AMG 以外では `None`。SPEC §5.5）
    pub amg: Option<AmgInfo>,
    /// 前処理の準備にかかった時間 [秒]（AMG では階層の構築が入る）
    pub setup_seconds: f64,
}

impl CgReport {
    /// 収束しなかった理由を説明する文字列。収束していれば `None`。
    pub fn failure_reason(&self) -> Option<String> {
        if self.converged {
            return None;
        }
        Some(if self.cancelled {
            crate::tr!("Cancelled by the user", "利用者によって中断されました")
        } else if self.stalled {
            crate::tr!(
                "The residual did not improve for {} iterations (stalled at relative residual {:.3e})",
                "残差が {} 反復のあいだ改善しませんでした（相対残差 {:.3e} で停滞）",
                STALL_WINDOW,
                self.relative_residual
            )
        } else {
            crate::tr!(
                "Reached the maximum number of iterations, {} (relative residual {:.3e})",
                "最大反復回数 {} に達しました（相対残差 {:.3e}）",
                self.iterations,
                self.relative_residual
            )
        })
    }
}

/// 進捗の通知先。`false` を返すと計算を中断する（SPEC §6.2 の「キャンセル可能」）。
pub type ProgressFn<'a> = &'a mut dyn FnMut(usize, f64) -> bool;

/// 解く系の特異性（SPEC §2.2 / §2.5 / §4.3）。
///
/// 利用者が選ぶ設定ではなく、**解く系が決める**。電極がすべて一様電流密度なら全 Neumann で
/// 零空間が定数ベクトルなので射影をかける。完全電極モデルの電極があれば、接触インピーダンスの
/// 項で行列が正則になり、電位の基準は境界条件が決めるので射影しない。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Gauge {
    /// 全 Neumann。射影 CG で「全節点電位の平均ゼロ」の代表元を選ぶ（SPEC §2.2 (b)）
    #[default]
    MeanZero,
    /// 行列が正則。射影をかけない（SPEC §2.5）
    NotNeeded,
}

impl Gauge {
    /// 記録用の名前（実行サマリの `solver.gauge`。SPEC §5.5）。
    pub const fn label(self) -> &'static str {
        match self {
            Gauge::MeanZero => "mean_zero",
            Gauge::NotNeeded => "none",
        }
    }
}

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
    let mut reports = solve_many(
        backend,
        matrix,
        &[rhs],
        &mut [phi],
        options,
        Gauge::MeanZero,
        progress,
    )?;
    Ok(reports.remove(0))
}

/// 同じ行列で右辺を替えて順に解く（完全電極モデルの重ね合わせ。SPEC §2.5 / §4.3）。
///
/// **前処理は 1 回だけ作って使い回す**（AMG の階層の構築と VRAM への転送を含む）。
/// 右辺は並びの順に 1 本ずつ解くので、何本渡しても 1 本ずつの答えは
/// それだけを解いたときとビット単位で同じ。
///
/// 進捗の通知は反復回数を**通し**で数える（GUI の進捗が戻らないように）。
/// 収束しなかった右辺があればそこで止め、そこまでの報告を返す（呼び出し側がエラーにする）。
/// 前処理の準備時間は 1 本目の報告にだけ入れる。
pub fn solve_many<B: LinearBackend>(
    backend: &B,
    matrix: &CsrMatrix,
    rhs: &[&[f64]],
    solutions: &mut [&mut [f64]],
    options: &SolverOptions,
    gauge: Gauge,
    progress: Option<ProgressFn<'_>>,
) -> Result<Vec<CgReport>> {
    let n = matrix.n_rows();
    if rhs.len() != solutions.len() {
        return Err(GfemError::internal(crate::tr!(
            "The numbers of right-hand sides ({}) and solutions ({}) do not match",
            "右辺（{} 本）と解（{} 本）の数が合いません",
            rhs.len(),
            solutions.len()
        )));
    }
    for (f, phi) in rhs.iter().zip(solutions.iter()) {
        if f.len() != n || phi.len() != n {
            return Err(GfemError::internal(crate::tr!(
                "Vector lengths do not match the matrix: matrix {n}, right-hand side {}, solution {}",
                "ベクトルの長さが行列と合いません: 行列 {n}, 右辺 {}, 解 {}",
                f.len(),
                phi.len()
            )));
        }
    }
    let mut progress = progress;

    // 前処理の準備。AMG の階層はホスト側で組み立てるので、
    // どのバックエンドで解いても同じものができる（SPEC §4.5）
    let setup_started = Instant::now();
    let hierarchy = precond::build_hierarchy(matrix, options, gauge)?;
    let a = backend.upload_matrix(matrix)?;
    let mut precond = precond::build(backend, matrix, options, hierarchy.as_ref())?;
    let setup_seconds = setup_started.elapsed().as_secs_f64();

    let mut reports = Vec::with_capacity(rhs.len());
    let mut offset = 0usize;
    for (k, (f, phi)) in rhs.iter().zip(solutions.iter_mut()).enumerate() {
        let mut report = iterate(
            backend,
            &a,
            &mut precond,
            f,
            phi,
            options,
            gauge,
            &mut progress,
            offset,
        )?;
        report.setup_seconds = if k == 0 { setup_seconds } else { 0.0 };
        offset += report.iterations;
        let stop = !report.converged;
        reports.push(report);
        if stop {
            break;
        }
    }
    Ok(reports)
}

/// 前処理を載せ終えた状態で、右辺 1 本を解く。
///
/// `offset` は進捗の通知に足す反復回数（[`solve_many`] で通しに数えるため）。
#[allow(clippy::too_many_arguments)] // CG の状態をまとめて渡すための内部関数
fn iterate<B: LinearBackend>(
    backend: &B,
    a: &B::Matrix<'_>,
    precond: &mut precond::DevicePreconditioner<'_, B>,
    rhs: &[f64],
    phi: &mut [f64],
    options: &SolverOptions,
    gauge: Gauge,
    progress: &mut Option<ProgressFn<'_>>,
    offset: usize,
) -> Result<CgReport> {
    let n = rhs.len();
    let preconditioner = precond.label();
    let amg_info = precond.amg_info();
    // 零空間に直交させる射影。正則な系（SPEC §2.5）では何もしない
    let project = |v: &mut B::Vector| -> Result<()> {
        match gauge {
            Gauge::MeanZero => backend.project_mean_zero(v),
            Gauge::NotNeeded => Ok(()),
        }
    };

    // 右辺と初期値を零空間に直交させる。
    // f は解析的には 1ᵀf = 0 を満たすが、面積按分の丸めでわずかにずれるため落としておく
    let mut f = backend.upload(rhs)?;
    project(&mut f)?;
    let mut x = backend.upload(phi)?;
    project(&mut x)?;

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
            preconditioner,
            amg: amg_info,
            setup_seconds: 0.0,
        });
    }

    let mut r = backend.zeros(n)?;
    let mut z = backend.zeros(n)?;
    let mut p = backend.zeros(n)?;
    let mut ap = backend.zeros(n)?;

    // r = P (f − K φ)
    backend.spmv(a, &x, &mut r)?;
    backend.scale(-1.0, &mut r)?;
    backend.axpy(1.0, &f, &mut r)?;
    project(&mut r)?;

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
            preconditioner,
            amg: amg_info,
            setup_seconds: 0.0,
        });
    }

    precond.apply(backend, a, &r, &mut z)?;
    project(&mut z)?;
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

        backend.spmv(a, &p, &mut ap)?;
        // K p は零空間に直交するはずだが、丸めで漏れる成分を毎回落とす
        project(&mut ap)?;

        let pap = backend.dot(&p, &ap)?;
        if pap <= 0.0 || pap.is_nan() {
            return Err(GfemError::internal(crate::tr!(
                "CG broke down: pᵀKp = {pap} is not positive (iteration {iteration})",
                "CG が破綻しました: pᵀKp = {pap} が正ではありません（反復 {iteration}）"
            ))
            .with_hint(crate::tr!(
                "The stiffness matrix may not be positive definite. Check for degenerate elements, \
                 negative conductivities and disconnected meshes",
                "剛性行列が正定値でない可能性があります。退化要素や負の導電率、\
                 メッシュの非連結を確認してください"
            )));
        }
        let alpha = rz / pap;

        backend.axpy(alpha, &p, &mut x)?;
        backend.axpy(-alpha, &ap, &mut r)?;
        project(&mut r)?;

        residual = backend.norm2(&r)? / norm_f;
        history.push(residual);

        if let Some(cb) = progress.as_mut()
            && !cb(offset + iteration, residual)
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

        precond.apply(backend, a, &r, &mut z)?;
        project(&mut z)?;
        let rz_next = backend.dot(&r, &z)?;
        let beta = rz_next / rz;
        rz = rz_next;
        // p = z + beta p
        backend.xpby(&z, beta, &mut p)?;
    }

    // 代表元を「平均ゼロ」に揃える（丸めで漏れた定数成分を落とす）
    project(&mut x)?;
    backend.download(&x, phi)?;

    Ok(CgReport {
        iterations: iteration,
        relative_residual: residual,
        converged,
        stalled,
        cancelled,
        history,
        backend: backend.name(),
        preconditioner,
        amg: amg_info,
        setup_seconds: 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{Backend, Preconditioner};
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
        };
        let report = solve(&CpuSerial, &k, &f, &mut phi, &opts, None).unwrap();
        assert!(report.converged, "{:?}", report.failure_reason());
    }
}
