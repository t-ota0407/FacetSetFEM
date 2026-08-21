//! V サイクルの実行（SPEC §4.5）。
//!
//! 階層（[`Hierarchy`]）はホスト側で作った「ただの行列の列」なので、
//! それをバックエンドに載せて回すのがここの仕事。使う演算は
//! [`LinearBackend`] にもとからあるものだけ（SpMV・要素積・axpy・xpby）なので、
//! **CPU 逐次・CPU 並列・GPU で自動的にビット単位一致する**。
//!
//! ```text
//! 細 ─┬─ 平滑化 ──→ 残差 ──→ R で制限 ─┐
//!     │                                 ↓
//!     │                            （1 つ粗いレベルで同じこと）
//!     │                                 ↓
//!     └─ 平滑化 ←── P で補間して加算 ←──┘
//! ```
//!
//! # CG の前処理として使えること
//!
//! CG は前処理作用素が**対称正定値**であることを要求する。ここでは
//!
//! - 下りと上りの平滑化回数を同じにする
//! - 制限を補間の転置にする（`R = Pᵀ`）
//! - 最粗レベルを直接解く
//! - 減衰 Jacobi の ω を `4/(3ρ) < 2/ρ` に採る（平滑化作用素が A に対して収束的）
//!
//! の 4 つで対称正定値になる。テスト（`crates/gfem-core/tests/amg.rs`）で
//! `⟨M r, s⟩ = ⟨r, M s⟩` と `⟨M r, r⟩ > 0` を実際に確かめている。
//!
//! # メモリ
//!
//! レベルごとに作業ベクトルを 4 本（x, rhs, tmp, tmp2）持つ。細かいレベルが
//! 支配的なので、おおよそ節点数 × 4 本ぶん。最細レベルの行列は CG が持つものと
//! 同じ内容を階層側でもう 1 つ持つ（改善の余地。SPEC §4.5）。

use crate::error::Result;
use crate::solver::amg::Hierarchy;
use crate::solver::backend::LinearBackend;

/// 最粗レベルを密分解できなかったときに代わりに回す平滑化の回数。
const COARSE_FALLBACK_SWEEPS: usize = 20;

/// バックエンドに載せた 1 レベル。
struct DeviceLevel<'h, B: LinearBackend> {
    a: B::Matrix<'h>,
    p: Option<B::Matrix<'h>>,
    r: Option<B::Matrix<'h>>,
    inv_diag: B::Vector,
    omega: f64,
}

/// レベルごとの作業ベクトル。
struct LevelWork<B: LinearBackend> {
    /// このレベルの補正量
    x: B::Vector,
    /// このレベルの右辺（1 つ細かいレベルの残差を制限したもの）
    rhs: B::Vector,
    tmp: B::Vector,
    tmp2: B::Vector,
}

/// バックエンドに載せた階層。
pub struct DeviceHierarchy<'h, B: LinearBackend> {
    host: &'h Hierarchy,
    levels: Vec<DeviceLevel<'h, B>>,
    work: Vec<LevelWork<B>>,
    /// 最粗レベルをホストで解くための受け皿
    coarse_rhs: Vec<f64>,
    coarse_x: Vec<f64>,
}

impl<'h, B: LinearBackend> DeviceHierarchy<'h, B> {
    /// 階層をバックエンドへ載せる（GPU なら VRAM へ転送する）。
    pub fn new(backend: &B, host: &'h Hierarchy) -> Result<Self> {
        let mut levels = Vec::with_capacity(host.levels.len());
        let mut work = Vec::with_capacity(host.levels.len());
        for level in &host.levels {
            let n = level.a.n_rows();
            levels.push(DeviceLevel {
                a: backend.upload_matrix(&level.a)?,
                p: match &level.transfer {
                    Some(t) => Some(backend.upload_matrix(&t.p)?),
                    None => None,
                },
                r: match &level.transfer {
                    Some(t) => Some(backend.upload_matrix(&t.r)?),
                    None => None,
                },
                inv_diag: backend.upload(&level.inv_diag)?,
                omega: level.omega,
            });
            work.push(LevelWork {
                x: backend.zeros(n)?,
                rhs: backend.zeros(n)?,
                tmp: backend.zeros(n)?,
                tmp2: backend.zeros(n)?,
            });
        }
        let coarse_n = host.levels.last().expect("レベルが 1 つも無い").a.n_rows();
        Ok(Self {
            host,
            levels,
            work,
            coarse_rhs: vec![0.0; coarse_n],
            coarse_x: vec![0.0; coarse_n],
        })
    }

    /// もとになった階層。
    pub fn host(&self) -> &'h Hierarchy {
        self.host
    }

    /// `z = M⁻¹ r`。V サイクルを 1 回まわす。
    pub fn apply(&mut self, backend: &B, r: &B::Vector, z: &mut B::Vector) -> Result<()> {
        backend.copy_from(r, &mut self.work[0].rhs)?;
        self.v_cycle(backend)?;
        backend.copy_from(&self.work[0].x, z)
    }

    fn v_cycle(&mut self, backend: &B) -> Result<()> {
        let last = self.levels.len() - 1;
        let (presmooth, postsmooth) = (self.host.presmooth, self.host.postsmooth);
        let levels = &self.levels;
        let work = &mut self.work;

        // --- 下り: 平滑化して残差を粗いレベルへ渡す -------------------------
        for l in 0..last {
            smooth_from_zero(backend, &levels[l], &mut work[l], presmooth)?;
            residual(backend, &levels[l], &mut work[l])?;
            let (fine, coarse) = work.split_at_mut(l + 1);
            let r = levels[l].r.as_ref().expect("最粗レベル以外には R がある");
            backend.spmv(r, &fine[l].tmp, &mut coarse[0].rhs)?;
        }

        // --- 最粗レベル -----------------------------------------------------
        if self.host.coarse.is_direct() {
            backend.download(&work[last].rhs, &mut self.coarse_rhs)?;
            self.host.coarse.solve(&self.coarse_rhs, &mut self.coarse_x);
            backend.upload_into(&self.coarse_x, &mut work[last].x)?;
        } else {
            // 分解できなかった場合の保険。反復回数は増えるが答えは変わらない
            smooth_from_zero(
                backend,
                &levels[last],
                &mut work[last],
                COARSE_FALLBACK_SWEEPS,
            )?;
        }

        // --- 上り: 補正を補間して足し、もう一度平滑化 -----------------------
        for l in (0..last).rev() {
            let (fine, coarse) = work.split_at_mut(l + 1);
            let p = levels[l].p.as_ref().expect("最粗レベル以外には P がある");
            backend.spmv(p, &coarse[0].x, &mut fine[l].tmp)?;
            let w = &mut fine[l];
            backend.axpy(1.0, &w.tmp, &mut w.x)?;
            smooth(backend, &levels[l], w, postsmooth)?;
        }
        Ok(())
    }
}

/// `x ← 0` から始める平滑化。1 回目は `A x = 0` なので SpMV を 1 回省ける。
fn smooth_from_zero<B: LinearBackend>(
    backend: &B,
    level: &DeviceLevel<'_, B>,
    w: &mut LevelWork<B>,
    sweeps: usize,
) -> Result<()> {
    if sweeps == 0 {
        return backend.fill(0.0, &mut w.x);
    }
    // x = ω D⁻¹ rhs
    backend.multiply(&level.inv_diag, &w.rhs, &mut w.x)?;
    backend.scale(level.omega, &mut w.x)?;
    for _ in 1..sweeps {
        sweep(backend, level, w)?;
    }
    Ok(())
}

/// 既に値の入っている `x` からの平滑化。
fn smooth<B: LinearBackend>(
    backend: &B,
    level: &DeviceLevel<'_, B>,
    w: &mut LevelWork<B>,
    sweeps: usize,
) -> Result<()> {
    for _ in 0..sweeps {
        sweep(backend, level, w)?;
    }
    Ok(())
}

/// 減衰 Jacobi を 1 回: `x += ω D⁻¹ (rhs − A x)`。
fn sweep<B: LinearBackend>(
    backend: &B,
    level: &DeviceLevel<'_, B>,
    w: &mut LevelWork<B>,
) -> Result<()> {
    residual(backend, level, w)?;
    backend.multiply(&level.inv_diag, &w.tmp, &mut w.tmp2)?;
    backend.axpy(level.omega, &w.tmp2, &mut w.x)
}

/// `tmp ← rhs − A x`。
fn residual<B: LinearBackend>(
    backend: &B,
    level: &DeviceLevel<'_, B>,
    w: &mut LevelWork<B>,
) -> Result<()> {
    backend.spmv(&level.a, &w.x, &mut w.tmp)?;
    backend.xpby(&w.rhs, -1.0, &mut w.tmp)
}
