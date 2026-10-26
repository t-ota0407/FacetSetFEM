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
//! の 4 つで対称正定値になる。テスト（`crates/facetsetfem-core/tests/amg.rs`）で
//! `⟨M r, s⟩ = ⟨r, M s⟩` と `⟨M r, r⟩ > 0` を実際に確かめている。
//!
//! # メモリ
//!
//! レベルごとに作業ベクトルを 4 本（x, rhs, tmp, tmp2）持つ。細かいレベルが
//! 支配的なので、おおよそ節点数 × 4 本ぶん。
//!
//! **最細レベルの行列は載せない。** CG がすでにバックエンドへ載せているものを
//! [`apply`](DeviceHierarchy::apply) で受け取って使う。以前は同じ行列を階層側でも
//! もう 1 つ持ち、VRAM へ 2 回転送していた（1000 万節点で約 1.8 GB。SPEC §4.5）。
//! 同じ行列なので、結果はビット単位で変わらない。

use crate::error::Result;
use crate::solver::amg::Hierarchy;
use crate::solver::backend::LinearBackend;

/// 最粗レベルを密分解できなかったときに代わりに回す平滑化の回数。
const COARSE_FALLBACK_SWEEPS: usize = 20;

/// バックエンドに載せた 1 レベル。
struct DeviceLevel<'h, B: LinearBackend> {
    /// このレベルの行列。最細レベルは CG が載せたものを借りるので持たない（`None`）
    a: Option<B::Matrix<'h>>,
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
    host: &'h Hierarchy<'h>,
    levels: Vec<DeviceLevel<'h, B>>,
    work: Vec<LevelWork<B>>,
    /// 最粗レベルをホストで解くための受け皿
    coarse_rhs: Vec<f64>,
    coarse_x: Vec<f64>,
}

impl<'h, B: LinearBackend> DeviceHierarchy<'h, B> {
    /// 階層をバックエンドへ載せる（GPU なら VRAM へ転送する）。最細レベルの行列は載せない。
    pub fn new(backend: &B, host: &'h Hierarchy<'h>) -> Result<Self> {
        let mut levels = Vec::with_capacity(host.levels.len());
        let mut work = Vec::with_capacity(host.levels.len());
        for (index, level) in host.levels.iter().enumerate() {
            let n = level.a.n_rows();
            levels.push(DeviceLevel {
                a: if index == 0 {
                    None
                } else {
                    Some(backend.upload_matrix(&level.a)?)
                },
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
    pub fn host(&self) -> &'h Hierarchy<'h> {
        self.host
    }

    /// `z = M⁻¹ r`。V サイクルを 1 回まわす。
    ///
    /// `fine` は最細レベルの行列（階層を作った行列）をバックエンドに載せたもの。
    /// CG が持っているものをそのまま渡す。
    pub fn apply(
        &mut self,
        backend: &B,
        fine: &B::Matrix<'_>,
        r: &B::Vector,
        z: &mut B::Vector,
    ) -> Result<()> {
        backend.copy_from(r, &mut self.work[0].rhs)?;
        self.v_cycle(backend, fine)?;
        backend.copy_from(&self.work[0].x, z)
    }

    fn v_cycle(&mut self, backend: &B, fine: &B::Matrix<'_>) -> Result<()> {
        let last = self.levels.len() - 1;
        let (presmooth, postsmooth) = (self.host.presmooth, self.host.postsmooth);
        let levels = &self.levels;
        let work = &mut self.work;

        // 各レベルの行列。最細レベルだけは借りたもの（寿命の違う型なので呼び分ける）
        macro_rules! with_matrix {
            ($level:expr, |$a:ident| $body:expr) => {
                match &$level.a {
                    Some($a) => $body,
                    None => {
                        let $a = fine;
                        $body
                    }
                }
            };
        }

        // --- 下り: 平滑化して残差を粗いレベルへ渡す -------------------------
        for (l, level) in levels.iter().enumerate().take(last) {
            with_matrix!(level, |a| descend(backend, a, level, work, l, presmooth))?;
        }

        // --- 最粗レベル -----------------------------------------------------
        if self.host.coarse.is_direct() {
            backend.download(&work[last].rhs, &mut self.coarse_rhs)?;
            self.host.coarse.solve(&self.coarse_rhs, &mut self.coarse_x);
            backend.upload_into(&self.coarse_x, &mut work[last].x)?;
        } else {
            // 分解できなかった場合の保険。反復回数は増えるが答えは変わらない
            with_matrix!(levels[last], |a| smooth_from_zero(
                backend,
                a,
                &levels[last],
                &mut work[last],
                COARSE_FALLBACK_SWEEPS,
            ))?;
        }

        // --- 上り: 補正を補間して足し、もう一度平滑化 -----------------------
        for (l, level) in levels.iter().enumerate().take(last).rev() {
            with_matrix!(level, |a| ascend(backend, a, level, work, l, postsmooth))?;
        }
        Ok(())
    }
}

/// 下り 1 レベル: 平滑化して残差を 1 つ粗いレベルの右辺へ制限する。
fn descend<B: LinearBackend>(
    backend: &B,
    a: &B::Matrix<'_>,
    level: &DeviceLevel<'_, B>,
    work: &mut [LevelWork<B>],
    l: usize,
    sweeps: usize,
) -> Result<()> {
    smooth_from_zero(backend, a, level, &mut work[l], sweeps)?;
    residual(backend, a, &mut work[l])?;
    let (fine, coarse) = work.split_at_mut(l + 1);
    let r = level.r.as_ref().expect("最粗レベル以外には R がある");
    backend.spmv(r, &fine[l].tmp, &mut coarse[0].rhs)
}

/// 上り 1 レベル: 1 つ粗いレベルの補正を補間して足し、もう一度平滑化する。
fn ascend<B: LinearBackend>(
    backend: &B,
    a: &B::Matrix<'_>,
    level: &DeviceLevel<'_, B>,
    work: &mut [LevelWork<B>],
    l: usize,
    sweeps: usize,
) -> Result<()> {
    let (fine, coarse) = work.split_at_mut(l + 1);
    let p = level.p.as_ref().expect("最粗レベル以外には P がある");
    backend.spmv(p, &coarse[0].x, &mut fine[l].tmp)?;
    let w = &mut fine[l];
    backend.axpy(1.0, &w.tmp, &mut w.x)?;
    smooth(backend, a, level, w, sweeps)
}

/// `x ← 0` から始める平滑化。1 回目は `A x = 0` なので SpMV を 1 回省ける。
fn smooth_from_zero<B: LinearBackend>(
    backend: &B,
    a: &B::Matrix<'_>,
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
        sweep(backend, a, level, w)?;
    }
    Ok(())
}

/// 既に値の入っている `x` からの平滑化。
fn smooth<B: LinearBackend>(
    backend: &B,
    a: &B::Matrix<'_>,
    level: &DeviceLevel<'_, B>,
    w: &mut LevelWork<B>,
    sweeps: usize,
) -> Result<()> {
    for _ in 0..sweeps {
        sweep(backend, a, level, w)?;
    }
    Ok(())
}

/// 減衰 Jacobi を 1 回: `x += ω D⁻¹ (rhs − A x)`。
fn sweep<B: LinearBackend>(
    backend: &B,
    a: &B::Matrix<'_>,
    level: &DeviceLevel<'_, B>,
    w: &mut LevelWork<B>,
) -> Result<()> {
    residual(backend, a, w)?;
    backend.multiply(&level.inv_diag, &w.tmp, &mut w.tmp2)?;
    backend.axpy(level.omega, &w.tmp2, &mut w.x)
}

/// `tmp ← rhs − A x`。
fn residual<B: LinearBackend>(backend: &B, a: &B::Matrix<'_>, w: &mut LevelWork<B>) -> Result<()> {
    backend.spmv(a, &w.x, &mut w.tmp)?;
    backend.xpby(&w.rhs, -1.0, &mut w.tmp)
}
