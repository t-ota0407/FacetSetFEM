//! 線形ソルバ（SPEC §4.3 / §4.4）。
//!
//! - [`reduce`] — 決定的な総和・内積。再現性の土台
//! - [`backend`] — 線形代数バックエンドの抽象化（CPU 逐次 / CPU 並列 / CUDA）
//! - [`cg`] — 射影付き前処理付き共役勾配法。アルゴリズム本体はここ 1 箇所だけ
//! - [`precond`] — 前処理の切り替え（なし / Jacobi / AMG）
//! - [`amg`] — 代数マルチグリッド（平滑化凝集）。SPEC §4.5
//! - [`external`] — 外部クレート（`gfem-cuda`）のバックエンドを差し込む口

pub mod amg;
pub mod backend;
pub mod cg;
pub mod external;
pub mod precond;
pub mod reduce;

pub use amg::{AmgInfo, AmgOptions, Hierarchy};
pub use backend::{CpuParallel, CpuSerial, LinearBackend};
pub use cg::{CgReport, ProgressFn};
pub use external::{BackendSolver, LinearSolver};
pub use precond::DevicePreconditioner;
