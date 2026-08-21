//! 線形ソルバ（SPEC §4.3 / §4.4）。
//!
//! - [`reduce`] — 決定的な総和・内積。再現性の土台
//! - [`backend`] — 線形代数バックエンドの抽象化（CPU 逐次 / CPU 並列 / CUDA）
//! - [`cg`] — 射影付き前処理付き共役勾配法。アルゴリズム本体はここ 1 箇所だけ

pub mod backend;
pub mod cg;
pub mod reduce;

pub use backend::{CpuParallel, CpuSerial, LinearBackend};
pub use cg::{CgReport, ProgressFn};
