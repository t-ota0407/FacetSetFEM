//! CUDA のエラーを `gfem-core` のエラーに直す。
//!
//! CLAUDE.md 設計原則 6 に従い、「何が」「どこで」「どうすれば直るか」を残す。
//! ドライバのエラーコードだけでは何も分からないので、必ず日本語の説明を添える。

use cudarc::driver::DriverError;
use gfem_core::error::GfemError;

/// CUDA ドライバのエラーを、文脈つきの [`GfemError`] にする。
pub(crate) fn cuda_error(what: impl std::fmt::Display, e: &DriverError) -> GfemError {
    GfemError::internal(gfem_core::tr!("{what} (CUDA: {e})", "{what}（CUDA: {e}）"))
}
