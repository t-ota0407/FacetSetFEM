//! CUDA のエラーを `facetsetfem-core` のエラーに直す。
//!
//! CLAUDE.md 設計原則 6 に従い、「何が」「どこで」「どうすれば直るか」を残す。
//! ドライバのエラーコードだけでは何も分からないので、必ず日本語の説明を添える。

use cudarc::driver::DriverError;
use facetsetfem_core::error::FacetSetFemError;

/// CUDA ドライバのエラーを、文脈つきの [`FacetSetFemError`] にする。
pub(crate) fn cuda_error(what: impl std::fmt::Display, e: &DriverError) -> FacetSetFemError {
    FacetSetFemError::internal(facetsetfem_core::tr!(
        "{what} (CUDA: {e})",
        "{what}（CUDA: {e}）"
    ))
}
