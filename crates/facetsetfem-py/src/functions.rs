//! モジュール直下の関数: 長さ単位・表示言語・ソルバと VTU の選択肢・GPU の問い合わせ（SPEC §6.5）。
//!
//! 選択肢と別名の表は `facetsetfem-core` の 1 か所にあり、ここはそれを Python に見せるだけ。

use crate::convert::to_py;
use crate::cuda;
use facetsetfem_core::case::{Backend, ElectrodeModel, Preconditioner};
use facetsetfem_core::{LengthUnit, WedgeOrder};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// 対応する長さ単位の一覧。
#[pyfunction]
pub(crate) fn length_units() -> Vec<&'static str> {
    LengthUnit::ALL.iter().map(|u| u.symbol()).collect()
}

/// 長さ単位 `unit` の 1 が何メートルか（`"mm"` なら 0.001）。
///
/// 換算表は Rust 側（`facetsetfem_core::units`）の 1 か所だけに置き、Python はここから引く。
#[pyfunction]
pub(crate) fn metres_per_unit(unit: &str) -> PyResult<f64> {
    Ok(LengthUnit::parse(unit).map_err(to_py)?.to_meters())
}

/// 表示言語を決める（`"en"` / `"ja"`。SPEC §6.6）。エラー・警告・レポートの文言が変わる。
#[pyfunction]
pub(crate) fn set_language(code: &str) -> PyResult<()> {
    let lang = facetsetfem_core::Language::parse(code).ok_or_else(|| {
        to_py(
            facetsetfem_core::FacetSetFemError::new(
                facetsetfem_core::ErrorKind::Case,
                facetsetfem_core::tr!("Unknown language: {code}", "知らない言語です: {code}"),
            )
            .with_hint(facetsetfem_core::tr!(
                "Use \"en\" or \"ja\"",
                "\"en\" か \"ja\" を指定してください"
            )),
        )
    })?;
    facetsetfem_core::set_language(lang);
    Ok(())
}

/// いまの表示言語（`"en"` / `"ja"`）。
#[pyfunction]
pub(crate) fn language() -> &'static str {
    facetsetfem_core::language().code()
}

/// 指定できる計算バックエンド（既定が先頭）。別名（`"gpu"` など）も解釈はできる。
#[pyfunction]
pub(crate) fn backend_choices() -> Vec<&'static str> {
    Backend::CHOICES.to_vec()
}

/// 指定できる前処理（既定が先頭）。別名（`"diagonal"` など）も解釈はできる。
#[pyfunction]
pub(crate) fn preconditioner_choices() -> Vec<&'static str> {
    Preconditioner::CHOICES.to_vec()
}

/// 電極の電流の配り方の選択肢（ケースファイルの `distribution`。SPEC §3.3）。
#[pyfunction]
pub(crate) fn distribution_choices() -> Vec<&'static str> {
    ElectrodeModel::CHOICES.to_vec()
}

/// バックエンド名を解釈して記録用の名前を返す。解釈できなければ `FacetSetFemError`。
///
/// 別名の表は Rust 側にしかないので、ケースファイルの検査もここを通す。
#[pyfunction]
pub(crate) fn parse_backend(name: &str) -> PyResult<&'static str> {
    Ok(Backend::parse(name).map_err(to_py)?.label())
}

/// 前処理の名前を解釈して記録用の名前を返す。解釈できなければ `FacetSetFemError`。
#[pyfunction]
pub(crate) fn parse_preconditioner(name: &str) -> PyResult<&'static str> {
    Ok(Preconditioner::parse(name).map_err(to_py)?.label())
}

/// VTU に書く五面体（CPENTA）の節点の並びの選択肢（既定が先頭。SPEC §5.5）。
///
/// `"vtk9.7"` は ParaView 6.2 以降（VTK 9.7 以降）、`"vtk9.6"` は ParaView 6.1 以前向け。
#[pyfunction]
pub(crate) fn wedge_order_choices() -> Vec<&'static str> {
    WedgeOrder::CHOICES.to_vec()
}

/// 五面体の並びの名前を解釈して記録用の名前を返す。解釈できなければ `FacetSetFemError`。
///
/// ケースファイルの `output.vtu_wedge_order` の検査もここを通す（SPEC §3.3）。
#[pyfunction]
pub(crate) fn parse_wedge_order(name: &str) -> PyResult<&'static str> {
    Ok(WedgeOrder::parse(name).map_err(to_py)?.label())
}

/// このビルドに CUDA バックエンドが入っているか（SPEC §8 M5）。
#[pyfunction]
pub(crate) fn cuda_available() -> bool {
    cuda::built_with_cuda()
}

/// 見えている GPU の一覧。
///
/// 各要素は `{"ordinal", "name", "compute_capability", "total_memory", "free_memory",
/// "usable"}`。`usable` は倍精度が使えるか（Compute Capability 6.0 以上）。
#[pyfunction]
pub(crate) fn cuda_devices(py: Python<'_>) -> PyResult<Py<PyList>> {
    let out = PyList::empty(py);
    for (ordinal, name, major, minor, total, free) in cuda::device_list() {
        let d = PyDict::new(py);
        d.set_item("ordinal", ordinal)?;
        d.set_item("name", name)?;
        d.set_item("compute_capability", (major, minor))?;
        d.set_item("total_memory", total)?;
        d.set_item("free_memory", free)?;
        d.set_item("usable", (major, minor) >= (6, 0))?;
        out.append(d)?;
    }
    Ok(out.unbind())
}

/// 節点数 `n_nodes` の問題を GPU で解けるか。
///
/// 解けるなら `None`、駄目ならその理由（日本語）を返す。AMG は階層を VRAM に置くので、
/// 前処理によって要る VRAM が違う（`preconditioner` は `"jacobi"` / `"amg"` / `"none"`）。
#[pyfunction]
#[pyo3(signature = (n_nodes, nnz = None, preconditioner = "jacobi"))]
pub(crate) fn cuda_check(
    n_nodes: usize,
    nnz: Option<usize>,
    preconditioner: &str,
) -> PyResult<Option<String>> {
    let preconditioner = Preconditioner::parse(preconditioner).map_err(to_py)?;
    Ok(cuda::check(n_nodes, nnz, preconditioner))
}
