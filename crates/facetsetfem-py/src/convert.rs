//! Rust と Python のあいだの受け渡し（エラーの変換・numpy 配列・遅延初期化）。
//!
//! `Mesh` / `Case` / `Solution` のどれからも使う小さな道具だけを置く。

use crate::{FacetSetFemError, SolverCancelled};
use numpy::ndarray::{ArrayView1, ArrayView2, Dimension};
use numpy::{IntoPyArray, PyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use std::sync::OnceLock;

/// `facetsetfem_core::FacetSetFemError` を Python 例外に変換する。
///
/// 本文には「何が」「どこで」「どうすれば直るか」がそのまま入る（CLAUDE.md 設計原則 6）。
pub(crate) fn to_py(e: facetsetfem_core::FacetSetFemError) -> PyErr {
    if e.kind() == facetsetfem_core::ErrorKind::Cancelled {
        SolverCancelled::new_err(e.to_string())
    } else {
        FacetSetFemError::new_err(e.to_string())
    }
}

pub(crate) fn array1(py: Python<'_>, v: Vec<f64>) -> Bound<'_, PyArray1<f64>> {
    v.into_pyarray(py)
}

/// `data` を、それを持つ Python オブジェクト `owner` を親にした**読み取り専用**の numpy 配列
/// として渡す。**複製しない**（SPEC §6.3 の zero-copy）。
///
/// 安全である理由: numpy 配列は `owner` を親として参照し続けるので、配列が生きている間
/// `owner` は解放されない。`owner` は `frozen` の pyclass（`Solution` / `Mesh`）で、`data` を
/// 書き換えたり作り直したりする手段が無いので、見せているメモリが動くことも変わることもない。
///
/// Python から書き換えると Rust 側の計算（領域統計・VTU の書き出しなど）が狂うので、
/// 書き込みは禁じる。親が numpy 配列ではないので、Python 側から書き込み可に戻すこともできない。
/// ただし pyvista（VTK）はこの印を見ずにメモリを共有するので、その先からは書き換えられる。
/// だから座標のように壊れると以後の解析すべてが狂うもの（`Mesh.points`）には使わない。
/// 配列を持っている間は `owner` 全体が解放されない（独立に持ちたければ `.copy()`）。
pub(crate) fn view1<'py>(owner: &Bound<'py, PyAny>, data: &[f64]) -> Bound<'py, PyArray1<f64>> {
    // SAFETY: 上の説明のとおり、`data` は `owner` が生きている間動かず、変わらない
    let array = unsafe { PyArray1::borrow_from_array(&ArrayView1::from(data), owner.clone()) };
    read_only(array)
}

/// [`view1`] の 2 次元版。`data` を `cols` 列ずつに区切った `(len / cols, cols)` の配列。
pub(crate) fn view2<'py>(
    owner: &Bound<'py, PyAny>,
    data: &[f64],
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let view = ArrayView2::from_shape((data.len() / cols, cols), data).map_err(|e| {
        to_py(facetsetfem_core::FacetSetFemError::internal(
            facetsetfem_core::tr!(
                "Cannot split the array into {cols} columns ({} elements): {e}",
                "配列を {cols} 列に区切れません（{} 要素）: {e}",
                data.len()
            ),
        ))
    })?;
    // SAFETY: view1 と同じ
    let array = unsafe { PyArray2::borrow_from_array(&view, owner.clone()) };
    Ok(read_only(array))
}

/// numpy 配列の書き込み可の印を外す。
pub(crate) fn read_only<D: Dimension>(
    array: Bound<'_, PyArray<f64, D>>,
) -> Bound<'_, PyArray<f64, D>> {
    let readonly = array.readwrite().make_nonwriteable();
    (*readonly).clone()
}

/// `cell` が空なら `build` で作って入れ、中身を返す（`OnceLock::get_or_try_init` の代わり）。
///
/// 作るのに失敗したら何も入れずにエラーを返すので、次の呼び出しでまた作り直す。
/// 2 つのスレッドが同時に作り始めたら、先に入れたほうが残る（どちらも同じものを作る）。
pub(crate) fn lazy<T, E>(
    cell: &OnceLock<T>,
    build: impl FnOnce() -> Result<T, E>,
) -> Result<&T, E> {
    if let Some(value) = cell.get() {
        return Ok(value);
    }
    let built = build()?;
    Ok(cell.get_or_init(|| built))
}
