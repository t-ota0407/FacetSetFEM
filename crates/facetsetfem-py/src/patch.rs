//! 体表面パッチの辞書の解釈（SPEC §11.1）。
//!
//! ケースファイル・`Case.add_electrode`・`Mesh.pick_patch`・`check_patch` が**同じ解釈**を通る。
//! 形の解釈をここ 1 か所に置き、値の範囲はコアの `SurfacePatch::check` が見る（SPEC §6.5）。

use crate::FacetSetFemError;
use crate::convert::to_py;
use facetsetfem_core::case::{PatchShape, RectShape, SurfacePatch};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// 体表面パッチの指定（辞書）を解釈する。
///
/// エラーは「どの電極の何番目のパッチが」「何が足りないか」まで出す
/// （CLAUDE.md 設計原則 6）。
pub(crate) fn parse_patch(
    obj: &Bound<'_, PyAny>,
    electrode: &str,
    index: usize,
) -> PyResult<SurfacePatch> {
    parse_patch_at(
        obj,
        &facetsetfem_core::tr!(
            "Patch {index} of electrode {electrode}",
            "電極 {electrode} のパッチ {index}"
        ),
    )
}

/// パッチの辞書で使えるキー。
pub(crate) const PATCH_KEYS: [&str; 9] = [
    "center",
    "shape",
    "radius",
    "width",
    "height",
    "corner_radius",
    "axis",
    "angle",
    "pid",
];

/// パッチの辞書を解釈する（ケースファイル・`Case.add_electrode`・`Mesh.pick_patch` の共通）。
///
/// 形は `shape` で選ぶ（省略すると円）。長さは m、角度は度。値の範囲（半径が負でない、
/// 角の半径が短い辺の半分まで、など）はコアの [`SurfacePatch::check`] が見る。
/// ここは「キーがそろっているか」「型が合っているか」だけを見る。
pub(crate) fn parse_patch_at(obj: &Bound<'_, PyAny>, where_: &str) -> PyResult<SurfacePatch> {
    let err = |msg: String| FacetSetFemError::new_err(msg);
    if let Ok(d) = obj.cast::<PyDict>() {
        for key in d.keys() {
            let key: String = key.extract().unwrap_or_default();
            if !PATCH_KEYS.contains(&key.as_str()) {
                return Err(err(facetsetfem_core::tr!(
                    "{where_}: unknown key {key:?}. Allowed keys are {}",
                    "{where_}に使えないキー {key:?} があります。使えるのは {} です",
                    PATCH_KEYS.join(" / ")
                )));
            }
        }
    }
    let get =
        |key: &str| -> Option<Bound<'_, PyAny>> { obj.get_item(key).ok().filter(|v| !v.is_none()) };
    let real = |key: &str, what: &str| -> PyResult<Option<f64>> {
        get(key)
            .map(|v| {
                v.extract::<f64>().map_err(|_| {
                    err(facetsetfem_core::tr!(
                        "{where_}: give {key} as a real number ({what})",
                        "{where_}の {key} は実数で指定してください（{what}）"
                    ))
                })
            })
            .transpose()
    };
    let vec3 = |key: &str, what: &str| -> PyResult<Option<[f64; 3]>> {
        let Some(v) = get(key) else { return Ok(None) };
        let v: Vec<f64> = v.extract().map_err(|_| {
            err(facetsetfem_core::tr!(
                "{where_}: give {key} as 3 real numbers [x, y, z] ({what})",
                "{where_}の {key} は 3 個の実数 [x, y, z] で指定してください（{what}）"
            ))
        })?;
        <[f64; 3]>::try_from(v.as_slice()).map(Some).map_err(|_| {
            err(facetsetfem_core::tr!(
                "{where_}: {key} has {} elements (give 3 as [x, y, z])",
                "{where_}の {key} の要素数が {} 個です（[x, y, z] の 3 個で指定してください）",
                v.len()
            ))
        })
    };

    let center = vec3("center", &facetsetfem_core::tr!("in m", "単位は m"))?.ok_or_else(|| {
        err(facetsetfem_core::tr!(
            "{where_}: center is missing. \
             Give it as {{\"center\": [x, y, z], \"radius\": r, \"pid\": 51}} (in m)",
            "{where_}に center がありません。\
             {{\"center\": [x, y, z], \"radius\": r, \"pid\": 51}} の形で指定してください（単位は m）"
        ))
    })?;
    let pid: Option<u32> = get("pid")
        .map(|v| {
            v.extract().map_err(|_| {
                err(facetsetfem_core::tr!(
                    "{where_}: give pid as an integer",
                    "{where_}の pid は整数で指定してください"
                ))
            })
        })
        .transpose()?;
    let shape: String = match get("shape") {
        Some(v) => v
            .extract::<String>()
            .map_err(|_| {
                err(facetsetfem_core::tr!(
                    "{where_}: give shape as \"circle\" or \"rect\"",
                    "{where_}の shape は \"circle\" か \"rect\" で指定してください"
                ))
            })?
            .trim()
            .to_ascii_lowercase(),
        None => "circle".to_string(),
    };
    let only = |allowed: &[&str], shape_name: &str| -> PyResult<()> {
        for key in [
            "radius",
            "width",
            "height",
            "corner_radius",
            "axis",
            "angle",
        ] {
            if !allowed.contains(&key) && get(key).is_some() {
                return Err(err(facetsetfem_core::tr!(
                    "{where_} is {shape_name}, so {key} cannot be used (allowed: {})",
                    "{where_}は {shape_name} なので {key} は使えません（使えるのは {}）",
                    allowed.join(" / ")
                )));
            }
        }
        Ok(())
    };
    match shape.as_str() {
        "circle" => {
            only(
                &["radius"],
                &facetsetfem_core::tr!("a circle (shape: circle)", "円（shape: circle）"),
            )?;
            let radius =
                real("radius", &facetsetfem_core::tr!("in m", "単位は m"))?.ok_or_else(|| {
                    err(facetsetfem_core::tr!(
                        "{where_}: radius is missing (in m)",
                        "{where_}に radius がありません（単位は m）"
                    ))
                })?;
            Ok(SurfacePatch::circle(center, radius, pid))
        }
        "rect" => {
            only(
                &["width", "height", "corner_radius", "axis", "angle"],
                &facetsetfem_core::tr!("a rectangle (shape: rect)", "矩形（shape: rect）"),
            )?;
            let need = |key: &str, what: &str| -> PyResult<f64> {
                real(key, what)?.ok_or_else(|| {
                    err(facetsetfem_core::tr!(
                        "{where_}: {key} is missing (a rectangle needs width and height; {what})",
                        "{where_}に {key} がありません（矩形は width と height が必要です。{what}）"
                    ))
                })
            };
            let rect = RectShape {
                width: need(
                    "width",
                    &facetsetfem_core::tr!("in m; 20 mm is 0.02", "単位は m。20 mm なら 0.02"),
                )?,
                height: need("height", &facetsetfem_core::tr!("in m", "単位は m"))?,
                corner_radius: real("corner_radius", &facetsetfem_core::tr!("in m", "単位は m"))?
                    .unwrap_or(0.0),
                axis: vec3(
                    "axis",
                    &facetsetfem_core::tr!("direction of the width edge", "幅の辺の向き"),
                )?
                .ok_or_else(|| {
                    err(facetsetfem_core::tr!(
                        "{where_}: axis is missing. For a rectangle, give the direction of the width \
                         edge as [x, y, z] (e.g. [1, 0, 0]; it is projected onto the body surface)",
                        "{where_}に axis がありません。矩形は幅の辺の向きを [x, y, z] で指定してください\
                         （例: [1, 0, 0]。体表面に沿う向きに投影して使います）"
                    ))
                })?,
                angle: real("angle", &facetsetfem_core::tr!("in degrees", "単位は度"))?.unwrap_or(0.0),
            };
            Ok(SurfacePatch::rect(center, rect, pid))
        }
        other => Err(err(facetsetfem_core::tr!(
            "{where_}: shape is {other:?}. Give \"circle\" or \"rect\"",
            "{where_}の shape が {other:?} です。\"circle\"（円）か \"rect\"（矩形）で指定してください"
        ))),
    }
}

/// パッチの形を辞書に書き出す（`pick_patch` の戻り値と `check_patch` が使う）。
pub(crate) fn describe_patch(d: &Bound<'_, PyDict>, patch: &SurfacePatch) -> PyResult<()> {
    d.set_item("center", patch.center.to_vec())?;
    d.set_item("pid", patch.pid)?;
    d.set_item("nominal_area", patch.nominal_area())?;
    match &patch.shape {
        PatchShape::Circle { radius } => {
            d.set_item("shape", "circle")?;
            d.set_item("radius", *radius)?;
        }
        PatchShape::Rect(r) => {
            d.set_item("shape", "rect")?;
            d.set_item("width", r.width)?;
            d.set_item("height", r.height)?;
            d.set_item("corner_radius", r.corner_radius)?;
            d.set_item("axis", r.axis.to_vec())?;
            d.set_item("angle", r.angle)?;
        }
    }
    Ok(())
}

/// 体表面パッチの辞書を検査し、正規化した辞書（`nominal_area` つき）を返す。
///
/// ケースファイルの読み込み（Python）もここを通す。形の解釈と値の範囲の検査を
/// Rust 側だけに置くため（SPEC §6.5「同じものを 2 か所に書かない」）。
/// 向きの軸が面の法線と平行かどうかはメッシュが無いと分からないので、solve 時に見る。
#[pyfunction]
pub(crate) fn check_patch(py: Python<'_>, patch: Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    let parsed = parse_patch_at(&patch, &facetsetfem_core::tr!("Patch", "パッチ"))?;
    parsed.check().map_err(to_py)?;
    let d = PyDict::new(py);
    describe_patch(&d, &parsed)?;
    Ok(d.unbind())
}
