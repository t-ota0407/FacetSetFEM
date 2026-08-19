//! GenericFEM の Python バインディング（SPEC §6.5）。
//!
//! ここには**ロジックを置かない**。`gfem-core` の型を Python から触れるように
//! 包むだけにして、GUI もバッチスクリプトも同じコアを叩く形にする（SPEC §7）。
//!
//! M0 の範囲では「Nastran を読む」「読み込みレポートを見る」「VTU を書く」まで。
//! 大規模データを numpy へ zero-copy で渡す口（SPEC §6.3）は、表示が必要になる
//! M1 で追加する。

use gfem_core::{LengthUnit, VtuOptions};
use pyo3::create_exception;
use pyo3::prelude::*;
use std::path::PathBuf;

/// バウンディングボックスを Python 側に返すときの形。`((xmin, ymin, zmin), (xmax, ymax, zmax))`。
type BoundingBox = ((f64, f64, f64), (f64, f64, f64));

create_exception!(
    _gfem,
    GfemError,
    pyo3::exceptions::PyException,
    "GenericFEM が投げるエラー。原因・場所・対処が本文に含まれる"
);

/// `gfem_core::GfemError` を Python 例外に変換する。
///
/// 本文には「何が」「どこで」「どうすれば直るか」がそのまま入る（CLAUDE.md 設計原則 6）。
fn to_py(e: gfem_core::GfemError) -> PyErr {
    GfemError::new_err(e.to_string())
}

/// 読み込みレポート（SPEC §6.2）。
#[pyclass(name = "ReadReport", module = "gfem._gfem", frozen)]
pub struct PyReadReport {
    inner: gfem_core::ReadReport,
}

#[pymethods]
impl PyReadReport {
    /// 読み込んだファイルのパス。
    #[getter]
    fn file(&self) -> String {
        self.inner.file.display().to_string()
    }

    /// 読んだ物理行数。
    #[getter]
    fn lines(&self) -> usize {
        self.inner.lines
    }

    /// 認識したカードごとの件数。
    #[getter]
    fn card_counts(&self) -> Vec<(String, usize)> {
        self.inner.card_counts.clone()
    }

    /// 未対応だったカードの集計。`(カード名, 件数, 最初の行番号)` の並び。
    #[getter]
    fn unsupported(&self) -> Vec<(String, usize, usize)> {
        self.inner
            .unsupported
            .iter()
            .map(|c| (c.name.clone(), c.count, c.first_line))
            .collect()
    }

    /// 警告の一覧。
    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.inner.warnings.clone()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "<ReadReport file={:?} lines={} warnings={}>",
            self.inner.file.display(),
            self.inner.lines,
            self.inner.warnings.len()
        )
    }
}

/// 有限要素メッシュ。座標は SI（m）に正規化済み。
#[pyclass(name = "Mesh", module = "gfem._gfem", frozen)]
pub struct PyMesh {
    inner: gfem_core::Mesh,
    report: gfem_core::ReadReport,
}

#[pymethods]
impl PyMesh {
    /// 節点数。
    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.nodes.len()
    }

    /// ソリッド要素数。
    #[getter]
    fn n_solid_elements(&self) -> usize {
        self.inner.n_solid_elements()
    }

    /// シェル要素数（電極候補面）。
    #[getter]
    fn n_shell_elements(&self) -> usize {
        self.inner.n_shell_elements()
    }

    /// 全要素数。
    #[getter]
    fn n_elements(&self) -> usize {
        self.inner.n_elements()
    }

    /// 読み込み時に指定した長さ単位。座標はすでに m に換算されている。
    #[getter]
    fn length_unit(&self) -> &'static str {
        self.inner.length_unit.symbol()
    }

    /// 読み込み元ファイル。
    #[getter]
    fn source(&self) -> Option<String> {
        self.inner.source.as_ref().map(|p| p.display().to_string())
    }

    /// 読み込みレポート。
    #[getter]
    fn report(&self) -> PyReadReport {
        PyReadReport {
            inner: self.report.clone(),
        }
    }

    /// バウンディングボックス `((xmin, ymin, zmin), (xmax, ymax, zmax))`。単位は **m**。
    ///
    /// 幾何量なので Python 側ではタプル（不変）で返す。
    fn bounding_box(&self) -> Option<BoundingBox> {
        self.inner
            .bounding_box()
            .map(|(lo, hi)| ((lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])))
    }

    /// ソリッド要素が使っている PID の一覧（＝領域の一覧、昇順）。
    fn solid_pids(&self) -> Vec<u32> {
        self.inner.solid_pids()
    }

    /// シェル要素が使っている PID の一覧（＝電極候補面の一覧、昇順）。
    fn shell_pids(&self) -> Vec<u32> {
        self.inner.shell_pids()
    }

    /// 要素種ごとの要素数。`[("CTETRA", 4032), ("CTRIA3", 48)]` のような並び。
    fn element_counts(&self) -> Vec<(&'static str, usize)> {
        self.inner
            .blocks()
            .map(|b| (b.kind().card_name(), b.len()))
            .collect()
    }

    /// VTU（ParaView で開ける形式）として書き出す。
    #[pyo3(signature = (path, compress = true, level = 6))]
    fn write_vtu(&self, py: Python<'_>, path: PathBuf, compress: bool, level: u32) -> PyResult<()> {
        // 書き出しは重いので GIL を離す（SPEC §6.4）
        py.detach(|| gfem_core::write_mesh_vtu(&path, &self.inner, &VtuOptions { compress, level }))
            .map_err(to_py)
    }

    fn __repr__(&self) -> String {
        format!(
            "<Mesh nodes={} solids={} shells={} unit={}>",
            self.inner.nodes.len(),
            self.inner.n_solid_elements(),
            self.inner.n_shell_elements(),
            self.inner.length_unit.symbol()
        )
    }
}

/// Nastran バルクデータ（.bdf / .nas）を読み込む。
///
/// `length_unit` はファイル中の座標値の単位（`"mm"` / `"cm"` / `"m"` / `"inch"`）。
/// Nastran ファイルには単位情報が無いため必ず指定する（SPEC §3.2）。
/// 読み込んだ座標は内部で SI（m）に換算される。
#[pyfunction]
#[pyo3(signature = (path, length_unit = "mm"))]
fn read_nastran(py: Python<'_>, path: PathBuf, length_unit: &str) -> PyResult<PyMesh> {
    let unit: LengthUnit = LengthUnit::parse(length_unit).map_err(to_py)?;
    // 1000 万節点の読み込みで GUI を固まらせないため GIL を離す（SPEC §6.4）
    let (mesh, report) = py
        .detach(|| gfem_core::read_nastran(&path, unit))
        .map_err(to_py)?;
    Ok(PyMesh {
        inner: mesh,
        report,
    })
}

/// 対応する長さ単位の一覧。
#[pyfunction]
fn length_units() -> Vec<&'static str> {
    LengthUnit::ALL.iter().map(|u| u.symbol()).collect()
}

#[pymodule]
fn _gfem(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", gfem_core::VERSION)?;
    m.add("GfemError", m.py().get_type::<GfemError>())?;
    m.add_class::<PyMesh>()?;
    m.add_class::<PyReadReport>()?;
    m.add_function(wrap_pyfunction!(read_nastran, m)?)?;
    m.add_function(wrap_pyfunction!(length_units, m)?)?;
    Ok(())
}
