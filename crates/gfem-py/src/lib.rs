//! GenericFEM の Python バインディング（SPEC §6.5）。
//!
//! ここには**ロジックを置かない**。`gfem-core` の型を Python から触れるように
//! 包むだけにして、GUI もバッチスクリプトも同じコアを叩く形にする（SPEC §7）。
//!
//! ```python
//! import gfem
//! mesh = gfem.read_nastran("model.bdf", length_unit="mm")
//! case = gfem.Case(mesh)
//! case.set_conductivity({1: 0.10, 2: 0.04, 3: 0.35})
//! case.add_electrode("E1", role="anode",   current=2e-3, shell_pids=[101])
//! case.add_electrode("E2", role="cathode", current=2e-3, shell_pids=[201])
//! result = case.solve()
//! print(result.impedance)
//! result.write_vtu("out.vtu")
//! ```

use gfem_core::case::{Electrode, ElectrodeRole, Preconditioner, ReferencePotential};
use gfem_core::{Analysis, LengthUnit, VtuOptions};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::create_exception;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;

create_exception!(
    _gfem,
    GfemError,
    pyo3::exceptions::PyException,
    "GenericFEM が投げるエラー。原因・場所・対処が本文に含まれる"
);

create_exception!(
    _gfem,
    SolverCancelled,
    GfemError,
    "進捗コールバックから計算が中断された"
);

/// バウンディングボックスを Python 側に返すときの形。
type BoundingBox = ((f64, f64, f64), (f64, f64, f64));

/// シェル要素の三角形分割を返すときの形。`(三角形, PID, 要素 ID)`。
type ShellTriangles<'py> = (
    Bound<'py, PyArray2<u32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
);

/// `gfem_core::GfemError` を Python 例外に変換する。
///
/// 本文には「何が」「どこで」「どうすれば直るか」がそのまま入る（CLAUDE.md 設計原則 6）。
fn to_py(e: gfem_core::GfemError) -> PyErr {
    if e.kind() == gfem_core::ErrorKind::Cancelled {
        SolverCancelled::new_err(e.to_string())
    } else {
        GfemError::new_err(e.to_string())
    }
}

fn array1(py: Python<'_>, v: Vec<f64>) -> Bound<'_, PyArray1<f64>> {
    v.into_pyarray(py)
}

fn array2(py: Python<'_>, v: Vec<f64>, cols: usize) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let rows = v.len() / cols;
    v.into_pyarray(py).reshape([rows, cols])
}

// --- 読み込みレポート -------------------------------------------------------

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

// --- メッシュ ---------------------------------------------------------------

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

    /// 節点座標 `(n_nodes, 3)` [m]。
    fn points<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        array2(py, self.inner.nodes.coords().to_vec(), 3)
    }

    /// ソリッド要素ごとの PID `(n_solid_elements,)`。要素値配列と同じ並び。
    fn solid_pid_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        let v: Vec<i32> = self
            .inner
            .solids
            .iter()
            .flat_map(|b| b.pids().iter().map(|&p| p as i32))
            .collect();
        v.into_pyarray(py)
    }

    /// 外表面を三角形に分割したもの `(n_tri, 3)`（節点索引）。
    ///
    /// 1000 万節点でも表面節点は数十万なので、GUI にはこれだけを渡す（SPEC §6.3）。
    fn surface_triangles<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let surface = gfem_core::mesh::extract_surface(&self.inner).map_err(to_py)?;
        let (tris, _) = surface.triangulate();
        let flat: Vec<u32> = tris.into_iter().flatten().collect();
        let rows = flat.len() / 3;
        flat.into_pyarray(py).reshape([rows, 3])
    }

    /// 外表面の三角形ごとの、元ソリッド要素の通し番号 `(n_tri,)`。
    ///
    /// 要素値（|E| など）を表面に貼り付けるときの索引に使う。
    fn surface_triangle_elements<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let surface = gfem_core::mesh::extract_surface(&self.inner).map_err(to_py)?;
        let (_, owners) = surface.triangulate();
        Ok(owners.into_pyarray(py))
    }

    /// シェル要素（電極候補面）を三角形に分割したもの。
    ///
    /// 戻り値は `(三角形 (n_tri, 3), PID (n_tri,), 要素 ID (n_tri,))`。
    /// 四辺形シェルは 2 枚の三角形に割る。電極面の色分け表示に使う（SPEC §6.2 の手順 3）。
    fn shell_triangles<'py>(&self, py: Python<'py>) -> PyResult<ShellTriangles<'py>> {
        let mut flat: Vec<u32> = Vec::new();
        let mut pids: Vec<i32> = Vec::new();
        let mut ids: Vec<i64> = Vec::new();
        for e in self.inner.iter_shells() {
            let c = e.conn;
            let mut push = |a: u32, b: u32, d: u32| {
                flat.extend_from_slice(&[a, b, d]);
                pids.push(e.pid as i32);
                ids.push(e.id as i64);
            };
            push(c[0], c[1], c[2]);
            if c.len() == 4 {
                push(c[0], c[2], c[3]);
            }
        }
        let rows = flat.len() / 3;
        Ok((
            flat.into_pyarray(py).reshape([rows, 3])?,
            pids.into_pyarray(py),
            ids.into_pyarray(py),
        ))
    }

    /// 四面体の接続 `(n_tet, 4)`（節点索引）。
    ///
    /// SPEC §6.3 の「小規模モデル用エスケープハッチ」。節点数が少ないうちは
    /// これで PyVista の `UnstructuredGrid` を組み立て、断面スライスなど
    /// VTK の全機能を使う。大規模モデルでは表示用データの抽出を Rust 側で行う。
    fn tetrahedra<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let others: Vec<&str> = self
            .inner
            .solids
            .iter()
            .filter(|b| b.kind() != gfem_core::ElementKind::Tet4)
            .map(|b| b.kind().card_name())
            .collect();
        if !others.is_empty() {
            return Err(GfemError::new_err(format!(
                "[未対応] 四面体以外のソリッド要素があります: {}
                   対処: tetrahedra() は CTETRA だけを返します。                 CHEXA / CPENTA を含むモデルの表示は M4 で対応します",
                others.join(", ")
            )));
        }
        let flat: Vec<u32> = self
            .inner
            .solids
            .iter()
            .flat_map(|b| b.conn().iter().copied())
            .collect();
        let rows = flat.len() / 4;
        flat.into_pyarray(py).reshape([rows, 4])
    }

    /// 外表面の総面積 [m²]。
    fn surface_area(&self) -> PyResult<f64> {
        let surface = gfem_core::mesh::extract_surface(&self.inner).map_err(to_py)?;
        Ok(surface.total_area(&self.inner))
    }

    /// VTU（ParaView で開ける形式）としてメッシュだけを書き出す。
    #[pyo3(signature = (path, compress = true, level = 6))]
    fn write_vtu(&self, py: Python<'_>, path: PathBuf, compress: bool, level: u32) -> PyResult<()> {
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

// --- 解析ケース -------------------------------------------------------------

/// 解析ケース（SPEC §3.3）。
#[pyclass(name = "Case", module = "gfem._gfem")]
pub struct PyCase {
    mesh: Py<PyMesh>,
    inner: gfem_core::Case,
}

#[pymethods]
impl PyCase {
    #[new]
    fn new(mesh: Py<PyMesh>) -> Self {
        Self {
            mesh,
            inner: gfem_core::Case::new(),
        }
    }

    /// 対象メッシュ。
    #[getter]
    fn mesh(&self, py: Python<'_>) -> Py<PyMesh> {
        self.mesh.clone_ref(py)
    }

    /// PID に導電率 σ [S/m] を割り当てる。
    ///
    /// `dict` を渡せばまとめて設定できる: `case.set_conductivity({1: 0.10, 2: 0.04})`
    #[pyo3(signature = (pid_or_map, sigma = None))]
    fn set_conductivity(
        &mut self,
        pid_or_map: &Bound<'_, PyAny>,
        sigma: Option<f64>,
    ) -> PyResult<()> {
        if let Ok(map) = pid_or_map.extract::<BTreeMap<u32, f64>>() {
            for (pid, s) in map {
                self.inner.set_conductivity(pid, s).map_err(to_py)?;
            }
            return Ok(());
        }
        let pid: u32 = pid_or_map.extract()?;
        let Some(s) = sigma else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "set_conductivity(pid, sigma) の形で σ [S/m] を指定してください",
            ));
        };
        self.inner.set_conductivity(pid, s).map_err(to_py)
    }

    /// 設定済みの導電率。
    #[getter]
    fn conductivity(&self) -> BTreeMap<u32, f64> {
        self.inner
            .conductivity
            .iter()
            .map(|(&pid, s)| (pid, s.scalar()))
            .collect()
    }

    /// 電極を追加する。
    ///
    /// `role` は `"anode"`（流入）か `"cathode"`（流出）。`current` は総電流 [A]（正の値）。
    /// 面はシェル PID の集合と個別の要素 ID の組み合わせで指定する（SPEC §3.3）。
    #[pyo3(signature = (name, role, current, shell_pids = None, element_ids = None))]
    fn add_electrode(
        &mut self,
        name: &str,
        role: &str,
        current: f64,
        shell_pids: Option<Vec<u32>>,
        element_ids: Option<Vec<u32>>,
    ) -> PyResult<()> {
        let electrode = Electrode {
            name: name.to_string(),
            role: parse_role(role)?,
            current,
            shell_pids: shell_pids.unwrap_or_default(),
            element_ids: element_ids.unwrap_or_default(),
        };
        self.inner.add_electrode(electrode);
        Ok(())
    }

    /// 定義済みの電極をすべて消す。
    fn clear_electrodes(&mut self) {
        self.inner.electrodes.clear();
    }

    /// ソルバ設定を変更する。
    #[pyo3(signature = (tolerance = None, max_iterations = None, preconditioner = None))]
    fn set_solver(
        &mut self,
        tolerance: Option<f64>,
        max_iterations: Option<usize>,
        preconditioner: Option<&str>,
    ) -> PyResult<()> {
        if let Some(t) = tolerance {
            self.inner.solver.tolerance = t;
        }
        if let Some(n) = max_iterations {
            self.inner.solver.max_iterations = n;
        }
        if let Some(p) = preconditioner {
            self.inner.solver.preconditioner = match p.to_ascii_lowercase().as_str() {
                "jacobi" => Preconditioner::Jacobi,
                "none" => Preconditioner::None,
                other => {
                    return Err(GfemError::new_err(format!(
                        "未知の前処理です: \"{other}\"（jacobi / none）"
                    )));
                }
            };
        }
        Ok(())
    }

    /// 現在のソルバ設定。
    #[getter]
    fn solver(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("tolerance", self.inner.solver.tolerance)?;
        d.set_item("max_iterations", self.inner.solver.max_iterations)?;
        d.set_item(
            "preconditioner",
            match self.inner.solver.preconditioner {
                Preconditioner::Jacobi => "jacobi",
                _ => "none",
            },
        )?;
        Ok(d.unbind())
    }

    /// 結果表示に使う電位の基準を設定する。
    ///
    /// `"zero_mean"` / `"cathode_mean"` / `"node:<節点 ID>"`（SPEC §3.3）。
    fn set_reference_potential(&mut self, reference: &str) -> PyResult<()> {
        self.inner.reference = parse_reference(reference)?;
        Ok(())
    }

    /// 実行前の整合性チェック（SPEC §6.2 の手順 4）。
    ///
    /// 不備があれば例外、問題なければ警告の一覧を返す。
    fn validate(&self) -> PyResult<Vec<String>> {
        let mesh = &self.mesh.get().inner;
        let surface = gfem_core::mesh::extract_surface(mesh).map_err(to_py)?;
        self.inner.validate(mesh, &surface).map_err(to_py)
    }

    /// 電極ごとの面積と電流密度。面を選びながら確認するために使う（SPEC §6.2 の手順 3）。
    fn electrode_info(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let mesh = &self.mesh.get().inner;
        let (_, geometries) =
            gfem_core::assemble::assemble_load(mesh, &self.inner).map_err(to_py)?;
        let list = PyList::empty(py);
        for g in geometries {
            let d = PyDict::new(py);
            d.set_item("name", g.name)?;
            d.set_item("role", g.role.label())?;
            d.set_item("current", g.signed_current)?;
            d.set_item("n_faces", g.n_faces)?;
            d.set_item("area", g.area)?;
            d.set_item("current_density", g.current_density)?;
            list.append(d)?;
        }
        Ok(list.unbind())
    }

    /// 解く（SPEC §4.1）。
    ///
    /// `progress` に `f(iteration, relative_residual) -> bool` を渡すと反復ごとに呼ばれ、
    /// `False` を返すと計算を中断する（`SolverCancelled` 例外）。
    #[pyo3(signature = (progress = None))]
    fn solve(&self, py: Python<'_>, progress: Option<Py<PyAny>>) -> PyResult<PySolution> {
        let mesh = &self.mesh.get().inner;
        let case = &self.inner;
        let callback_error: Mutex<Option<PyErr>> = Mutex::new(None);

        // 1000 万節点でも GUI が固まらないよう GIL を離す（SPEC §6.4）
        let analysis = py.detach(|| {
            let mut cb = |iteration: usize, residual: f64| -> bool {
                let Some(obj) = progress.as_ref() else {
                    return true;
                };
                Python::attach(|py| match obj.call1(py, (iteration, residual)) {
                    Ok(v) => v.is_truthy(py).unwrap_or(true),
                    Err(e) => {
                        *callback_error.lock().expect("進捗コールバックのエラー保持") = Some(e);
                        false
                    }
                })
            };
            gfem_core::analysis::solve(mesh, case, Some(&mut cb))
        });

        if let Some(e) = callback_error
            .into_inner()
            .expect("進捗コールバックのエラー保持")
        {
            return Err(e);
        }
        let analysis = analysis.map_err(to_py)?;
        Ok(PySolution {
            mesh: self.mesh.clone_ref(py),
            inner: analysis,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "<Case regions={} electrodes={}>",
            self.inner.conductivity.len(),
            self.inner.electrodes.len()
        )
    }
}

// --- 解析結果 ---------------------------------------------------------------

/// 解析結果（SPEC §5）。
#[pyclass(name = "Solution", module = "gfem._gfem", frozen)]
pub struct PySolution {
    mesh: Py<PyMesh>,
    inner: Analysis,
}

#[pymethods]
impl PySolution {
    /// 対象メッシュ。
    #[getter]
    fn mesh(&self, py: Python<'_>) -> Py<PyMesh> {
        self.mesh.clone_ref(py)
    }

    /// 節点電位 φ [V]、`(n_nodes,)`。
    #[getter]
    fn potential<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array1(py, self.inner.potential.clone())
    }

    /// 電界 E [V/m]、`(n_elements, 3)`。Tet4 では要素内で一定。
    #[getter]
    fn electric_field<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        array2(py, self.inner.fields.electric_field.clone(), 3)
    }

    /// 電流密度 J [A/m²]、`(n_elements, 3)`。
    #[getter]
    fn current_density<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        array2(py, self.inner.fields.current_density.clone(), 3)
    }

    /// |E| [V/m]、`(n_elements,)`。**最大値の評価にはこの要素値を使うこと**。
    #[getter]
    fn e_magnitude<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array1(py, self.inner.fields.e_magnitude.clone())
    }

    /// |J| [A/m²]、`(n_elements,)`。
    #[getter]
    fn j_magnitude<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array1(py, self.inner.fields.j_magnitude.clone())
    }

    /// 電力密度 σ|E|² [W/m³]、`(n_elements,)`。
    #[getter]
    fn power_density<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array1(py, self.inner.fields.power_density.clone())
    }

    /// 要素体積 [m³]、`(n_elements,)`。
    #[getter]
    fn element_volume<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array1(py, self.inner.fields.volume.clone())
    }

    /// 要素値を体積重み平均で節点に外挿した値（表示用）。
    ///
    /// 平滑化されているので**最大値の評価には使わないこと**（SPEC §5.1）。
    fn nodal_average<'py>(
        &self,
        py: Python<'py>,
        quantity: &str,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let f = &self.inner.fields;
        let values = match quantity {
            "e_magnitude" => &f.e_magnitude,
            "j_magnitude" => &f.j_magnitude,
            "power_density" => &f.power_density,
            other => {
                return Err(GfemError::new_err(format!(
                    "未知の量です: \"{other}\"（e_magnitude / j_magnitude / power_density）"
                )));
            }
        };
        let mesh = &self.mesh.get().inner;
        let nodal = gfem_core::post::nodal_average(mesh, values, &f.volume, 1);
        Ok(array1(py, nodal))
    }

    /// 電極間の等価抵抗 R = ΔV / I [Ω]（SPEC §5.3）。
    #[getter]
    fn impedance(&self) -> f64 {
        self.inner.conservation.impedance
    }

    /// 電極間電位差 ΔV [V]。
    #[getter]
    fn voltage(&self) -> f64 {
        self.inner.conservation.voltage
    }

    /// 総注入電流 [A]。
    #[getter]
    fn total_current(&self) -> f64 {
        self.inner.conservation.total_current
    }

    /// 消費電力 ∫σ|E|² dΩ [W]。
    #[getter]
    fn power(&self) -> f64 {
        self.inner.conservation.power_from_field
    }

    /// 電極ごとの面積重み平均電位 [V]。
    #[getter]
    fn electrode_potentials(&self) -> Vec<(String, f64)> {
        self.inner.conservation.electrode_potentials.clone()
    }

    /// CG の反復回数。
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.solver.iterations
    }

    /// 最終的な相対残差。
    #[getter]
    fn residual(&self) -> f64 {
        self.inner.solver.relative_residual
    }

    /// 収束履歴（GUI の収束曲線用）。
    #[getter]
    fn history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array1(py, self.inner.solver.history.clone())
    }

    /// 使用したバックエンド名。
    #[getter]
    fn backend(&self) -> &'static str {
        self.inner.solver.backend
    }

    /// 保存則チェックの結果（SPEC §5.4）。
    #[getter]
    fn conservation(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let c = &self.inner.conservation;
        let d = PyDict::new(py);
        d.set_item("current_balance", c.current_balance)?;
        d.set_item("power_from_field", c.power_from_field)?;
        d.set_item("power_from_load", c.power_from_load)?;
        d.set_item("energy_relative_error", c.energy_relative_error)?;
        d.set_item("discrete_residual", c.discrete_residual)?;
        d.set_item("voltage", c.voltage)?;
        d.set_item("impedance", c.impedance)?;
        d.set_item("total_current", c.total_current)?;
        d.set_item("warnings", c.warnings.clone())?;
        Ok(d.unbind())
    }

    /// 各段階の所要時間 [秒]。
    #[getter]
    fn timings(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let t = &self.inner.timings;
        let d = PyDict::new(py);
        d.set_item("prepare", t.prepare)?;
        d.set_item("assemble", t.assemble)?;
        d.set_item("solve", t.solve)?;
        d.set_item("post", t.post)?;
        d.set_item("total", t.total)?;
        Ok(d.unbind())
    }

    /// 実行中に出た警告。
    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.inner.warnings.clone()
    }

    /// 結果を VTU として書き出す（節点値と要素値の両方を含む）。
    #[pyo3(signature = (path, compress = true, level = 6))]
    fn write_vtu(&self, py: Python<'_>, path: PathBuf, compress: bool, level: u32) -> PyResult<()> {
        let mesh = &self.mesh.get().inner;
        py.detach(|| {
            gfem_core::write_analysis_vtu(&path, mesh, &self.inner, &VtuOptions { compress, level })
        })
        .map_err(to_py)
    }

    fn __repr__(&self) -> String {
        format!(
            "<Solution iterations={} residual={:.2e} R={:.4} ohm>",
            self.inner.solver.iterations,
            self.inner.solver.relative_residual,
            self.inner.conservation.impedance
        )
    }
}

// --- モジュール関数 ---------------------------------------------------------

fn parse_role(role: &str) -> PyResult<ElectrodeRole> {
    match role.to_ascii_lowercase().as_str() {
        "anode" => Ok(ElectrodeRole::Anode),
        "cathode" => Ok(ElectrodeRole::Cathode),
        other => Err(GfemError::new_err(format!(
            "未知の電極の役割です: \"{other}\"（anode / cathode）"
        ))),
    }
}

fn parse_reference(reference: &str) -> PyResult<ReferencePotential> {
    let r = reference.trim();
    if let Some(rest) = r.strip_prefix("node:") {
        let id: u32 = rest
            .trim()
            .parse()
            .map_err(|_| GfemError::new_err(format!("節点 ID として解釈できません: \"{rest}\"")))?;
        return Ok(ReferencePotential::Node(id));
    }
    match r.to_ascii_lowercase().as_str() {
        "zero_mean" => Ok(ReferencePotential::ZeroMean),
        "cathode_mean" => Ok(ReferencePotential::CathodeMean),
        other => Err(GfemError::new_err(format!(
            "未知の基準電位です: \"{other}\"（zero_mean / cathode_mean / node:<ID>）"
        ))),
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
    m.add("SolverCancelled", m.py().get_type::<SolverCancelled>())?;
    m.add_class::<PyMesh>()?;
    m.add_class::<PyReadReport>()?;
    m.add_class::<PyCase>()?;
    m.add_class::<PySolution>()?;
    m.add_function(wrap_pyfunction!(read_nastran, m)?)?;
    m.add_function(wrap_pyfunction!(length_units, m)?)?;
    Ok(())
}
