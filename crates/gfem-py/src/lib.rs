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

mod cuda;

use gfem_core::case::{Backend, Electrode, ElectrodeRole, Preconditioner, ReferencePotential};
use gfem_core::mesh::streamline::{TraceDirection, TraceOptions};
use gfem_core::post::RegionStats;
use gfem_core::provenance::Provenance;
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
/// 等値面を Python 側に返すときの形。`(頂点, 三角形, 頂点のスカラー)`。
type IsoSurfaceArrays<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<u32>>,
    Bound<'py, PyArray1<f64>>,
);

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

    /// ソリッド要素を VTK のセル配列として返す `(cells, celltypes)`。
    ///
    /// SPEC §6.3 の「小規模モデル用エスケープハッチ」。節点数が少ないうちは
    /// これで PyVista の `UnstructuredGrid` を組み立て、断面スライスなど
    /// VTK の全機能を使う。大規模モデルでは表示用データの抽出を Rust 側で行う。
    ///
    /// `cells` は VTK の並び `[節点数, 節点索引..., 節点数, ...]`、
    /// `celltypes` は要素ごとの VTK セル型。CTETRA / CHEXA / CPENTA はそれぞれ
    /// VTK_TETRA / VTK_HEXAHEDRON / VTK_WEDGE に対応するので、
    /// 四面体へ切り分ける必要はない（M4 で追加）。節点の並び替えは
    /// `ElementKind::vtk_node_order`（VTU 出力と共通）。
    ///
    /// 並びは要素値配列（`solid_pid_array` や結果の要素値）と同じ。
    fn solid_cells<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<u8>>) {
        let mut cells: Vec<i64> = Vec::new();
        let mut types: Vec<u8> = Vec::with_capacity(self.inner.n_solid_elements());
        for e in self.inner.iter_solids() {
            cells.push(e.conn.len() as i64);
            cells.extend(e.kind.vtk_node_order().iter().map(|&l| e.conn[l] as i64));
            types.push(e.kind.vtk_cell_type());
        }
        (cells.into_pyarray(py), types.into_pyarray(py))
    }

    /// 四面体の接続 `(n_tet, 4)`（節点索引）。
    ///
    /// CTETRA だけのモデル用。要素種が混ざっていてもよい表示経路は
    /// [`solid_cells`] を使う。
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
                "[未対応] 四面体以外のソリッド要素があります: {}\n\
                 対処: tetrahedra() は CTETRA だけを返します。\
                 要素種が混ざったモデルは solid_cells() を使ってください",
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

    /// ソリッド要素の重心 `(n_solid_elements, 3)` [m]。
    ///
    /// 要素値（E, J など）は要素内で一定なので、解析解と突き合わせるときは
    /// この重心で真値を評価すればよい（SPEC §8 M2 の検証で使う）。
    fn element_centroids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mut flat: Vec<f64> = Vec::with_capacity(self.inner.n_solid_elements() * 3);
        for e in self.inner.iter_solids() {
            let mut c = [0.0f64; 3];
            let scale = 1.0 / e.conn.len() as f64;
            for &node in e.conn {
                let p = self.inner.nodes.coord(node);
                for (d, cd) in c.iter_mut().enumerate() {
                    *cd += scale * p[d];
                }
            }
            flat.extend_from_slice(&c);
        }
        let rows = flat.len() / 3;
        flat.into_pyarray(py).reshape([rows, 3])
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

    /// PID に領域名を付ける（計算には使わないが、統計と表示に出る）。
    fn set_region_name(&mut self, pid: u32, name: &str) {
        self.inner.set_region_name(pid, name);
    }

    /// 設定済みの領域名。
    #[getter]
    fn region_names(&self) -> BTreeMap<u32, String> {
        self.inner.region_names.clone()
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
    ///
    /// `backend` は `"auto"` / `"cpu"` / `"cpu_parallel"`。
    /// **どれを選んでも結果はビット単位で同じ**で、速度だけが変わる（SPEC §4.4）。
    #[pyo3(signature = (
        tolerance = None,
        max_iterations = None,
        preconditioner = None,
        backend = None,
    ))]
    fn set_solver(
        &mut self,
        tolerance: Option<f64>,
        max_iterations: Option<usize>,
        preconditioner: Option<&str>,
        backend: Option<&str>,
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
        if let Some(b) = backend {
            self.inner.solver.backend = Backend::parse(b).map_err(to_py)?;
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
        d.set_item("backend", self.inner.solver.backend.label())?;
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
            // GPU を使うかはここで決める。gfem-core は CUDA を知らない（SPEC §7）
            let (solver, note) = cuda::pick(case.solver.backend, mesh.nodes.len(), None);
            let analysis =
                gfem_core::analysis::solve_with(mesh, case, Some(&mut cb), solver.as_deref());
            (analysis, note)
        });
        let (analysis, note) = analysis;

        if let Some(e) = callback_error
            .into_inner()
            .expect("進捗コールバックのエラー保持")
        {
            return Err(e);
        }
        let mut analysis = analysis.map_err(to_py)?;
        // GPU に落ちた理由は必ず結果に残す（SPEC §4.4 のフォールバック）
        if let Some(note) = note {
            analysis.warnings.push(note);
        }
        Ok(PySolution {
            mesh: self.mesh.clone_ref(py),
            case: case.clone(),
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
    /// 解いたときのケース。領域統計や実行サマリに σ と領域名が要る
    case: gfem_core::Case,
    inner: Analysis,
}

impl PySolution {
    /// 表示・等値面用の節点値。要素値の量は体積重み平均で節点に移す（SPEC §5.1）。
    fn nodal_values(&self, quantity: &str) -> PyResult<Vec<f64>> {
        if quantity == "phi" {
            return Ok(self.inner.potential.clone());
        }
        let f = &self.inner.fields;
        let element_values = match quantity {
            "e_magnitude" => &f.e_magnitude,
            "j_magnitude" => &f.j_magnitude,
            "power_density" => &f.power_density,
            other => {
                return Err(GfemError::new_err(format!(
                    "未知の量です: \"{other}\"（phi / e_magnitude / j_magnitude / power_density）"
                )));
            }
        };
        let mesh = &self.mesh.get().inner;
        Ok(gfem_core::post::nodal_average(
            mesh,
            element_values,
            &f.volume,
            1,
        ))
    }
}

fn region_stats_dict<'py>(py: Python<'py>, s: &RegionStats) -> PyResult<Bound<'py, PyDict>> {
    let quantity = |q: &gfem_core::post::QuantityStats| -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("max", q.max)?;
        d.set_item("mean", q.mean)?;
        d.set_item("median", q.median)?;
        d.set_item("p95", q.p95)?;
        d.set_item("p99", q.p99)?;
        Ok(d)
    };
    let hotspot = PyDict::new(py);
    hotspot.set_item("element_id", s.hotspot.element_id)?;
    hotspot.set_item("centroid", s.hotspot.centroid.to_vec())?;
    hotspot.set_item("j_magnitude", s.hotspot.j_magnitude)?;
    hotspot.set_item("e_magnitude", s.hotspot.e_magnitude)?;

    let d = PyDict::new(py);
    d.set_item("pid", s.pid)?;
    d.set_item("name", s.name.clone())?;
    d.set_item("sigma", s.sigma)?;
    d.set_item("n_elements", s.n_elements)?;
    d.set_item("volume", s.volume)?;
    d.set_item("power", s.power)?;
    d.set_item("j_magnitude", quantity(&s.current_density)?)?;
    d.set_item("e_magnitude", quantity(&s.electric_field)?)?;
    d.set_item("hotspot", hotspot)?;
    Ok(d)
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

    /// 領域（PID）ごとの統計（SPEC §5.2）。
    ///
    /// パーセンタイルは**体積重み**。要素サイズが不均一なので、
    /// 要素数ベースの分位点では細かく切られた領域の値が過大に効いてしまう。
    fn region_stats(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let mesh = &self.mesh.get().inner;
        let stats = self.inner.region_stats(mesh, &self.case).map_err(to_py)?;
        let list = PyList::empty(py);
        for s in &stats {
            list.append(region_stats_dict(py, s)?)?;
        }
        Ok(list.unbind())
    }

    /// 領域統計を CSV に書き出す（SPEC §5.5）。
    fn write_region_stats_csv(&self, path: PathBuf) -> PyResult<()> {
        let mesh = &self.mesh.get().inner;
        self.inner
            .write_region_stats_csv(&path, mesh, &self.case)
            .map_err(to_py)
    }

    /// 収束履歴を CSV に書き出す。
    fn write_convergence_csv(&self, path: PathBuf) -> PyResult<()> {
        self.inner.write_convergence_csv(&path).map_err(to_py)
    }

    /// 実行サマリ JSON（プロビナンス付き）を書き出す（SPEC §5.5）。
    ///
    /// 入力メッシュの SHA-256、ケースファイルの全文、Git のコミットハッシュと
    /// 作業ツリーの汚れ、実行時刻を記録するので、「この図はどの入力から出たか」を
    /// 後から追える。
    #[pyo3(signature = (path, case_text = None))]
    fn write_summary_json(&self, path: PathBuf, case_text: Option<String>) -> PyResult<()> {
        let mesh = &self.mesh.get().inner;
        let mut provenance = Provenance::capture();
        if let Some(source) = &mesh.source {
            provenance = provenance.with_mesh(source).map_err(to_py)?;
        }
        if let Some(text) = case_text {
            provenance = provenance.with_case_text(text);
        }
        self.inner
            .write_summary_json(&path, mesh, &self.case, &provenance)
            .map_err(to_py)
    }

    /// 実行サマリを JSON 文字列として返す。
    #[pyo3(signature = (case_text = None))]
    fn summary_json(&self, case_text: Option<String>) -> PyResult<String> {
        let mesh = &self.mesh.get().inner;
        let mut provenance = Provenance::capture();
        if let Some(source) = &mesh.source {
            provenance = provenance.with_mesh(source).map_err(to_py)?;
        }
        if let Some(text) = case_text {
            provenance = provenance.with_case_text(text);
        }
        Ok(self
            .inner
            .summary_json(mesh, &self.case, &provenance)
            .map_err(to_py)?
            .to_pretty_string())
    }

    /// 等値面を Marching Tetrahedra で抽出する（SPEC §6.3）。
    ///
    /// `quantity` は `"phi"` / `"e_magnitude"` / `"j_magnitude"` / `"power_density"`。
    /// 要素値の量は表示のために節点へ体積重み平均してから等値面を取る。
    /// 戻り値は `(頂点 (n, 3), 三角形 (m, 3), 頂点のスカラー (n,))`。
    fn iso_surface<'py>(
        &self,
        py: Python<'py>,
        quantity: &str,
        level: f64,
    ) -> PyResult<IsoSurfaceArrays<'py>> {
        let mesh = &self.mesh.get().inner;
        let values = self.nodal_values(quantity)?;
        let surface =
            gfem_core::mesh::iso_surface(mesh, &values, level, &[&values]).map_err(to_py)?;
        let n_points = surface.n_points();
        let n_triangles = surface.n_triangles();
        Ok((
            surface.points.into_pyarray(py).reshape([n_points, 3])?,
            surface
                .triangles
                .into_pyarray(py)
                .reshape([n_triangles, 3])?,
            surface
                .scalars
                .into_iter()
                .next()
                .unwrap_or_default()
                .into_pyarray(py),
        ))
    }

    /// 指定した量の値域 `(最小, 最大)`。等値面のしきい値を決めるのに使う。
    fn value_range(&self, quantity: &str) -> PyResult<(f64, f64)> {
        let values = self.nodal_values(quantity)?;
        Ok((
            values.iter().copied().fold(f64::INFINITY, f64::min),
            values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        ))
    }

    /// 電流のストリームラインを引く（SPEC §6.3）。
    ///
    /// Tet4 では J が要素内で一定なので、要素ごとに直線として**厳密に**追跡できる。
    /// `seeds` は始点の配列 `(k, 3)`。戻り値は折れ線の頂点配列のリスト。
    #[pyo3(signature = (seeds, direction = "forward", max_steps = 100_000, max_length = f64::INFINITY))]
    fn streamlines(
        &self,
        py: Python<'_>,
        seeds: Vec<[f64; 3]>,
        direction: &str,
        max_steps: usize,
        max_length: f64,
    ) -> PyResult<Py<PyList>> {
        let mesh = &self.mesh.get().inner;
        let options = TraceOptions {
            max_steps,
            max_length,
            direction: match direction.to_ascii_lowercase().as_str() {
                "forward" => TraceDirection::Forward,
                "backward" => TraceDirection::Backward,
                other => {
                    return Err(GfemError::new_err(format!(
                        "未知の向きです: \"{other}\"（forward / backward）"
                    )));
                }
            },
        };
        let adjacency = gfem_core::mesh::build_face_adjacency(mesh).map_err(to_py)?;
        let lines = gfem_core::mesh::trace_streamlines(
            mesh,
            &adjacency,
            &self.inner.fields.current_density,
            &seeds,
            &options,
        )
        .map_err(to_py)?;

        let out = PyList::empty(py);
        for line in lines {
            let n = line.n_points();
            out.append(line.points.into_pyarray(py).reshape([n, 3])?)?;
        }
        Ok(out.unbind())
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

/// このビルドに CUDA バックエンドが入っているか（SPEC §8 M5）。
#[pyfunction]
fn cuda_available() -> bool {
    cuda::built_with_cuda()
}

/// 見えている GPU の一覧。
///
/// 各要素は `{"ordinal", "name", "compute_capability", "total_memory", "free_memory",
/// "usable"}`。`usable` は倍精度が使えるか（Compute Capability 6.0 以上）。
#[pyfunction]
fn cuda_devices(py: Python<'_>) -> PyResult<Py<PyList>> {
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
/// 解けるなら `None`、駄目ならその理由（日本語）を返す。
#[pyfunction]
#[pyo3(signature = (n_nodes, nnz = None))]
fn cuda_check(n_nodes: usize, nnz: Option<usize>) -> Option<String> {
    cuda::check(n_nodes, nnz)
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
    m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_devices, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_check, m)?)?;
    Ok(())
}
