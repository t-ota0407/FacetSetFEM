//! `Case` の Python 向けの顔: 導電率・電極・ソルバ設定・検証・求解（SPEC §3.3 / §6.5）。

use crate::convert::to_py;
use crate::mesh::PyMesh;
use crate::patch::parse_patch;
use crate::solution::PySolution;
use crate::{FacetSetFemError, cuda};
use facetsetfem_core::LengthUnit;
use facetsetfem_core::case::{
    Backend, Electrode, ElectrodeModel, ElectrodeRole, FaceRef, Preconditioner, ReferencePotential,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

// --- 解析ケース -------------------------------------------------------------

/// 解析ケース（SPEC §3.3）。
#[pyclass(name = "Case", module = "facetsetfem._core")]
pub struct PyCase {
    pub(crate) mesh: Py<PyMesh>,
    pub(crate) inner: facetsetfem_core::Case,
}

#[pymethods]
impl PyCase {
    #[new]
    fn new(mesh: Py<PyMesh>) -> Self {
        Self {
            mesh,
            inner: facetsetfem_core::Case::new(),
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
                facetsetfem_core::tr!(
                    "Specify σ [S/m] as set_conductivity(pid, sigma)",
                    "set_conductivity(pid, sigma) の形で σ [S/m] を指定してください"
                ),
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
    /// 面はシェル PID の集合・個別の要素 ID・体表面のパッチの組み合わせで指定する
    /// （SPEC §3.3）。
    ///
    /// `patches` は `{"center": [x, y, z], "radius": r, "pid": 51}` の一覧。
    /// **シェル要素を持たないモデル（人体モデルはこれが普通）はこちらを使う**
    /// （SPEC §11.1）:
    ///
    /// ```python
    /// case.add_electrode("E1", role="anode", current=2e-3,
    ///                    patches=[{"center": [0.0, 1.76, 0.02], "radius": 0.025, "pid": 51}])
    /// ```
    ///
    /// `distribution` は `"uniform_current_density"`（既定。SPEC §2.4）か
    /// `"complete_electrode"`（完全電極モデル。SPEC §2.5）。後者には
    /// `contact_impedance`（接触インピーダンス [Ω·m²]）が必須。
    #[pyo3(signature = (
        name, role, current,
        shell_pids = None, element_ids = None, patches = None, added = None, removed = None,
        distribution = None, contact_impedance = None
    ))]
    #[allow(clippy::too_many_arguments)] // Python 側の名前つき引数に合わせる
    fn add_electrode(
        &mut self,
        name: &str,
        role: &str,
        current: f64,
        shell_pids: Option<Vec<u32>>,
        element_ids: Option<Vec<u32>>,
        patches: Option<Vec<Bound<'_, PyAny>>>,
        added: Option<Vec<(u32, u8)>>,
        removed: Option<Vec<(u32, u8)>>,
        distribution: Option<&str>,
        contact_impedance: Option<f64>,
    ) -> PyResult<()> {
        let mut electrode = build_electrode(
            name,
            parse_role(role)?,
            current,
            shell_pids,
            element_ids,
            patches,
            added,
            removed,
        )?;
        electrode.model = ElectrodeModel::parse(
            distribution.unwrap_or(ElectrodeModel::UniformCurrentDensity.label()),
            contact_impedance,
        )
        .map_err(to_py)?;
        self.inner.add_electrode(electrode);
        Ok(())
    }

    /// 定義済みの電極をすべて消す。
    fn clear_electrodes(&mut self) {
        self.inner.electrodes.clear();
    }

    /// ソルバ設定を変更する。
    ///
    /// `backend` は `"auto"` / `"cpu"` / `"cpu_parallel"` / `"cuda"`。
    /// **どれを選んでも結果はビット単位で同じ**で、速度だけが変わる（SPEC §4.4）。
    ///
    /// `preconditioner` は `"jacobi"`（既定）/ `"amg"` / `"none"`。
    /// こちらは反復の道筋が変わるので、収束判定の範囲内で答えが揺れる。
    /// `"amg"` は反復回数がメッシュの細かさにほとんど依存しなくなる代わりに
    /// 1 反復が重くなる（SPEC §4.5）。
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
            self.inner.solver.preconditioner = Preconditioner::parse(p).map_err(to_py)?;
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
        d.set_item("preconditioner", self.inner.solver.preconditioner.label())?;
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
    fn validate(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        let mesh = self.mesh.get();
        // 外表面の抽出（初回）は大規模モデルで数秒かかるので GIL を離す（SPEC §6.4）
        py.detach(|| self.inner.validate_with(&mesh.lookup()?))
            .map_err(to_py)
    }

    /// メッシュに電極の面をシェル要素として書き足し、Nastran（自由形式）で書き出す（SPEC §3.1）。
    ///
    /// GUI で体表面から選んだ電極を、ほかのソフトウェア（COMSOL など）に持っていくための経路。
    /// 電極ごとに新しい PSHELL の PID を振る。`pids` で `{電極名: PID}` を指定でき、
    /// 指定の無い電極にはメッシュの PID と重ならない番号を順に振る。座標は `length_unit`
    /// （省略時はメッシュを読み込んだ単位）で書く。戻り値は `{電極名: PID}`。
    /// 2 次要素のメッシュでは、電極の面は辺の中点つきの CTRIA6 になる。
    #[pyo3(signature = (path, pids = None, length_unit = None))]
    fn write_nastran(
        &self,
        py: Python<'_>,
        path: PathBuf,
        pids: Option<std::collections::BTreeMap<String, u32>>,
        length_unit: Option<&str>,
    ) -> PyResult<Vec<(String, u32)>> {
        let mesh = self.mesh.get();
        let unit = match length_unit {
            Some(u) => LengthUnit::parse(u).map_err(to_py)?,
            None => mesh.inner.length_unit,
        };
        let pids = pids.unwrap_or_default();
        py.detach(|| -> facetsetfem_core::Result<Vec<(String, u32)>> {
            let lookup = mesh.lookup()?;
            // 指定の無い電極に振る番号。指定された PID とも重ならないよう、まとめて多めに取る
            let mut free = facetsetfem_core::io::nastran::free_pids(
                &mesh.inner,
                self.inner.electrodes.len() + pids.len(),
            )
            .into_iter()
            .filter(|p| !pids.values().any(|q| q == p));
            let mut sets = Vec::with_capacity(self.inner.electrodes.len());
            for e in &self.inner.electrodes {
                let faces = facetsetfem_core::case::resolve_faces(&lookup, e)?;
                let pid = match pids.get(&e.name) {
                    Some(&pid) => pid,
                    None => free.next().expect("候補は電極の数より多く取ってある"),
                };
                sets.push(facetsetfem_core::io::nastran::ShellSet {
                    name: e.name.clone(),
                    pid,
                    faces: faces.iter().map(|&f| lookup.face_trace(f)).collect(),
                });
            }
            facetsetfem_core::io::nastran::write_nastran(&path, &mesh.inner, &sets, unit)?;
            Ok(sets.into_iter().map(|s| (s.name, s.pid)).collect())
        })
        .map_err(to_py)
    }

    /// 実行前の整合性チェックの不備を**全部**集める（SPEC §6.2 の手順 4）。
    ///
    /// `(不備, 警告)` をどちらも文字列の一覧で返す（例外は出さない）。不備の 1 件目は
    /// `validate` が例外にするものと同じ。GUI はこれで不備を一覧にして一度に直させる。
    fn problems(&self, py: Python<'_>) -> PyResult<(Vec<String>, Vec<String>)> {
        let mesh = self.mesh.get();
        let (problems, warnings) = py
            .detach(|| {
                mesh.lookup()
                    .map(|lookup| self.inner.problems_with(&lookup))
            })
            .map_err(to_py)?;
        Ok((problems.iter().map(|p| p.to_string()).collect(), warnings))
    }

    /// 電極ごとの面積と電流密度。面を選びながら確認するために使う（SPEC §6.2 の手順 3）。
    fn electrode_info(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let mesh = self.mesh.get();
        let lookup = mesh.lookup().map_err(to_py)?;
        let (_, geometries) =
            facetsetfem_core::assemble::assemble_load(&mesh.inner, &self.inner, &lookup)
                .map_err(to_py)?;
        let list = PyList::empty(py);
        for g in geometries {
            let d = PyDict::new(py);
            d.set_item("name", &g.name)?;
            d.set_item("role", g.role.label())?;
            d.set_item("current", g.signed_current)?;
            d.set_item("n_faces", g.n_faces)?;
            d.set_item("area", g.area)?;
            d.set_item("current_density", g.current_density)?;
            d.set_item("distribution", g.model.label())?;
            d.set_item("contact_impedance", g.contact_impedance())?;
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
        let py_mesh = self.mesh.get();
        let mesh = &py_mesh.inner;
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
            // GPU を使うかはここで決める。facetsetfem-core は CUDA を知らない（SPEC §7）
            let (solver, note) = cuda::pick(
                case.solver.backend,
                case.solver.preconditioner,
                mesh.nodes.len(),
                // 組み立てる前に決めるので、非ゼロ数は要素の種類から見積もる
                Some(mesh.estimated_nnz()),
            );
            // 外表面と電極面の索引は Mesh が作り置きしたものを使う。何度解いても
            // 抽出し直さない（大規模モデルでは solve の準備の大半を占める）
            let analysis = py_mesh.lookup().and_then(|lookup| {
                facetsetfem_core::analysis::solve_prepared(
                    &lookup,
                    case,
                    Some(&mut cb),
                    solver.as_deref(),
                )
            });
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
            stats: OnceLock::new(),
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

pub(crate) fn parse_role(role: &str) -> PyResult<ElectrodeRole> {
    match role.to_ascii_lowercase().as_str() {
        "anode" => Ok(ElectrodeRole::Anode),
        "cathode" => Ok(ElectrodeRole::Cathode),
        other => Err(FacetSetFemError::new_err(facetsetfem_core::tr!(
            "Unknown electrode role: \"{other}\" (anode / cathode)",
            "未知の電極の役割です: \"{other}\"（anode / cathode）"
        ))),
    }
}

/// 電極 1 個を組み立てる。`Case.add_electrode` と `Mesh.resolve_electrode_faces` で
/// **同じ解釈**を通すために 1 か所にまとめてある。
///
/// `added` / `removed` の面番号は **1 始まり**（ケースファイルとエラーメッセージに
/// 合わせる）。コアの [`FaceRef`] は 0 始まりなので、ここで直す。
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_electrode(
    name: &str,
    role: ElectrodeRole,
    current: f64,
    shell_pids: Option<Vec<u32>>,
    element_ids: Option<Vec<u32>>,
    patches: Option<Vec<Bound<'_, PyAny>>>,
    added: Option<Vec<(u32, u8)>>,
    removed: Option<Vec<(u32, u8)>>,
) -> PyResult<Electrode> {
    let mut parsed = Vec::new();
    for (i, patch) in patches.unwrap_or_default().iter().enumerate() {
        parsed.push(parse_patch(patch, name, i + 1)?);
    }
    let to_refs = |v: Option<Vec<(u32, u8)>>, what: String| -> PyResult<Vec<FaceRef>> {
        v.unwrap_or_default()
            .into_iter()
            .map(|(eid, local)| {
                if local == 0 {
                    return Err(FacetSetFemError::new_err(facetsetfem_core::tr!(
                        "A face number in the {what} of electrode {name} is 0 (EID={eid}). \
                         Face numbers start at 1",
                        "電極 {name} の{what}の面番号が 0 です（EID={eid}）。\
                         面番号は 1 始まりで指定してください"
                    )));
                }
                Ok(FaceRef::new(eid, local - 1))
            })
            .collect()
    };
    Ok(Electrode {
        name: name.to_string(),
        role,
        current,
        shell_pids: shell_pids.unwrap_or_default(),
        element_ids: element_ids.unwrap_or_default(),
        patches: parsed,
        added_faces: to_refs(
            added,
            facetsetfem_core::tr!("manually added faces", "手で足した面"),
        )?,
        removed_faces: to_refs(
            removed,
            facetsetfem_core::tr!("manually removed faces", "手で外した面"),
        )?,
        model: ElectrodeModel::default(),
    })
}

pub(crate) fn parse_reference(reference: &str) -> PyResult<ReferencePotential> {
    let r = reference.trim();
    if let Some(rest) = r.strip_prefix("node:") {
        let id: u32 = rest.trim().parse().map_err(|_| {
            FacetSetFemError::new_err(facetsetfem_core::tr!(
                "Cannot interpret as a node ID: \"{rest}\"",
                "節点 ID として解釈できません: \"{rest}\""
            ))
        })?;
        return Ok(ReferencePotential::Node(id));
    }
    match r.to_ascii_lowercase().as_str() {
        "zero_mean" => Ok(ReferencePotential::ZeroMean),
        "cathode_mean" => Ok(ReferencePotential::CathodeMean),
        other => Err(FacetSetFemError::new_err(facetsetfem_core::tr!(
            "Unknown reference potential: \"{other}\" (zero_mean / cathode_mean / node:<ID>)",
            "未知の基準電位です: \"{other}\"（zero_mean / cathode_mean / node:<ID>）"
        ))),
    }
}
