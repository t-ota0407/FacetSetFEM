//! `Solution` の Python 向けの顔: 結果の配列・統計・書き出し・等値面・流線（SPEC §5 / §6.3）。

use crate::FacetSetFemError;
use crate::convert::{array1, lazy, to_py, view1, view2};
use crate::mesh::PyMesh;
use facetsetfem_core::mesh::streamline::{TraceDirection, TraceOptions};
use facetsetfem_core::post::RegionStats;
use facetsetfem_core::provenance::{Provenance, SourceTree};
use facetsetfem_core::{Analysis, VtuOptions};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::PathBuf;
use std::sync::OnceLock;

/// 等値面を Python 側に返すときの形。`(頂点, 三角形, 頂点のスカラー)`。
pub(crate) type IsoSurfaceArrays<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<u32>>,
    Bound<'py, PyArray1<f64>>,
);

// --- 解析結果 ---------------------------------------------------------------

/// 解析結果（SPEC §5）。
#[pyclass(name = "Solution", module = "facetsetfem._core", frozen)]
pub struct PySolution {
    pub(crate) mesh: Py<PyMesh>,
    /// 解いたときのケース。領域統計や実行サマリに σ と領域名が要る
    pub(crate) case: facetsetfem_core::Case,
    pub(crate) inner: Analysis,
    /// 領域統計（初回だけ求める）。表・CSV・実行サマリで使い回す。
    /// 大規模モデルでは領域ごとのソートに数百ミリ秒かかる
    pub(crate) stats: OnceLock<Vec<RegionStats>>,
}

impl PySolution {
    /// 領域統計（初回だけ求める）。
    pub(crate) fn stats(&self) -> PyResult<&[RegionStats]> {
        lazy(&self.stats, || {
            self.inner.region_stats(&self.mesh.get().inner, &self.case)
        })
        .map(Vec::as_slice)
        .map_err(to_py)
    }

    /// 実行サマリに添えるプロビナンス（SPEC §5.5）。
    ///
    /// - Git の状態は facetsetfem-core 単体ではなく、**この拡張モジュール全体**
    ///   （facetsetfem-core・facetsetfem-cuda・facetsetfem-py）を作ったソースのもの（`build.rs`）
    /// - メッシュのハッシュは**読み込んだときに取ったもの**。いまファイルを読み直すと、
    ///   読み込み後に書き換えられていた場合に、解いていないメッシュのハッシュになる
    /// - 開発環境では、いまの作業ツリーの状態も添える（`facetsetfem._provenance`）
    pub(crate) fn provenance(
        &self,
        py: Python<'_>,
        case_text: Option<String>,
        case_file: Option<PathBuf>,
    ) -> PyResult<Provenance> {
        let mesh = self.mesh.get();
        let mut provenance = Provenance::capture().with_git_state(
            option_env!("FACETSETFEM_GIT_COMMIT"),
            option_env!("FACETSETFEM_GIT_DIRTY"),
            option_env!("FACETSETFEM_GIT_SCOPE"),
        );
        if let Some(tree) = source_tree(py)? {
            provenance = provenance.with_source_tree(tree);
        }
        if let Some(source) = &mesh.inner.source {
            provenance = match &mesh.report.sha256 {
                Some(hash) => provenance.with_mesh_hash(source, hash.clone()),
                None => provenance.with_mesh(source).map_err(to_py)?,
            };
        }
        if let Some(text) = case_text {
            provenance = provenance.with_case_text(text);
        }
        provenance.case_file = case_file.map(|p| p.display().to_string());
        Ok(provenance)
    }

    /// 表示・等値面用の節点値。要素値の量は体積重み平均で節点に移す（SPEC §5.1）。
    pub(crate) fn nodal_values(&self, quantity: &str) -> PyResult<Vec<f64>> {
        if quantity == "phi" {
            return Ok(self.inner.potential.clone());
        }
        let f = &self.inner.fields;
        let element_values = match quantity {
            "e_magnitude" => &f.e_magnitude,
            "j_magnitude" => &f.j_magnitude,
            "power_density" => &f.power_density,
            other => {
                return Err(FacetSetFemError::new_err(facetsetfem_core::tr!(
                    "Unknown quantity: \"{other}\" (phi / e_magnitude / j_magnitude / power_density)",
                    "未知の量です: \"{other}\"（phi / e_magnitude / j_magnitude / power_density）"
                )));
            }
        };
        let mesh = &self.mesh.get().inner;
        Ok(facetsetfem_core::post::nodal_average(
            mesh,
            element_values,
            &f.volume,
            1,
        ))
    }
}

pub(crate) fn region_stats_dict<'py>(
    py: Python<'py>,
    s: &RegionStats,
) -> PyResult<Bound<'py, PyDict>> {
    let quantity = |q: &facetsetfem_core::post::QuantityStats| -> PyResult<Bound<'py, PyDict>> {
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
    ///
    /// この getter を含め、結果の配列は**複製せずに読み取り専用で**渡す（SPEC §6.3）。
    /// 何度呼んでも同じメモリを指すので速いが、書き換えはできない（要るなら `.copy()`）。
    /// pyvista などに渡してその先で書き換えるなら、先に `.copy()` すること（pyvista は
    /// 読み取り専用の印を見ずにメモリを共有するので、この結果そのものが書き換わる）。
    #[getter]
    fn potential<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().inner.potential)
    }

    /// 電界 E [V/m]、`(n_elements, 3)`。Tet4 では要素内で一定。
    #[getter]
    fn electric_field<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        view2(slf.as_any(), &slf.get().inner.fields.electric_field, 3)
    }

    /// 電流密度 J [A/m²]、`(n_elements, 3)`。
    #[getter]
    fn current_density<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        view2(slf.as_any(), &slf.get().inner.fields.current_density, 3)
    }

    /// |E| [V/m]、`(n_elements,)`。**最大値の評価にはこの要素値を使うこと**。
    #[getter]
    fn e_magnitude<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().inner.fields.e_magnitude)
    }

    /// |J| [A/m²]、`(n_elements,)`。
    #[getter]
    fn j_magnitude<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().inner.fields.j_magnitude)
    }

    /// 要素内の |E| の最大 [V/m]、`(n_elements,)`（SPEC §5.1）。
    ///
    /// 2 次要素では E が要素内で変わるので、頂点での値の最大（中心の値の `e_magnitude` は
    /// 最大を過小評価する）。1 次要素では `e_magnitude` と同じ。
    #[getter]
    fn e_peak<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        let fields = &slf.get().inner.fields;
        if fields.e_peak.is_empty() {
            view1(slf.as_any(), &fields.e_magnitude)
        } else {
            view1(slf.as_any(), &fields.e_peak)
        }
    }

    /// 要素内の |J| の最大 [A/m²]、`(n_elements,)`（[`e_peak`](Self::e_peak) と同じ扱い）。
    #[getter]
    fn j_peak<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        let fields = &slf.get().inner.fields;
        if fields.j_peak.is_empty() {
            view1(slf.as_any(), &fields.j_magnitude)
        } else {
            view1(slf.as_any(), &fields.j_peak)
        }
    }

    /// 電力密度 σ|E|² [W/m³]、`(n_elements,)`。E が要素内で変わる要素では要素内の平均。
    #[getter]
    fn power_density<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().inner.fields.power_density)
    }

    /// 要素体積 [m³]、`(n_elements,)`。
    #[getter]
    fn element_volume<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().inner.fields.volume)
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
                return Err(FacetSetFemError::new_err(facetsetfem_core::tr!(
                    "Unknown quantity: \"{other}\" (e_magnitude / j_magnitude / power_density)",
                    "未知の量です: \"{other}\"（e_magnitude / j_magnitude / power_density）"
                )));
            }
        };
        let mesh = &self.mesh.get().inner;
        let nodal = facetsetfem_core::post::nodal_average(mesh, values, &f.volume, 1);
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

    /// 電極ごとの電位 [V]。一様電流密度は面積重み平均電位、完全電極モデルは板の電位（SPEC §5.3）。
    #[getter]
    fn electrode_potentials(&self) -> Vec<(String, f64)> {
        self.inner.conservation.electrode_potentials.clone()
    }

    /// 電極ごとの状態（SPEC §2.5 / §5.5）。
    ///
    /// 辞書の項目: `name` / `role` / `distribution` / `contact_impedance`（Ω·m²。一様電流密度は
    /// `None`）/ `current`（指定した流入電流 [A]。陰極は負）/ `current_from_field`（φ から求め直した
    /// 電流）/ `area` [m²] / `n_faces` / `potential`（電極の電位 [V]）/ `mean_surface_potential`
    /// （電極の下の皮膚の平均電位）/ `mean_current_density`（I/A）/ `peak_current_density`
    /// （皮膚を通って流れ込む電流密度の最大 [A/m²]）/ `contact_power`（接触面の消費電力 [W]）。
    #[getter]
    fn electrodes(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let list = PyList::empty(py);
        for (e, state) in self
            .inner
            .electrodes
            .iter()
            .zip(self.inner.electrode_states())
        {
            let d = PyDict::new(py);
            d.set_item("name", &e.name)?;
            d.set_item("role", e.role.label())?;
            d.set_item("distribution", e.model.label())?;
            d.set_item("contact_impedance", e.contact_impedance())?;
            d.set_item("current", e.signed_current)?;
            d.set_item("current_from_field", state.current)?;
            d.set_item("area", e.area)?;
            d.set_item("n_faces", e.n_faces)?;
            d.set_item("potential", state.potential)?;
            d.set_item("mean_surface_potential", state.mean_surface_potential)?;
            d.set_item("mean_current_density", e.current_density)?;
            d.set_item("peak_current_density", state.peak_current_density)?;
            d.set_item("contact_power", state.contact_power)?;
            list.append(d)?;
        }
        Ok(list.unbind())
    }

    /// 電極面の節点ごとの、皮膚を通って流れ込む電流密度 [A/m²]（流入を正、電極の外は 0。SPEC §5.5）。
    ///
    /// 完全電極モデルは `(U − φ)/z`、一様電流密度は `I/A`。呼ぶたびに作る（複製ではない）。
    #[getter]
    fn electrode_current_density<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array1(
            py,
            self.inner.electrode_current_density(&self.mesh.get().inner),
        )
    }

    /// 解いた系の数（重ね合わせで何度か解いたときは 2 以上。SPEC §2.5）。
    #[getter]
    fn n_solves(&self) -> usize {
        self.inner.solves.len()
    }

    /// 右辺ごとの `(反復回数, 相対残差)`（解いた順）。
    #[getter]
    fn solves(&self) -> Vec<(usize, f64)> {
        self.inner.solves.clone()
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
    fn history<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        view1(slf.as_any(), &slf.get().inner.solver.history)
    }

    /// 使用したバックエンド名。
    #[getter]
    fn backend(&self) -> &'static str {
        self.inner.solver.backend
    }

    /// 使用した前処理の名前（`"jacobi"` / `"amg"` / `"none"`）。
    #[getter]
    fn preconditioner(&self) -> &'static str {
        self.inner.solver.preconditioner
    }

    /// AMG の階層の要約。AMG 以外で解いた場合は `None`（SPEC §4.5）。
    ///
    /// `levels` はレベルごとの `(節点数, 非ゼロ数)`。`operator_complexity` は
    /// V サイクル 1 回が最細レベルの SpMV 何回ぶんに相当するかの目安。
    #[getter]
    fn amg(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let Some(info) = &self.inner.solver.amg else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        d.set_item("levels", info.levels.clone())?;
        d.set_item("operator_complexity", info.operator_complexity)?;
        d.set_item("grid_complexity", info.grid_complexity)?;
        d.set_item("direct_coarse_solve", info.direct_coarse_solve)?;
        d.set_item("setup_seconds", self.inner.solver.setup_seconds)?;
        Ok(Some(d.unbind()))
    }

    /// 保存則チェックの結果（SPEC §5.4）。
    #[getter]
    fn conservation(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let c = &self.inner.conservation;
        let d = PyDict::new(py);
        d.set_item("current_balance", c.current_balance)?;
        d.set_item("electrode_current_error", c.electrode_current_error)?;
        d.set_item("power_from_field", c.power_from_field)?;
        d.set_item("contact_power", c.contact_power)?;
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
        let list = PyList::empty(py);
        for s in self.stats()? {
            list.append(region_stats_dict(py, s)?)?;
        }
        Ok(list.unbind())
    }

    /// 領域統計を CSV に書き出す（SPEC §5.5）。
    fn write_region_stats_csv(&self, path: PathBuf) -> PyResult<()> {
        facetsetfem_core::post::write_region_stats_csv(&path, self.stats()?).map_err(to_py)
    }

    /// 節点ごとの結果を CSV に書き出す: `node_id,x_m,y_m,z_m,phi_V`（SPEC §5.5）。
    ///
    /// 値は読み直すと同じ値に戻る最短の表記で書く。座標の列があるので、ほかのソフトウェアに
    /// 「この点で値を出す」点の一覧としてそのまま渡せる。
    fn write_nodes_csv(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let mesh = &self.mesh.get().inner;
        py.detach(|| facetsetfem_core::post::write_node_table(&path, mesh, &self.inner.potential))
            .map_err(to_py)
    }

    /// 要素ごとの結果を CSV に書き出す（重心の座標と E・J・|E|・|J|・要素内の最大・
    /// 電力密度・体積。SPEC §5.5）。重心は頂点の平均で、1 次と 2 次で同じ点になる。
    fn write_elements_csv(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let mesh = &self.mesh.get().inner;
        py.detach(|| facetsetfem_core::post::write_element_table(&path, mesh, &self.inner.fields))
            .map_err(to_py)
    }

    /// 収束履歴を CSV に書き出す。
    fn write_convergence_csv(&self, path: PathBuf) -> PyResult<()> {
        self.inner.write_convergence_csv(&path).map_err(to_py)
    }

    /// 実行サマリ JSON（プロビナンス付き）を書き出す（SPEC §5.5）。
    ///
    /// 入力メッシュの SHA-256、解析条件の全文、Git のコミットハッシュと
    /// 作業ツリーの汚れ、実行時刻を記録するので、「この図はどの入力から出たか」を
    /// 後から追える。
    ///
    /// `case_text` には**解いたときの解析条件**を渡すこと（`Study.to_yaml()`）。
    /// ディスク上のケースファイルは、読み込んだ後に条件を変えて解いた場合に食い違う。
    /// `case_file` はその条件をどのファイルから読んだか（あれば）。
    #[pyo3(signature = (path, case_text = None, case_file = None))]
    fn write_summary_json(
        &self,
        py: Python<'_>,
        path: PathBuf,
        case_text: Option<String>,
        case_file: Option<PathBuf>,
    ) -> PyResult<()> {
        let provenance = self.provenance(py, case_text, case_file)?;
        let mesh = &self.mesh.get().inner;
        self.inner
            .write_summary_json_with_stats(&path, mesh, &self.case, &provenance, self.stats()?)
            .map_err(to_py)
    }

    /// 実行サマリを JSON 文字列として返す（引数は [`write_summary_json`] と同じ）。
    #[pyo3(signature = (case_text = None, case_file = None))]
    fn summary_json(
        &self,
        py: Python<'_>,
        case_text: Option<String>,
        case_file: Option<PathBuf>,
    ) -> PyResult<String> {
        let provenance = self.provenance(py, case_text, case_file)?;
        let mesh = &self.mesh.get().inner;
        Ok(self
            .inner
            .summary_json_with_stats(mesh, &self.case, &provenance, self.stats()?)
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
        let tets = self.mesh.get().tets()?;
        let surface = facetsetfem_core::mesh::iso_surface(mesh, tets, &values, level, &[&values])
            .map_err(to_py)?;
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
    /// J は要素内で一定（Hex8 / Penta6 では要素中心の代表値）なので、
    /// 要素ごとに直線として**厳密に**追跡できる。Hex8 / Penta6 を含むメッシュは
    /// 面共役な四面体分割を通す（SPEC §5.1）。
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
                    return Err(FacetSetFemError::new_err(facetsetfem_core::tr!(
                        "Unknown direction: \"{other}\" (forward / backward)",
                        "未知の向きです: \"{other}\"（forward / backward）"
                    )));
                }
            },
        };
        let lines = self
            .mesh
            .get()
            .navigator()?
            .trace_many(mesh, &self.inner.fields.current_density, &seeds, &options)
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
            facetsetfem_core::write_analysis_vtu(
                &path,
                mesh,
                &self.inner,
                &VtuOptions { compress, level },
            )
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

/// 実行時の作業ツリー（`facetsetfem._provenance.source_tree()`）。開発環境でなければ `None`。
pub(crate) fn source_tree(py: Python<'_>) -> PyResult<Option<SourceTree>> {
    let value = py
        .import("facetsetfem._provenance")?
        .call_method0("source_tree")?;
    let tree: Option<(String, Option<String>, Option<bool>)> = value.extract()?;
    Ok(tree.map(|(path, git_commit, git_dirty)| SourceTree {
        path,
        git_commit,
        git_dirty,
    }))
}
