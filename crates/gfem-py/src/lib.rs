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

use gfem_core::case::{
    Backend, Electrode, ElectrodeRole, FaceLookup, FaceRef, Preconditioner, ReferencePotential,
    SurfaceIndex, SurfacePatch,
};
use gfem_core::mesh::streamline::{TraceDirection, TraceOptions};
use gfem_core::mesh::{Plane, TriangleMesh};
use gfem_core::post::RegionStats;
use gfem_core::provenance::{Provenance, SourceTree};
use gfem_core::{Analysis, LengthUnit, VtuOptions};
use numpy::ndarray::{ArrayView1, ArrayView2, Dimension};
use numpy::{IntoPyArray, PyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::create_exception;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

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

/// 等値面を Python 側に返すときの形。`(頂点, 三角形, 頂点のスカラー)`。
type IsoSurfaceArrays<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<u32>>,
    Bound<'py, PyArray1<f64>>,
);

/// `Mesh.region_surface` の戻り値: (三角形, 元のソリッド要素)。
type RegionSurface<'py> = (Bound<'py, PyArray2<u32>>, Bound<'py, PyArray1<u32>>);

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
fn view1<'py>(owner: &Bound<'py, PyAny>, data: &[f64]) -> Bound<'py, PyArray1<f64>> {
    // SAFETY: 上の説明のとおり、`data` は `owner` が生きている間動かず、変わらない
    let array = unsafe { PyArray1::borrow_from_array(&ArrayView1::from(data), owner.clone()) };
    read_only(array)
}

/// [`view1`] の 2 次元版。`data` を `cols` 列ずつに区切った `(len / cols, cols)` の配列。
fn view2<'py>(
    owner: &Bound<'py, PyAny>,
    data: &[f64],
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let view = ArrayView2::from_shape((data.len() / cols, cols), data).map_err(|e| {
        to_py(gfem_core::GfemError::internal(format!(
            "配列を {cols} 列に区切れません（{} 要素）: {e}",
            data.len()
        )))
    })?;
    // SAFETY: view1 と同じ
    let array = unsafe { PyArray2::borrow_from_array(&view, owner.clone()) };
    Ok(read_only(array))
}

/// numpy 配列の書き込み可の印を外す。
fn read_only<D: Dimension>(array: Bound<'_, PyArray<f64, D>>) -> Bound<'_, PyArray<f64, D>> {
    let readonly = array.readwrite().make_nonwriteable();
    (*readonly).clone()
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

    /// 読み込んだファイル全体の SHA-256（16 進小文字。`sha256sum` と同じ値）。
    ///
    /// 読み込むのと同じ 1 回の読み取りで取るので、解いたメッシュそのもののハッシュになる。
    #[getter]
    fn sha256(&self) -> Option<String> {
        self.inner.sha256.clone()
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
    /// 面共役な四面体分割（SPEC §5.1）。等値面と流線が使う。
    ///
    /// メッシュだけで決まるので、呼ばれるたびに作り直さず 1 度だけ作って使い回す。
    /// 使わなければ作られない。
    tets: OnceLock<gfem_core::mesh::TetView>,
    /// 流線の下ごしらえ（四面体分割 + 面隣接 + 空間分割）
    navigator: OnceLock<gfem_core::mesh::Navigator>,
    /// 外表面（SPEC §4.1）。
    ///
    /// 抽出は 4 百万四面体で 3.8 秒かかる。表示にも電極のパッチ選択にも使うので、
    /// 1 度だけ作って使い回す
    surface: OnceLock<gfem_core::mesh::Surface>,
    /// 外表面から電極面を引く索引（パッチ選択と面の識別子の逆引き。SPEC §11.1）。
    ///
    /// solve・検証・クリックのたびに作り直さないよう、ここで 1 つ持って貸す
    /// （[`FaceLookup::with_index`]）。以前は毎回作り直していた
    surface_index: SurfaceIndex,
}

impl PyMesh {
    /// 読み込んだメッシュを包む。キャッシュは使われたときに作る。
    fn new(inner: gfem_core::Mesh, report: gfem_core::ReadReport) -> Self {
        Self {
            inner,
            report,
            tets: OnceLock::new(),
            navigator: OnceLock::new(),
            surface: OnceLock::new(),
            surface_index: SurfaceIndex::new(),
        }
    }

    /// 面共役な四面体分割（初回だけ作る）。
    fn tets(&self) -> PyResult<&gfem_core::mesh::TetView> {
        lazy(&self.tets, || gfem_core::mesh::TetView::build(&self.inner)).map_err(to_py)
    }

    /// 流線の下ごしらえ（初回だけ作る）。
    fn navigator(&self) -> PyResult<&gfem_core::mesh::Navigator> {
        lazy(&self.navigator, || {
            gfem_core::mesh::Navigator::build(&self.inner)
        })
        .map_err(to_py)
    }

    /// 外表面（初回だけ抽出する）。GIL を手放した中からも呼べるよう、コアのエラーで返す。
    fn surface_core(&self) -> gfem_core::Result<&gfem_core::mesh::Surface> {
        lazy(&self.surface, || {
            gfem_core::mesh::extract_surface(&self.inner)
        })
    }

    /// 外表面（初回だけ抽出する）。
    pub(crate) fn surface(&self) -> PyResult<&gfem_core::mesh::Surface> {
        self.surface_core().map_err(to_py)
    }

    /// 体表面から面を選ぶ索引（初回だけ作る）。
    fn picker(&self) -> PyResult<&gfem_core::mesh::SurfacePicker> {
        Ok(self.surface_index.picker(&self.inner, self.surface()?))
    }

    /// 電極面を引く下ごしらえ。外表面と索引は作り置きのものを貸す。
    fn lookup(&self) -> gfem_core::Result<FaceLookup<'_>> {
        Ok(FaceLookup::with_index(
            &self.inner,
            self.surface_core()?,
            &self.surface_index,
        ))
    }

    /// 外表面の面の集合を Python に返す形にまとめる。
    ///
    /// `faces` は [`Surface::faces`] の通し番号。三角形は強調表示に、
    /// `face_refs` は「手で足した面・外した面」を覚えておくために使う
    /// （**面番号は 1 始まり**。エラーメッセージやケースファイルと揃える）。
    fn describe_faces<'py>(&self, py: Python<'py>, faces: &[u32]) -> PyResult<Bound<'py, PyDict>> {
        let surface = self.surface()?;
        let picker = self.picker()?;
        let mut flat: Vec<u32> = Vec::new();
        let mut refs: Vec<i64> = Vec::with_capacity(faces.len() * 2);
        for &f in faces {
            let face = &surface.faces()[f as usize];
            let n = face.nodes();
            flat.extend_from_slice(&[n[0], n[1], n[2]]);
            if n.len() == 4 {
                flat.extend_from_slice(&[n[0], n[2], n[3]]);
            }
            refs.push(self.inner.solid(face.element as usize).id as i64);
            refs.push(face.local_face as i64 + 1);
        }
        let rows = flat.len() / 3;
        let n_refs = refs.len() / 2;

        let d = PyDict::new(py);
        d.set_item("n_faces", faces.len())?;
        d.set_item("area", picker.total_area(faces))?;
        d.set_item("triangles", flat.into_pyarray(py).reshape([rows, 3])?)?;
        d.set_item("face_refs", refs.into_pyarray(py).reshape([n_refs, 2])?)?;
        Ok(d)
    }
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

    /// **m として読んだ**メッシュを、座標の単位が `length_unit` だったものとして返す（SPEC §3.2）。
    ///
    /// GUI は単位を選ばせる前に寸法を見せるため、まず m として読む。選ばれた単位で
    /// 読み直すとファイルを 2 回パースすることになるので、代わりにこれを使う。
    /// 最初からその単位で読んだものと座標はビット単位で同じになる。
    fn with_length_unit(&self, py: Python<'_>, length_unit: &str) -> PyResult<PyMesh> {
        let unit = LengthUnit::parse(length_unit).map_err(to_py)?;
        let mesh = py
            .detach(|| self.inner.reinterpret_length_unit(unit))
            .map_err(to_py)?;
        Ok(PyMesh::new(mesh, self.report.clone()))
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
    ///
    /// **呼ぶたびに複製する**（`Solution` の配列と違って zero-copy にしない）。pyvista などは
    /// numpy の読み取り専用の印を見ずにメモリを共有し、その先から書き換えられてしまう。
    /// 座標が書き換わると、このメッシュで解くものが以後すべて黙って狂うので、ここは
    /// 安全を取る。節点座標は要素ごとの配列より小さい（頭部クラスで 16 MB）。
    fn points<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rows = self.inner.nodes.len();
        self.inner
            .nodes
            .coords()
            .to_vec()
            .into_pyarray(py)
            .reshape([rows, 3])
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
        let (tris, _) = self.surface()?.triangulate();
        let flat: Vec<u32> = tris.into_iter().flatten().collect();
        let rows = flat.len() / 3;
        flat.into_pyarray(py).reshape([rows, 3])
    }

    /// 指定した領域（PID）の要素の集まりの境界面 `(三角形 (n, 3), 元のソリッド要素 (n,))`。
    ///
    /// 体表面に出ていない領域をハイライトするために使う（SPEC §6.3）。体積メッシュを
    /// 組み立てずに、Rust 側で領域の境界だけを取り出す。向きは領域の外向き。
    fn region_surface<'py>(&self, py: Python<'py>, pid: u32) -> PyResult<RegionSurface<'py>> {
        let surface = py
            .detach(|| gfem_core::mesh::extract_region_surface(&self.inner, pid))
            .map_err(to_py)?;
        let (tris, owners) = surface.triangulate();
        let flat: Vec<u32> = tris.into_iter().flatten().collect();
        let rows = flat.len() / 3;
        Ok((
            flat.into_pyarray(py).reshape([rows, 3])?,
            owners.into_pyarray(py),
        ))
    }

    /// 外表面の三角形ごとの、元ソリッド要素の通し番号 `(n_tri,)`。
    ///
    /// 要素値（|E| など）を表面に貼り付けるときの索引に使う。
    fn surface_triangle_elements<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let (_, owners) = self.surface()?.triangulate();
        Ok(owners.into_pyarray(py))
    }

    /// 平面で体積メッシュを切った断面（SPEC §6.3）。
    ///
    /// `origin` は平面上の 1 点 [m]、`normal` は法線（長さは問わない）。戻り値は
    /// `{"points": (n, 3), "triangles": (m, 3), "elements": (m,), "edges": (n, 2), "t": (n,)}`。
    /// 要素値は `elements`（三角形ごとの元ソリッド要素の通し番号）で引き、節点値 `v` は
    /// `v[a] + t * (v[b] - v[a])`（`a, b = edges[i]`）で頂点に補間する。
    ///
    /// 体積メッシュを VTK に渡して切るより桁違いに軽い（切り口の三角形しか作らない）。
    fn slice(&self, py: Python<'_>, origin: [f64; 3], normal: [f64; 3]) -> PyResult<Py<PyDict>> {
        let plane = Plane::new(origin, normal).map_err(to_py)?;
        let tets = self.tets()?;
        let section = py.detach(|| gfem_core::mesh::slice(&self.inner, tets, &plane));
        section_dict(py, section)
    }

    /// 平面の**法線の向きの側を切り落とした**クリップの見た目（SPEC §6.3）。
    ///
    /// 外表面のうち残す側と、切り口を合わせた三角形。戻り値の形は [`Self::slice`] と同じ。
    fn clip(&self, py: Python<'_>, origin: [f64; 3], normal: [f64; 3]) -> PyResult<Py<PyDict>> {
        let plane = Plane::new(origin, normal).map_err(to_py)?;
        let tets = self.tets()?;
        let surface = self.surface()?;
        let section = py.detach(|| gfem_core::mesh::clip(&self.inner, tets, surface, &plane));
        section_dict(py, section)
    }

    /// 外表面に顔を出している PID と、その面積・面数（面積の大きい順）。
    ///
    /// 電極パッチを PID で絞り込むときの手がかり。人体モデルでは首の切断面などで
    /// 内部組織も外表面に現れるので、**皮膚がどの PID かを面積で見当づける**ために使う
    /// （SPEC §11.1）。
    fn surface_pids(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let picker = self.picker()?;
        let mut by_pid: BTreeMap<u32, (usize, f64)> = BTreeMap::new();
        for face in 0..picker.len() as u32 {
            let e = by_pid.entry(picker.pid(face)).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += picker.area(face);
        }
        let mut rows: Vec<(u32, usize, f64)> =
            by_pid.into_iter().map(|(p, (n, a))| (p, n, a)).collect();
        // 面積の大きい順。同じ面積なら PID 順（決定性のため）
        rows.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));

        let list = PyList::empty(py);
        for (pid, n_faces, area) in rows {
            let d = PyDict::new(py);
            d.set_item("pid", pid)?;
            d.set_item("n_faces", n_faces)?;
            d.set_item("area", area)?;
            list.append(d)?;
        }
        Ok(list.unbind())
    }

    /// 体表面の 1 点を中心に、半径の中でつながっている面を選ぶ（SPEC §11.1）。
    ///
    /// GUI の 3D ビューでクリックした点をそのまま渡す。戻り値は
    /// `{"n_faces", "area", "triangles", "center", "radius", "pid"}` で、
    /// `triangles` は選ばれた面を三角形に割った `(n, 3)` の節点索引（強調表示用）。
    ///
    /// **球に入っているかだけでは選ばない。** クリックした面から稜線でつながって
    /// いる面だけを辿るので、首の切断面や耳の裏へ飛び移らない。
    #[pyo3(signature = (center, radius, pid = None))]
    fn pick_patch(
        &self,
        py: Python<'_>,
        center: [f64; 3],
        radius: f64,
        pid: Option<u32>,
    ) -> PyResult<Py<PyDict>> {
        // 半径と中心（ピックに失敗した NaN を含む）の検証は、電極面の解決と同じ関数で行う
        SurfacePatch {
            center,
            radius,
            pid,
        }
        .check()
        .map_err(to_py)?;
        let faces = self.picker()?.select(center, radius, pid);
        let d = self.describe_faces(py, &faces)?;
        d.set_item("center", center.to_vec())?;
        d.set_item("radius", radius)?;
        d.set_item("pid", pid)?;
        Ok(d.unbind())
    }

    /// 電極の面指定を解決して、面数・面積・三角形・面の識別子を返す。
    ///
    /// **ソルバが荷重を作るときと同じ関数**（`gfem_core::case::resolve_faces`）を通す。
    /// GUI の表示だけ別に計算すると、「25 cm² のつもりで置いた電極が実は違う面積
    /// だった」が静かに起きるため（SPEC §6.2 の手順 3）。導電率が未設定でも呼べる。
    ///
    /// `added` / `removed` は `(EID, 面番号)` の並び。**面番号は 1 始まり**。
    #[pyo3(signature = (
        shell_pids = None, element_ids = None, patches = None, added = None, removed = None,
        name = "（名前なし）"
    ))]
    #[allow(clippy::too_many_arguments)] // Python 側の名前つき引数に合わせる
    fn resolve_electrode_faces(
        &self,
        py: Python<'_>,
        shell_pids: Option<Vec<u32>>,
        element_ids: Option<Vec<u32>>,
        patches: Option<Vec<Bound<'_, PyAny>>>,
        added: Option<Vec<(u32, u8)>>,
        removed: Option<Vec<(u32, u8)>>,
        name: &str,
    ) -> PyResult<Py<PyDict>> {
        let electrode = build_electrode(
            name,
            ElectrodeRole::Anode,
            1.0,
            shell_pids,
            element_ids,
            patches,
            added,
            removed,
        )?;
        let lookup = self.lookup().map_err(to_py)?;
        let faces = gfem_core::case::resolve_faces(&lookup, &electrode).map_err(to_py)?;

        // 体表面の面だけ強調表示できる。シェル要素は従来どおり別経路で描く
        let surface_faces: Vec<u32> = faces
            .iter()
            .filter_map(|f| match f {
                gfem_core::case::ElectrodeFace::Surface(i) => Some(*i),
                gfem_core::case::ElectrodeFace::Shell(_) => None,
            })
            .collect();
        let d = self.describe_faces(py, &surface_faces)?;
        // 面数と面積はシェルを含めた全体で出す（ソルバが使う値と一致させる）
        let mut area = 0.0;
        let mut points = [[0.0f64; 3]; 4];
        for &face in &faces {
            let (kind, conn) = lookup.face_nodes(face);
            self.inner.gather_coords(conn, &mut points[..conn.len()]);
            area += gfem_core::element::shell::face_integral(kind, &points[..conn.len()]).area;
        }
        d.set_item("n_faces", faces.len())?;
        d.set_item("area", area)?;
        Ok(d.unbind())
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
        Ok(self.surface()?.total_area(&self.inner))
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
    #[pyo3(signature = (
        name, role, current,
        shell_pids = None, element_ids = None, patches = None, added = None, removed = None
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
    ) -> PyResult<()> {
        let electrode = build_electrode(
            name,
            parse_role(role)?,
            current,
            shell_pids,
            element_ids,
            patches,
            added,
            removed,
        )?;
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

    /// 電極ごとの面積と電流密度。面を選びながら確認するために使う（SPEC §6.2 の手順 3）。
    fn electrode_info(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let mesh = self.mesh.get();
        let lookup = mesh.lookup().map_err(to_py)?;
        let (_, geometries) =
            gfem_core::assemble::assemble_load(&mesh.inner, &self.inner, &lookup).map_err(to_py)?;
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
            // GPU を使うかはここで決める。gfem-core は CUDA を知らない（SPEC §7）
            let (solver, note) = cuda::pick(
                case.solver.backend,
                case.solver.preconditioner,
                mesh.nodes.len(),
                None,
            );
            // 外表面と電極面の索引は Mesh が作り置きしたものを使う。何度解いても
            // 抽出し直さない（大規模モデルでは solve の準備の大半を占める）
            let analysis = py_mesh.lookup().and_then(|lookup| {
                gfem_core::analysis::solve_prepared(&lookup, case, Some(&mut cb), solver.as_deref())
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

// --- 解析結果 ---------------------------------------------------------------

/// 解析結果（SPEC §5）。
#[pyclass(name = "Solution", module = "gfem._gfem", frozen)]
pub struct PySolution {
    mesh: Py<PyMesh>,
    /// 解いたときのケース。領域統計や実行サマリに σ と領域名が要る
    case: gfem_core::Case,
    inner: Analysis,
    /// 領域統計（初回だけ求める）。表・CSV・実行サマリで使い回す。
    /// 大規模モデルでは領域ごとのソートに数百ミリ秒かかる
    stats: OnceLock<Vec<RegionStats>>,
}

impl PySolution {
    /// 領域統計（初回だけ求める）。
    fn stats(&self) -> PyResult<&[RegionStats]> {
        lazy(&self.stats, || {
            self.inner.region_stats(&self.mesh.get().inner, &self.case)
        })
        .map(Vec::as_slice)
        .map_err(to_py)
    }

    /// 実行サマリに添えるプロビナンス（SPEC §5.5）。
    ///
    /// - Git の状態は gfem-core 単体ではなく、**この拡張モジュール全体**
    ///   （gfem-core・gfem-cuda・gfem-py）を作ったソースのもの（`build.rs`）
    /// - メッシュのハッシュは**読み込んだときに取ったもの**。いまファイルを読み直すと、
    ///   読み込み後に書き換えられていた場合に、解いていないメッシュのハッシュになる
    /// - 開発環境では、いまの作業ツリーの状態も添える（`gfem._provenance`）
    fn provenance(
        &self,
        py: Python<'_>,
        case_text: Option<String>,
        case_file: Option<PathBuf>,
    ) -> PyResult<Provenance> {
        let mesh = self.mesh.get();
        let mut provenance = Provenance::capture().with_git_state(
            option_env!("GFEM_GIT_COMMIT"),
            option_env!("GFEM_GIT_DIRTY"),
            option_env!("GFEM_GIT_SCOPE"),
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

    /// 電力密度 σ|E|² [W/m³]、`(n_elements,)`。
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
        let list = PyList::empty(py);
        for s in self.stats()? {
            list.append(region_stats_dict(py, s)?)?;
        }
        Ok(list.unbind())
    }

    /// 領域統計を CSV に書き出す（SPEC §5.5）。
    fn write_region_stats_csv(&self, path: PathBuf) -> PyResult<()> {
        gfem_core::post::write_region_stats_csv(&path, self.stats()?).map_err(to_py)
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
        let surface =
            gfem_core::mesh::iso_surface(mesh, tets, &values, level, &[&values]).map_err(to_py)?;
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
                    return Err(GfemError::new_err(format!(
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

/// 実行時の作業ツリー（`gfem._provenance.source_tree()`）。開発環境でなければ `None`。
fn source_tree(py: Python<'_>) -> PyResult<Option<SourceTree>> {
    let value = py.import("gfem._provenance")?.call_method0("source_tree")?;
    let tree: Option<(String, Option<String>, Option<bool>)> = value.extract()?;
    Ok(tree.map(|(path, git_commit, git_dirty)| SourceTree {
        path,
        git_commit,
        git_dirty,
    }))
}

fn parse_role(role: &str) -> PyResult<ElectrodeRole> {
    match role.to_ascii_lowercase().as_str() {
        "anode" => Ok(ElectrodeRole::Anode),
        "cathode" => Ok(ElectrodeRole::Cathode),
        other => Err(GfemError::new_err(format!(
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
fn build_electrode(
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
    let to_refs = |v: Option<Vec<(u32, u8)>>, what: &str| -> PyResult<Vec<FaceRef>> {
        v.unwrap_or_default()
            .into_iter()
            .map(|(eid, local)| {
                if local == 0 {
                    return Err(GfemError::new_err(format!(
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
        added_faces: to_refs(added, "手で足した面")?,
        removed_faces: to_refs(removed, "手で外した面")?,
    })
}

/// 体表面パッチの指定（辞書）を解釈する。
///
/// エラーは「どの電極の何番目のパッチが」「何が足りないか」まで出す
/// （CLAUDE.md 設計原則 6）。
fn parse_patch(obj: &Bound<'_, PyAny>, electrode: &str, index: usize) -> PyResult<SurfacePatch> {
    let where_ = format!("電極 {electrode} のパッチ {index}");
    let get = |key: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
        match obj.get_item(key) {
            Ok(v) => Ok(if v.is_none() { None } else { Some(v) }),
            Err(_) => Ok(None),
        }
    };
    let center = get("center")?.ok_or_else(|| {
        GfemError::new_err(format!(
            "{where_}に center がありません。\
             {{\"center\": [x, y, z], \"radius\": r, \"pid\": 51}} の形で指定してください（単位は m）"
        ))
    })?;
    let center: Vec<f64> = center.extract().map_err(|_| {
        GfemError::new_err(format!(
            "{where_}の center は 3 個の実数 [x, y, z] で指定してください"
        ))
    })?;
    if center.len() != 3 {
        return Err(GfemError::new_err(format!(
            "{where_}の center の要素数が {} 個です（3 個で指定してください）",
            center.len()
        )));
    }
    let radius: f64 = get("radius")?
        .ok_or_else(|| GfemError::new_err(format!("{where_}に radius がありません（単位は m）")))?
        .extract()
        .map_err(|_| GfemError::new_err(format!("{where_}の radius は実数で指定してください")))?;
    let pid: Option<u32> =
        match get("pid")? {
            Some(v) => Some(v.extract().map_err(|_| {
                GfemError::new_err(format!("{where_}の pid は整数で指定してください"))
            })?),
            None => None,
        };
    Ok(SurfacePatch {
        center: [center[0], center[1], center[2]],
        radius,
        pid,
    })
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

/// 断面・クリップを Python へ返す形にする（[`PyMesh::slice`] の説明を参照）。
fn section_dict(py: Python<'_>, section: TriangleMesh) -> PyResult<Py<PyDict>> {
    let (n, m) = (section.n_points(), section.n_triangles());
    let d = PyDict::new(py);
    d.set_item("points", section.points.into_pyarray(py).reshape([n, 3])?)?;
    d.set_item(
        "triangles",
        section.triangles.into_pyarray(py).reshape([m, 3])?,
    )?;
    d.set_item("elements", section.source_elements.into_pyarray(py))?;
    let edges: Vec<u32> = section.point_edges.into_iter().flatten().collect();
    d.set_item("edges", edges.into_pyarray(py).reshape([n, 2])?)?;
    d.set_item("t", section.point_t.into_pyarray(py))?;
    Ok(d.unbind())
}

/// `cell` が空なら `build` で作って入れ、中身を返す（`OnceLock::get_or_try_init` の代わり）。
///
/// 作るのに失敗したら何も入れずにエラーを返すので、次の呼び出しでまた作り直す。
/// 2 つのスレッドが同時に作り始めたら、先に入れたほうが残る（どちらも同じものを作る）。
fn lazy<T, E>(cell: &OnceLock<T>, build: impl FnOnce() -> Result<T, E>) -> Result<&T, E> {
    if let Some(value) = cell.get() {
        return Ok(value);
    }
    let built = build()?;
    Ok(cell.get_or_init(|| built))
}

/// Nastran バルクデータ（.bdf / .nas）を読み込む。
///
/// `length_unit` はファイル中の座標値の単位（`"mm"` / `"cm"` / `"m"` / `"inch"`）。
/// Nastran ファイルには単位情報が無いため**必ず指定する**（SPEC §3.2）。既定値は
/// 置かない——黙って mm とみなすと、m で書かれたモデルが 1000 倍に膨らんだまま、
/// もっともらしい絵が出てしまう。読み込んだ座標は内部で SI（m）に換算される。
#[pyfunction]
#[pyo3(signature = (path, length_unit = None))]
fn read_nastran(py: Python<'_>, path: PathBuf, length_unit: Option<&str>) -> PyResult<PyMesh> {
    let Some(length_unit) = length_unit else {
        return Err(to_py(
            gfem_core::GfemError::new(
                gfem_core::ErrorKind::Case,
                "長さ単位が指定されていません",
            )
            .with_file(&path)
            .with_hint(
                "read_nastran(path, length_unit=\"mm\") のように mm / cm / m / inch を指定して\
                 ください。Nastran ファイルには単位情報が無いので、既定値は置いていません（SPEC §3.2）",
            ),
        ));
    };
    let unit: LengthUnit = LengthUnit::parse(length_unit).map_err(to_py)?;
    let (mesh, report) = py
        .detach(|| gfem_core::read_nastran(&path, unit))
        .map_err(to_py)?;
    Ok(PyMesh::new(mesh, report))
}

/// 対応する長さ単位の一覧。
#[pyfunction]
fn length_units() -> Vec<&'static str> {
    LengthUnit::ALL.iter().map(|u| u.symbol()).collect()
}

/// 長さ単位 `unit` の 1 が何メートルか（`"mm"` なら 0.001）。
///
/// 換算表は Rust 側（`gfem_core::units`）の 1 か所だけに置き、Python はここから引く。
#[pyfunction]
fn metres_per_unit(unit: &str) -> PyResult<f64> {
    Ok(LengthUnit::parse(unit).map_err(to_py)?.to_meters())
}

/// 指定できる計算バックエンド（既定が先頭）。別名（`"gpu"` など）も解釈はできる。
#[pyfunction]
fn backend_choices() -> Vec<&'static str> {
    Backend::CHOICES.to_vec()
}

/// 指定できる前処理（既定が先頭）。別名（`"diagonal"` など）も解釈はできる。
#[pyfunction]
fn preconditioner_choices() -> Vec<&'static str> {
    Preconditioner::CHOICES.to_vec()
}

/// バックエンド名を解釈して記録用の名前を返す。解釈できなければ `GfemError`。
///
/// 別名の表は Rust 側にしかないので、ケースファイルの検査もここを通す。
#[pyfunction]
fn parse_backend(name: &str) -> PyResult<&'static str> {
    Ok(Backend::parse(name).map_err(to_py)?.label())
}

/// 前処理の名前を解釈して記録用の名前を返す。解釈できなければ `GfemError`。
#[pyfunction]
fn parse_preconditioner(name: &str) -> PyResult<&'static str> {
    Ok(Preconditioner::parse(name).map_err(to_py)?.label())
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
/// 解けるなら `None`、駄目ならその理由（日本語）を返す。AMG は階層を VRAM に置くので、
/// 前処理によって要る VRAM が違う（`preconditioner` は `"jacobi"` / `"amg"` / `"none"`）。
#[pyfunction]
#[pyo3(signature = (n_nodes, nnz = None, preconditioner = "jacobi"))]
fn cuda_check(
    n_nodes: usize,
    nnz: Option<usize>,
    preconditioner: &str,
) -> PyResult<Option<String>> {
    let preconditioner = Preconditioner::parse(preconditioner).map_err(to_py)?;
    Ok(cuda::check(n_nodes, nnz, preconditioner))
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
    m.add_function(wrap_pyfunction!(metres_per_unit, m)?)?;
    m.add_function(wrap_pyfunction!(backend_choices, m)?)?;
    m.add_function(wrap_pyfunction!(preconditioner_choices, m)?)?;
    m.add_function(wrap_pyfunction!(parse_backend, m)?)?;
    m.add_function(wrap_pyfunction!(parse_preconditioner, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_devices, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_check, m)?)?;
    Ok(())
}
