//! `Mesh` と `ReadReport` の Python 向けの顔、Nastran の読み込み（SPEC §3.1 / §6.3）。

use crate::FacetSetFemError;
use crate::case::build_electrode;
use crate::convert::{lazy, to_py};
use crate::patch::{describe_patch, parse_patch_at};
use facetsetfem_core::case::{ElectrodeRole, FaceLookup, SurfaceIndex};
use facetsetfem_core::mesh::{Plane, TriangleMesh};
use facetsetfem_core::{LengthUnit, VtuOptions};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::OnceLock;

/// バウンディングボックスを Python 側に返すときの形。
pub(crate) type BoundingBox = ((f64, f64, f64), (f64, f64, f64));

/// `Mesh.region_surface` の戻り値: (三角形, 元のソリッド要素)。
pub(crate) type RegionSurface<'py> = (Bound<'py, PyArray2<u32>>, Bound<'py, PyArray1<u32>>);

/// シェル要素の三角形分割を返すときの形。`(三角形, PID, 要素 ID)`。
pub(crate) type ShellTriangles<'py> = (
    Bound<'py, PyArray2<u32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
);

// --- 読み込みレポート -------------------------------------------------------

/// 読み込みレポート（SPEC §6.2）。
#[pyclass(name = "ReadReport", module = "facetsetfem._core", frozen)]
pub struct PyReadReport {
    pub(crate) inner: facetsetfem_core::ReadReport,
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
#[pyclass(name = "Mesh", module = "facetsetfem._core", frozen)]
pub struct PyMesh {
    pub(crate) inner: facetsetfem_core::Mesh,
    pub(crate) report: facetsetfem_core::ReadReport,
    /// 面共役な四面体分割（SPEC §5.1）。等値面と流線が使う。
    ///
    /// メッシュだけで決まるので、呼ばれるたびに作り直さず 1 度だけ作って使い回す。
    /// 使わなければ作られない。
    pub(crate) tets: OnceLock<facetsetfem_core::mesh::TetView>,
    /// 流線の下ごしらえ（四面体分割 + 面隣接 + 空間分割）
    pub(crate) navigator: OnceLock<facetsetfem_core::mesh::Navigator>,
    /// 外表面（SPEC §4.1）。
    ///
    /// 抽出は 4 百万四面体で 3.8 秒かかる。表示にも電極のパッチ選択にも使うので、
    /// 1 度だけ作って使い回す
    pub(crate) surface: OnceLock<facetsetfem_core::mesh::Surface>,
    /// 外表面から電極面を引く索引（パッチ選択と面の識別子の逆引き。SPEC §11.1）。
    ///
    /// solve・検証・クリックのたびに作り直さないよう、ここで 1 つ持って貸す
    /// （[`FaceLookup::with_index`]）。以前は毎回作り直していた
    pub(crate) surface_index: SurfaceIndex,
}

impl PyMesh {
    /// 読み込んだメッシュを包む。キャッシュは使われたときに作る。
    pub(crate) fn new(inner: facetsetfem_core::Mesh, report: facetsetfem_core::ReadReport) -> Self {
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
    pub(crate) fn tets(&self) -> PyResult<&facetsetfem_core::mesh::TetView> {
        lazy(&self.tets, || {
            facetsetfem_core::mesh::TetView::build(&self.inner)
        })
        .map_err(to_py)
    }

    /// 流線の下ごしらえ（初回だけ作る）。
    pub(crate) fn navigator(&self) -> PyResult<&facetsetfem_core::mesh::Navigator> {
        lazy(&self.navigator, || {
            facetsetfem_core::mesh::Navigator::build(&self.inner)
        })
        .map_err(to_py)
    }

    /// 外表面（初回だけ抽出する）。GIL を手放した中からも呼べるよう、コアのエラーで返す。
    pub(crate) fn surface_core(
        &self,
    ) -> facetsetfem_core::Result<&facetsetfem_core::mesh::Surface> {
        lazy(&self.surface, || {
            facetsetfem_core::mesh::extract_surface(&self.inner)
        })
    }

    /// 外表面（初回だけ抽出する）。
    pub(crate) fn surface(&self) -> PyResult<&facetsetfem_core::mesh::Surface> {
        self.surface_core().map_err(to_py)
    }

    /// 体表面から面を選ぶ索引（初回だけ作る）。
    pub(crate) fn picker(&self) -> PyResult<&facetsetfem_core::mesh::SurfacePicker> {
        Ok(self.surface_index.picker(&self.inner, self.surface()?))
    }

    /// 電極面を引く下ごしらえ。外表面と索引は作り置きのものを貸す。
    pub(crate) fn lookup(&self) -> facetsetfem_core::Result<FaceLookup<'_>> {
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
    pub(crate) fn describe_faces<'py>(
        &self,
        py: Python<'py>,
        faces: &[u32],
    ) -> PyResult<Bound<'py, PyDict>> {
        let surface = self.surface()?;
        let picker = self.picker()?;
        let mut flat: Vec<u32> = Vec::new();
        let mut refs: Vec<i64> = Vec::with_capacity(faces.len() * 2);
        for &f in faces {
            let face = &surface.faces()[f as usize];
            // 体表面の表示（`surface_triangles`）と同じ割り方にする。2 次要素の面は 4 枚
            let (pieces, n) = face.display_triangles(&self.inner);
            for tri in &pieces[..n] {
                flat.extend_from_slice(tri);
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

    /// 1 次の四面体メッシュの各辺の中点に節点を足し、2 次（10 節点の CTETRA）にしたもの
    /// （SPEC §2.3。要素の形は 1 次のまま（辺は真っ直ぐ）で、電位の近似だけを 2 次にする）。
    ///
    /// 要素の ID・PID・面番号は変わらないので、電極の指定はそのまま使える。
    /// 読み込みレポートは元のものを引き継ぐ。
    fn elevate(&self, py: Python<'_>) -> PyResult<PyMesh> {
        let mesh = py
            .detach(|| facetsetfem_core::mesh::elevate(&self.inner))
            .map_err(to_py)?;
        Ok(PyMesh::new(mesh, self.report.clone()))
    }

    /// 要素の次数（2 次の要素があれば 2、なければ 1）。
    #[getter]
    fn element_order(&self) -> u32 {
        if self.inner.solids.iter().any(|b| b.kind().is_quadratic()) {
            2
        } else {
            1
        }
    }

    /// 節点が要素の頂点か（辺の中点なら False）、`(n_nodes,)`。
    ///
    /// 2 次要素の表示で、外表面を 4 分割した三角形の辺のうち、元の要素の辺
    /// （頂点と中点を結ぶもの）だけを線で描くのに使う。1 次要素のメッシュでは全て True。
    fn corner_node_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        let mut mask = vec![false; self.inner.nodes.len()];
        for e in self.inner.iter_solids() {
            for &node in &e.conn[..e.kind.corner_count()] {
                mask[node as usize] = true;
            }
        }
        mask.into_pyarray(py)
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
    /// 10 節点の CTETRA は `"CTETRA(10)"` と書いて 4 節点のものと区別する。
    fn element_counts(&self) -> Vec<(&'static str, usize)> {
        self.inner
            .blocks()
            .map(|b| (b.kind().label(), b.len()))
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
        let (tris, _) = self.surface()?.triangulate(&self.inner);
        let flat: Vec<u32> = tris.into_iter().flatten().collect();
        let rows = flat.len() / 3;
        flat.into_pyarray(py).reshape([rows, 3])
    }

    /// 全領域の境界面 `(三角形 (n, 3), 元のソリッド要素 (n,))`（SPEC §6.3）。
    ///
    /// 外表面と、隣り合う要素の PID が違う面を両側とも。元の要素の PID で振り分ければ、
    /// 領域ごとの `region_surface` を全領域ぶん呼んだのと同じ面になる（1 回の並べ替えで済む）。
    /// 領域ごとに表示・半透明・非表示を選ぶ描画に使う。
    fn region_boundaries<'py>(&self, py: Python<'py>) -> PyResult<RegionSurface<'py>> {
        let surface = py
            .detach(|| facetsetfem_core::mesh::extract_region_boundaries(&self.inner))
            .map_err(to_py)?;
        let (tris, owners) = surface.triangulate(&self.inner);
        let flat: Vec<u32> = tris.into_iter().flatten().collect();
        let rows = flat.len() / 3;
        Ok((
            flat.into_pyarray(py).reshape([rows, 3])?,
            owners.into_pyarray(py),
        ))
    }

    /// 指定した領域（PID）の要素の集まりの境界面 `(三角形 (n, 3), 元のソリッド要素 (n,))`。
    ///
    /// 体表面に出ていない領域をハイライトするために使う（SPEC §6.3）。体積メッシュを
    /// 組み立てずに、Rust 側で領域の境界だけを取り出す。向きは領域の外向き。
    fn region_surface<'py>(&self, py: Python<'py>, pid: u32) -> PyResult<RegionSurface<'py>> {
        let surface = py
            .detach(|| facetsetfem_core::mesh::extract_region_surface(&self.inner, pid))
            .map_err(to_py)?;
        let (tris, owners) = surface.triangulate(&self.inner);
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
        let (_, owners) = self.surface()?.triangulate(&self.inner);
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
        let section = py.detach(|| facetsetfem_core::mesh::slice(&self.inner, tets, &plane));
        section_dict(py, section)
    }

    /// 平面の**法線の向きの側を切り落とした**クリップの見た目（SPEC §6.3）。
    ///
    /// 外表面のうち残す側と、切り口を合わせた三角形。戻り値の形は [`Self::slice`] と同じ。
    fn clip(&self, py: Python<'_>, origin: [f64; 3], normal: [f64; 3]) -> PyResult<Py<PyDict>> {
        let plane = Plane::new(origin, normal).map_err(to_py)?;
        let tets = self.tets()?;
        let surface = self.surface()?;
        let section =
            py.detach(|| facetsetfem_core::mesh::clip(&self.inner, tets, surface, &plane));
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

    /// 体表面の 1 点を中心に、形の中でつながっている面を選ぶ（SPEC §11.1）。
    ///
    /// GUI の 3D ビューでクリックした点をそのまま渡す。形は円（`radius`）か、
    /// `shape="rect"` と `width` / `height` / `corner_radius` / `axis` / `angle` の矩形。
    /// 指定の仕方はケースファイルのパッチと同じ（同じ関数で解釈する）。
    ///
    /// 戻り値は `{"n_faces", "area", "nominal_area", "triangles", "face_refs",
    /// "outline", "normal", "center", "pid", "shape", ...形の寸法}`。
    /// `triangles` は選ばれた面を三角形に割った `(n, 3)` の節点索引（強調表示用）、
    /// `outline` は形の輪郭の `(m, 3)` の座標（接平面上。表示用）、`nominal_area` は
    /// 指定どおりの面積。実際に選ばれた面積 `area` との差がメッシュの粗さの目安になる。
    ///
    /// **球に入っているかだけでは選ばない。** クリックした面から稜線でつながって
    /// いる面だけを辿るので、首の切断面や耳の裏へ飛び移らない。
    #[pyo3(signature = (center, radius = None, pid = None, **shape))]
    fn pick_patch(
        &self,
        py: Python<'_>,
        center: Bound<'_, PyAny>,
        radius: Option<f64>,
        pid: Option<u32>,
        shape: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDict>> {
        let spec = PyDict::new(py);
        if let Some(shape) = shape {
            spec.update(shape.as_mapping())?;
        }
        spec.set_item("center", center)?;
        if let Some(r) = radius {
            spec.set_item("radius", r)?;
        }
        if let Some(p) = pid {
            spec.set_item("pid", p)?;
        }
        // 形と中心（ピックに失敗した NaN を含む）の検証は、電極面の解決と同じ関数で行う
        let patch = parse_patch_at(spec.as_any(), &facetsetfem_core::tr!("Patch", "パッチ"))?;
        let picker = self.picker()?;
        let faces = patch.select(picker).map_err(to_py)?;
        let d = self.describe_faces(py, &faces)?;
        describe_patch(&d, &patch)?;
        match patch.outline(picker).map_err(to_py)? {
            Some((frame, points)) => {
                let rows = points.len();
                let flat: Vec<f64> = points.into_iter().flatten().collect();
                d.set_item("outline", flat.into_pyarray(py).reshape([rows, 3])?)?;
                d.set_item("normal", frame.normal.to_vec())?;
            }
            None => {
                d.set_item("outline", py.None())?;
                d.set_item("normal", py.None())?;
            }
        }
        Ok(d.unbind())
    }

    /// 電極の面指定を解決して、面数・面積・三角形・面の識別子を返す。
    ///
    /// **ソルバが荷重を作るときと同じ関数**（`facetsetfem_core::case::resolve_faces`）を通す。
    /// GUI の表示だけ別に計算すると、「25 cm² のつもりで置いた電極が実は違う面積
    /// だった」が静かに起きるため（SPEC §6.2 の手順 3）。導電率が未設定でも呼べる。
    ///
    /// `added` / `removed` は `(EID, 面番号)` の並び。**面番号は 1 始まり**。
    #[pyo3(signature = (
        shell_pids = None, element_ids = None, patches = None, added = None, removed = None,
        name = None
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
        name: Option<&str>,
    ) -> PyResult<Py<PyDict>> {
        // 名前はエラーの中で電極を指すためだけに使う（省略時の呼び名は表示言語で）
        let unnamed = facetsetfem_core::tr!("(unnamed)", "（名前なし）");
        let electrode = build_electrode(
            name.unwrap_or(&unnamed),
            ElectrodeRole::Anode,
            1.0,
            shell_pids,
            element_ids,
            patches,
            added,
            removed,
        )?;
        let lookup = self.lookup().map_err(to_py)?;
        let faces = facetsetfem_core::case::resolve_faces(&lookup, &electrode).map_err(to_py)?;

        // 体表面の面だけ強調表示できる。シェル要素は従来どおり別経路で描く
        let surface_faces: Vec<u32> = faces
            .iter()
            .filter_map(|f| match f {
                facetsetfem_core::case::ElectrodeFace::Surface(i) => Some(*i),
                facetsetfem_core::case::ElectrodeFace::Shell(_) => None,
            })
            .collect();
        let d = self.describe_faces(py, &surface_faces)?;
        // 面数と面積はシェルを含めた全体で出す（ソルバが使う値と一致させる）
        let mut area = 0.0;
        for &face in &faces {
            area += lookup.face_trace(face).integral(&self.inner).area;
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
            .filter(|b| b.kind() != facetsetfem_core::ElementKind::Tet4)
            .map(|b| b.kind().card_name())
            .collect();
        if !others.is_empty() {
            return Err(FacetSetFemError::new_err(facetsetfem_core::tr!(
                "[Unsupported] There are solid elements other than tetrahedra: {}\n\
                 Fix: tetrahedra() returns CTETRA only. \
                 For a model with mixed element types, use solid_cells()",
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
        py.detach(|| {
            facetsetfem_core::write_mesh_vtu(&path, &self.inner, &VtuOptions { compress, level })
        })
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

/// 断面・クリップを Python へ返す形にする（[`PyMesh::slice`] の説明を参照）。
pub(crate) fn section_dict(py: Python<'_>, section: TriangleMesh) -> PyResult<Py<PyDict>> {
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

/// Nastran バルクデータ（.bdf / .nas）を読み込む。
///
/// `length_unit` はファイル中の座標値の単位（`"mm"` / `"cm"` / `"m"` / `"inch"`）。
/// Nastran ファイルには単位情報が無いため**必ず指定する**（SPEC §3.2）。既定値は
/// 置かない——黙って mm とみなすと、m で書かれたモデルが 1000 倍に膨らんだまま、
/// もっともらしい絵が出てしまう。読み込んだ座標は内部で SI（m）に換算される。
#[pyfunction]
#[pyo3(signature = (path, length_unit = None))]
pub(crate) fn read_nastran(
    py: Python<'_>,
    path: PathBuf,
    length_unit: Option<&str>,
) -> PyResult<PyMesh> {
    let Some(length_unit) = length_unit else {
        return Err(to_py(
            facetsetfem_core::FacetSetFemError::new(
                facetsetfem_core::ErrorKind::Case,
                facetsetfem_core::tr!("No length unit was given", "長さ単位が指定されていません"),
            )
            .with_file(&path)
            .with_hint(facetsetfem_core::tr!(
                "Give mm / cm / m / inch, as in read_nastran(path, length_unit=\"mm\"). Nastran \
                 files carry no unit information, so there is no default (SPEC §3.2)",
                "read_nastran(path, length_unit=\"mm\") のように mm / cm / m / inch を指定して\
                 ください。Nastran ファイルには単位情報が無いので、既定値は置いていません（SPEC §3.2）"
            )),
        ));
    };
    let unit: LengthUnit = LengthUnit::parse(length_unit).map_err(to_py)?;
    let (mesh, report) = py
        .detach(|| facetsetfem_core::read_nastran(&path, unit))
        .map_err(to_py)?;
    Ok(PyMesh::new(mesh, report))
}
