//! 解析ケースの定義と、実行前の整合性チェック（SPEC §3.3 / §6.2）。
//!
//! Nastran ファイルは形状と領域分けだけを担い、解析条件はこちらが持つ。
//! Nastran ファイルは書き換えない。
//!
//! YAML の読み書きは Python 側（PyYAML）で行い、ここはプログラム的に組み立てる形にする。
//! `gfem-core` に YAML パーサを入れると依存が増えるうえ、行番号つきの親切なエラー
//! （CLAUDE.md 設計原則 6）を Rust と Python の二重に作ることになるため。

use crate::error::{ErrorKind, GfemError, Result, format_sample_list};
use crate::material::Conductivity;
use crate::mesh::pick::SurfacePicker;
use crate::mesh::topology::{FaceKey, Surface, face_key};
use crate::mesh::{ElemId, Mesh, NodeId, NodeIndex, PropId};
use crate::solver::amg::AmgOptions;
use std::collections::{BTreeMap, BTreeSet};

/// エラー一覧に並べる最大件数。
const SAMPLE_LIMIT: usize = 20;

/// 電流保存（`1ᵀf = 0`）を満たしているとみなす相対しきい値（SPEC §2.2）。
pub const CURRENT_BALANCE_TOLERANCE: f64 = 1.0e-12;

/// 電極の役割。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElectrodeRole {
    /// 陽極。電流が流入する
    Anode,
    /// 陰極。電流が流出する
    Cathode,
}

impl ElectrodeRole {
    /// 総電流に掛ける符号。流入を正とする。
    pub const fn sign(self) -> f64 {
        match self {
            ElectrodeRole::Anode => 1.0,
            ElectrodeRole::Cathode => -1.0,
        }
    }

    /// 表示用の名前。
    pub const fn label(self) -> &'static str {
        match self {
            ElectrodeRole::Anode => "anode",
            ElectrodeRole::Cathode => "cathode",
        }
    }
}

/// 体表面に置く電極パッチ 1 枚（SPEC §3.3 / §11.1）。
///
/// **シェル要素を持たないモデルのための指定方法。** 人体モデルはソリッド要素だけで
/// 配られるのが普通で、その場合シェル PID では電極を 1 つも置けない。外表面は
/// ソリッドの面から抽出できているので、そこから直接選ぶ。
///
/// GUI では 3D ビュー上のクリックが `center` を、スピンボックスが `radius` を決める。
/// **保存するのは面の一覧ではなく、この中心と半径**。同じメッシュなら同じ面が選ばれ、
/// ケースファイルを読んでも人が意味を追えるため（面番号の羅列は追えない）。
#[derive(Debug, Clone, PartialEq)]
pub struct SurfacePatch {
    /// 中心 [m]。3D ビューでクリックした点
    pub center: [f64; 3],
    /// 半径 [m]。中心からこの距離までの面を、体表面伝いに広げて選ぶ
    pub radius: f64,
    /// この PID のソリッドが持つ面だけに限る（`None` なら制限しない）
    ///
    /// 首の切断面のように「外表面だが体表面ではない」面を避けるために使う。
    pub pid: Option<PropId>,
}

/// 体表面の面 1 枚の指し方（SPEC §11.1）。
///
/// **内部の通し番号ではなく Nastran の識別子で持つ。** ソリッド要素の EID と、
/// その要素内での面番号（0 始まり）なら、読み込み順や内部の索引づけが変わっても
/// 壊れず、ケースファイルを人が読んでも意味を追える。
///
/// ケースファイル（YAML）とエラーメッセージでは**面番号を 1 始まり**で表す
/// （「CTETRA EID=123 の第 2 面」）。0 始まりなのはこの構造体の中だけ。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FaceRef {
    /// ソリッド要素の EID
    pub element_id: ElemId,
    /// 要素内での面番号（0 始まり）
    pub local_face: u8,
}

impl FaceRef {
    /// 0 始まりの面番号から作る。
    pub fn new(element_id: ElemId, local_face: u8) -> Self {
        Self {
            element_id,
            local_face,
        }
    }
}

/// 電極 1 個。
///
/// 面は「シェル PID の集合」「個別のシェル要素 ID」「体表面のパッチ」の**組み合わせ**で
/// 指定し、さらに**手で足した面・外した面**で調整する（SPEC §3.3 / §11.1）。
///
/// 手編集を面の一覧そのものではなく**差分**で持つのは、パッチの「中心と半径」という
/// 読める指定を残したまま縁だけ直せるようにするため。実測で、半径 25 mm の電極は
/// 923 面あり、解決後の一覧を書き出すと 1 電極で 6〜12 KB になる。
#[derive(Debug, Clone)]
pub struct Electrode {
    /// 表示名
    pub name: String,
    /// 陽極 / 陰極
    pub role: ElectrodeRole,
    /// 総電流 [A]（正の値）
    pub current: f64,
    /// 電極面に含めるシェル PID
    pub shell_pids: Vec<PropId>,
    /// 個別に追加するシェル要素 ID
    pub element_ids: Vec<ElemId>,
    /// 体表面から直接選んだパッチ
    pub patches: Vec<SurfacePatch>,
    /// 手で足した体表面の面
    pub added_faces: Vec<FaceRef>,
    /// 手で外した体表面の面。パッチや `added_faces` より優先する
    pub removed_faces: Vec<FaceRef>,
}

impl Electrode {
    /// 陽極を作る。
    pub fn anode(name: impl Into<String>, current: f64) -> Self {
        Self::new(name, ElectrodeRole::Anode, current)
    }

    /// 陰極を作る。
    pub fn cathode(name: impl Into<String>, current: f64) -> Self {
        Self::new(name, ElectrodeRole::Cathode, current)
    }

    fn new(name: impl Into<String>, role: ElectrodeRole, current: f64) -> Self {
        Self {
            name: name.into(),
            role,
            current,
            shell_pids: Vec::new(),
            element_ids: Vec::new(),
            patches: Vec::new(),
            added_faces: Vec::new(),
            removed_faces: Vec::new(),
        }
    }

    /// シェル PID を面として追加する。
    pub fn with_shell_pids(mut self, pids: impl IntoIterator<Item = PropId>) -> Self {
        self.shell_pids.extend(pids);
        self
    }

    /// シェル要素 ID を面として追加する。
    pub fn with_element_ids(mut self, ids: impl IntoIterator<Item = ElemId>) -> Self {
        self.element_ids.extend(ids);
        self
    }

    /// 体表面のパッチを面として追加する（SPEC §11.1）。
    pub fn with_patches(mut self, patches: impl IntoIterator<Item = SurfacePatch>) -> Self {
        self.patches.extend(patches);
        self
    }

    /// 手で足した面。
    pub fn with_added_faces(mut self, faces: impl IntoIterator<Item = FaceRef>) -> Self {
        self.added_faces.extend(faces);
        self
    }

    /// 手で外した面。
    pub fn with_removed_faces(mut self, faces: impl IntoIterator<Item = FaceRef>) -> Self {
        self.removed_faces.extend(faces);
        self
    }

    /// 流入電流（陰極は負）。
    pub fn signed_current(&self) -> f64 {
        self.role.sign() * self.current
    }
}

/// 結果表示に使う電位の基準（SPEC §2.2 / §3.3）。
///
/// E, J は基準に依存しないので、ここを変えても物理量は変わらない。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReferencePotential {
    /// 全節点電位の平均をゼロにする（ソルバの射影と同じ代表元）
    #[default]
    ZeroMean,
    /// 陰極面の面積重み平均電位をゼロにする
    CathodeMean,
    /// 指定した節点の電位をゼロにする
    Node(NodeId),
}

/// 前処理（SPEC §4.3 / §4.5）。
///
/// **どれを選んでも収束後の答えは同じ**（同じ方程式を解いている）。変わるのは
/// 反復回数と 1 反復のコストだけ。ただしバックエンドの違いと違って、
/// 前処理を変えると反復の道筋が変わるので**ビット列は一致しない**——
/// 収束判定の範囲内で一致する。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Preconditioner {
    /// 前処理なし
    None,
    /// 対角スケーリング
    Jacobi,
    /// 代数マルチグリッド（平滑化凝集。SPEC §4.5）
    ///
    /// 反復回数がメッシュの細かさにほとんど依存しなくなる。1 反復は重くなるので、
    /// 小さい問題や導電率が一様な問題では Jacobi のほうが速い。
    Amg,
}

impl Preconditioner {
    /// 表示・記録用の名前。
    pub const fn label(self) -> &'static str {
        match self {
            Preconditioner::None => "none",
            Preconditioner::Jacobi => "jacobi",
            Preconditioner::Amg => "amg",
        }
    }

    /// 文字列から解釈する。
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Preconditioner::None),
            "jacobi" | "diagonal" => Ok(Preconditioner::Jacobi),
            "amg" | "smoothed_aggregation" | "sa" => Ok(Preconditioner::Amg),
            other => Err(
                GfemError::new(ErrorKind::Case, format!("未知の前処理です: \"{other}\""))
                    .with_hint("jacobi / amg / none のいずれかを指定してください"),
            ),
        }
    }
}

/// 計算バックエンド（SPEC §3.3 / §4.4）。
///
/// **どれを選んでも結果はビット単位で同じ**（CLAUDE.md 設計原則 1）。
/// 速度だけが変わる。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Backend {
    /// 規模を見て自動で選ぶ
    #[default]
    Auto,
    /// CPU 逐次
    CpuSerial,
    /// CPU 並列（rayon）
    CpuParallel,
    /// CUDA（`gfem-cuda` クレート）
    ///
    /// `gfem-core` は CUDA に依存しないので、この値を指定しただけでは GPU では動かない。
    /// 呼び出し側が [`crate::solve_with`] に `gfem-cuda` のソルバを渡す必要がある
    /// （SPEC §7 の設計原則）。渡されなければ理由つきのエラーになる。
    Cuda,
}

impl Backend {
    /// 表示・記録用の名前。
    pub const fn label(self) -> &'static str {
        match self {
            Backend::Auto => "auto",
            Backend::CpuSerial => "cpu_serial",
            Backend::CpuParallel => "cpu_parallel",
            Backend::Cuda => "cuda",
        }
    }

    /// 文字列から解釈する。
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Backend::Auto),
            "cpu" | "cpu_serial" | "serial" => Ok(Backend::CpuSerial),
            "cpu_parallel" | "parallel" => Ok(Backend::CpuParallel),
            "cuda" | "gpu" => Ok(Backend::Cuda),
            other => Err(GfemError::new(
                ErrorKind::Case,
                format!("このビルドでは使えないバックエンドです: \"{other}\""),
            )
            .with_hint("auto / cpu / cpu_parallel / cuda のいずれかを指定してください")),
        }
    }

    /// 並列アセンブリと CPU 並列に切り替える節点数（実測の損益分岐点。SPEC §8 M4）。
    pub const PARALLEL_THRESHOLD: usize = 20_000;

    /// GPU に載せる価値が出る節点数（実測の損益分岐点。SPEC §8 M5）。
    ///
    /// これより小さいと、VRAM への転送とカーネル起動の往復が効いて、
    /// 20 コアの CPU 並列に負ける。実測では 15 万節点で CPU 並列が、
    /// 30 万節点で GPU が勝つので、その間に置いてある。
    pub const GPU_THRESHOLD: usize = 250_000;

    /// `Auto` を CPU のバックエンドに解決する。
    ///
    /// 小さい問題では並列化のオーバーヘッドの方が高くつくので、しきい値で切り替える。
    /// どちらでも結果は同じなので、この判断が答えを変えることはない。
    ///
    /// **`Auto` がここで `Cuda` を選ぶことはない。** `gfem-core` は GPU があるか
    /// どうかを知らないため、GPU を使う判断は呼び出し側（`gfem-py`）が行う。
    pub fn resolve(self, n_nodes: usize) -> Backend {
        match self {
            Backend::Auto => {
                if n_nodes >= Self::PARALLEL_THRESHOLD {
                    Backend::CpuParallel
                } else {
                    Backend::CpuSerial
                }
            }
            other => other,
        }
    }

    /// `Auto` のとき、この規模なら GPU を試す価値があるか。
    ///
    /// 判断材料を 1 箇所にまとめておくための小さな関数。実際に GPU が使えるかは
    /// `gfem-cuda` に聞く。
    pub fn wants_gpu(self, n_nodes: usize) -> bool {
        match self {
            Backend::Cuda => true,
            Backend::Auto => n_nodes >= Self::GPU_THRESHOLD,
            _ => false,
        }
    }
}

/// ソルバ設定（SPEC §3.3 の `solver` 節）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverOptions {
    /// 相対残差の収束判定しきい値
    pub tolerance: f64,
    /// 最大反復回数
    pub max_iterations: usize,
    /// 前処理
    pub preconditioner: Preconditioner,
    /// AMG の設定（`preconditioner: amg` のときだけ使う）
    pub amg: AmgOptions,
    /// 計算バックエンド
    pub backend: Backend,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            tolerance: 1.0e-10,
            max_iterations: 50_000,
            preconditioner: Preconditioner::Jacobi,
            amg: AmgOptions::default(),
            backend: Backend::Auto,
        }
    }
}

/// 解析ケース。
#[derive(Debug, Clone, Default)]
pub struct Case {
    /// PID → 導電率。PSOLID の PID が領域の単位（SPEC §3.1）
    pub conductivity: BTreeMap<PropId, Conductivity>,
    /// PID → 領域名。計算には使わないが、領域統計（SPEC §5.2）や GUI の表示に出す
    pub region_names: BTreeMap<PropId, String>,
    /// 電極
    pub electrodes: Vec<Electrode>,
    /// ソルバ設定
    pub solver: SolverOptions,
    /// 結果表示の基準電位
    pub reference: ReferencePotential,
}

impl Case {
    /// 空のケース。
    pub fn new() -> Self {
        Self::default()
    }

    /// PID に導電率を割り当てる。
    pub fn set_conductivity(&mut self, pid: PropId, sigma: f64) -> Result<()> {
        self.conductivity
            .insert(pid, Conductivity::isotropic(sigma)?);
        Ok(())
    }

    /// PID に領域名を付ける（表示と統計出力のためだけに使う）。
    pub fn set_region_name(&mut self, pid: PropId, name: impl Into<String>) {
        self.region_names.insert(pid, name.into());
    }

    /// 電極を追加する。
    pub fn add_electrode(&mut self, electrode: Electrode) {
        self.electrodes.push(electrode);
    }

    /// 陽極の総電流 [A]。
    pub fn total_anode_current(&self) -> f64 {
        self.sum_current(ElectrodeRole::Anode)
    }

    /// 陰極の総電流 [A]。
    pub fn total_cathode_current(&self) -> f64 {
        self.sum_current(ElectrodeRole::Cathode)
    }

    fn sum_current(&self, role: ElectrodeRole) -> f64 {
        self.electrodes
            .iter()
            .filter(|e| e.role == role)
            .map(|e| e.current)
            .sum()
    }

    /// 実行前の整合性チェック（SPEC §6.2 の手順 4）。
    ///
    /// エラーは 1 件目で止めず、**同じ種類の不備はまとめて列挙する**。
    /// GUI でリスト表示して一度に直せるようにするため。
    /// 戻り値は「実行はできるが知らせるべきこと」の警告一覧。
    pub fn validate(&self, mesh: &Mesh, surface: &Surface) -> Result<Vec<String>> {
        let mut warnings = Vec::new();

        // --- 導電率 ---
        let region_pids = mesh.solid_pids();
        let missing: Vec<String> = region_pids
            .iter()
            .filter(|pid| !self.conductivity.contains_key(pid))
            .map(|pid| pid.to_string())
            .collect();
        if !missing.is_empty() {
            return Err(GfemError::new(
                ErrorKind::Case,
                format!(
                    "導電率が未設定の領域があります（{} 件）: PID={}",
                    missing.len(),
                    format_sample_list(&missing, SAMPLE_LIMIT)
                ),
            )
            .with_hint("全ての PSOLID の PID に σ [S/m] を割り当ててください"));
        }
        for pid in self.conductivity.keys() {
            if !region_pids.contains(pid) {
                warnings.push(format!(
                    "PID={pid} に導電率が設定されていますが、その PID のソリッド要素はありません"
                ));
            }
        }

        // --- 電極の有無 ---
        if self.electrodes.is_empty() {
            return Err(GfemError::new(ErrorKind::Case, "電極が定義されていません")
                .with_hint("陽極（anode）と陰極（cathode）を最低 1 つずつ定義してください"));
        }
        for role in [ElectrodeRole::Anode, ElectrodeRole::Cathode] {
            if !self.electrodes.iter().any(|e| e.role == role) {
                return Err(GfemError::new(
                    ErrorKind::Case,
                    format!("{} が定義されていません", role.label()),
                )
                .with_hint("電流の流入口と流出口の両方が必要です"));
            }
        }
        for e in &self.electrodes {
            if !e.current.is_finite() || e.current <= 0.0 {
                return Err(GfemError::new(
                    ErrorKind::Case,
                    format!(
                        "電極 {} の総電流が正の有限値ではありません: {} A",
                        e.name, e.current
                    ),
                )
                .with_hint(
                    "総電流は正の値で指定し、向きは role（anode / cathode）で表してください",
                ));
            }
        }

        // --- 電流保存（SPEC §2.2 の適合条件）---
        let anode = self.total_anode_current();
        let cathode = self.total_cathode_current();
        let scale = anode.max(cathode);
        if (anode - cathode).abs() > CURRENT_BALANCE_TOLERANCE * scale {
            return Err(GfemError::new(
                ErrorKind::Case,
                format!("総流入電流と総流出電流が一致しません: 陽極 {anode} A / 陰極 {cathode} A"),
            )
            .with_hint(
                "全 Neumann 境界の問題なので、Σ I_k = 0 でなければ解が存在しません（SPEC §2.2）。\
                 電極の電流値を見直してください",
            ));
        }

        // --- 電極面 ---
        let lookup = FaceLookup::new(mesh, surface);
        // 面の重なりは「体表面のどの面か」で見る。シェル要素とパッチが同じ面を
        // 指していても重複として捕まえられるようにするため
        let mut used_faces: BTreeMap<FaceKey, String> = BTreeMap::new();
        for e in &self.electrodes {
            let faces = self.resolve_faces(&lookup, e)?;
            let mut off_surface: Vec<String> = Vec::new();
            for &face in &faces {
                let (_, nodes) = lookup.face_nodes(face);
                if !surface.contains(nodes) {
                    // パッチは外表面から選ぶので、ここに来るのはシェル要素だけ
                    off_surface.push(lookup.label(face));
                }
                if let Some(other) = used_faces.insert(face_key(nodes), e.name.clone())
                    && other != e.name
                {
                    return Err(GfemError::new(
                        ErrorKind::Case,
                        format!(
                            "{} が電極 {other} と {} の両方に含まれています",
                            lookup.label(face),
                            e.name
                        ),
                    )
                    .with_hint("1 つの面を 2 つの電極に割り当てることはできません"));
                }
            }
            if !off_surface.is_empty() {
                // SPEC §11.1: 電極は必ず体表面にある
                return Err(GfemError::new(
                    ErrorKind::Case,
                    format!(
                        "電極 {} の面のうち {} 枚が体表面（外表面）に載っていません: {}",
                        e.name,
                        off_surface.len(),
                        format_sample_list(&off_surface, SAMPLE_LIMIT)
                    ),
                )
                .with_hint(
                    "経皮電気刺激では電極は必ず体表面にあります（SPEC §11.1）。\
                     内部面を指しているか、シェル要素の節点がソリッドの面と一致していません",
                ));
            }
        }

        // --- ソルバ設定 ---
        if !(self.solver.tolerance > 0.0 && self.solver.tolerance < 1.0) {
            return Err(GfemError::new(
                ErrorKind::Case,
                format!("収束判定しきい値が範囲外です: {}", self.solver.tolerance),
            )
            .with_hint("0 < tolerance < 1 の相対残差で指定してください（既定 1e-10）"));
        }
        if self.solver.max_iterations == 0 {
            return Err(GfemError::new(ErrorKind::Case, "最大反復回数が 0 です")
                .with_hint("1 以上を指定してください（既定 50000）"));
        }

        Ok(warnings)
    }

    /// 電極に属する面（昇順・重複なし）を求める。
    ///
    /// 中身は [`resolve_faces`]。ケースの設定には依存しないので、
    /// 表示のためだけに面を引きたいとき（導電率をまだ入れていない GUI など）は
    /// そちらを直接呼べる。
    pub fn resolve_faces(
        &self,
        lookup: &FaceLookup,
        electrode: &Electrode,
    ) -> Result<Vec<ElectrodeFace>> {
        resolve_faces(lookup, electrode)
    }
}

/// 電極に属する面（昇順・重複なし）を求める。
///
/// 通し番号順に並べることで、荷重ベクトルへの加算順序が入力の書き順に依存しなくなる
/// （CLAUDE.md 設計原則 1）。
///
/// 面の出どころは 4 つあり、この順で効く:
///
/// 1. シェル PID / シェル要素 ID
/// 2. 体表面のパッチ（中心・半径・PID）
/// 3. 手で足した面
/// 4. 手で外した面（**上の 3 つより優先する**）
pub fn resolve_faces(lookup: &FaceLookup, electrode: &Electrode) -> Result<Vec<ElectrodeFace>> {
    {
        let mesh = lookup.mesh;
        let wanted_pids: BTreeSet<PropId> = electrode.shell_pids.iter().copied().collect();
        let wanted_ids: BTreeSet<ElemId> = electrode.element_ids.iter().copied().collect();

        let mut faces: Vec<ElectrodeFace> = Vec::new();
        let mut seen_pids: BTreeSet<PropId> = BTreeSet::new();
        let mut seen_ids: BTreeSet<ElemId> = BTreeSet::new();
        for s in mesh.iter_shells() {
            let by_pid = wanted_pids.contains(&s.pid);
            let by_id = wanted_ids.contains(&s.id);
            if by_pid {
                seen_pids.insert(s.pid);
            }
            if by_id {
                seen_ids.insert(s.id);
            }
            if by_pid || by_id {
                faces.push(ElectrodeFace::Shell(s.index as u32));
            }
        }

        // --- 体表面のパッチ（SPEC §11.1）---
        for (k, patch) in electrode.patches.iter().enumerate() {
            if !patch.radius.is_finite() || patch.radius < 0.0 {
                return Err(GfemError::new(
                    ErrorKind::Case,
                    format!(
                        "電極 {} のパッチ {} の半径が不正です: {} m",
                        electrode.name,
                        k + 1,
                        patch.radius
                    ),
                )
                .with_hint("半径は 0 以上の有限値で指定してください（単位は m）"));
            }
            if patch.center.iter().any(|v| !v.is_finite()) {
                return Err(GfemError::new(
                    ErrorKind::Case,
                    format!(
                        "電極 {} のパッチ {} の中心が不正です: {:?}",
                        electrode.name,
                        k + 1,
                        patch.center
                    ),
                )
                .with_hint("中心は体表面上の点を m で指定してください"));
            }
            let selected = lookup
                .picker()
                .select(patch.center, patch.radius, patch.pid);
            if selected.is_empty() {
                return Err(GfemError::new(
                    ErrorKind::Case,
                    format!(
                        "電極 {} のパッチ {} が体表面の面を 1 枚も選べませんでした（中心 {:?} m, 半径 {} m{}）",
                        electrode.name,
                        k + 1,
                        patch.center,
                        patch.radius,
                        match patch.pid {
                            Some(p) => format!(", PID={p}"),
                            None => String::new(),
                        }
                    ),
                )
                .with_hint(
                    "PID で絞り込んでいる場合は、その領域が体表面に露出しているか確認してください\
                     （読み込みレポートの領域一覧と、3D ビューの外表面表示で分かります）",
                ));
            }
            faces.extend(selected.into_iter().map(ElectrodeFace::Surface));
        }

        // --- 手で足した面・外した面（SPEC §11.1）---
        for face in &electrode.added_faces {
            faces.push(ElectrodeFace::Surface(lookup.face_of(*face, electrode)?));
        }
        faces.sort_unstable();
        faces.dedup();
        if !electrode.removed_faces.is_empty() {
            let mut dropped: BTreeSet<u32> = BTreeSet::new();
            for face in &electrode.removed_faces {
                // 選ばれていない面を外そうとしても黙って通す。半径を変えれば
                // もともと選ばれなくなるので、そこで文句を言う筋合いはない
                dropped.insert(lookup.face_of(*face, electrode)?);
            }
            faces.retain(|f| !matches!(f, ElectrodeFace::Surface(i) if dropped.contains(i)));
        }

        // シェル要素と体表面の面が同じ 1 枚を指していたら、体表面側を落とす。
        // 残すと**同じ面を 2 回積分して面積が二重に数えられる**
        if faces.iter().any(|f| matches!(f, ElectrodeFace::Shell(_))) {
            let shells: BTreeSet<FaceKey> = faces
                .iter()
                .filter(|f| matches!(f, ElectrodeFace::Shell(_)))
                .map(|f| face_key(lookup.face_nodes(*f).1))
                .collect();
            faces.retain(|f| match f {
                ElectrodeFace::Shell(_) => true,
                ElectrodeFace::Surface(_) => !shells.contains(&face_key(lookup.face_nodes(*f).1)),
            });
        }

        let unknown_pids: Vec<String> = wanted_pids
            .difference(&seen_pids)
            .map(|p| p.to_string())
            .collect();
        if !unknown_pids.is_empty() {
            return Err(GfemError::new(
                ErrorKind::Case,
                format!(
                    "電極 {} が参照するシェル PID がメッシュにありません: PID={}",
                    electrode.name,
                    format_sample_list(&unknown_pids, SAMPLE_LIMIT)
                ),
            )
            .with_hint("読み込みレポートの PSHELL 一覧から選び直してください"));
        }
        let unknown_ids: Vec<String> = wanted_ids
            .difference(&seen_ids)
            .map(|i| i.to_string())
            .collect();
        if !unknown_ids.is_empty() {
            return Err(GfemError::new(
                ErrorKind::Case,
                format!(
                    "電極 {} が参照するシェル要素がメッシュにありません: EID={}",
                    electrode.name,
                    format_sample_list(&unknown_ids, SAMPLE_LIMIT)
                ),
            )
            .with_hint("CTRIA3 / CQUAD4 として存在する要素 ID を指定してください"));
        }
        if faces.is_empty() {
            return Err(GfemError::new(
                ErrorKind::Case,
                format!(
                    "電極 {} に面が 1 枚も割り当てられていません",
                    electrode.name
                ),
            )
            .with_hint(
                "シェル PID・シェル要素 ID・体表面のパッチ（3D ビューでクリック）の\
                 いずれかで電極面を指定してください",
            ));
        }
        Ok(faces)
    }
}

/// 電極を構成する面 1 枚。
///
/// シェル要素（CTRIA3 / CQUAD4）と、ソリッドの外表面から選んだ面の両方がありうる。
/// **後者はシェル要素を持たないモデルのための経路**で、荷重の作り方（面積按分。
/// SPEC §2.4）はどちらも同じ。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ElectrodeFace {
    /// シェル要素の通し番号（`Mesh::iter_shells` の `index`）
    Shell(u32),
    /// 外表面の面の通し番号（[`Surface::faces`] の索引）
    Surface(u32),
}

/// 電極面を引くための下ごしらえ。
///
/// 外表面は解析の準備段階で 1 度だけ作る。パッチ選択に使う索引
/// （[`SurfacePicker`]）は**パッチを使う電極があるときだけ**作るので、
/// シェルだけのモデルでは何も余計に計算しない。
pub struct FaceLookup<'a> {
    mesh: &'a Mesh,
    surface: &'a Surface,
    picker: std::sync::OnceLock<SurfacePicker>,
    /// (ソリッドの EID, 局所面番号) → 外表面の面の通し番号。
    /// 手で足した / 外した面を引くときだけ作る
    by_element: std::sync::OnceLock<BTreeMap<(ElemId, u8), u32>>,
}

impl<'a> FaceLookup<'a> {
    /// メッシュと、そこから抽出済みの外表面から作る。
    pub fn new(mesh: &'a Mesh, surface: &'a Surface) -> Self {
        Self {
            mesh,
            surface,
            picker: std::sync::OnceLock::new(),
            by_element: std::sync::OnceLock::new(),
        }
    }

    /// 手で指した面（EID と面番号）を外表面の通し番号に直す。
    ///
    /// 見つからなければ**どの電極のどの面か**が分かるエラーにする（設計原則 6）。
    /// 外表面に無い面を指しているということは、内部の面を指しているか、
    /// メッシュを差し替えて番号が変わったかのどちらか（SPEC §11.1）。
    pub fn face_of(&self, face: FaceRef, electrode: &Electrode) -> Result<u32> {
        let index = self.by_element.get_or_init(|| {
            let mut map = BTreeMap::new();
            for (i, f) in self.surface.faces().iter().enumerate() {
                let id = self.mesh.solid(f.element as usize).id;
                map.insert((id, f.local_face), i as u32);
            }
            map
        });
        index
            .get(&(face.element_id, face.local_face))
            .copied()
            .ok_or_else(|| {
                GfemError::new(
                    ErrorKind::Case,
                    format!(
                        "電極 {} が指している面が体表面にありません: EID={} の第 {} 面",
                        electrode.name,
                        face.element_id,
                        face.local_face + 1
                    ),
                )
                .with_hint(
                    "3D ビューで選び直してください。メッシュを差し替えた場合は、\
                     手で足した面・外した面の指定が古くなっている可能性があります",
                )
            })
    }

    /// 外表面。
    pub fn surface(&self) -> &'a Surface {
        self.surface
    }

    /// パッチ選択の索引（初回だけ作る）。
    pub fn picker(&self) -> &SurfacePicker {
        if let Some(p) = self.picker.get() {
            return p;
        }
        let built = SurfacePicker::build(self.mesh, self.surface);
        self.picker.get_or_init(|| built)
    }

    /// 面の種類と節点（面積分に渡す形）。
    pub fn face_nodes(
        &self,
        face: ElectrodeFace,
    ) -> (crate::element::ElementKind, &'a [NodeIndex]) {
        match face {
            ElectrodeFace::Shell(i) => {
                let s = shell_at(self.mesh, i as usize);
                (s.kind, s.conn)
            }
            ElectrodeFace::Surface(i) => {
                let f = &self.surface.faces()[i as usize];
                (f.kind(), f.nodes())
            }
        }
    }

    /// エラー表示に使う面の呼び名（CLAUDE.md 設計原則 6）。
    pub fn label(&self, face: ElectrodeFace) -> String {
        match face {
            ElectrodeFace::Shell(i) => {
                let s = shell_at(self.mesh, i as usize);
                format!("{} EID={}", s.kind.card_name(), s.id)
            }
            ElectrodeFace::Surface(i) => {
                let f = &self.surface.faces()[i as usize];
                let solid = self.mesh.solid(f.element as usize);
                format!(
                    "体表面の面（{} EID={} の第 {} 面）",
                    solid.kind.card_name(),
                    solid.id,
                    f.local_face + 1
                )
            }
        }
    }
}

/// 通し番号でシェル要素を引く。
pub(crate) fn shell_at(mesh: &Mesh, index: usize) -> crate::mesh::ElementRef<'_> {
    let mut offset = 0;
    for block in &mesh.shells {
        if index < offset + block.len() {
            let i = index - offset;
            return crate::mesh::ElementRef {
                index,
                kind: block.kind(),
                id: block.ids()[i],
                pid: block.pids()[i],
                conn: block.conn_of(i),
            };
        }
        offset += block.len();
    }
    panic!("シェル要素の通し番号が範囲外です: {index}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::nastran::read_nastran_from;
    use crate::mesh::extract_surface;
    use crate::units::LengthUnit;
    use std::path::Path;

    /// 四面体 1 個 + 面 (1,2,3) にシェル + 面 (1,2,4) にシェル
    const DECK: &str = "GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
GRID,4,,0.0,0.0,1.0
PSOLID,1,1
PSHELL,101,1,0.1
PSHELL,201,1,0.1
CTETRA,10,1,1,2,3,4
CTRIA3,100,101,1,3,2
CTRIA3,200,201,1,2,4
";

    fn setup() -> (Mesh, Surface) {
        let (mesh, _) =
            read_nastran_from(DECK.as_bytes(), Path::new("<test>"), LengthUnit::Meter).unwrap();
        let surface = extract_surface(&mesh).unwrap();
        (mesh, surface)
    }

    fn valid_case() -> Case {
        let mut case = Case::new();
        case.set_conductivity(1, 0.35).unwrap();
        case.add_electrode(Electrode::anode("E1", 2.0e-3).with_shell_pids([101]));
        case.add_electrode(Electrode::cathode("E2", 2.0e-3).with_shell_pids([201]));
        case
    }

    #[test]
    fn accepts_a_well_formed_case() {
        let (mesh, surface) = setup();
        let warnings = valid_case().validate(&mesh, &surface).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn rejects_missing_conductivity() {
        let (mesh, surface) = setup();
        let mut case = valid_case();
        case.conductivity.clear();
        let e = case.validate(&mesh, &surface).unwrap_err();
        assert!(e.to_string().contains("導電率が未設定"), "{e}");
        assert!(e.to_string().contains("PID=1"), "{e}");
    }

    #[test]
    fn rejects_unbalanced_current() {
        let (mesh, surface) = setup();
        let mut case = valid_case();
        case.electrodes[1].current = 1.0e-3;
        let e = case.validate(&mesh, &surface).unwrap_err();
        assert!(e.to_string().contains("一致しません"), "{e}");
        assert!(e.to_string().contains("SPEC §2.2"), "{e}");
    }

    #[test]
    fn rejects_missing_role() {
        let (mesh, surface) = setup();
        let mut case = valid_case();
        case.electrodes[1].role = ElectrodeRole::Anode;
        let e = case.validate(&mesh, &surface).unwrap_err();
        assert!(e.to_string().contains("cathode"), "{e}");
    }

    #[test]
    fn rejects_overlapping_electrodes() {
        let (mesh, surface) = setup();
        let mut case = valid_case();
        case.electrodes[1].shell_pids = vec![101];
        let e = case.validate(&mesh, &surface).unwrap_err();
        assert!(e.to_string().contains("両方に含まれています"), "{e}");
    }

    #[test]
    fn rejects_unknown_shell_pid() {
        let (mesh, surface) = setup();
        let mut case = valid_case();
        case.electrodes[0].shell_pids = vec![999];
        let e = case.validate(&mesh, &surface).unwrap_err();
        assert!(e.to_string().contains("PID=999"), "{e}");
    }

    #[test]
    fn rejects_interior_electrode_face() {
        // 面 (1,2,3) を共有する 2 個目の四面体を足すと、そのシェルは内部面になる
        let deck = DECK.replace(
            "CTRIA3,100,101,1,3,2",
            "GRID,5,,0.5,0.5,-1.0\nCTETRA,11,1,5,1,2,3\nCTRIA3,100,101,1,3,2",
        );
        let (mesh, _) =
            read_nastran_from(deck.as_bytes(), Path::new("<test>"), LengthUnit::Meter).unwrap();
        let surface = extract_surface(&mesh).unwrap();
        let e = valid_case().validate(&mesh, &surface).unwrap_err();
        assert!(e.to_string().contains("体表面"), "{e}");
        assert!(e.to_string().contains("SPEC §11.1"), "{e}");
    }

    #[test]
    fn resolves_faces_in_index_order() {
        let (mesh, surface) = setup();
        let lookup = FaceLookup::new(&mesh, &surface);
        let case = valid_case();
        let faces = case.resolve_faces(&lookup, &case.electrodes[0]).unwrap();
        assert_eq!(faces, vec![ElectrodeFace::Shell(0)]);
        let faces = case.resolve_faces(&lookup, &case.electrodes[1]).unwrap();
        assert_eq!(faces, vec![ElectrodeFace::Shell(1)]);
    }

    #[test]
    fn warns_about_conductivity_for_absent_pid() {
        let (mesh, surface) = setup();
        let mut case = valid_case();
        case.set_conductivity(42, 1.0).unwrap();
        let warnings = case.validate(&mesh, &surface).unwrap();
        assert!(
            warnings.iter().any(|w| w.contains("PID=42")),
            "{warnings:?}"
        );
    }
}
