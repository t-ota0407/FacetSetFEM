//! 解析ケースの定義と、実行前の整合性チェック（SPEC §3.3 / §6.2）。
//!
//! Nastran ファイルは形状と領域分けだけを担い、解析条件はこちらが持つ。
//! Nastran ファイルは書き換えない。
//!
//! YAML の読み書きは Python 側（PyYAML）で行い、ここはプログラム的に組み立てる形にする。
//! `facetsetfem-core` に YAML パーサを入れると依存が増えるうえ、行番号つきの親切なエラー
//! （CLAUDE.md 設計原則 6）を Rust と Python の二重に作ることになるため。

use crate::error::{ErrorKind, FacetSetFemError, Result, format_sample_list};
use crate::material::Conductivity;
pub use crate::mesh::pick::RectShape;
use crate::mesh::pick::{PatchFrame, SurfacePicker};
use crate::mesh::topology::{FaceKey, FaceTrace, Surface, SurfaceFace, face_key};
use crate::mesh::{ElemId, Mesh, NodeId, NodeIndex, PropId};
use crate::solver::amg::AmgOptions;
use std::collections::{BTreeMap, BTreeSet};

/// エラー一覧に並べる最大件数。
const SAMPLE_LIMIT: usize = 20;

/// 電流保存を満たしているとみなす相対しきい値（SPEC §2.2 (a)）。
///
/// ケースの検証（陽極と陰極の総電流の一致）と、保存則チェック（`1ᵀf / I`。
/// [`crate::post::check_conservation`]）の両方で使う。
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

/// 電極のモデル（電流の配り方）。ケースファイルの `distribution`（SPEC §3.3）。
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ElectrodeModel {
    /// 一様電流密度（面積按分。SPEC §2.4）。既定
    #[default]
    UniformCurrentDensity,
    /// 完全電極モデル（SPEC §2.5）。板は等電位で、皮膚との境目に接触インピーダンスがある
    Complete {
        /// 接触インピーダンス z [Ω·m²]（面積あたりの抵抗。正の有限値）
        contact_impedance: f64,
    },
}

impl ElectrodeModel {
    /// ケースファイルと GUI の選択肢（Python は `facetsetfem.distribution_choices()` で引く。SPEC §6.5）。
    pub const CHOICES: [&'static str; 2] = ["uniform_current_density", "complete_electrode"];

    /// ケースファイルの名前。
    pub const fn label(self) -> &'static str {
        match self {
            ElectrodeModel::UniformCurrentDensity => "uniform_current_density",
            ElectrodeModel::Complete { .. } => "complete_electrode",
        }
    }

    /// 接触インピーダンス [Ω·m²]。一様電流密度なら `None`。
    pub const fn contact_impedance(self) -> Option<f64> {
        match self {
            ElectrodeModel::UniformCurrentDensity => None,
            ElectrodeModel::Complete { contact_impedance } => Some(contact_impedance),
        }
    }

    /// 名前と接触インピーダンスから組み立てる。
    ///
    /// `complete_electrode` には接触インピーダンスが必須で、`uniform_current_density` に
    /// 渡すとエラーにする（書いた値が黙って使われないのを避ける。SPEC §3.3）。
    /// 値そのものの範囲は [`Case::validate`] が確かめる。
    pub fn parse(name: &str, contact_impedance: Option<f64>) -> Result<Self> {
        match (name, contact_impedance) {
            ("uniform_current_density", None) => Ok(ElectrodeModel::UniformCurrentDensity),
            ("uniform_current_density", Some(_)) => Err(FacetSetFemError::new(
                ErrorKind::Case,
                crate::tr!(
                    "contact_impedance is given, but the current distribution is uniform_current_density",
                    "contact_impedance が指定されていますが、電流の配り方が uniform_current_density です"
                ),
            )
            .with_hint(crate::tr!(
                "The contact impedance is used only by complete_electrode (SPEC §2.5).                  Either set distribution: complete_electrode or remove contact_impedance",
                "接触インピーダンスを使うのは complete_electrode だけです（SPEC §2.5）。                 distribution: complete_electrode にするか、contact_impedance を消してください"
            ))),
            ("complete_electrode", Some(z)) => Ok(ElectrodeModel::Complete {
                contact_impedance: z,
            }),
            ("complete_electrode", None) => Err(FacetSetFemError::new(
                ErrorKind::Case,
                crate::tr!(
                    "complete_electrode needs a contact impedance, but none is given",
                    "complete_electrode には接触インピーダンスが必要ですが、指定されていません"
                ),
            )
            .with_hint(crate::tr!(
                "Give contact_impedance in Ω·m² (e.g. 0.01). Fitting it to the measured                  inter-electrode resistance is the most reliable way (SPEC §2.5)",
                "contact_impedance を Ω·m² で指定してください（例: 0.01）。                 実測の電極間抵抗に合わせて決めるのが確実です（SPEC §2.5）"
            ))),
            (other, _) => Err(FacetSetFemError::new(
                ErrorKind::Case,
                crate::tr!(
                    "Unknown current distribution: \"{other}\"",
                    "未知の電流の配り方です: \"{other}\""
                ),
            )
            .with_hint(crate::tr!(
                "Choose one of {}",
                "{} のどれかを指定してください",
                Self::CHOICES.join(" / ")
            ))),
        }
    }
}

/// 体表面に置く電極パッチ 1 枚（SPEC §3.3 / §11.1）。
///
/// **シェル要素を持たないモデルのための指定方法。** 人体モデルはソリッド要素だけで
/// 配られるのが普通で、その場合シェル PID では電極を 1 つも置けない。外表面は
/// ソリッドの面から抽出できているので、そこから直接選ぶ。
///
/// GUI では 3D ビュー上のクリックが `center` を、パネルの入力が形（円の半径、または
/// 矩形の寸法と向き）を決める。**保存するのは面の一覧ではなく、この中心と形**。
/// 同じメッシュなら同じ面が選ばれ、ケースファイルを読んでも人が意味を追えるため
/// （面番号の羅列は追えない）。
#[derive(Debug, Clone, PartialEq)]
pub struct SurfacePatch {
    /// 中心 [m]。3D ビューでクリックした点
    pub center: [f64; 3],
    /// 形
    pub shape: PatchShape,
    /// この PID のソリッドが持つ面だけに限る（`None` なら制限しない）
    ///
    /// 首の切断面のように「外表面だが体表面ではない」面を避けるために使う。
    pub pid: Option<PropId>,
}

/// パッチの形（SPEC §11.1）。
#[derive(Debug, Clone, PartialEq)]
pub enum PatchShape {
    /// 円。中心から 3 次元の距離 `radius` [m] までの面を、体表面伝いに広げて選ぶ
    Circle {
        /// 半径 [m]
        radius: f64,
    },
    /// 矩形（角丸可）。面の重心を接平面に投影して選ぶ
    Rect(RectShape),
}

impl SurfacePatch {
    /// 円のパッチ。
    pub fn circle(center: [f64; 3], radius: f64, pid: Option<PropId>) -> Self {
        Self {
            center,
            shape: PatchShape::Circle { radius },
            pid,
        }
    }

    /// 矩形のパッチ。
    pub fn rect(center: [f64; 3], rect: RectShape, pid: Option<PropId>) -> Self {
        Self {
            center,
            shape: PatchShape::Rect(rect),
            pid,
        }
    }

    /// 指定どおりの面積 [m²]。
    ///
    /// 実際に選ばれる面は面 1 枚単位なので、この値とは縁のぶんだけずれる。GUI はこの差を
    /// 見せて、メッシュが電極に対して粗すぎないかを判断させる（SPEC §11.1）。
    pub fn nominal_area(&self) -> f64 {
        match &self.shape {
            PatchShape::Circle { radius } => std::f64::consts::PI * radius * radius,
            PatchShape::Rect(rect) => rect.area(),
        }
    }

    /// エラーメッセージ用の形の説明（例: 「半径 0.025 m」「矩形 0.02 m × 0.02 m」）。
    pub fn describe_shape(&self) -> String {
        match &self.shape {
            PatchShape::Circle { radius } => crate::tr!("radius {radius} m", "半径 {radius} m"),
            PatchShape::Rect(r) => {
                let mut s = crate::tr!(
                    "rectangle {} m × {} m",
                    "矩形 {} m × {} m",
                    r.width,
                    r.height
                );
                if r.corner_radius > 0.0 {
                    s += &crate::tr!(", corner radius {} m", "・角の半径 {} m", r.corner_radius);
                }
                s += &crate::tr!(
                    ", turned {1}° from {0:?}",
                    "・向き {:?} から {}°",
                    r.axis,
                    r.angle
                );
                s
            }
        }
    }

    /// 体表面の面を選ぶ（面の通し番号の昇順）。電極面の解決と GUI のクリックが共に通る。
    pub fn select(&self, picker: &SurfacePicker) -> Result<Vec<u32>> {
        self.check()?;
        match &self.shape {
            PatchShape::Circle { radius } => Ok(picker.select(self.center, *radius, self.pid)),
            PatchShape::Rect(rect) => picker.select_rect(self.center, rect, self.pid),
        }
    }

    /// 接平面の座標系と輪郭（表示用）。クリック点の近くに面が無ければ `None`。
    pub fn outline(&self, picker: &SurfacePicker) -> Result<Option<(PatchFrame, Vec<[f64; 3]>)>> {
        self.check()?;
        Ok(match &self.shape {
            PatchShape::Circle { radius } => picker
                .circle_frame(self.center, *radius, self.pid)
                .map(|f| (f, f.circle_outline(*radius))),
            PatchShape::Rect(rect) => picker
                .rect_frame(self.center, rect, self.pid)?
                .map(|f| (f, f.rect_outline(rect))),
        })
    }

    /// 中心と形が使える値か確かめる。
    ///
    /// 電極面を解決するとき（[`resolve_faces`]）と、GUI のクリックで面を選ぶとき
    /// （Python の `Mesh.pick_patch`）の両方がここを通る。
    ///
    /// **中心が NaN のまま通してはいけない。** ピックに失敗すると NaN が返ることがあり
    /// （実測: `vtkHardwarePicker`）、通すと距離の比較が全て false になって先頭の面が
    /// 選ばれ、見当違いの場所に電極ができる（SPEC §11.1）。
    pub fn check(&self) -> Result<()> {
        match &self.shape {
            PatchShape::Circle { radius } => {
                if !radius.is_finite() || *radius < 0.0 {
                    return Err(FacetSetFemError::new(
                        ErrorKind::Case,
                        crate::tr!(
                            "Invalid patch radius: {radius} m",
                            "パッチの半径が不正です: {radius} m"
                        ),
                    )
                    .with_hint(crate::tr!(
                        "Give the radius as a finite value of 0 or more (in m)",
                        "半径は 0 以上の有限値で指定してください（単位は m）"
                    )));
                }
            }
            PatchShape::Rect(rect) => rect.check()?,
        }
        if self.center.iter().any(|v| !v.is_finite()) {
            return Err(FacetSetFemError::new(
                ErrorKind::Case,
                crate::tr!(
                    "The patch center is not a number: {:?}",
                    "パッチの中心が数値になっていません: {:?}",
                    self.center
                ),
            )
            .with_hint(crate::tr!(
                "Picking in the 3D view may have failed. Click on the body surface again \
                 (in a case file, give a point on the body surface in m)",
                "3D ビューのピックに失敗した可能性があります。体表面の上をもう一度クリック\
                 してください（ケースファイルでは、体表面上の点を m で書いてください）"
            )));
        }
        Ok(())
    }
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
    /// 電流の配り方（SPEC §2.4 / §2.5）
    pub model: ElectrodeModel,
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
            model: ElectrodeModel::default(),
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

    /// 完全電極モデルにする（SPEC §2.5）。`contact_impedance` は Ω·m²。
    pub fn with_contact_impedance(mut self, contact_impedance: f64) -> Self {
        self.model = ElectrodeModel::Complete { contact_impedance };
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
    /// 選択肢として示す名前（既定を先頭に）。GUI の一覧とエラーの対処に使う。
    ///
    /// 別名（`"diagonal"` など）は [`Self::parse`] が受け付けるが、ここには載せない。
    pub const CHOICES: [&'static str; 3] = ["jacobi", "amg", "none"];

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
            other => Err(FacetSetFemError::new(
                ErrorKind::Case,
                crate::tr!(
                    "Unknown preconditioner: \"{other}\"",
                    "未知の前処理です: \"{other}\""
                ),
            )
            .with_hint(crate::tr!(
                "Specify one of {}",
                "{} のいずれかを指定してください",
                Self::CHOICES.join(" / ")
            ))),
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
    /// CUDA（`facetsetfem-cuda` クレート）
    ///
    /// `facetsetfem-core` は CUDA に依存しないので、この値を指定しただけでは GPU では動かない。
    /// 呼び出し側が [`crate::solve_with`] に `facetsetfem-cuda` のソルバを渡す必要がある
    /// （SPEC §7 の設計原則）。渡されなければ理由つきのエラーになる。
    Cuda,
}

impl Backend {
    /// 選択肢として示す名前（既定を先頭に）。GUI の一覧とエラーの対処に使う。
    ///
    /// 別名（`"gpu"` など）は [`Self::parse`] が受け付けるが、ここには載せない。
    /// CPU 逐次は打ちやすい `"cpu"` で示す（記録用の名前 [`Self::label`] は `"cpu_serial"`）。
    pub const CHOICES: [&'static str; 4] = ["auto", "cpu", "cpu_parallel", "cuda"];

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
            other => Err(FacetSetFemError::new(
                ErrorKind::Case,
                crate::tr!(
                    "Backend not available in this build: \"{other}\"",
                    "このビルドでは使えないバックエンドです: \"{other}\""
                ),
            )
            .with_hint(crate::tr!(
                "Specify one of {}",
                "{} のいずれかを指定してください",
                Self::CHOICES.join(" / ")
            ))),
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
    /// **`Auto` がここで `Cuda` を選ぶことはない。** `facetsetfem-core` は GPU があるか
    /// どうかを知らないため、GPU を使う判断は呼び出し側（`facetsetfem-py`）が行う。
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
    /// `facetsetfem-cuda` に聞く。
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

    /// 完全電極モデルの電極が 1 つでもあるか（あれば行列が正則になる。SPEC §2.5）。
    pub fn has_complete_electrodes(&self) -> bool {
        self.electrodes
            .iter()
            .any(|e| matches!(e.model, ElectrodeModel::Complete { .. }))
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
        self.validate_with(&FaceLookup::new(mesh, surface))
    }

    /// [`validate`](Self::validate) の、電極面の下ごしらえを渡す版。
    ///
    /// 解析（[`crate::analysis`]）は同じ `FaceLookup` で検証と荷重の組み立てをするので、
    /// パッチ選択の索引を 2 度作らずに済む。
    pub fn validate_with(&self, lookup: &FaceLookup) -> Result<Vec<String>> {
        let (mut problems, warnings) = self.problems_with(lookup);
        if problems.is_empty() {
            Ok(warnings)
        } else {
            Err(problems.remove(0))
        }
    }

    /// 実行前の整合性チェックの**不備を全部**集める（SPEC §6.2 の手順 4）。
    ///
    /// 戻り値は `(不備, 警告)`。[`validate_with`](Self::validate_with) は不備の 1 件目を
    /// エラーとして返すが、GUI はこれで全部を一覧にして一度に直せるようにする
    /// （以前は 1 件ずつしか出ず、直しては F5 を押すことを繰り返した）。
    /// 並びは `validate_with` が返してきた順（1 件目は以前と同じ不備になる）。
    /// ほかの不備から当然に従うもの（陰極が無いときの電流の不一致など）は重ねて出さない。
    pub fn problems_with(&self, lookup: &FaceLookup) -> (Vec<FacetSetFemError>, Vec<String>) {
        let (mesh, surface) = (lookup.mesh(), lookup.surface());
        let mut problems: Vec<FacetSetFemError> = Vec::new();
        let mut warnings = Vec::new();

        // --- 導電率 ---
        let region_pids = mesh.solid_pids();
        let missing: Vec<String> = region_pids
            .iter()
            .filter(|pid| !self.conductivity.contains_key(pid))
            .map(|pid| pid.to_string())
            .collect();
        if !missing.is_empty() {
            problems.push(
                FacetSetFemError::new(
                    ErrorKind::Case,
                    crate::tr!(
                        "Some regions have no conductivity ({}): PID={}",
                        "導電率が未設定の領域があります（{} 件）: PID={}",
                        missing.len(),
                        format_sample_list(&missing, SAMPLE_LIMIT)
                    ),
                )
                .with_hint(crate::tr!(
                    "Assign a conductivity σ [S/m] to every PSOLID PID",
                    "全ての PSOLID の PID に σ [S/m] を割り当ててください"
                )),
            );
        }
        for pid in self.conductivity.keys() {
            if !region_pids.contains(pid) {
                warnings.push(crate::tr!(
                    "A conductivity is set for PID={pid}, but there are no solid elements with that PID",
                    "PID={pid} に導電率が設定されていますが、その PID のソリッド要素はありません"
                ));
            }
        }

        // --- 電極の有無 ---
        if self.electrodes.is_empty() {
            problems.push(
                FacetSetFemError::new(
                    ErrorKind::Case,
                    crate::tr!("No electrodes are defined", "電極が定義されていません"),
                )
                .with_hint(crate::tr!(
                    "Define at least one anode and one cathode",
                    "陽極（anode）と陰極（cathode）を最低 1 つずつ定義してください"
                )),
            );
        } else {
            self.electrode_problems(lookup, surface, &mut problems);
        }

        // --- ソルバ設定 ---
        if !(self.solver.tolerance > 0.0 && self.solver.tolerance < 1.0) {
            problems.push(
                FacetSetFemError::new(
                    ErrorKind::Case,
                    crate::tr!(
                        "The convergence tolerance is out of range: {}",
                        "収束判定しきい値が範囲外です: {}",
                        self.solver.tolerance
                    ),
                )
                .with_hint(crate::tr!(
                    "Give a relative residual with 0 < tolerance < 1 (default 1e-10)",
                    "0 < tolerance < 1 の相対残差で指定してください（既定 1e-10）"
                )),
            );
        }
        if self.solver.max_iterations == 0 {
            problems.push(
                FacetSetFemError::new(
                    ErrorKind::Case,
                    crate::tr!(
                        "The maximum number of iterations is 0",
                        "最大反復回数が 0 です"
                    ),
                )
                .with_hint(crate::tr!(
                    "Specify 1 or more (default 50000)",
                    "1 以上を指定してください（既定 50000）"
                )),
            );
        }

        (problems, warnings)
    }

    /// 電極の不備（役割・名前・電流・電流保存・電極面）を `problems` に足す。
    fn electrode_problems(
        &self,
        lookup: &FaceLookup,
        surface: &Surface,
        problems: &mut Vec<FacetSetFemError>,
    ) {
        let mut roles_complete = true;
        for role in [ElectrodeRole::Anode, ElectrodeRole::Cathode] {
            if !self.electrodes.iter().any(|e| e.role == role) {
                roles_complete = false;
                problems.push(
                    FacetSetFemError::new(
                        ErrorKind::Case,
                        crate::tr!("No {} is defined", "{} が定義されていません", role.label()),
                    )
                    .with_hint(crate::tr!(
                        "Both a current inlet and a current outlet are required",
                        "電流の流入口と流出口の両方が必要です"
                    )),
                );
            }
        }
        // 電極名は結果（平均電位・実行サマリ）とケースファイルで電極を指す鍵。
        // 重複を許すと、同名の陽極と陰極が同じ面を指していても重なりとして
        // 捕まらず、荷重が打ち消し合って「R = 0 Ω・警告なし」という結果が出てしまう
        let mut seen_names: BTreeSet<&str> = BTreeSet::new();
        let mut duplicated: BTreeSet<&str> = BTreeSet::new();
        let mut blank_name = false;
        for e in &self.electrodes {
            if e.name.trim().is_empty() {
                blank_name = true;
            } else if !seen_names.insert(e.name.as_str()) {
                duplicated.insert(e.name.as_str());
            }
        }
        if blank_name {
            problems.push(
                FacetSetFemError::new(
                    ErrorKind::Case,
                    crate::tr!("An electrode has an empty name", "名前が空の電極があります"),
                )
                .with_hint(crate::tr!(
                    "Give each electrode a name such as E1 or E2",
                    "電極には E1・E2 のような名前を付けてください"
                )),
            );
        }
        if !duplicated.is_empty() {
            let names: Vec<String> = duplicated.iter().map(|n| n.to_string()).collect();
            problems.push(
                FacetSetFemError::new(
                    ErrorKind::Case,
                    crate::tr!(
                        "Several electrodes have the same name: {}",
                        "同じ名前の電極が複数あります: {}",
                        format_sample_list(&names, SAMPLE_LIMIT)
                    ),
                )
                .with_hint(crate::tr!(
                    "Electrode names identify electrodes in the results and the case file, \
                     so give each electrode a different name",
                    "電極名は結果とケースファイルで電極を指す鍵なので、電極ごとに違う名前にしてください"
                )),
            );
        }
        let mut currents_valid = true;
        for e in &self.electrodes {
            if !e.current.is_finite() || e.current <= 0.0 {
                currents_valid = false;
                problems.push(
                    FacetSetFemError::new(
                        ErrorKind::Case,
                        crate::tr!(
                            "The total current of electrode {} is not a positive finite value: {} A",
                            "電極 {} の総電流が正の有限値ではありません: {} A",
                            e.name,
                            e.current
                        ),
                    )
                    .with_hint(crate::tr!(
                        "Give the total current as a positive value and express its direction \
                         with role (anode / cathode)",
                        "総電流は正の値で指定し、向きは role（anode / cathode）で表してください"
                    )),
                );
            }
        }

        for e in &self.electrodes {
            if let ElectrodeModel::Complete { contact_impedance } = e.model
                && !(contact_impedance.is_finite() && contact_impedance > 0.0)
            {
                problems.push(
                    FacetSetFemError::new(
                        ErrorKind::Case,
                        crate::tr!(
                            "The contact impedance of electrode {} is not a positive finite value: {} Ω·m²",
                            "電極 {} の接触インピーダンスが正の有限値ではありません: {} Ω·m²",
                            e.name,
                            contact_impedance
                        ),
                    )
                    .with_hint(crate::tr!(
                        "Give the contact impedance as a positive value in Ω·m² (SPEC §2.5).                          For an ideal equipotential electrode use a small positive value instead of 0",
                        "接触インピーダンスは Ω·m² の正の値で指定してください（SPEC §2.5）。                         理想的な等電位電極にしたいときは、0 ではなく小さな正の値にしてください"
                    )),
                );
            }
        }

        // --- 電流保存（SPEC §2.2 の適合条件）---
        // 役割が欠けている・電流が不正なときは、不一致はその結果なので重ねて出さない
        let anode = self.total_anode_current();
        let cathode = self.total_cathode_current();
        let scale = anode.max(cathode);
        if roles_complete
            && currents_valid
            && (anode - cathode).abs() > CURRENT_BALANCE_TOLERANCE * scale
        {
            problems.push(
                FacetSetFemError::new(
                    ErrorKind::Case,
                    crate::tr!(
                        "Total inflow and outflow currents do not match: anodes {anode} A / cathodes {cathode} A",
                        "総流入電流と総流出電流が一致しません: 陽極 {anode} A / 陰極 {cathode} A"
                    ),
                )
                .with_hint(crate::tr!(
                    "This is an all-Neumann problem, so no solution exists unless Σ I_k = 0 \
                     (SPEC §2.2). Review the electrode currents",
                    "全 Neumann 境界の問題なので、Σ I_k = 0 でなければ解が存在しません（SPEC §2.2）。\
                     電極の電流値を見直してください"
                )),
            );
        }

        // --- 電極面 ---
        // 面の重なりは「体表面のどの面か」で見る。シェル要素とパッチが同じ面を
        // 指していても重複として捕まえられるようにするため。
        // 値は電極の通し番号（名前で比べると、名前の検査をすり抜けたときに見逃す）
        let mut used_faces: BTreeMap<FaceKey, usize> = BTreeMap::new();
        let mut overlapping: BTreeSet<(usize, usize)> = BTreeSet::new();
        for (index, e) in self.electrodes.iter().enumerate() {
            let faces = match resolve_faces(lookup, e) {
                Ok(faces) => faces,
                Err(err) => {
                    problems.push(err);
                    continue;
                }
            };
            let mut off_surface: Vec<String> = Vec::new();
            let mut mismatched: Vec<String> = Vec::new();
            for &face in &faces {
                let nodes = lookup.face_corners(face);
                match surface.find(nodes) {
                    // パッチは外表面から選ぶので、ここに来るのはシェル要素だけ
                    None => off_surface.push(lookup.label(face)),
                    Some(f) => {
                        if !mid_nodes_match(lookup, face, f) {
                            mismatched.push(lookup.label(face));
                        }
                    }
                }
                if let Some(other) = used_faces.insert(face_key(nodes), index)
                    && other != index
                    && overlapping.insert((other, index))
                {
                    // 同じ組の重なりは最初の 1 面だけ出す（全部出すと一覧が埋まる）
                    problems.push(
                        FacetSetFemError::new(
                            ErrorKind::Case,
                            crate::tr!(
                                "{} belongs to both electrode {} and electrode {}",
                                "{} が電極 {} と {} の両方に含まれています",
                                lookup.label(face),
                                self.electrodes[other].name,
                                e.name
                            ),
                        )
                        .with_hint(crate::tr!(
                            "A face cannot be assigned to two electrodes",
                            "1 つの面を 2 つの電極に割り当てることはできません"
                        )),
                    );
                }
            }
            if !off_surface.is_empty() {
                // SPEC §11.1: 電極は必ず体表面にある
                problems.push(
                    FacetSetFemError::new(
                        ErrorKind::Case,
                        crate::tr!(
                            "{1} face(s) of electrode {0} do not lie on the body surface (outer surface): {2}",
                            "電極 {} の面のうち {} 枚が体表面（外表面）に載っていません: {}",
                            e.name,
                            off_surface.len(),
                            format_sample_list(&off_surface, SAMPLE_LIMIT)
                        ),
                    )
                    .with_hint(crate::tr!(
                        "In transcutaneous electrical stimulation, electrodes are always on the body \
                         surface (SPEC §11.1). Either an internal face is referenced, or the nodes of \
                         the shell elements do not coincide with the faces of the solids",
                        "経皮電気刺激では電極は必ず体表面にあります（SPEC §11.1）。\
                         内部面を指しているか、シェル要素の節点がソリッドの面と一致していません"
                    )),
                );
            }
            if !mismatched.is_empty() {
                problems.push(
                    FacetSetFemError::new(
                        ErrorKind::Case,
                        crate::tr!(
                            "The mid-side nodes of {1} CTRIA6 face(s) of electrode {0} do not match \
                             the solid element below: {2}",
                            "電極 {} の CTRIA6 のうち {} 枚で、辺の中点の節点が下のソリッド要素と\
                             一致しません: {}",
                            e.name,
                            mismatched.len(),
                            format_sample_list(&mismatched, SAMPLE_LIMIT)
                        ),
                    )
                    .with_hint(crate::tr!(
                        "A CTRIA6 can only be placed on a face of a 10-node CTETRA, and its mid-side \
                         nodes G4–G6 must be the same GRIDs as those of the solid. On a first-order \
                         mesh, use CTRIA3",
                        "CTRIA6 は 10 節点の CTETRA の面にだけ置けて、辺の中点 G4〜G6 は\
                         ソリッドと同じ GRID でなければなりません。1 次要素のメッシュでは CTRIA3 を使ってください"
                    )),
                );
            }
        }
    }
}

/// シェル要素の辺の中点が、下のソリッド要素の面の中点と一致するか。
///
/// 1 次のシェル要素（CTRIA3 / CQUAD4）と体表面の面はいつも通す（頂点の照合で足りる。
/// 2 次要素の面に載った CTRIA3 は、ソリッドの面で積分する。[`FaceLookup::face_trace`]）。
/// CTRIA6 は、下が 2 次要素で、辺ごとの中点の節点がソリッドと同じときだけ通す。
fn mid_nodes_match(lookup: &FaceLookup, face: ElectrodeFace, below: &SurfaceFace) -> bool {
    let ElectrodeFace::Shell(i) = face else {
        return true;
    };
    let shell = lookup.mesh().shell(i as usize);
    if !shell.kind.is_quadratic() {
        return true;
    }
    let solid = below.trace(lookup.mesh());
    if !solid.kind().is_quadratic() {
        return false;
    }
    // 辺（両端の頂点の組）→ 中点。向きが逆でも同じ辺として比べる
    let edges = |nodes: &[NodeIndex]| {
        let mut out: Vec<([NodeIndex; 2], NodeIndex)> = (0..3)
            .map(|k| {
                let (a, b) = (nodes[k], nodes[(k + 1) % 3]);
                ([a.min(b), a.max(b)], nodes[3 + k])
            })
            .collect();
        out.sort_unstable();
        out
    };
    edges(shell.conn) == edges(solid.nodes())
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
        let selected = patch.select(lookup.picker()).map_err(|e| {
            e.with_entity(crate::tr!(
                "patch {1} of electrode {0}",
                "電極 {} のパッチ {}",
                electrode.name,
                k + 1
            ))
        })?;
        if selected.is_empty() {
            return Err(FacetSetFemError::new(
                ErrorKind::Case,
                crate::tr!(
                    "Patch {1} of electrode {0} did not select any face of the body surface \
                     (center {2:?} m, {3}{4})",
                    "電極 {} のパッチ {} が体表面の面を 1 枚も選べませんでした（中心 {:?} m, {}{}）",
                    electrode.name,
                    k + 1,
                    patch.center,
                    patch.describe_shape(),
                    match patch.pid {
                        Some(p) => format!(", PID={p}"),
                        None => String::new(),
                    }
                ),
            )
            .with_hint(crate::tr!(
                "If you filter by PID, check that the region is exposed on the body surface \
                 (see the region list in the load report and the outer surface in the 3D view)",
                "PID で絞り込んでいる場合は、その領域が体表面に露出しているか確認してください\
                 （読み込みレポートの領域一覧と、3D ビューの外表面表示で分かります）"
            )));
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
            .map(|f| face_key(lookup.face_corners(*f)))
            .collect();
        faces.retain(|f| match f {
            ElectrodeFace::Shell(_) => true,
            ElectrodeFace::Surface(_) => !shells.contains(&face_key(lookup.face_corners(*f))),
        });
    }

    let unknown_pids: Vec<String> = wanted_pids
        .difference(&seen_pids)
        .map(|p| p.to_string())
        .collect();
    if !unknown_pids.is_empty() {
        return Err(FacetSetFemError::new(
            ErrorKind::Case,
            crate::tr!(
                "Shell PIDs referenced by electrode {} are not in the mesh: PID={}",
                "電極 {} が参照するシェル PID がメッシュにありません: PID={}",
                electrode.name,
                format_sample_list(&unknown_pids, SAMPLE_LIMIT)
            ),
        )
        .with_hint(crate::tr!(
            "Choose again from the PSHELL list in the load report",
            "読み込みレポートの PSHELL 一覧から選び直してください"
        )));
    }
    let unknown_ids: Vec<String> = wanted_ids
        .difference(&seen_ids)
        .map(|i| i.to_string())
        .collect();
    if !unknown_ids.is_empty() {
        return Err(FacetSetFemError::new(
            ErrorKind::Case,
            crate::tr!(
                "Shell elements referenced by electrode {} are not in the mesh: EID={}",
                "電極 {} が参照するシェル要素がメッシュにありません: EID={}",
                electrode.name,
                format_sample_list(&unknown_ids, SAMPLE_LIMIT)
            ),
        )
        .with_hint(crate::tr!(
            "Specify IDs of existing CTRIA3 / CQUAD4 / CTRIA6 elements",
            "CTRIA3 / CQUAD4 / CTRIA6 として存在する要素 ID を指定してください"
        )));
    }
    if faces.is_empty() {
        return Err(FacetSetFemError::new(
            ErrorKind::Case,
            crate::tr!(
                "No faces are assigned to electrode {}",
                "電極 {} に面が 1 枚も割り当てられていません",
                electrode.name
            ),
        )
        .with_hint(crate::tr!(
            "Specify the electrode faces by shell PID, shell element ID, or a patch on the body \
             surface (click in the 3D view)",
            "シェル PID・シェル要素 ID・体表面のパッチ（3D ビューでクリック）の\
             いずれかで電極面を指定してください"
        )));
    }
    Ok(faces)
}

/// 電極を構成する面 1 枚。
///
/// シェル要素（CTRIA3 / CQUAD4 / CTRIA6）と、ソリッドの外表面から選んだ面の両方がありうる。
/// **後者はシェル要素を持たないモデルのための経路**で、荷重の作り方（面積按分。
/// SPEC §2.4）はどちらも同じ。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ElectrodeFace {
    /// シェル要素の通し番号（`Mesh::iter_shells` の `index`）
    Shell(u32),
    /// 外表面の面の通し番号（[`Surface::faces`] の索引）
    Surface(u32),
}

/// 外表面から電極面を引くための索引（パッチ選択と、面の識別子からの逆引き）。
///
/// **メッシュと外表面だけで決まる**ので、1 度作れば電極を何個置いても、何度解いても
/// 使い回せる。GUI や Python の `Mesh` は 1 つ持っておき、[`FaceLookup::with_index`] で
/// 貸す（以前は検証・solve・クリックのたびに作り直していた）。どちらも使われたときに
/// 初めて作るので、シェルだけのモデルでは何も余計に計算しない。
///
/// 作ったときと違うメッシュ・外表面と組み合わせてはいけない。
#[derive(Debug, Default)]
pub struct SurfaceIndex {
    picker: std::sync::OnceLock<SurfacePicker>,
    /// (ソリッドの EID, 局所面番号) → 外表面の面の通し番号。
    /// 手で足した / 外した面を引くときだけ作る
    by_element: std::sync::OnceLock<BTreeMap<(ElemId, u8), u32>>,
}

impl SurfaceIndex {
    /// 空の索引。中身は使われたときに作る。
    pub fn new() -> Self {
        Self::default()
    }

    /// パッチ選択の索引（初回だけ作る）。
    pub fn picker(&self, mesh: &Mesh, surface: &Surface) -> &SurfacePicker {
        self.picker
            .get_or_init(|| SurfacePicker::build(mesh, surface))
    }

    fn by_element(&self, mesh: &Mesh, surface: &Surface) -> &BTreeMap<(ElemId, u8), u32> {
        self.by_element.get_or_init(|| {
            surface
                .faces()
                .iter()
                .enumerate()
                .map(|(i, f)| ((mesh.solid(f.element as usize).id, f.local_face), i as u32))
                .collect()
        })
    }
}

/// [`FaceLookup`] が索引を自分で持つか、借りるか。
enum IndexSlot<'a> {
    Owned(SurfaceIndex),
    Shared(&'a SurfaceIndex),
}

/// 電極面を引くための下ごしらえ（メッシュ・外表面・[`SurfaceIndex`] の組）。
pub struct FaceLookup<'a> {
    mesh: &'a Mesh,
    surface: &'a Surface,
    index: IndexSlot<'a>,
}

impl<'a> FaceLookup<'a> {
    /// メッシュと、そこから抽出済みの外表面から作る。索引はこの `FaceLookup` が持つ。
    pub fn new(mesh: &'a Mesh, surface: &'a Surface) -> Self {
        Self {
            mesh,
            surface,
            index: IndexSlot::Owned(SurfaceIndex::new()),
        }
    }

    /// 作り置きの索引を借りて作る（同じメッシュで何度も解く・電極面を選び直す場合）。
    pub fn with_index(mesh: &'a Mesh, surface: &'a Surface, index: &'a SurfaceIndex) -> Self {
        Self {
            mesh,
            surface,
            index: IndexSlot::Shared(index),
        }
    }

    fn index(&self) -> &SurfaceIndex {
        match &self.index {
            IndexSlot::Owned(index) => index,
            IndexSlot::Shared(index) => index,
        }
    }

    /// 手で指した面（EID と面番号）を外表面の通し番号に直す。
    ///
    /// 見つからなければ**どの電極のどの面か**が分かるエラーにする（設計原則 6）。
    /// 外表面に無い面を指しているということは、内部の面を指しているか、
    /// メッシュを差し替えて番号が変わったかのどちらか（SPEC §11.1）。
    pub fn face_of(&self, face: FaceRef, electrode: &Electrode) -> Result<u32> {
        self.index()
            .by_element(self.mesh, self.surface)
            .get(&(face.element_id, face.local_face))
            .copied()
            .ok_or_else(|| {
                FacetSetFemError::new(
                    ErrorKind::Case,
                    crate::tr!(
                        "A face referenced by electrode {} is not on the body surface: face {2} of EID={1}",
                        "電極 {} が指している面が体表面にありません: EID={} の第 {} 面",
                        electrode.name,
                        face.element_id,
                        face.local_face + 1
                    ),
                )
                .with_hint(crate::tr!(
                    "Select it again in the 3D view. If the mesh was replaced, the manually added or \
                     removed faces may be out of date",
                    "3D ビューで選び直してください。メッシュを差し替えた場合は、\
                     手で足した面・外した面の指定が古くなっている可能性があります"
                ))
            })
    }

    /// メッシュ。
    pub fn mesh(&self) -> &'a Mesh {
        self.mesh
    }

    /// 外表面。
    pub fn surface(&self) -> &'a Surface {
        self.surface
    }

    /// パッチ選択の索引（初回だけ作る）。
    pub fn picker(&self) -> &SurfacePicker {
        self.index().picker(self.mesh, self.surface)
    }

    /// 面の頂点（体表面との照合・面の重なりの判定に使う）。
    ///
    /// 2 次の面（CTRIA6・Tet10 の面）でも頂点 3 つだけを返す。面は頂点で決まるので、
    /// CTRIA3 で指した面と Tet10 の体表面の面が同じ 1 枚として照合できる。
    pub fn face_corners(&self, face: ElectrodeFace) -> &'a [NodeIndex] {
        match face {
            ElectrodeFace::Shell(i) => {
                let s = self.mesh.shell(i as usize);
                &s.conn[..s.kind.corner_count()]
            }
            ElectrodeFace::Surface(i) => self.surface.faces()[i as usize].nodes(),
        }
    }

    /// 面積分に使う節点と面の種類（SPEC §2.4）。
    ///
    /// **2 次要素の面に載ったシェル要素は、ソリッドの面（辺の中点つきの Tria6）で積分する。**
    /// シェル要素は面を選ぶためだけに使う。1 次のメッシュから 2 次に上げたものでは
    /// 電極が CTRIA3 のまま残るが、それを頂点だけで積分すると、電流の総量は合うのに
    /// 配り方が 2 次の形状関数と食い違う（頂点に A/3 ずつ配り、中点に 0）。
    /// 1 次要素の面に載ったシェル要素は、これまでどおりシェル要素そのもので積分する。
    pub fn face_trace(&self, face: ElectrodeFace) -> FaceTrace {
        match face {
            ElectrodeFace::Shell(i) => {
                let s = self.mesh.shell(i as usize);
                let on_quadratic = self
                    .surface
                    .find(&s.conn[..s.kind.corner_count()])
                    .filter(|f| self.mesh.solid(f.element as usize).kind.is_quadratic());
                match on_quadratic {
                    Some(f) => f.trace(self.mesh),
                    None => FaceTrace::new(s.kind, s.conn),
                }
            }
            ElectrodeFace::Surface(i) => self.surface.faces()[i as usize].trace(self.mesh),
        }
    }

    /// エラー表示に使う面の呼び名（CLAUDE.md 設計原則 6）。
    pub fn label(&self, face: ElectrodeFace) -> String {
        match face {
            ElectrodeFace::Shell(i) => {
                let s = self.mesh.shell(i as usize);
                format!("{} EID={}", s.kind.card_name(), s.id)
            }
            ElectrodeFace::Surface(i) => {
                let f = &self.surface.faces()[i as usize];
                let solid = self.mesh.solid(f.element as usize);
                crate::tr!(
                    "body-surface face (face {2} of {0} EID={1})",
                    "体表面の面（{} EID={} の第 {} 面）",
                    solid.kind.card_name(),
                    solid.id,
                    f.local_face + 1
                )
            }
        }
    }
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
    fn lists_every_problem_at_once() {
        // GUI は不備を一覧にして一度に直させる。validate は 1 件目だけを返す
        let (mesh, surface) = setup();
        let mut case = valid_case();
        case.conductivity.clear();
        case.electrodes[0].current = 0.0;
        case.electrodes[1].shell_pids = vec![101];
        case.solver.tolerance = 5.0;
        let lookup = FaceLookup::new(&mesh, &surface);
        let (problems, _) = case.problems_with(&lookup);
        let text: Vec<String> = problems.iter().map(|p| p.to_string()).collect();
        assert_eq!(problems.len(), 4, "{text:#?}");
        assert!(text[0].contains("導電率が未設定"), "{text:#?}");
        assert!(
            text[1].contains("総電流が正の有限値ではありません"),
            "{text:#?}"
        );
        assert!(text[2].contains("両方に含まれています"), "{text:#?}");
        assert!(text[3].contains("収束判定しきい値が範囲外"), "{text:#?}");
        // 電流が不正なら、電流の不一致は重ねて出さない
        assert!(
            !text.iter().any(|t| t.contains("一致しません")),
            "{text:#?}"
        );
        // validate は同じ 1 件目を返す
        let first = case.validate(&mesh, &surface).unwrap_err().to_string();
        assert_eq!(first, text[0]);
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
    fn rejects_duplicate_electrode_names() {
        // 同名の陽極と陰極が同じ面を指すと、荷重が打ち消し合って
        // 「R = 0 Ω・警告なし」の結果が出ていた。名前の重複そのものを止める
        let (mesh, surface) = setup();
        let mut case = valid_case();
        case.electrodes[1].name = "E1".to_string();
        let e = case.validate(&mesh, &surface).unwrap_err();
        assert!(e.to_string().contains("同じ名前の電極"), "{e}");
        assert!(e.to_string().contains("E1"), "{e}");

        case.electrodes[1].shell_pids = vec![101];
        let e = case.validate(&mesh, &surface).unwrap_err();
        assert!(e.to_string().contains("同じ名前の電極"), "{e}");
    }

    #[test]
    fn rejects_blank_electrode_names() {
        let (mesh, surface) = setup();
        let mut case = valid_case();
        case.electrodes[0].name = "  ".to_string();
        let e = case.validate(&mesh, &surface).unwrap_err();
        assert!(e.to_string().contains("名前が空"), "{e}");
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
        let faces = resolve_faces(&lookup, &case.electrodes[0]).unwrap();
        assert_eq!(faces, vec![ElectrodeFace::Shell(0)]);
        let faces = resolve_faces(&lookup, &case.electrodes[1]).unwrap();
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

    #[test]
    fn every_listed_choice_parses_and_the_list_covers_every_value() {
        // GUI の選択肢とエラーの対処はこの一覧から作るので、読めない名前が
        // 載っていても、選べない値があってもいけない
        let backends: Vec<_> = Backend::CHOICES
            .iter()
            .map(|c| Backend::parse(c).unwrap())
            .collect();
        assert_eq!(backends[0], Backend::default());
        for b in [
            Backend::Auto,
            Backend::CpuSerial,
            Backend::CpuParallel,
            Backend::Cuda,
        ] {
            assert!(backends.contains(&b), "{b:?} が選択肢にない");
        }

        let preconditioners: Vec<_> = Preconditioner::CHOICES
            .iter()
            .map(|c| Preconditioner::parse(c).unwrap())
            .collect();
        assert_eq!(preconditioners[0], SolverOptions::default().preconditioner);
        for p in [
            Preconditioner::None,
            Preconditioner::Jacobi,
            Preconditioner::Amg,
        ] {
            assert!(preconditioners.contains(&p), "{p:?} が選択肢にない");
        }

        let e = Preconditioner::parse("ilu").unwrap_err().to_string();
        assert!(e.contains("jacobi / amg / none"), "{e}");
    }
}
