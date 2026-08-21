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
use crate::mesh::topology::Surface;
use crate::mesh::{ElemId, Mesh, NodeId, PropId};
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

/// 電極 1 個。
///
/// 面は「シェル PID の集合」と「個別のシェル要素 ID」の**組み合わせ**で指定する
/// （SPEC §3.3）。複数パッチを 1 つの電極として束ねられる。
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

/// 前処理（SPEC §4.3）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Preconditioner {
    /// 前処理なし
    None,
    /// 対角スケーリング
    Jacobi,
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
}

impl Backend {
    /// 表示・記録用の名前。
    pub const fn label(self) -> &'static str {
        match self {
            Backend::Auto => "auto",
            Backend::CpuSerial => "cpu_serial",
            Backend::CpuParallel => "cpu_parallel",
        }
    }

    /// 文字列から解釈する。
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Backend::Auto),
            "cpu" | "cpu_serial" | "serial" => Ok(Backend::CpuSerial),
            "cpu_parallel" | "parallel" => Ok(Backend::CpuParallel),
            other => Err(GfemError::new(
                ErrorKind::Case,
                format!("このビルドでは使えないバックエンドです: \"{other}\""),
            )
            .with_hint(
                "auto / cpu / cpu_parallel のいずれかを指定してください。\
                 cuda は M5 で対応します（SPEC §8）",
            )),
        }
    }

    /// `Auto` を実際のバックエンドに解決する。
    ///
    /// 小さい問題では並列化のオーバーヘッドの方が高くつくので、しきい値で切り替える。
    /// どちらでも結果は同じなので、この判断が答えを変えることはない。
    pub fn resolve(self, n_nodes: usize) -> Backend {
        const PARALLEL_THRESHOLD: usize = 20_000;
        match self {
            Backend::Auto => {
                if n_nodes >= PARALLEL_THRESHOLD {
                    Backend::CpuParallel
                } else {
                    Backend::CpuSerial
                }
            }
            other => other,
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
    /// 計算バックエンド
    pub backend: Backend,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            tolerance: 1.0e-10,
            max_iterations: 50_000,
            preconditioner: Preconditioner::Jacobi,
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
        let mut used_faces: BTreeMap<u32, String> = BTreeMap::new();
        for e in &self.electrodes {
            let faces = self.resolve_faces(mesh, e)?;
            let mut off_surface: Vec<String> = Vec::new();
            for &shell in &faces {
                let s = shell_at(mesh, shell as usize);
                if !surface.contains(s.conn) {
                    off_surface.push(s.id.to_string());
                }
                if let Some(other) = used_faces.insert(shell, e.name.clone())
                    && other != e.name
                {
                    return Err(GfemError::new(
                        ErrorKind::Case,
                        format!(
                            "シェル要素 EID={} が電極 {other} と {} の両方に含まれています",
                            s.id, e.name
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
                        "電極 {} の面のうち {} 枚が体表面（外表面）に載っていません: EID={}",
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

    /// 電極に属するシェル要素の通し番号（昇順・重複なし）を求める。
    ///
    /// 通し番号順に並べることで、荷重ベクトルへの加算順序が入力の書き順に依存しなくなる
    /// （CLAUDE.md 設計原則 1）。
    pub fn resolve_faces(&self, mesh: &Mesh, electrode: &Electrode) -> Result<Vec<u32>> {
        let wanted_pids: BTreeSet<PropId> = electrode.shell_pids.iter().copied().collect();
        let wanted_ids: BTreeSet<ElemId> = electrode.element_ids.iter().copied().collect();

        let mut faces: Vec<u32> = Vec::new();
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
                faces.push(s.index as u32);
            }
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
            .with_hint("シェル PID か要素 ID で電極面を指定してください"));
        }
        Ok(faces)
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
        let (mesh, _) = setup();
        let case = valid_case();
        let faces = case.resolve_faces(&mesh, &case.electrodes[0]).unwrap();
        assert_eq!(faces, vec![0]);
        let faces = case.resolve_faces(&mesh, &case.electrodes[1]).unwrap();
        assert_eq!(faces, vec![1]);
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
