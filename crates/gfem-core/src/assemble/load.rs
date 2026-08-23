//! 電極荷重ベクトルの構築（SPEC §2.4）。
//!
//! 電極面の面積を合計して |Γ| とし、一様電流密度 `J = I / |Γ|` を与える:
//!
//! ```text
//! f_i += J ∫_F N_i dS
//! ```
//!
//! **節点均等分配ではなく面積按分**にする理由は SPEC §2.4 のとおり。
//! メッシュが不均一な場合、節点均等では細かく切られた部分に過剰な電流が集まり、
//! 電流密度分布がメッシュ密度に依存して歪む。
//!
//! なおこのモデルは「理想電流源電極」の近似である。実際の金属電極は等電位面になり、
//! 電流は電極エッジに集中する（理論上は発散する）ため、エッジ近傍の電流密度は
//! 過小評価になる。将来の完全電極モデル（CEM）は SPEC §9 の予定項目。

use crate::case::{Case, ElectrodeRole, FaceLookup};
use crate::element::shell::face_integral;
use crate::error::{ErrorKind, GfemError, Result};
use crate::mesh::{Mesh, NodeIndex};
use std::collections::BTreeMap;

/// 面積がゼロとみなすしきい値との比較に使う相対値。
const MIN_AREA_RATIO: f64 = 1.0e-12;

/// 電極 1 個ぶんの幾何情報。ポスト処理（平均電位・インピーダンス）でも使う。
#[derive(Debug, Clone)]
pub struct ElectrodeGeometry {
    /// 電極名
    pub name: String,
    /// 陽極 / 陰極
    pub role: ElectrodeRole,
    /// 流入電流 [A]。陰極は負
    pub signed_current: f64,
    /// 電極面の総面積 [m²]
    pub area: f64,
    /// 一様電流密度 [A/m²]（符号付き）
    pub current_density: f64,
    /// 電極を構成する面の枚数
    pub n_faces: usize,
    /// 節点索引 → `∫N_i dS` [m²]（節点索引の昇順）
    ///
    /// 荷重ベクトルの分配にも、面積重み平均電位 `∫φ dS / A` の計算にも使う。
    pub node_weights: Vec<(NodeIndex, f64)>,
}

impl ElectrodeGeometry {
    /// 面積重み平均電位 [V]（SPEC §5.3）。
    pub fn mean_potential(&self, phi: &[f64]) -> f64 {
        if self.area <= 0.0 {
            return 0.0;
        }
        let sum: f64 = self
            .node_weights
            .iter()
            .map(|&(node, w)| w * phi[node as usize])
            .sum();
        sum / self.area
    }
}

/// 荷重ベクトル f と、電極ごとの幾何情報を作る。
///
/// 電極面は [`Case::resolve_faces`] が返す通し番号の昇順にたどるので、
/// 同じ入力からは常に同じ加算順序になる（CLAUDE.md 設計原則 1）。
pub fn assemble_load(
    mesh: &Mesh,
    case: &Case,
    lookup: &FaceLookup,
) -> Result<(Vec<f64>, Vec<ElectrodeGeometry>)> {
    let mut f = vec![0.0; mesh.nodes.len()];
    let mut geometries = Vec::with_capacity(case.electrodes.len());

    for electrode in &case.electrodes {
        let faces = case.resolve_faces(lookup, electrode)?;

        // 1. 面積と ∫N_i dS を集める
        let mut weights: BTreeMap<NodeIndex, f64> = BTreeMap::new();
        let mut area = 0.0;
        let mut points = [[0.0f64; 3]; 4];
        for &face in &faces {
            let (kind, conn) = lookup.face_nodes(face);
            let n = conn.len();
            mesh.gather_coords(conn, &mut points[..n]);
            let integral = face_integral(kind, &points[..n]);
            if integral.area <= 0.0 {
                return Err(GfemError::model(format!(
                    "電極 {} の面の面積がゼロです",
                    electrode.name
                ))
                .with_card(kind.card_name())
                .with_entity(lookup.label(face))
                .with_hint("退化した面です。メッシュ生成側で修正してください"));
            }
            area += integral.area;
            for (k, &node) in conn.iter().enumerate() {
                *weights.entry(node).or_insert(0.0) += integral.shape_integral[k];
            }
        }

        if area <= MIN_AREA_RATIO {
            return Err(GfemError::new(
                ErrorKind::Case,
                format!("電極 {} の総面積がゼロです: {area} m²", electrode.name),
            )
            .with_hint("電極面の指定を見直してください"));
        }

        // 2. 一様電流密度で配る（SPEC §2.4）
        let signed_current = electrode.signed_current();
        let current_density = signed_current / area;
        let node_weights: Vec<(NodeIndex, f64)> = weights.into_iter().collect();
        for &(node, w) in &node_weights {
            f[node as usize] += current_density * w;
        }

        geometries.push(ElectrodeGeometry {
            name: electrode.name.clone(),
            role: electrode.role,
            signed_current,
            area,
            current_density,
            n_faces: faces.len(),
            node_weights,
        });
    }

    Ok((f, geometries))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::Electrode;
    use crate::io::nastran::read_nastran_from;
    use crate::mesh::topology::{Surface, extract_surface};
    use crate::units::LengthUnit;
    use std::path::Path;

    /// 単位四面体。面 (1,3,2) と (1,2,4) にシェルを貼る
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

    /// 面を引く下ごしらえ。外表面を持ち回るのが煩わしいので、テストではまとめて作る。
    fn lookup_of<'a>(mesh: &'a Mesh, surface: &'a Surface) -> FaceLookup<'a> {
        FaceLookup::new(mesh, surface)
    }

    fn setup() -> (Mesh, Case) {
        let (mesh, _) =
            read_nastran_from(DECK.as_bytes(), Path::new("<test>"), LengthUnit::Meter).unwrap();
        let mut case = Case::new();
        case.set_conductivity(1, 0.35).unwrap();
        case.add_electrode(Electrode::anode("E1", 2.0e-3).with_shell_pids([101]));
        case.add_electrode(Electrode::cathode("E2", 2.0e-3).with_shell_pids([201]));
        (mesh, case)
    }

    #[test]
    fn load_satisfies_the_solvability_condition() {
        let (mesh, case) = setup();
        let surface = extract_surface(&mesh).unwrap();
        let (f, geo) = assemble_load(&mesh, &case, &lookup_of(&mesh, &surface)).unwrap();
        // SPEC §2.2 (a): 1ᵀf = 0
        let sum: f64 = f.iter().sum();
        let scale = f.iter().fold(0.0f64, |a, b| a.max(b.abs()));
        assert!(sum.abs() <= 1e-15 * scale, "1ᵀf = {sum}");
        assert_eq!(geo.len(), 2);
        assert_eq!(geo[0].n_faces, 1);
    }

    #[test]
    fn total_injected_current_matches_the_setting() {
        let (mesh, case) = setup();
        let surface = extract_surface(&mesh).unwrap();
        let (_, geo) = assemble_load(&mesh, &case, &lookup_of(&mesh, &surface)).unwrap();
        // この電極が配った電流の合計が I。
        // f の節点値を足すのでは駄目で、2 つの電極が節点を共有していると相殺されてしまう
        let injected: f64 = geo[0]
            .node_weights
            .iter()
            .map(|&(_, w)| w * geo[0].current_density)
            .sum();
        assert!((injected - 2.0e-3).abs() < 1e-18, "{injected}");
        // ∫N_i dS の総和は面積に一致する
        let weight_sum: f64 = geo[0].node_weights.iter().map(|&(_, w)| w).sum();
        assert!((weight_sum - geo[0].area).abs() < 1e-18);
        assert!((geo[0].area - 0.5).abs() < 1e-15);
        assert!((geo[0].current_density - 4.0e-3).abs() < 1e-18);
        assert!(geo[1].signed_current < 0.0);
    }

    #[test]
    fn triangle_load_is_split_evenly_between_its_nodes() {
        let (mesh, case) = setup();
        let surface = extract_surface(&mesh).unwrap();
        let (f, _) = assemble_load(&mesh, &case, &lookup_of(&mesh, &surface)).unwrap();
        // 陽極面 (節点 0,2,1) の 3 節点に I/3 ずつ入るはずだが、
        // 節点 0 と 1 は陰極面にも属するので、単独で確認できるのは節点 2
        assert!((f[2] - 2.0e-3 / 3.0).abs() < 1e-18, "{}", f[2]);
    }

    #[test]
    fn mean_potential_is_area_weighted() {
        let (mesh, case) = setup();
        let surface = extract_surface(&mesh).unwrap();
        let (_, geo) = assemble_load(&mesh, &case, &lookup_of(&mesh, &surface)).unwrap();
        // 一定電位ならその値がそのまま平均になる
        let phi = vec![2.5; mesh.nodes.len()];
        assert!((geo[0].mean_potential(&phi) - 2.5).abs() < 1e-15);
    }
}
