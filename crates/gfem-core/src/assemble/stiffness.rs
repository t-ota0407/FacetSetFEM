//! 剛性行列のアセンブリ（SPEC §2.3 / §4.1）。

use crate::assemble::csr::CsrMatrix;
use crate::element::shape::{self, MAX_ELEMENT_NODES};
use crate::element::{ElementKind, Hex8, Penta6, tet4};
use crate::error::{GfemError, Result};
use crate::material::Conductivity;
use crate::mesh::{ElementRef, Mesh, PropId};
use std::collections::BTreeMap;

/// 剛性行列 K を組み立てる。
///
/// # 加算順序
///
/// 要素は [`Mesh::iter_solids`] の通し番号順にたどり、各要素内では `(i, j)` を
/// 行優先でたどる。CSR の格納位置は構築時に固定されているので、この順序を守れば
/// CPU 逐次でも行ごとの並列でも GPU でもビット単位で同じ行列になる
/// （CLAUDE.md 設計原則 1 / SPEC §4.2）。
pub fn assemble_stiffness(
    mesh: &Mesh,
    conductivity: &BTreeMap<PropId, Conductivity>,
    matrix: &mut CsrMatrix,
) -> Result<()> {
    matrix.values_mut().fill(0.0);
    let mut local = [[0.0f64; MAX_ELEMENT_NODES]; MAX_ELEMENT_NODES];

    for e in mesh.iter_solids() {
        let Some(sigma) = conductivity.get(&e.pid) else {
            return Err(GfemError::new(
                crate::error::ErrorKind::Case,
                format!("PID={} に導電率が設定されていません", e.pid),
            )
            .with_entity(format!("要素 EID={}", e.id))
            .with_hint("ケースの materials に該当 PID の σ [S/m] を追加してください"));
        };

        element_stiffness(mesh, &e, sigma, &mut local)?;
        scatter(matrix, e.conn, &local)?;
    }
    Ok(())
}

/// 四面体の要素剛性の `row` 行目だけ（並列アセンブリ用。SPEC §2.3 Tet4 要素剛性）。
///
/// 並列アセンブリは行ごとに要素をたどるので、要素 1 つにつき要るのはその行だけ。
/// 値は [`element_stiffness`] で全体を作ったときの同じ行とビット単位で同じ
/// （同じ式を同じ順で計算する。`element::tet4` のテストで確かめている）。
pub(crate) fn tet4_row(
    mesh: &Mesh,
    e: &ElementRef<'_>,
    sigma: &Conductivity,
    row: usize,
) -> Result<[f64; 4]> {
    let p = mesh.tet4_coords(e.conn);
    let shape = tet4::tet4_shape(&p);
    if shape.volume <= 0.0 {
        return Err(degenerate_tet(e));
    }
    Ok(tet4::tet4_stiffness_row(&shape, sigma, row))
}

/// 体積がゼロ以下の四面体のエラー。
fn degenerate_tet(e: &ElementRef<'_>) -> GfemError {
    GfemError::model("体積がゼロの要素があります")
        .with_card("CTETRA")
        .with_entity(format!("要素 EID={}", e.id))
        .with_hint("退化した四面体です。メッシュ生成側で修正してください")
}

/// 1 要素分の剛性行列を `local` に書く。
///
/// 要素種ごとに違うのは形状関数と積分則だけなので、Hex8 / Penta6 は共通の
/// アイソパラメトリック経路（[`shape::stiffness`]）に流す。Tet4 は ∇N が
/// 要素内で一定で解析的に書けるため、専用の速い経路を使う（結果は一致することを
/// `element::tet4` のテストで確認している）。
pub(crate) fn element_stiffness(
    mesh: &Mesh,
    e: &ElementRef<'_>,
    sigma: &Conductivity,
    local: &mut [[f64; MAX_ELEMENT_NODES]; MAX_ELEMENT_NODES],
) -> Result<()> {
    let mut points = [[0.0f64; 3]; MAX_ELEMENT_NODES];
    let n = e.conn.len();
    mesh.gather_coords(e.conn, &mut points[..n]);

    match e.kind {
        ElementKind::Tet4 => {
            // SPEC §2.3 Tet4 要素剛性
            let p = mesh.tet4_coords(e.conn);
            let shape = tet4::tet4_shape(&p);
            if shape.volume <= 0.0 {
                return Err(degenerate_tet(e));
            }
            let ke = tet4::tet4_stiffness(&shape, sigma);
            for row in local.iter_mut() {
                row.fill(0.0);
            }
            for i in 0..4 {
                local[i][..4].copy_from_slice(&ke[i]);
            }
        }
        // SPEC §2.3 Hex8 / Penta6 アイソパラメトリック要素剛性
        ElementKind::Hex8 => {
            shape::stiffness::<Hex8>(&points[..n], sigma, local)
                .map_err(|e2| with_element(e2, e))?;
        }
        ElementKind::Penta6 => {
            shape::stiffness::<Penta6>(&points[..n], sigma, local)
                .map_err(|e2| with_element(e2, e))?;
        }
        other => {
            return Err(GfemError::unsupported(format!(
                "{} は解析に使えません",
                other.card_name()
            ))
            .with_entity(format!("要素 EID={}", e.id))
            .with_hint("ソリッド要素は CTETRA / CHEXA / CPENTA に対応しています"));
        }
    }
    Ok(())
}

/// エラーに要素の情報を足す。
fn with_element(error: GfemError, e: &ElementRef<'_>) -> GfemError {
    error
        .with_card(e.kind.card_name())
        .with_entity(format!("要素 EID={}", e.id))
}

/// 要素行列を CSR に足し込む。
fn scatter(
    matrix: &mut CsrMatrix,
    conn: &[u32],
    ke: &[[f64; MAX_ELEMENT_NODES]; MAX_ELEMENT_NODES],
) -> Result<()> {
    for (i, &row) in conn.iter().enumerate() {
        for (j, &col) in conn.iter().enumerate() {
            let Some(k) = matrix.position(row as usize, col) else {
                return Err(GfemError::internal(format!(
                    "CSR 構造に成分 ({row}, {col}) がありません"
                ))
                .with_hint("非ゼロ構造の構築とアセンブリで要素の集合が食い違っています"));
            };
            matrix.values_mut()[k] += ke[i][j];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assemble::sparsity::build_sparsity;
    use crate::io::nastran::read_nastran_from;
    use crate::units::LengthUnit;
    use std::path::Path;

    fn read(text: &str) -> Mesh {
        read_nastran_from(text.as_bytes(), Path::new("<test>"), LengthUnit::Meter)
            .unwrap()
            .0
    }

    fn sigma_map(pid: PropId, value: f64) -> BTreeMap<PropId, Conductivity> {
        let mut m = BTreeMap::new();
        m.insert(pid, Conductivity::isotropic(value).unwrap());
        m
    }

    const TWO_TETS: &str = "GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
GRID,4,,0.0,0.0,1.0
GRID,5,,1.0,1.0,1.0
PSOLID,1,1
CTETRA,10,1,1,2,3,4
CTETRA,11,1,5,4,3,2
";

    #[test]
    fn stiffness_is_symmetric_and_has_the_constant_nullspace() {
        let mesh = read(TWO_TETS);
        let mut k = build_sparsity(&mesh).unwrap();
        assemble_stiffness(&mesh, &sigma_map(1, 0.35), &mut k).unwrap();
        assert!(k.is_symmetric(1e-18));

        // K·1 = 0（SPEC §2.2: 零空間は定数ベクトル）
        let ones = vec![1.0; k.n_rows()];
        let mut y = vec![0.0; k.n_rows()];
        k.spmv(&ones, &mut y);
        for v in &y {
            assert!(v.abs() < 1e-16, "K·1 = {y:?}");
        }
        // 対角は正
        assert!(k.diagonal().iter().all(|&d| d > 0.0));
    }

    #[test]
    fn assembly_is_bitwise_repeatable() {
        let mesh = read(TWO_TETS);
        let mut a = build_sparsity(&mesh).unwrap();
        let mut b = build_sparsity(&mesh).unwrap();
        assemble_stiffness(&mesh, &sigma_map(1, 0.35), &mut a).unwrap();
        assemble_stiffness(&mesh, &sigma_map(1, 0.35), &mut b).unwrap();
        assert_eq!(a.values(), b.values());
    }

    #[test]
    fn energy_is_positive_for_a_non_constant_field() {
        let mesh = read(TWO_TETS);
        let mut k = build_sparsity(&mesh).unwrap();
        assemble_stiffness(&mesh, &sigma_map(1, 1.0), &mut k).unwrap();
        let x: Vec<f64> = (0..k.n_rows()).map(|i| i as f64).collect();
        let mut y = vec![0.0; k.n_rows()];
        k.spmv(&x, &mut y);
        let energy: f64 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
        assert!(energy > 0.0, "{energy}");
    }

    const UNIT_CUBE_HEX: &str = "GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,1.0,1.0,0.0
GRID,4,,0.0,1.0,0.0
GRID,5,,0.0,0.0,1.0
GRID,6,,1.0,0.0,1.0
GRID,7,,1.0,1.0,1.0
GRID,8,,0.0,1.0,1.0
PSOLID,1,1
CHEXA,10,1,1,2,3,4,5,6,+C
+C,7,8
";

    #[test]
    fn hex_elements_are_assembled() {
        let mesh = read(UNIT_CUBE_HEX);
        let mut k = build_sparsity(&mesh).unwrap();
        assemble_stiffness(&mesh, &sigma_map(1, 0.35), &mut k).unwrap();
        assert!(k.is_symmetric(1e-18));

        // K·1 = 0（SPEC §2.2）
        let ones = vec![1.0; k.n_rows()];
        let mut y = vec![0.0; k.n_rows()];
        k.spmv(&ones, &mut y);
        assert!(y.iter().all(|v| v.abs() < 1e-16), "K·1 = {y:?}");
        assert!(k.diagonal().iter().all(|&d| d > 0.0));
    }

    #[test]
    fn penta_elements_are_assembled() {
        let mesh = read(
            "GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
GRID,4,,0.0,0.0,1.0
GRID,5,,1.0,0.0,1.0
GRID,6,,0.0,1.0,1.0
PSOLID,1,1
CPENTA,10,1,1,2,3,4,5,6
",
        );
        let mut k = build_sparsity(&mesh).unwrap();
        assemble_stiffness(&mesh, &sigma_map(1, 0.35), &mut k).unwrap();
        assert!(k.is_symmetric(1e-18));
        let ones = vec![1.0; k.n_rows()];
        let mut y = vec![0.0; k.n_rows()];
        k.spmv(&ones, &mut y);
        assert!(y.iter().all(|v| v.abs() < 1e-16));
    }

    #[test]
    fn inverted_hex_never_reaches_assembly() {
        // 上面と底面を入れ替えた CHEXA。読み込み時点で弾かれるので、
        // アセンブリまで届かない（parse::check_isoparametric_geometry）
        let e = read_nastran_from(
            UNIT_CUBE_HEX
                .replace("CHEXA,10,1,1,2,3,4,5,6", "CHEXA,42,1,5,6,7,8,1,2")
                .replace("+C,7,8", "+C,3,4")
                .as_bytes(),
            Path::new("<test>"),
            LengthUnit::Meter,
        )
        .unwrap_err();
        assert!(e.to_string().contains("ヤコビアン"), "{e}");
        assert!(e.to_string().contains("42"), "{e}");
        assert!(e.to_string().contains("CHEXA"), "{e}");
    }
}
