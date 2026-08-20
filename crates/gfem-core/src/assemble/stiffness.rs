//! 剛性行列のアセンブリ（SPEC §2.3 / §4.1）。

use crate::assemble::csr::CsrMatrix;
use crate::element::{ElementKind, tet4};
use crate::error::{GfemError, Result};
use crate::material::Conductivity;
use crate::mesh::{Mesh, PropId};
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

    for e in mesh.iter_solids() {
        let Some(sigma) = conductivity.get(&e.pid) else {
            return Err(GfemError::new(
                crate::error::ErrorKind::Case,
                format!("PID={} に導電率が設定されていません", e.pid),
            )
            .with_entity(format!("要素 EID={}", e.id))
            .with_hint("ケースの materials に該当 PID の σ [S/m] を追加してください"));
        };

        match e.kind {
            ElementKind::Tet4 => {
                // SPEC §2.3 Tet4 要素剛性
                let p = mesh.tet4_coords(e.conn);
                let shape = tet4::tet4_shape(&p);
                if shape.volume <= 0.0 {
                    return Err(GfemError::model("体積がゼロの要素があります")
                        .with_card("CTETRA")
                        .with_entity(format!("要素 EID={}", e.id))
                        .with_hint("退化した四面体です。メッシュ生成側で修正してください"));
                }
                let ke = tet4::tet4_stiffness(&shape, sigma);
                scatter(matrix, e.conn, &ke)?;
            }
            other => {
                return Err(GfemError::unsupported(format!(
                    "{} の要素剛性は未実装です",
                    other.card_name()
                ))
                .with_entity(format!("要素 EID={}", e.id))
                .with_hint(
                    "v1.0 の M1 時点で解けるのは CTETRA（1 次四面体）だけです。\
                     CHEXA / CPENTA は M4 で対応します（SPEC §8）",
                ));
            }
        }
    }
    Ok(())
}

/// 要素行列を CSR に足し込む。
fn scatter(matrix: &mut CsrMatrix, conn: &[u32], ke: &[[f64; 4]; 4]) -> Result<()> {
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

    #[test]
    fn hex_elements_are_reported_as_not_yet_implemented() {
        let mesh = read(
            "GRID,1,,0.0,0.0,0.0
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
",
        );
        let mut k = build_sparsity(&mesh).unwrap();
        let e = assemble_stiffness(&mesh, &sigma_map(1, 1.0), &mut k).unwrap_err();
        assert!(e.to_string().contains("CHEXA"), "{e}");
        assert!(e.to_string().contains("M4"), "{e}");
    }
}
