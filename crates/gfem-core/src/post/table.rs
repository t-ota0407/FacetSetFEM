//! 節点・要素ごとの結果の表（CSV。SPEC §5.5）。
//!
//! 他のソフトウェア（COMSOL・MATLAB など）で結果を扱う・突き合わせるための書き出し。
//! 節点の表は電位、要素の表は重心での E・J など。座標の列があるので、そのまま
//! 他のソフトウェアに「この点で値を出す」点の一覧として渡せる。
//!
//! 列名は ASCII で単位付き（領域統計の CSV と同じ取り決め）。値は**読み直すと同じ値に戻る
//! 最短の表記**で書く（突き合わせで 10⁻¹⁰ より小さい差を見たいので、桁を丸めない）。

use crate::error::{GfemError, Result};
use crate::mesh::Mesh;
use crate::post::FieldResult;
use std::io::Write;
use std::path::Path;

/// 節点の表: `node_id,x_m,y_m,z_m,phi_V`。
pub fn write_node_table(path: impl AsRef<Path>, mesh: &Mesh, potential: &[f64]) -> Result<()> {
    let path = path.as_ref();
    if potential.len() != mesh.nodes.len() {
        return Err(GfemError::internal(crate::tr!(
            "The number of nodal values does not match the mesh: {} given, but there are {} nodes",
            "節点値の個数がメッシュと合いません: {} 個ですが節点は {} 個です",
            potential.len(),
            mesh.nodes.len()
        )));
    }
    write_csv(path, |w| {
        writeln!(w, "node_id,x_m,y_m,z_m,phi_V")?;
        for (i, &phi) in potential.iter().enumerate() {
            let p = mesh.nodes.coord(i as u32);
            writeln!(
                w,
                "{},{:e},{:e},{:e},{:e}",
                mesh.nodes.id(i as u32),
                p[0],
                p[1],
                p[2],
                phi
            )?;
        }
        Ok(())
    })
}

/// 要素の表（ソリッド要素のみ）。重心の座標と、要素の代表値（重心の値。SPEC §5.1）。
///
/// `element_id,pid,x_m,y_m,z_m,Ex_V_per_m,Ey_V_per_m,Ez_V_per_m,Jx_A_per_m2,Jy_A_per_m2,
/// Jz_A_per_m2,E_V_per_m,J_A_per_m2,E_peak_V_per_m,J_peak_A_per_m2,power_density_W_per_m3,volume_m3`
///
/// `*_peak` は要素内の最大（2 次要素では頂点での値。1 次要素では代表値と同じ）。
pub fn write_element_table(
    path: impl AsRef<Path>,
    mesh: &Mesh,
    fields: &FieldResult,
) -> Result<()> {
    let path = path.as_ref();
    if fields.n_elements != mesh.n_solid_elements() {
        return Err(GfemError::internal(crate::tr!(
            "The number of element values does not match the mesh: {} given, but there are {} solid elements",
            "要素値の個数がメッシュと合いません: {} 個ですがソリッド要素は {} 個です",
            fields.n_elements,
            mesh.n_solid_elements()
        )));
    }
    write_csv(path, |w| {
        writeln!(
            w,
            "element_id,pid,x_m,y_m,z_m,Ex_V_per_m,Ey_V_per_m,Ez_V_per_m,\
             Jx_A_per_m2,Jy_A_per_m2,Jz_A_per_m2,E_V_per_m,J_A_per_m2,\
             E_peak_V_per_m,J_peak_A_per_m2,power_density_W_per_m3,volume_m3"
        )?;
        for e in mesh.iter_solids() {
            let k = e.index;
            let c = centroid(mesh, &e);
            let (ef, j) = (
                &fields.electric_field[3 * k..3 * k + 3],
                &fields.current_density[3 * k..3 * k + 3],
            );
            writeln!(
                w,
                "{},{},{:e},{:e},{:e},{:e},{:e},{:e},{:e},{:e},{:e},{:e},{:e},{:e},{:e},{:e},{:e}",
                e.id,
                e.pid,
                c[0],
                c[1],
                c[2],
                ef[0],
                ef[1],
                ef[2],
                j[0],
                j[1],
                j[2],
                fields.e_magnitude[k],
                fields.j_magnitude[k],
                fields.e_peak_of(k),
                fields.j_peak_of(k),
                fields.power_density[k],
                fields.volume[k]
            )?;
        }
        Ok(())
    })
}

/// 要素の重心（頂点の平均）。代表値を評価した点（局所座標の中心）と同じ点になる。
///
/// 2 次要素で全節点の平均を取ると、辺が曲がった要素では中心からずれる。頂点だけで取れば
/// 1 次要素から上げたものと同じ点になり、1 次と 2 次の表を同じ点で突き合わせられる。
fn centroid(mesh: &Mesh, e: &crate::mesh::ElementRef<'_>) -> [f64; 3] {
    mesh.element_centroid(&e.conn[..e.kind.corner_count()])
}

fn write_csv(
    path: &Path,
    body: impl FnOnce(&mut std::io::BufWriter<std::fs::File>) -> std::io::Result<()>,
) -> Result<()> {
    let file = std::fs::File::create(path).map_err(|e| {
        GfemError::io(
            path,
            crate::tr!("Cannot create the CSV file", "CSV ファイルを作成できません"),
            e,
        )
        .with_hint(crate::tr!(
            "Check that the output directory exists and that you have write permission",
            "出力先のディレクトリが存在し、書き込み権限があるかを確認してください"
        ))
    })?;
    let mut w = std::io::BufWriter::new(file);
    body(&mut w).and_then(|()| w.flush()).map_err(|e| {
        GfemError::io(
            path,
            crate::tr!(
                "Failed to write the CSV file",
                "CSV ファイルの書き込みに失敗しました"
            ),
            e,
        )
    })
}
