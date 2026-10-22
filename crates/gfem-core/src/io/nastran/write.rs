//! Nastran バルクデータの書き出し（SPEC §3.1 / §5.5）。
//!
//! 読み込んだメッシュに**電極の面をシェル要素として書き足して**書き出す。
//! GenericFEM で体表面から選んだ電極を、ほかのソフトウェア（COMSOL など）に
//! そのまま持っていくための経路。電極ごとに新しい PSHELL の PID を振るので、
//! 読み込んだ側では「その PID の面」として電極を選べる。
//!
//! # 書式
//!
//! **自由形式**（カンマ区切り）で、実数は**読み直すと同じ値に戻る最短の表記**で書く
//! （Rust の `{:e}`。仮数に小数点が無ければ `.0` を足す。Nastran の実数は小数点が要る）。
//! 小フィールド（8 文字）では座標が 7 桁ほどに丸まり、書き出したメッシュと元のメッシュで
//! 解が食い違う。1 行に入るのはカード名のあとに 8 欄までで、続きは先頭が `,` の継続行。
//!
//! 座標は**元のファイルの長さ単位**に戻して書く（内部の SI から換算する。設計原則 7）。
//! ケースファイルの `length_unit` と同じ単位のまま、書き出したファイルを読み直せる。
//! m → mm → m の換算で、座標は最後の 1 ビットほど変わりうる。
//!
//! # 書かないもの
//!
//! 座標系（CORD2*）は書かない。座標は基本座標系に変換済みのものを書く。
//! 材料カード（MAT1 など）は読み込みで捨てているので書かない。

use crate::element::ElementKind;
use crate::error::{GfemError, Result};
use crate::mesh::topology::FaceTrace;
use crate::mesh::{ElemId, Mesh, PropId};
use crate::units::LengthUnit;
use std::io::Write;
use std::path::Path;

/// 書き足すシェル要素の組（電極 1 つぶん）。
#[derive(Debug, Clone)]
pub struct ShellSet {
    /// 名前（ファイルの先頭のコメントに書く）
    pub name: String,
    /// PSHELL の PID
    pub pid: PropId,
    /// 面（体表面の要素面。2 次要素の面は CTRIA6 になる）
    pub faces: Vec<FaceTrace>,
}

/// 書き足すシェル要素に振る PID の候補。メッシュの PID（ソリッド・シェルとも）と重ならない、
/// いちばん大きい PID の次から `n` 個。
pub fn free_pids(mesh: &Mesh, n: usize) -> Vec<PropId> {
    let max = mesh
        .solid_pids()
        .into_iter()
        .chain(mesh.shell_pids())
        .max()
        .unwrap_or(0);
    (1..=n as u32).map(|k| max + k).collect()
}

/// メッシュに `extra` のシェル要素を書き足して、Nastran の自由形式で書き出す。
///
/// 書き足すシェル要素の EID は、メッシュの最大の EID の次から振る。PID がメッシュの PID と
/// 重なっていればエラー（読み込んだ側で、電極の面と元の面が区別できなくなる）。
pub fn write_nastran(
    path: impl AsRef<Path>,
    mesh: &Mesh,
    extra: &[ShellSet],
    unit: LengthUnit,
) -> Result<()> {
    let path = path.as_ref();
    let taken: std::collections::BTreeSet<PropId> = mesh
        .solid_pids()
        .into_iter()
        .chain(mesh.shell_pids())
        .collect();
    let mut seen = std::collections::BTreeSet::new();
    for set in extra {
        if taken.contains(&set.pid) || !seen.insert(set.pid) {
            return Err(GfemError::model(crate::tr!(
                "PID={} for {} is already used in the mesh or by another electrode",
                "{1} に振った PID={0} は、メッシュかほかの電極ですでに使われています",
                set.pid,
                set.name
            ))
            .with_hint(crate::tr!(
                "Choose a PID that is not used yet (for example, from gfem's free_pids)",
                "まだ使われていない PID を選んでください（free_pids で候補を出せます）"
            )));
        }
    }

    let file = std::fs::File::create(path).map_err(|e| {
        GfemError::io(
            path,
            crate::tr!(
                "Cannot create the Nastran file",
                "Nastran ファイルを作成できません"
            ),
            e,
        )
        .with_hint(crate::tr!(
            "Check that the output directory exists and that you have write permission",
            "出力先のディレクトリが存在し、書き込み権限があるかを確認してください"
        ))
    })?;
    let mut w = std::io::BufWriter::new(file);
    let io_err = |e: std::io::Error| {
        GfemError::io(
            path,
            crate::tr!(
                "Failed to write the Nastran file",
                "Nastran ファイルの書き込みに失敗しました"
            ),
            e,
        )
    };
    write_all(&mut w, mesh, extra, unit).map_err(io_err)?;
    w.flush().map_err(io_err)
}

fn write_all(
    w: &mut impl Write,
    mesh: &Mesh,
    extra: &[ShellSet],
    unit: LengthUnit,
) -> std::io::Result<()> {
    // コメントは ASCII に限る（データ行と同じく、読む側の文字コードに依らないように）
    writeln!(w, "$ Written by GenericFEM {}", env!("CARGO_PKG_VERSION"))?;
    if let Some(source) = &mesh.source {
        writeln!(w, "$ Source mesh: {}", source.display())?;
    }
    writeln!(w, "$ Length unit: {}", unit.symbol())?;
    for set in extra {
        writeln!(
            w,
            "$ Electrode {}: PSHELL PID={} ({} faces)",
            ascii_comment(&set.name),
            set.pid,
            set.faces.len()
        )?;
    }
    writeln!(w, "BEGIN BULK")?;

    // --- 節点 ---
    let scale = 1.0 / unit.to_meters();
    for i in 0..mesh.nodes.len() {
        let p = mesh.nodes.coord(i as u32);
        writeln!(
            w,
            "GRID,{},,{},{},{}",
            mesh.nodes.id(i as u32),
            real(p[0] * scale),
            real(p[1] * scale),
            real(p[2] * scale)
        )?;
    }

    // --- プロパティ ---
    for p in &mesh.solid_properties {
        writeln!(w, "PSOLID,{},{}", p.pid, p.mid.unwrap_or(p.pid))?;
    }
    for p in &mesh.shell_properties {
        writeln!(w, "PSHELL,{},{},1.0", p.pid, p.mid.unwrap_or(p.pid))?;
    }
    for set in extra {
        writeln!(w, "PSHELL,{},{},1.0", set.pid, set.pid)?;
    }

    // --- 要素 ---
    let mut max_eid: ElemId = 0;
    for block in mesh.blocks() {
        let card = block.kind().card_name();
        for e in 0..block.len() {
            let id = block.ids()[e];
            max_eid = max_eid.max(id);
            let nodes: Vec<u32> = block.conn_of(e).iter().map(|&n| mesh.nodes.id(n)).collect();
            element(w, card, id, block.pids()[e], &nodes)?;
        }
    }
    let mut eid = max_eid;
    for set in extra {
        for face in &set.faces {
            eid += 1;
            let card = match face.kind() {
                ElementKind::Tria6 => "CTRIA6",
                other => other.card_name(),
            };
            let nodes: Vec<u32> = face.nodes().iter().map(|&n| mesh.nodes.id(n)).collect();
            element(w, card, eid, set.pid, &nodes)?;
        }
    }
    writeln!(w, "ENDDATA")
}

/// 要素カード 1 枚。カード名のあとに 8 欄まで書き、続きは継続行（先頭が `,`）。
fn element(
    w: &mut impl Write,
    card: &str,
    id: ElemId,
    pid: PropId,
    nodes: &[u32],
) -> std::io::Result<()> {
    let fields: Vec<String> = [id, pid]
        .into_iter()
        .chain(nodes.iter().copied())
        .map(|v| v.to_string())
        .collect();
    let (first, rest) = fields.split_at(fields.len().min(8));
    write!(w, "{card},{}", first.join(","))?;
    for chunk in rest.chunks(8) {
        write!(w, "\n,{}", chunk.join(","))?;
    }
    writeln!(w)
}

/// 実数を、読み直すと同じ値に戻る最短の表記で書く（仮数には必ず小数点を入れる）。
fn real(v: f64) -> String {
    let s = format!("{v:e}");
    match s.split_once('e') {
        Some((mantissa, exponent)) if !mantissa.contains('.') => {
            format!("{mantissa}.0e{exponent}")
        }
        _ => s,
    }
}

/// コメントに書く名前を ASCII に限る（ほかの文字は `?`）。
fn ascii_comment(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii() && !c.is_ascii_control() {
                c
            } else {
                '?'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::nastran::read_nastran;
    use crate::mesh::topology::extract_surface;
    use crate::testing::kuhn_box;

    fn temp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("gfem-write-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn reals_round_trip_and_always_have_a_decimal_point() {
        for v in [
            0.0,
            1.0,
            -2.5,
            1e-5,
            0.1,
            1.0 / 3.0,
            123456.789,
            -7e300,
            5e-324,
        ] {
            let s = real(v);
            let mantissa = s.split('e').next().unwrap();
            assert!(mantissa.contains('.'), "{s}");
            assert_eq!(crate::io::nastran::field::parse_real(&s), Some(v), "{s}");
        }
    }

    #[test]
    fn a_written_mesh_reads_back_with_the_electrode_faces() {
        let mesh = kuhn_box([2, 2, 1], [0.02, 0.02, 0.01]);
        let surface = extract_surface(&mesh).unwrap();
        // 上面（z 最大）の面を電極にする
        let top: Vec<FaceTrace> = surface
            .faces()
            .iter()
            .filter(|f| f.nodes().iter().all(|&n| mesh.nodes.coord(n)[2] > 0.0099))
            .map(|f| f.trace(&mesh))
            .collect();
        assert!(!top.is_empty());
        let pid = free_pids(&mesh, 1)[0];
        let path = temp("box.bdf");
        let sets = [ShellSet {
            name: "E1".into(),
            pid,
            faces: top.clone(),
        }];
        write_nastran(&path, &mesh, &sets, LengthUnit::Millimeter).unwrap();

        let (again, report) = read_nastran(&path, LengthUnit::Millimeter).unwrap();
        assert!(report.unsupported.is_empty(), "{:?}", report.unsupported);
        assert_eq!(again.nodes.ids(), mesh.nodes.ids());
        for (a, b) in again.nodes.coords().iter().zip(mesh.nodes.coords()) {
            assert!((a - b).abs() <= 1e-15 * b.abs().max(1e-3), "{a} vs {b}");
        }
        assert_eq!(again.n_solid_elements(), mesh.n_solid_elements());
        assert_eq!(again.solids[0].conn(), mesh.solids[0].conn());
        assert_eq!(again.shell_pids(), vec![pid]);
        let shells = again.block(ElementKind::Tria3).unwrap();
        assert_eq!(shells.len(), top.len());
        // 書き足したシェルは、元の体表面の面と同じ節点（向きも同じ）
        for (k, face) in top.iter().enumerate() {
            assert_eq!(shells.conn_of(k), face.nodes());
        }
        // EID は元の最大の次から
        let max = *mesh.solids[0].ids().iter().max().unwrap();
        assert_eq!(shells.ids()[0], max + 1);
    }

    #[test]
    fn metres_are_written_exactly() {
        let mesh = kuhn_box([1, 1, 1], [0.013, 0.017, 0.019]);
        let path = temp("exact.bdf");
        write_nastran(&path, &mesh, &[], LengthUnit::Meter).unwrap();
        let (again, _) = read_nastran(&path, LengthUnit::Meter).unwrap();
        let bits = |m: &Mesh| {
            m.nodes
                .coords()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&again), bits(&mesh));
    }

    #[test]
    fn second_order_meshes_are_written_with_ten_node_ctetra_and_ctria6() {
        let mesh = crate::mesh::elevate(&kuhn_box([1, 1, 1], [1.0, 1.0, 1.0])).unwrap();
        let surface = extract_surface(&mesh).unwrap();
        let faces: Vec<FaceTrace> = surface.faces()[..2]
            .iter()
            .map(|f| f.trace(&mesh))
            .collect();
        let path = temp("tet10.bdf");
        let sets = [ShellSet {
            name: "E".into(),
            pid: 50,
            faces,
        }];
        write_nastran(&path, &mesh, &sets, LengthUnit::Meter).unwrap();
        let (again, report) = read_nastran(&path, LengthUnit::Meter).unwrap();
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        assert_eq!(again.solids[0].kind(), ElementKind::Tet10);
        assert_eq!(again.solids[0].conn(), mesh.solids[0].conn());
        assert_eq!(again.block(ElementKind::Tria6).unwrap().len(), 2);
    }

    #[test]
    fn a_pid_that_is_already_used_is_an_error() {
        let mesh = kuhn_box([1, 1, 1], [1.0, 1.0, 1.0]);
        let sets = [ShellSet {
            name: "E".into(),
            pid: 1,
            faces: Vec::new(),
        }];
        let e = write_nastran(temp("dup.bdf"), &mesh, &sets, LengthUnit::Meter).unwrap_err();
        assert!(e.to_string().contains("すでに使われています"), "{e}");
    }
}
