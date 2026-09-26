//! 体表面から直接選ぶ電極（SPEC §2.4 / §11.1）。
//!
//! シェル要素を持たないモデル（人体モデルはこれが普通）でも電極を置けること、
//! そして**シェルで指定したときと物理的に同じ荷重になる**ことを確かめる。

use gfem_core::assemble::load::assemble_load;
use gfem_core::case::{Case, Electrode, FaceLookup, SurfacePatch};
use gfem_core::io::nastran::read_nastran_from;
use gfem_core::mesh::topology::extract_surface;
use gfem_core::mesh::{Mesh, SurfacePicker};
use gfem_core::units::LengthUnit;
use std::path::Path;

/// 1 辺 1 m の立方体。上面と下面にはシェルも貼ってある（比較のため）。
const DECK: &str = "\
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,1.0,1.0,0.0
GRID,4,,0.0,1.0,0.0
GRID,5,,0.0,0.0,1.0
GRID,6,,1.0,0.0,1.0
GRID,7,,1.0,1.0,1.0
GRID,8,,0.0,1.0,1.0
CHEXA,10,1,1,2,3,4,5,6,7,8
CQUAD4,100,101,5,6,7,8
CQUAD4,200,201,1,2,3,4
PSOLID,1,1
PSHELL,101,1,0.1
PSHELL,201,1,0.1
";

/// シェルを一切持たないモデル（実際の人体モデルはこの形で配られる）。
const SOLID_ONLY: &str = "\
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,1.0,1.0,0.0
GRID,4,,0.0,1.0,0.0
GRID,5,,0.0,0.0,1.0
GRID,6,,1.0,0.0,1.0
GRID,7,,1.0,1.0,1.0
GRID,8,,0.0,1.0,1.0
CHEXA,10,1,1,2,3,4,5,6,7,8
PSOLID,1,1
";

fn read(deck: &str) -> Mesh {
    read_nastran_from(deck.as_bytes(), Path::new("<test>"), LengthUnit::Meter)
        .unwrap()
        .0
}

/// 上面・下面をそれぞれ 1 枚の面として選ぶパッチ。
///
/// 立方体の側面の重心は上面の重心から 0.707 m 離れているので、
/// 半径 0.5 m なら上面 1 枚だけが選ばれる。
fn top_patch() -> SurfacePatch {
    SurfacePatch {
        center: [0.5, 0.5, 1.0],
        radius: 0.5,
        pid: None,
    }
}

fn bottom_patch() -> SurfacePatch {
    SurfacePatch {
        center: [0.5, 0.5, 0.0],
        radius: 0.5,
        pid: None,
    }
}

fn case_with(anode: Electrode, cathode: Electrode) -> Case {
    let mut case = Case::new();
    case.set_conductivity(1, 0.35).unwrap();
    case.add_electrode(anode);
    case.add_electrode(cathode);
    case
}

#[test]
fn a_patch_gives_bitwise_the_same_load_as_the_shell_it_covers() {
    // 同じ面を「シェル要素で」と「体表面から選んで」の 2 通りで指定し、
    // 荷重ベクトルがビット単位で一致することを見る（CLAUDE.md 設計原則 1）。
    let mesh = read(DECK);
    let surface = extract_surface(&mesh).unwrap();
    let lookup = FaceLookup::new(&mesh, &surface);

    let by_shell = case_with(
        Electrode::anode("E1", 2.0e-3).with_shell_pids([101]),
        Electrode::cathode("E2", 2.0e-3).with_shell_pids([201]),
    );
    let by_patch = case_with(
        Electrode::anode("E1", 2.0e-3).with_patches([top_patch()]),
        Electrode::cathode("E2", 2.0e-3).with_patches([bottom_patch()]),
    );

    let (f_shell, geo_shell) = assemble_load(&mesh, &by_shell, &lookup).unwrap();
    let (f_patch, geo_patch) = assemble_load(&mesh, &by_patch, &lookup).unwrap();

    assert_eq!(f_shell, f_patch, "荷重ベクトルが一致しない");
    for (a, b) in geo_shell.iter().zip(&geo_patch) {
        assert_eq!(a.area, b.area, "面積が一致しない");
        assert_eq!(a.n_faces, b.n_faces);
        assert_eq!(a.current_density, b.current_density);
        assert_eq!(a.node_weights, b.node_weights);
    }
    assert_eq!(geo_patch[0].area, 1.0);
}

#[test]
fn electrodes_can_be_placed_on_a_model_without_any_shells() {
    // 本題。シェルが 1 つも無くても電極を定義して解ける
    let mesh = read(SOLID_ONLY);
    assert_eq!(mesh.shell_pids().len(), 0, "シェルが無いモデルのはず");

    let case = case_with(
        Electrode::anode("E1", 2.0e-3).with_patches([top_patch()]),
        Electrode::cathode("E2", 2.0e-3).with_patches([bottom_patch()]),
    );
    let surface = extract_surface(&mesh).unwrap();
    case.validate(&mesh, &surface).unwrap();

    let analysis = gfem_core::solve(&mesh, &case, None).unwrap();
    // 一様な柱の 1 次元問題: R = L / (σ A) = 1 / (0.35 × 1) [Ω]
    let expected = 1.0 / (0.35 * 1.0);
    let got = analysis.impedance();
    assert!(
        (got - expected).abs() / expected < 1e-9,
        "R = {got} Ω（理論値 {expected} Ω）"
    );
}

#[test]
fn a_patch_stays_on_the_body_surface() {
    // SPEC §11.1: パッチは外表面から選ぶので、内部面が混ざることはない
    let mesh = read(SOLID_ONLY);
    let surface = extract_surface(&mesh).unwrap();
    let picker = SurfacePicker::build(&mesh, &surface);
    // 立方体の中心から探しても、選ばれるのは外表面の面
    let faces = picker.select([0.5, 0.5, 0.5], 0.6, None);
    assert!(!faces.is_empty());
    for f in faces {
        let nodes = surface.faces()[f as usize].nodes();
        assert!(surface.contains(nodes), "外表面に載っていない面が選ばれた");
    }
}

#[test]
fn an_empty_patch_says_why() {
    // 存在しない PID で絞り込んだら、理由と直し方が分かるエラーにする
    // （CLAUDE.md 設計原則 6）
    let mesh = read(SOLID_ONLY);
    let surface = extract_surface(&mesh).unwrap();
    let lookup = FaceLookup::new(&mesh, &surface);
    let case = case_with(
        Electrode::anode("E1", 2.0e-3).with_patches([SurfacePatch {
            center: [0.5, 0.5, 1.0],
            radius: 0.5,
            pid: Some(999),
        }]),
        Electrode::cathode("E2", 2.0e-3).with_patches([bottom_patch()]),
    );
    let err = gfem_core::case::resolve_faces(&lookup, &case.electrodes[0])
        .unwrap_err()
        .to_string();
    assert!(err.contains("1 枚も選べませんでした"), "{err}");
    assert!(err.contains("PID=999"), "{err}");
    assert!(err.contains("体表面に露出しているか"), "{err}");
}

/// 立方体の上面（z = 1）を持つソリッド要素の EID と面番号。
fn top_face_ref(mesh: &Mesh) -> gfem_core::case::FaceRef {
    let surface = extract_surface(mesh).unwrap();
    let picker = SurfacePicker::build(mesh, &surface);
    let face = picker.nearest([0.5, 0.5, 1.0], None).unwrap();
    let f = &surface.faces()[face as usize];
    gfem_core::case::FaceRef::new(mesh.solid(f.element as usize).id, f.local_face)
}

#[test]
fn a_face_can_be_added_by_hand() {
    // パッチを使わず、面を 1 枚ずつ指定するだけでも電極になる
    let mesh = read(SOLID_ONLY);
    let surface = extract_surface(&mesh).unwrap();
    let lookup = FaceLookup::new(&mesh, &surface);
    let top = top_face_ref(&mesh);

    let by_hand = Electrode::anode("E1", 2.0e-3).with_added_faces([top]);
    let by_patch = Electrode::anode("E1", 2.0e-3).with_patches([top_patch()]);
    assert_eq!(
        gfem_core::case::resolve_faces(&lookup, &by_hand).unwrap(),
        gfem_core::case::resolve_faces(&lookup, &by_patch).unwrap(),
        "同じ 1 枚を指しているのに結果が違う"
    );
}

#[test]
fn a_face_removed_by_hand_drops_out_of_the_patch() {
    let mesh = read(SOLID_ONLY);
    let surface = extract_surface(&mesh).unwrap();
    let lookup = FaceLookup::new(&mesh, &surface);
    let top = top_face_ref(&mesh);

    // 立方体全体を覆うパッチから、上面だけ外す
    let whole = SurfacePatch {
        center: [0.5, 0.5, 0.5],
        radius: 10.0,
        pid: None,
    };
    let all = Electrode::anode("E1", 2.0e-3).with_patches([whole.clone()]);
    let minus_top = Electrode::anode("E1", 2.0e-3)
        .with_patches([whole])
        .with_removed_faces([top]);

    let all_faces = gfem_core::case::resolve_faces(&lookup, &all).unwrap();
    let kept = gfem_core::case::resolve_faces(&lookup, &minus_top).unwrap();
    assert_eq!(all_faces.len(), 6, "立方体の外表面は 6 面");
    assert_eq!(kept.len(), 5, "外した 1 枚だけ減るはず");
    assert!(kept.iter().all(|f| all_faces.contains(f)));
}

#[test]
fn removing_wins_over_adding() {
    // 同じ面を足しても外してもいる場合は「外す」を採る（迷ったら安全側）
    let mesh = read(SOLID_ONLY);
    let surface = extract_surface(&mesh).unwrap();
    let lookup = FaceLookup::new(&mesh, &surface);
    let top = top_face_ref(&mesh);

    let electrode = Electrode::anode("E1", 2.0e-3)
        .with_patches([top_patch()])
        .with_added_faces([top])
        .with_removed_faces([top]);
    let err = gfem_core::case::resolve_faces(&lookup, &electrode).unwrap_err();
    assert!(
        err.to_string()
            .contains("面が 1 枚も割り当てられていません"),
        "{err}"
    );
}

#[test]
fn a_stale_face_reference_says_which_one() {
    let mesh = read(SOLID_ONLY);
    let surface = extract_surface(&mesh).unwrap();
    let lookup = FaceLookup::new(&mesh, &surface);
    let electrode =
        Electrode::anode("E1", 2.0e-3).with_added_faces([gfem_core::case::FaceRef::new(9999, 0)]);
    let err = gfem_core::case::resolve_faces(&lookup, &electrode)
        .unwrap_err()
        .to_string();
    assert!(err.contains("EID=9999 の第 1 面"), "{err}");
    assert!(err.contains("体表面にありません"), "{err}");
}

#[test]
fn a_shell_and_a_patch_covering_the_same_face_are_not_counted_twice() {
    // シェルとパッチが同じ 1 枚を指しても、面積が二重にならないこと
    let mesh = read(DECK);
    let surface = extract_surface(&mesh).unwrap();
    let lookup = FaceLookup::new(&mesh, &surface);
    let both = Electrode::anode("E1", 2.0e-3)
        .with_shell_pids([101])
        .with_patches([top_patch()]);
    let faces = gfem_core::case::resolve_faces(&lookup, &both).unwrap();
    assert_eq!(faces.len(), 1, "同じ面が 2 枚に数えられている: {faces:?}");

    let case = case_with(
        both,
        Electrode::cathode("E2", 2.0e-3).with_patches([bottom_patch()]),
    );
    let (_, geo) = assemble_load(&mesh, &case, &lookup).unwrap();
    assert_eq!(geo[0].area, 1.0, "上面 1 枚ぶんの面積になっていない");
}

#[test]
fn the_same_face_cannot_belong_to_two_electrodes() {
    // シェルとパッチで同じ面を指しても重複として捕まえる
    let mesh = read(DECK);
    let surface = extract_surface(&mesh).unwrap();
    let case = case_with(
        Electrode::anode("E1", 2.0e-3).with_shell_pids([101]),
        Electrode::cathode("E2", 2.0e-3).with_patches([top_patch()]),
    );
    let err = case.validate(&mesh, &surface).unwrap_err().to_string();
    assert!(err.contains("両方に含まれています"), "{err}");
}

#[test]
fn solving_with_a_prepared_surface_gives_bitwise_the_same_answer() {
    // GUI や Python の Mesh は外表面と電極面の索引を作り置きし、何度解いても
    // 抽出し直さない（solve_prepared）。毎回作り直して解いた結果と 1 ビットも
    // 違ってはいけない。同じ索引で 2 回解いても同じ（索引を使い回しても壊れない）
    let mesh = read(SOLID_ONLY);
    let case = case_with(
        Electrode::anode("E1", 2.0e-3).with_patches([top_patch()]),
        Electrode::cathode("E2", 2.0e-3).with_patches([bottom_patch()]),
    );
    let fresh = gfem_core::solve(&mesh, &case, None).unwrap();

    let surface = extract_surface(&mesh).unwrap();
    let index = gfem_core::case::SurfaceIndex::new();
    for _ in 0..2 {
        let lookup = FaceLookup::with_index(&mesh, &surface, &index);
        let prepared = gfem_core::solve_prepared(&lookup, &case, None, None).unwrap();
        assert_eq!(prepared.potential, fresh.potential);
        assert_eq!(
            prepared.fields.current_density,
            fresh.fields.current_density
        );
        assert_eq!(prepared.solver.history, fresh.solver.history);
        assert_eq!(
            prepared.conservation.impedance,
            fresh.conservation.impedance
        );
    }
}
