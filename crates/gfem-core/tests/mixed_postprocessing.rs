//! 要素種を混ぜたメッシュでのポスト処理（SPEC §5.1 / §8 M7）。
//!
//! M6 までは等値面・流線・面隣接・L2 誤差評価が Tet4 専用だった。
//! M7 で**面共役な四面体分割**を通すようにしたので、
//! Hex8 / Penta6 を含むメッシュでも同じことができる。
//!
//! 中心にあるのは「隣り合う要素が共有する四角形面で同じ対角線を選ぶ」こと。
//! これが崩れると
//!
//! - 等値面に隙間ができる（面積が理論値からずれる）
//! - 流線が面を越えられない（境界に届く前に止まる）
//!
//! という形で必ず現れるので、そこを見張っている。

mod common;

use common::{BOTTOM_PID, Layer, Patch, TOP_PID, box_mesh_of, slab_of};
use gfem_core::case::{
    Backend, Case, Electrode, Preconditioner, ReferencePotential, SolverOptions,
};
use gfem_core::mesh::{Navigator, TetView, iso_surface};
use gfem_core::post::{l2_error_field, l2_error_potential, mean_element_size};
use gfem_core::{ElementKind, Mesh, solve};

const SOLID_KINDS: [ElementKind; 3] = [ElementKind::Tet4, ElementKind::Hex8, ElementKind::Penta6];

const LX: f64 = 0.060;
const LY: f64 = 0.040;
const CURRENT: f64 = 2.0e-3;

/// 一辺 1 の立方体を `n` 分割した単位立方体メッシュ。
fn unit_cube(kind: ElementKind, n: usize) -> Mesh {
    let line: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64).collect();
    box_mesh_of(
        kind,
        &line,
        &line,
        &line,
        &vec![1u32; n],
        &[Patch::whole(TOP_PID, true), Patch::whole(BOTTOM_PID, false)],
    )
}

/// 3 層スラブ。
fn layered(kind: ElementKind, refine: usize) -> Mesh {
    slab_of(
        kind,
        LX,
        LY,
        6,
        4,
        &[
            Layer {
                pid: 1,
                thickness: 0.002,
                divisions: refine,
            },
            Layer {
                pid: 2,
                thickness: 0.004,
                divisions: refine,
            },
            Layer {
                pid: 3,
                thickness: 0.014,
                divisions: 2 * refine,
            },
        ],
    )
}

fn layered_case() -> Case {
    let mut case = Case::new();
    for (pid, sigma) in [(1u32, 0.10), (2, 0.04), (3, 0.35)] {
        case.set_conductivity(pid, sigma).unwrap();
    }
    case.add_electrode(Electrode::anode("top", CURRENT).with_shell_pids([TOP_PID]));
    case.add_electrode(Electrode::cathode("bottom", CURRENT).with_shell_pids([BOTTOM_PID]));
    case.solver = SolverOptions {
        tolerance: 1.0e-12,
        max_iterations: 50_000,
        preconditioner: Preconditioner::Jacobi,
        backend: Backend::CpuSerial,
        ..Default::default()
    };
    case.reference = ReferencePotential::CathodeMean;
    case
}

// --- 分割そのもの -----------------------------------------------------------

#[test]
fn the_split_fills_the_mesh_without_gaps_or_overlaps() {
    // 体積が一致することが、隙間も重なりも無いことの一番わかりやすい証拠
    for kind in SOLID_KINDS {
        let mesh = unit_cube(kind, 4);
        let tets = TetView::build(&mesh).unwrap();
        assert!(
            tets.find_inverted(&mesh).is_none(),
            "{kind:?}: 反転した四面体がある"
        );
        let volume = tets.total_volume(&mesh);
        assert!(
            (volume - 1.0).abs() < 1e-12,
            "{kind:?}: 体積 {volume}（1.0 のはず）"
        );
        assert_eq!(tets.is_identity(), kind == ElementKind::Tet4);
    }
}

#[test]
fn every_interior_face_is_shared_by_exactly_two_tetrahedra() {
    // 対角線の選び方が両側で食い違うと、面が 1 回しか現れない三角形ができる。
    // 面隣接を作って「内部の面はすべて相手を持つ」ことを確かめる
    for kind in SOLID_KINDS {
        let mesh = unit_cube(kind, 3);
        let nav = Navigator::build(&mesh).unwrap();
        let tets = nav.tets();

        // 境界（立方体の表面）にある面の枚数を数える
        let mut boundary = 0usize;
        for t in 0..tets.len() {
            for f in 0..4 {
                if nav.adjacency().neighbour(t, f).is_none() {
                    boundary += 1;
                }
            }
        }
        // 立方体の 6 面 × 3×3 セル × 三角形 2 枚 = 108 枚
        assert_eq!(
            boundary, 108,
            "{kind:?}: 相手のいない面が {boundary} 枚（境界の 108 枚だけのはず）"
        );
    }
}

// --- 等値面 -----------------------------------------------------------------

#[test]
fn an_isosurface_of_a_linear_field_is_a_flat_plane() {
    // φ = z の等値面 z = 0.5 は 1×1 の正方形。切り方が食い違えば面積がずれる
    for kind in SOLID_KINDS {
        let mesh = unit_cube(kind, 4);
        let tets = TetView::build(&mesh).unwrap();
        let values: Vec<f64> = (0..mesh.nodes.len())
            .map(|i| mesh.nodes.coord(i as u32)[2])
            .collect();
        let surface = iso_surface(&mesh, &tets, &values, 0.5, &[&values]).unwrap();
        assert!(!surface.is_empty(), "{kind:?}: 等値面が空");
        assert!(
            (surface.area() - 1.0).abs() < 1e-12,
            "{kind:?}: 面積 {}（1.0 のはず）",
            surface.area()
        );
        // 全ての頂点が z = 0.5 に載る
        for p in surface.points.chunks_exact(3) {
            assert!((p[2] - 0.5).abs() < 1e-12, "{kind:?}: 頂点 {p:?}");
        }
        // 色として載せた量も同じ補間で来る
        let scalars = &surface.scalars[0];
        assert_eq!(scalars.len(), surface.n_points());
        assert!(scalars.iter().all(|v| (v - 0.5).abs() < 1e-12));
    }
}

#[test]
fn the_isosurface_is_watertight() {
    // 隙間があると「1 度しか現れない辺」ができる。閉じた面なら
    // すべての辺がちょうど 2 枚の三角形に共有される
    for kind in SOLID_KINDS {
        let mesh = unit_cube(kind, 4);
        let tets = TetView::build(&mesh).unwrap();
        // 球状の等値面（境界を切らないように内側だけ。半径 √0.1 ≈ 0.32）
        let centre = [0.5, 0.5, 0.5];
        let values: Vec<f64> = (0..mesh.nodes.len())
            .map(|i| {
                let p = mesh.nodes.coord(i as u32);
                (0..3).map(|d| (p[d] - centre[d]).powi(2)).sum::<f64>()
            })
            .collect();
        let surface = iso_surface(&mesh, &tets, &values, 0.1, &[]).unwrap();
        assert!(!surface.is_empty(), "{kind:?}: 等値面が空");

        let mut edges: std::collections::HashMap<(u32, u32), usize> =
            std::collections::HashMap::new();
        for t in surface.triangles.chunks_exact(3) {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let key = if a < b { (a, b) } else { (b, a) };
                *edges.entry(key).or_default() += 1;
            }
        }
        let open: Vec<_> = edges.iter().filter(|&(_, &c)| c != 2).collect();
        assert!(
            open.is_empty(),
            "{kind:?}: 閉じていない辺が {} 本ある（等値面に隙間がある）",
            open.len()
        );
    }
}

#[test]
fn the_source_element_is_the_original_element_not_the_sub_tetrahedron() {
    let mesh = unit_cube(ElementKind::Hex8, 3);
    let tets = TetView::build(&mesh).unwrap();
    let values: Vec<f64> = (0..mesh.nodes.len())
        .map(|i| mesh.nodes.coord(i as u32)[2])
        .collect();
    let surface = iso_surface(&mesh, &tets, &values, 0.5, &[]).unwrap();
    assert_eq!(surface.source_elements.len(), surface.n_triangles());
    let worst = surface.source_elements.iter().copied().max().unwrap() as usize;
    assert!(
        worst < mesh.n_solid_elements(),
        "元要素の番号 {worst} が要素数 {} を超えている",
        mesh.n_solid_elements()
    );
}

// --- 流線 -------------------------------------------------------------------

#[test]
fn streamlines_cross_element_boundaries_in_every_element_type() {
    // 一様場なら、始点から境界まで真っ直ぐ抜けるはず。
    // 面の対角線が食い違うと途中で隣が見つからず、境界の手前で止まる
    for kind in SOLID_KINDS {
        let mesh = unit_cube(kind, 4);
        let nav = Navigator::build(&mesh).unwrap();
        let n = mesh.n_solid_elements();
        let field: Vec<f64> = (0..n).flat_map(|_| [1.0, 0.0, 0.0]).collect();

        let line = nav
            .trace(&mesh, &field, [0.05, 0.5, 0.5], &Default::default())
            .unwrap()
            .expect("始点はメッシュの中");
        assert_eq!(
            line.stop_reason,
            gfem_core::mesh::streamline::StopReason::Boundary,
            "{kind:?}: 境界まで届いていない"
        );
        let last = &line.points[line.points.len() - 3..];
        assert!((last[0] - 1.0).abs() < 1e-12, "{kind:?}: 終点 {last:?}");
        assert!(
            (line.length - 0.95).abs() < 1e-12,
            "{kind:?}: 長さ {}",
            line.length
        );
        // y, z はずれない
        for p in line.points.chunks_exact(3) {
            assert!((p[1] - 0.5).abs() < 1e-12, "{kind:?}: {p:?}");
            assert!((p[2] - 0.5).abs() < 1e-12, "{kind:?}: {p:?}");
        }
        // 区間が属するのは**元の要素**
        assert!(line.elements.iter().all(|&e| (e as usize) < n));
    }
}

#[test]
fn locating_a_point_agrees_with_the_element_that_contains_it() {
    for kind in SOLID_KINDS {
        let mesh = unit_cube(kind, 3);
        let nav = Navigator::build(&mesh).unwrap();
        for e in 0..mesh.n_solid_elements() {
            let element = mesh.solid(e);
            let centre = mesh.element_centroid(element.conn);
            let found = nav
                .locate_element(&mesh, centre)
                .unwrap_or_else(|| panic!("{kind:?}: 要素 {e} の重心が見つからない"));
            assert_eq!(
                found, e,
                "{kind:?}: 要素 {e} の重心が要素 {found} と判定された"
            );
        }
        assert!(nav.locate_element(&mesh, [5.0, 0.5, 0.5]).is_none());
        assert!(nav.locate_element(&mesh, [0.5, -1.0, 0.5]).is_none());
    }
}

#[test]
fn streamlines_follow_the_current_in_a_layered_slab() {
    // 実際に解いた場で、上面から入れた電流が下面まで抜けること
    for kind in SOLID_KINDS {
        let mesh = layered(kind, 1);
        let analysis = solve(&mesh, &layered_case(), None).unwrap();
        let nav = Navigator::build(&mesh).unwrap();
        // 節点面にちょうど載る点は稜線の上になるので、少しずらした一般の位置から引く
        let seeds = [[LX * 0.37, LY * 0.41, 0.0195]];
        let lines = nav
            .trace_many(
                &mesh,
                &analysis.fields.current_density,
                &seeds,
                &Default::default(),
            )
            .unwrap();
        assert_eq!(lines.len(), 1, "{kind:?}");
        let line = &lines[0];
        assert_eq!(
            line.stop_reason,
            gfem_core::mesh::streamline::StopReason::Boundary,
            "{kind:?}: 境界まで届いていない"
        );
        // 一次元問題なので z が単調に下がる
        let zs: Vec<f64> = line.points.chunks_exact(3).map(|p| p[2]).collect();
        assert!(
            zs.windows(2).all(|w| w[1] <= w[0] + 1e-12),
            "{kind:?}: z が単調に下がっていない"
        );
        assert!(
            zs.last().unwrap().abs() < 1e-9,
            "{kind:?}: 下面に届いていない"
        );
    }
}

// --- L2 誤差評価 ------------------------------------------------------------

#[test]
fn a_linear_potential_has_no_error_in_any_element_type() {
    // 1 次要素は線形場を厳密に再現する。分割しても求積が正しければ誤差はゼロ
    for kind in SOLID_KINDS {
        let mesh = unit_cube(kind, 3);
        let exact = |x: [f64; 3]| 2.0 * x[0] - 3.0 * x[1] + 0.5 * x[2] + 7.0;
        let phi: Vec<f64> = (0..mesh.nodes.len())
            .map(|i| exact(mesh.nodes.coord(i as u32)))
            .collect();
        let e = l2_error_potential(&mesh, &phi, exact).unwrap();
        assert!(e.absolute < 1e-14, "{kind:?}: {e:?}");

        // h は「分割前の要素 1 個あたりの体積」の 3 乗根。要素種で要素数が違う
        let h = mean_element_size(&mesh).unwrap();
        let expected = (1.0 / mesh.n_solid_elements() as f64).cbrt();
        assert!(
            (h - expected).abs() < 1e-12,
            "{kind:?}: h = {h}（{expected} のはず）"
        );
    }
}

#[test]
fn the_field_error_uses_the_original_element_values() {
    for kind in SOLID_KINDS {
        let mesh = unit_cube(kind, 2);
        let n = mesh.n_solid_elements();
        let field: Vec<f64> = (0..n).flat_map(|_| [1.0, -2.0, 0.5]).collect();
        let e = l2_error_field(&mesh, &field, |_| [1.0, -2.0, 0.5]).unwrap();
        assert!(e.absolute < 1e-14, "{kind:?}: {e:?}");

        // 長さが要素数と合っていなければエラー
        assert!(l2_error_field(&mesh, &field[3..], |_| [0.0; 3]).is_err());
    }
}

// --- 要素種をまたいだ一致 ---------------------------------------------------

#[test]
fn every_element_type_gives_the_same_isosurface_of_the_solved_potential() {
    // 層状スラブは一次元問題なので φ は z だけの関数。
    // どの要素種で刻んでも、同じ電位の等値面は同じ平面になるはず
    let mut areas = Vec::new();
    for kind in SOLID_KINDS {
        let mesh = layered(kind, 1);
        let analysis = solve(&mesh, &layered_case(), None).unwrap();
        let tets = TetView::build(&mesh).unwrap();
        let (lo, hi) = analysis.potential_range();
        let level = 0.5 * (lo + hi);
        let surface = iso_surface(&mesh, &tets, &analysis.potential, level, &[]).unwrap();
        assert!(!surface.is_empty(), "{kind:?}");
        areas.push((kind, surface.area()));
    }
    let reference = areas[0].1;
    for &(kind, area) in &areas {
        assert!(
            (area - reference).abs() < 1e-9 * reference,
            "{kind:?}: 面積 {area}（Tet4 は {reference}）"
        );
        // 断面は 60 × 40 mm の矩形
        assert!(
            (area - LX * LY).abs() < 1e-9 * LX * LY,
            "{kind:?}: 面積 {area}（{} のはず）",
            LX * LY
        );
    }
}

#[test]
fn a_seed_sitting_exactly_on_a_mesh_edge_still_traces() {
    // GUI は始点を格子状に置くので、節点面にぴったり載ることが実際にある。
    // 直す前は Hex8 で 10²⁹ m 先へ飛んでいた（出口面の分母がほぼゼロになる）
    for kind in SOLID_KINDS {
        let mesh = layered(kind, 1);
        let analysis = solve(&mesh, &layered_case(), None).unwrap();
        let nav = Navigator::build(&mesh).unwrap();
        // x も y も節点面の上（＝稜線の上）
        let seed = [LX * 0.5, LY * 0.5, 0.0195];
        let line = nav
            .trace(
                &mesh,
                &analysis.fields.current_density,
                seed,
                &Default::default(),
            )
            .unwrap()
            .expect("始点はメッシュの中");

        assert_eq!(
            line.stop_reason,
            gfem_core::mesh::streamline::StopReason::Boundary,
            "{kind:?}: 下面まで届いていない"
        );
        // 一次元問題なので、上面 z = 0.0195 から下面 z = 0 まで真っ直ぐ 0.0195 m
        assert!(
            (line.length - 0.0195).abs() < 1e-9,
            "{kind:?}: 長さ {}",
            line.length
        );
        for p in line.points.chunks_exact(3) {
            assert!(
                (p[0] - seed[0]).abs() < 1e-9 && (p[1] - seed[1]).abs() < 1e-9,
                "{kind:?}: 横にずれた {p:?}"
            );
            assert!(
                (-1e-9..=0.0195 + 1e-9).contains(&p[2]),
                "{kind:?}: メッシュの外へ出た {p:?}"
            );
        }
    }
}

#[test]
fn streamlines_never_leave_the_mesh() {
    // 出口面の選び方を誤ると、面を 1 枚突き抜けてメッシュの外に点が出る。
    // 目に見える壊れ方ではないが、外に出た点は物理的に意味を持たない
    for kind in SOLID_KINDS {
        let mesh = layered(kind, 1);
        let analysis = solve(&mesh, &layered_case(), None).unwrap();
        let nav = Navigator::build(&mesh).unwrap();
        let (lo, hi) = mesh.bounding_box().unwrap();

        // 上面の格子状の点から、両向きに引く
        let mut seeds = Vec::new();
        for i in 1..5 {
            for j in 1..4 {
                seeds.push([LX * i as f64 / 5.0, LY * j as f64 / 4.0, 0.0195]);
            }
        }
        for direction in [
            gfem_core::mesh::TraceDirection::Forward,
            gfem_core::mesh::TraceDirection::Backward,
        ] {
            let lines = nav
                .trace_many(
                    &mesh,
                    &analysis.fields.current_density,
                    &seeds,
                    &gfem_core::mesh::TraceOptions {
                        direction,
                        ..Default::default()
                    },
                )
                .unwrap();
            assert_eq!(lines.len(), seeds.len(), "{kind:?} {direction:?}");
            for line in &lines {
                for p in line.points.chunks_exact(3) {
                    for d in 0..3 {
                        assert!(
                            p[d] >= lo[d] - 1e-9 && p[d] <= hi[d] + 1e-9,
                            "{kind:?} {direction:?}: メッシュの外に出た {p:?}（範囲 {lo:?}..{hi:?}）"
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "計測用。--ignored で実行する"]
fn measure_seed_location() {
    // M6 までは始点の探索が全要素走査だった（SPEC §8 M7）。
    // 1000 万要素で流線を 100 本引くと、それだけで数十秒かかっていた
    use std::time::Instant;
    println!(
        "{:>10} {:>10} {:>10} {:>12} {:>12} {:>8}",
        "節点", "四面体", "構築[s]", "格子[s]", "全走査[s]", "倍率"
    );
    for (nx, ny, refine) in [(30, 20, 2), (60, 40, 3), (120, 80, 4)] {
        let mesh = slab_of(
            ElementKind::Tet4,
            LX,
            LY,
            nx,
            ny,
            &[
                Layer {
                    pid: 1,
                    thickness: 0.002,
                    divisions: refine,
                },
                Layer {
                    pid: 2,
                    thickness: 0.004,
                    divisions: refine,
                },
                Layer {
                    pid: 3,
                    thickness: 0.014,
                    divisions: 2 * refine,
                },
            ],
        );

        let t = Instant::now();
        let nav = Navigator::build(&mesh).unwrap();
        let build = t.elapsed().as_secs_f64();
        let tets = nav.tets();

        let seeds: Vec<[f64; 3]> = (0..200)
            .map(|i| {
                let f = i as f64 / 200.0;
                [LX * (0.1 + 0.8 * f), LY * 0.37, 0.0195 * (0.05 + 0.9 * f)]
            })
            .collect();

        let t = Instant::now();
        let with_grid: Vec<Option<usize>> = seeds
            .iter()
            .map(|&s| nav.locate_element(&mesh, s))
            .collect();
        let grid = t.elapsed().as_secs_f64();

        // M6 までのやり方（全四面体を昇順に走査）
        let t = Instant::now();
        let by_scan: Vec<Option<usize>> = seeds
            .iter()
            .map(|&x| {
                (0..tets.len())
                    .find(|&e| {
                        let p = tets.coords(&mesh, e);
                        inside(&p, x)
                    })
                    .map(|e| tets.parent(e) as usize)
            })
            .collect();
        let scan = t.elapsed().as_secs_f64();

        // 速いだけでなく、同じ答えが出ること
        assert_eq!(with_grid, by_scan, "格子と全走査で答えが違う");

        println!(
            "{:>10} {:>10} {:>10.3} {:>12.4} {:>12.4} {:>8.1}",
            mesh.nodes.len(),
            tets.len(),
            build,
            grid,
            scan,
            scan / grid.max(1e-12)
        );
    }
}

/// 点が四面体の中にあるか（全走査版の判定。`Navigator::locate_tet` と同じ規則）。
fn inside(p: &[[f64; 3]; 4], x: [f64; 3]) -> bool {
    let volume = gfem_core::element::tet4_signed_volume(p);
    if volume <= 0.0 {
        return false;
    }
    // 各面の対頂点側にあるかを符号付き体積で見る
    for i in 0..4 {
        let mut q = *p;
        q[i] = x;
        let sub = gfem_core::element::tet4_signed_volume(&q);
        if sub / volume < -1.0e-12 {
            return false;
        }
    }
    true
}
