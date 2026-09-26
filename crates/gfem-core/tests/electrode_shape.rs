//! 矩形（角丸可）の電極パッチ（SPEC §11.1）。
//!
//! 実験で使う電極は 2 cm × 2 cm の正方形のような矩形が多い。円のパッチしか無いと
//! 形も面積も合わせられないので、中心・寸法・角の半径・向きで矩形を選べるようにした。
//! ここでは、平らな体表面で**指定どおりの面積・形・向き**が選ばれることを確かめる。

mod common;

use common::uniform_slab;
use gfem_core::case::{FaceLookup, PatchShape, RectShape, SurfacePatch};
use gfem_core::mesh::SurfacePicker;
use gfem_core::mesh::topology::extract_surface;

/// 10 cm 角・厚さ 2 cm のスラブ。上面は 1 mm 間隔の格子を四面体に割った三角形。
fn picker() -> SurfacePicker {
    let mesh = uniform_slab(0.1, 0.1, 0.02, 100, 100, 2);
    let surface = extract_surface(&mesh).unwrap();
    SurfacePicker::build(&mesh, &surface)
}

/// 上面の中央。
const CENTER: [f64; 3] = [0.05, 0.05, 0.02];

fn rect(width: f64, height: f64, corner_radius: f64, axis: [f64; 3], angle: f64) -> RectShape {
    RectShape {
        width,
        height,
        corner_radius,
        axis,
        angle,
    }
}

fn select(picker: &SurfacePicker, r: RectShape) -> Vec<u32> {
    SurfacePatch::rect(CENTER, r, None).select(picker).unwrap()
}

#[test]
fn a_square_has_the_area_it_was_given() {
    // 2 cm × 2 cm。格子の線に辺が載らないよう中心を少しずらしても、面積は数 % 以内
    let picker = picker();
    let r = rect(0.02, 0.02, 0.0, [1.0, 0.0, 0.0], 0.0);
    let faces = select(&picker, r);
    let area = picker.total_area(&faces);
    let nominal = SurfacePatch::rect(CENTER, r, None).nominal_area();
    assert!((nominal - 4.0e-4).abs() < 1e-15);
    assert!(
        (area - nominal).abs() / nominal < 0.03,
        "面積 {area} m²（指定 {nominal} m²）"
    );
    // 選ばれた面はどれも上面にあり、正方形の中（縁の面 1 枚ぶんの余裕）に収まる
    for &f in &faces {
        let c = picker.centroid(f);
        assert!(
            (c[2] - 0.02).abs() < 1e-12,
            "上面でない面が入っている: {c:?}"
        );
        assert!((c[0] - CENTER[0]).abs() <= 0.01 && (c[1] - CENTER[1]).abs() <= 0.01);
    }
}

#[test]
fn a_rectangle_follows_its_axis_and_angle() {
    // 30 mm × 10 mm を x 向きに置き、90° 回すと y 向きに置いたのと同じ面になる
    let picker = picker();
    let along_x = select(&picker, rect(0.03, 0.01, 0.0, [1.0, 0.0, 0.0], 0.0));
    let along_y = select(&picker, rect(0.03, 0.01, 0.0, [0.0, 1.0, 0.0], 0.0));
    let turned = select(&picker, rect(0.03, 0.01, 0.0, [1.0, 0.0, 0.0], 90.0));
    assert_eq!(turned, along_y, "90° 回すと y 向きと同じ面になるはず");
    assert_ne!(along_x, along_y);

    // x 向きなら x に長く、y に短い
    let extent = |faces: &[u32], k: usize| {
        let v: Vec<f64> = faces.iter().map(|&f| picker.centroid(f)[k]).collect();
        v.iter().cloned().fold(f64::MIN, f64::max) - v.iter().cloned().fold(f64::MAX, f64::min)
    };
    assert!(extent(&along_x, 0) > 0.025 && extent(&along_x, 1) < 0.011);

    // 向きの軸は長さも、法線方向の成分も問わない（接平面に投影して使う）
    let skewed = select(&picker, rect(0.03, 0.01, 0.0, [2.0, 0.0, 0.7], 0.0));
    assert_eq!(skewed, along_x);
}

#[test]
fn a_positive_angle_turns_counterclockwise_seen_from_outside() {
    // 上面の外向き法線は +z。外（上）から見て反時計回りに 90° 回すと、x 軸は +y を向く
    let picker = picker();
    let r = rect(0.02, 0.01, 0.0, [1.0, 0.0, 0.0], 90.0);
    let (frame, _) = SurfacePatch::rect(CENTER, r, None)
        .outline(&picker)
        .unwrap()
        .unwrap();
    assert!(
        (frame.normal[2] - 1.0).abs() < 1e-12,
        "法線 {:?}",
        frame.normal
    );
    assert!((frame.u[1] - 1.0).abs() < 1e-12, "u {:?}", frame.u);
    assert!((frame.v[0] + 1.0).abs() < 1e-12, "v {:?}", frame.v);
}

#[test]
fn rounded_corners_take_the_corners_off() {
    // 角の半径 5 mm の 2 cm 角は、面積が (4 − π) r² だけ減り、角の面が入らない
    let picker = picker();
    let sharp = select(&picker, rect(0.02, 0.02, 0.0, [1.0, 0.0, 0.0], 0.0));
    let round = select(&picker, rect(0.02, 0.02, 0.005, [1.0, 0.0, 0.0], 0.0));
    let r = rect(0.02, 0.02, 0.005, [1.0, 0.0, 0.0], 0.0);
    let nominal = r.area();
    assert!((nominal - (4.0e-4 - (4.0 - std::f64::consts::PI) * 25.0e-6)).abs() < 1e-15);
    let area = picker.total_area(&round);
    assert!(
        (area - nominal).abs() / nominal < 0.03,
        "面積 {area} m²（指定 {nominal} m²）"
    );
    assert!(round.len() < sharp.len());
    assert!(
        round.iter().all(|f| sharp.contains(f)),
        "角丸は尖った矩形の内側に収まる"
    );
    // 角の点（中心から 9.5 mm, 9.5 mm）の近くの面は入らない
    assert!(round.iter().all(|&f| {
        let c = picker.centroid(f);
        let (dx, dy) = ((c[0] - CENTER[0]).abs(), (c[1] - CENTER[1]).abs());
        !(dx > 0.009 && dy > 0.009)
    }));
    // 角の半径を短い辺の半分にした正方形は円になる
    let disc = select(&picker, rect(0.02, 0.02, 0.01, [1.0, 0.0, 0.0], 0.0));
    let circle = SurfacePatch::circle(CENTER, 0.01, None)
        .select(&picker)
        .unwrap();
    let (a, b) = (picker.total_area(&disc), picker.total_area(&circle));
    assert!((a - b).abs() / b < 0.02, "角丸の円 {a} m² と円 {b} m²");
}

#[test]
fn a_rectangle_does_not_wrap_around_an_edge() {
    // 上面の縁に大きな矩形を置いても、側面（法線が直交）には回り込まない。
    // 円は 3 次元の距離で選ぶので回り込む——耳の裏を拾わないための違い
    let picker = picker();
    let edge = [0.05, 0.001, 0.02];
    let faces = SurfacePatch::rect(edge, rect(0.02, 0.02, 0.0, [1.0, 0.0, 0.0], 0.0), None)
        .select(&picker)
        .unwrap();
    assert!(
        faces
            .iter()
            .all(|&f| (picker.centroid(f)[2] - 0.02).abs() < 1e-12)
    );
    let circle = SurfacePatch::circle(edge, 0.01, None)
        .select(&picker)
        .unwrap();
    assert!(circle.iter().any(|&f| picker.centroid(f)[2] < 0.0199));
}

#[test]
fn an_axis_along_the_normal_is_refused_with_a_hint() {
    // 向きの軸が法線と平行だと辺の向きが決まらない。黙って決めずに、直し方を示す
    let picker = picker();
    let err = SurfacePatch::rect(CENTER, rect(0.02, 0.02, 0.0, [0.0, 0.0, 1.0], 0.0), None)
        .select(&picker)
        .unwrap_err()
        .to_string();
    assert!(err.contains("ほぼ平行"), "{err}");
    assert!(err.contains("正面から見る"), "{err}");
}

#[test]
fn bad_dimensions_are_refused() {
    let picker = picker();
    for (r, needle) in [
        (rect(0.0, 0.02, 0.0, [1.0, 0.0, 0.0], 0.0), "寸法"),
        (rect(0.02, f64::NAN, 0.0, [1.0, 0.0, 0.0], 0.0), "寸法"),
        (rect(0.02, 0.01, 0.006, [1.0, 0.0, 0.0], 0.0), "角の半径"),
        (rect(0.02, 0.01, -0.001, [1.0, 0.0, 0.0], 0.0), "角の半径"),
        (rect(0.02, 0.01, 0.0, [0.0, 0.0, 0.0], 0.0), "向きの軸"),
        (
            rect(0.02, 0.01, 0.0, [1.0, 0.0, 0.0], f64::INFINITY),
            "角度",
        ),
    ] {
        let err = SurfacePatch::rect(CENTER, r, None)
            .select(&picker)
            .unwrap_err()
            .to_string();
        assert!(err.contains(needle), "{needle}: {err}");
    }
}

#[test]
fn the_selection_is_deterministic_and_sorted() {
    let picker = picker();
    let r = rect(0.02, 0.015, 0.003, [1.0, 0.3, 0.0], 17.0);
    let a = select(&picker, r);
    let b = select(&picker, r);
    assert_eq!(a, b);
    assert!(a.windows(2).all(|w| w[0] < w[1]), "昇順・重複なしのこと");
}

#[test]
fn a_rectangle_electrode_resolves_like_any_patch() {
    // 電極面の解決（ソルバが荷重を作る経路）でも同じ面が選ばれる
    let mesh = uniform_slab(0.1, 0.1, 0.02, 100, 100, 2);
    let surface = extract_surface(&mesh).unwrap();
    let lookup = FaceLookup::new(&mesh, &surface);
    let r = rect(0.02, 0.02, 0.002, [1.0, 0.0, 0.0], 30.0);
    let patch = SurfacePatch::rect(CENTER, r, None);
    let electrode = gfem_core::case::Electrode::anode("E1", 1.0e-3).with_patches([patch.clone()]);
    let resolved = gfem_core::case::resolve_faces(&lookup, &electrode).unwrap();
    let picked = patch
        .select(&SurfacePicker::build(&mesh, &surface))
        .unwrap();
    assert_eq!(resolved.len(), picked.len());
    assert!(matches!(patch.shape, PatchShape::Rect(_)));
}
