//! 表示言語の切り替え（SPEC §6.6）。
//!
//! 代表的なエラー・警告・レポートが、英語では日本語を 1 文字も含まず、
//! 日本語では従来の文言のまま出ることを確かめる。
//!
//! `with_language` はスレッドごとの上書きなので、並列に走るほかのテストの言語は変えない。
//! ただし rayon のワーカースレッドには引き継がれないので、ここでは逐次で組み立て・求解
//! する規模と設定を使う（並列の経路で作られる文言は、プロセス全体の言語で出る）。

mod common;

use common::{BOTTOM_PID, Layer, TOP_PID, slab};
use gfem_core::case::{Backend, Case, Electrode, Preconditioner, SolverOptions};
use gfem_core::io::nastran::read_nastran_from;
use gfem_core::units::LengthUnit;
use gfem_core::{ErrorKind, GfemError, Language, Mesh, solve, with_language};

/// 日本語の文字（かな・漢字・全角記号）を含むか。
fn has_japanese(s: &str) -> bool {
    s.chars().any(|c| {
        matches!(c,
            '\u{3000}'..='\u{30FF}' | '\u{3400}'..='\u{9FFF}' | '\u{FF00}'..='\u{FFEF}')
    })
}

/// 英語で出たことの確認（日本語が混ざっていない・見出しが英語）。
fn assert_english(e: &GfemError) -> String {
    let s = e.to_string();
    assert!(!has_japanese(&s), "日本語が混ざっている:\n{s}");
    s
}

fn read(text: &str) -> gfem_core::Result<(Mesh, gfem_core::ReadReport)> {
    read_nastran_from(text.as_bytes(), "model.bdf", LengthUnit::Meter)
}

/// 一様な 2 層スラブ（上面に陽極、下面に陰極のシェル）。
fn two_layer() -> Mesh {
    slab(
        0.02,
        0.02,
        2,
        2,
        &[
            Layer {
                pid: 1,
                thickness: 0.002,
                divisions: 1,
            },
            Layer {
                pid: 2,
                thickness: 0.004,
                divisions: 2,
            },
        ],
    )
}

fn complete_case() -> Case {
    let mut case = Case::new();
    case.set_conductivity(1, 0.1).unwrap();
    case.set_conductivity(2, 0.35).unwrap();
    case.add_electrode(Electrode::anode("top", 1.0e-3).with_shell_pids([TOP_PID]));
    case.add_electrode(Electrode::cathode("bottom", 1.0e-3).with_shell_pids([BOTTOM_PID]));
    case.solver = SolverOptions {
        tolerance: 1.0e-12,
        max_iterations: 10_000,
        preconditioner: Preconditioner::Jacobi,
        backend: Backend::CpuSerial,
        ..Default::default()
    };
    case
}

#[test]
fn a_nastran_syntax_error_comes_out_in_the_chosen_language() {
    let text = "GRID,1,,1.0.0,0.0,0.0\n";
    let en = with_language(Language::En, || read(text).unwrap_err());
    assert_eq!(en.kind(), ErrorKind::Syntax);
    let s = with_language(Language::En, || assert_english(&en));
    assert!(
        s.starts_with("[Syntax error] Cannot interpret as a real number"),
        "{s}"
    );
    assert!(s.contains("Where: model.bdf:1 GRID card field 3"), "{s}");
    assert!(s.contains("Fix: Write it in a Nastran real format"), "{s}");

    let ja = with_language(Language::Ja, || read(text).unwrap_err());
    let s = with_language(Language::Ja, || ja.to_string());
    assert!(
        s.starts_with("[構文エラー] 実数として解釈できません: \"1.0.0\""),
        "{s}"
    );
    assert!(
        s.contains("場所: model.bdf:1 GRID カード フィールド 3"),
        "{s}"
    );
    assert!(s.contains("対処: Nastran の実数形式"), "{s}");
}

#[test]
fn the_load_report_and_its_warnings_follow_the_language() {
    // PSOLID の無い PID と、どの要素からも参照されない節点がある
    let text = "\
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,0.0,1.0,0.0
GRID,4,,0.0,0.0,1.0
GRID,5,,5.0,5.0,5.0
CTETRA,10,1,1,2,3,4
FOO,1
";
    let en = with_language(Language::En, || {
        let (_, report) = read(text).unwrap();
        report.to_string()
    });
    assert!(!has_japanese(&en), "{en}");
    assert!(en.starts_with("Read: model.bdf ("), "{en}");
    assert!(
        en.contains("[unsupported] FOO ×1 (first seen at line 7)"),
        "{en}"
    );
    assert!(en.contains("[warning] Some isolated nodes"), "{en}");
    assert!(en.contains("no matching PSOLID card"), "{en}");

    let ja = with_language(Language::Ja, || {
        let (_, report) = read(text).unwrap();
        report.to_string()
    });
    assert!(ja.starts_with("読み込み: model.bdf （"), "{ja}");
    assert!(
        ja.contains("[未対応] FOO 1 件（最初の出現: 7 行目）"),
        "{ja}"
    );
    assert!(
        ja.contains("[警告] どの要素からも参照されていない孤立節点"),
        "{ja}"
    );
}

#[test]
fn a_missing_conductivity_is_reported_in_the_chosen_language() {
    let mesh = two_layer();
    let mut case = Case::new();
    case.set_conductivity(1, 0.1).unwrap();
    case.add_electrode(Electrode::anode("top", 1.0e-3).with_shell_pids([TOP_PID]));
    case.add_electrode(Electrode::cathode("bottom", 1.0e-3).with_shell_pids([BOTTOM_PID]));
    case.solver.backend = Backend::CpuSerial;

    let en = with_language(Language::En, || solve(&mesh, &case, None).unwrap_err());
    assert_eq!(en.kind(), ErrorKind::Case);
    let s = with_language(Language::En, || assert_english(&en));
    assert!(
        s.starts_with("[Setup error] Some regions have no conductivity (1): PID=2"),
        "{s}"
    );
    assert!(
        s.contains("Fix: Assign a conductivity σ [S/m] to every PSOLID PID"),
        "{s}"
    );

    let ja = with_language(Language::Ja, || solve(&mesh, &case, None).unwrap_err());
    let s = with_language(Language::Ja, || ja.to_string());
    assert!(
        s.starts_with("[設定エラー] 導電率が未設定の領域があります（1 件）: PID=2"),
        "{s}"
    );
    assert!(
        s.contains("対処: 全ての PSOLID の PID に σ [S/m] を割り当ててください"),
        "{s}"
    );
}

#[test]
fn non_convergence_is_reported_in_the_chosen_language() {
    let mesh = two_layer();
    let mut case = complete_case();
    case.solver.max_iterations = 1;

    let en = with_language(Language::En, || solve(&mesh, &case, None).unwrap_err());
    assert_eq!(en.kind(), ErrorKind::Convergence);
    let s = with_language(Language::En, || assert_english(&en));
    assert!(
        s.starts_with(
            "[Convergence error] CG did not converge: Reached the maximum number of iterations, 1"
        ),
        "{s}"
    );
    assert!(
        s.contains("Fix: Increase the maximum number of iterations"),
        "{s}"
    );
    assert!(s.contains("(currently 1e-12)"), "{s}");

    let ja = with_language(Language::Ja, || solve(&mesh, &case, None).unwrap_err());
    let s = with_language(Language::Ja, || ja.to_string());
    assert!(
        s.starts_with("[収束エラー] CG が収束しませんでした: 最大反復回数 1 に達しました"),
        "{s}"
    );
    assert!(s.contains("（現在 1e-12）"), "{s}");
}

#[test]
fn cancellation_is_reported_in_the_chosen_language() {
    let mesh = two_layer();
    let case = complete_case();
    let run = || {
        let mut stop = |_: usize, _: f64| false;
        solve(&mesh, &case, Some(&mut stop)).unwrap_err()
    };

    let en = with_language(Language::En, run);
    assert_eq!(en.kind(), ErrorKind::Cancelled);
    let s = with_language(Language::En, || assert_english(&en));
    assert!(
        s.starts_with("[Cancelled] CG did not converge: Cancelled by the user"),
        "{s}"
    );
    assert!(
        s.contains("Fix: The solution at the point of cancellation cannot be used"),
        "{s}"
    );

    let ja = with_language(Language::Ja, run);
    let s = with_language(Language::Ja, || ja.to_string());
    assert!(
        s.starts_with("[中断] CG が収束しませんでした: 利用者によって中断されました"),
        "{s}"
    );
}

#[test]
fn the_message_is_fixed_when_the_error_is_made() {
    // 文言は作った時点の言語で組み立てる（見出しだけは表示するときの言語）
    let e = with_language(Language::En, || LengthUnit::parse("furlong").unwrap_err());
    assert_eq!(
        e.message(),
        "Cannot interpret as a length unit: \"furlong\""
    );
    assert_eq!(e.hint(), Some("Specify one of mm / cm / m / inch"));
    let e = with_language(Language::Ja, || LengthUnit::parse("furlong").unwrap_err());
    assert_eq!(e.message(), "長さ単位として解釈できません: \"furlong\"");
}
