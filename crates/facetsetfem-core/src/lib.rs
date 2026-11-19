//! FacetSetFEM のコア。
//!
//! 経皮電気刺激の準静的電流場 ∇·(σ∇φ) = 0 を有限要素法で解く（SPEC §2）。
//!
//! # 設計上の約束
//!
//! - このクレートは **CUDA にも Python にも依存しない**。CPU だけで完全にテストできる
//!   （CLAUDE.md 設計原則 5 / SPEC §7）
//! - 全ての計算は f64（設計原則 2）
//! - 内部の単位は必ず SI（m, S/m, A）。単位換算は入力を読む時点で済ませる（設計原則 7）
//! - 同一入力からはビット単位で同一の結果を出す（設計原則 1）
//!
//! # 入口
//!
//! 読み込み [`read_nastran`] → ケースの組み立て [`Case`] → 解く [`solve`]
//! （GPU などの外部ソルバは [`solve_with`]）→ 結果 [`Analysis`]（VTU・統計・実行サマリ）。
//! 流れ全体は SPEC §4.1、各モジュールの役割は SPEC §7 を参照。

pub mod analysis;
pub mod assemble;
pub mod case;
pub mod contact;
pub mod element;
pub mod error;
mod geom;
pub mod i18n;
pub mod io;
pub mod material;
pub mod mesh;
mod parallel;
pub mod post;
pub mod provenance;
pub mod solver;
#[cfg(test)]
mod testing;
pub mod units;

pub use analysis::{Analysis, Timings, solve, solve_prepared, solve_with, write_analysis_vtu};
pub use assemble::{CsrMatrix, LinearSystem, assemble};
pub use case::{
    Case, Electrode, ElectrodeModel, ElectrodeRole, Preconditioner, ReferencePotential,
    SolverOptions,
};
pub use element::{ElementKind, WedgeOrder};
pub use error::{ErrorKind, FacetSetFemError, Location, Result};
pub use i18n::{Language, language, set_language, with_language};
pub use io::nastran::{ReadReport, read_nastran};
pub use io::vtu::{VtuData, VtuOptions, write_mesh_vtu, write_vtu};
pub use material::Conductivity;
pub use mesh::{ElementBlock, ElementRef, Mesh, Nodes, Property};
pub use post::{ConservationReport, ErrorNorms, FieldResult, convergence_order};
pub use provenance::Provenance;
pub use solver::{
    BackendSolver, CgReport, CpuParallel, CpuSerial, Gauge, LinearBackend, LinearSolver,
};
pub use units::LengthUnit;

/// facetsetfem-core のバージョン。出力 JSON のプロビナンス記録に使う（SPEC §5.5）。
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
