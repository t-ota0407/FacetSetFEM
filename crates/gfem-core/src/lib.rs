//! GenericFEM のコア。
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
//! # 現在の実装範囲（M0）
//!
//! Nastran バルクデータの読み込みと VTU 出力まで。
//! アセンブリ・ソルバ・ポスト処理は M1 以降で追加する（SPEC §8）。

pub mod analysis;
pub mod assemble;
pub mod case;
pub mod element;
pub mod error;
pub mod io;
pub mod material;
pub mod mesh;
pub mod post;
pub mod provenance;
pub mod solver;
pub mod units;

pub use analysis::{Analysis, Timings, solve, solve_with, write_analysis_vtu};
pub use assemble::{CsrMatrix, LinearSystem, assemble};
pub use case::{Case, Electrode, ElectrodeRole, Preconditioner, ReferencePotential, SolverOptions};
pub use element::ElementKind;
pub use error::{ErrorKind, GfemError, Location, Result};
pub use io::nastran::{ReadReport, read_nastran};
pub use io::vtu::{VtuData, VtuOptions, write_mesh_vtu, write_vtu};
pub use material::Conductivity;
pub use mesh::{ElementBlock, ElementRef, Mesh, Nodes, Property};
pub use post::{ConservationReport, ErrorNorms, FieldResult, convergence_order};
pub use provenance::Provenance;
pub use solver::{BackendSolver, CgReport, CpuParallel, CpuSerial, LinearBackend, LinearSolver};
pub use units::LengthUnit;

/// gfem-core のバージョン。出力 JSON のプロビナンス記録に使う（SPEC §5.5）。
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
