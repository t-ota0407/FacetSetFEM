//! 入出力。
//!
//! 内部のメッシュ表現をファイル形式から独立させ、形式ごとにモジュールを分ける
//! （SPEC §9「他のメッシュ形式」への備え）。

pub mod nastran;
pub mod vtu;

pub use nastran::{ReadReport, read_nastran};
pub use vtu::{VtuData, VtuOptions, write_mesh_vtu, write_vtu};
