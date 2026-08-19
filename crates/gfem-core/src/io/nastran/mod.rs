//! Nastran バルクデータのパーサ（SPEC §3.1）。
//!
//! 1000 万節点を実用時間で読むため自前実装にしている（pyNastran 等は使わない）。
//!
//! - [`field`]: 1 行のフィールド分解と Fortran 由来の数値表記の解釈
//! - [`reader`]: 物理行 → 論理カード（継続行・書式混在・セクションの吸収）
//! - [`parse`]: 論理カード → [`crate::mesh::Mesh`]

pub mod field;
pub mod parse;
pub mod reader;

pub use parse::{CardCount, ReadReport, read_nastran, read_nastran_from};
pub use reader::{BulkReader, Card};
