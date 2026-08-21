//! Nastran バルクデータのパーサ（SPEC §3.1）。
//!
//! 1000 万節点を実用時間で読むため自前実装にしている（pyNastran 等は使わない）。
//!
//! - [`coord`] — CORD2R / CORD2C / CORD2S による座標変換
//! - [`field`] — 1 行のフィールド分解と Fortran 由来の数値表記の解釈
//! - [`reader`] — 物理行 → 論理カード（継続行・書式混在・セクションの吸収）
//! - [`parse`] — 論理カード → [`crate::mesh::Mesh`]

pub mod coord;
pub mod field;
pub mod parse;
pub mod reader;

pub use coord::{CoordKind, CoordSystem, CoordSystems};
pub use parse::{CardCount, ReadReport, read_nastran, read_nastran_from};
pub use reader::{BulkReader, Card};
