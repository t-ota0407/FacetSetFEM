//! ユニットテストで使う小さなメッシュ（`cfg(test)` のときだけ入る）。
//!
//! 以前は同じ Kuhn 分割の生成が、各モジュールのテストに 4 か所コピーされていた。
//! 結合テスト（`tests/`）は別のクレートとしてビルドされるので、ここは使えない。
//! そちらは `tests/common` にある。

use crate::io::nastran::read_nastran_from;
use crate::mesh::Mesh;
use crate::units::LengthUnit;
use std::path::Path;

/// 立方体を 6 個の四面体に分ける Kuhn 分割（局所節点 `c = bx + 2 by + 4 bz`）。
///
/// どの四面体も対角（局所節点 0 と 7）を共有し、正の体積になる向きで並べてある。
/// 隣り合う立方体の共有面は同じ対角線で割られるので、格子全体で適合メッシュになる。
const KUHN: [[usize; 4]; 6] = [
    [0, 1, 3, 7],
    [0, 1, 7, 5],
    [0, 5, 7, 4],
    [0, 3, 2, 7],
    [0, 6, 4, 7],
    [0, 2, 6, 7],
];

/// `n[0] × n[1] × n[2]` の格子を Kuhn 分割した四面体メッシュ。
///
/// 原点から各方向に長さ `size` [m] の直方体。節点 ID は `i + (nx+1)(j + (ny+1) k) + 1`、
/// 要素 ID は 1 から格子の順（x → y → z、各立方体の中は [`KUHN`] の順）。PID は 1。
pub(crate) fn kuhn_box(n: [usize; 3], size: [f64; 3]) -> Mesh {
    let [nx, ny, nz] = n;
    let node = |i: usize, j: usize, k: usize| i + (nx + 1) * (j + (ny + 1) * k) + 1;
    let mut deck = String::from("PSOLID,1,1\n");
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                deck.push_str(&format!(
                    "GRID,{},,{},{},{}\n",
                    node(i, j, k),
                    size[0] * i as f64 / nx as f64,
                    size[1] * j as f64 / ny as f64,
                    size[2] * k as f64 / nz as f64
                ));
            }
        }
    }
    let mut eid = 1;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let corner = |c: usize| node(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));
                for t in KUHN {
                    deck.push_str(&format!(
                        "CTETRA,{eid},1,{},{},{},{}\n",
                        corner(t[0]),
                        corner(t[1]),
                        corner(t[2]),
                        corner(t[3])
                    ));
                    eid += 1;
                }
            }
        }
    }
    read_nastran_from(deck.as_bytes(), Path::new("<test>"), LengthUnit::Meter)
        .expect("テスト用のメッシュは読める")
        .0
}

#[test]
fn the_kuhn_box_fills_its_volume_with_positive_tetrahedra() {
    let mesh = kuhn_box([3, 2, 2], [3.0, 2.0, 1.0]);
    assert_eq!(mesh.n_solid_elements(), 3 * 2 * 2 * 6);
    let volumes = mesh.tet4_signed_volumes();
    assert!(volumes.iter().all(|&v| v > 0.0), "負の体積の四面体がある");
    let total: f64 = volumes.iter().sum();
    assert!((total - 6.0).abs() < 1e-12, "体積 {total}");
}
