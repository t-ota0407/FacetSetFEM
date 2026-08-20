//! テスト用のメッシュ生成。
//!
//! 直方体を六面体格子に切り、**Kuhn の 6 分割**で四面体化する。全ての立方体で
//! 同じ主対角線を使うので、隣り合う立方体の共有面が同じ対角線で割れ、
//! メッシュ全体が適合（conforming）になる。
//!
//! 上面（z = 最大）に PSHELL 101、下面（z = 0）に PSHELL 201 のシェルを貼るので、
//! そのまま「上下面に電極を置いた一次元問題」の解析解と突き合わせられる（SPEC §8 M2）。

use gfem_core::element::ElementKind;
use gfem_core::mesh::{ElementBlock, Mesh, Nodes, PropId, Property};
use gfem_core::units::LengthUnit;

/// 上面（陽極側）に貼るシェルの PID。
pub const TOP_PID: PropId = 101;
/// 下面（陰極側）に貼るシェルの PID。
pub const BOTTOM_PID: PropId = 201;

/// z 方向に積む層。
#[derive(Debug, Clone, Copy)]
pub struct Layer {
    /// PSOLID の PID
    pub pid: PropId,
    /// 厚さ [m]
    pub thickness: f64,
    /// z 方向の分割数
    pub divisions: usize,
}

/// Kuhn の 6 分割（局所節点番号 n = bx + 2·by + 4·bz）。
const KUHN: [[usize; 4]; 6] = [
    [0, 1, 3, 7],
    [0, 1, 5, 7],
    [0, 2, 3, 7],
    [0, 2, 6, 7],
    [0, 4, 5, 7],
    [0, 4, 6, 7],
];

/// 直方体の多層四面体メッシュを作る。座標は m。
pub fn slab(lx: f64, ly: f64, nx: usize, ny: usize, layers: &[Layer]) -> Mesh {
    // --- 節点 ---
    let mut z_levels = vec![0.0f64];
    let mut layer_of_cell: Vec<PropId> = Vec::new();
    for layer in layers {
        let z0 = *z_levels.last().unwrap();
        for k in 1..=layer.divisions {
            z_levels.push(z0 + layer.thickness * k as f64 / layer.divisions as f64);
            layer_of_cell.push(layer.pid);
        }
    }
    let nz = z_levels.len() - 1;
    let xs: Vec<f64> = (0..=nx).map(|i| lx * i as f64 / nx as f64).collect();
    let ys: Vec<f64> = (0..=ny).map(|j| ly * j as f64 / ny as f64).collect();

    let nid = |i: usize, j: usize, k: usize| -> u32 { (i + (nx + 1) * (j + (ny + 1) * k)) as u32 };

    let mut coords = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1) * 3);
    for &z in &z_levels {
        for &y in &ys {
            for &x in &xs {
                coords.push(x);
                coords.push(y);
                coords.push(z);
            }
        }
    }
    let n_nodes = coords.len() / 3;
    let ids: Vec<u32> = (1..=n_nodes as u32).collect();

    // --- 四面体 ---
    let mut tet_conn: Vec<u32> = Vec::new();
    let mut tet_pids: Vec<PropId> = Vec::new();
    let mut tet_ids: Vec<u32> = Vec::new();
    let mut next_eid = 1u32;
    for (k, &pid) in layer_of_cell.iter().enumerate() {
        for j in 0..ny {
            for i in 0..nx {
                let corner: [u32; 8] = std::array::from_fn(|n| {
                    nid(i + (n & 1), j + ((n >> 1) & 1), k + ((n >> 2) & 1))
                });
                for t in &KUHN {
                    let mut q = [corner[t[0]], corner[t[1]], corner[t[2]], corner[t[3]]];
                    if signed_volume(&coords, &q) < 0.0 {
                        q.swap(2, 3);
                    }
                    tet_conn.extend_from_slice(&q);
                    tet_pids.push(pid);
                    tet_ids.push(next_eid);
                    next_eid += 1;
                }
            }
        }
    }

    // --- 上下面のシェル ---
    // Kuhn 分割では z = 一定の面が (i,j)-(i+1,j+1) の対角線で割れる
    let mut tri_conn: Vec<u32> = Vec::new();
    let mut tri_pids: Vec<PropId> = Vec::new();
    let mut tri_ids: Vec<u32> = Vec::new();
    for (k, pid, flip) in [(nz, TOP_PID, false), (0, BOTTOM_PID, true)] {
        for j in 0..ny {
            for i in 0..nx {
                let (a, b, c, d) = (
                    nid(i, j, k),
                    nid(i + 1, j, k),
                    nid(i + 1, j + 1, k),
                    nid(i, j + 1, k),
                );
                let faces = if flip {
                    [[a, c, b], [a, d, c]]
                } else {
                    [[a, b, c], [a, c, d]]
                };
                for f in faces {
                    tri_conn.extend_from_slice(&f);
                    tri_pids.push(pid);
                    tri_ids.push(next_eid);
                    next_eid += 1;
                }
            }
        }
    }

    let mut solid_properties: Vec<Property> = layers
        .iter()
        .map(|l| Property {
            pid: l.pid,
            mid: None,
            card: "PSOLID",
        })
        .collect();
    solid_properties.dedup_by_key(|p| p.pid);

    Mesh {
        nodes: Nodes::new(ids, coords),
        solids: vec![ElementBlock::new(
            ElementKind::Tet4,
            tet_ids,
            tet_pids,
            tet_conn,
        )],
        shells: vec![ElementBlock::new(
            ElementKind::Tria3,
            tri_ids,
            tri_pids,
            tri_conn,
        )],
        solid_properties,
        shell_properties: vec![
            Property {
                pid: TOP_PID,
                mid: None,
                card: "PSHELL",
            },
            Property {
                pid: BOTTOM_PID,
                mid: None,
                card: "PSHELL",
            },
        ],
        length_unit: LengthUnit::Meter,
        source: None,
    }
}

/// 単一材料の直方体。
pub fn uniform_slab(lx: f64, ly: f64, lz: f64, nx: usize, ny: usize, nz: usize) -> Mesh {
    slab(
        lx,
        ly,
        nx,
        ny,
        &[Layer {
            pid: 1,
            thickness: lz,
            divisions: nz,
        }],
    )
}

fn signed_volume(coords: &[f64], q: &[u32; 4]) -> f64 {
    let p: Vec<[f64; 3]> = q
        .iter()
        .map(|&n| {
            let k = n as usize * 3;
            [coords[k], coords[k + 1], coords[k + 2]]
        })
        .collect();
    let a = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
    let b = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
    let c = [p[3][0] - p[0][0], p[3][1] - p[0][1], p[3][2] - p[0][2]];
    let cr = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    (cr[0] * c[0] + cr[1] * c[1] + cr[2] * c[2]) / 6.0
}
