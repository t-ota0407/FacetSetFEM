//! テスト用のメッシュ生成。
//!
//! - [`box_mesh`] / [`slab`] — 直方体（層構造つき、格子は任意）
//! - [`spherical_shell`] — 同心球殻・半球殻
//!
//! 直方体は**Kuhn の 6 分割**、球殻は**三角柱の 3 分割**で四面体化する。
//! どちらも「立方体／三角柱ごとに同じ規則で対角線を選ぶ」ことで、
//! 隣り合う要素の共有面が同じ対角線で割れ、適合（conforming）メッシュになる。

#![allow(dead_code)] // テストファイルごとに使う関数が違う

use facetsetfem_core::assemble::CsrMatrix;
use facetsetfem_core::element::ElementKind;
use facetsetfem_core::mesh::{ElementBlock, Mesh, Nodes, PropId, Property};
use facetsetfem_core::units::LengthUnit;

// --- 直方体 -----------------------------------------------------------------

/// 直方体の上面（陽極側）に貼るシェルの PID。
pub const TOP_PID: PropId = 101;
/// 直方体の下面（陰極側）に貼るシェルの PID。
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

/// 面に貼る電極パッチ。
#[derive(Debug, Clone, Copy)]
pub struct Patch {
    /// PSHELL の PID
    pub pid: PropId,
    /// x の範囲 [m]
    pub x: [f64; 2],
    /// y の範囲 [m]
    pub y: [f64; 2],
    /// true なら上面（z 最大）、false なら下面（z 最小）
    pub top: bool,
}

impl Patch {
    /// 面全体を覆うパッチ。
    pub fn whole(pid: PropId, top: bool) -> Self {
        Self {
            pid,
            x: [f64::NEG_INFINITY, f64::INFINITY],
            y: [f64::NEG_INFINITY, f64::INFINITY],
            top,
        }
    }

    fn covers(&self, x0: f64, x1: f64, y0: f64, y1: f64) -> bool {
        const EPS: f64 = 1.0e-9;
        x0 >= self.x[0] - EPS
            && x1 <= self.x[1] + EPS
            && y0 >= self.y[0] - EPS
            && y1 <= self.y[1] + EPS
    }
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

/// 任意の格子で直方体の四面体メッシュを作る。
///
/// `xs` / `ys` / `zs` は昇順の格子座標 [m]、`cell_pid` は z セルごとの PID
/// （長さは `zs.len() - 1`）、`patches` は上下面に貼る電極面。
pub fn box_mesh(
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    cell_pid: &[PropId],
    patches: &[Patch],
) -> Mesh {
    box_mesh_of(ElementKind::Tet4, xs, ys, zs, cell_pid, patches)
}

/// 要素種を選んで直方体メッシュを作る（SPEC §8 M4 の検証用）。
///
/// - `Tet4`: Kuhn の 6 分割。電極面は CTRIA3
/// - `Hex8`: 格子セルをそのまま 1 要素に。電極面は CQUAD4
/// - `Penta6`: セルを対角線で 2 つの三角柱に割る。電極面は CTRIA3
///
/// どの分け方でも「セルごとに同じ規則で対角線を選ぶ」ので適合メッシュになる。
pub fn box_mesh_of(
    kind: ElementKind,
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    cell_pid: &[PropId],
    patches: &[Patch],
) -> Mesh {
    match kind {
        ElementKind::Tet4 => box_mesh_tet4(xs, ys, zs, cell_pid, patches),
        ElementKind::Hex8 => box_mesh_hex8(xs, ys, zs, cell_pid, patches),
        ElementKind::Penta6 => box_mesh_penta6(xs, ys, zs, cell_pid, patches),
        other => panic!("ソリッド要素ではありません: {other:?}"),
    }
}

/// 直方体を六面体そのままで刻む。電極面は CQUAD4。
fn box_mesh_hex8(
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    cell_pid: &[PropId],
    patches: &[Patch],
) -> Mesh {
    let (nx, ny, nz) = (xs.len() - 1, ys.len() - 1, zs.len() - 1);
    assert_eq!(cell_pid.len(), nz);
    let nid = |i: usize, j: usize, k: usize| -> u32 { (i + (nx + 1) * (j + (ny + 1) * k)) as u32 };
    let coords = grid_coords(xs, ys, zs);

    let mut conn: Vec<u32> = Vec::new();
    let mut pids: Vec<PropId> = Vec::new();
    let mut ids: Vec<u32> = Vec::new();
    let mut next_eid = 1u32;
    for (k, &pid) in cell_pid.iter().enumerate() {
        for j in 0..ny {
            for i in 0..nx {
                // CHEXA G1..G4 が底面、G5..G8 が上面（どちらも反時計回り）
                conn.extend_from_slice(&[
                    nid(i, j, k),
                    nid(i + 1, j, k),
                    nid(i + 1, j + 1, k),
                    nid(i, j + 1, k),
                    nid(i, j, k + 1),
                    nid(i + 1, j, k + 1),
                    nid(i + 1, j + 1, k + 1),
                    nid(i, j + 1, k + 1),
                ]);
                pids.push(pid);
                ids.push(next_eid);
                next_eid += 1;
            }
        }
    }

    let mut shell_conn: Vec<u32> = Vec::new();
    let mut shell_pids: Vec<PropId> = Vec::new();
    let mut shell_ids: Vec<u32> = Vec::new();
    for patch in patches {
        let k = if patch.top { nz } else { 0 };
        for j in 0..ny {
            for i in 0..nx {
                if !patch.covers(xs[i], xs[i + 1], ys[j], ys[j + 1]) {
                    continue;
                }
                let quad = [
                    nid(i, j, k),
                    nid(i + 1, j, k),
                    nid(i + 1, j + 1, k),
                    nid(i, j + 1, k),
                ];
                shell_conn.extend_from_slice(&quad);
                shell_pids.push(patch.pid);
                shell_ids.push(next_eid);
                next_eid += 1;
            }
        }
    }

    assemble_mesh(
        coords,
        ElementKind::Hex8,
        ids,
        pids,
        conn,
        ElementKind::Quad4,
        shell_ids,
        shell_pids,
        shell_conn,
        cell_pid,
        patches,
    )
}

/// 直方体を三角柱で刻む。各セルを対角線 (i,j)-(i+1,j+1) で 2 つに割る。
fn box_mesh_penta6(
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    cell_pid: &[PropId],
    patches: &[Patch],
) -> Mesh {
    let (nx, ny, nz) = (xs.len() - 1, ys.len() - 1, zs.len() - 1);
    assert_eq!(cell_pid.len(), nz);
    let nid = |i: usize, j: usize, k: usize| -> u32 { (i + (nx + 1) * (j + (ny + 1) * k)) as u32 };
    let coords = grid_coords(xs, ys, zs);

    let mut conn: Vec<u32> = Vec::new();
    let mut pids: Vec<PropId> = Vec::new();
    let mut ids: Vec<u32> = Vec::new();
    let mut next_eid = 1u32;
    for (k, &pid) in cell_pid.iter().enumerate() {
        for j in 0..ny {
            for i in 0..nx {
                // 底面の四角形を対角線 (i,j)-(i+1,j+1) で 2 つに割る。
                // 全セルで同じ規則なので隣接セルと整合する
                let corners = [
                    [(i, j), (i + 1, j), (i + 1, j + 1)],
                    [(i, j), (i + 1, j + 1), (i, j + 1)],
                ];
                for tri in corners {
                    // CPENTA G1..G3 が底面、G4..G6 が上面
                    for level in [k, k + 1] {
                        for (a, b) in tri {
                            conn.push(nid(a, b, level));
                        }
                    }
                    pids.push(pid);
                    ids.push(next_eid);
                    next_eid += 1;
                }
            }
        }
    }

    let mut shell_conn: Vec<u32> = Vec::new();
    let mut shell_pids: Vec<PropId> = Vec::new();
    let mut shell_ids: Vec<u32> = Vec::new();
    for patch in patches {
        let k = if patch.top { nz } else { 0 };
        for j in 0..ny {
            for i in 0..nx {
                if !patch.covers(xs[i], xs[i + 1], ys[j], ys[j + 1]) {
                    continue;
                }
                let (a, b, c, d) = (
                    nid(i, j, k),
                    nid(i + 1, j, k),
                    nid(i + 1, j + 1, k),
                    nid(i, j + 1, k),
                );
                let faces = if patch.top {
                    [[a, b, c], [a, c, d]]
                } else {
                    [[a, c, b], [a, d, c]]
                };
                for f in faces {
                    shell_conn.extend_from_slice(&f);
                    shell_pids.push(patch.pid);
                    shell_ids.push(next_eid);
                    next_eid += 1;
                }
            }
        }
    }

    assemble_mesh(
        coords,
        ElementKind::Penta6,
        ids,
        pids,
        conn,
        ElementKind::Tria3,
        shell_ids,
        shell_pids,
        shell_conn,
        cell_pid,
        patches,
    )
}

/// 格子の節点座標を作る。
fn grid_coords(xs: &[f64], ys: &[f64], zs: &[f64]) -> Vec<f64> {
    let mut coords = Vec::with_capacity(xs.len() * ys.len() * zs.len() * 3);
    for &z in zs {
        for &y in ys {
            for &x in xs {
                coords.push(x);
                coords.push(y);
                coords.push(z);
            }
        }
    }
    coords
}

/// ブロックから `Mesh` を組み立てる。
#[allow(clippy::too_many_arguments)]
fn assemble_mesh(
    coords: Vec<f64>,
    solid_kind: ElementKind,
    solid_ids: Vec<u32>,
    solid_pids: Vec<PropId>,
    solid_conn: Vec<u32>,
    shell_kind: ElementKind,
    shell_ids: Vec<u32>,
    shell_pids: Vec<PropId>,
    shell_conn: Vec<u32>,
    cell_pid: &[PropId],
    patches: &[Patch],
) -> Mesh {
    let n_nodes = coords.len() / 3;
    let ids: Vec<u32> = (1..=n_nodes as u32).collect();

    let mut solid_properties: Vec<Property> = cell_pid
        .iter()
        .map(|&pid| Property {
            pid,
            mid: None,
            card: "PSOLID",
        })
        .collect();
    solid_properties.sort_by_key(|p| p.pid);
    solid_properties.dedup_by_key(|p| p.pid);

    let mut shell_properties: Vec<Property> = patches
        .iter()
        .map(|p| Property {
            pid: p.pid,
            mid: None,
            card: "PSHELL",
        })
        .collect();
    shell_properties.sort_by_key(|p| p.pid);
    shell_properties.dedup_by_key(|p| p.pid);

    Mesh {
        nodes: Nodes::new(ids, coords),
        solids: vec![ElementBlock::new(
            solid_kind, solid_ids, solid_pids, solid_conn,
        )],
        shells: vec![ElementBlock::new(
            shell_kind, shell_ids, shell_pids, shell_conn,
        )],
        solid_properties,
        shell_properties,
        length_unit: LengthUnit::Meter,
        source: None,
    }
}

fn box_mesh_tet4(
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    cell_pid: &[PropId],
    patches: &[Patch],
) -> Mesh {
    let (nx, ny, nz) = (xs.len() - 1, ys.len() - 1, zs.len() - 1);
    assert_eq!(cell_pid.len(), nz);
    let nid = |i: usize, j: usize, k: usize| -> u32 { (i + (nx + 1) * (j + (ny + 1) * k)) as u32 };

    let mut coords = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1) * 3);
    for &z in zs {
        for &y in ys {
            for &x in xs {
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
    for (k, &pid) in cell_pid.iter().enumerate() {
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
                    assert!(signed_volume(&coords, &q) > 0.0, "退化した四面体");
                    tet_conn.extend_from_slice(&q);
                    tet_pids.push(pid);
                    tet_ids.push(next_eid);
                    next_eid += 1;
                }
            }
        }
    }

    // --- 上下面の電極パッチ ---
    // Kuhn 分割では z = 一定の面が (i,j)-(i+1,j+1) の対角線で割れる。
    // シェル要素はこの三角形分割に一致していなければならない（SPEC §11.1）
    let mut tri_conn: Vec<u32> = Vec::new();
    let mut tri_pids: Vec<PropId> = Vec::new();
    let mut tri_ids: Vec<u32> = Vec::new();
    for patch in patches {
        let k = if patch.top { nz } else { 0 };
        for j in 0..ny {
            for i in 0..nx {
                if !patch.covers(xs[i], xs[i + 1], ys[j], ys[j + 1]) {
                    continue;
                }
                let (a, b, c, d) = (
                    nid(i, j, k),
                    nid(i + 1, j, k),
                    nid(i + 1, j + 1, k),
                    nid(i, j + 1, k),
                );
                let faces = if patch.top {
                    [[a, b, c], [a, c, d]]
                } else {
                    [[a, c, b], [a, d, c]]
                };
                for f in faces {
                    tri_conn.extend_from_slice(&f);
                    tri_pids.push(patch.pid);
                    tri_ids.push(next_eid);
                    next_eid += 1;
                }
            }
        }
    }

    let mut solid_properties: Vec<Property> = cell_pid
        .iter()
        .map(|&pid| Property {
            pid,
            mid: None,
            card: "PSOLID",
        })
        .collect();
    solid_properties.sort_by_key(|p| p.pid);
    solid_properties.dedup_by_key(|p| p.pid);

    let mut shell_properties: Vec<Property> = patches
        .iter()
        .map(|p| Property {
            pid: p.pid,
            mid: None,
            card: "PSHELL",
        })
        .collect();
    shell_properties.sort_by_key(|p| p.pid);
    shell_properties.dedup_by_key(|p| p.pid);

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
        shell_properties,
        length_unit: LengthUnit::Meter,
        source: None,
    }
}

/// 等分割の格子。
pub fn linear_grid(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..=n)
        .map(|i| lo + (hi - lo) * i as f64 / n as f64)
        .collect()
}

/// `0` から `fine_to` までを `fine_step` の等分割、そこから `outer` まで等比数列で伸ばす格子。
///
/// 電極付近だけ細かく、遠方は粗くしたいときに使う。
pub fn graded_grid(fine_to: f64, fine_step: f64, outer: f64, ratio: f64) -> Vec<f64> {
    assert!(fine_step > 0.0 && ratio > 1.0 && outer > fine_to && fine_to >= 0.0);
    let mut grid = vec![0.0];
    while *grid.last().unwrap() < fine_to - 1e-12 {
        let next = grid.last().unwrap() + fine_step;
        grid.push(next);
    }
    let mut step = fine_step;
    while *grid.last().unwrap() < outer - 1e-12 {
        step *= ratio;
        let next = (grid.last().unwrap() + step).min(outer);
        grid.push(next);
    }
    grid
}

/// [`graded_grid`] を原点対称に広げた格子。
pub fn symmetric_graded_grid(fine_to: f64, fine_step: f64, outer: f64, ratio: f64) -> Vec<f64> {
    let half = graded_grid(fine_to, fine_step, outer, ratio);
    let mut grid: Vec<f64> = half.iter().skip(1).rev().map(|v| -v).collect();
    grid.extend(half);
    grid
}

/// 直方体の多層四面体メッシュ。上面に PSHELL 101、下面に PSHELL 201 を貼る。
pub fn slab(lx: f64, ly: f64, nx: usize, ny: usize, layers: &[Layer]) -> Mesh {
    let mut zs = vec![0.0f64];
    let mut cell_pid: Vec<PropId> = Vec::new();
    for layer in layers {
        let z0 = *zs.last().unwrap();
        for k in 1..=layer.divisions {
            zs.push(z0 + layer.thickness * k as f64 / layer.divisions as f64);
            cell_pid.push(layer.pid);
        }
    }
    box_mesh(
        &linear_grid(0.0, lx, nx),
        &linear_grid(0.0, ly, ny),
        &zs,
        &cell_pid,
        &[Patch::whole(TOP_PID, true), Patch::whole(BOTTOM_PID, false)],
    )
}

/// 要素種を選べる多層スラブ（SPEC §8 M4 の検証用）。
pub fn slab_of(
    kind: ElementKind,
    lx: f64,
    ly: f64,
    nx: usize,
    ny: usize,
    layers: &[Layer],
) -> Mesh {
    let mut zs = vec![0.0f64];
    let mut cell_pid: Vec<PropId> = Vec::new();
    for layer in layers {
        let z0 = *zs.last().unwrap();
        for k in 1..=layer.divisions {
            zs.push(z0 + layer.thickness * k as f64 / layer.divisions as f64);
            cell_pid.push(layer.pid);
        }
    }
    box_mesh_of(
        kind,
        &linear_grid(0.0, lx, nx),
        &linear_grid(0.0, ly, ny),
        &zs,
        &cell_pid,
        &[Patch::whole(TOP_PID, true), Patch::whole(BOTTOM_PID, false)],
    )
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

// --- 同心球殻（SPEC §8 M2 の 3・4）-----------------------------------------

/// 球殻の内側表面に貼るシェルの PID。
pub const INNER_PID: PropId = 101;
/// 球殻の外側表面に貼るシェルの PID。
pub const OUTER_PID: PropId = 201;

/// 球殻の 1 層。
#[derive(Debug, Clone, Copy)]
pub struct ShellLayer {
    /// PSOLID の PID
    pub pid: PropId,
    /// この層の外半径 [m]
    pub outer_radius: f64,
    /// 半径方向の分割数
    pub divisions: usize,
}

/// 同心球殻（`hemisphere` が true なら上半球殻）の四面体メッシュ。
///
/// 表面は緯度経度分割（極は 1 節点）。それを半径方向に積んで三角柱を作り、
/// 3 つの四面体に割る。三角柱の分割は**節点番号の昇順に並べてから**行うので、
/// 隣り合う三角柱が共有する四角形面が同じ対角線で割れ、適合メッシュになる。
///
/// 半球殻の赤道面（z = 0 のドーナツ状の面）は電極を貼らないので絶縁境界になり、
/// そのまま「半無限媒質上の電極」の模型になる（SPEC §8 M2 の 4）。
pub fn spherical_shell(
    inner_radius: f64,
    layers: &[ShellLayer],
    n_theta: usize,
    n_phi: usize,
    hemisphere: bool,
) -> Mesh {
    assert!(inner_radius > 0.0 && n_theta >= 2 && n_phi >= 3);

    // --- 単位球面上の方向ベクトルと三角形分割 ---
    let mut dirs: Vec<[f64; 3]> = vec![[0.0, 0.0, 1.0]]; // 北極
    let n_rings = if hemisphere { n_theta } else { n_theta - 1 };
    let theta_max = if hemisphere {
        std::f64::consts::FRAC_PI_2
    } else {
        std::f64::consts::PI
    };
    let mut ring_start: Vec<u32> = Vec::with_capacity(n_rings);
    for i in 1..=n_rings {
        let theta = theta_max * i as f64 / n_theta as f64;
        ring_start.push(dirs.len() as u32);
        for j in 0..n_phi {
            let phi = std::f64::consts::TAU * j as f64 / n_phi as f64;
            dirs.push([
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            ]);
        }
    }
    let south = if hemisphere {
        None
    } else {
        let index = dirs.len() as u32;
        dirs.push([0.0, 0.0, -1.0]);
        Some(index)
    };

    let mut surface: Vec<[u32; 3]> = Vec::new();
    let next = |j: usize| (j + 1) % n_phi;
    for j in 0..n_phi {
        surface.push([0, ring_start[0] + j as u32, ring_start[0] + next(j) as u32]);
    }
    for i in 0..n_rings.saturating_sub(1) {
        for j in 0..n_phi {
            let (a0, b0) = (ring_start[i] + j as u32, ring_start[i] + next(j) as u32);
            let (a1, b1) = (
                ring_start[i + 1] + j as u32,
                ring_start[i + 1] + next(j) as u32,
            );
            surface.push([a0, a1, b1]);
            surface.push([a0, b1, b0]);
        }
    }
    if let Some(s) = south {
        let last = ring_start[n_rings - 1];
        for j in 0..n_phi {
            surface.push([s, last + next(j) as u32, last + j as u32]);
        }
    }

    // --- 半径方向の格子 ---
    let mut radii = vec![inner_radius];
    let mut cell_pid: Vec<PropId> = Vec::new();
    for layer in layers {
        let r0 = *radii.last().unwrap();
        assert!(layer.outer_radius > r0 && layer.divisions >= 1);
        for k in 1..=layer.divisions {
            radii.push(r0 + (layer.outer_radius - r0) * k as f64 / layer.divisions as f64);
            cell_pid.push(layer.pid);
        }
    }
    let n_cells = radii.len() - 1;
    let per_layer = dirs.len();

    let mut coords = Vec::with_capacity(per_layer * radii.len() * 3);
    for &r in &radii {
        for d in &dirs {
            coords.push(r * d[0]);
            coords.push(r * d[1]);
            coords.push(r * d[2]);
        }
    }
    let n_nodes = coords.len() / 3;
    let ids: Vec<u32> = (1..=n_nodes as u32).collect();
    let node = |k: usize, s: u32| (k * per_layer) as u32 + s;

    // --- 三角柱を 3 つの四面体に割る ---
    let mut tet_conn: Vec<u32> = Vec::new();
    let mut tet_pids: Vec<PropId> = Vec::new();
    let mut tet_ids: Vec<u32> = Vec::new();
    let mut next_eid = 1u32;
    for (k, &pid) in cell_pid.iter().enumerate() {
        for tri in &surface {
            // 節点番号の昇順に並べると、隣接する三角柱と対角線の選び方が揃う
            let mut t = *tri;
            t.sort_unstable();
            let v = [
                node(k, t[0]),
                node(k, t[1]),
                node(k, t[2]),
                node(k + 1, t[0]),
                node(k + 1, t[1]),
                node(k + 1, t[2]),
            ];
            for local in [[0, 1, 2, 5], [0, 1, 5, 4], [0, 4, 5, 3]] {
                let mut q = [v[local[0]], v[local[1]], v[local[2]], v[local[3]]];
                if signed_volume(&coords, &q) < 0.0 {
                    q.swap(2, 3);
                }
                assert!(signed_volume(&coords, &q) > 0.0, "退化した四面体");
                tet_conn.extend_from_slice(&q);
                tet_pids.push(pid);
                tet_ids.push(next_eid);
                next_eid += 1;
            }
        }
    }

    // --- 内側・外側の電極面 ---
    let mut tri_conn: Vec<u32> = Vec::new();
    let mut tri_pids: Vec<PropId> = Vec::new();
    let mut tri_ids: Vec<u32> = Vec::new();
    for (k, pid) in [(0usize, INNER_PID), (n_cells, OUTER_PID)] {
        for tri in &surface {
            tri_conn.extend_from_slice(&[node(k, tri[0]), node(k, tri[1]), node(k, tri[2])]);
            tri_pids.push(pid);
            tri_ids.push(next_eid);
            next_eid += 1;
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
                pid: INNER_PID,
                mid: None,
                card: "PSHELL",
            },
            Property {
                pid: OUTER_PID,
                mid: None,
                card: "PSHELL",
            },
        ],
        length_unit: LengthUnit::Meter,
        source: None,
    }
}

/// 単一材料の球殻。
pub fn uniform_shell(
    inner_radius: f64,
    outer_radius: f64,
    n_radial: usize,
    n_theta: usize,
    n_phi: usize,
    hemisphere: bool,
) -> Mesh {
    spherical_shell(
        inner_radius,
        &[ShellLayer {
            pid: 1,
            outer_radius,
            divisions: n_radial,
        }],
        n_theta,
        n_phi,
        hemisphere,
    )
}

// --- 共通の補助 -------------------------------------------------------------

/// 節点番号の組から符号付き体積を求める。
pub fn signed_volume(coords: &[f64], q: &[u32; 4]) -> f64 {
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

/// 指定した座標に最も近い節点の索引。
pub fn nearest_node(mesh: &Mesh, target: [f64; 3]) -> u32 {
    let mut best = 0u32;
    let mut best_distance = f64::INFINITY;
    for i in 0..mesh.nodes.len() as u32 {
        let p = mesh.nodes.coord(i);
        let d =
            (p[0] - target[0]).powi(2) + (p[1] - target[1]).powi(2) + (p[2] - target[2]).powi(2);
        if d < best_distance {
            best_distance = d;
            best = i;
        }
    }
    best
}

/// 要素の重心。
pub fn centroid(mesh: &Mesh, element: usize) -> [f64; 3] {
    let block = mesh.block(ElementKind::Tet4).unwrap();
    let p = mesh.tet4_coords(block.conn_of(element));
    let mut c = [0.0f64; 3];
    for q in &p {
        for (d, cd) in c.iter_mut().enumerate() {
            *cd += q[d] / 4.0;
        }
    }
    c
}

/// 1 次元の格子ラプラシアン（全 Neumann）。`tridiag(−1, 2, −1)` の両端の対角を 1 にしたもの。
///
/// 零空間がちょうど定数ベクトルになる。ユニットテスト用の `facetsetfem_core` 内の
/// `testing::laplacian_1d` と同じ行列（結合テストは別クレートなので、そちらは使えない）。
pub fn laplacian_1d(n: usize) -> CsrMatrix {
    let mut row_ptr = vec![0usize];
    let mut col_idx: Vec<u32> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    for i in 0..n {
        let mut diag = 0.0;
        if i > 0 {
            col_idx.push(i as u32 - 1);
            values.push(-1.0);
            diag += 1.0;
        }
        let slot = col_idx.len();
        col_idx.push(i as u32);
        values.push(0.0);
        if i + 1 < n {
            col_idx.push(i as u32 + 1);
            values.push(-1.0);
            diag += 1.0;
        }
        values[slot] = diag;
        row_ptr.push(col_idx.len());
    }
    let mut m = CsrMatrix::from_structure(n, row_ptr, col_idx);
    m.values_mut().copy_from_slice(&values);
    m
}
