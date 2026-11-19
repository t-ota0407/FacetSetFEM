//! 座標系カード CORD2R / CORD2C / CORD2S（SPEC §3.1）。
//!
//! GRID の CP フィールドが 0 以外のとき、その座標系で書かれた値を
//! デカルト（基本座標系）に直す。カードは
//!
//! ```text
//! CORD2R CID RID A1 A2 A3 B1 B2 B3
//! +      C1 C2 C3
//! ```
//!
//! の形で、**A・B・C は座標系 RID の値**として書かれている。したがって
//! RID を先に解決してからでないと自分を組み立てられない。入れ子は何段でもよいが、
//! 循環していたら CID を挙げてエラーにする。
//!
//! - A: 新しい座標系の原点
//! - B: +z 軸上の点
//! - C: xz 平面の +x 側にある点
//!
//! から `ez = normalize(B−A)`、`ey = normalize(ez × (C−A))`、`ex = ey × ez` を作る。
//!
//! # 単位について
//!
//! 変換は**ファイルに書かれた単位のまま**行い、SI への換算は全部終わってから
//! まとめて掛ける（角度は常に度なので単位に依存しない）。こうしないと
//! 「A・B・C は換算済み、GRID は未換算」のような食い違いが起きる。

use crate::error::{FacetSetFemError, Result};
use crate::geom::{cross, norm, sub};
use std::collections::BTreeMap;

/// 座標系の種類。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordKind {
    /// CORD2R: 直交
    Rectangular,
    /// CORD2C: 円筒 `(r, θ[deg], z)`
    Cylindrical,
    /// CORD2S: 球 `(r, θ[deg], φ[deg])`
    Spherical,
}

impl CoordKind {
    /// 対応する Nastran のカード名。
    pub const fn card_name(self) -> &'static str {
        match self {
            CoordKind::Rectangular => "CORD2R",
            CoordKind::Cylindrical => "CORD2C",
            CoordKind::Spherical => "CORD2S",
        }
    }
}

/// カードから読んだままの定義（まだ解決していない）。
#[derive(Debug, Clone, Copy)]
pub struct CoordDefinition {
    /// この座標系の ID
    pub cid: u32,
    /// A・B・C が書かれている座標系の ID（0 は基本座標系）
    pub rid: u32,
    /// 種類
    pub kind: CoordKind,
    /// 原点
    pub a: [f64; 3],
    /// +z 軸上の点
    pub b: [f64; 3],
    /// xz 平面の +x 側の点
    pub c: [f64; 3],
    /// カードの行番号（エラー報告用）
    pub line: usize,
}

/// 解決済みの座標系。
#[derive(Debug, Clone, Copy)]
pub struct CoordSystem {
    /// 基本座標系での原点
    pub origin: [f64; 3],
    /// 基本座標系での基底 `[ex, ey, ez]`
    pub axes: [[f64; 3]; 3],
    /// 種類
    pub kind: CoordKind,
}

impl CoordSystem {
    /// 基本座標系そのもの。
    pub const BASIC: CoordSystem = CoordSystem {
        origin: [0.0, 0.0, 0.0],
        axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        kind: CoordKind::Rectangular,
    };

    /// この座標系で書かれた値を基本座標系に直す。
    pub fn to_basic(&self, local: [f64; 3]) -> [f64; 3] {
        let cartesian = match self.kind {
            CoordKind::Rectangular => local,
            CoordKind::Cylindrical => {
                let (r, theta) = (local[0], local[1].to_radians());
                [r * theta.cos(), r * theta.sin(), local[2]]
            }
            CoordKind::Spherical => {
                let (r, theta, phi) = (local[0], local[1].to_radians(), local[2].to_radians());
                [
                    r * theta.sin() * phi.cos(),
                    r * theta.sin() * phi.sin(),
                    r * theta.cos(),
                ]
            }
        };
        let mut out = self.origin;
        for (axis, value) in self.axes.iter().zip(cartesian) {
            for d in 0..3 {
                out[d] += axis[d] * value;
            }
        }
        out
    }
}

/// 解決済みの座標系の集まり。
#[derive(Debug, Clone, Default)]
pub struct CoordSystems {
    systems: BTreeMap<u32, CoordSystem>,
}

impl CoordSystems {
    /// CID から座標系を引く。0 は基本座標系。
    pub fn get(&self, cid: u32) -> Option<CoordSystem> {
        if cid == 0 {
            return Some(CoordSystem::BASIC);
        }
        self.systems.get(&cid).copied()
    }

    /// 定義されている座標系の数（基本座標系は数えない）。
    pub fn len(&self) -> usize {
        self.systems.len()
    }

    /// 何も定義されていないか。
    pub fn is_empty(&self) -> bool {
        self.systems.is_empty()
    }
}

/// 定義を解決して座標系を組み立てる。
///
/// RID の入れ子をたどるので、定義の順序には依存しない。
pub fn resolve(definitions: &BTreeMap<u32, CoordDefinition>) -> Result<CoordSystems> {
    let mut resolved: BTreeMap<u32, CoordSystem> = BTreeMap::new();
    for &cid in definitions.keys() {
        let mut stack: Vec<u32> = Vec::new();
        resolve_one(cid, definitions, &mut resolved, &mut stack)?;
    }
    Ok(CoordSystems { systems: resolved })
}

fn resolve_one(
    cid: u32,
    definitions: &BTreeMap<u32, CoordDefinition>,
    resolved: &mut BTreeMap<u32, CoordSystem>,
    stack: &mut Vec<u32>,
) -> Result<()> {
    if cid == 0 || resolved.contains_key(&cid) {
        return Ok(());
    }
    if stack.contains(&cid) {
        stack.push(cid);
        let chain: Vec<String> = stack.iter().map(|c| c.to_string()).collect();
        return Err(FacetSetFemError::model(crate::tr!(
            "Coordinate system references form a cycle: CID {}",
            "座標系の参照が循環しています: CID {}",
            chain.join(" → ")
        ))
        .with_hint(crate::tr!(
            "Fix the RIDs of CORD2* so that following them never leads back to the same system",
            "CORD2* の RID がたどって自分自身に戻らないように直してください"
        )));
    }
    let Some(definition) = definitions.get(&cid) else {
        return Err(FacetSetFemError::model(crate::tr!(
            "The referenced coordinate system CID={cid} is not defined",
            "参照されている座標系 CID={cid} が定義されていません"
        ))
        .with_hint(crate::tr!(
            "Define that CID with CORD2R / CORD2C / CORD2S",
            "CORD2R / CORD2C / CORD2S でその CID を定義してください"
        )));
    };

    stack.push(cid);
    resolve_one(definition.rid, definitions, resolved, stack)?;
    stack.pop();

    let parent = if definition.rid == 0 {
        CoordSystem::BASIC
    } else {
        *resolved
            .get(&definition.rid)
            .expect("親は先に解決されている")
    };

    let a = parent.to_basic(definition.a);
    let b = parent.to_basic(definition.b);
    let c = parent.to_basic(definition.c);
    resolved.insert(cid, build(definition, a, b, c)?);
    Ok(())
}

/// 3 点から座標系を組み立てる。
fn build(
    definition: &CoordDefinition,
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
) -> Result<CoordSystem> {
    let fail = |what: String, hint: String| {
        FacetSetFemError::model(crate::tr!(
            "The definition of {} is degenerate: {what}",
            "{} の定義が退化しています: {what}",
            definition.kind.card_name()
        ))
        .with_line(definition.line)
        .with_card(definition.kind.card_name())
        .with_entity(format!("CID={}", definition.cid))
        .with_hint(hint)
    };

    let ez = normalize(sub(b, a)).ok_or_else(|| {
        fail(
            crate::tr!(
                "points A and B are at the same position",
                "点 A と点 B が同じ位置にあります"
            ),
            crate::tr!(
                "B is a point on the +z axis seen from the origin A. Give it a position different from A",
                "B は原点 A から見た +z 軸上の点です。A と別の位置を指定してください"
            ),
        )
    })?;
    let ac = sub(c, a);
    let ey = normalize(cross(ez, ac)).ok_or_else(|| {
        fail(
            crate::tr!("point C lies on the z axis", "点 C が z 軸上にあります"),
            crate::tr!(
                "C fixes the +x side of the xz plane, so it must lie off the line through A and B",
                "C は xz 平面の +x 側を決める点なので、A と B を結ぶ直線から外れている必要があります"
            ),
        )
    })?;
    let ex = cross(ey, ez);
    Ok(CoordSystem {
        origin: a,
        axes: [ex, ey, ez],
        kind: definition.kind,
    })
}

/// 単位ベクトルにする。長さが 0（または有限でない）なら `None`。
///
/// 差・外積・長さは [`crate::geom`] のものを使う（以前はここに同じ式の写しがあった。
/// 式の形は同じなので、座標変換の結果はビット単位で変わらない）。
fn normalize(v: [f64; 3]) -> Option<[f64; 3]> {
    let length = norm(v);
    if length <= 0.0 || !length.is_finite() {
        return None;
    }
    Some([v[0] / length, v[1] / length, v[2] / length])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn definition(
        cid: u32,
        rid: u32,
        kind: CoordKind,
        a: [f64; 3],
        b: [f64; 3],
        c: [f64; 3],
    ) -> CoordDefinition {
        CoordDefinition {
            cid,
            rid,
            kind,
            a,
            b,
            c,
            line: 1,
        }
    }

    fn resolve_one_definition(d: CoordDefinition) -> CoordSystem {
        let mut map = BTreeMap::new();
        map.insert(d.cid, d);
        resolve(&map).unwrap().get(d.cid).unwrap()
    }

    fn close(a: [f64; 3], b: [f64; 3]) -> bool {
        (0..3).all(|d| (a[d] - b[d]).abs() < 1e-12)
    }

    #[test]
    fn basic_system_is_the_identity() {
        let basic = CoordSystem::BASIC;
        assert!(close(basic.to_basic([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]));
    }

    #[test]
    fn rectangular_system_translates_and_rotates() {
        // 原点 (1,2,3)、z 軸は基本の +y、x 軸は基本の +z
        let system = resolve_one_definition(definition(
            10,
            0,
            CoordKind::Rectangular,
            [1.0, 2.0, 3.0],
            [1.0, 3.0, 3.0], // A から +y へ → ez = +y
            [1.0, 2.0, 4.0], // A から +z へ → ex = +z
        ));
        assert!(close(system.axes[2], [0.0, 1.0, 0.0]), "{:?}", system.axes);
        assert!(close(system.axes[0], [0.0, 0.0, 1.0]), "{:?}", system.axes);
        // 局所 (0,0,0) は原点
        assert!(close(system.to_basic([0.0, 0.0, 0.0]), [1.0, 2.0, 3.0]));
        // 局所 x は基本の +z
        assert!(close(system.to_basic([2.0, 0.0, 0.0]), [1.0, 2.0, 5.0]));
        // 局所 z は基本の +y
        assert!(close(system.to_basic([0.0, 0.0, 5.0]), [1.0, 7.0, 3.0]));
    }

    #[test]
    fn cylindrical_system_converts_angles_in_degrees() {
        // 基本座標系に揃えた円筒座標
        let system = resolve_one_definition(definition(
            20,
            0,
            CoordKind::Cylindrical,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ));
        assert!(close(system.to_basic([2.0, 0.0, 0.0]), [2.0, 0.0, 0.0]));
        assert!(close(system.to_basic([2.0, 90.0, 0.0]), [0.0, 2.0, 0.0]));
        assert!(close(system.to_basic([2.0, 180.0, 3.0]), [-2.0, 0.0, 3.0]));
    }

    #[test]
    fn spherical_system_converts_angles_in_degrees() {
        let system = resolve_one_definition(definition(
            30,
            0,
            CoordKind::Spherical,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ));
        // θ は +z からの角、φ は +x からの方位角
        assert!(close(system.to_basic([1.0, 0.0, 0.0]), [0.0, 0.0, 1.0]));
        assert!(close(system.to_basic([1.0, 90.0, 0.0]), [1.0, 0.0, 0.0]));
        assert!(close(system.to_basic([1.0, 90.0, 90.0]), [0.0, 1.0, 0.0]));
        assert!(close(system.to_basic([2.0, 180.0, 0.0]), [0.0, 0.0, -2.0]));
    }

    #[test]
    fn nested_systems_are_resolved_in_any_order() {
        // CID 2 は CID 1 の中で定義されている
        let mut map = BTreeMap::new();
        map.insert(
            2,
            definition(
                2,
                1,
                CoordKind::Rectangular,
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 0.0],
            ),
        );
        map.insert(
            1,
            definition(
                1,
                0,
                CoordKind::Rectangular,
                [10.0, 0.0, 0.0],
                [10.0, 0.0, 1.0],
                [11.0, 0.0, 0.0],
            ),
        );
        let systems = resolve(&map).unwrap();
        // CID 1 の原点は (10,0,0)、CID 2 はそこからさらに (1,0,0)
        assert!(close(systems.get(2).unwrap().origin, [11.0, 0.0, 0.0]));
        assert!(close(
            systems.get(2).unwrap().to_basic([0.0, 0.0, 0.0]),
            [11.0, 0.0, 0.0]
        ));
    }

    #[test]
    fn cylindrical_parent_is_taken_into_account() {
        // 親が円筒座標系なら、子の A・B・C も (r, θ, z) として読む
        let mut map = BTreeMap::new();
        map.insert(
            1,
            definition(
                1,
                0,
                CoordKind::Cylindrical,
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ),
        );
        map.insert(
            2,
            definition(
                2,
                1,
                CoordKind::Rectangular,
                [2.0, 90.0, 0.0], // 親の円筒座標で (r=2, θ=90°) → 基本の (0,2,0)
                [2.0, 90.0, 1.0],
                [3.0, 90.0, 0.0],
            ),
        );
        let systems = resolve(&map).unwrap();
        assert!(close(systems.get(2).unwrap().origin, [0.0, 2.0, 0.0]));
    }

    #[test]
    fn cycles_are_reported() {
        let mut map = BTreeMap::new();
        for (cid, rid) in [(1u32, 2u32), (2, 1)] {
            map.insert(
                cid,
                definition(
                    cid,
                    rid,
                    CoordKind::Rectangular,
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                ),
            );
        }
        let e = resolve(&map).unwrap_err();
        assert!(e.to_string().contains("循環"), "{e}");
    }

    #[test]
    fn missing_reference_is_reported() {
        let mut map = BTreeMap::new();
        map.insert(
            1,
            definition(
                1,
                99,
                CoordKind::Rectangular,
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ),
        );
        let e = resolve(&map).unwrap_err();
        assert!(e.to_string().contains("CID=99"), "{e}");
    }

    #[test]
    fn degenerate_definitions_are_reported() {
        // A と B が同じ
        let mut map = BTreeMap::new();
        map.insert(
            1,
            definition(
                1,
                0,
                CoordKind::Rectangular,
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ),
        );
        let e = resolve(&map).unwrap_err();
        assert!(e.to_string().contains("A と点 B"), "{e}");

        // C が z 軸上
        let mut map = BTreeMap::new();
        map.insert(
            1,
            definition(
                1,
                0,
                CoordKind::Rectangular,
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
            ),
        );
        let e = resolve(&map).unwrap_err();
        assert!(e.to_string().contains("z 軸上"), "{e}");
    }

    #[test]
    fn axes_are_orthonormal() {
        let system = resolve_one_definition(definition(
            1,
            0,
            CoordKind::Rectangular,
            [0.3, -1.2, 4.0],
            [1.7, 2.2, 0.5],
            [-3.0, 0.1, 2.2],
        ));
        for axis in &system.axes {
            let length = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
            assert!((length - 1.0).abs() < 1e-14, "長さ {length}");
        }
        for (i, j) in [(0, 1), (1, 2), (0, 2)] {
            let dot: f64 = (0..3).map(|d| system.axes[i][d] * system.axes[j][d]).sum();
            assert!(dot.abs() < 1e-14, "軸 {i}·{j} = {dot}");
        }
        // 右手系
        let ex_cross_ey = cross(system.axes[0], system.axes[1]);
        assert!(close(ex_cross_ey, system.axes[2]));
    }
}
