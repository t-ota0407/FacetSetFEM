//! 領域ごとの統計量（SPEC §5.2）。
//!
//! PID ごとに |J|・|E| の分布と消費電力を集計し、ホットスポット（最大値をとった要素）
//! の ID と重心を添える。研究では「どの組織にどれだけの電流密度が生じたか」を
//! 数字で押さえる必要があるため、これが主要な出力になる。
//!
//! # パーセンタイルは体積重み
//!
//! 要素サイズが不均一なので、**単純な要素数ベースの分位点は意味を持たない**
//! （細かく切った領域の値が過大に効いてしまう）。そこで値の昇順に並べたときの
//! **累積体積**が全体の p % に達する点の値をパーセンタイルとする。
//! 「体積の p % はこの値以下」という素直な定義になる。
//!
//! 要素値は区分定数なので補間はせず、階段的に定める。同じ値が並んだときは
//! 要素の通し番号順に並べる（安定ソート）ので、同じ入力からは常に同じ値になる。

use crate::case::Case;
use crate::error::{GfemError, Result};
use crate::mesh::{ElemId, Mesh, PropId};
use crate::post::FieldResult;
use crate::solver::reduce;
use std::collections::BTreeMap;

/// 1 つの量（|J| や |E|）の分布。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct QuantityStats {
    /// 最大値。**要素値**なので平滑化されていない（SPEC §5.1）
    pub max: f64,
    /// 体積重み平均
    pub mean: f64,
    /// 体積重み中央値
    pub median: f64,
    /// 体積重み 95 パーセンタイル
    pub p95: f64,
    /// 体積重み 99 パーセンタイル
    pub p99: f64,
}

/// 最大値をとった要素（ホットスポット）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hotspot {
    /// Nastran の要素 ID
    pub element_id: ElemId,
    /// 要素の重心 [m]
    pub centroid: [f64; 3],
    /// そこでの |J| [A/m²]
    pub j_magnitude: f64,
    /// そこでの |E| [V/m]
    pub e_magnitude: f64,
}

/// 1 領域（PID）の統計。
#[derive(Debug, Clone)]
pub struct RegionStats {
    /// プロパティ ID
    pub pid: PropId,
    /// 領域名（ケースで付けた名前。無ければ空）
    pub name: String,
    /// 導電率 [S/m]
    pub sigma: f64,
    /// 要素数
    pub n_elements: usize,
    /// 体積 [m³]
    pub volume: f64,
    /// 消費電力 ∫σ|E|² dΩ [W]
    pub power: f64,
    /// 電流密度 |J| [A/m²]
    pub current_density: QuantityStats,
    /// 電界 |E| [V/m]
    pub electric_field: QuantityStats,
    /// |J| が最大だった要素
    pub hotspot: Hotspot,
}

/// 領域ごとの統計を求める。並びは PID の昇順。
///
/// 要素種は問わない（Tet4 / Hex8 / Penta6）。要素値の並びは
/// [`Mesh::iter_solids`] の通し番号順であることを前提にする。
pub fn region_stats(mesh: &Mesh, case: &Case, fields: &FieldResult) -> Result<Vec<RegionStats>> {
    check_lengths(mesh, fields)?;

    // PID ごとに要素の通し番号を集める（昇順のまま）
    let mut by_pid: BTreeMap<PropId, Vec<usize>> = BTreeMap::new();
    for e in mesh.iter_solids() {
        by_pid.entry(e.pid).or_default().push(e.index);
    }

    // 領域ごとに独立なので並列に集計する（並びは PID の昇順のまま集まる）
    use rayon::prelude::*;
    let regions: Vec<(PropId, Vec<usize>)> = by_pid.into_iter().collect();
    Ok(regions
        .into_par_iter()
        .map(|(pid, elements)| RegionStats {
            pid,
            name: case.region_names.get(&pid).cloned().unwrap_or_default(),
            sigma: case
                .conductivity
                .get(&pid)
                .map(|s| s.scalar())
                .unwrap_or(f64::NAN),
            ..summarise_elements(mesh, fields, &elements)
        })
        .collect())
}

/// 全領域をひとまとめにした統計。
///
/// **パーセンタイルは全要素の値から求め直す。** 領域ごとの統計の要約からは
/// 全体の中央値・p95・p99 は決まらない（以前は先頭の領域の値を写していた）。
/// `pid` は 0、`sigma` は NaN（領域をまたぐので 1 つに決まらない）。
/// 名前は [`ALL_REGIONS_NAME`]。**表示言語に依らない**——領域統計の CSV と実行サマリの
/// JSON に出るので、後から読むスクリプトが言語で迷わないようにする（GUI が表示で訳す）。
/// 全領域の行の名前（言語に依らない。SPEC §6.6）。
pub const ALL_REGIONS_NAME: &str = "(all regions)";

pub fn total_stats(mesh: &Mesh, fields: &FieldResult) -> Result<RegionStats> {
    check_lengths(mesh, fields)?;
    if fields.n_elements == 0 {
        return Err(GfemError::model(crate::tr!(
            "There are no solid elements",
            "ソリッド要素が 1 つもありません"
        )));
    }
    let all: Vec<usize> = (0..fields.n_elements).collect();
    Ok(RegionStats {
        pid: 0,
        name: ALL_REGIONS_NAME.to_string(),
        sigma: f64::NAN,
        ..summarise_elements(mesh, fields, &all)
    })
}

fn check_lengths(mesh: &Mesh, fields: &FieldResult) -> Result<()> {
    if fields.n_elements != mesh.n_solid_elements() {
        return Err(GfemError::internal(crate::tr!(
            "The number of element values does not match the mesh: {} given, but there are {} solid elements",
            "要素値の個数がメッシュと合いません: {} 個ですがソリッド要素は {} 個です",
            fields.n_elements,
            mesh.n_solid_elements()
        )));
    }
    Ok(())
}

/// 要素の集合（通し番号の昇順、1 つ以上）の統計。`pid`・`name`・`sigma` は呼び出し側で埋める。
fn summarise_elements(mesh: &Mesh, fields: &FieldResult, elements: &[usize]) -> RegionStats {
    let volumes: Vec<f64> = elements.iter().map(|&e| fields.volume[e]).collect();
    let j: Vec<f64> = elements.iter().map(|&e| fields.j_magnitude[e]).collect();
    let e_mag: Vec<f64> = elements.iter().map(|&e| fields.e_magnitude[e]).collect();
    let power_terms: Vec<f64> = elements
        .iter()
        .map(|&e| fields.power_density[e] * fields.volume[e])
        .collect();
    let volume = reduce::sum(&volumes);

    // ホットスポットは |J| が最大の要素。同値なら通し番号の小さい方を採る
    let hottest = elements
        .iter()
        .copied()
        .reduce(|a, b| {
            if fields.j_magnitude[b] > fields.j_magnitude[a] {
                b
            } else {
                a
            }
        })
        .expect("要素が 1 つ以上ある");
    // 重心が要るのはホットスポットの 1 要素だけ
    let hot = mesh.solid(hottest);

    RegionStats {
        pid: 0,
        name: String::new(),
        sigma: f64::NAN,
        n_elements: elements.len(),
        volume,
        power: reduce::sum(&power_terms),
        current_density: summarise(&j, &volumes, volume),
        electric_field: summarise(&e_mag, &volumes, volume),
        hotspot: Hotspot {
            element_id: hot.id,
            centroid: mesh.element_centroid(hot.conn),
            j_magnitude: fields.j_magnitude[hottest],
            e_magnitude: fields.e_magnitude[hottest],
        },
    }
}

fn summarise(values: &[f64], volumes: &[f64], total_volume: f64) -> QuantityStats {
    let weighted: Vec<f64> = values.iter().zip(volumes).map(|(v, w)| v * w).collect();
    let quantiles = weighted_quantiles(values, volumes, &[0.5, 0.95, 0.99]);
    QuantityStats {
        max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        mean: if total_volume > 0.0 {
            reduce::sum(&weighted) / total_volume
        } else {
            f64::NAN
        },
        median: quantiles[0],
        p95: quantiles[1],
        p99: quantiles[2],
    }
}

/// 体積重みパーセンタイル。`quantiles` は昇順で渡すこと。
pub fn weighted_quantiles(values: &[f64], volumes: &[f64], quantiles: &[f64]) -> Vec<f64> {
    debug_assert_eq!(values.len(), volumes.len());
    if values.is_empty() {
        return vec![f64::NAN; quantiles.len()];
    }
    // 安定ソートなので、同じ値のときは元の要素順（＝通し番号順）が保たれる。
    // rayon の par_sort_by も安定ソートなので、並列にしても並び方は逐次と同じ
    use rayon::prelude::*;
    let mut order: Vec<u32> = (0..values.len() as u32).collect();
    order.par_sort_by(|&a, &b| values[a as usize].total_cmp(&values[b as usize]));

    let total = reduce::sum(volumes);
    let mut out = Vec::with_capacity(quantiles.len());
    let mut cumulative = 0.0;
    let mut k = 0usize;
    for &q in quantiles {
        let target = q * total;
        while k + 1 < order.len() && cumulative + volumes[order[k] as usize] < target {
            cumulative += volumes[order[k] as usize];
            k += 1;
        }
        out.push(values[order[k] as usize]);
    }
    out
}

/// 領域統計を CSV に書き出す（SPEC §5.5）。
///
/// 列名は ASCII で単位付き。pandas などにそのまま読ませられるようにしてある。
pub fn write_region_stats_csv(
    path: impl AsRef<std::path::Path>,
    stats: &[RegionStats],
) -> Result<()> {
    use std::io::Write;
    let path = path.as_ref();
    let file = std::fs::File::create(path).map_err(|e| {
        GfemError::io(
            path,
            crate::tr!(
                "Cannot create the region statistics CSV",
                "領域統計 CSV を作成できません"
            ),
            e,
        )
        .with_hint(crate::tr!(
            "Check that the output directory exists and that you have write permission",
            "出力先のディレクトリが存在し、書き込み権限があるかを確認してください"
        ))
    })?;
    let mut w = std::io::BufWriter::new(file);
    let io_err = |e: std::io::Error| {
        GfemError::io(
            path,
            crate::tr!(
                "Failed to write the region statistics CSV",
                "領域統計 CSV の書き込みに失敗しました"
            ),
            e,
        )
    };

    writeln!(
        w,
        "pid,name,sigma_S_per_m,n_elements,volume_m3,power_W,\
         J_max_A_per_m2,J_mean_A_per_m2,J_median_A_per_m2,J_p95_A_per_m2,J_p99_A_per_m2,\
         E_max_V_per_m,E_mean_V_per_m,E_median_V_per_m,E_p95_V_per_m,E_p99_V_per_m,\
         hotspot_element_id,hotspot_x_m,hotspot_y_m,hotspot_z_m"
    )
    .map_err(io_err)?;

    for s in stats {
        writeln!(
            w,
            "{},{},{:.9e},{},{:.9e},{:.9e},\
             {:.9e},{:.9e},{:.9e},{:.9e},{:.9e},\
             {:.9e},{:.9e},{:.9e},{:.9e},{:.9e},\
             {},{:.9e},{:.9e},{:.9e}",
            s.pid,
            escape_csv(&s.name),
            s.sigma,
            s.n_elements,
            s.volume,
            s.power,
            s.current_density.max,
            s.current_density.mean,
            s.current_density.median,
            s.current_density.p95,
            s.current_density.p99,
            s.electric_field.max,
            s.electric_field.mean,
            s.electric_field.median,
            s.electric_field.p95,
            s.electric_field.p99,
            s.hotspot.element_id,
            s.hotspot.centroid[0],
            s.hotspot.centroid[1],
            s.hotspot.centroid[2],
        )
        .map_err(io_err)?;
    }
    w.flush().map_err(io_err)
}

/// 収束履歴を CSV に書き出す（SPEC §5.5）。
pub fn write_convergence_csv(path: impl AsRef<std::path::Path>, history: &[f64]) -> Result<()> {
    use std::io::Write;
    let path = path.as_ref();
    let file = std::fs::File::create(path).map_err(|e| {
        GfemError::io(
            path,
            crate::tr!(
                "Cannot create the convergence history CSV",
                "収束履歴 CSV を作成できません"
            ),
            e,
        )
    })?;
    let mut w = std::io::BufWriter::new(file);
    let io_err = |e: std::io::Error| {
        GfemError::io(
            path,
            crate::tr!(
                "Failed to write the convergence history CSV",
                "収束履歴 CSV の書き込みに失敗しました"
            ),
            e,
        )
    };
    writeln!(w, "iteration,relative_residual").map_err(io_err)?;
    for (i, r) in history.iter().enumerate() {
        writeln!(w, "{i},{r:.9e}").map_err(io_err)?;
    }
    w.flush().map_err(io_err)
}

/// CSV のフィールドとして安全な形にする。
fn escape_csv(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_are_volume_weighted() {
        // 値 1 の要素が体積 1、値 10 の要素が体積 9
        // → 体積の 90 % は値 10 なので、中央値も 95 % も 10 になる
        let values = [1.0, 10.0];
        let volumes = [1.0, 9.0];
        let q = weighted_quantiles(&values, &volumes, &[0.05, 0.5, 0.95]);
        assert_eq!(q, vec![1.0, 10.0, 10.0]);
    }

    #[test]
    fn element_count_based_quantiles_would_differ() {
        // 単純な要素数ベースなら中央値は 1 と 10 の間だが、
        // 体積重みでは大きい要素の値が中央値になる
        let values = [1.0, 1.0, 1.0, 100.0];
        let volumes = [0.01, 0.01, 0.01, 10.0];
        let q = weighted_quantiles(&values, &volumes, &[0.5]);
        assert_eq!(q, vec![100.0]);
    }

    #[test]
    fn quantiles_of_a_uniform_field() {
        let values = vec![3.0; 10];
        let volumes = vec![0.5; 10];
        let q = weighted_quantiles(&values, &volumes, &[0.0, 0.5, 1.0]);
        assert_eq!(q, vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn quantiles_are_monotonic() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let volumes = vec![1.0; 100];
        let q = weighted_quantiles(&values, &volumes, &[0.1, 0.5, 0.9, 0.99]);
        assert!(q.windows(2).all(|w| w[0] <= w[1]), "{q:?}");
        // 一様な体積なら要素数ベースと一致する
        assert_eq!(q[1], 49.0);
    }

    #[test]
    fn empty_input_gives_nan() {
        let q = weighted_quantiles(&[], &[], &[0.5]);
        assert!(q[0].is_nan());
    }

    #[test]
    fn csv_escaping() {
        assert_eq!(escape_csv("skin"), "skin");
        assert_eq!(escape_csv("a,b"), "\"a,b\"");
        assert_eq!(escape_csv("say \"hi\""), "\"say \"\"hi\"\"\"");
    }
}
