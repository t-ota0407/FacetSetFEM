//! GPU と CPU がビット単位で一致することの検証（SPEC §8 M5 / §10）。
//!
//! CLAUDE.md 設計原則 1 は「同一入力から CPU 逐次・CPU 並列・GPU のいずれでも
//! ビット単位で同じ結果」を要求している。M5 でいちばん大事なテストはこれ。
//!
//! GPU が無い環境ではテストごとに `skip` して通す（`facetsetfem-core` は CPU だけで
//! 完全にテストできる状態を保つ。設計原則 5）。GPU があるのに落ちたら本物の不具合。
//!
//! ```text
//! cargo test -p facetsetfem-cuda --release
//! ```

#[path = "../../facetsetfem-core/tests/common/mod.rs"]
mod common;

use common::laplacian_1d;
use facetsetfem_core::assemble::CsrMatrix;
use facetsetfem_core::solver::reduce::BLOCK;
use facetsetfem_core::solver::{CpuParallel, CpuSerial, LinearBackend};
use facetsetfem_cuda::CudaBackend;

/// GPU が使えなければ `None`。使えなければテストは黙って通す。
fn gpu() -> Option<CudaBackend> {
    if facetsetfem_cuda::devices()
        .iter()
        .all(|d| !d.supports_f64())
    {
        eprintln!("倍精度が使える GPU が無いので飛ばします");
        return None;
    }
    match CudaBackend::new(0) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("GPU を初期化できないので飛ばします: {e}");
            None
        }
    }
}

macro_rules! gpu_or_skip {
    () => {
        match gpu() {
            Some(b) => b,
            None => return,
        }
    };
}

/// 丸めが効きやすい、桁の散らばった試験ベクトル。
fn spread(n: usize, seed: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 * seed;
            // 10^-8 〜 10^8 にまたがらせて、加算順序の違いが出やすいようにする
            t.sin() * 10f64.powf((t.cos() * 8.0).round())
        })
        .collect()
}

#[test]
fn every_vector_operation_matches_the_cpu_bitwise() {
    let gpu = gpu_or_skip!();
    // BLOCK の倍数ちょうど・端数あり・1 段の再帰では終わらない大きさを含める
    for n in [
        1usize,
        7,
        BLOCK - 1,
        BLOCK,
        BLOCK + 1,
        4096,
        BLOCK * BLOCK + 3,
    ] {
        let x = spread(n, 0.37);
        let y = spread(n, 1.13);
        let matrix = laplacian_1d(n);

        let cpu = CpuSerial;
        let (mc, mg) = (
            cpu.upload_matrix(&matrix).unwrap(),
            gpu.upload_matrix(&matrix).unwrap(),
        );
        let (xc, xg) = (cpu.upload(&x).unwrap(), gpu.upload(&x).unwrap());
        let (yc, yg) = (cpu.upload(&y).unwrap(), gpu.upload(&y).unwrap());

        // --- スカラーを返すもの ---
        assert_eq!(
            cpu.dot(&xc, &yc).unwrap(),
            gpu.dot(&xg, &yg).unwrap(),
            "内積 (n = {n})"
        );
        assert_eq!(
            cpu.sum(&xc).unwrap(),
            gpu.sum(&xg).unwrap(),
            "総和 (n = {n})"
        );
        assert_eq!(
            cpu.norm2(&xc).unwrap(),
            gpu.norm2(&xg).unwrap(),
            "ノルム (n = {n})"
        );

        // --- ベクトルを返すもの ---
        let check = |name: &str, c: &Vec<f64>, g: &facetsetfem_cuda::CudaVector| {
            let mut back = vec![0.0; n];
            gpu.download(g, &mut back).unwrap();
            assert_eq!(c, &back, "{name} (n = {n})");
        };

        let mut sc = cpu.zeros(n).unwrap();
        let mut sg = gpu.zeros(n).unwrap();
        cpu.spmv(&mc, &xc, &mut sc).unwrap();
        gpu.spmv(&mg, &xg, &mut sg).unwrap();
        check("SpMV", &sc, &sg);

        let (mut ac, mut ag) = (cpu.upload(&y).unwrap(), gpu.upload(&y).unwrap());
        cpu.axpy(-0.375, &xc, &mut ac).unwrap();
        gpu.axpy(-0.375, &xg, &mut ag).unwrap();
        check("axpy", &ac, &ag);

        cpu.xpby(&xc, 1.0 / 3.0, &mut ac).unwrap();
        gpu.xpby(&xg, 1.0 / 3.0, &mut ag).unwrap();
        check("xpby", &ac, &ag);

        cpu.scale(1.0 / 7.0, &mut ac).unwrap();
        gpu.scale(1.0 / 7.0, &mut ag).unwrap();
        check("scale", &ac, &ag);

        cpu.multiply(&xc, &yc, &mut sc).unwrap();
        gpu.multiply(&xg, &yg, &mut sg).unwrap();
        check("multiply", &sc, &sg);

        cpu.shift(-1.0 / 9.0, &mut ac).unwrap();
        gpu.shift(-1.0 / 9.0, &mut ag).unwrap();
        check("shift", &ac, &ag);

        cpu.fill(-0.0, &mut ac).unwrap();
        gpu.fill(-0.0, &mut ag).unwrap();
        check("fill", &ac, &ag);

        // 射影は sum と shift の組み合わせ（既定実装）
        let (mut pc, mut pg) = (cpu.upload(&x).unwrap(), gpu.upload(&x).unwrap());
        cpu.project_mean_zero(&mut pc).unwrap();
        gpu.project_mean_zero(&mut pg).unwrap();
        check("射影", &pc, &pg);
    }
}

#[test]
fn transfers_round_trip() {
    let gpu = gpu_or_skip!();
    let host = spread(10_000, 0.91);
    let v = gpu.upload(&host).unwrap();
    assert_eq!(gpu.len(&v), host.len());

    let mut back = vec![0.0; host.len()];
    gpu.download(&v, &mut back).unwrap();
    assert_eq!(host, back, "転送で値が変わってはいけない");

    let mut z = gpu.zeros(host.len()).unwrap();
    let mut zeros = vec![1.0; host.len()];
    gpu.download(&z, &mut zeros).unwrap();
    assert!(zeros.iter().all(|&v| v == 0.0));

    gpu.copy_from(&v, &mut z).unwrap();
    gpu.download(&z, &mut back).unwrap();
    assert_eq!(host, back, "VRAM 内のコピー");

    // 既存のベクトルへの書き戻し（AMG の最粗レベルがこれを毎サイクル使う）
    let other = spread(host.len(), 0.13);
    gpu.upload_into(&other, &mut z).unwrap();
    gpu.download(&z, &mut back).unwrap();
    assert_eq!(other, back, "既存ベクトルへの書き戻し");
}

#[test]
fn rectangular_matrices_multiply_the_same_way() {
    // AMG の補間 P（細 × 粗）と制限 R（粗 × 細）は矩形（SPEC §4.5）。
    // GPU の SpMV は行ごと 1 スレッドなので矩形でもそのまま動くはずだが、
    // 長さの検査を間違えると静かに壊れるので固定しておく
    let gpu = gpu_or_skip!();
    let (n_fine, n_coarse) = (5_000usize, 617);

    // 各行 1 非ゼロの試行補間そのもの
    let row_ptr: Vec<usize> = (0..=n_fine).collect();
    let col_idx: Vec<u32> = (0..n_fine).map(|i| (i % n_coarse) as u32).collect();
    let mut p = CsrMatrix::from_structure_rect(n_fine, n_coarse, row_ptr, col_idx);
    let values = spread(n_fine, 0.29);
    p.values_mut().copy_from_slice(&values);
    let r = p.transpose();

    let cpu = CpuSerial;
    for m in [&p, &r] {
        let (mc, mg) = (cpu.upload_matrix(m).unwrap(), gpu.upload_matrix(m).unwrap());
        let x = spread(m.n_cols(), 0.71);
        let (xc, xg) = (cpu.upload(&x).unwrap(), gpu.upload(&x).unwrap());
        let mut yc = cpu.zeros(m.n_rows()).unwrap();
        let mut yg = gpu.zeros(m.n_rows()).unwrap();
        cpu.spmv(&mc, &xc, &mut yc).unwrap();
        gpu.spmv(&mg, &xg, &mut yg).unwrap();
        let mut back = vec![0.0; m.n_rows()];
        gpu.download(&yg, &mut back).unwrap();
        assert_eq!(yc, back, "{} x {} の SpMV", m.n_rows(), m.n_cols());
    }
}

#[test]
fn the_reduction_tree_has_the_same_shape_as_the_cpu() {
    // 総和の木の形が同じであることは、丸めの出方でしか確かめられない。
    // 素朴な逐次和とは違う値になり、かつ CPU と一致する——という形で見る
    let gpu = gpu_or_skip!();
    let n = BLOCK * 40 + 17;
    // 先頭に巨大な値を置く。素朴な逐次和では以降の 1.0 が丸めで飲まれて消えるが、
    // ブロックごとに足してから合わせる木では 512 ずつ積み上がって残る
    let v: Vec<f64> = (0..n).map(|i| if i == 0 { 1.0e16 } else { 1.0 }).collect();

    let cpu = CpuSerial;
    let vc = cpu.upload(&v).unwrap();
    let vg = gpu.upload(&v).unwrap();
    let expected = cpu.sum(&vc).unwrap();

    let naive = v.iter().fold(0.0f64, |a, &b| a + b);
    assert_ne!(
        naive, expected,
        "この入力では素朴な逐次和と木の総和が違うはず"
    );
    assert_eq!(gpu.sum(&vg).unwrap(), expected, "GPU の総和");
    assert_eq!(
        gpu.sum(&vg).unwrap(),
        CpuParallel.sum(&v.to_vec()).unwrap(),
        "CPU 並列とも一致すること"
    );
}

#[test]
fn repeated_runs_give_identical_bits() {
    // GPU のスレッド割り当ては実行ごとに変わりうる
    let gpu = gpu_or_skip!();
    let n = 100_000;
    let x = spread(n, 0.53);
    let v = gpu.upload(&x).unwrap();
    let first = gpu.dot(&v, &v).unwrap();
    for _ in 0..5 {
        assert_eq!(gpu.dot(&v, &v).unwrap(), first);
    }
}
