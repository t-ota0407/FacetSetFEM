//! GPU と CPU がビット単位で一致することの検証（SPEC §8 M5 / §10）。
//!
//! CLAUDE.md 設計原則 1 は「同一入力から CPU 逐次・CPU 並列・GPU のいずれでも
//! ビット単位で同じ結果」を要求している。M5 でいちばん大事なテストはこれ。
//!
//! GPU が無い環境ではテストごとに `skip` して通す（`gfem-core` は CPU だけで
//! 完全にテストできる状態を保つ。設計原則 5）。GPU があるのに落ちたら本物の不具合。
//!
//! ```text
//! cargo test -p gfem-cuda --release
//! ```

use gfem_core::assemble::CsrMatrix;
use gfem_core::solver::reduce::BLOCK;
use gfem_core::solver::{CpuParallel, CpuSerial, LinearBackend};
use gfem_cuda::CudaBackend;

/// GPU が使えなければ `None`。使えなければテストは黙って通す。
fn gpu() -> Option<CudaBackend> {
    if gfem_cuda::devices().iter().all(|d| !d.supports_f64()) {
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

/// 1 次元格子ラプラシアン（全 Neumann）。零空間がちょうど定数ベクトル。
fn laplacian_1d(n: usize) -> CsrMatrix {
    let mut row_ptr = vec![0usize];
    let mut col_idx: Vec<u32> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    for i in 0..n {
        let mut entries: Vec<(u32, f64)> = Vec::new();
        let mut diag = 0.0;
        if i > 0 {
            entries.push(((i - 1) as u32, -1.0));
            diag += 1.0;
        }
        if i + 1 < n {
            entries.push(((i + 1) as u32, -1.0));
            diag += 1.0;
        }
        entries.push((i as u32, diag));
        entries.sort_by_key(|&(c, _)| c);
        for (c, v) in entries {
            col_idx.push(c);
            values.push(v);
        }
        row_ptr.push(col_idx.len());
    }
    let mut m = CsrMatrix::from_structure(n, row_ptr, col_idx);
    m.values_mut().copy_from_slice(&values);
    m
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
        let check = |name: &str, c: &Vec<f64>, g: &gfem_cuda::CudaVector| {
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
