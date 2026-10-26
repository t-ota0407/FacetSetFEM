//! CUDA バックエンド（SPEC §4.4 / §8 M5）。
//!
//! [`facetsetfem_core::LinearBackend`] を実装する。CG 本体は `facetsetfem-core` の 1 箇所のまま
//! （SPEC §4.4）で、ここはベクトル演算の中身だけを差し替える。
//!
//! # VRAM 常駐
//!
//! CSR（row_ptr / col_idx / values）と作業ベクトルは最初に一度だけ転送し、
//! **反復の間はホストへ戻さない**。ホストへ落ちるのは内積・総和の結果という
//! スカラーだけで、その 1 個のためだけに毎反復 4 KB 以下の転送が 1 回入る。
//!
//! # ビット単位一致
//!
//! カーネルの書き方（FMA を作らせない・総和の木を CPU と揃える）は
//! `kernels.cu` の冒頭に書いた。ここでは**総和の再帰をホスト側で CPU と
//! 同じ形にたどる**ことを担当する。

use crate::device::{Device, THREADS};
use crate::error::cuda_error;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use facetsetfem_core::assemble::CsrMatrix;
use facetsetfem_core::error::{FacetSetFemError, Result};
use facetsetfem_core::solver::LinearBackend;
use facetsetfem_core::solver::reduce::{self, BLOCK};
use std::sync::Arc;

/// CSR の転送に失敗したときのエラー。VRAM 不足が圧倒的に多いので、そう書く。
///
/// `what` は（英語, 日本語）の組。エラーのときだけ片方を選ぶ。
fn upload_err(what: (&str, &str), e: &cudarc::driver::DriverError) -> FacetSetFemError {
    let message = if facetsetfem_core::i18n::is_ja() {
        format!("{}を VRAM へ転送できませんでした", what.1)
    } else {
        format!("Could not transfer the {} to VRAM", what.0)
    };
    cuda_error(message, e).with_hint(facetsetfem_core::tr!(
        "VRAM may be insufficient. backend: cpu_parallel solves using main memory only",
        "VRAM が足りない可能性があります。backend: cpu_parallel なら主記憶だけで解けます"
    ))
}

/// VRAM 上の CSR 行列。
#[derive(Debug)]
pub struct CudaCsr {
    row_ptr: CudaSlice<u64>,
    col_idx: CudaSlice<u32>,
    values: CudaSlice<f64>,
    n_rows: usize,
    /// 列数。AMG の補間・制限は矩形なので行数と違う（SPEC §4.5）
    n_cols: usize,
}

impl CudaCsr {
    /// VRAM 上での大きさ [byte]。
    pub fn bytes(&self) -> usize {
        self.row_ptr.len() * 8 + self.col_idx.len() * 4 + self.values.len() * 8
    }
}

/// VRAM 上のベクトル。
#[derive(Debug)]
pub struct CudaVector {
    data: CudaSlice<f64>,
    len: usize,
}

/// CUDA バックエンド。
///
/// デバイス（コンテキストとコンパイル済みのカーネル）は同じ GPU を使うバックエンドどうしで
/// 共有する（[`Device::shared`]）。別のスレッドで同時に解いても、ストリームは 1 本なので
/// カーネルは順に流れ、それぞれの VRAM は別なので答えは変わらない。
#[derive(Debug)]
pub struct CudaBackend {
    device: Arc<Device>,
}

impl CudaBackend {
    /// デバイス `ordinal` を使う。初期化はプロセスで初回だけ（[`Device::shared`]）。
    pub fn new(ordinal: usize) -> Result<Self> {
        Ok(Self {
            device: Device::shared(ordinal)?,
        })
    }

    /// 使っている GPU。
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// `n` 要素を 1 スレッド 1 要素で処理する起動設定。
    fn config(n: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (n.div_ceil(THREADS as usize).max(1) as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// ブロック和・ブロック内積を 1 段だけ走らせる。
    ///
    /// `v` の長さ `n` を長さ [`BLOCK`] の連続ブロックに切り、ブロックごとに
    /// 1 スレッドが**先頭から順に**足す。CPU の `reduce::sum_block` と同じ。
    fn block_reduce(
        &self,
        v: &CudaSlice<f64>,
        second: Option<&CudaSlice<f64>>,
        n: usize,
    ) -> Result<CudaSlice<f64>> {
        let blocks = n.div_ceil(BLOCK);
        let mut out = self.device.stream.alloc_zeros::<f64>(blocks).map_err(|e| {
            cuda_error(
                facetsetfem_core::tr!(
                    "Could not allocate VRAM for the reduction",
                    "リダクション用の VRAM を確保できませんでした"
                ),
                &e,
            )
        })?;
        let (n64, block64) = (n as u64, BLOCK as u64);
        let cfg = Self::config(blocks);
        let launch = match second {
            Some(y) => {
                let mut b = self.device.stream.launch_builder(&self.device.block_dot);
                b.arg(&n64).arg(&block64).arg(v).arg(y).arg(&mut out);
                unsafe { b.launch(cfg) }
            }
            None => {
                let mut b = self.device.stream.launch_builder(&self.device.block_sum);
                b.arg(&n64).arg(&block64).arg(v).arg(&mut out);
                unsafe { b.launch(cfg) }
            }
        };
        launch.map_err(|e| {
            cuda_error(
                facetsetfem_core::tr!(
                    "Failed to launch the reduction kernel",
                    "リダクションのカーネル起動に失敗しました"
                ),
                &e,
            )
        })?;
        Ok(out)
    }

    /// ブロック和の列を、CPU の [`reduce::sum`] と同じ木でたたむ。
    ///
    /// `reduce::sum` は「長さが [`BLOCK`] 以下になるまでブロック和を繰り返し、
    /// 最後に逐次で足す」。同じ手順をここでもたどる。最後の 512 要素以下は
    /// ホストへ落として CPU 側の関数をそのまま呼ぶので、規則がずれようがない。
    fn finish(&self, partial: CudaSlice<f64>) -> Result<f64> {
        let mut current = partial;
        while current.len() > BLOCK {
            let n = current.len();
            current = self.block_reduce(&current, None, n)?;
        }
        let host = self.device.stream.clone_dtoh(&current).map_err(|e| {
            cuda_error(
                facetsetfem_core::tr!(
                    "Could not retrieve the reduction result",
                    "リダクション結果を取り出せませんでした"
                ),
                &e,
            )
        })?;
        self.sync()?;
        Ok(reduce::sum(&host))
    }

    fn sync(&self) -> Result<()> {
        self.device.stream.synchronize().map_err(|e| {
            cuda_error(
                facetsetfem_core::tr!("Failed to synchronize the GPU", "GPU の同期に失敗しました"),
                &e,
            )
        })
    }

    /// 同じ長さであることを確かめる。
    fn same_len(a: usize, b: usize) -> Result<()> {
        if a != b {
            return Err(FacetSetFemError::internal(facetsetfem_core::tr!(
                "Vector lengths differ: {a} and {b}",
                "ベクトルの長さが違います: {a} と {b}"
            )));
        }
        Ok(())
    }
}

impl Drop for CudaBackend {
    /// 解くのに使った VRAM を、プールから GPU へ返させる。
    ///
    /// cudarc は VRAM をストリーム順序付きのプール（`cuMemAllocAsync`）から取り、解放も
    /// ストリームに積むだけにする。プールが VRAM を実際に手放すのは次の同期のとき
    /// （既定の release threshold は 0）。デバイスを使い回すようになってからは、ここで
    /// 同期しないと、解き終わったあとも VRAM を掴んだままになる（以前はコンテキストごと
    /// 捨てていたので問題にならなかった）。
    fn drop(&mut self) {
        let _ = self.device.stream.synchronize();
    }
}

impl LinearBackend for CudaBackend {
    type Vector = CudaVector;
    type Matrix<'a> = CudaCsr;

    fn name(&self) -> &'static str {
        "cuda"
    }

    fn upload_matrix(&self, a: &CsrMatrix) -> Result<CudaCsr> {
        let row_ptr: Vec<u64> = a.row_ptr().iter().map(|&v| v as u64).collect();
        let stream = &self.device.stream;
        Ok(CudaCsr {
            row_ptr: stream
                .clone_htod(&row_ptr)
                .map_err(|e| upload_err(("row pointers", "行ポインタ"), &e))?,
            col_idx: stream
                .clone_htod(a.col_idx())
                .map_err(|e| upload_err(("column indices", "列番号"), &e))?,
            values: stream
                .clone_htod(a.values())
                .map_err(|e| upload_err(("matrix values", "成分"), &e))?,
            n_rows: a.n_rows(),
            n_cols: a.n_cols(),
        })
    }

    fn zeros(&self, n: usize) -> Result<CudaVector> {
        let data = self.device.stream.alloc_zeros::<f64>(n).map_err(|e| {
            cuda_error(
                facetsetfem_core::tr!(
                    "Could not allocate VRAM for {n} elements",
                    "{n} 要素の VRAM を確保できませんでした"
                ),
                &e,
            )
        })?;
        Ok(CudaVector { data, len: n })
    }

    fn upload(&self, host: &[f64]) -> Result<CudaVector> {
        let data = self.device.stream.clone_htod(host).map_err(|e| {
            cuda_error(
                facetsetfem_core::tr!(
                    "Could not transfer the vector to VRAM",
                    "ベクトルを VRAM へ転送できませんでした"
                ),
                &e,
            )
        })?;
        Ok(CudaVector {
            data,
            len: host.len(),
        })
    }

    fn download(&self, x: &CudaVector, host: &mut [f64]) -> Result<()> {
        Self::same_len(x.len, host.len())?;
        self.device.stream.memcpy_dtoh(&x.data, host).map_err(|e| {
            cuda_error(
                facetsetfem_core::tr!(
                    "Could not copy the vector back to the host",
                    "ベクトルをホストへ戻せませんでした"
                ),
                &e,
            )
        })?;
        self.sync()
    }

    fn upload_into(&self, host: &[f64], x: &mut CudaVector) -> Result<()> {
        Self::same_len(x.len, host.len())?;
        self.device
            .stream
            .memcpy_htod(host, &mut x.data)
            .map_err(|e| {
                cuda_error(
                    facetsetfem_core::tr!(
                        "Could not write the vector back to VRAM",
                        "ベクトルを VRAM へ書き戻せませんでした"
                    ),
                    &e,
                )
            })
    }

    fn copy_from(&self, src: &CudaVector, dst: &mut CudaVector) -> Result<()> {
        Self::same_len(src.len, dst.len)?;
        self.device
            .stream
            .memcpy_dtod(&src.data, &mut dst.data)
            .map_err(|e| {
                cuda_error(
                    facetsetfem_core::tr!(
                        "Failed to copy within VRAM",
                        "VRAM 内のコピーに失敗しました"
                    ),
                    &e,
                )
            })
    }

    fn len(&self, x: &CudaVector) -> usize {
        x.len
    }

    fn spmv(&self, a: &CudaCsr, x: &CudaVector, y: &mut CudaVector) -> Result<()> {
        Self::same_len(a.n_cols, x.len)?;
        Self::same_len(a.n_rows, y.len)?;
        let n = a.n_rows as u64;
        let mut b = self.device.stream.launch_builder(&self.device.spmv);
        b.arg(&n)
            .arg(&a.row_ptr)
            .arg(&a.col_idx)
            .arg(&a.values)
            .arg(&x.data)
            .arg(&mut y.data);
        unsafe { b.launch(Self::config(a.n_rows)) }
            .map(|_| ())
            .map_err(|e| {
                cuda_error(
                    facetsetfem_core::tr!(
                        "Failed to launch the SpMV kernel",
                        "SpMV のカーネル起動に失敗しました"
                    ),
                    &e,
                )
            })
    }

    fn axpy(&self, alpha: f64, x: &CudaVector, y: &mut CudaVector) -> Result<()> {
        Self::same_len(x.len, y.len)?;
        let n = x.len as u64;
        let mut b = self.device.stream.launch_builder(&self.device.axpy);
        b.arg(&n).arg(&alpha).arg(&x.data).arg(&mut y.data);
        unsafe { b.launch(Self::config(x.len)) }
            .map(|_| ())
            .map_err(|e| {
                cuda_error(
                    facetsetfem_core::tr!(
                        "Failed to launch the axpy kernel",
                        "axpy のカーネル起動に失敗しました"
                    ),
                    &e,
                )
            })
    }

    fn xpby(&self, x: &CudaVector, beta: f64, y: &mut CudaVector) -> Result<()> {
        Self::same_len(x.len, y.len)?;
        let n = x.len as u64;
        let mut b = self.device.stream.launch_builder(&self.device.xpby);
        b.arg(&n).arg(&x.data).arg(&beta).arg(&mut y.data);
        unsafe { b.launch(Self::config(x.len)) }
            .map(|_| ())
            .map_err(|e| {
                cuda_error(
                    facetsetfem_core::tr!(
                        "Failed to launch the xpby kernel",
                        "xpby のカーネル起動に失敗しました"
                    ),
                    &e,
                )
            })
    }

    fn dot(&self, x: &CudaVector, y: &CudaVector) -> Result<f64> {
        Self::same_len(x.len, y.len)?;
        if x.len == 0 {
            return Ok(0.0);
        }
        let partial = self.block_reduce(&x.data, Some(&y.data), x.len)?;
        if x.len <= BLOCK {
            // CPU 側は dot_block の結果をそのまま返す（総和の再帰に入らない）
            let host = self.device.stream.clone_dtoh(&partial).map_err(|e| {
                cuda_error(
                    facetsetfem_core::tr!(
                        "Could not retrieve the dot product result",
                        "内積の結果を取り出せませんでした"
                    ),
                    &e,
                )
            })?;
            self.sync()?;
            return Ok(host[0]);
        }
        self.finish(partial)
    }

    fn sum(&self, x: &CudaVector) -> Result<f64> {
        if x.len == 0 {
            return Ok(0.0);
        }
        let partial = self.block_reduce(&x.data, None, x.len)?;
        if x.len <= BLOCK {
            let host = self.device.stream.clone_dtoh(&partial).map_err(|e| {
                cuda_error(
                    facetsetfem_core::tr!(
                        "Could not retrieve the sum result",
                        "総和の結果を取り出せませんでした"
                    ),
                    &e,
                )
            })?;
            self.sync()?;
            return Ok(host[0]);
        }
        self.finish(partial)
    }

    fn scale(&self, alpha: f64, x: &mut CudaVector) -> Result<()> {
        let n = x.len as u64;
        let mut b = self.device.stream.launch_builder(&self.device.scale);
        b.arg(&n).arg(&alpha).arg(&mut x.data);
        unsafe { b.launch(Self::config(x.len)) }
            .map(|_| ())
            .map_err(|e| {
                cuda_error(
                    facetsetfem_core::tr!(
                        "Failed to launch the scale kernel",
                        "scale のカーネル起動に失敗しました"
                    ),
                    &e,
                )
            })
    }

    fn multiply(&self, a: &CudaVector, x: &CudaVector, y: &mut CudaVector) -> Result<()> {
        Self::same_len(a.len, x.len)?;
        Self::same_len(a.len, y.len)?;
        let n = a.len as u64;
        let mut b = self.device.stream.launch_builder(&self.device.multiply);
        b.arg(&n).arg(&a.data).arg(&x.data).arg(&mut y.data);
        unsafe { b.launch(Self::config(a.len)) }
            .map(|_| ())
            .map_err(|e| {
                cuda_error(
                    facetsetfem_core::tr!(
                        "Failed to launch the multiply kernel",
                        "multiply のカーネル起動に失敗しました"
                    ),
                    &e,
                )
            })
    }

    fn shift(&self, delta: f64, x: &mut CudaVector) -> Result<()> {
        let n = x.len as u64;
        let mut b = self.device.stream.launch_builder(&self.device.shift);
        b.arg(&n).arg(&delta).arg(&mut x.data);
        unsafe { b.launch(Self::config(x.len)) }
            .map(|_| ())
            .map_err(|e| {
                cuda_error(
                    facetsetfem_core::tr!(
                        "Failed to launch the shift kernel",
                        "shift のカーネル起動に失敗しました"
                    ),
                    &e,
                )
            })
    }

    fn fill(&self, value: f64, x: &mut CudaVector) -> Result<()> {
        let n = x.len as u64;
        let mut b = self.device.stream.launch_builder(&self.device.fill);
        b.arg(&n).arg(&value).arg(&mut x.data);
        unsafe { b.launch(Self::config(x.len)) }
            .map(|_| ())
            .map_err(|e| {
                cuda_error(
                    facetsetfem_core::tr!(
                        "Failed to launch the fill kernel",
                        "fill のカーネル起動に失敗しました"
                    ),
                    &e,
                )
            })
    }
}
