//! CUDA コンテキストの初期化と、使えるかどうかの判定（SPEC §4.4 / §8 M5）。

use crate::error::cuda_error;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use gfem_core::error::{GfemError, Result};
use std::sync::Arc;

/// 倍精度演算に必要な Compute Capability（SPEC §4.4）。
pub const MIN_COMPUTE_CAPABILITY: (i32, i32) = (6, 0);

/// カーネルのソース。実行時に NVRTC でコンパイルする。
const KERNELS: &str = include_str!("kernels.cu");

/// カーネル 1 つあたりの CUDA スレッド数。
pub const THREADS: u32 = 256;

/// 使う GPU の素性。エラーメッセージとプロビナンスに載せる。
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// デバイス番号
    pub ordinal: usize,
    /// 製品名
    pub name: String,
    /// Compute Capability
    pub compute_capability: (i32, i32),
    /// 総 VRAM [byte]
    pub total_memory: usize,
    /// 空き VRAM [byte]
    pub free_memory: usize,
}

impl DeviceInfo {
    /// 倍精度で使えるか。
    pub fn supports_f64(&self) -> bool {
        self.compute_capability >= MIN_COMPUTE_CAPABILITY
    }
}

/// 見えている GPU を列挙する。CUDA が無い環境では空を返す。
pub fn devices() -> Vec<DeviceInfo> {
    let Ok(count) = CudaContext::device_count() else {
        return Vec::new();
    };
    (0..count.max(0) as usize).filter_map(probe).collect()
}

fn probe(ordinal: usize) -> Option<DeviceInfo> {
    let ctx = CudaContext::new(ordinal).ok()?;
    let (free_memory, total_memory) = ctx.mem_get_info().ok()?;
    Some(DeviceInfo {
        ordinal,
        name: ctx.name().unwrap_or_else(|_| "不明な GPU".into()),
        compute_capability: ctx.compute_capability().ok()?,
        total_memory,
        free_memory,
    })
}

/// カーネル一式を読み込んだデバイス。
pub struct Device {
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) info: DeviceInfo,
    pub(crate) spmv: CudaFunction,
    pub(crate) axpy: CudaFunction,
    pub(crate) xpby: CudaFunction,
    pub(crate) scale: CudaFunction,
    pub(crate) multiply: CudaFunction,
    pub(crate) shift: CudaFunction,
    pub(crate) block_sum: CudaFunction,
    pub(crate) block_dot: CudaFunction,
    /// モジュールを生かしておく（関数はモジュールから借りている）
    _module: Arc<CudaModule>,
    _ctx: Arc<CudaContext>,
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device").field("info", &self.info).finish()
    }
}

impl Device {
    /// デバイス `ordinal` を初期化し、カーネルをコンパイルして読み込む。
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal).map_err(|e| {
            cuda_error(format!("GPU {ordinal} を初期化できませんでした"), &e).with_hint(
                "NVIDIA ドライバが入っているか、GPU が他のプロセスに占有されていないか\
                 確認してください（nvidia-smi）",
            )
        })?;
        let info = DeviceInfo {
            ordinal,
            name: ctx.name().unwrap_or_else(|_| "不明な GPU".into()),
            compute_capability: ctx
                .compute_capability()
                .map_err(|e| cuda_error("Compute Capability を取得できませんでした", &e))?,
            total_memory: ctx.total_mem().unwrap_or(0),
            free_memory: ctx.mem_get_info().map(|(free, _)| free).unwrap_or(0),
        };
        if !info.supports_f64() {
            let (major, minor) = info.compute_capability;
            let (rmajor, rminor) = MIN_COMPUTE_CAPABILITY;
            return Err(GfemError::unsupported(format!(
                "GPU {ordinal}（{}）の Compute Capability は {major}.{minor} で、\
                 倍精度演算に必要な {rmajor}.{rminor} に足りません",
                info.name
            ))
            .with_hint(
                "全計算を f64 で行うため単精度への切り替えはしません（CLAUDE.md 設計原則 2）。\
                 backend: cpu_parallel を使ってください",
            ));
        }

        let ptx = compile_ptx_with_opts(
            KERNELS,
            CompileOptions {
                // 積和を FMA にまとめると CPU と結果が変わる（kernels.cu の説明を参照）。
                // カーネル側でも __dmul_rn / __dadd_rn を明示しているが、二重に止める
                fmad: Some(false),
                // 非正規化数をゼロに丸めない。除算・平方根も正しく丸める
                ftz: Some(false),
                prec_div: Some(true),
                prec_sqrt: Some(true),
                use_fast_math: Some(false),
                ..Default::default()
            },
        )
        .map_err(|e| {
            GfemError::internal(format!("CUDA カーネルのコンパイルに失敗しました: {e}")).with_hint(
                "CUDA Toolkit の nvrtc が見つからない可能性があります。\
                 CUDA_PATH が通っているか確認してください",
            )
        })?;

        let module = ctx
            .load_module(ptx)
            .map_err(|e| cuda_error("CUDA モジュールを読み込めませんでした", &e))?;
        let load = |name: &str| -> Result<CudaFunction> {
            module
                .load_function(name)
                .map_err(|e| cuda_error(format!("カーネル {name} が見つかりません"), &e))
        };

        Ok(Self {
            stream: ctx.default_stream(),
            spmv: load("gfem_spmv")?,
            axpy: load("gfem_axpy")?,
            xpby: load("gfem_xpby")?,
            scale: load("gfem_scale")?,
            multiply: load("gfem_multiply")?,
            shift: load("gfem_shift")?,
            block_sum: load("gfem_block_sum")?,
            block_dot: load("gfem_block_dot")?,
            info,
            _module: module,
            _ctx: ctx,
        })
    }

    /// 使っている GPU の素性。
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }
}
