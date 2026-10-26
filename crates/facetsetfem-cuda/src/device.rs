//! CUDA コンテキストの初期化と、使えるかどうかの判定（SPEC §4.4 / §8 M5）。
//!
//! # 初期化は 1 回だけ
//!
//! コンテキストの作成と NVRTC によるカーネルのコンパイルは合わせて 0.2 秒ほどかかる。
//! 以前は解くたびにやり直していて、GPU を調べ直すぶん（1 枚 60 ms 前後）と合わせると
//! 1 回 0.37 秒——28 万節点の解析本体（約 1 秒）の 4 割近くを毎回払っていた。
//! いまは初期化済みのデバイスを [`Device::shared`] でプロセスが終わるまで使い回す。
//! どの GPU で・何度目に解いても答えはビット単位で同じなので、結果には影響しない。

use crate::error::cuda_error;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use facetsetfem_core::error::{FacetSetFemError, Result};
use std::sync::{Arc, Mutex};

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
    /// 空き VRAM [byte]（調べた時点の値）
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
        name: ctx
            .name()
            .unwrap_or_else(|_| facetsetfem_core::tr!("Unknown GPU", "不明な GPU")),
        compute_capability: ctx.compute_capability().ok()?,
        total_memory,
        free_memory,
    })
}

/// 初期化済みのデバイス（デバイス番号ごとに 1 つ）。プロセスが終わるまで持ち続ける。
///
/// 持ち続けるので、一度 GPU で解いたプロセスはコンテキストのぶんの VRAM（数百 MB）を
/// 掴んだままになる。解くための VRAM は解き終わるたびに返す（[`crate::CudaBackend`] の `Drop`）。
static READY: Mutex<Vec<Arc<Device>>> = Mutex::new(Vec::new());

/// 初期化済みのデバイスの素性。空き VRAM は今の値で、調べられなかったものは除く。
///
/// コンテキストが生きているので、ここは GPU を調べ直すより桁違いに速い（0.1 ms 未満）。
pub(crate) fn ready_devices() -> Vec<DeviceInfo> {
    let ready = READY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    ready
        .iter()
        .filter_map(|d| {
            Some(DeviceInfo {
                free_memory: d.free_memory().ok()?,
                ..d.info.clone()
            })
        })
        .collect()
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
    pub(crate) fill: CudaFunction,
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
    /// デバイス `ordinal` を返す。初回だけ初期化し、以後は同じものを返す。
    ///
    /// 同時に呼ばれても初期化は 1 回だけ（初期化の間はロックを持つ）。初期化に失敗したら
    /// 何も覚えないので、次の呼び出しでやり直す。
    pub fn shared(ordinal: usize) -> Result<Arc<Device>> {
        let mut ready = READY
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(device) = ready.iter().find(|d| d.info.ordinal == ordinal) {
            return Ok(Arc::clone(device));
        }
        let device = Arc::new(Device::new(ordinal)?);
        ready.push(Arc::clone(&device));
        Ok(device)
    }

    /// デバイス `ordinal` を新しく初期化し、カーネルをコンパイルして読み込む。
    ///
    /// ふつうは使い回す [`Device::shared`] を使う。こちらは呼ぶたびに 0.2 秒ほどかかる。
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal).map_err(|e| {
            cuda_error(
                facetsetfem_core::tr!(
                    "Could not initialize GPU {ordinal}",
                    "GPU {ordinal} を初期化できませんでした"
                ),
                &e,
            )
            .with_hint(facetsetfem_core::tr!(
                "Check that the NVIDIA driver is installed and that the GPU is not occupied by \
                 another process (nvidia-smi)",
                "NVIDIA ドライバが入っているか、GPU が他のプロセスに占有されていないか\
                 確認してください（nvidia-smi）"
            ))
        })?;
        let info = DeviceInfo {
            ordinal,
            name: ctx
                .name()
                .unwrap_or_else(|_| facetsetfem_core::tr!("Unknown GPU", "不明な GPU")),
            compute_capability: ctx.compute_capability().map_err(|e| {
                cuda_error(
                    facetsetfem_core::tr!(
                        "Could not get the Compute Capability",
                        "Compute Capability を取得できませんでした"
                    ),
                    &e,
                )
            })?,
            total_memory: ctx.total_mem().unwrap_or(0),
            free_memory: ctx.mem_get_info().map(|(free, _)| free).unwrap_or(0),
        };
        if !info.supports_f64() {
            let (major, minor) = info.compute_capability;
            let (rmajor, rminor) = MIN_COMPUTE_CAPABILITY;
            return Err(FacetSetFemError::unsupported(facetsetfem_core::tr!(
                "The Compute Capability of GPU {ordinal} ({}) is {major}.{minor}, below the \
                 {rmajor}.{rminor} needed for double-precision arithmetic",
                "GPU {ordinal}（{}）の Compute Capability は {major}.{minor} で、\
                 倍精度演算に必要な {rmajor}.{rminor} に足りません",
                info.name
            ))
            .with_hint(facetsetfem_core::tr!(
                "All computations are done in f64, so there is no fallback to single precision \
                 (CLAUDE.md design principle 2). Use backend: cpu_parallel",
                "全計算を f64 で行うため単精度への切り替えはしません（CLAUDE.md 設計原則 2）。\
                 backend: cpu_parallel を使ってください"
            )));
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
            FacetSetFemError::internal(facetsetfem_core::tr!(
                "Failed to compile the CUDA kernels: {e}",
                "CUDA カーネルのコンパイルに失敗しました: {e}"
            ))
            .with_hint(facetsetfem_core::tr!(
                "nvrtc from the CUDA Toolkit may not be found. Check that CUDA_PATH is set",
                "CUDA Toolkit の nvrtc が見つからない可能性があります。\
                 CUDA_PATH が通っているか確認してください"
            ))
        })?;

        let module = ctx.load_module(ptx).map_err(|e| {
            cuda_error(
                facetsetfem_core::tr!(
                    "Could not load the CUDA module",
                    "CUDA モジュールを読み込めませんでした"
                ),
                &e,
            )
        })?;
        let load = |name: &str| -> Result<CudaFunction> {
            module.load_function(name).map_err(|e| {
                cuda_error(
                    facetsetfem_core::tr!(
                        "Kernel {name} not found",
                        "カーネル {name} が見つかりません"
                    ),
                    &e,
                )
            })
        };

        Ok(Self {
            stream: ctx.default_stream(),
            spmv: load("facetsetfem_spmv")?,
            axpy: load("facetsetfem_axpy")?,
            xpby: load("facetsetfem_xpby")?,
            scale: load("facetsetfem_scale")?,
            multiply: load("facetsetfem_multiply")?,
            shift: load("facetsetfem_shift")?,
            fill: load("facetsetfem_fill")?,
            block_sum: load("facetsetfem_block_sum")?,
            block_dot: load("facetsetfem_block_dot")?,
            info,
            _module: module,
            _ctx: ctx,
        })
    }

    /// 使っている GPU の素性（空き VRAM は初期化した時点の値）。
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// いまの空き VRAM [byte]。
    pub fn free_memory(&self) -> Result<usize> {
        self._ctx.mem_get_info().map(|(free, _)| free).map_err(|e| {
            cuda_error(
                facetsetfem_core::tr!(
                    "Could not get the free VRAM",
                    "空き VRAM を取得できませんでした"
                ),
                &e,
            )
        })
    }
}
