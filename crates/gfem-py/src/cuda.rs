//! GPU を使うかどうかの判断（SPEC §4.4 / §7 / §8 M5）。
//!
//! `gfem-core` は CUDA に一切依存しない（CLAUDE.md 設計原則 5）ので、
//! 「GPU があるか」「VRAM が足りるか」「駄目なら CPU に落とす」の判断はここで行う。
//! 判断の結果は必ず文章で返し、黙って CPU に落ちることはない（SPEC §4.4）。
//!
//! `cuda` フィーチャを外すと、常に「CUDA 無しでビルドされている」と答える版になる。

use gfem_core::case::Backend;

/// GPU を試すか判断し、使えるならソルバを、駄目なら理由を返す。
///
/// 返り値の第 2 要素は、`Some` なら**必ず結果の警告に載せる**こと。
#[cfg(feature = "cuda")]
pub fn pick(
    backend: Backend,
    n_nodes: usize,
    nnz: Option<usize>,
) -> (Option<Box<dyn gfem_core::LinearSolver>>, Option<String>) {
    if !backend.wants_gpu(n_nodes) {
        return (None, None);
    }
    let (solver, note) = gfem_cuda::solver_or_fallback(n_nodes, nnz);
    match (solver, note) {
        (Some(s), _) => (Some(s), None),
        // auto のときは「GPU が無いので CPU で解いた」は当たり前なので黙っている。
        // 明示的に cuda を指定されたときだけ、落ちた理由を必ず伝える
        (None, note) if backend == Backend::Cuda => (None, note),
        (None, _) => (None, None),
    }
}

/// CUDA 無しでビルドした版。
#[cfg(not(feature = "cuda"))]
pub fn pick(
    backend: Backend,
    n_nodes: usize,
    _nnz: Option<usize>,
) -> (Option<Box<dyn gfem_core::LinearSolver>>, Option<String>) {
    if backend == Backend::Cuda {
        return (
            None,
            Some(
                "このビルドには CUDA バックエンドが入っていません（cuda フィーチャ無し）。\
                 CPU で解きます"
                    .to_string(),
            ),
        );
    }
    let _ = n_nodes;
    (None, None)
}

/// この規模を GPU で解けるか。解けるなら `None`、駄目なら理由。
#[cfg(feature = "cuda")]
pub fn check(n_nodes: usize, nnz: Option<usize>) -> Option<String> {
    gfem_cuda::can_use_gpu(n_nodes, nnz).err()
}

/// CUDA 無しでビルドした版。
#[cfg(not(feature = "cuda"))]
pub fn check(_n_nodes: usize, _nnz: Option<usize>) -> Option<String> {
    Some("このビルドには CUDA バックエンドが入っていません".to_string())
}

/// 見えている GPU の一覧（Python へ返す用）。
#[cfg(feature = "cuda")]
pub fn device_list() -> Vec<(usize, String, i32, i32, usize, usize)> {
    gfem_cuda::devices()
        .into_iter()
        .map(|d| {
            (
                d.ordinal,
                d.name,
                d.compute_capability.0,
                d.compute_capability.1,
                d.total_memory,
                d.free_memory,
            )
        })
        .collect()
}

/// CUDA 無しでビルドした版。
#[cfg(not(feature = "cuda"))]
pub fn device_list() -> Vec<(usize, String, i32, i32, usize, usize)> {
    Vec::new()
}

/// このビルドに CUDA バックエンドが入っているか。
pub const fn built_with_cuda() -> bool {
    cfg!(feature = "cuda")
}
