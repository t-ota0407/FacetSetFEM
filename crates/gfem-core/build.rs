//! ビルド時に Git の状態とターゲットを埋め込む（SPEC §5.5 のプロビナンス記録）。
//!
//! 仕組みは `build_git.rs`（`gfem-py` と共有）を参照。ここでは **gfem-core 自身の
//! ソース**だけを数える。gfem-core を編集すればどのみち再コンパイルになるので、
//! 監視のせいでビルドが余計に走ることはない。
//!
//! Python から書く実行サマリは、`gfem-cuda`・`gfem-py` まで含めた最終バイナリの
//! 状態（`crates/gfem-py/build.rs`）で上書きする。

include!("build_git.rs");

fn main() {
    println!(
        "cargo:rustc-env=GFEM_TARGET={}",
        std::env::var("TARGET").unwrap_or_else(|_| "unknown".into())
    );
    embed_git_state(&[
        "crates/gfem-core/src",
        "crates/gfem-core/build.rs",
        "crates/gfem-core/build_git.rs",
        "crates/gfem-core/Cargo.toml",
        "Cargo.toml",
        "Cargo.lock",
    ]);
}
