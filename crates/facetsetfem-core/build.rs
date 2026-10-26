//! ビルド時に Git の状態とターゲットを埋め込む（SPEC §5.5 のプロビナンス記録）。
//!
//! 仕組みは `build_git.rs`（`facetsetfem-py` と共有）を参照。ここでは **facetsetfem-core 自身の
//! ソース**だけを数える。facetsetfem-core を編集すればどのみち再コンパイルになるので、
//! 監視のせいでビルドが余計に走ることはない。
//!
//! Python から書く実行サマリは、`facetsetfem-cuda`・`facetsetfem-py` まで含めた最終バイナリの
//! 状態（`crates/facetsetfem-py/build.rs`）で上書きする。

include!("build_git.rs");

fn main() {
    println!(
        "cargo:rustc-env=FACETSETFEM_TARGET={}",
        std::env::var("TARGET").unwrap_or_else(|_| "unknown".into())
    );
    embed_git_state(&[
        "crates/facetsetfem-core/src",
        "crates/facetsetfem-core/build.rs",
        "crates/facetsetfem-core/build_git.rs",
        "crates/facetsetfem-core/Cargo.toml",
        "Cargo.toml",
        "Cargo.lock",
    ]);
}
