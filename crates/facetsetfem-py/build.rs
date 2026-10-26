//! 最終バイナリ（Python 拡張）を作ったソースの Git の状態を埋め込む（SPEC §5.5）。
//!
//! facetsetfem-core の `build.rs` が埋め込むのは facetsetfem-core 自身の状態だけなので、
//! facetsetfem-cuda・facetsetfem-py を編集した場合はそこに現れない。Python から書く実行サマリでは
//! こちらの値を使う。facetsetfem-py はどのクレートを編集しても再コンパイルされるので、
//! 監視のせいでビルドが余計に走ることはない。仕組みは `build_git.rs` を参照。

include!("../facetsetfem-core/build_git.rs");

fn main() {
    embed_git_state(&[
        "crates/facetsetfem-core/src",
        "crates/facetsetfem-core/build.rs",
        "crates/facetsetfem-core/build_git.rs",
        "crates/facetsetfem-core/Cargo.toml",
        "crates/facetsetfem-cuda/src",
        "crates/facetsetfem-cuda/Cargo.toml",
        "crates/facetsetfem-py/src",
        "crates/facetsetfem-py/build.rs",
        "crates/facetsetfem-py/Cargo.toml",
        "Cargo.toml",
        "Cargo.lock",
    ]);
}
