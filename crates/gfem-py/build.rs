//! 最終バイナリ（Python 拡張）を作ったソースの Git の状態を埋め込む（SPEC §5.5）。
//!
//! gfem-core の `build.rs` が埋め込むのは gfem-core 自身の状態だけなので、
//! gfem-cuda・gfem-py を編集した場合はそこに現れない。Python から書く実行サマリでは
//! こちらの値を使う。gfem-py はどのクレートを編集しても再コンパイルされるので、
//! 監視のせいでビルドが余計に走ることはない。仕組みは `build_git.rs` を参照。

include!("../gfem-core/build_git.rs");

fn main() {
    embed_git_state(&[
        "crates/gfem-core/src",
        "crates/gfem-core/build.rs",
        "crates/gfem-core/build_git.rs",
        "crates/gfem-core/Cargo.toml",
        "crates/gfem-cuda/src",
        "crates/gfem-cuda/Cargo.toml",
        "crates/gfem-py/src",
        "crates/gfem-py/build.rs",
        "crates/gfem-py/Cargo.toml",
        "Cargo.toml",
        "Cargo.lock",
    ]);
}
