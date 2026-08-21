//! ビルド時に Git の状態とターゲットを埋め込む（SPEC §5.5 のプロビナンス記録）。
//!
//! git が無い環境やリポジトリ外でのビルドでも失敗しないよう、取れなければ
//! 環境変数を設定しないだけにする（実行時は `None` になる）。

use std::process::Command;

fn main() {
    println!(
        "cargo:rustc-env=GFEM_TARGET={}",
        std::env::var("TARGET").unwrap_or_else(|_| "unknown".into())
    );

    // HEAD が動いたら埋め込み直す
    for path in ["../../.git/HEAD", "../../.git/index"] {
        if std::path::Path::new(path).exists() {
            println!("cargo:rerun-if-changed={path}");
        }
    }

    if let Some(commit) = git(&["rev-parse", "HEAD"]) {
        println!("cargo:rustc-env=GFEM_GIT_COMMIT={commit}");
    }
    // 未コミットの変更があると、コミットハッシュだけでは結果を再現できない。
    // 「汚れていたか」も残しておく
    if let Some(status) = git(&["status", "--porcelain"]) {
        let dirty = if status.is_empty() { "0" } else { "1" };
        println!("cargo:rustc-env=GFEM_GIT_DIRTY={dirty}");
    }
}

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}
