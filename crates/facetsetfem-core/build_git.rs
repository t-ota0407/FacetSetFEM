// ビルド時に Git の状態を埋め込む処理（SPEC §5.5 のプロビナンス記録）。
//
// `crates/facetsetfem-core/build.rs` と `crates/facetsetfem-py/build.rs` の両方から
// `include!` で取り込む。クレートごとに「どのソースの状態を記録するか」だけが違う。
//
// # 何を「汚れ」とみなし、いつ埋め込み直すか
//
// 数えるのは**そのバイナリを作ったソース**（呼び出し側が渡す `sources`）に限る。
// SPEC.md や tests/ の編集は結果に効かないので数えない。
//
// `cargo:rerun-if-changed` を 1 つでも出すと、Cargo はそこに挙げたものが変わった
// ときにしかビルドスクリプトを走らせ直さない。以前は `.git/HEAD` と `.git/index`
// だけを挙げていたので、
//
// - ソースを編集して再ビルドしても走り直さず、`git_dirty = false` のまま埋め込まれた
// - ブランチ上でコミットしても `.git/HEAD`（`ref: refs/heads/main` と書いてあるだけ）は
//   変わらないので、コミットハッシュが古いままになりえた
//
// いまは `sources` そのものと、HEAD が指すブランチの参照ファイルも監視する。
//
// git が無い環境やリポジトリ外でのビルドでも失敗しないよう、取れなければ
// 環境変数を設定しないだけにする（実行時は `None` になる）。

/// `sources`（ワークスペースのルートからの相対パス）の Git の状態を埋め込む。
///
/// 埋め込む環境変数:
///
/// - `FACETSETFEM_GIT_COMMIT`: HEAD のコミットハッシュ
/// - `FACETSETFEM_GIT_DIRTY`: `sources` に未コミットの変更があれば `1`
/// - `FACETSETFEM_GIT_SCOPE`: `sources` を `;` でつないだもの（何を数えたかの記録）
fn embed_git_state(sources: &[&str]) {
    let root = std::path::Path::new("../..");
    println!("cargo:rustc-env=FACETSETFEM_GIT_SCOPE={}", sources.join(";"));

    // ソースが変わったら埋め込み直す（ディレクトリは中身ごと監視される）
    for source in sources {
        watch(&root.join(source));
    }
    // HEAD が動いたら（コミット・チェックアウト）埋め込み直す
    watch_git_refs();

    if let Some(commit) = git_in(root, &["rev-parse", "HEAD"]) {
        println!("cargo:rustc-env=FACETSETFEM_GIT_COMMIT={commit}");
    }
    // 未コミットの変更があると、コミットハッシュだけでは結果を再現できない。
    // 「汚れていたか」も残しておく
    let mut status = vec!["status", "--porcelain", "--"];
    status.extend(sources);
    if let Some(output) = git_in(root, &status) {
        let dirty = if output.is_empty() { "0" } else { "1" };
        println!("cargo:rustc-env=FACETSETFEM_GIT_DIRTY={dirty}");
    }
}

/// HEAD と、HEAD が指すブランチの参照を監視する。
///
/// 場所は `git rev-parse --git-path` に聞く（worktree でも正しい場所が返る）。
/// ブランチの参照は `git gc` で packed-refs に移ることがあるので、そちらも見る。
fn watch_git_refs() {
    let here = std::path::Path::new(".");
    let mut names = vec![
        "HEAD".to_string(),
        "index".to_string(),
        "packed-refs".to_string(),
    ];
    if let Some(branch) = git_in(here, &["symbolic-ref", "-q", "HEAD"]) {
        names.push(branch);
    }
    for name in names {
        if let Some(path) = git_in(here, &["rev-parse", "--git-path", &name]) {
            watch(std::path::Path::new(&path));
        }
    }
}

/// 存在するパスだけを監視対象にする。
///
/// 存在しないパスを挙げると Cargo は毎回「変わった」とみなし、ビルドのたびに
/// 走り直してしまう。
fn watch(path: &std::path::Path) {
    if path.exists() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn git_in(dir: &std::path::Path, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}
