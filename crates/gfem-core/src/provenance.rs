//! 実行のプロビナンス記録（SPEC §5.5 / §10）。
//!
//! 研究用途では「この図はどの入力から出たか」が後で追えることが不可欠なので、
//! 出力 JSON に
//!
//! - 入力ファイルの SHA-256
//! - ケースファイルの全文
//! - Git のコミットハッシュと、作業ツリーが汚れていたかどうか
//! - ライブラリのバージョンとビルド対象
//! - 実行時刻
//!
//! を必ず記録する。**コミットハッシュだけでは足りない**——未コミットの変更が
//! あった状態で回した結果は、そのハッシュからは再現できないため、
//! `git_dirty` も併せて残す。
//!
//! # Git の状態は 2 種類ある
//!
//! - **ビルド時**（`git_commit` / `git_dirty`）: そのバイナリを作ったソースの状態。
//!   ビルドスクリプトが埋め込む（`crates/gfem-core/build_git.rs`）。何を数えたかは
//!   `git_dirty_scope` に残す
//! - **実行時**（[`SourceTree`]）: `maturin develop` の開発環境では Python のコードが
//!   作業ツリーから直接読まれるので、ビルドし直さずに編集・コミットすると
//!   ビルド時の状態とずれる。サマリを書く時点の作業ツリーも記録し、
//!   ビルド後にコミットが進んでいたら（拡張のビルドし忘れ）その旨を書き添える

use crate::error::{GfemError, Result};
use crate::io::json::Json;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// 実行環境と入力の来歴。
#[derive(Debug, Clone, Default)]
pub struct Provenance {
    /// `gfem-core` のバージョン
    pub gfem_version: String,
    /// ビルド時の Git コミットハッシュ
    pub git_commit: Option<String>,
    /// ビルド時に未コミットの変更があったか。`None` は判定できなかったことを表す
    pub git_dirty: Option<bool>,
    /// `git_dirty` の判定に数えたソース（ワークスペースのルートからの相対パス）
    pub git_dirty_scope: Vec<String>,
    /// 実行時の作業ツリー（開発環境でだけ分かる）
    pub source_tree: Option<SourceTree>,
    /// ビルド対象（ターゲットトリプル）
    pub target: String,
    /// 実行時刻（UTC、ISO 8601）
    pub timestamp_utc: String,
    /// 実行時刻（UNIX epoch 秒）
    pub unix_time: i64,
    /// 入力メッシュのパス
    pub mesh_file: Option<String>,
    /// 入力メッシュの SHA-256（16 進小文字）
    pub mesh_sha256: Option<String>,
    /// ケースファイルのパス
    pub case_file: Option<String>,
    /// ケースの全文
    pub case_text: Option<String>,
}

/// 実行時に読み込まれているソースの作業ツリー（開発環境でだけ分かる）。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceTree {
    /// リポジトリのルート
    pub path: String,
    /// その時点の HEAD
    pub git_commit: Option<String>,
    /// 結果に効くソースに未コミットの変更があるか
    pub git_dirty: Option<bool>,
}

impl Provenance {
    /// 実行環境の情報を埋めたものを作る。
    ///
    /// Git の状態は gfem-core 自身のビルド時のもの。最終バイナリを作る側
    /// （`gfem-py`）は [`with_git_state`](Self::with_git_state) で自分の値に差し替える。
    pub fn capture() -> Self {
        let unix_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        Self {
            gfem_version: crate::VERSION.to_string(),
            target: option_env!("GFEM_TARGET").unwrap_or("unknown").to_string(),
            timestamp_utc: format_utc(unix_time),
            unix_time,
            ..Self::default()
        }
        .with_git_state(
            option_env!("GFEM_GIT_COMMIT"),
            option_env!("GFEM_GIT_DIRTY"),
            option_env!("GFEM_GIT_SCOPE"),
        )
    }

    /// ビルド時の Git の状態を設定する（`build_git.rs` が埋め込む環境変数の値をそのまま渡す）。
    pub fn with_git_state(
        mut self,
        commit: Option<&str>,
        dirty: Option<&str>,
        scope: Option<&str>,
    ) -> Self {
        self.git_commit = commit.map(str::to_string);
        self.git_dirty = dirty.map(|v| v == "1");
        self.git_dirty_scope = scope
            .map(|s| s.split(';').map(str::to_string).collect())
            .unwrap_or_default();
        self
    }

    /// 実行時の作業ツリーを記録する。
    pub fn with_source_tree(mut self, tree: SourceTree) -> Self {
        self.source_tree = Some(tree);
        self
    }

    /// 入力メッシュのパスとハッシュを記録する（ここでファイルを読んでハッシュを取る）。
    ///
    /// 読み込んだときのハッシュが手元にあるなら [`with_mesh_hash`](Self::with_mesh_hash)
    /// を使うこと。読み込んだ後にファイルが書き換えられていると、ここで取るハッシュは
    /// 解いたメッシュのものではなくなる。
    pub fn with_mesh(mut self, path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        self.mesh_sha256 = Some(sha256_file(path)?);
        self.mesh_file = Some(path.display().to_string());
        Ok(self)
    }

    /// 入力メッシュのパスと、読み込んだときに取ったハッシュを記録する
    /// （[`crate::ReadReport::sha256`]）。
    pub fn with_mesh_hash(mut self, path: impl AsRef<Path>, sha256: impl Into<String>) -> Self {
        self.mesh_file = Some(path.as_ref().display().to_string());
        self.mesh_sha256 = Some(sha256.into());
        self
    }

    /// ケースファイルのパスと全文を記録する。
    pub fn with_case_file(mut self, path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| GfemError::io(path, "ケースファイルを読めません", e))?;
        self.case_file = Some(path.display().to_string());
        self.case_text = Some(text);
        Ok(self)
    }

    /// ケースの内容を文字列として記録する。
    ///
    /// **解いたときの解析条件そのもの**を渡すこと。ディスク上のケースファイルは、
    /// 読み込んだ後に（GUI やスクリプトで）条件を変えて解いた場合に食い違う。
    pub fn with_case_text(mut self, text: impl Into<String>) -> Self {
        self.case_text = Some(text.into());
        self
    }

    /// ビルド後にコミットが進んでいるか（拡張モジュールのビルドし忘れ）。
    ///
    /// どちらかのコミットが分からなければ `None`。
    pub fn source_matches_build(&self) -> Option<bool> {
        let tree = self.source_tree.as_ref()?;
        Some(tree.git_commit.as_ref()? == self.git_commit.as_ref()?)
    }

    /// JSON に変換する。
    pub fn to_json(&self) -> Json {
        let mut json = Json::object();
        json.set("gfem_version", self.gfem_version.clone())
            .set_option("git_commit", self.git_commit.clone())
            .set_option("git_dirty", self.git_dirty)
            .set("git_dirty_scope", self.git_dirty_scope.clone());
        if let Some(tree) = &self.source_tree {
            let mut t = Json::object();
            t.set("path", tree.path.clone())
                .set_option("git_commit", tree.git_commit.clone())
                .set_option("git_dirty", tree.git_dirty)
                .set_option("same_commit_as_build", self.source_matches_build());
            if self.source_matches_build() == Some(false) {
                t.set(
                    "note",
                    "拡張モジュールのビルド後にコミットが進んでいます。Rust 側は古いソースのまま\
                     動いている可能性があるので、maturin develop でビルドし直してください",
                );
            }
            json.set("source_tree", t);
        }
        json.set("target", self.target.clone())
            .set("timestamp_utc", self.timestamp_utc.clone())
            .set("unix_time", self.unix_time)
            .set_option("mesh_file", self.mesh_file.clone())
            .set_option("mesh_sha256", self.mesh_sha256.clone())
            .set_option("case_file", self.case_file.clone())
            .set_option("case_text", self.case_text.clone());
        json
    }
}

/// ファイルの SHA-256 を 16 進小文字で返す。
pub fn sha256_file(path: impl AsRef<Path>) -> Result<String> {
    use std::io::Read;
    let path = path.as_ref();
    let mut file = std::fs::File::open(path)
        .map_err(|e| GfemError::io(path, "ハッシュを取るためにファイルを開けません", e))?;
    let mut hasher = Sha256::new();
    // 1000 万節点のファイルでも一度に読み込まないよう、チャンクで流す
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let n = file
            .read(&mut buffer)
            .map_err(|e| GfemError::io(path, "ハッシュの計算中に読み取りに失敗しました", e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

/// 読んだバイト列をそのまま SHA-256 に流す [`Read`](std::io::Read)。
///
/// メッシュを読むのと同じ 1 回の読み取りでハッシュを取るために使う
/// （[`crate::read_nastran`]）。読み終えた後にファイルを読み直してハッシュを取ると、
/// その間にファイルが書き換えられていても気づけない。
pub struct HashingReader<R> {
    inner: R,
    hasher: Sha256,
}

impl<R> HashingReader<R> {
    /// `inner` から読むたびにハッシュを更新する。
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
        }
    }

    /// ここまでに読んだバイト列の SHA-256（16 進小文字）。
    pub fn finish(self) -> String {
        hex(&self.hasher.finalize())
    }
}

impl<R: std::io::Read> std::io::Read for HashingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }
}

/// バイト列の SHA-256 を 16 進小文字で返す。
pub fn sha256_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex(&hasher.finalize())
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// UNIX epoch 秒を UTC の ISO 8601 文字列にする。
///
/// 日付の変換は Howard Hinnant の `civil_from_days` を使う（うるう年・世紀の
/// 例外込みで閉じた式になっており、テーブルを持たずに済む）。
pub fn format_utc(unix_time: i64) -> String {
    let days = unix_time.div_euclid(86_400);
    let seconds = unix_time.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        seconds / 3600,
        (seconds / 60) % 60,
        seconds % 60
    )
}

/// 1970-01-01 からの日数を (年, 月, 日) に変換する。
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097); // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m as u32, d as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_the_nist_vectors() {
        // FIPS 180-4 の既知の値
        assert_eq!(
            sha256_bytes(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_bytes(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn hashes_a_file_in_chunks() {
        let mut path = std::env::temp_dir();
        path.push(format!("gfem_hash_{}.txt", std::process::id()));
        std::fs::write(&path, b"abc").unwrap();
        assert_eq!(
            sha256_file(&path).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn missing_file_is_reported() {
        let e = sha256_file("does_not_exist_12345.bdf").unwrap_err();
        assert!(e.to_string().contains("開けません"), "{e}");
    }

    #[test]
    fn formats_known_timestamps() {
        assert_eq!(format_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(format_utc(1_000_000_000), "2001-09-09T01:46:40Z");
        // うるう日
        assert_eq!(format_utc(951_782_400), "2000-02-29T00:00:00Z");
        // 世紀の例外（1900 はうるう年ではない）を跨ぐ日付
        assert_eq!(format_utc(-2_208_988_800), "1900-01-01T00:00:00Z");
    }

    #[test]
    fn capture_fills_the_environment() {
        let p = Provenance::capture();
        assert_eq!(p.gfem_version, crate::VERSION);
        assert!(p.timestamp_utc.ends_with('Z'));
        assert!(p.unix_time > 1_600_000_000, "実行時刻が取れていません");
        // ビルドスクリプトが通っていればターゲットが入る
        assert!(!p.target.is_empty());
    }

    #[test]
    fn the_embedded_git_state_matches_the_working_tree() {
        // build.rs が埋め込んだ状態が、いまの作業ツリーと一致すること。
        // 以前はソースを編集して再ビルドしても build.rs が走り直さず、
        // 編集済みのソースから作ったバイナリが git_dirty = false を名乗っていた
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let git = |args: &[&str]| -> Option<String> {
            let output = std::process::Command::new("git")
                .args(args)
                .current_dir(&root)
                .output()
                .ok()?;
            output
                .status
                .success()
                .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
        };
        let Some(head) = git(&["rev-parse", "HEAD"]) else {
            return; // git もリポジトリも無い環境（配布物からのビルドなど）
        };
        let p = Provenance::capture();
        assert_eq!(
            p.git_commit.as_deref(),
            Some(head.as_str()),
            "コミットが古い"
        );

        assert!(
            p.git_dirty_scope
                .iter()
                .any(|s| s == "crates/gfem-core/src")
        );
        let mut args = vec!["status", "--porcelain", "--"];
        args.extend(p.git_dirty_scope.iter().map(String::as_str));
        let status = git(&args).expect("git status");
        assert_eq!(
            p.git_dirty,
            Some(!status.is_empty()),
            "埋め込まれた git_dirty が作業ツリーと食い違う:\n{status}"
        );
    }

    #[test]
    fn a_stale_extension_is_called_out() {
        let tree = SourceTree {
            path: "/repo".to_string(),
            git_commit: Some("bbbb".to_string()),
            git_dirty: Some(false),
        };
        let stale = Provenance::capture()
            .with_git_state(Some("aaaa"), Some("0"), Some("crates"))
            .with_source_tree(tree.clone());
        assert_eq!(stale.source_matches_build(), Some(false));
        let text = stale.to_json().to_pretty_string();
        assert!(text.contains("\"same_commit_as_build\": false"), "{text}");
        assert!(text.contains("maturin develop"), "{text}");

        let fresh = Provenance::capture()
            .with_git_state(Some("bbbb"), Some("0"), Some("crates"))
            .with_source_tree(tree);
        assert_eq!(fresh.source_matches_build(), Some(true));
        assert!(
            !fresh
                .to_json()
                .to_pretty_string()
                .contains("maturin develop")
        );
    }

    #[test]
    fn hashing_reader_hashes_what_it_reads() {
        use std::io::Read;
        let mut reader = HashingReader::new(&b"abc"[..]);
        let mut text = String::new();
        reader.read_to_string(&mut text).unwrap();
        assert_eq!(text, "abc");
        assert_eq!(reader.finish(), sha256_bytes(b"abc"));
    }

    #[test]
    fn json_contains_the_case_text() {
        let p = Provenance::capture().with_case_text("version: 1\nmesh:\n  file: a.bdf\n");
        let text = p.to_json().to_pretty_string();
        assert!(text.contains("\"case_text\""));
        assert!(text.contains("version: 1"));
        assert!(text.contains("\\n"), "改行がエスケープされていません");
    }
}
