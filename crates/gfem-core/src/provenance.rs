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
    /// ケースファイルの全文
    pub case_text: Option<String>,
}

impl Provenance {
    /// 実行環境の情報を埋めたものを作る。
    pub fn capture() -> Self {
        let unix_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        Self {
            gfem_version: crate::VERSION.to_string(),
            git_commit: option_env!("GFEM_GIT_COMMIT").map(str::to_string),
            git_dirty: option_env!("GFEM_GIT_DIRTY").map(|v| v == "1"),
            target: option_env!("GFEM_TARGET").unwrap_or("unknown").to_string(),
            timestamp_utc: format_utc(unix_time),
            unix_time,
            mesh_file: None,
            mesh_sha256: None,
            case_file: None,
            case_text: None,
        }
    }

    /// 入力メッシュのパスとハッシュを記録する。
    pub fn with_mesh(mut self, path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        self.mesh_sha256 = Some(sha256_file(path)?);
        self.mesh_file = Some(path.display().to_string());
        Ok(self)
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

    /// ケースの内容を文字列として記録する（ファイルが無い場合に使う）。
    pub fn with_case_text(mut self, text: impl Into<String>) -> Self {
        self.case_text = Some(text.into());
        self
    }

    /// JSON に変換する。
    pub fn to_json(&self) -> Json {
        let mut json = Json::object();
        json.set("gfem_version", self.gfem_version.clone())
            .set_option("git_commit", self.git_commit.clone())
            .set_option("git_dirty", self.git_dirty)
            .set("target", self.target.clone())
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
    fn json_contains_the_case_text() {
        let p = Provenance::capture().with_case_text("version: 1\nmesh:\n  file: a.bdf\n");
        let text = p.to_json().to_pretty_string();
        assert!(text.contains("\"case_text\""));
        assert!(text.contains("version: 1"));
        assert!(text.contains("\\n"), "改行がエスケープされていません");
    }
}
