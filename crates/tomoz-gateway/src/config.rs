//! Gateway configuration.
//!
//! Settings come from built-in defaults, an optional TOML file and
//! environment variables of the form `TOMOZ__SECTION__KEY`, in that order.
//! Unknown keys are rejected. Secrets are never part of the configuration:
//! credentials are read from the file named by `auth.credentials_file`.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Complete gateway configuration.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct Config {
    /// Address of the S3 endpoint.
    pub listen: SocketAddr,
    /// Directory holding the index, raw objects and archives.
    pub data_dir: PathBuf,
    /// Region reported to clients and used in request signatures.
    pub region: String,
    /// Buckets created at startup if they do not exist.
    pub buckets: Vec<String>,
    /// Authentication.
    pub auth: AuthConfig,
    /// Background compaction of DICOM series.
    pub compaction: CompactionConfig,
    /// Cache of decoded archive slices.
    pub cache: CacheConfig,
    /// Request limits.
    pub limits: LimitsConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen: SocketAddr::from(([127, 0, 0, 1], 9100)),
            data_dir: PathBuf::from("tomoz-data"),
            region: "us-east-1".into(),
            buckets: Vec::new(),
            auth: AuthConfig::default(),
            compaction: CompactionConfig::default(),
            cache: CacheConfig::default(),
            limits: LimitsConfig::default(),
        }
    }
}

/// How requests are authenticated.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct AuthConfig {
    /// `sigv4` (AWS Signature Version 4) or `none` (any request is accepted;
    /// only for trusted networks).
    pub mode: AuthMode,
    /// File with one `ACCESS_KEY_ID:SECRET_ACCESS_KEY` per line.
    pub credentials_file: Option<PathBuf>,
    /// Largest accepted difference between the request date and the clock.
    pub max_clock_skew_seconds: u64,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self { mode: AuthMode::Sigv4, credentials_file: None, max_clock_skew_seconds: 900 }
    }
}

/// Authentication mode.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AuthMode {
    /// AWS Signature Version 4.
    Sigv4,
    /// No authentication.
    None,
}

/// When and how series are compacted into archives.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct CompactionConfig {
    /// Whether compaction runs.
    pub enabled: bool,
    /// A series is compacted once it received no object for this long.
    pub quiet_seconds: u64,
    /// Smallest number of objects worth compacting.
    pub min_objects: u32,
    /// How often the compactor looks for work.
    pub interval_seconds: u64,
    /// Series compacted at the same time.
    pub workers: usize,
    /// zstd level of archive metadata.
    pub zstd_level: i32,
    /// Slices per tile in archives (smaller tiles make single-object reads
    /// cheaper; larger ones compress slightly better).
    pub slab: u16,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            quiet_seconds: 120,
            min_objects: 4,
            interval_seconds: 10,
            workers: 1,
            zstd_level: 19,
            slab: 16,
        }
    }
}

/// Cache of decoded data.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct CacheConfig {
    /// Memory budget for decoded slabs and archive metadata.
    pub max_bytes: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self { max_bytes: 512 << 20 }
    }
}

/// Request limits.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct LimitsConfig {
    /// Largest object accepted by a single PUT or part upload.
    pub max_object_bytes: u64,
    /// Largest number of keys returned by one listing.
    pub max_keys: u32,
    /// Concurrent connections.
    pub max_connections: usize,
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self { max_object_bytes: 4 << 30, max_keys: 1000, max_connections: 1024 }
    }
}

/// Errors while loading the configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The file could not be read.
    #[error("reading {path}: {source}")]
    Io {
        /// File.
        path: PathBuf,
        /// Cause.
        source: std::io::Error,
    },
    /// The configuration is invalid.
    #[error("invalid configuration: {0}")]
    Invalid(String),
}

impl Config {
    /// Defaults, then `file`, then `TOMOZ__*` variables from `env`.
    ///
    /// # Errors
    ///
    /// [`ConfigError`] for unreadable files, syntax errors, unknown keys and
    /// invalid values.
    pub fn load(file: Option<&Path>, env: impl IntoIterator<Item = (String, String)>) -> Result<Self, ConfigError> {
        let mut table = toml::Table::try_from(Self::default()).map_err(|e| ConfigError::Invalid(e.to_string()))?;
        if let Some(path) = file {
            let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io { path: path.into(), source })?;
            let overlay: toml::Table =
                text.parse().map_err(|e| ConfigError::Invalid(format!("{}: {e}", path.display())))?;
            merge(&mut table, overlay);
        }
        for (name, value) in env {
            let Some(rest) = name.strip_prefix("TOMOZ__") else { continue };
            let path: Vec<String> = rest.split("__").map(str::to_lowercase).collect();
            set_path(&mut table, &path, parse_env_value(&value))?;
        }
        let config: Self = table.try_into().map_err(|e: toml::de::Error| ConfigError::Invalid(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        let bad = |m: &str| Err(ConfigError::Invalid(m.to_owned()));
        if self.auth.mode == AuthMode::Sigv4 && self.auth.credentials_file.is_none() {
            return bad("auth.mode = \"sigv4\" needs auth.credentials_file");
        }
        if self.compaction.workers == 0 || self.compaction.slab == 0 || self.compaction.interval_seconds == 0 {
            return bad("compaction.workers, compaction.slab and compaction.interval_seconds must be positive");
        }
        if !(1..=22).contains(&self.compaction.zstd_level) {
            return bad("compaction.zstd_level must be within 1..=22");
        }
        if self.limits.max_keys == 0 || self.limits.max_connections == 0 {
            return bad("limits.max_keys and limits.max_connections must be positive");
        }
        for b in &self.buckets {
            if !crate::store::valid_bucket(b) {
                return bad(&format!("invalid bucket name {b:?}"));
            }
        }
        Ok(())
    }

    /// The configuration as TOML (secrets are never part of it).
    #[must_use]
    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).unwrap_or_default()
    }
}

fn merge(base: &mut toml::Table, overlay: toml::Table) {
    for (k, v) in overlay {
        match (base.get_mut(&k), v) {
            (Some(toml::Value::Table(b)), toml::Value::Table(o)) => merge(b, o),
            (_, v) => {
                base.insert(k, v);
            }
        }
    }
}

fn set_path(table: &mut toml::Table, path: &[String], value: toml::Value) -> Result<(), ConfigError> {
    match path {
        [] => Err(ConfigError::Invalid("empty environment override".into())),
        [last] => {
            table.insert(last.clone(), value);
            Ok(())
        }
        [head, rest @ ..] => {
            match table.entry(head.clone()).or_insert_with(|| toml::Value::Table(toml::Table::new())) {
                toml::Value::Table(t) => set_path(t, rest, value),
                _ => Err(ConfigError::Invalid(format!("{head} is not a section"))),
            }
        }
    }
}

/// Environment values are TOML values when they parse as such (numbers,
/// booleans, arrays), plain strings otherwise.
fn parse_env_value(raw: &str) -> toml::Value {
    format!("v = {raw}")
        .parse::<toml::Table>()
        .ok()
        .and_then(|mut t| t.remove("v"))
        .unwrap_or_else(|| toml::Value::String(raw.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layers_override_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("gw.toml");
        std::fs::write(&file, "data_dir = \"/srv/tomoz\"\n[compaction]\nquiet_seconds = 30\n[auth]\nmode = \"none\"\n")
            .unwrap();
        let env = [
            ("TOMOZ__COMPACTION__QUIET_SECONDS".to_owned(), "5".to_owned()),
            ("TOMOZ__REGION".to_owned(), "eu-west-1".to_owned()),
        ];
        let c = Config::load(Some(&file), env).unwrap();
        assert_eq!(c.data_dir, PathBuf::from("/srv/tomoz"));
        assert_eq!(c.compaction.quiet_seconds, 5);
        assert_eq!(c.region, "eu-west-1");
        assert_eq!(c.auth.mode, AuthMode::None);
    }

    #[test]
    fn unknown_keys_and_missing_credentials_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("gw.toml");
        std::fs::write(&file, "[auth]\nmode = \"none\"\n[cache]\nmax_byte = 1\n").unwrap();
        assert!(Config::load(Some(&file), []).is_err());
        assert!(Config::load(None, []).is_err(), "sigv4 without credentials");
        assert!(Config::load(None, [("TOMOZ__AUTH__MODE".to_owned(), "none".to_owned())]).is_ok());
    }
}
