//! S3-compatible storage gateway with transparent, byte-exact compression of
//! DICOM series.
//!
//! The gateway speaks a subset of the S3 API (buckets, objects, ranges,
//! listings, multi-object delete, multipart uploads, copy) authenticated
//! with AWS Signature Version 4, so that PACS and pipelines that store DICOM
//! instances in object storage can use it unchanged. Objects are stored as
//! received; a background compactor packs each quiet DICOM series into a
//! Tomoz archive, where every slice is predicted from its neighbours, and
//! reads restore the original bytes exactly (verified against their SHA-256
//! on every read).
//!
//! See `docs/gateway.md` for the design, the consistency argument and the
//! supported API.

pub mod cache;
pub mod config;
pub mod metrics;
pub mod s3;
pub mod server;
pub mod sigv4;
pub mod store;
pub mod xml;

use std::sync::Arc;
use std::time::Duration;

pub use config::Config;

use tomoz_codec::ModelSet;

/// Errors starting the gateway.
#[derive(Debug, thiserror::Error)]
pub enum StartError {
    /// Credentials could not be loaded.
    #[error("credentials: {0}")]
    Credentials(std::io::Error),
    /// The store could not be opened.
    #[error("store: {0}")]
    Store(#[from] store::StoreError),
    /// The listener could not be bound.
    #[error("listening on {addr}: {source}")]
    Bind {
        /// Address.
        addr: std::net::SocketAddr,
        /// Cause.
        source: std::io::Error,
    },
}

/// Runs the gateway until `shutdown` resolves.
///
/// # Errors
///
/// [`StartError`] if the gateway cannot start.
pub async fn run(config: Config, models: ModelSet, shutdown: impl Future<Output = ()>) -> Result<(), StartError> {
    let credentials = match &config.auth.credentials_file {
        Some(p) if config.auth.mode == config::AuthMode::Sigv4 => {
            Some(sigv4::Credentials::from_file(p).map_err(StartError::Credentials)?)
        }
        _ => None,
    };
    if config.auth.mode == config::AuthMode::None {
        tracing::warn!("authentication is disabled: every request is accepted");
    }
    let metrics = Arc::new(metrics::Metrics::default());
    let options = store::StoreOptions {
        dir: config.data_dir.clone(),
        zstd_level: config.compaction.zstd_level,
        slab: config.compaction.slab,
        cache_bytes: config.cache.max_bytes,
    };
    let s = Arc::new(
        tokio::task::spawn_blocking({
            let metrics = metrics.clone();
            move || store::Store::open(&options, models, metrics)
        })
        .await
        .map_err(|e| StartError::Store(store::StoreError::Io(std::io::Error::other(e))))??,
    );
    for b in &config.buckets {
        s.create_bucket(b)?;
    }
    let listener = tokio::net::TcpListener::bind(config.listen)
        .await
        .map_err(|source| StartError::Bind { addr: config.listen, source })?;
    tracing::info!(addr = %config.listen, data_dir = %config.data_dir.display(), "gateway listening");
    let compactor = config.compaction.enabled.then(|| {
        tokio::spawn(server::compactor(
            s.clone(),
            config.compaction.quiet_seconds,
            config.compaction.min_objects,
            Duration::from_secs(config.compaction.interval_seconds),
            config.compaction.workers,
        ))
    });
    let gauges = tokio::spawn({
        let s = s.clone();
        async move {
            loop {
                tokio::time::sleep(Duration::from_secs(15)).await;
                let s = s.clone();
                let _ = tokio::task::spawn_blocking(move || s.refresh_gauges()).await;
            }
        }
    });
    let gateway = Arc::new(s3::Gateway::new(s, Arc::new(config), credentials, metrics));
    let result = server::serve(gateway, listener, shutdown).await;
    if let Some(c) = compactor {
        c.abort();
    }
    gauges.abort();
    result.map_err(|source| StartError::Bind { addr: std::net::SocketAddr::from(([0, 0, 0, 0], 0)), source })
}
