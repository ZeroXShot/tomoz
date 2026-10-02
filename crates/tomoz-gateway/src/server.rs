//! The HTTP server and the compaction loop.

use std::sync::Arc;
use std::time::Duration;

use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;

use crate::s3::{Gateway, handle};
use crate::store::Store;

/// Serves `gateway` on `listener` until `shutdown` resolves, then stops
/// accepting connections and waits (up to 30 s) for requests in flight.
///
/// # Errors
///
/// I/O errors of the listener.
pub async fn serve(
    gateway: Arc<Gateway>,
    listener: TcpListener,
    shutdown: impl Future<Output = ()>,
) -> std::io::Result<()> {
    let limit = Arc::new(Semaphore::new(gateway.config.limits.max_connections));
    let graceful = hyper_util::server::graceful::GracefulShutdown::new();
    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            () = &mut shutdown => break,
            accepted = listener.accept() => {
                let (stream, peer) = match accepted {
                    Ok(a) => a,
                    Err(e) => {
                        tracing::warn!(error = %e, "accept failed");
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        continue;
                    }
                };
                let Ok(permit) = limit.clone().try_acquire_owned() else {
                    tracing::warn!(%peer, "connection limit reached, dropping connection");
                    continue;
                };
                let _ = stream.set_nodelay(true);
                let gw = gateway.clone();
                let conn = http1::Builder::new()
                    .keep_alive(true)
                    .serve_connection(TokioIo::new(stream), service_fn(move |req| handle(gw.clone(), req)));
                let watched = graceful.watch(conn);
                tokio::spawn(async move {
                    if let Err(e) = watched.await {
                        tracing::debug!(%peer, error = %e, "connection closed with an error");
                    }
                    drop(permit);
                });
            }
        }
    }
    tracing::info!("shutting down: waiting for requests in flight");
    tokio::select! {
        () = graceful.shutdown() => {}
        () = tokio::time::sleep(Duration::from_secs(30)) => tracing::warn!("requests still in flight after 30 s"),
    }
    Ok(())
}

/// Compacts quiet series forever (cancel the task to stop it).
pub async fn compactor(store: Arc<Store>, quiet: u64, min_objects: u32, interval: Duration, workers: usize) {
    let slots = Arc::new(Semaphore::new(workers));
    let running = Arc::new(std::sync::Mutex::new(std::collections::HashSet::<String>::new()));
    loop {
        tokio::time::sleep(interval).await;
        let s = store.clone();
        let due = match tokio::task::spawn_blocking(move || s.due(quiet, min_objects)).await {
            Ok(Ok(d)) => d,
            Ok(Err(e)) => {
                tracing::error!(error = %e, "listing series due for compaction failed");
                continue;
            }
            Err(_) => continue,
        };
        for series in due {
            if !running.lock().unwrap_or_else(std::sync::PoisonError::into_inner).insert(series.clone()) {
                continue;
            }
            let Ok(permit) = slots.clone().acquire_owned().await else { return };
            let (s, r) = (store.clone(), running.clone());
            tokio::spawn(async move {
                let name = series.replace('\u{0}', "/");
                let key = series.clone();
                let s2 = s.clone();
                let result = tokio::task::spawn_blocking(move || s2.compact(&key)).await;
                match result {
                    Ok(Ok(c)) if c.objects > 0 => tracing::info!(
                        series = %name,
                        objects = c.objects,
                        input_bytes = c.input_bytes,
                        archive_bytes = c.archive_bytes,
                        ratio = c.input_bytes as f64 / c.archive_bytes.max(1) as f64,
                        "compacted series"
                    ),
                    Ok(Ok(_)) => {}
                    Ok(Err(e)) => {
                        s.metrics().compaction_failures.inc();
                        tracing::error!(series = %name, error = %e, "compaction failed");
                    }
                    Err(e) => tracing::error!(series = %name, error = %e, "compaction task panicked"),
                }
                r.lock().unwrap_or_else(std::sync::PoisonError::into_inner).remove(&series);
                drop(permit);
            });
        }
    }
}
