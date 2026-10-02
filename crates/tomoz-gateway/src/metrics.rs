//! Prometheus metrics, served at `/_tomoz/metrics`.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};

/// A monotonic counter.
#[derive(Default)]
pub struct Counter(AtomicU64);

impl Counter {
    /// Adds one.
    pub fn inc(&self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    /// Adds `n`.
    pub fn add(&self, n: u64) {
        self.0.fetch_add(n, Ordering::Relaxed);
    }

    /// Current value.
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

/// S3 operations, as metric labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    /// PutObject.
    Put,
    /// GetObject.
    Get,
    /// HeadObject.
    Head,
    /// DeleteObject(s).
    Delete,
    /// ListObjectsV2 and ListBuckets.
    List,
    /// Multipart upload operations.
    Multipart,
    /// Bucket operations.
    Bucket,
    /// Anything else.
    Other,
}

const OPERATIONS: [(&str, Operation); 8] = [
    ("put", Operation::Put),
    ("get", Operation::Get),
    ("head", Operation::Head),
    ("delete", Operation::Delete),
    ("list", Operation::List),
    ("multipart", Operation::Multipart),
    ("bucket", Operation::Bucket),
    ("other", Operation::Other),
];

/// Upper bounds of the latency histogram buckets, in seconds.
const BUCKETS: [f64; 10] = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1.0, 5.0];

#[derive(Default)]
struct Histogram {
    buckets: [AtomicU64; BUCKETS.len()],
    count: AtomicU64,
    sum_micros: AtomicU64,
}

/// Every metric of the gateway.
#[derive(Default)]
pub struct Metrics {
    requests: [[Counter; 4]; 8],
    latency: [Histogram; 8],
    /// Objects stored.
    pub objects_put: Counter,
    /// Bytes received in object bodies.
    pub bytes_in: Counter,
    /// Bytes sent in object bodies.
    pub bytes_out: Counter,
    /// Requests rejected by authentication.
    pub auth_failures: Counter,
    /// Compactions completed.
    pub compactions: Counter,
    /// Compactions that failed.
    pub compaction_failures: Counter,
    /// Bytes of objects compacted.
    pub compaction_input_bytes: Counter,
    /// Bytes of archives written.
    pub compaction_output_bytes: Counter,
    /// Cache hits.
    pub cache_hits: Counter,
    /// Cache misses.
    pub cache_misses: Counter,
    store: [AtomicU64; 7],
}

fn op_index(op: Operation) -> usize {
    OPERATIONS.iter().position(|(_, o)| *o == op).unwrap_or(7)
}

impl Metrics {
    /// Records a finished request.
    pub fn request(&self, op: Operation, status: u16, seconds: f64) {
        let i = op_index(op);
        let class = usize::from((status / 100).clamp(2, 5) - 2);
        self.requests[i][class].inc();
        let h = &self.latency[i];
        if let Some(b) = BUCKETS.iter().position(|&le| seconds <= le) {
            h.buckets[b].fetch_add(1, Ordering::Relaxed);
        }
        h.count.fetch_add(1, Ordering::Relaxed);
        h.sum_micros.fetch_add((seconds * 1e6) as u64, Ordering::Relaxed);
    }

    /// Updates the store gauges.
    #[allow(clippy::too_many_arguments)]
    pub fn set_store(
        &self,
        raw: u64,
        raw_bytes: u64,
        archived: u64,
        archived_bytes: u64,
        archives: u64,
        archive_bytes: u64,
        pending_series: u64,
    ) {
        for (slot, v) in
            self.store.iter().zip([raw, raw_bytes, archived, archived_bytes, archives, archive_bytes, pending_series])
        {
            slot.store(v, Ordering::Relaxed);
        }
    }

    /// The metrics in the Prometheus text format.
    #[must_use]
    pub fn render(&self, cache_bytes: u64) -> String {
        let mut o = String::new();
        let _ = writeln!(o, "# HELP tomoz_requests_total S3 requests by operation and status class.");
        let _ = writeln!(o, "# TYPE tomoz_requests_total counter");
        for (i, (name, _)) in OPERATIONS.iter().enumerate() {
            for (c, class) in ["2xx", "3xx", "4xx", "5xx"].iter().enumerate() {
                let _ = writeln!(
                    o,
                    "tomoz_requests_total{{operation=\"{name}\",status=\"{class}\"}} {}",
                    self.requests[i][c].get()
                );
            }
        }
        let _ = writeln!(o, "# HELP tomoz_request_duration_seconds Request latency.");
        let _ = writeln!(o, "# TYPE tomoz_request_duration_seconds histogram");
        for (i, (name, _)) in OPERATIONS.iter().enumerate() {
            let h = &self.latency[i];
            let mut cumulative = 0;
            for (b, le) in BUCKETS.iter().enumerate() {
                cumulative += h.buckets[b].load(Ordering::Relaxed);
                let _ = writeln!(
                    o,
                    "tomoz_request_duration_seconds_bucket{{operation=\"{name}\",le=\"{le}\"}} {cumulative}"
                );
            }
            let count = h.count.load(Ordering::Relaxed);
            let _ = writeln!(o, "tomoz_request_duration_seconds_bucket{{operation=\"{name}\",le=\"+Inf\"}} {count}");
            let _ = writeln!(
                o,
                "tomoz_request_duration_seconds_sum{{operation=\"{name}\"}} {}",
                h.sum_micros.load(Ordering::Relaxed) as f64 / 1e6
            );
            let _ = writeln!(o, "tomoz_request_duration_seconds_count{{operation=\"{name}\"}} {count}");
        }
        let counters = [
            ("tomoz_objects_put_total", "Objects stored.", &self.objects_put),
            ("tomoz_object_bytes_in_total", "Bytes received in object bodies.", &self.bytes_in),
            ("tomoz_object_bytes_out_total", "Bytes sent in object bodies.", &self.bytes_out),
            ("tomoz_auth_failures_total", "Requests rejected by authentication.", &self.auth_failures),
            ("tomoz_compactions_total", "Series compacted into archives.", &self.compactions),
            ("tomoz_compaction_failures_total", "Compactions that failed.", &self.compaction_failures),
            ("tomoz_compaction_input_bytes_total", "Bytes of objects compacted.", &self.compaction_input_bytes),
            ("tomoz_compaction_output_bytes_total", "Bytes of archives written.", &self.compaction_output_bytes),
            ("tomoz_cache_hits_total", "Cache hits.", &self.cache_hits),
            ("tomoz_cache_misses_total", "Cache misses.", &self.cache_misses),
        ];
        for (name, help, c) in counters {
            let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} counter\n{name} {}", c.get());
        }
        let gauges = [
            ("tomoz_raw_objects", "Objects stored raw."),
            ("tomoz_raw_object_bytes", "Bytes of objects stored raw."),
            ("tomoz_archived_objects", "Objects held in archives."),
            ("tomoz_archived_object_bytes", "Original bytes of objects held in archives."),
            ("tomoz_archives", "Archive files."),
            ("tomoz_archive_bytes", "Bytes of archive files."),
            ("tomoz_pending_series", "Series with raw objects waiting for compaction."),
        ];
        for ((name, help), v) in gauges.iter().zip(&self.store) {
            let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {}", v.load(Ordering::Relaxed));
        }
        let _ = writeln!(
            o,
            "# HELP tomoz_cache_bytes Bytes held by the cache.\n# TYPE tomoz_cache_bytes gauge\ntomoz_cache_bytes {cache_bytes}"
        );
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_counters_and_histograms() {
        let m = Metrics::default();
        m.request(Operation::Get, 200, 0.003);
        m.request(Operation::Get, 404, 2.0);
        m.objects_put.inc();
        let text = m.render(5);
        assert!(text.contains("tomoz_requests_total{operation=\"get\",status=\"2xx\"} 1"));
        assert!(text.contains("tomoz_requests_total{operation=\"get\",status=\"4xx\"} 1"));
        assert!(text.contains("tomoz_request_duration_seconds_bucket{operation=\"get\",le=\"0.005\"} 1"));
        assert!(text.contains("tomoz_request_duration_seconds_count{operation=\"get\"} 2"));
        assert!(text.contains("tomoz_objects_put_total 1"));
        assert!(text.contains("tomoz_cache_bytes 5"));
    }
}
