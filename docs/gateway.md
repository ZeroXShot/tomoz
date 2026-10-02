# The S3 gateway

`tomoz serve` runs an S3-compatible object store for DICOM. Clients write
DICOM files as objects, exactly as they would to any S3 service; the gateway
stores them as they arrive and later, in the background, packs each quiet
series into a Tomoz archive. Reads always return the original bytes.

It is meant to sit behind a PACS or a research pipeline that can already
store DICOM in S3 — for example Orthanc with its S3 storage plugin, or scripts
using boto3, the AWS CLI or rclone — and to make that storage several times
smaller without changing the client.

## Contents

- [S3 compatibility](#s3-compatibility)
- [Storage design](#storage-design)
- [Compaction](#compaction)
- [Consistency and crash safety](#consistency-and-crash-safety)
- [Reads and caching](#reads-and-caching)
- [Configuration](#configuration)
- [Operations](#operations)
- [Limitations](#limitations)

## S3 compatibility

Requests use path-style addressing (`http://host:9100/bucket/key`) and AWS
Signature Version 4 in the `Authorization` header. Supported:

| Area | Operations and features |
|---|---|
| Service | ListBuckets |
| Buckets | CreateBucket, HeadBucket, DeleteBucket (empty only), GetBucketLocation |
| Listing | ListObjects (v1, marker) and ListObjectsV2 (continuation token, `start-after`), `prefix`, `delimiter` with common prefixes, `max-keys`, `encoding-type=url` |
| Objects | PutObject, GetObject (with `Range: bytes=`), HeadObject, DeleteObject, DeleteObjects (quiet or verbose), CopyObject |
| Multipart | CreateMultipartUpload, UploadPart, CompleteMultipartUpload, AbortMultipartUpload (ETag `"md5-of-md5s-N"`) |
| Payloads | Signed (`x-amz-content-sha256` = hash), `UNSIGNED-PAYLOAD`, `aws-chunked` with per-chunk signatures, unsigned `aws-chunked` with trailers; `Content-MD5`; `x-amz-checksum-crc32`, `-crc32c` and `-sha256` as headers or trailers |
| Errors | S3 XML errors with the usual codes (`NoSuchKey`, `SignatureDoesNotMatch`, `RequestTimeTooSkewed`, `InvalidRange`, `EntityTooLarge`, …) and `x-amz-request-id` |

Not supported: virtual-hosted addressing, presigned URLs, versioning, ACLs and
bucket policies, object tags, lifecycle rules, server-side encryption,
conditional requests, ListParts and ListMultipartUploads. Unsupported
operations answer `501 NotImplemented`.

Compatibility is tested with boto3 against a running gateway in CI
(`lab/tests/gateway_compat.py`), and the signature code against the examples
of the AWS documentation (unit tests in `sigv4.rs`).

## Storage design

Everything lives under `data_dir`:

```text
index.sqlite        the index: the single source of truth (SQLite, WAL)
raw/ab/<id>         objects as uploaded
archives/ab/<id>.tmzd   Tomoz DICOM archives
parts/ab/<id>       parts of multipart uploads in progress
tmp/                files being written
```

The index maps every `(bucket, key)` to exactly one of a raw file or an
instance of an archive (a `CHECK` constraint enforces it), with size, ETag
(MD5, as S3 clients expect), content type, modification time and the SHA-256
of the object. Archives carry a count of the objects that still refer to
them.

When an object arrives, the gateway parses its DICOM header. If the object is
a DICOM instance whose pixel data Tomoz can code (native grayscale 8- or
16-bit), it is recorded under its series (bucket + Series Instance UID) and
the series' "last write" time is updated. Anything else — non-DICOM objects,
compressed transfer syntaxes, colour images — is just stored.

## Compaction

A background loop (every `compaction.interval_seconds`) looks for series that
have received no write for `compaction.quiet_seconds` and have at least
`compaction.min_objects` raw objects, oldest first, and hands them to
`compaction.workers` threads. For each series:

1. read its raw objects and check each against its SHA-256;
2. pack them into an archive ([format](format.md#3-dicom-archive-tmzd)):
   instances sorted into stacks by geometry and position along the slice
   normal, pixel data coded with Tomoz, all other bytes zstd-compressed
   together;
3. open the archive again and restore every object, checking each against
   its SHA-256 — nothing is switched unless the round trip is exact;
4. write the archive atomically (temporary file, `fsync`, rename, `fsync` of
   the directory);
5. in one transaction, move each object from its raw file to the archive
   *only if it still points to the raw file that was read*
   (`UPDATE … WHERE blob = <old>`), and record the archive with the number of
   objects that moved;
6. delete the raw files of the objects that moved.

An object overwritten or deleted while its series was being compacted keeps
its new state; if no object moved, the archive is discarded. When the last
object of an archive is overwritten or deleted, the archive is deleted.

Compressed sizes depend on the data; the evaluation reports Tomoz on real
series ([evaluation](evaluation.md)). Compacted objects answer reads with the
header `x-tomoz-storage: archive`.

## Consistency and crash safety

- **Durability.** A PUT is acknowledged only after the object file and the
  directory entry are synced and the index transaction is committed.
- **Atomicity.** Files are written under a temporary name and renamed into
  place before the index refers to them; they are deleted only after the
  index stops referring to them. A crash between those steps leaves
  unreferenced files, which the gateway deletes at startup (and logs).
- **Isolation.** Each write is one SQLite transaction. Readers look the key
  up, then read the file; if a concurrent compaction, overwrite or delete has
  removed that file in between, the read looks the key up again and returns
  the current version (or `NoSuchKey`). A reader never sees a mix of two
  versions, and never an internal error caused by a concurrent writer — a
  stress test races readers against overwrites and compaction.
- **Integrity.** Every read of a raw object checks its SHA-256; every
  restored archive instance checks its SHA-256 (and the codec its tile and
  volume checksums). Corruption is reported as `500 InternalError` and
  logged, never served.
- **Graceful shutdown.** On SIGTERM or Ctrl-C the gateway stops accepting
  connections, lets requests in flight finish and waits for the compaction in
  progress, then exits. A compaction killed midway (power loss, SIGKILL)
  changes nothing: its archive is not referenced and is removed at the next
  start.

## Reads and caching

A read of a compacted object decodes only the tiles of the slab that holds
its slices. An LRU cache bounded in bytes (`cache.max_bytes`) keeps opened
archives (their decompressed metadata) and decoded slabs, so reading a whole
series decodes each slab once: the first instance of a slab pays for it, the
others are served from memory.

## Configuration

Settings come from defaults, then a TOML file (`--config`), then environment
variables `TOMOZ__<SECTION>__<KEY>` (for example
`TOMOZ__COMPACTION__QUIET_SECONDS=300`). `tomoz serve --print-config` prints
the effective configuration.

```toml
listen = "127.0.0.1:9100"   # loopback by default
data_dir = "tomoz-data"
region = "us-east-1"        # the region clients sign for
buckets = ["dicom"]         # created at startup if missing

[auth]
mode = "sigv4"              # "none" only for local experiments
credentials_file = "/run/secrets/tomoz-credentials"
max_clock_skew_seconds = 900

[compaction]
enabled = true
quiet_seconds = 120         # a series must be idle this long
min_objects = 4
interval_seconds = 10
workers = 1
zstd_level = 19             # headers and stored files
slab = 16                   # slices per tile: random-access granularity

[cache]
max_bytes = 536870912       # 512 MiB

[limits]
max_object_bytes = 4294967296
max_keys = 1000
max_connections = 1024
```

The credentials file holds one `ACCESS_KEY_ID:SECRET` per line (`#` starts a
comment). Generate secrets randomly, e.g. `openssl rand -hex 32`, and keep the
file readable only by the service.

## Operations

- **Health:** `GET /_tomoz/health` answers `ok`.
- **Metrics:** `GET /_tomoz/metrics` in the Prometheus text format:

  | Metric | Type | Meaning |
  |---|---|---|
  | `tomoz_requests_total{operation,status}` | counter | Requests by operation and status class |
  | `tomoz_request_duration_seconds{operation}` | histogram | Latency |
  | `tomoz_objects_put_total`, `tomoz_object_bytes_in_total`, `tomoz_object_bytes_out_total` | counter | Traffic |
  | `tomoz_auth_failures_total` | counter | Rejected signatures |
  | `tomoz_compactions_total`, `tomoz_compaction_failures_total` | counter | Compactions |
  | `tomoz_compaction_input_bytes_total`, `tomoz_compaction_output_bytes_total` | counter | Bytes before and after compaction |
  | `tomoz_cache_hits_total`, `tomoz_cache_misses_total`, `tomoz_cache_bytes` | counter, gauge | Cache |
  | `tomoz_raw_objects`, `tomoz_raw_object_bytes` | gauge | Objects waiting as uploaded |
  | `tomoz_archived_objects`, `tomoz_archived_object_bytes` | gauge | Objects in archives (original bytes) |
  | `tomoz_archives`, `tomoz_archive_bytes` | gauge | Archive files |
  | `tomoz_pending_series` | gauge | Series waiting for compaction |

  The storage saving is
  `1 − tomoz_archive_bytes / tomoz_archived_object_bytes`.
- **Logs:** to stderr, `info` by default for `serve`; `tomoz --log debug serve`
  (any `tracing` filter, e.g. `--log tomoz_gateway=debug,info`) for more.
- **Backups:** stop the gateway (or use SQLite's online backup of
  `index.sqlite` followed by the files) and copy `data_dir`. The index must be
  restored together with the files it refers to.
- **Container:** see the [Dockerfile](../Dockerfile) and the
  [Orthanc example](../examples/orthanc/README.md).

## Limitations

- One node: the index is a local SQLite database, so there is no clustering.
  For availability, keep `data_dir` on replicated storage and restart the
  gateway elsewhere on failure.
- A compaction holds the raw objects of one series in memory (plus the
  archive), so very large series need correspondingly large memory.
- An archive whose objects are mostly overwritten keeps its file until the
  last of them goes; there is no re-compaction of sparse archives yet.
- Listing is served from the index and scales with it; the S3 limit of 1000
  keys per page applies.
