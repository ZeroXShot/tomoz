# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's "Report a
vulnerability" button (Security → Advisories) on this repository, not in
public issues. Include the affected component, version or commit, and a
reproducer if you have one (a crashing input from the fuzzers is ideal). You
should get an acknowledgement within a week.

## What counts

Tomoz reads untrusted input in several places, and each of them is expected to
fail cleanly — an error, never a crash, a hang or an unbounded allocation:

| Input | Component | Guard |
|---|---|---|
| Tomoz containers (`.tmz`) | `tomoz-codec` | Header CRC, per-tile stream and sample CRCs, SHA-256 of the volume, `DecodeOptions::max_samples`, fallible allocation |
| DICOM archives (`.tmzd`) | `tomoz-archive` | Header and table CRCs, zstd sizes bounded by the frame (no trust in declared sizes), SHA-256 per restored file, `Archive::with_max_samples` |
| DICOM files | `tomoz-dicom` | Bounds-checked parser that never allocates from declared lengths |
| Model files (`.tzm`) | `tomoz-codec` | Layer shapes validated; the loader proves that no integer accumulator can overflow |
| S3 requests | `tomoz-gateway` | AWS Signature V4 (header and per-chunk), clock-skew window, body size limit, connection limit, payload hashes and checksums |

Each of these has a fuzz target in `fuzz/`, run weekly in CI.

## Deployment notes

- The gateway listens on `127.0.0.1` by default. Expose it only behind TLS
  termination (a reverse proxy or a service mesh); it speaks plain HTTP.
- Authentication is on by default (`auth.mode = "sigv4"`) and the gateway
  refuses to start without a credentials file. Keep that file readable only by
  the service user; never bake it into an image.
- `auth.mode = "none"` exists for local experiments only.
- Health and metrics (`/_tomoz/health`, `/_tomoz/metrics`) are not
  authenticated and reveal object counts and sizes, not object contents or
  keys. Restrict them at the proxy if that matters to you.
- Medical images are personal data. Tomoz compresses them losslessly and does
  not de-identify them: apply your institution's de-identification and access
  policies before and after storage.
