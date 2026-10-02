# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/). The container, archive and
model formats are versioned separately (see [docs/format.md](docs/format.md)).

## [Unreleased]

## [0.1.1] - 2026-10-02

### Changed

- The JavaScript package is published as `@zeroxshot/tomoz` on npm.
- Dependencies updated: `lru` 0.18 (fixes RUSTSEC-2026-0253), `rusqlite`
  0.38, `zstd` 0.14, `toml`, and the GitHub Actions used by CI.

### Added

- Releases publish to crates.io, PyPI and npm automatically
  (`scripts/release.sh`); Dependabot proposes weekly updates.

## [0.1.0] - 2026-10-02

### Added

- Volume codec: integer TZ1 predictor (12/30-48-48-6 MLP) with a linear head
  over the left neighbours, adaptive range coding, flat-region flags,
  histogram packing, independent tiles for parallelism and random access,
  container format version 1 with CRC-32C per tile and SHA-256 per volume.
- SIMD kernels for AArch64 (NEON, dot product), x86-64 (AVX2) and
  WebAssembly, bit-identical to the scalar reference; `TOMOZ_KERNEL` to
  force one.
- Built-in models trained on 17 public TCIA collections (CC BY).
- Byte-exact DICOM archives (`.tmzd`), with projection images coded
  slice-independently.
- S3-compatible gateway with SigV4, aws-chunked uploads, multipart uploads,
  background series compaction, LRU cache and Prometheus metrics.
- `tomoz` command line, Python bindings (NumPy), WebAssembly build with a
  JavaScript API and a browser viewer.
- Lab: reproducible TCIA datasets, quantisation-aware training, golden
  vectors, evaluation against JPEG-LS, JPEG 2000, HTJ2K, JPEG-XL and zstd.
- Conformance vectors, fuzz targets for every parser of untrusted input,
  CI on Linux (x86-64, AArch64), macOS, Windows and WebAssembly.
