# Architecture

## The problem and its constraints

Medical images must be stored losslessly, for years, and hospitals and
research archives hold petabytes of them. The standard lossless codecs used
in DICOM (JPEG-LS, JPEG 2000, HTJ2K, JPEG-XL) code each slice as a 2-D image,
although CT, MR and PET series are volumes whose neighbouring slices are
highly correlated. Learned codecs exploit that and more, but published ones
run on GPUs at well under a megabyte per second and use floating point, so
their output depends on the hardware they ran on — unacceptable for an
archive that must decode in ten years on machines that do not exist yet.

Tomoz is designed around five constraints:

1. **Lossless, verifiably.** Every decode is checked against checksums
   stored with the data; a DICOM file comes back byte for byte.
2. **Deterministic everywhere.** The same input gives the same bytes on
   x86-64, AArch64 and WebAssembly, with or without SIMD, with any number of
   threads. Conformance hashes enforce it in CI.
3. **Fast on CPUs.** Hospitals do not put GPUs in their storage path. The
   predictor is a few thousand integer multiply-accumulates per sample, in
   SIMD.
4. **Random access.** Viewers read single slices; a slice must decode without
   the whole series.
5. **Drop-in deployment.** No change to PACS software or clients: Tomoz
   stores through the S3 API they already speak.

## Components

![Architecture and data flow](images/architecture.svg)

```text
                     ┌──────────────────────────────────────────────┐
  PACS / scripts ──▶ │ tomoz-gateway   S3 API, SigV4, SQLite index,  │
  (S3 clients)       │                 compaction, cache, metrics    │
                     └───────────────┬──────────────────────────────┘
                                     │ packs series
  tomoz-cli ─────────┐      ┌────────▼────────┐      ┌───────────────┐
  tomoz-python ──────┼────▶ │ tomoz-archive   │────▶ │ tomoz-dicom   │
  tomoz-wasm (JS) ───┤      │ byte-exact DICOM│      │ parser, pixel │
                     │      └────────┬────────┘      │ layouts       │
                     │               │ codes stacks  └───────────────┘
                     │      ┌────────▼────────┐
                     └────▶ │ tomoz-codec     │  container, tiles, context
                            │                 │  model, residual coding
                            └───┬─────────┬───┘
                     ┌──────────▼──┐   ┌──▼─────────────┐
                     │ tomoz-nn    │   │ tomoz-entropy  │
                     │ integer MLP │   │ range coder,   │
                     │ SIMD kernels│   │ adaptive models│
                     └─────────────┘   └────────────────┘

  lab/ (Python): TCIA datasets → training (PyTorch, QAT) → .tzm models
                 → golden vectors for the Rust codec → evaluation
```

| Crate | Role | Notes |
|---|---|---|
| `tomoz-entropy` | Range coder, adaptive 16-symbol and binary models, bypass bits | `no_std`; never panics on malformed input |
| `tomoz-nn` | Integer MLP runtime: 8-bit weights and activations, 32-bit accumulators | Scalar, NEON (i16 and dot-product), AVX2 and WASM SIMD kernels, all bit-identical; loader proves accumulators cannot overflow |
| `tomoz-codec` | The volume codec and `.tmz` container | Tiles coded in parallel (rayon); value mapping and histogram packing; model registry by content hash |
| `tomoz-dicom` | Minimal, robust DICOM Part 10 parser and native pixel layouts | Explicit/implicit VR, big endian, undefined lengths, sequences; keeps exact byte offsets |
| `tomoz-archive` | `.tmzd` archives: series → stacks → Tomoz, everything else zstd | Byte-exact restoration, SHA-256 per file |
| `tomoz-gateway` | S3-compatible server with background compaction | tokio + hyper, SQLite (WAL), LRU cache, Prometheus metrics |
| `tomoz-cli` | `tomoz` command: encode/decode `.npy`/NIfTI, DICOM archives, benchmarks, `serve` | |
| `tomoz-python` | Python bindings (PyO3, abi3 wheels) with NumPy arrays | Releases the GIL while coding |
| `tomoz-wasm` | WebAssembly build with a C ABI and a JavaScript wrapper | Browser and Node.js; demo viewer |

The format specification is in [format.md](format.md); the gateway in
[gateway.md](gateway.md); the evaluation in [evaluation.md](evaluation.md);
design decisions in [decisions/](decisions/).

## How a volume is coded

A volume is cut into tiles of `slab` slices × `stripe` rows (32 × 512 by
default). Each tile is coded independently, sample by sample in raster
order:

1. **Context.** From the causal neighbourhood — rows above in the current
   slice and, from the second slice of a tile on, the two previous slices — a
   *reference* prediction `r` (the sample above, or the median of above,
   previous-slice and their planar combination) and an *activity* `a` (sum of
   local gradients) are computed. Neighbours are expressed relative to `r`
   and divided by `a`, then companded to 8 bits: the network sees the local
   *shape* of the signal, independent of its absolute intensity and
   contrast. (Feeding absolute intensities made the models overfit to
   scanners; see [ADR 0003](decisions/0003-relative-inputs.md).)
2. **Network.** A 12-input (first slice of a tile) or 30-input (other slices)
   integer MLP with two hidden layers of 48 units outputs six numbers.
3. **Head.** Four outputs weight a linear predictor over the left neighbours
   (`W`, `WW`, `WWW`, relative to `r`), giving the predicted mean in 1/16
   units; two outputs give the log-scale of the residual distribution. The
   left neighbours are not network inputs: they are the most informative
   context, and keeping them in an exact linear head lets the network be
   computed for a whole row at once, before the row is coded.
4. **Flat regions.** Where the neighbourhood is constant (air, padding,
   background), a single adaptive bit says "same as the left neighbour".
5. **Residual coding.** The residual is coded with adaptive models selected
   by the predicted scale: a 16-symbol token for the high part, bypass bits
   for the low bits, an adaptive sign bit whose context includes the
   fractional part of the predicted mean.

Every step is integer arithmetic with fully specified rounding. The Python
training code in `lab/` implements the same integer computation, and golden
vectors (`crates/tomoz-codec/tests/golden/`) check that Rust and Python
predict exactly the same values for every sample.

## Determinism

Floating point is not reproducible across instruction sets, compilers and
WebAssembly engines (FMA contraction, vectorised reductions, transcendental
functions). An entropy decoder that disagrees with its encoder by one bit in
one probability produces garbage from then on. So:

- the network is quantised (QAT in training) and runs on integers; the model
  loader proves no accumulator can overflow, so SIMD kernels that reorder the
  additions compute exactly the scalar result;
- all codec arithmetic is integer, with explicit rounding;
- adaptive models update with integer shifts;
- parallelism is over independent tiles, whose order in the file is fixed.

Three layers of tests enforce it: kernel tests (every SIMD kernel against the
scalar one), golden vectors (Rust against the Python reference), and
conformance hashes (`tests/conformance/cases.json`: the exact SHA-256 of the
containers produced for fixed inputs, checked natively on x86-64, AArch64,
macOS and Windows and in WebAssembly by the JavaScript tests).

## Concurrency and resources

- **Codec.** Tiles are coded in parallel with rayon (`threads` option; the
  output does not depend on it). A tile allocates its working buffers (three
  padded slice planes, row context, network scratch) once and reuses them for
  every row. Decoding a whole volume holds the decoded tiles and the
  assembled volume as `i32`, about 8 bytes per sample at peak; the sample
  limit and fallible allocation bound what a hostile header can request.
- **Gateway.** tokio runs the HTTP server; every store operation (SQLite,
  files, coding) runs on the blocking pool so the reactor never stalls.
  Compaction runs on `workers` blocking tasks, one series each, deduplicated.
  SQLite is in WAL mode behind one connection and a mutex: transactions are
  short (index updates only — no I/O or coding inside them). Connections are
  capped (`max_connections`), bodies are size-limited before being read.
- **Cache.** One LRU, bounded in bytes, shared by opened archives and decoded
  slabs; entries are reference-counted so eviction never invalidates a reader.

## Integrity and failure handling

| Layer | Check |
|---|---|
| Container header | CRC-32C |
| Tile stream | CRC-32C of the bytes; the range decoder must end exactly at the end |
| Tile samples | CRC-32C of the decoded samples |
| Volume | SHA-256 of all samples |
| Archive header and tables | CRC-32C |
| Archive instance | SHA-256 of the restored file |
| Gateway raw object | SHA-256 on every read |
| Compaction | full restore and SHA-256 of every object before switching |

Parsers of untrusted input (containers, archives, DICOM, models, S3 requests)
return errors and never panic, never allocate from unvalidated sizes, and are
fuzzed (`fuzz/`). See [SECURITY.md](../SECURITY.md).

## Training and evaluation pipeline

`lab/` is a Python package (`tomoz-lab`):

1. `select` queries the public TCIA API and writes a manifest of series,
   chosen deterministically (hash order) from collections under CC BY
   licences, with training and test collections disjoint by site.
2. `fetch` downloads the manifest's series and pins the SHA-256 of their
   samples, so the data cannot silently change.
3. `train` samples context windows exactly as the codec does, trains the
   float model, then quantisation-aware training with power-of-two scales,
   and exports `.tzm` files with entropy-coder priors estimated on held-out
   samples.
4. `golden` writes the vectors that tie the Rust codec to the Python
   reference.
5. `eval` runs Tomoz and the baselines (JPEG-LS, JPEG 2000, HTJ2K, JPEG-XL
   at efforts 3 and 7, zstd) on the test collections and reports bits per
   sample with bootstrap confidence intervals.
