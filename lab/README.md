# tomoz-lab

Datasets, training and evaluation for the Tomoz codec: everything needed to
rebuild the built-in models and the published results from public data.

```sh
cd lab
uv sync --group dev                    # numpy, pydicom, pytest, ruff
uv sync --extra train --extra baselines # + PyTorch (CPU) and imagecodecs
```

The Tomoz Python bindings are needed for evaluation (`maturin develop` in
`crates/tomoz-python`, or the published wheel).

## Pipeline

| Step | Command | Output |
|---|---|---|
| Select series from TCIA | `tomoz-lab select --out manifests/tz1.json` | Manifest of series (already committed) |
| Download | `tomoz-lab fetch --manifest manifests/tz1.json --out ../data/tz1` | `<id>.npy` + `<id>.json` per series, checked against the pinned SHA-256 |
| Train | `tomoz-lab train --data ../data/tz1 --out models --threads 3` | `tz1-2d.tzm`, `tz1-3d.tzm`, `report.json` |
| Golden vectors | `tomoz-lab golden ../crates/tomoz-codec/tests/golden/predictions.json` | Rust ↔ Python prediction checks |
| Evaluate | `tomoz-lab eval --data ../data/tz1 --out ../eval/tz1` | `results.csv`, `summary.json`, `summary.md` |

`fetch` uses the public NBIA API of The Cancer Imaging Archive; no account
is needed. The TZ1 manifest holds 118 usable series (2 more are marked
excluded: RGB screen captures filed as CT); downloading them takes about an
hour and 2.7 GB of disk. `--split train` or `--split test` fetches one split.

`train` reads only the training split; `eval` only the test split, whose
collections never appear in training (see
[ADR 0006](../docs/decisions/0006-data.md)). `eval --models DIR` evaluates
other models than the built-in ones; `--limit N` the first N volumes.

## Layout

```text
src/tomoz_lab/
  datasets/tcia.py      NBIA API client
  datasets/volumes.py   DICOM series → volume of stored values
  datasets/manifest.py  selection rules, manifest, fetch with hash checks
  features.py           context features (twin of the Rust codec)
  model.py              TZ1 network, quantisation, integer export
  train.py              sampling, float training, QAT, integer evaluation
  export.py             .tzm writer, entropy-coder priors
  golden.py             golden vectors for the Rust tests
  baselines.py          JPEG-LS, JPEG 2000, HTJ2K, JPEG-XL, zstd (imagecodecs)
  evaluate.py           benchmark harness and statistics
  synthetic.py          synthetic DICOM series for tests
tests/
  test_features.py      features against a sample-by-sample reference
  gateway_compat.py     S3 compatibility check against a running gateway
```

The integer computation in `features.py`, `model.py` and the head must stay
identical to the Rust codec: after changing any of them, regenerate the
golden vectors and run `cargo test -p tomoz-codec --test golden`.
