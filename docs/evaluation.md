# Evaluation

How much smaller does Tomoz make real medical images than the lossless codecs
already used with DICOM, on data its models have never seen, and at what
speed? This page describes the protocol and the results; the raw numbers are
in [`eval/tz1/`](../eval/tz1/) (`results.csv` per volume and codec,
`summary.json`, `summary.md`).

## Data

The TZ1 manifest ([`lab/manifests/tz1.json`](../lab/manifests/tz1.json))
pins 118 series from public collections of The Cancer Imaging Archive, all
under CC BY licences. Training and test collections are disjoint — different
sites, scanners and protocols ([ADR 0006](decisions/0006-data.md)):

| | Training | Test |
|---|---|---|
| Volumes | 81 (585 M voxels; 40 central slices per volume) | 37 (850 M voxels; whole series) |
| CT | 38 from 10 collections; GE, Siemens, Toshiba | 12 from 6 collections (CPTAC-PDA, Colorectal-Liver-Metastases, LIDC-IDRI, PSMA-PET-CT-Lesions, Spine-Mets-CT-SEG, StageII-Colorectal-CT); GE, **Philips**, Siemens; 94–344 slices, 1.25–5 mm |
| MR | 21 from 6 collections; GE, Siemens | 9 from 3 collections (Prostate-MRI-US-Biopsy, QIN-BREAST, Vestibular-Schwannoma-SEG); **Philips**, Siemens; 20–200 slices |
| PET | 8 from 2 FDG collections; Siemens | 6 from 2 collections (ACRIN-NSCLC-FDG-PET, PSMA-PET-CT-Lesions: **another tracer**); CPS, GE, Siemens; 128–452 slices |
| Radiography | 8 (CR, DX; Carestream) | 6 (COVID-19-AR; **Fujifilm** CR, **Philips** DX, 10–15 bits) |
| Mammography | 6 (CBIS-DDSM, digitised film, 16 bits) | 4 (CMMD, digital, 8 bits; 2–4 views per series) |

Vendors in bold never appear in training. Samples are the stored values
(no rescale), so every codec sees exactly the bytes a DICOM file holds.

## Protocol

- **Codecs.** Tomoz with its built-in models, single-threaded (`tomoz`) and
  on all cores (`tomoz-mt`, same bytes). Baselines through
  [imagecodecs](https://github.com/cgohlke/imagecodecs) 2026.8.16:
  JPEG-LS (CharLS, lossless), JPEG 2000 (OpenJPEG, reversible 5/3), HTJ2K
  (reversible), JPEG-XL (libjxl, lossless, efforts 3 and 7) and zstd level 19
  on the raw samples. The 2-D codecs code each slice as an image; their sizes
  are summed over slices.
- **Projection images.** A radiography or mammography "volume" is a stack of
  different views, so Tomoz codes it one tile per image (no inter-image
  prediction), exactly as its DICOM archives do.
- **Losslessness.** Every decoded volume is compared with the original; any
  mismatch is recorded. There were none, for any codec.
- **Size.** Bits per sample = 8 × compressed bytes / samples, per volume.
  Group values are means over volumes.
- **Savings.** For each volume, `1 − tomoz_bytes / baseline_bytes`; the
  table reports the mean over volumes (each volume counts once, whatever its
  size), a 95 % bootstrap confidence interval (10,000 resamples of volumes)
  and on how many volumes Tomoz was smaller.
- **Speed.** Raw megabytes (2 bytes per sample) per second of wall time, on
  an Arm Neoverse-N1 server (4 cores, shared with other services, run at low
  priority), one thread per codec unless stated. Tomoz uses the AArch64
  dot-product kernel. Baselines see the samples shifted to start at zero, as
  Tomoz does.

## Results

Mean bits per sample (lower is better), Tomoz's mean saving and its 95 %
confidence interval:

### Volumes

| Codec | CT (12) | MR (9) | PET (6) |
|---|---|---|---|
| **Tomoz** | **4.166** | **5.052** | **3.218** |
| JPEG-XL e7 | 4.366 · **4.6 %** [3.1, 6.2] · 11/12 | 5.201 · **3.5 %** [0.0, 6.9] · 7/9 | 3.598 · **10.8 %** [7.1, 14.8] · 6/6 |
| JPEG-XL e3 | 4.459 · 6.6 % [5.0, 8.3] · 12/12 | 5.320 · 6.1 % [3.5, 9.0] · 9/9 | 3.792 · 17.2 % [12.6, 23.4] · 6/6 |
| JPEG-LS | 4.592 · 9.4 % [7.7, 11.3] · 12/12 | 5.601 · 11.5 % [5.6, 18.5] · 9/9 | 4.071 · 23.2 % [18.8, 28.6] · 6/6 |
| JPEG 2000 | 4.736 · 12.1 % [10.0, 14.5] · 12/12 | 5.643 · 11.8 % [8.2, 15.7] · 9/9 | 4.209 · 25.0 % [21.9, 29.7] · 6/6 |
| HTJ2K | 5.058 · 17.9 % [15.7, 20.2] · 12/12 | 5.933 · 16.2 % [12.6, 20.3] · 9/9 | 4.399 · 28.2 % [25.1, 32.7] · 6/6 |
| zstd-19 | 6.321 · 34.6 % [31.1, 37.1] · 12/12 | 7.073 · 29.1 % [23.4, 35.1] · 9/9 | 5.095 · 37.8 % [35.1, 41.1] · 6/6 |

### 2-D images

| Codec | CR (3) | DX (3) | MG (4) |
|---|---|---|---|
| **Tomoz** | 4.384 | 9.942 | 1.185 |
| JPEG-XL e7 | 4.371 · −0.3 % [−0.9, 0.8] | 10.013 · 0.8 % [0.4, 1.5] | 1.183 · −0.2 % [−0.6, 0.2] |
| JPEG-XL e3 | 4.455 · 1.5 % [−0.4, 5.4] | 9.894 · −0.4 % [−0.7, 0.2] | 1.189 · 0.4 % [0.0, 0.9] |
| JPEG-LS | 4.569 · 4.0 % [2.7, 6.0] | 9.928 · −0.2 % [−0.6, 0.1] | 1.198 · 1.1 % [0.1, 1.8] |
| JPEG 2000 | 4.532 · 3.2 % [1.2, 7.0] | 10.124 · 1.7 % [1.2, 2.2] | 1.199 · 1.2 % [0.6, 1.7] |
| HTJ2K | 4.838 · 9.3 % [7.4, 12.6] | 10.421 · 4.7 % [4.0, 5.5] | 1.267 · 6.5 % [6.0, 7.0] |

### All 37 test series

| Codec | Bits/sample | Tomoz saving | 95 % CI | Tomoz smaller | Encode MB/s | Decode MB/s |
|---|---|---|---|---|---|---|
| **Tomoz** (1 thread) | **4.392** | – | – | – | 9.8 | 10.1 |
| Tomoz (4 threads) | 4.392 | – | – | – | 25.8 | 27.8 |
| JPEG-XL e7 | 4.559 | 4.1 % | [2.5, 5.8] | 30/37 | 3.3 | 23.9 |
| JPEG-XL e3 | 4.647 | 6.5 % | [4.5, 8.8] | 32/37 | 31.4 | 44.4 |
| JPEG-LS | 4.817 | 10.0 % | [7.2, 13.0] | 34/37 | 109.1 | 130.8 |
| JPEG 2000 | 4.909 | 11.4 % | [8.7, 14.1] | 37/37 | 18.5 | 20.9 |
| HTJ2K | 5.171 | 16.1 % | [13.5, 18.8] | 37/37 | 121.4 | 260.6 |
| zstd-19 | 6.320 | 31.2 % | [28.7, 33.6] | 37/37 | 1.9 | 261.9 |

## Reading the results

- **Volumes are where Tomoz wins.** Against the strongest classical codec,
  JPEG-XL at effort 7, Tomoz is 4.6 % smaller on CT, 3.5 % on MR and 10.8 % on
  PET, on scanners, sites and (for PET) a tracer it never saw in training.
  Against the codecs most PACS actually use (JPEG-LS, JPEG 2000), the saving
  is 9–25 %. Coding the same volumes slice by slice with Tomoz's 2-D model
  would cost 3.5–9 % more ([ADR 0002](decisions/0002-tiles.md)): most of the
  gain over 2-D codecs comes from inter-slice prediction.
- **On 2-D images it is on par**, within ±1 % of the best of JPEG-XL and
  JPEG-LS. The TZ1 2-D model is small and trained on few radiographs; this is
  the obvious place for a bigger model.
- **MR is the most variable group** (CI from 0.0 % to 6.9 % against JPEG-XL
  e7). Tomoz saves 6–11 % on the thin-slice breast and prostate series, and
  loses on the two small 20-slice prostate series with 3.6 mm slices, whose
  neighbouring slices share little (−6.2 % and −0.1 %); on the noisy
  high-resolution Vestibular-Schwannoma-SEG series (6–7 bits per sample)
  every codec is within 3 %.
- **Speed.** Tomoz encodes three times faster than JPEG-XL e7 and decodes at
  about 10 MB/s per core — slower than JPEG-XL decoding (24 MB/s), and much
  slower than JPEG-LS and HTJ2K (130–260 MB/s), which trade compression for
  speed. Tomoz is symmetric (decoding costs what encoding costs) and parallel
  across tiles: 26–28 MB/s on four cores. For archives, where data is
  written once and read rarely, this is the trade-off Tomoz is built for.

## Limitations of this evaluation

- 37 test series is enough for the confidence intervals above, not for
  claims about every protocol; the per-volume results are in
  `results.csv` for closer inspection.
- Speeds were measured on one machine, shared with other services; they are
  indicative, and the AVX2 kernel (x86-64) is tested for exactness but was
  not benchmarked here.
- Baselines run with the settings above through one library build; other
  builds or settings may differ slightly.

## Reproducing

```sh
cd lab
uv sync --extra baselines
uv run tomoz-lab fetch --manifest manifests/tz1.json --out ../data/tz1 --split test
uv run tomoz-lab eval --data ../data/tz1 --out ../eval/tz1
```

The evaluation needs the Tomoz Python bindings (`maturin develop --release`
in `crates/tomoz-python`). It resumes where it stopped if interrupted. Sizes
are deterministic and reproduce exactly; times depend on the machine.
