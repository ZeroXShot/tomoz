# 0002 — Volumes are coded as independent tiles

## Context

Viewers read single slices, often out of order. Storage servers want to use
all cores. A sequential codec over a whole volume gives the best compression
but neither random access nor parallelism.

## Decision

A volume is cut into tiles of `slab` slices × `stripe` rows (full width).
Each tile is coded independently — fresh adaptive models, padding instead of
neighbours across its borders — and the container indexes the tiles. A range
of slices decodes only the tiles of its slabs; tiles are coded in parallel.
The first slice of each tile uses the 2-D model (no previous slice exists
inside the tile), the others the 3-D model.

Defaults: 32 × 512 for files; the gateway uses slabs of 16 for finer random
access. Stacks whose slices are not known to be spatially coherent
(projection images: different views of one series, without positions) use
slabs of 1, so no image is predicted from another.

## Measurements

Bits per sample with the TZ1 models, relative to one tile per volume
(up to 128 central slices of held-out series):

| Tiling | CT (LIDC-IDRI) | MR (Prostate-MRI-US-Biopsy) | PET (ACRIN-NSCLC-FDG-PET) |
|---|---|---|---|
| 1 slice per tile (2-D only) | +3.5 % | +6.4 % | +8.9 % |
| 4 × full height | +1.1 % | +1.8 % | +2.3 % |
| 16 × full height | +0.3 % | +0.4 % | +0.5 % |
| **32 × 512 (default)** | **+0.1 %** | **+0.1 %** | **+0.2 %** |
| 32 × 128 | +0.4 % | +0.2 % | +0.2 % |
| 32 × 64 | +0.8 % | +0.4 % | +0.6 % |

The first row is also the value of inter-slice prediction: coding each slice
as an image costs 3.5–8.9 % more.

## Alternatives

- *One stream per volume*: 0.1–0.2 % smaller, no random access, one core.
- *Random-access points inside one stream* (resetting models periodically
  but keeping context): similar cost, but decoding still needs the previous
  slices' samples, so a single slice cannot be decoded alone.

## Consequences

- Decoding one slice costs at most one slab; the gateway caches decoded
  slabs so neighbouring slices are free.
- Throughput scales with cores up to the number of tiles; a small volume
  has few tiles.
- The tiling is recorded in the container and does not affect the decoded
  samples, so encoders may choose it freely.
