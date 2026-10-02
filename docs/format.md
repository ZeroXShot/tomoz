# Tomoz formats

This document specifies the three file formats of Tomoz precisely enough to
write an independent decoder:

1. the volume container (`.tmz`, magic `0x89 'T' 'M' 'Z'`);
2. the model file (`.tzm`, magic `TZM1`);
3. the DICOM archive (`.tmzd`, magic `0x89 'T' 'Z' 'D'`).

The reference implementation is the Rust code cited in each section; where
this text and the code disagree, the code is right and this text has a bug.
Conformance vectors that any implementation must reproduce bit for bit are in
[`crates/tomoz-codec/tests/conformance/cases.json`](../crates/tomoz-codec/tests/conformance/cases.json).

All integers are little endian. Division of negative numbers is never used;
`>>` on signed values is an arithmetic shift (rounds towards −∞).

---

## 1. Volume container (`.tmz`)

Source: [`container.rs`](../crates/tomoz-codec/src/container.rs),
[`lib.rs`](../crates/tomoz-codec/src/lib.rs),
[`volume.rs`](../crates/tomoz-codec/src/volume.rs).

A volume is `depth × height × width` integer samples of `bits` bits (1 to 16),
signed (two's complement range) or unsigned, stored slice after slice, row
after row.

### 1.1 Header

```text
offset  size  field
     0     4  magic: 0x89 'T' 'M' 'Z'
     4     2  format version (1)
     6     2  flags: bit 0 histogram packing, bit 1 application metadata
     8     4  depth
    12     4  height
    16     4  width
    20     1  bits per sample (1..=16)
    21     1  signed (0 or 1)
    22     2  reserved (0)
    24     4  offset (i32): the sample value of mapped value 0
    28     4  max_mapped (i32): the largest mapped value
    32     2  slab: slices per tile (> 0)
    34     2  stripe: rows per tile (> 0)
    36     4  number of tiles
    40     4  length of the variable part (everything after offset 44)
    44    16  identifier of the 2-D model
    60    16  identifier of the 3-D model
    76    32  SHA-256 of the samples, each as a 16-bit little-endian integer
   108     4  length p of the packing table (0 unless flag bit 0)
   112     p  packing table (§1.3)
     .     4  length m of the application metadata (0 unless flag bit 1)
     .     m  application metadata (opaque; at most 16 MiB)
     .  20×t  tile table, per tile: offset (u64, from the end of the header),
              length (u32), CRC-32C of the tile stream (u32), CRC-32C of the
              tile's original samples as 16-bit little endian (u32)
     .     4  CRC-32C of every header byte before this field
```

A decoder must reject: unknown flags; `depth`, `height` or `width` of zero, or
whose product overflows or exceeds its sample limit; `bits` outside 1..=16;
`slab` or `stripe` of zero; a mapping outside the value range
(`offset < lo`, `max_mapped < 0` or `offset + max_mapped > hi`); a tile count
different from `ceil(depth / slab) × ceil(height / stripe)`; tile offsets
that are not contiguous starting at zero; a variable-part length that does
not end exactly after the CRC; and a CRC mismatch.

### 1.2 Tiles

Tile `i` covers slab `i / stripes` and stripe `i % stripes`, where
`stripes = ceil(height / stripe)`: slices `[slab_index × slab, …)` and rows
`[stripe_index × stripe, …)`, clipped to the volume, full width. Each tile is
an independent range-coded stream (§1.5): tiles can be decoded in parallel
and in any order, and a range of slices needs only the tiles of its slabs.

### 1.3 Value mapping

The codec works on non-negative *mapped* values `0..=max_mapped`:

* without packing, `mapped = sample − offset` with `offset` the smallest
  sample;
* with packing (flag bit 0), `mapped` is the rank of `sample − offset` among
  the distinct values present. The encoder packs when the span of values is at
  least 64 and fewer than three quarters of it are used (all-even images,
  padding values far from the data).

The packing table is a range-coded stream (§1.5): `span` as 17 bypass bits,
then for each `v` in `0..=span` one binary decision "v is present" with four
`BitModel`s (all starting at one half) selected by the previous two
decisions, `ctx = ((ctx << 1) | present) & 3`, starting at 0. A valid table
has `max_mapped + 1` entries, includes 0 and `span`, and `span ≤ 2^16`.

### 1.4 Coding a tile

Let `M = max_mapped`, `B = max(1, bit_length(M))`. The first sample of the
tile, `t0`, is written as `B` bypass bits. Then every sample is coded in
raster order (slice, row, column) — including the first one again.

**Models.** The first slice of a tile uses the 2-D model; the others use the
3-D model. Each model has its own set of adaptive entropy models (§1.6),
initialised from the model's priors at the start of the tile.

**Planes and padding.** Each slice lives in a buffer with 3 samples of
padding on every side. While slice `z` is coded, the rows above row 0 hold
`t0`; before row `y` is coded, its 3 left padding samples are set to the first
sample of row `y − 1` (`t0` for row 0); after it is coded, its 3 right padding
samples are set to its last sample; padding below the coded rows is not read.
When the slice is complete, its padding is rebuilt by replicating the edge
samples (columns, then rows), and it becomes the previous slice `P1` (the one
before becomes `P2`; for `z = 1`, `P2 = P1`).

Notation for sample `(y, x)` of the current slice `C`: `N = C[y−1][x]`,
`NN = C[y−2][x]`, `NW = C[y−1][x−1]`, `NE = C[y−1][x+1]`, `W = C[y][x−1]`,
`WW = C[y][x−2]`, `WWW = C[y][x−3]`, and `P = P1[y][x]`, `PN = P1[y−1][x]`.

**Row context** (computed for the whole row before coding it):

| | 2-D model | 3-D model |
|---|---|---|
| reference `r` | `N` | `median(N, P, N + P − PN)` |
| activity `act` | `|N−NW| + |N−NE| + |NN−N|` | that `+ |P−PN| + |N−PN|` |
| `a4` | `max(4·act/3, 2) + 2` | `max(4·act/5, 2) + 2` |
| flat value | `N` if `N = NW = NE`, else none | `N` if `N = NW = NE = P`, else none |

`inv = floor(2^22 / a4)`. Each network input is a companded difference: for a
neighbour value `v`, `d = v − r`, `m = (|d| · inv) >> 16` (computed exactly;
`|d| < 2^16`), `q = min(m, 32 + ((m − 32) >> 2), 64 + ((m − 160) >> 5), 127)`,
input `= sign(d) · q`. The inputs, in order:

1. the 11 neighbours of the current slice at (dy, dx) = (−1,0) (−2,0)
   (−1,−1) (−1,1) (−2,−1) (−2,1) (−1,−2) (−1,2) (−3,0) (−2,−2) (−2,2);
2. 3-D only: 13 neighbours of `P1` at (0,0) (0,−1) (0,1) (−1,0) (1,0)
   (−1,−1) (−1,1) (1,−1) (1,1) (0,−2) (0,2) (−2,0) (2,0);
3. 3-D only: 5 neighbours of `P2` at (0,0) (0,−1) (0,1) (−1,0) (1,0);
4. `clamp(ilog2_8(a4) − 16, −127, 127)`.

That is 12 inputs for the 2-D model and 30 for the 3-D model. `ilog2_8(x)` is
`8·e + T[m]` where `e` is the position of the leading one of `x`, `m` the
three bits after it (zero-filled for small `x`) and
`T = [0, 1, 3, 4, 5, 6, 6, 7]`.

**Flat regions.** If the row context has a flat value equal to `W` (and, for
the 2-D model, `WW = W` too), a binary flag is coded first with flag context
`2·is_3d + previous_was_flat`, where `previous_was_flat` tells whether the
previous sample of the row was coded by this flag alone. Bit 0 means "the
sample equals `W`": the sample is `W` and nothing else is coded for it.

**Prediction.** Otherwise the network (§2.2) maps the inputs to six
32-bit outputs `o0..o5`, and with
`near_i = clamp(((v_i − r) · inv) >> 12, −4096, 4096)` for
`v = (W, WW, WWW)` and `s` the model's output shift:

```text
m      = clamp((o0 << 8) + o1·near0 + o2·near1 + o3·near2, −2^40, 2^40)
mu16   = clamp(16·r + ((a4·m + 2^(s+3)) >> (s+4)), 0, 16·M)
lg_w8  = ilog2_8(256 + |near0|) − 64
ls8    = (8·o4 + o5·lg_w8 + 2^(s−1)) >> s
log_scale8 = ilog2_8(a4) − 16 + ls8
```

`mu16` is the predicted mean in 1/16 units; `log_scale8` is `8·log2` of the
predicted scale.

**Residual.** With `predicted = (mu16 + 8) >> 4`, `frac = mu16 − 16·predicted`
(in −8..=7), `l = log_scale8`:

```text
k         = clamp((l >> 3) − 1, 0, 14)           low bits sent as bypass
token_ctx = k == 0 ? clamp((l + 16) >> 1, 0, 15) : 16 + ((l & 7) >> 1)
sign_ctx  = ((frac + 8) >> 2) · 17 + (k == 0 ? token_ctx : 16)
```

For the residual `e = x − predicted`, `m = |e|`, `q = m >> k`:

1. token `min(q, 14)`, or 15 if `q ≥ 30`, with the 16-symbol model
   `token[token_ctx]`;
2. token 14: `q − 14` as 4 bypass bits; token 15: with `v = q − 29`,
   `n = floor(log2 v)` as 5 bypass bits then `v − 2^n` as `n` bypass bits
   (`n ≤ 20`);
3. if `k > 0`, the low `k` bits of `m` as bypass bits;
4. if `m ≠ 0`, the sign (1 = negative) with `sign[sign_ctx]`.

A decoder must reject decoded samples outside `0..=M`.

**Checks.** After decoding a tile, the CRC-32C of its samples (unmapped,
each as 16-bit little endian) must match the tile table; the stream itself
must end exactly where the encoder ended it (§1.5). A full decode also checks
the SHA-256 of the volume.

### 1.5 Range coder

Source: [`tomoz-entropy`](../crates/tomoz-entropy/src/range.rs).

A byte-oriented range coder with carry propagation (the LZMA scheme): `low`
is 33 bits wide, `range` 32 bits, renormalisation by bytes whenever
`range < 2^24`. A sub-interval `[lo, hi)` of `[0, 2^b)` (`b ≤ 16`) is coded as
`r = range >> b; low += r·lo; range = (hi == 2^b) ? range − r·lo : r·(hi − lo)`.
Probabilities have 15 bits. The first byte produced is always zero and is not
written; `finish` shifts out 5 bytes. Bypass values of up to 32 bits are coded
16 bits at a time, most significant first.

### 1.6 Adaptive models

* **16-symbol model** (`Cdf`): cumulative values `f[0..16]` in units of
  2⁻¹⁵ with `f[0] = 0`. Symbol `i` occupies
  `[b(i), b(i+1))` with `b(i) = i + ((f[i] · (2^15 − 16)) >> 15)` and
  `b(16) = 2^15`, so every symbol stays codable. After coding symbol `s`, with
  `rate = 5 + (count ≥ 16) + (count ≥ 32)` and `count` saturating at 32,
  every `f[i]` (`i ≥ 1`) moves: `f += (2^15 − f) >> rate` if `i > s`,
  else `f −= f >> rate`.
* **Binary model** (`BitModel`): two 24-bit estimates of P(0) updated with
  rates 4 and 7 (`p += (2^24 − p) >> rate` after a 0, `p −= p >> rate` after
  a 1); the coder uses `clamp((fast + slow) >> 10, 1, 2^15 − 1)`.

---

## 2. Model file (`.tzm`, TZM1)

Source: [`model.rs`](../crates/tomoz-codec/src/model.rs),
[`tomoz-nn`](../crates/tomoz-nn/src/lib.rs).

### 2.1 Layout

```text
4      magic "TZM1"
1      kind: 2 (2-D model) or 3 (3-D model)
1      output shift s (1..=30)
2      reserved (0)
1      length n of the name (≤ 64)
n      name (UTF-8)
…      network (§2.2)
640    token priors: 20 contexts × 16 cumulative values (u16)
136    sign priors: 68 × P(positive) (u16, units of 2⁻¹⁵)
8      flat priors: 4 × P(not equal) (u16)
```

The model identifier stored in containers is the first 16 bytes of the
SHA-256 of the whole file: a container names exactly the models it needs, and
a decoder refuses to decode with any other.

### 2.2 Network

```text
1      number of layers L (1..=8)
per layer:
  2    inputs (1..=256)
  2    outputs (1..=256)
  1    requantisation shift (≤ 24)
  1    activation: 0 linear (last layer), 1 clipped ReLU (other layers)
  i×o  weights, i8, row-major by output
  4×o  biases, i32
```

Inputs and hidden activations are `i8`, accumulators `i32`:
`acc_j = bias_j + Σ w_ji · x_i`. Hidden layers apply
`clamp((acc + 2^(shift−1)) >> shift, 0, 127)` (no rounding term when
`shift = 0`); the last layer returns the accumulators. The first layer must
take the model kind's input count and the last must produce 6 outputs; the
priors must be valid distributions. Loaders must prove that no accumulator
can exceed 2³⁰ in magnitude for any input: for every output,
`|bias| + Σ |w| · 128` (first layer, inputs in −128..=127) or
`|bias| + Σ |w| · 127` (hidden activations in 0..=127) must be at most 2³⁰.
This is what makes every SIMD kernel compute exactly the scalar result, and
leaves room for the rounding term.

---

## 3. DICOM archive (`.tmzd`)

Source: [`tomoz-archive`](../crates/tomoz-archive/src/).

An archive restores a set of files byte for byte. DICOM instances with
native, uncompressed, one-sample 8- or 16-bit pixel data are grouped into
*stacks* (same series, rows, columns, bits stored, signedness, orientation;
each multi-frame instance on its own) ordered by position along the slice
normal, then instance number, then SOP Instance UID. Each stack is a volume
container (§1). Stacks whose spatial coherence is not established (some
instance without a position) are coded with one tile per slice, so no image
is predicted from another. Everything else is kept verbatim.

```text
offset  size  field
     0     4  magic: 0x89 'T' 'Z' 'D'
     4     2  version (1)
     6     2  flags (0)
     8     4  instance count
    12     4  stack count
    16    16  instance table: offset, length (u64 each)
    32    16  metadata: offset, length (one zstd frame)
    48    16  stored files: offset, length (concatenated zstd frames)
    64     4  CRC-32C of the instance table
    68     4  CRC-32C of bytes 0..68 and the stack table
    72  16×s  stack table: offset, length (u64 each) of each container
```

Instance table entry:

```text
8     size of the original file
32    SHA-256 of the original file
2+n   name (u16 length, UTF-8)
1     kind: 0 stored, 1 coded
stored: 8 offset, 8 length of its zstd frame in the stored-files section
coded:  4 stack, 4 first slice, 4 frames, 4 rows, 4 columns,
        1 bits allocated, 1 bits stored, 1 high bit, 1 signed, 1 big endian,
        1 bit convention (0 zero, 1 sign-extended, 2 irregular),
        8 metadata offset, 8 prefix length, 8 suffix length, 4 trailing length
```

The metadata region of a coded instance is: the bytes of the file before the
pixel data value (`prefix`), the bytes after it (`suffix`), the bytes of the
value after the last cell (`trailing`, e.g. the padding byte of an odd
length) and, for the irregular convention, one u16 per cell holding the cell
XOR its zero-convention reconstruction (overlay bits or garbage outside the
stored bits). A file is rebuilt as `prefix + cells + trailing + suffix` with
the cells re-created from the decoded samples, and must match its SHA-256.

Readers must not trust declared sizes for allocation: a zstd frame cannot
expand more than 2¹⁶ times (an RLE block codes 128 KiB in 4 bytes), so larger
declared sizes are corrupt.
