# 0004 — Learned mean and scale, adaptive classical entropy coding

## Context

Learned lossless codecs in the literature typically let the network output a
full distribution (a mixture of logistics, or a discretised CDF) per sample.
That makes the network larger and puts all the adaptation burden on it: a
distribution learned on training data stays fixed on test data.

## Decision

The network outputs six numbers: four weights of a linear predictor over the
left neighbours (giving the predicted mean in 1/16 units) and two terms of a
log-scale. The residual is then coded with *adaptive* models selected by the
predicted scale — a 16-symbol token model for the high part of the
magnitude, bypass bits for the low part, a binary sign model whose context
includes the fractional part of the predicted mean — initialised from priors
estimated at training time and updated as the tile is coded. Flat regions
are coded with an adaptive "same as left" flag before the network is
consulted.

## Alternatives

- *Network-predicted CDFs*: the most general, but needs many outputs per
  sample (or a mixture whose CDF needs `exp`), and cannot adapt to a volume's
  statistics while decoding.
- *Context mixing (PAQ-style) of many models*: excellent compression, an
  order of magnitude slower, hard to make bit-exact with SIMD.

## Consequences

- The network is small enough (30-48-48-6: about 4,000 multiply-accumulates
  per sample for the 3-D model) to run at several MB/s per core on CPUs.
- The adaptive models absorb distribution shift: a scanner whose noise is
  heavier than anything in training costs a few symbols of adaptation, not a
  systematic loss.
- The left neighbours enter only through the exact linear head, so the
  network for a whole row is computed before the row is coded, in one batch.
