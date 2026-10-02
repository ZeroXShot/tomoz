# 0003 — The network sees relative, contrast-normalised context

## Context

The models must work on scanners, protocols, sites and modalities they were
not trained on, without retraining. Intensities mean different things across
modalities (Hounsfield units, arbitrary MR units, PET counts) and their
ranges differ across scanners even within a modality.

## Decision

The network never sees absolute values. For each sample, a reference
prediction `r` (the sample above in 2-D; the median of above, previous slice
and their planar combination in 3-D) and a local activity `a` (sum of
gradients) are computed; each neighbour enters as `(v − r) / a`, companded to
8 bits, plus `log₂ a` itself. The output is a correction to `r` in units of
`a`, and a log-scale relative to `a`. The network therefore learns the local
shape of the signal — edges, textures, noise structure — which transfers
across intensity scales.

## Negative results

Two variants were tried during the feasibility study and dropped:

- **Absolute intensity as an input** (the reference value, normalised by the
  bit depth): better on the training scanners, worse on held-out
  collections — the network learned scanner-specific intensity statistics.
- **Acquisition metadata as an input** (slice spacing relative to pixel
  spacing, to tell the 3-D model how much to trust the previous slice): it
  made held-out results much worse — most likely because the training
  collections cover few distinct protocols, so the network keyed on values
  that held-out series do not share. The reference predictor's planar term
  and the activity of the previous-slice gradients already adapt to slice
  spacing locally.

## Consequences

- One pair of models covers CT, MR, PET, radiography and mammography.
- Values far outside the training distribution of *differences* (e.g. synthetic
  images with huge steps) are clamped by the companding, so predictions
  degrade gracefully to the reference predictor rather than failing.
