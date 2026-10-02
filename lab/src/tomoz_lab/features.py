"""Reference implementation of the Tomoz context model (model family TZ1).

This module defines, in integer arithmetic on int64 arrays, everything the
codec computes before the entropy coder: the causal neighbourhood of each
sample, the reference prediction and local activity, the 8-bit network
inputs, and the prediction head that turns network outputs into a predicted
mean and a log-scale. The Rust implementation in ``crates/tomoz-codec`` must
match it bit for bit; ``tomoz-lab golden`` writes test vectors from this
module that the Rust tests replay.

Conventions (see docs/algorithms.md):

* A tile is a stack of slices coded independently. Slice 0 of a tile is
  coded with the 2-D model (rows above only); later slices with the 3-D model,
  which also sees the two previous slices (slice 1 sees slice 0 twice).
* Values outside the current slice are padded causally: rows above the tile
  hold ``t0`` (the first sample of the tile), the columns left of row ``y``
  hold the first sample of row ``y - 1``, the columns right of row ``y`` hold
  its last sample. Previous slices are complete and padded by replication.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PAD = 3

#: Neighbours in the rows above, (dy, dx).
FAR_2D = (
    (-1, 0),
    (-2, 0),
    (-1, -1),
    (-1, 1),
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (-3, 0),
    (-2, -2),
    (-2, 2),
)
#: Neighbours in the previous slice.
FAR_P1 = (
    (0, 0),
    (0, -1),
    (0, 1),
    (-1, 0),
    (1, 0),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
    (0, -2),
    (0, 2),
    (-2, 0),
    (2, 0),
)
#: Neighbours two slices back.
FAR_P2 = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0))
#: Neighbours to the left in the current row; they enter the linear head only.
NEAR = ((0, -1), (0, -2), (0, -3))

#: Network inputs: one per far neighbour plus the log-activity feature.
INPUTS_2D = len(FAR_2D) + 1
INPUTS_3D = len(FAR_2D) + len(FAR_P1) + len(FAR_P2) + 1
#: Network outputs: mean offset, three near coefficients, log-scale and its
#: dependence on the left neighbour.
OUTPUTS = 6

_LOG8_MANTISSA = np.array([0, 1, 3, 4, 5, 6, 6, 7], dtype=np.int64)


def ilog2_8(x: np.ndarray) -> np.ndarray:
    """8 * log2(x) for x >= 1, from the position of the leading one and the
    three bits after it."""
    x = np.asarray(x, dtype=np.int64)
    if np.any(x < 1):
        raise ValueError("ilog2_8 needs x >= 1")
    # frexp is exact for integers below 2**53: x = m * 2**e with m in [0.5, 1).
    _, e = np.frexp(x.astype(np.float64))
    e = e.astype(np.int64) - 1
    m = np.where(e >= 3, x >> np.maximum(e - 3, 0), x << np.maximum(3 - e, 0)) & 7
    return 8 * e + _LOG8_MANTISSA[m]


def pwl(v: np.ndarray) -> np.ndarray:
    """Piecewise-linear companding of a magnitude in 1/16 units to [0, 127]."""
    return np.where(
        v < 32, v, np.where(v < 160, 32 + ((v - 32) >> 2), np.minimum(127, 64 + ((v - 160) >> 5)))
    )


def pad_current(s: np.ndarray, t0: int) -> np.ndarray:
    """The current slice as the decoder sees it, with causal padding."""
    h, w = s.shape
    c = np.empty((h + PAD, w + 2 * PAD), dtype=np.int64)
    c[:PAD, :] = t0
    c[PAD:, PAD : PAD + w] = s
    c[PAD:, PAD + w :] = s[:, w - 1 : w]
    left = np.empty(h, dtype=np.int64)
    left[0] = t0
    left[1:] = s[:-1, 0]
    c[PAD:, :PAD] = left[:, None]
    return c


def pad_previous(s: np.ndarray) -> np.ndarray:
    """A complete slice padded by replication."""
    return np.pad(s.astype(np.int64), PAD, mode="edge")


@dataclass
class SliceFeatures:
    """Context of every sample of one slice."""

    #: Network inputs, (H, W, n_inputs) int8.
    inputs: np.ndarray
    #: Reference prediction r, (H, W).
    r: np.ndarray
    #: Activity A in units of 1/4, (H, W), at least 4.
    a4: np.ndarray
    #: Near neighbours relative to r in units of A/256, (H, W, 3), within +-4096.
    near: np.ndarray
    #: Samples whose neighbourhood is flat (coded with a single flag when
    #: they equal their left neighbour), (H, W) bool.
    flat: np.ndarray
    #: Left neighbour W, (H, W).
    west: np.ndarray


def slice_features(
    cur: np.ndarray,
    t0: int,
    prev1: np.ndarray | None,
    prev2: np.ndarray | None,
    points: tuple[np.ndarray, np.ndarray] | None = None,
) -> SliceFeatures:
    """Features of the samples of ``cur`` (H, W) given the previous slices of
    the tile (None for the first slice; ``prev2`` defaults to ``prev1``).

    With ``points`` = (ys, xs), only those samples are computed and every
    output has one entry per point instead of the shape of the slice."""
    h, w = cur.shape
    c = pad_current(cur, t0)
    three_d = prev1 is not None

    def at(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
        if points is not None:
            return arr[PAD + dy + points[0], PAD + dx + points[1]]
        return arr[PAD + dy : PAD + dy + h, PAD + dx : PAD + dx + w]

    far = [at(c, dy, dx) for dy, dx in FAR_2D]
    n, nn, nw, ne = far[0], far[1], far[2], far[3]
    if three_d:
        p1 = pad_previous(prev1)
        p2 = pad_previous(prev2 if prev2 is not None else prev1)
        far += [at(p1, dy, dx) for dy, dx in FAR_P1]
        far += [at(p2, dy, dx) for dy, dx in FAR_P2]
        p, pn = at(p1, 0, 0), at(p1, -1, 0)
        g = n + p - pn
        r = np.maximum(np.minimum(n, p), np.minimum(np.maximum(n, p), g))
        act = np.abs(n - nw) + np.abs(n - ne) + np.abs(nn - n) + np.abs(p - pn) + np.abs(n - pn)
        a4 = np.maximum(4 * act // 5, 2) + 2
    else:
        r = n
        act = np.abs(n - nw) + np.abs(n - ne) + np.abs(nn - n)
        a4 = np.maximum(4 * act // 3, 2) + 2

    inv = (1 << 22) // a4
    d = np.stack(far, axis=-1) - r[..., None]
    v = (64 * np.abs(d) * inv[..., None]) >> 22
    q = np.sign(d) * pwl(v)
    qa = np.clip(ilog2_8(a4) - 16, -127, 127)
    inputs = np.concatenate([q, qa[..., None]], axis=-1).astype(np.int8)

    near_vals = [at(c, dy, dx) for dy, dx in NEAR]
    near = np.stack([np.clip(((x - r) * inv) >> 12, -4096, 4096) for x in near_vals], axis=-1)
    west, ww = near_vals[0], near_vals[1]
    flat = (n == west) & (nw == west) & (ne == west)
    flat &= (at(p1, 0, 0) == west) if three_d else (ww == west)
    return SliceFeatures(inputs=inputs, r=r, a4=a4, near=near, flat=flat, west=west)


@dataclass
class HeadOutput:
    """Predicted mean (1/16 units, clamped to the sample range) and log2 of
    the scale in 1/8 octaves."""

    mu16: np.ndarray
    log_scale8: np.ndarray


def head(out: np.ndarray, f: SliceFeatures, out_shift: int, max_value: int) -> HeadOutput:
    """Applies the prediction head to network outputs ``out`` (..., 6) int."""
    o = out.astype(np.int64)
    dw, dww, dwww = f.near[..., 0], f.near[..., 1], f.near[..., 2]
    m = (o[..., 0] << 8) + o[..., 1] * dw + o[..., 2] * dww + o[..., 3] * dwww
    m = np.clip(m, -(1 << 40), 1 << 40)
    mu16 = 16 * f.r + ((f.a4 * m + (1 << (out_shift + 3))) >> (out_shift + 4))
    mu16 = np.clip(mu16, 0, 16 * max_value)
    lg_w8 = ilog2_8(256 + np.abs(dw)) - 64
    ls8 = (8 * o[..., 4] + o[..., 5] * lg_w8 + (1 << (out_shift - 1))) >> out_shift
    return HeadOutput(mu16=mu16, log_scale8=ilog2_8(f.a4) - 16 + ls8)
