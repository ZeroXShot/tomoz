"""The vectorised feature extraction matches a direct per-sample reading of
the specification."""

import math

import numpy as np
import pytest

from tomoz_lab import features as F


def cur_value(s, t0, y, x):
    """Causal padding of the current slice, sample by sample."""
    w = s.shape[1]
    if y < 0:
        return t0
    if x < 0:
        return t0 if y == 0 else int(s[y - 1, 0])
    if x >= w:
        return int(s[y, w - 1])
    return int(s[y, x])


def prev_value(p, y, x):
    h, w = p.shape
    return int(p[min(max(y, 0), h - 1), min(max(x, 0), w - 1)])


def ilog2_8_scalar(x):
    e = x.bit_length() - 1
    m = (x >> (e - 3)) & 7 if e >= 3 else (x << (3 - e)) & 7
    return 8 * e + [0, 1, 3, 4, 5, 6, 6, 7][m]


def pwl_scalar(v):
    if v < 32:
        return v
    if v < 160:
        return 32 + ((v - 32) >> 2)
    return min(127, 64 + ((v - 160) >> 5))


def sample_features(s, t0, p1, p2, y, x):
    far = [cur_value(s, t0, y + dy, x + dx) for dy, dx in F.FAR_2D]
    n, nn, nw, ne = far[0], far[1], far[2], far[3]
    if p1 is not None:
        p2 = p1 if p2 is None else p2
        far += [prev_value(p1, y + dy, x + dx) for dy, dx in F.FAR_P1]
        far += [prev_value(p2, y + dy, x + dx) for dy, dx in F.FAR_P2]
        p, pn = prev_value(p1, y, x), prev_value(p1, y - 1, x)
        r = sorted([n, p, n + p - pn])[1]
        act = abs(n - nw) + abs(n - ne) + abs(nn - n) + abs(p - pn) + abs(n - pn)
        a4 = max(4 * act // 5, 2) + 2
    else:
        r = n
        act = abs(n - nw) + abs(n - ne) + abs(nn - n)
        a4 = max(4 * act // 3, 2) + 2
    inv = (1 << 22) // a4
    q = []
    for v in far:
        d = v - r
        mag = pwl_scalar((64 * abs(d) * inv) >> 22)
        q.append(int(math.copysign(mag, d)) if d else 0)
    q.append(max(-127, min(127, ilog2_8_scalar(a4) - 16)))
    near = [max(-4096, min(4096, ((cur_value(s, t0, y + dy, x + dx) - r) * inv) >> 12)) for dy, dx in F.NEAR]
    west = cur_value(s, t0, y, x - 1)
    flat = n == west and nw == west and ne == west
    flat = flat and (prev_value(p1, y, x) == west if p1 is not None else cur_value(s, t0, y, x - 2) == west)
    return q, r, a4, near, flat


@pytest.mark.parametrize("three_d", [False, True])
@pytest.mark.parametrize("second_slice", [False, True])
def test_vectorised_matches_scalar(three_d, second_slice):
    rng = np.random.default_rng(3 + three_d + 2 * second_slice)
    vol = rng.integers(0, 4000, size=(3, 9, 11))
    vol[:, 2:5, 3:8] = 1000  # a flat patch
    t0 = int(vol[0, 0, 0])
    p1 = vol[0] if three_d else None
    p2 = None if second_slice or not three_d else vol[1]
    s = vol[2]
    f = F.slice_features(s, t0, p1, p2)
    for y in range(s.shape[0]):
        for x in range(s.shape[1]):
            q, r, a4, near, flat = sample_features(s, t0, p1, p2, y, x)
            assert list(f.inputs[y, x]) == q, (y, x)
            assert (f.r[y, x], f.a4[y, x], list(f.near[y, x]), bool(f.flat[y, x])) == (r, a4, near, flat), (
                y,
                x,
            )


def test_ilog2_8():
    xs = np.array([1, 2, 3, 7, 8, 9, 15, 16, 1000, 65535, 1 << 40])
    assert list(F.ilog2_8(xs)) == [ilog2_8_scalar(int(x)) for x in xs]
    assert F.ilog2_8(np.array([1024]))[0] == 80


def test_head_identity():
    """Zero outputs predict the reference with the activity as scale."""
    rng = np.random.default_rng(0)
    vol = rng.integers(0, 300, size=(2, 6, 7))
    f = F.slice_features(vol[1], int(vol[0, 0, 0]), vol[0], None)
    h = F.head(np.zeros((6, 7, 6), np.int64), f, out_shift=12, max_value=299)
    assert np.array_equal(h.mu16, 16 * np.clip(f.r, 0, 299))
    assert np.array_equal(h.log_scale8, F.ilog2_8(f.a4) - 16)


def test_points_match_full_slice():
    rng = np.random.default_rng(11)
    vol = rng.integers(0, 900, size=(3, 15, 12))
    full = F.slice_features(vol[2], 5, vol[1], vol[0])
    ys = rng.integers(0, 15, 40)
    xs = rng.integers(0, 12, 40)
    pts = F.slice_features(vol[2], 5, vol[1], vol[0], points=(ys, xs))
    assert np.array_equal(pts.inputs, full.inputs[ys, xs])
    assert np.array_equal(pts.near, full.near[ys, xs])
    assert np.array_equal(pts.r, full.r[ys, xs])
    assert np.array_equal(pts.flat, full.flat[ys, xs])
