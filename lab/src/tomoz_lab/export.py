"""Serialisation of trained predictors to the TZM1 model format and
estimation of the entropy coder priors.

The format is defined by ``crates/tomoz-codec/src/model.rs``; the network
encoding by ``crates/tomoz-nn``. The model identifier used in bitstreams is
the first 16 bytes of the SHA-256 of the file.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field

import numpy as np

from . import features as F_
from .model import IntegerNet

TOKEN_CONTEXTS = 20
TOKENS = 16
SIGN_CONTEXTS = 68
FLAT_CONTEXTS = 4
PROB_ONE = 1 << 15
ACC_BOUND = 1 << 30


@dataclass
class Priors:
    token: np.ndarray = field(
        default_factory=lambda: np.tile((np.arange(TOKENS) * PROB_ONE // TOKENS), (TOKEN_CONTEXTS, 1))
    )
    sign: np.ndarray = field(default_factory=lambda: np.full(SIGN_CONTEXTS, PROB_ONE // 2))
    flat: np.ndarray = field(default_factory=lambda: np.full(FLAT_CONTEXTS, PROB_ONE // 2))


def plan(mu16: np.ndarray, log_scale8: np.ndarray):
    """Mirror of ``Prediction::plan`` in residual.rs."""
    predicted = (mu16 + 8) >> 4
    frac = mu16 - 16 * predicted
    k = np.clip((log_scale8 >> 3) - 1, 0, 14)
    token_ctx = np.where(k == 0, np.clip((log_scale8 + 16) >> 1, 0, 15), 16 + ((log_scale8 & 7) >> 1))
    sign_ctx = ((frac + 8) >> 2) * 17 + np.where(k == 0, token_ctx, 16)
    return predicted, k, token_ctx, sign_ctx


def tokens(x: np.ndarray, predicted: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e = x - predicted
    q = np.abs(e) >> k
    return np.minimum(q, 14) + (q >= 30), e


def cumulative(counts: np.ndarray) -> np.ndarray:
    """Cumulative 15-bit frequencies with every symbol at least 1/1024 likely
    (matching how quickly the adaptive model would recover)."""
    c = counts.astype(np.float64) + counts.sum() / 1024.0 + 1e-9
    f = np.concatenate([[0.0], np.cumsum(c)[:-1]]) / c.sum() * PROB_ONE
    return np.floor(f).astype(np.int64)


def probability(zeros: np.ndarray, total: np.ndarray) -> np.ndarray:
    p = (zeros + 1.0) / (total + 2.0)
    return np.clip(np.round(p * PROB_ONE), 32, PROB_ONE - 32).astype(np.int64)


def estimate_priors(mu16, log_scale8, x, flat_counts: np.ndarray | None = None) -> Priors:
    """Priors from integer predictions on a sample of training data.
    ``flat_counts`` is (FLAT_CONTEXTS, 2): counts of equal / not equal."""
    predicted, k, token_ctx, sign_ctx = plan(mu16, log_scale8)
    tok, e = tokens(x, predicted, k)
    pri = Priors()
    for c in range(TOKEN_CONTEXTS):
        counts = np.bincount(tok[token_ctx == c], minlength=TOKENS)[:TOKENS]
        if counts.sum() > 0:
            pri.token[c] = cumulative(counts)
    nz = e != 0
    pos = np.bincount(sign_ctx[nz & (e > 0)], minlength=SIGN_CONTEXTS)[:SIGN_CONTEXTS]
    tot = np.bincount(sign_ctx[nz], minlength=SIGN_CONTEXTS)[:SIGN_CONTEXTS]
    pri.sign = probability(pos, tot)
    if flat_counts is not None:
        # Bit zero means "equal to the left neighbour".
        pri.flat = probability(flat_counts[:, 0], flat_counts.sum(axis=1))
    return pri


def network_bytes(net: IntegerNet) -> bytes:
    out = bytearray([len(net.layers)])
    for i, (w, b, shift) in enumerate(net.layers):
        last = i + 1 == len(net.layers)
        n_out, n_in = w.shape
        max_in = 128 if i == 0 else 127
        bound = np.abs(b.astype(np.int64)) + np.abs(w.astype(np.int64)).sum(axis=1) * max_in
        if bound.max() > ACC_BOUND:
            raise ValueError(f"layer {i}: accumulator bound {bound.max()} exceeds 2^30")
        out += struct.pack("<HHBB", n_in, n_out, 0 if last else shift, 0 if last else 1)
        out += w.astype(np.int8).tobytes()
        out += b.astype("<i4").tobytes()
    return bytes(out)


def model_bytes(kind: int, name: str, out_shift: int, net: IntegerNet, priors: Priors) -> bytes:
    if kind not in (2, 3):
        raise ValueError("kind is 2 (2-D) or 3 (3-D)")
    raw = name.encode()
    if len(raw) > 64:
        raise ValueError("model name longer than 64 bytes")
    expected = F_.INPUTS_2D if kind == 2 else F_.INPUTS_3D
    if net.layers[0][0].shape[1] != expected or net.layers[-1][0].shape[0] != F_.OUTPUTS:
        raise ValueError("network shape does not match the model kind")
    out = bytearray(b"TZM1")
    out += struct.pack("<BBBBB", kind, out_shift, 0, 0, len(raw)) + raw
    out += network_bytes(net)
    for arr in (priors.token.reshape(-1), priors.sign, priors.flat):
        out += np.asarray(arr, dtype="<u2").tobytes()
    return bytes(out)


def model_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:32]


def read_model(data: bytes) -> tuple[int, str, int, IntegerNet, Priors]:
    """Parses a TZM1 file (used by tests and the golden vector generator)."""
    if data[:4] != b"TZM1":
        raise ValueError("not a TZM1 model")
    kind, out_shift, _, _, name_len = struct.unpack_from("<BBBBB", data, 4)
    pos = 9
    name = data[pos : pos + name_len].decode()
    pos += name_len
    n_layers = data[pos]
    pos += 1
    layers = []
    for _ in range(n_layers):
        n_in, n_out, shift, _act = struct.unpack_from("<HHBB", data, pos)
        pos += 6
        w = np.frombuffer(data, dtype=np.int8, count=n_in * n_out, offset=pos).reshape(n_out, n_in).copy()
        pos += n_in * n_out
        b = np.frombuffer(data, dtype="<i4", count=n_out, offset=pos).astype(np.int32)
        pos += 4 * n_out
        layers.append((w, b, shift))
    rest = np.frombuffer(data, dtype="<u2", offset=pos).astype(np.int64)
    pri = Priors(
        token=rest[: TOKEN_CONTEXTS * TOKENS].reshape(TOKEN_CONTEXTS, TOKENS),
        sign=rest[TOKEN_CONTEXTS * TOKENS : TOKEN_CONTEXTS * TOKENS + SIGN_CONTEXTS],
        flat=rest[TOKEN_CONTEXTS * TOKENS + SIGN_CONTEXTS :],
    )
    return kind, name, out_shift, IntegerNet(layers), pri
