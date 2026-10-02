"""Classical lossless codecs used as baselines, through imagecodecs.

Each slice is coded as a separate image, as these codecs are used in DICOM
(one compressed frame per image). Values are shifted to start at zero, the
form these codecs expect for unsigned samples; the shift is free information
(a DICOM file records it as the pixel padding or rescale), so it does not
favour or penalise any codec.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class CodecResult:
    bytes: int
    encode_s: float
    decode_s: float
    lossless: bool


def _codecs(bits: int) -> dict[str, tuple[Callable, Callable]]:
    import imagecodecs as ic

    return {
        "jpeg-ls": (lambda s: ic.jpegls_encode(s, level=0), ic.jpegls_decode),
        "jpeg-2000": (
            lambda s: ic.jpeg2k_encode(s, level=0, reversible=True, bitspersample=bits, numthreads=1),
            lambda b: ic.jpeg2k_decode(b, numthreads=1),
        ),
        "htj2k": (lambda s: ic.htj2k_encode(s, reversible=True), ic.htj2k_decode),
        "jpeg-xl-e3": (
            lambda s: ic.jpegxl_encode(s, lossless=True, effort=3, bitspersample=bits, numthreads=1),
            lambda b: ic.jpegxl_decode(b, numthreads=1),
        ),
        "jpeg-xl-e7": (
            lambda s: ic.jpegxl_encode(s, lossless=True, effort=7, bitspersample=bits, numthreads=1),
            lambda b: ic.jpegxl_decode(b, numthreads=1),
        ),
    }


def names() -> list[str]:
    return [*_codecs(16).keys(), "zstd-19"]


def run(volume: np.ndarray) -> dict[str, CodecResult | None]:
    """Codes ``volume`` (Z, H, W) with every baseline; None where a codec
    rejects the input."""
    import imagecodecs as ic

    v = volume.astype(np.int64)
    u = v - v.min()
    bits = max(1, int(u.max()).bit_length())
    u = u.astype(np.uint8 if bits <= 8 else np.uint16)
    out: dict[str, CodecResult | None] = {}
    for name, (enc, dec) in _codecs(bits).items():
        try:
            t0 = time.perf_counter()
            streams = [enc(np.ascontiguousarray(s)) for s in u]
            t1 = time.perf_counter()
            back = [dec(b) for b in streams]
            t2 = time.perf_counter()
            ok = all(np.array_equal(np.asarray(b).reshape(s.shape), s) for b, s in zip(back, u, strict=True))
            out[name] = CodecResult(sum(len(b) for b in streams), t1 - t0, t2 - t1, ok)
        except Exception:
            out[name] = None
    raw = u.tobytes()
    t0 = time.perf_counter()
    z = ic.zstd_encode(raw, level=19)
    t1 = time.perf_counter()
    back = ic.zstd_decode(z)
    t2 = time.perf_counter()
    out["zstd-19"] = CodecResult(len(z), t1 - t0, t2 - t1, back == raw)
    return out
