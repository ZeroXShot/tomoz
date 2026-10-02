"""Golden vectors: predictions computed by the Python reference that the Rust
codec must reproduce exactly (``crates/tomoz-codec/tests/golden.rs``).

The vectors use random networks rather than the released models, so that
they exercise every term of the head with large weights and do not change
when models are retrained.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import features as F_
from .export import Priors, model_bytes
from .model import IntegerNet


def random_net(n_in: int, hidden: tuple[int, ...], rng: np.random.Generator) -> IntegerNet:
    widths = (n_in, *hidden, F_.OUTPUTS)
    layers = []
    for i in range(len(widths) - 1):
        last = i + 2 == len(widths)
        w = rng.integers(-60, 61, size=(widths[i + 1], widths[i])).astype(np.int8)
        b = rng.integers(-3000, 3001, size=widths[i + 1]).astype(np.int32)
        layers.append((w, b, 0 if last else 9))
    return IntegerNet(layers)


def predictions(vol: np.ndarray, nets: dict[int, tuple[IntegerNet, int]]) -> list[list[int]]:
    """Per-sample [z, y, x, flat_equal, mu16, log_scale8] for one tile."""
    vol = vol.astype(np.int64)
    max_value = int(vol.max())
    t0 = int(vol[0, 0, 0])
    out = []
    for z in range(vol.shape[0]):
        three_d = z > 0
        net, out_shift = nets[3 if three_d else 2]
        f = F_.slice_features(vol[z], t0, vol[z - 1] if three_d else None, vol[z - 2] if z >= 2 else None)
        o = net(f.inputs.reshape(-1, f.inputs.shape[-1])).reshape(*vol[z].shape, F_.OUTPUTS)
        h = F_.head(o, f, out_shift, max_value)
        eq = f.flat & (vol[z] == f.west)
        for y in range(vol.shape[1]):
            for x in range(vol.shape[2]):
                out.append([z, y, x, int(eq[y, x]), int(h.mu16[y, x]), int(h.log_scale8[y, x])])
    return out


def structured_volume(rng: np.random.Generator, shape: tuple[int, int, int], max_value: int) -> np.ndarray:
    z, y, x = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    base = (max_value // 3 + 40 * np.sin(x / 3.0 + z) + 25 * y).astype(np.int64)
    vol = base + rng.integers(-6, 7, size=shape)
    vol[:, :, : shape[2] // 3] = 0  # flat padding region
    vol[:, 3:5, :] = rng.integers(0, max_value + 1, size=(shape[0], 2, shape[2]))  # edges and outliers
    vol[0, 0, 0] = max_value // 2
    return np.clip(vol, 0, max_value)


def write(path: Path, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    cases = []
    for shape, max_value in [((4, 11, 13), 4095), ((3, 9, 17), 255)]:
        nets = {
            2: (random_net(F_.INPUTS_2D, (16,), rng), 11),
            3: (random_net(F_.INPUTS_3D, (32, 16), rng), 13),
        }
        vol = structured_volume(rng, shape, max_value)
        # Make sure the full value range is present so that the codec maps
        # values with offset 0 and no packing.
        vol[-1, -1, -1] = max_value
        vol[-1, -1, -2] = 0
        cases.append(
            {
                "shape": list(shape),
                "bits": int(max_value).bit_length(),
                "samples": vol.reshape(-1).tolist(),
                "model_2d": model_bytes(2, "golden-2d", nets[2][1], nets[2][0], Priors()).hex(),
                "model_3d": model_bytes(3, "golden-3d", nets[3][1], nets[3][0], Priors()).hex(),
                "predictions": predictions(vol, nets),
            }
        )
    path.write_text(json.dumps({"cases": cases}) + "\n")
