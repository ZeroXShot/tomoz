"""Training of the TZ1 predictors.

Samples are drawn from volumes exactly as the codec sees them: each training
volume is one tile; slice 0 is coded with the 2-D model and the following
slices with the 3-D model. Samples coded by the flat-region flag alone are
left out, since the network never sees them.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import features as F_
from .model import IntegerNet, TZ1Net, choose_quantization, code_length, export_integer, head_float

log = logging.getLogger(__name__)


@dataclass
class Samples:
    """Training samples of one model kind (arrays of equal length)."""

    inputs: np.ndarray  # (n, n_in) int8
    near: np.ndarray  # (n, 3) int32
    r: np.ndarray  # (n,) int64
    a4: np.ndarray  # (n,) int64
    x: np.ndarray  # (n,) int64
    max_value: np.ndarray  # (n,) int64
    volume: np.ndarray  # (n,) int32, index of the source volume

    def __len__(self) -> int:
        return len(self.x)

    def subset(self, idx: np.ndarray) -> Samples:
        return Samples(**{k: v[idx] for k, v in asdict(self).items()})

    @staticmethod
    def concat(parts: list[Samples]) -> Samples:
        return Samples(
            **{k: np.concatenate([getattr(p, k) for p in parts]) for k in Samples.__dataclass_fields__}
        )


def volume_samples(
    vol: np.ndarray,
    three_d: bool,
    per_slice: int,
    rng: np.random.Generator,
    index: int,
    max_slices: int | None = None,
) -> Samples | None:
    """Random samples from the slices of one volume (Z, H, W) of non-negative
    integers, treated as one tile: up to ``per_slice`` per 512 × 512 pixels of
    a slice (at most 40 times as many for large images such as mammograms),
    from at most ``max_slices`` random slices. None if no sample qualifies
    (e.g. a 2-D image for the 3-D model)."""
    vol = vol.astype(np.int64)
    max_value = int(vol.max())
    t0 = int(vol[0, 0, 0])
    parts = []
    zs = list(range(1, vol.shape[0]) if three_d else range(vol.shape[0]))
    if max_slices is not None and len(zs) > max_slices:
        zs = sorted(int(z) for z in rng.choice(zs, size=max_slices, replace=False))
    h, w = vol.shape[1:]
    budget = per_slice * min(40, max(1, (h * w) // (512 * 512)))
    for z in zs:
        prev1 = vol[z - 1] if three_d else None
        prev2 = vol[z - 2] if three_d and z >= 2 else None
        n = min(h * w, int(budget * 1.5))
        flat_idx = rng.choice(h * w, size=n, replace=False)
        ys, xs = flat_idx // w, flat_idx % w
        f = F_.slice_features(vol[z], t0, prev1, prev2, points=(ys, xs))
        x = vol[z][ys, xs]
        keep = np.nonzero(~(f.flat & (x == f.west)))[0][:budget]
        if len(keep) == 0:
            continue
        parts.append(
            Samples(
                inputs=f.inputs[keep],
                near=f.near[keep].astype(np.int32),
                r=f.r[keep],
                a4=f.a4[keep],
                x=x[keep],
                max_value=np.full(len(keep), max_value, dtype=np.int64),
                volume=np.full(len(keep), index, dtype=np.int32),
            )
        )
    return Samples.concat(parts) if parts else None


@dataclass
class TrainConfig:
    hidden: tuple[int, ...] = (48, 48)
    batch: int = 8192
    float_steps: int = 12_000
    quant_steps: int = 4_000
    lr: float = 4e-3
    quant_lr: float = 3e-4
    seed: int = 0
    threads: int = 0
    log_every: int = 1000
    history: list[dict] = field(default_factory=list)


def _tensors(s: Samples, idx: np.ndarray):
    t = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a[idx])).to(dt)  # noqa: E731
    return t(s.inputs), t(s.near), t(s.r), t(s.a4), t(s.x), t(s.max_value)


def train(train_set: Samples, val_set: Samples, n_in: int, cfg: TrainConfig) -> TZ1Net:
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    rng = np.random.default_rng(cfg.seed)
    net = TZ1Net(n_in, cfg.hidden)

    def run(steps: int, lr: float, phase: str) -> None:
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
        t0 = time.time()
        for step in range(steps):
            idx = rng.integers(0, len(train_set), cfg.batch)
            q, near, r, a4, x, mx = _tensors(train_set, idx)
            mu, log2s = head_float(net(q), r, a4, near)
            loss = code_length(mu, log2s, x, mx).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if step % cfg.log_every == 0 or step + 1 == steps:
                entry = {
                    "phase": phase,
                    "step": step,
                    "bits": float(loss.detach()),
                    "seconds": round(time.time() - t0, 1),
                }
                cfg.history.append(entry)
                log.info("%s step %d: %.4f bits/sample (%.0fs)", phase, step, entry["bits"], entry["seconds"])

    run(cfg.float_steps, cfg.lr, "float")
    log.info("float validation: %.4f bits/sample", evaluate_float(net, val_set))
    sample = torch.from_numpy(train_set.inputs[rng.integers(0, len(train_set), 200_000)]).float()
    net.quant = choose_quantization(net, sample)
    log.info("quantisation: %s", net.quant)
    run(cfg.quant_steps, cfg.quant_lr, "quant")
    return net


def evaluate_float(net: TZ1Net, s: Samples, chunk: int = 1 << 16) -> float:
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(s), chunk):
            idx = np.arange(i, min(i + chunk, len(s)))
            q, near, r, a4, x, mx = _tensors(s, idx)
            mu, log2s = head_float(net(q), r, a4, near)
            total += float(code_length(mu, log2s, x, mx).sum())
    return total / max(len(s), 1)


def integer_predictions(inet: IntegerNet, out_shift: int, s: Samples, chunk: int = 1 << 16):
    """Exact integer (mu16, log_scale8) of every sample."""
    mus, lss = [], []
    for i in range(0, len(s), chunk):
        sl = slice(i, min(i + chunk, len(s)))
        out = inet(s.inputs[sl])
        n = len(s.r[sl])
        f = F_.SliceFeatures(
            inputs=s.inputs[sl],
            r=s.r[sl],
            a4=s.a4[sl],
            near=s.near[sl].astype(np.int64),
            flat=np.zeros(n, bool),
            west=np.zeros(n, np.int64),
        )
        h = F_.head(out, f, out_shift, s.max_value[sl])
        mus.append(h.mu16)
        lss.append(h.log_scale8)
    return np.concatenate(mus), np.concatenate(lss)


def evaluate_integer(inet: IntegerNet, out_shift: int, s: Samples) -> float:
    """Bits per sample with the exact integer network and head."""
    if len(s) == 0:
        return 0.0
    mu16, ls8 = integer_predictions(inet, out_shift, s)
    bits = code_length(
        torch.from_numpy(mu16 / 16.0),
        torch.from_numpy(ls8 / 8.0),
        torch.from_numpy(s.x.astype(np.float64)),
        torch.from_numpy(s.max_value.astype(np.float64)),
    )
    return float(bits.mean())


def flat_statistics(vols, three_d: bool, slices_per_volume: int = 8) -> np.ndarray:
    """Counts (4, 2) of flat-region samples equal / not equal to their left
    neighbour, per flag context (2-D/3-D x previous sample flat-equal)."""
    counts = np.zeros((4, 2), dtype=np.int64)
    for _, _, vol in vols:
        if three_d and vol.shape[0] < 2:
            continue
        zs = np.linspace(
            1 if three_d else 0, vol.shape[0] - 1, num=min(slices_per_volume, vol.shape[0]), dtype=int
        )
        t0 = int(vol[0, 0, 0])
        for z in np.unique(zs):
            if three_d and z == 0:
                continue
            f = F_.slice_features(
                vol[z], t0, vol[z - 1] if three_d else None, vol[z - 2] if three_d and z >= 2 else None
            )
            eq = f.flat & (vol[z] == f.west)
            prev = np.zeros_like(eq)
            prev[:, 1:] = eq[:, :-1]
            for p in (0, 1):
                sel = f.flat & (prev == bool(p))
                ctx = 2 * int(three_d) + p
                counts[ctx, 0] += int((eq & sel).sum())
                counts[ctx, 1] += int((~eq & sel).sum())
    return counts


def save_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2) + "\n")


def export_layers(net: TZ1Net) -> tuple[IntegerNet, int]:
    return export_integer(net)
