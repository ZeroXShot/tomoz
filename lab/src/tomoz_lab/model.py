"""The TZ1 predictor network, its training objective and quantisation.

The network maps the 8-bit context features of a sample to six numbers:
the offset of the predicted mean from the reference prediction, three
coefficients that weigh the left neighbours, a log-scale and the dependence
of that log-scale on the left neighbour (see ``features.head``). It is
trained to minimise the code length of the data: the negative log2
likelihood of each sample under a discretised logistic distribution.

Training runs in two phases. The float phase learns the function; the
quantisation-aware phase fixes power-of-two scales for every layer and
fine-tunes with fake quantisation (rounding and clamping in the forward
pass, straight-through gradients), so that the exported integer network
computes what was trained. ``IntegerNet`` evaluates the exported network
with exactly the arithmetic of the Rust runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from . import features as F_

LN2 = math.log(2.0)


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round half up in the forward pass, identity gradient."""
    return x + (torch.floor(x + 0.5) - x).detach()


def clamp_ste(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Clamp whose gradient is one inside the range and zero outside."""
    return torch.clamp(x, lo, hi)


class TZ1Net(nn.Module):
    """Multilayer perceptron with clipped ReLU hidden layers."""

    def __init__(self, n_in: int, hidden: tuple[int, ...] = (48, 48), n_out: int = F_.OUTPUTS):
        super().__init__()
        widths = (n_in, *hidden, n_out)
        self.layers = nn.ModuleList(nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1))
        last = self.layers[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        # Start from "predict the reference with the activity as scale".
        self.quant: QuantSpec | None = None
        self.act_max = [6.0] * (len(widths) - 2)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """``q``: (B, n_in) integer-valued inputs. Returns the six head
        parameters c (float, in units of the head)."""
        if self.quant is not None:
            return self.forward_quant(q)
        x = q / 16.0
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers):
                x = torch.clamp(x, 0.0, self.act_max[i])
        return x

    def forward_quant(self, q: torch.Tensor) -> torch.Tensor:
        spec = self.quant
        assert spec is not None
        a = q
        for i, layer in enumerate(self.layers):
            w = clamp_ste(round_ste(layer.weight * 2.0 ** spec.weight_exp[i]), -127, 127)
            b = round_ste(layer.bias / 2.0 ** -spec.acc_exp[i])
            acc = a @ w.t() + b
            if i + 1 < len(self.layers):
                a = clamp_ste(round_ste(acc / 2.0 ** spec.shifts[i]), 0, 127)
            else:
                return acc / 2.0**spec.out_shift
        raise AssertionError("unreachable")


@dataclass
class QuantSpec:
    """Power-of-two scales: weights are ``round(w * 2**weight_exp)``, the
    accumulator of layer i has scale ``2**-acc_exp[i]``, hidden layers shift
    right by ``shifts[i]``, and outputs are divided by ``2**out_shift``."""

    weight_exp: list[int]
    acc_exp: list[int]
    shifts: list[int]
    out_shift: int


def choose_quantization(net: TZ1Net, sample_q: torch.Tensor) -> QuantSpec:
    """Picks scales from the float weights and the activations on a sample."""
    weight_exp, acc_exp, shifts = [], [], []
    in_exp = 4  # inputs are q = 16 * x
    x = sample_q / 16.0
    with torch.no_grad():
        for i, layer in enumerate(net.layers):
            wmax = float(layer.weight.abs().max()) or 1e-3
            we = math.floor(math.log2(127.0 / wmax))
            weight_exp.append(we)
            acc_exp.append(we + in_exp)
            x = layer(x)
            if i + 1 < len(net.layers):
                x = torch.clamp(x, 0.0, net.act_max[i])
                # Activation scale: 127 steps should cover the clip value.
                hi = max(float(torch.quantile(x.flatten()[:200_000], 0.9999)), 1e-3)
                act_exp = math.floor(math.log2(127.0 / hi))
                shift = we + in_exp - act_exp
                if shift < 0:
                    act_exp += shift
                    shift = 0
                shifts.append(shift)
                in_exp = act_exp
    return QuantSpec(weight_exp=weight_exp, acc_exp=acc_exp, shifts=shifts, out_shift=acc_exp[-1])


def head_float(c: torch.Tensor, r: torch.Tensor, a4: torch.Tensor, near: torch.Tensor):
    """Float version of ``features.head``: returns (mu, log2 scale)."""
    a = a4 / 4.0
    d = near / 256.0
    mu = r + 4.0 * a * (c[:, 0] + c[:, 1] * d[:, 0] + c[:, 2] * d[:, 1] + c[:, 3] * d[:, 2])
    log2s = torch.log2(a) + c[:, 4] + c[:, 5] * torch.log2(1.0 + d[:, 0].abs())
    return mu, log2s


def code_length(
    mu: torch.Tensor, log2s: torch.Tensor, x: torch.Tensor, max_value: torch.Tensor
) -> torch.Tensor:
    """Bits to code integer ``x`` under a discretised logistic distribution;
    the end bins absorb the tails."""
    s = torch.exp2(log2s.clamp(-6.0, 16.0))
    up = torch.where(x >= max_value, torch.ones_like(x), torch.sigmoid((x + 0.5 - mu) / s))
    lo = torch.where(x <= 0, torch.zeros_like(x), torch.sigmoid((x - 0.5 - mu) / s))
    return -torch.log2((up - lo).clamp_min(1e-10))


class IntegerNet:
    """The exported network evaluated with the arithmetic of the Rust runtime."""

    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray, int]]):
        # (weights int8 (out, in), bias int32 (out,), shift)
        self.layers = layers

    def __call__(self, q: np.ndarray) -> np.ndarray:
        a = q.astype(np.int64)
        for i, (w, b, shift) in enumerate(self.layers):
            acc = a @ w.astype(np.int64).T + b.astype(np.int64)
            if i + 1 < len(self.layers):
                acc = acc if shift == 0 else (acc + (1 << (shift - 1))) >> shift
                a = np.clip(acc, 0, 127)
            else:
                return acc
        raise AssertionError("unreachable")


def export_integer(net: TZ1Net) -> tuple[IntegerNet, int]:
    """Integer weights of a network in its quantised phase."""
    spec = net.quant
    assert spec is not None, "quantise the network first"
    layers = []
    with torch.no_grad():
        for i, layer in enumerate(net.layers):
            w = torch.clamp(torch.floor(layer.weight * 2.0 ** spec.weight_exp[i] + 0.5), -127, 127)
            b = torch.floor(layer.bias / 2.0 ** -spec.acc_exp[i] + 0.5)
            shift = spec.shifts[i] if i + 1 < len(net.layers) else 0
            layers.append((w.numpy().astype(np.int8), b.numpy().astype(np.int64).astype(np.int32), shift))
    return IntegerNet(layers), spec.out_shift
