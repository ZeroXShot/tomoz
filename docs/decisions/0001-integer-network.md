# 0001 — The predictor runs on integers only

## Context

An entropy decoder must reproduce the encoder's probability for every
symbol exactly; one differing bit desynchronises everything after it. The
predictor runs once per sample, and its output feeds those probabilities.
Archives are decoded years after they are written, on other hardware, and in
browsers (WebAssembly). Floating point gives no such guarantee: compilers
contract `a*b+c` into FMA or not, SIMD reductions reorder additions, `exp`
and `log` differ between libraries, and WebAssembly engines differ again.

## Decision

The network is an integer MLP: 8-bit weights and activations, 32-bit
accumulators, hidden layers requantised with a rounding shift and clamped to
`[0, 127]`, a linear output layer returning accumulators. Everything around
it (context features, head, residual coding) is integer arithmetic with
explicit rounding. Models are trained in floating point, then fine-tuned with
quantisation-aware training using power-of-two scales so that every rescale
is a shift.

The model loader computes, for every output of every layer, the largest
accumulator any input could produce, and refuses models where it exceeds
2³⁰. Additions of integers that cannot overflow are associative, so every
SIMD kernel (NEON with 16-bit multiplies or the dot-product extension, AVX2,
WebAssembly SIMD) may order them as it likes and still compute exactly the
scalar result.

## Alternatives

- *Floating point with a fixed operation order* (no FMA, scalar loops): slow,
  and still at the mercy of compilers and engines; WebAssembly guarantees
  IEEE results for basic operations but transcendental functions are not part
  of it.
- *16-bit fixed point*: easier training, but half the SIMD throughput.
- *Lookup tables instead of a network*: exact and fast, but the context is
  too large to tabulate; quantising it coarsely loses most of the gain.

## Consequences

- Determinism is testable: conformance hashes pin the exact container bytes
  for fixed inputs, and CI checks them on x86-64, AArch64, macOS, Windows and
  WebAssembly, with the best and the scalar kernel.
- Quantisation costs about 1 % of compressed size compared with the float
  model (TZ1 3-D model on validation data: 5.156 → 5.207 bits per sample).
- The Python training code must implement the same integer computation;
  golden vectors check prediction by prediction that it does.
