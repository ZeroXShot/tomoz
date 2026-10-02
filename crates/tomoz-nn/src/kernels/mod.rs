//! Layer kernels and their dispatch.
//!
//! Every kernel computes exactly the same integers as [`scalar`]; the SIMD
//! kernels only change the order of additions, which is exact for integers
//! that cannot overflow (guaranteed by [`crate::Network::new`]).

use crate::{Layer, padded};

mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm;

/// An implementation of the layer arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kernel {
    /// Portable reference.
    Scalar,
    /// AArch64 Advanced SIMD with 16-bit multiply-accumulate.
    Neon,
    /// AArch64 Advanced SIMD with the 8-bit dot-product extension.
    NeonDot,
    /// x86-64 AVX2.
    Avx2,
    /// WebAssembly 128-bit SIMD.
    Wasm128,
}

impl Kernel {
    /// The fastest kernel supported by this machine, unless the environment
    /// variable `TOMOZ_KERNEL` names another available one (see
    /// [`Kernel::name`]). Every kernel computes the same results; forcing one
    /// is for tests, benchmarks and diagnosis.
    #[must_use]
    pub fn detect() -> Self {
        let forced = std::env::var("TOMOZ_KERNEL").ok().and_then(|n| Self::from_name(&n));
        if let Some(k) = forced.filter(|k| k.is_available()) {
            return k;
        }
        [Self::NeonDot, Self::Neon, Self::Avx2, Self::Wasm128]
            .into_iter()
            .find(|k| k.is_available())
            .unwrap_or(Self::Scalar)
    }

    /// Lower-case name: `scalar`, `neon`, `neon-dot`, `avx2` or `wasm128`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Neon => "neon",
            Self::NeonDot => "neon-dot",
            Self::Avx2 => "avx2",
            Self::Wasm128 => "wasm128",
        }
    }

    /// The kernel called `name` (see [`Kernel::name`]).
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        [Self::Scalar, Self::Neon, Self::NeonDot, Self::Avx2, Self::Wasm128].into_iter().find(|k| k.name() == name)
    }

    /// Whether this machine can run the kernel.
    #[must_use]
    pub fn is_available(self) -> bool {
        match self {
            Self::Scalar => true,
            #[cfg(target_arch = "aarch64")]
            Self::Neon => true,
            #[cfg(target_arch = "aarch64")]
            Self::NeonDot => std::arch::is_aarch64_feature_detected!("dotprod"),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => std::arch::is_x86_feature_detected!("avx2"),
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Self::Wasm128 => true,
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    /// Every kernel this machine can run, the scalar reference first.
    #[must_use]
    pub fn available() -> Vec<Self> {
        [Self::Scalar, Self::Neon, Self::NeonDot, Self::Avx2, Self::Wasm128]
            .into_iter()
            .filter(|k| k.is_available())
            .collect()
    }
}

/// Weights of one layer in the layout a kernel reads.
#[derive(Clone, Debug)]
pub(crate) struct Packed {
    /// Inputs rounded up to a multiple of 16.
    pub in_pad: usize,
    /// Outputs rounded up to a multiple of 16.
    pub out_pad: usize,
    /// Biases, `out_pad` of them (zero beyond the real outputs).
    pub bias: Vec<i32>,
    /// Row-major `out_pad × in_pad` weights (scalar kernel), or 4×4 blocks
    /// (`NeonDot`).
    pub w8: Vec<i8>,
    /// Interleaved 16-bit weights (`Neon`, `Avx2`, `Wasm128`).
    pub w16: Vec<i16>,
}

impl Packed {
    pub(crate) fn new(layer: &Layer, kernel: Kernel) -> Self {
        let in_pad = padded(layer.inputs);
        let out_pad = padded(layer.outputs);
        let w = |o: usize, i: usize| -> i8 {
            if o < layer.outputs && i < layer.inputs { layer.weights[o * layer.inputs + i] } else { 0 }
        };
        let mut bias = layer.bias.clone();
        bias.resize(out_pad, 0);
        let mut p = Self { in_pad, out_pad, bias, w8: Vec::new(), w16: Vec::new() };
        match kernel {
            Kernel::Scalar => {
                p.w8 = (0..out_pad).flat_map(|o| (0..in_pad).map(move |i| (o, i))).map(|(o, i)| w(o, i)).collect();
            }
            Kernel::NeonDot => {
                // [output block of 4][input group of 4][4 outputs × 4 inputs]
                for o4 in (0..out_pad).step_by(4) {
                    for g in (0..in_pad).step_by(4) {
                        for o in o4..o4 + 4 {
                            for i in g..g + 4 {
                                p.w8.push(w(o, i));
                            }
                        }
                    }
                }
            }
            Kernel::Neon => {
                // [output block of 4][input][4 outputs]
                for o4 in (0..out_pad).step_by(4) {
                    for i in 0..in_pad {
                        for o in o4..o4 + 4 {
                            p.w16.push(i16::from(w(o, i)));
                        }
                    }
                }
            }
            Kernel::Avx2 | Kernel::Wasm128 => {
                // [output block of B][input pair][B outputs × 2 inputs]
                let b = if kernel == Kernel::Avx2 { 8 } else { 4 };
                for ob in (0..out_pad).step_by(b) {
                    for i in (0..in_pad).step_by(2) {
                        for o in ob..ob + b {
                            p.w16.push(i16::from(w(o, i)));
                            p.w16.push(i16::from(w(o, i + 1)));
                        }
                    }
                }
            }
        }
        p
    }
}

/// Hidden layer: `dst` receives `batch` rows of `out_pad` activations.
#[allow(unsafe_code)]
pub(crate) fn hidden(kernel: Kernel, p: &Packed, layer: &Layer, src: &[i8], batch: usize, dst: &mut [i8]) {
    debug_assert!(src.len() >= batch * p.in_pad && dst.len() >= batch * p.out_pad);
    match kernel {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: the kernel is only selected when the CPU supports it.
        Kernel::NeonDot => unsafe { neon::hidden_dot(p, layer.shift, src, batch, dst) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: Advanced SIMD is part of the AArch64 baseline.
        Kernel::Neon => unsafe { neon::hidden(p, layer.shift, src, batch, dst) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the kernel is only selected when the CPU supports AVX2.
        Kernel::Avx2 => unsafe { avx2::hidden(p, layer.shift, src, batch, dst) },
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Kernel::Wasm128 => wasm::hidden(p, layer.shift, src, batch, dst),
        _ => scalar::hidden(p, layer.shift, src, batch, dst),
    }
}

/// Output layer: `out` receives `batch` rows of `layer.outputs` accumulators.
#[allow(unsafe_code)]
pub(crate) fn linear(kernel: Kernel, p: &Packed, layer: &Layer, src: &[i8], batch: usize, out: &mut [i32]) {
    debug_assert!(src.len() >= batch * p.in_pad && out.len() >= batch * layer.outputs);
    let n = layer.outputs;
    match kernel {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: the kernel is only selected when the CPU supports it.
        Kernel::NeonDot => unsafe { neon::linear_dot(p, n, src, batch, out) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: Advanced SIMD is part of the AArch64 baseline.
        Kernel::Neon => unsafe { neon::linear(p, n, src, batch, out) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the kernel is only selected when the CPU supports AVX2.
        Kernel::Avx2 => unsafe { avx2::linear(p, n, src, batch, out) },
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Kernel::Wasm128 => wasm::linear(p, n, src, batch, out),
        _ => scalar::linear(p, n, src, batch, out),
    }
}
