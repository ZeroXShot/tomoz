//! Integer neural network runtime.
//!
//! Tomoz predicts every voxel with a small multilayer perceptron, and the
//! decoder must reproduce the encoder's predictions exactly: a single
//! differing bit anywhere desynchronises the entropy decoder. Floating point
//! cannot promise that across instruction sets, compilers and WebAssembly
//! engines, so the network runs entirely on integers:
//!
//! * weights and activations are 8-bit, accumulators 32-bit;
//! * hidden layers requantise with a rounding right shift and clamp to
//!   `[0, 127]` (a clipped ReLU);
//! * the output layer returns raw 32-bit accumulators.
//!
//! [`Network::new`] checks that no accumulator can overflow for any input, so
//! every kernel computes the same function. Kernels for AArch64 (with and
//! without the dot-product extension), x86-64 AVX2 and WebAssembly SIMD are
//! selected at run time and tested against the scalar reference.
//!
//! The binary format of a network ([`Network::from_bytes`]) is part of the
//! Tomoz model format and is stable.

mod kernels;

pub use kernels::Kernel;

/// Largest number of inputs or outputs of a layer.
pub const MAX_WIDTH: usize = 256;

/// Largest number of layers.
pub const MAX_LAYERS: usize = 8;

/// Inputs and hidden widths are padded to a multiple of this.
pub const LANES: usize = 16;

/// Bound on the magnitude of any accumulator, checked when a network is
/// built. Leaves room for the rounding term of the requantisation.
const ACC_BOUND: i64 = 1 << 30;

/// Errors raised when a network is built or decoded.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The network has no layers or too many.
    #[error("a network needs 1 to {MAX_LAYERS} layers, got {0}")]
    LayerCount(usize),
    /// A layer is too wide or empty.
    #[error("layer {layer}: width {width} outside 1..={MAX_WIDTH}")]
    Width {
        /// Index of the layer.
        layer: usize,
        /// Offending width.
        width: usize,
    },
    /// Consecutive layers do not connect.
    #[error("layer {layer} expects {expected} inputs but the previous layer has {actual} outputs")]
    Shape {
        /// Index of the layer.
        layer: usize,
        /// Inputs of the layer.
        expected: usize,
        /// Outputs of the previous layer.
        actual: usize,
    },
    /// Weight or bias arrays have the wrong length.
    #[error("layer {0}: weight or bias count does not match its shape")]
    Parameters(usize),
    /// Hidden layers must use the clipped ReLU, the output layer none.
    #[error("layer {0}: invalid activation for its position")]
    Activation(usize),
    /// A requantisation shift is out of range.
    #[error("layer {layer}: shift {shift} outside 0..=24")]
    Shift {
        /// Index of the layer.
        layer: usize,
        /// Offending shift.
        shift: u8,
    },
    /// Some input could overflow a 32-bit accumulator.
    #[error("layer {0}: accumulator could exceed 2^30")]
    Overflow(usize),
    /// The serialised network is malformed.
    #[error("malformed network encoding: {0}")]
    Encoding(&'static str),
}

/// Activation of a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    /// Output layer: the accumulators are returned as they are.
    Linear,
    /// Hidden layer: `clamp((acc + 2^(shift-1)) >> shift, 0, 127)`.
    ClippedRelu,
}

/// A dense layer in its portable, row-major form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layer {
    /// Number of inputs.
    pub inputs: usize,
    /// Number of outputs.
    pub outputs: usize,
    /// `outputs × inputs` weights, row by row.
    pub weights: Vec<i8>,
    /// One bias per output.
    pub bias: Vec<i32>,
    /// Requantisation shift (hidden layers).
    pub shift: u8,
    /// Activation.
    pub activation: Activation,
}

/// A validated network, with weights repacked for the selected kernel.
#[derive(Clone, Debug)]
pub struct Network {
    layers: Vec<Layer>,
    packed: Vec<kernels::Packed>,
    kernel: Kernel,
}

/// Reusable buffers for [`Network::forward`].
#[derive(Clone, Debug, Default)]
pub struct Scratch {
    a: Vec<i8>,
    b: Vec<i8>,
}

/// Rounds `n` up to a multiple of [`LANES`].
#[must_use]
pub const fn padded(n: usize) -> usize {
    n.div_ceil(LANES) * LANES
}

impl Network {
    /// Validates `layers` and prepares them for the best kernel available on
    /// this machine.
    ///
    /// # Errors
    ///
    /// Any [`Error`] describing why the layers do not form a valid network,
    /// including the possibility of accumulator overflow.
    pub fn new(layers: Vec<Layer>) -> Result<Self, Error> {
        Self::with_kernel(layers, Kernel::detect())
    }

    /// Like [`Network::new`] with an explicit kernel; unsupported kernels fall
    /// back to [`Kernel::Scalar`].
    ///
    /// # Errors
    ///
    /// As [`Network::new`].
    pub fn with_kernel(layers: Vec<Layer>, kernel: Kernel) -> Result<Self, Error> {
        validate(&layers)?;
        let kernel = if kernel.is_available() { kernel } else { Kernel::Scalar };
        let packed = layers.iter().map(|l| kernels::Packed::new(l, kernel)).collect();
        Ok(Self { layers, packed, kernel })
    }

    /// The kernel in use.
    #[must_use]
    pub fn kernel(&self) -> Kernel {
        self.kernel
    }

    /// The layers in portable form.
    #[must_use]
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Number of inputs.
    #[must_use]
    pub fn inputs(&self) -> usize {
        self.layers[0].inputs
    }

    /// Distance between consecutive samples in the input buffer of
    /// [`Network::forward`]: the number of inputs rounded up to [`LANES`].
    #[must_use]
    pub fn input_stride(&self) -> usize {
        padded(self.inputs())
    }

    /// Number of outputs.
    #[must_use]
    pub fn outputs(&self) -> usize {
        self.layers[self.layers.len() - 1].outputs
    }

    /// Evaluates the network on `batch` samples.
    ///
    /// `inputs` holds `batch` rows of [`Network::input_stride`] values; the
    /// padding at the end of each row must be zero. `out` receives `batch`
    /// rows of [`Network::outputs`] accumulators.
    ///
    /// # Panics
    ///
    /// Panics if the buffers are shorter than `batch` rows.
    pub fn forward(&self, inputs: &[i8], batch: usize, scratch: &mut Scratch, out: &mut [i32]) {
        let stride = self.input_stride();
        assert!(inputs.len() >= batch * stride, "input buffer too short");
        assert!(out.len() >= batch * self.outputs(), "output buffer too short");
        let n = self.layers.len();
        let Scratch { a, b } = scratch;
        if n == 1 {
            kernels::linear(self.kernel, &self.packed[0], &self.layers[0], inputs, batch, out);
            return;
        }
        Self::hidden_into(self.kernel, &self.packed[0], &self.layers[0], inputs, batch, a);
        for i in 1..n - 1 {
            Self::hidden_into(self.kernel, &self.packed[i], &self.layers[i], a, batch, b);
            core::mem::swap(a, b);
        }
        kernels::linear(self.kernel, &self.packed[n - 1], &self.layers[n - 1], a, batch, out);
    }

    fn hidden_into(
        kernel: Kernel,
        packed: &kernels::Packed,
        layer: &Layer,
        src: &[i8],
        batch: usize,
        dst: &mut Vec<i8>,
    ) {
        // Kernels write every lane, padding included: no need to clear.
        let n = batch * padded(layer.outputs);
        if dst.len() < n {
            dst.resize(n, 0);
        }
        kernels::hidden(kernel, packed, layer, src, batch, &mut dst[..n]);
    }

    /// Decodes a network from the start of `bytes`; returns it with the
    /// number of bytes read.
    ///
    /// The encoding is: layer count (`u8`), then per layer: inputs (`u16`),
    /// outputs (`u16`), shift (`u8`), activation (`u8`: 0 linear, 1 clipped
    /// ReLU), `outputs × inputs` weights (`i8`), `outputs` biases (`i32`). All
    /// integers are little endian.
    ///
    /// # Errors
    ///
    /// [`Error::Encoding`] for truncated or inconsistent data, or any
    /// validation error.
    pub fn from_bytes(bytes: &[u8]) -> Result<(Self, usize), Error> {
        let mut r = Reader { bytes, pos: 0 };
        let count = usize::from(r.u8()?);
        if count == 0 || count > MAX_LAYERS {
            return Err(Error::LayerCount(count));
        }
        let mut layers = Vec::with_capacity(count);
        for _ in 0..count {
            let inputs = usize::from(r.u16()?);
            let outputs = usize::from(r.u16()?);
            if inputs == 0 || inputs > MAX_WIDTH || outputs == 0 || outputs > MAX_WIDTH {
                return Err(Error::Encoding("layer width out of range"));
            }
            let shift = r.u8()?;
            let activation = match r.u8()? {
                0 => Activation::Linear,
                1 => Activation::ClippedRelu,
                _ => return Err(Error::Encoding("unknown activation")),
            };
            let weights = r.take(inputs * outputs)?.iter().map(|&b| b as i8).collect();
            let bias = (0..outputs).map(|_| r.i32()).collect::<Result<_, _>>()?;
            layers.push(Layer { inputs, outputs, weights, bias, shift, activation });
        }
        Ok((Self::new(layers)?, r.pos))
    }

    /// Encodes the network in the format read by [`Network::from_bytes`].
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = vec![self.layers.len() as u8];
        for l in &self.layers {
            out.extend_from_slice(&(l.inputs as u16).to_le_bytes());
            out.extend_from_slice(&(l.outputs as u16).to_le_bytes());
            out.push(l.shift);
            out.push(match l.activation {
                Activation::Linear => 0,
                Activation::ClippedRelu => 1,
            });
            out.extend(l.weights.iter().map(|&w| w as u8));
            for b in &l.bias {
                out.extend_from_slice(&b.to_le_bytes());
            }
        }
        out
    }
}

fn validate(layers: &[Layer]) -> Result<(), Error> {
    if layers.is_empty() || layers.len() > MAX_LAYERS {
        return Err(Error::LayerCount(layers.len()));
    }
    for (i, l) in layers.iter().enumerate() {
        for width in [l.inputs, l.outputs] {
            if width == 0 || width > MAX_WIDTH {
                return Err(Error::Width { layer: i, width });
            }
        }
        if i > 0 && l.inputs != layers[i - 1].outputs {
            return Err(Error::Shape { layer: i, expected: l.inputs, actual: layers[i - 1].outputs });
        }
        if l.weights.len() != l.inputs * l.outputs || l.bias.len() != l.outputs {
            return Err(Error::Parameters(i));
        }
        let last = i + 1 == layers.len();
        if (l.activation == Activation::Linear) != last {
            return Err(Error::Activation(i));
        }
        if l.shift > 24 {
            return Err(Error::Shift { layer: i, shift: l.shift });
        }
        // Inputs of the first layer span [-128, 127]; hidden activations
        // [0, 127].
        let max_input: i64 = if i == 0 { 128 } else { 127 };
        for (o, row) in l.weights.chunks(l.inputs).enumerate() {
            let bound = i64::from(l.bias[o]).abs() + row.iter().map(|&w| i64::from(w).abs() * max_input).sum::<i64>();
            if bound > ACC_BOUND {
                return Err(Error::Overflow(i));
            }
        }
    }
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let s = self.bytes.get(self.pos..self.pos + n).ok_or(Error::Encoding("truncated"))?;
        self.pos += n;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, Error> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn i32(&mut self) -> Result<i32, Error> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

/// Requantises a hidden accumulator.
#[inline]
#[must_use]
pub fn requantize(acc: i32, shift: u8) -> i8 {
    let v = if shift == 0 { acc } else { (acc + (1 << (shift - 1))) >> shift };
    v.clamp(0, 127) as i8
}
