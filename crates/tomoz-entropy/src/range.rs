//! Range encoder and decoder.

use alloc::vec::Vec;
use core::fmt;

use crate::model::{BitModel, Cdf};
use crate::{PROB_BITS, PROB_ONE};

/// The range is renormalised whenever it drops below this value, which keeps
/// at least 9 bits of resolution for every 15-bit probability.
const TOP: u32 = 1 << 24;

/// Maximum number of bits coded by one bypass operation.
const MAX_BYPASS: u32 = 16;

/// Errors reported when a decoded stream turns out to be malformed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The decoder needed more bytes than the stream contains.
    Truncated {
        /// Number of missing bytes.
        missing: usize,
    },
    /// The stream continues after the point where the decoder stopped.
    TrailingBytes {
        /// Number of bytes that were not consumed.
        unread: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { missing } => write!(f, "entropy-coded stream is truncated ({missing} bytes missing)"),
            Self::TrailingBytes { unread } => write!(f, "entropy-coded stream has {unread} unexpected trailing bytes"),
        }
    }
}

impl core::error::Error for Error {}

/// Range encoder.
///
/// Symbols are coded in the order of the calls; the matching [`Decoder`] must
/// make the same sequence of calls with models in the same state.
#[derive(Clone, Debug)]
pub struct Encoder {
    /// Low end of the interval. Bit 32 holds a carry not yet propagated into
    /// the bytes held back in `cache` and `pending`.
    low: u64,
    range: u32,
    /// Last byte that might still receive a carry.
    cache: u8,
    /// Number of bytes held back: `cache` plus a run of `0xFF` bytes.
    pending: u64,
    out: Vec<u8>,
    /// The very first byte produced by the scheme is always zero and is not
    /// written.
    skip_first: bool,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    /// An encoder with an empty output buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// An encoder whose output buffer has room for `capacity` bytes.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self { low: 0, range: u32::MAX, cache: 0, pending: 1, out: Vec::with_capacity(capacity), skip_first: true }
    }

    /// Number of bytes produced so far (excluding bytes held back for carry
    /// propagation).
    #[must_use]
    pub fn bytes_written(&self) -> usize {
        self.out.len()
    }

    /// Codes the sub-interval `[lo, hi)` of `[0, 2^bits)`; `hi == 2^bits` gives
    /// the last sub-interval the remainder of the range.
    #[inline]
    fn encode_interval(&mut self, lo: u32, hi: u32, bits: u32) {
        debug_assert!(lo < hi && hi <= 1 << bits && bits <= MAX_BYPASS);
        let r = self.range >> bits;
        self.low += u64::from(r * lo);
        self.range = if hi == 1 << bits { self.range - r * lo } else { r * (hi - lo) };
        while self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
    }

    #[inline]
    fn shift_low(&mut self) {
        if (self.low as u32) < 0xFF00_0000 || (self.low >> 32) != 0 {
            let carry = (self.low >> 32) as u8;
            let mut byte = self.cache;
            loop {
                if self.skip_first {
                    self.skip_first = false;
                } else {
                    self.out.push(byte.wrapping_add(carry));
                }
                byte = 0xFF;
                self.pending -= 1;
                if self.pending == 0 {
                    break;
                }
            }
            self.cache = (self.low >> 24) as u8;
        }
        self.pending += 1;
        self.low = (self.low & 0x00FF_FFFF) << 8;
    }

    /// Codes `symbol` with `model` and updates the model.
    ///
    /// # Panics
    ///
    /// Panics if `symbol >= N`.
    #[inline]
    pub fn encode_symbol<const N: usize>(&mut self, model: &mut Cdf<N>, symbol: usize) {
        assert!(symbol < N, "symbol {symbol} out of range for a {N}-symbol model");
        self.encode_interval(model.boundary(symbol), model.boundary(symbol + 1), PROB_BITS);
        model.update(symbol);
    }

    /// Codes `bit` with `model` and updates the model.
    #[inline]
    pub fn encode_bit(&mut self, model: &mut BitModel, bit: bool) {
        let p0 = model.p0();
        if bit {
            self.encode_interval(p0, PROB_ONE, PROB_BITS);
        } else {
            self.encode_interval(0, p0, PROB_BITS);
        }
        model.update(bit);
    }

    /// Codes the low `bits` bits of `value` with a uniform distribution.
    ///
    /// # Panics
    ///
    /// Panics if `bits > 32`.
    #[inline]
    pub fn encode_bypass(&mut self, value: u32, bits: u32) {
        assert!(bits <= 32, "at most 32 bypass bits per call");
        let mut bits = bits;
        while bits > 0 {
            let n = bits.min(MAX_BYPASS);
            bits -= n;
            let v = (value >> bits) & ((1 << n) - 1);
            self.encode_interval(v, v + 1, n);
        }
    }

    /// Flushes the coder state and returns the coded bytes.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        self.out
    }
}

/// Range decoder over a byte slice.
#[derive(Clone, Debug)]
pub struct Decoder<'a> {
    code: u32,
    range: u32,
    input: &'a [u8],
    pos: usize,
    missing: usize,
}

impl<'a> Decoder<'a> {
    /// Starts decoding `input`. Never fails: problems surface in
    /// [`Decoder::finish`].
    #[must_use]
    pub fn new(input: &'a [u8]) -> Self {
        let mut d = Self { code: 0, range: u32::MAX, input, pos: 0, missing: 0 };
        for _ in 0..4 {
            d.code = (d.code << 8) | u32::from(d.next_byte());
        }
        d
    }

    #[inline]
    fn next_byte(&mut self) -> u8 {
        if let Some(&b) = self.input.get(self.pos) {
            self.pos += 1;
            b
        } else {
            self.missing += 1;
            0
        }
    }

    #[inline]
    fn normalize(&mut self) {
        while self.range < TOP {
            self.range <<= 8;
            self.code = (self.code << 8) | u32::from(self.next_byte());
        }
    }

    /// Decodes a symbol coded with `model` and updates the model.
    #[inline]
    pub fn decode_symbol<const N: usize>(&mut self, model: &mut Cdf<N>) -> usize {
        let r = self.range >> PROB_BITS;
        let mut symbol = 0;
        let mut lo = 0;
        let mut hi = model.boundary(1);
        while symbol + 1 < N && r * hi <= self.code {
            symbol += 1;
            lo = hi;
            hi = model.boundary(symbol + 1);
        }
        self.code -= r * lo;
        self.range = if symbol + 1 == N { self.range - r * lo } else { r * (hi - lo) };
        self.normalize();
        model.update(symbol);
        symbol
    }

    /// Decodes a bit coded with `model` and updates the model.
    #[inline]
    pub fn decode_bit(&mut self, model: &mut BitModel) -> bool {
        let bound = (self.range >> PROB_BITS) * model.p0();
        let bit = self.code >= bound;
        if bit {
            self.code -= bound;
            self.range -= bound;
        } else {
            self.range = bound;
        }
        self.normalize();
        model.update(bit);
        bit
    }

    /// Decodes `bits` bypass bits.
    ///
    /// # Panics
    ///
    /// Panics if `bits > 32`.
    #[inline]
    pub fn decode_bypass(&mut self, bits: u32) -> u32 {
        assert!(bits <= 32, "at most 32 bypass bits per call");
        let mut bits = bits;
        let mut value = 0u32;
        while bits > 0 {
            let n = bits.min(MAX_BYPASS);
            bits -= n;
            let r = self.range >> n;
            // A corrupt stream can put the code beyond the range; clamp
            // instead of failing, `finish` reports the corruption.
            let v = (self.code / r).min((1 << n) - 1);
            self.code -= r * v;
            self.range = if v == (1 << n) - 1 { self.range - r * v } else { r };
            self.normalize();
            value = (value << n) | v;
        }
        value
    }

    /// Number of input bytes consumed so far.
    #[must_use]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Checks that the stream was consumed exactly.
    ///
    /// # Errors
    ///
    /// [`Error::Truncated`] if the decoder read past the end of the input and
    /// [`Error::TrailingBytes`] if input remains.
    pub fn finish(&self) -> Result<(), Error> {
        if self.missing > 0 {
            Err(Error::Truncated { missing: self.missing })
        } else if self.pos < self.input.len() {
            Err(Error::TrailingBytes { unread: self.input.len() - self.pos })
        } else {
            Ok(())
        }
    }
}
