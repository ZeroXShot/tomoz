//! Range coding with adaptive models.
//!
//! This crate provides the entropy coding layer of Tomoz:
//!
//! * [`Encoder`] and [`Decoder`]: a byte-oriented range coder with carry
//!   propagation in the encoder (the scheme used by LZMA), 32-bit range and
//!   15-bit probabilities.
//! * [`Cdf`]: an adaptive model over up to 16 symbols whose cumulative
//!   distribution moves towards each coded symbol, with a learning rate that
//!   slows down as the model sees more data (as in AV1).
//! * [`BitModel`]: an adaptive binary model.
//! * Bypass bits for values that are close to uniform.
//!
//! All arithmetic is on integers with fully specified rounding, so a stream
//! encoded on one platform decodes to the same symbols on every other one, and
//! the same input always produces the same bytes. The decoder never panics on
//! malformed input: missing bytes read as zero and are counted, and
//! [`Decoder::finish`] reports streams that were truncated or did not end
//! where the encoder ended them.

#![no_std]

extern crate alloc;

mod model;
mod range;

pub use model::{BitModel, Cdf, MAX_SYMBOLS};
pub use range::{Decoder, Encoder, Error};

/// Number of bits of probability precision.
pub const PROB_BITS: u32 = 15;

/// Probability of one, in the fixed-point representation of [`PROB_BITS`] bits.
pub const PROB_ONE: u32 = 1 << PROB_BITS;
