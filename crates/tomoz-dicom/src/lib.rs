//! DICOM Part 10 parsing for lossless archiving.
//!
//! Tomoz needs three things from a DICOM file: the attributes that describe
//! its pixel data and place it in a series, the exact byte span of the pixel
//! data, and the guarantee that the bytes around that span can be stored
//! verbatim and put back. This crate provides them without interpreting more
//! of the file than necessary:
//!
//! * [`parse`] walks the data set of a Part 10 file (implicit and explicit VR,
//!   little and big endian, nested sequences of defined and undefined length,
//!   encapsulated pixel data) and returns the [`Attributes`] Tomoz uses and the
//!   [`PixelSpan`] of the top-level Pixel Data element. Every length is
//!   checked against the buffer, nesting is bounded, and malformed input is an
//!   error, never a panic.
//! * [`PixelLayout`] converts native pixel data to integer samples and back,
//!   including files whose unused high bits carry data, so that
//!   reconstruction is byte-exact.
//!
//! Files in transfer syntaxes whose data set cannot be stored verbatim around
//! the pixel data (deflated data sets) or whose pixel data is already
//! compressed are reported as such; the archive layer stores them unchanged.

mod attrs;
mod parse;
mod pixel;
pub mod tag;

pub use attrs::Attributes;
pub use parse::{DatasetEncoding, Error, ParsedFile, PixelSpan, parse};
pub use pixel::{BitConvention, LayoutError, PixelLayout, Samples};
pub use tag::Tag;

/// Well-known transfer syntax UIDs.
pub mod transfer_syntax {
    /// Implicit VR Little Endian, the default DICOM transfer syntax.
    pub const IMPLICIT_VR_LITTLE_ENDIAN: &str = "1.2.840.10008.1.2";
    /// Explicit VR Little Endian.
    pub const EXPLICIT_VR_LITTLE_ENDIAN: &str = "1.2.840.10008.1.2.1";
    /// Deflated Explicit VR Little Endian.
    pub const DEFLATED_EXPLICIT_VR_LITTLE_ENDIAN: &str = "1.2.840.10008.1.2.1.99";
    /// Explicit VR Big Endian (retired, still found in old archives).
    pub const EXPLICIT_VR_BIG_ENDIAN: &str = "1.2.840.10008.1.2.2";
}
