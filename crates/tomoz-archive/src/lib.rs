//! Byte-exact DICOM archives with Tomoz-coded pixel data.
//!
//! An archive holds a set of DICOM files, usually the instances of a series,
//! and restores each of them byte for byte. Instances with native grayscale
//! pixel data are sorted into stacks (one per geometry, ordered along the
//! slice normal) and coded with the Tomoz volume codec, so that each slice is
//! predicted from its neighbours. Everything else in each file (the preamble,
//! the file meta information, every data element before and after the pixel
//! data, the bits outside the stored value) is kept verbatim and compressed
//! together with zstd, which removes the redundancy between the headers of a
//! series. Files that cannot be coded (compressed transfer syntaxes, colour,
//! float pixels, malformed data) are stored whole, compressed with zstd.
//!
//! Every instance carries the SHA-256 of its original bytes; restoring an
//! instance checks it.
//!
//! ```text
//! offset  size  field
//!      0     4  magic: 0x89 'T' 'Z' 'D'
//!      4     2  version (1)
//!      6     2  flags (0)
//!      8     4  instance count
//!     12     4  stack count
//!     16    16  instance table: offset, length (u64 each)
//!     32    16  metadata (one zstd frame): offset, length
//!     48    16  stored files (zstd frames): offset, length
//!     64     4  CRC-32C of the instance table
//!     68     4  CRC-32C of bytes 0..68 and the stack table
//!     72  16×s  stack table: offset, length (u64 each) of each volume
//!               container
//! ```

mod format;
mod pack;

pub use format::{Archive, InstanceEntry, InstanceKind};
pub use pack::{PackOptions, PackReport, StoredReason, pack};

/// Errors of the archive layer.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The data is not a Tomoz DICOM archive.
    #[error("not a Tomoz DICOM archive")]
    NotArchive,
    /// The archive is internally inconsistent.
    #[error("corrupt archive: {0}")]
    Corrupt(&'static str),
    /// The archive uses a feature this version does not support.
    #[error("unsupported archive: {0}")]
    Unsupported(String),
    /// A restored file does not match its recorded digest.
    #[error("instance {0} does not match its SHA-256")]
    Digest(usize),
    /// The volume codec failed.
    #[error(transparent)]
    Codec(#[from] tomoz_codec::Error),
    /// zstd failed.
    #[error("zstd: {0}")]
    Zstd(String),
    /// The input set is too large for the format.
    #[error("too large: {0}")]
    TooLarge(&'static str),
}

pub(crate) const MAGIC: [u8; 4] = [0x89, b'T', b'Z', b'D'];
pub(crate) const VERSION: u16 = 1;
