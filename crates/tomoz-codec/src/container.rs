//! The `.tmz` volume container.
//!
//! ```text
//! offset  size  field
//!      0     4  magic: 0x89 'T' 'M' 'Z'
//!      4     2  format version (1)
//!      6     2  flags (bit 0: histogram packing, bit 1: metadata)
//!      8     4  depth
//!     12     4  height
//!     16     4  width
//!     20     1  bits per sample
//!     21     1  signed (0 or 1)
//!     22     2  reserved (0)
//!     24     4  offset (i32): value of mapped zero
//!     28     4  largest mapped value
//!     32     2  slab depth (slices per tile)
//!     34     2  stripe height (rows per tile)
//!     36     4  number of tiles
//!     40     4  length of the variable header part
//!     44    16  2-D model identifier
//!     60    16  3-D model identifier
//!     76    32  SHA-256 of the samples (16-bit little endian)
//!    108     4  length of the packing table (0 without packing)
//!    112     n  packing table
//!      .     4  length of the application metadata (0 without)
//!      .     m  application metadata (opaque to the codec)
//!      .  20×t  tile table: offset (u64, from the end of the header),
//!               length (u32), CRC-32C of the stream (u32), CRC-32C of the
//!               decoded samples as 16-bit little endian (u32)
//!      .     4  CRC-32C of every header byte before this field
//! ```
//!
//! All integers are little endian. Tiles cover `slab` slices and `stripe`
//! rows; tile `i` is slab `i / stripes`, stripe `i % stripes`.

use crate::Error;
use crate::model::ModelId;

pub(crate) const MAGIC: [u8; 4] = [0x89, b'T', b'M', b'Z'];
pub(crate) const VERSION: u16 = 1;
const FIXED: usize = 44;
const FLAG_PACKED: u16 = 1;
const FLAG_METADATA: u16 = 2;
const KNOWN_FLAGS: u16 = FLAG_PACKED | FLAG_METADATA;
/// Upper bound on the samples of a volume the decoder accepts by default,
/// which bounds the memory a crafted header can make it allocate.
pub(crate) const DEFAULT_MAX_SAMPLES: u64 = 1 << 34;

/// Entry of the tile table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TileEntry {
    pub offset: u64,
    pub length: u32,
    pub stream_crc: u32,
    pub samples_crc: u32,
}

/// Decoded container header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    /// Format version.
    pub version: u16,
    /// Number of slices.
    pub depth: u32,
    /// Rows per slice.
    pub height: u32,
    /// Columns per row.
    pub width: u32,
    /// Bits per sample.
    pub bits: u8,
    /// Whether samples are signed.
    pub signed: bool,
    /// Value of mapped zero.
    pub offset: i32,
    /// Largest mapped value.
    pub max_mapped: i32,
    /// Slices per tile.
    pub slab: u16,
    /// Rows per tile.
    pub stripe: u16,
    /// 2-D model.
    pub model_2d: ModelId,
    /// 3-D model.
    pub model_3d: ModelId,
    /// SHA-256 of the samples.
    pub sha256: [u8; 32],
    pub(crate) packing: Vec<u8>,
    /// Application metadata stored with the volume (for example the header
    /// of the file it came from).
    pub metadata: Vec<u8>,
    pub(crate) tiles: Vec<TileEntry>,
    /// Total header length; tile offsets start here.
    pub header_len: usize,
}

impl Header {
    /// Number of slabs.
    #[must_use]
    pub fn slabs(&self) -> usize {
        (self.depth as usize).div_ceil(usize::from(self.slab))
    }

    /// Number of stripes per slab.
    #[must_use]
    pub fn stripes(&self) -> usize {
        (self.height as usize).div_ceil(usize::from(self.stripe))
    }

    /// Number of tiles.
    #[must_use]
    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// Whether the volume uses histogram packing.
    #[must_use]
    pub fn packed(&self) -> bool {
        !self.packing.is_empty()
    }

    /// Bytes of tile data.
    #[must_use]
    pub fn data_len(&self) -> u64 {
        self.tiles.iter().map(|t| u64::from(t.length)).sum()
    }

    pub(crate) fn write(&self) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(FIXED + 76 + self.packing.len() + self.metadata.len() + 20 * self.tiles.len() + 4);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&self.version.to_le_bytes());
        let flags = if self.packing.is_empty() { 0 } else { FLAG_PACKED }
            | if self.metadata.is_empty() { 0 } else { FLAG_METADATA };
        out.extend_from_slice(&flags.to_le_bytes());
        for v in [self.depth, self.height, self.width] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&[self.bits, u8::from(self.signed), 0, 0]);
        out.extend_from_slice(&self.offset.to_le_bytes());
        out.extend_from_slice(&self.max_mapped.to_le_bytes());
        out.extend_from_slice(&self.slab.to_le_bytes());
        out.extend_from_slice(&self.stripe.to_le_bytes());
        out.extend_from_slice(&(self.tiles.len() as u32).to_le_bytes());
        let variable = 16 + 16 + 32 + 4 + self.packing.len() + 4 + self.metadata.len() + 20 * self.tiles.len() + 4;
        out.extend_from_slice(&(variable as u32).to_le_bytes());
        out.extend_from_slice(&self.model_2d.0);
        out.extend_from_slice(&self.model_3d.0);
        out.extend_from_slice(&self.sha256);
        out.extend_from_slice(&(self.packing.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.packing);
        out.extend_from_slice(&(self.metadata.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.metadata);
        for t in &self.tiles {
            out.extend_from_slice(&t.offset.to_le_bytes());
            out.extend_from_slice(&t.length.to_le_bytes());
            out.extend_from_slice(&t.stream_crc.to_le_bytes());
            out.extend_from_slice(&t.samples_crc.to_le_bytes());
        }
        let crc = crc32c::crc32c(&out);
        out.extend_from_slice(&crc.to_le_bytes());
        out
    }

    /// Parses and validates the header at the start of `bytes`.
    ///
    /// # Errors
    ///
    /// [`Error::NotTomoz`] if the magic is missing, [`Error::Unsupported`] for
    /// newer format versions, [`Error::Corrupt`] for inconsistent headers.
    pub fn parse(bytes: &[u8], max_samples: u64) -> Result<Self, Error> {
        if bytes.len() < 4 || bytes[..4] != MAGIC {
            return Err(Error::NotTomoz);
        }
        let mut r = Reader { bytes, pos: 4 };
        let version = r.u16()?;
        if version != VERSION {
            return Err(Error::Unsupported(format!("container version {version}")));
        }
        let flags = r.u16()?;
        if flags & !KNOWN_FLAGS != 0 {
            return Err(Error::Unsupported(format!("container flags {flags:#06x}")));
        }
        let (depth, height, width) = (r.u32()?, r.u32()?, r.u32()?);
        let bits = r.u8()?;
        let signed = match r.u8()? {
            0 => false,
            1 => true,
            _ => return Err(Error::Corrupt("signedness flag")),
        };
        if r.u16()? != 0 {
            return Err(Error::Corrupt("reserved header field"));
        }
        let offset = r.i32()?;
        let max_mapped = r.i32()?;
        let slab = r.u16()?;
        let stripe = r.u16()?;
        let tile_count = r.u32()? as usize;
        let variable = r.u32()? as usize;
        // Three u32 can overflow even a u64.
        let samples = u64::from(depth).checked_mul(u64::from(height)).and_then(|n| n.checked_mul(u64::from(width)));
        if depth == 0 || height == 0 || width == 0 || samples.is_none_or(|n| n > max_samples) {
            return Err(Error::Corrupt("volume dimensions"));
        }
        if bits == 0 || bits > crate::MAX_BITS || slab == 0 || stripe == 0 {
            return Err(Error::Corrupt("sample format or tiling"));
        }
        let (lo, hi) = crate::value_range(bits, signed);
        if offset < lo || max_mapped < 0 || i64::from(offset) + i64::from(max_mapped) > i64::from(hi) {
            return Err(Error::Corrupt("value mapping"));
        }
        // In u64: usize is 32 bits on WebAssembly.
        let expected_tiles = u64::from(depth).div_ceil(u64::from(slab)) * u64::from(height).div_ceil(u64::from(stripe));
        if tile_count as u64 != expected_tiles {
            return Err(Error::Corrupt("tile count does not match the tiling"));
        }
        if variable.checked_add(FIXED).is_none_or(|n| bytes.len() < n) {
            return Err(Error::Truncated);
        }
        let mut id = || -> Result<[u8; 16], Error> { r.take(16)?.try_into().map_err(|_| Error::Truncated) };
        let model_2d = ModelId(id()?);
        let model_3d = ModelId(id()?);
        let sha256: [u8; 32] = r.take(32)?.try_into().map_err(|_| Error::Truncated)?;
        let packing_len = r.u32()? as usize;
        let packing = r.take(packing_len)?.to_vec();
        if (flags & FLAG_PACKED != 0) != (packing_len > 0) {
            return Err(Error::Corrupt("packing flag and table disagree"));
        }
        let metadata_len = r.u32()? as usize;
        let metadata = r.take(metadata_len)?.to_vec();
        if (flags & FLAG_METADATA != 0) != (metadata_len > 0) {
            return Err(Error::Corrupt("metadata flag and length disagree"));
        }
        let mut tiles = Vec::with_capacity(tile_count.min(1 << 20));
        let mut next_offset = 0u64;
        for _ in 0..tile_count {
            let t = TileEntry { offset: r.u64()?, length: r.u32()?, stream_crc: r.u32()?, samples_crc: r.u32()? };
            if t.offset != next_offset {
                return Err(Error::Corrupt("tile table is not contiguous"));
            }
            next_offset = next_offset.checked_add(u64::from(t.length)).ok_or(Error::Corrupt("tile table"))?;
            tiles.push(t);
        }
        let crc_pos = r.pos;
        let crc = r.u32()?;
        if r.pos != FIXED + variable {
            return Err(Error::Corrupt("header length"));
        }
        if crc32c::crc32c(&bytes[..crc_pos]) != crc {
            return Err(Error::Corrupt("header checksum mismatch"));
        }
        Ok(Self {
            version,
            depth,
            height,
            width,
            bits,
            signed,
            offset,
            max_mapped,
            slab,
            stripe,
            model_2d,
            model_3d,
            sha256,
            packing,
            metadata,
            tiles,
            header_len: r.pos,
        })
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.pos.checked_add(n).ok_or(Error::Truncated)?;
        let s = self.bytes.get(self.pos..end).ok_or(Error::Truncated)?;
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, Error> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().map_err(|_| Error::Truncated)?))
    }

    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().map_err(|_| Error::Truncated)?))
    }

    fn i32(&mut self) -> Result<i32, Error> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().map_err(|_| Error::Truncated)?))
    }

    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().map_err(|_| Error::Truncated)?))
    }
}
