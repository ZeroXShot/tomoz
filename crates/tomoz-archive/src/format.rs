//! Archive layout: instance table and reader.

use sha2::{Digest, Sha256};
use tomoz_codec::{DecodeOptions, ModelRegistry, Volume};
use tomoz_dicom::{BitConvention, PixelLayout, Samples};

use crate::{Error, MAGIC, VERSION};

pub(crate) const FIXED: usize = 72;
/// Largest decompressed metadata accepted (a guard against zstd bombs).
const MAX_METADATA: usize = 1 << 31;

/// zstd cannot expand data by more than this (an RLE block codes 128 KiB in
/// 4 bytes), so a larger declared size is corrupt and is refused before any
/// allocation.
const MAX_ZSTD_RATIO: u64 = 1 << 16;

/// Decompresses a zstd frame that must hold exactly `size` bytes, without
/// trusting `size` for the allocation.
fn unzstd(frame: &[u8], size: u64) -> Result<Vec<u8>, Error> {
    if size > (frame.len() as u64).saturating_mul(MAX_ZSTD_RATIO).saturating_add(1 << 16) {
        return Err(Error::Corrupt("declared size exceeds what the zstd frame can hold"));
    }
    let size = usize::try_from(size).map_err(|_| Error::TooLarge("instance does not fit in memory"))?;
    let mut out = Vec::new();
    out.try_reserve_exact(size).map_err(|_| Error::TooLarge("instance does not fit in memory"))?;
    // Decompresses into the spare capacity reserved above.
    zstd::bulk::Decompressor::new()
        .and_then(|mut d| d.decompress_to_buffer(frame, &mut out))
        .map_err(|e| Error::Zstd(e.to_string()))?;
    if out.len() != size {
        return Err(Error::Corrupt("zstd frame length"));
    }
    Ok(out)
}

/// How an instance is stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstanceKind {
    /// The whole file, compressed with zstd.
    Stored {
        /// Offset of its zstd frame in the stored-files section.
        offset: u64,
        /// Length of the frame.
        length: u64,
    },
    /// Pixel data coded in a stack, the rest of the file in the metadata.
    Coded {
        /// Stack holding the pixel data.
        stack: u32,
        /// First slice of the instance in the stack.
        first_slice: u32,
        /// Bit layout of the pixel cells.
        layout: PixelLayout,
        /// How the bits outside the stored values are filled.
        convention: BitConvention,
        /// Offset of the instance's region in the decompressed metadata.
        meta_offset: u64,
        /// Bytes before the pixel data value.
        prefix: u64,
        /// Bytes after the pixel data value.
        suffix: u64,
        /// Bytes of the value after the last cell.
        trailing: u32,
    },
}

/// One file of the archive.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstanceEntry {
    /// Name given when packing (typically the relative path).
    pub name: String,
    /// Size of the original file.
    pub size: u64,
    /// SHA-256 of the original file.
    pub sha256: [u8; 32],
    /// Storage of the instance.
    pub kind: InstanceKind,
}

impl InstanceEntry {
    /// Bytes of the instance's region in the decompressed metadata.
    fn meta_len(&self) -> u64 {
        match &self.kind {
            InstanceKind::Stored { .. } => 0,
            InstanceKind::Coded { layout, convention, prefix, suffix, trailing, .. } => {
                let irregular = if *convention == BitConvention::Irregular { 2 * layout.samples() as u64 } else { 0 };
                prefix + suffix + u64::from(*trailing) + irregular
            }
        }
    }

    pub(crate) fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.size.to_le_bytes());
        out.extend_from_slice(&self.sha256);
        let name = self.name.as_bytes();
        let name = &name[..name.len().min(u16::MAX as usize)];
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name);
        match &self.kind {
            InstanceKind::Stored { offset, length } => {
                out.push(0);
                out.extend_from_slice(&offset.to_le_bytes());
                out.extend_from_slice(&length.to_le_bytes());
            }
            InstanceKind::Coded { stack, first_slice, layout, convention, meta_offset, prefix, suffix, trailing } => {
                out.push(1);
                out.extend_from_slice(&stack.to_le_bytes());
                out.extend_from_slice(&first_slice.to_le_bytes());
                out.extend_from_slice(&layout.frames.to_le_bytes());
                out.extend_from_slice(&layout.rows.to_le_bytes());
                out.extend_from_slice(&layout.columns.to_le_bytes());
                out.extend_from_slice(&[
                    layout.bits_allocated,
                    layout.bits_stored,
                    layout.high_bit,
                    u8::from(layout.signed),
                    u8::from(layout.big_endian),
                    match convention {
                        BitConvention::Zero => 0,
                        BitConvention::SignExtend => 1,
                        BitConvention::Irregular => 2,
                    },
                ]);
                out.extend_from_slice(&meta_offset.to_le_bytes());
                out.extend_from_slice(&prefix.to_le_bytes());
                out.extend_from_slice(&suffix.to_le_bytes());
                out.extend_from_slice(&trailing.to_le_bytes());
            }
        }
    }

    fn read(r: &mut Reader<'_>) -> Result<Self, Error> {
        let size = r.u64()?;
        let sha256: [u8; 32] = r.take(32)?.try_into().map_err(|_| Error::Corrupt("instance digest"))?;
        let name_len = usize::from(r.u16()?);
        let name = String::from_utf8_lossy(r.take(name_len)?).into_owned();
        let kind = match r.u8()? {
            0 => InstanceKind::Stored { offset: r.u64()?, length: r.u64()? },
            1 => {
                let stack = r.u32()?;
                let first_slice = r.u32()?;
                let frames = r.u32()?;
                let rows = r.u32()?;
                let columns = r.u32()?;
                let b = r.take(6)?;
                let layout = PixelLayout {
                    rows,
                    columns,
                    frames,
                    bits_allocated: b[0],
                    bits_stored: b[1],
                    high_bit: b[2],
                    signed: b[3] == 1,
                    big_endian: b[4] == 1,
                };
                let valid = matches!(layout.bits_allocated, 8 | 16)
                    && layout.bits_stored >= 1
                    && layout.bits_stored <= layout.bits_allocated
                    && layout.high_bit < layout.bits_allocated
                    && layout.high_bit + 1 >= layout.bits_stored
                    && b[3] <= 1
                    && b[4] <= 1
                    && rows > 0
                    && columns > 0
                    && frames > 0;
                if !valid {
                    return Err(Error::Corrupt("pixel layout of an instance"));
                }
                let convention = match b[5] {
                    0 => BitConvention::Zero,
                    1 => BitConvention::SignExtend,
                    2 => BitConvention::Irregular,
                    _ => return Err(Error::Corrupt("bit convention of an instance")),
                };
                InstanceKind::Coded {
                    stack,
                    first_slice,
                    layout,
                    convention,
                    meta_offset: r.u64()?,
                    prefix: r.u64()?,
                    suffix: r.u64()?,
                    trailing: r.u32()?,
                }
            }
            _ => return Err(Error::Corrupt("instance kind")),
        };
        Ok(Self { name, size, sha256, kind })
    }
}

/// An archive opened for reading.
pub struct Archive<'a> {
    bytes: &'a [u8],
    instances: Vec<InstanceEntry>,
    stacks: Vec<(u64, u64)>,
    metadata: (u64, u64),
    stored: (u64, u64),
    max_samples: Option<u64>,
}

fn section(bytes: &[u8], (offset, length): (u64, u64)) -> Result<&[u8], Error> {
    let end = offset.checked_add(length).ok_or(Error::Corrupt("section bounds"))?;
    bytes.get(offset as usize..end as usize).ok_or(Error::Corrupt("section outside the archive"))
}

impl<'a> Archive<'a> {
    /// Parses the header, stack table and instance table.
    ///
    /// # Errors
    ///
    /// [`Error::NotArchive`], [`Error::Unsupported`] or [`Error::Corrupt`].
    pub fn open(bytes: &'a [u8]) -> Result<Self, Error> {
        if bytes.len() < FIXED || bytes[..4] != MAGIC {
            return Err(Error::NotArchive);
        }
        let mut r = Reader { bytes, pos: 4 };
        let version = r.u16()?;
        if version != VERSION {
            return Err(Error::Unsupported(format!("archive version {version}")));
        }
        if r.u16()? != 0 {
            return Err(Error::Unsupported("archive flags".into()));
        }
        let instance_count = r.u32()? as usize;
        let stack_count = r.u32()? as usize;
        let table = (r.u64()?, r.u64()?);
        let metadata = (r.u64()?, r.u64()?);
        let stored = (r.u64()?, r.u64()?);
        let table_crc = r.u32()?;
        let header_crc = r.u32()?;
        let mut stacks = Vec::with_capacity(stack_count.min(1 << 16));
        for _ in 0..stack_count {
            stacks.push((r.u64()?, r.u64()?));
        }
        let mut crc = crc32c::crc32c(&bytes[..68]);
        crc = crc32c::crc32c_append(crc, &bytes[FIXED..r.pos]);
        if crc != header_crc {
            return Err(Error::Corrupt("header checksum mismatch"));
        }
        let table_bytes = section(bytes, table)?;
        if crc32c::crc32c(table_bytes) != table_crc {
            return Err(Error::Corrupt("instance table checksum mismatch"));
        }
        let mut tr = Reader { bytes: table_bytes, pos: 0 };
        let mut instances = Vec::with_capacity(instance_count.min(1 << 20));
        for _ in 0..instance_count {
            instances.push(InstanceEntry::read(&mut tr)?);
        }
        if tr.pos != table_bytes.len() {
            return Err(Error::Corrupt("instance table length"));
        }
        if instances
            .iter()
            .any(|e| matches!(e.kind, InstanceKind::Coded { stack, .. } if stack as usize >= stack_count))
        {
            return Err(Error::Corrupt("instance refers to a missing stack"));
        }
        for s in &stacks {
            section(bytes, *s)?;
        }
        section(bytes, metadata)?;
        section(bytes, stored)?;
        Ok(Self { bytes, instances, stacks, metadata, stored, max_samples: None })
    }

    /// Limits the samples of a stack the decoder accepts (see
    /// [`DecodeOptions::max_samples`]); set it when reading untrusted
    /// archives.
    #[must_use]
    pub fn with_max_samples(mut self, max_samples: u64) -> Self {
        self.max_samples = Some(max_samples);
        self
    }

    fn decode_options<'r>(&self, registry: &'r dyn ModelRegistry) -> DecodeOptions<'r> {
        let mut options = DecodeOptions::with_registry(registry);
        if let Some(max) = self.max_samples {
            options.max_samples = max;
        }
        options
    }

    /// The instances, in packing order.
    #[must_use]
    pub fn instances(&self) -> &[InstanceEntry] {
        &self.instances
    }

    /// Number of stacks.
    #[must_use]
    pub fn stack_count(&self) -> usize {
        self.stacks.len()
    }

    /// The volume container of stack `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i` is not a stack index.
    #[must_use]
    pub fn stack(&self, i: usize) -> &'a [u8] {
        let (o, l) = self.stacks[i];
        &self.bytes[o as usize..(o + l) as usize]
    }

    /// The decompressed metadata of all coded instances (headers and the
    /// other bytes stored verbatim). Callers restoring many instances should
    /// keep it and use [`Archive::rebuild`].
    ///
    /// # Errors
    ///
    /// zstd and consistency errors.
    pub fn metadata(&self) -> Result<Vec<u8>, Error> {
        let frame = section(self.bytes, self.metadata)?;
        if frame.is_empty() {
            return Ok(Vec::new());
        }
        let expected: u64 = self.instances.iter().map(InstanceEntry::meta_len).sum();
        if expected > MAX_METADATA as u64 {
            return Err(Error::Corrupt("metadata size"));
        }
        unzstd(frame, expected)
    }

    /// The stack and slice range holding the pixel data of instance `index`
    /// (`None` for stored instances).
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of range.
    #[must_use]
    pub fn instance_slices(&self, index: usize) -> Option<(usize, std::ops::Range<usize>)> {
        match &self.instances[index].kind {
            InstanceKind::Stored { .. } => None,
            InstanceKind::Coded { stack, first_slice, layout, .. } => {
                let first = *first_slice as usize;
                Some((*stack as usize, first..first + layout.frames as usize))
            }
        }
    }

    /// Restores every instance, decoding each stack once.
    ///
    /// # Errors
    ///
    /// Codec, zstd and consistency errors, and [`Error::Digest`] if a restored
    /// file does not match its SHA-256.
    pub fn restore_all(&self, registry: &dyn ModelRegistry) -> Result<Vec<Vec<u8>>, Error> {
        let options = self.decode_options(registry);
        let volumes = (0..self.stacks.len())
            .map(|i| tomoz_codec::decode(self.stack(i), &options).map_err(Error::from))
            .collect::<Result<Vec<_>, _>>()?;
        let metadata = self.metadata()?;
        (0..self.instances.len())
            .map(|i| {
                let pixels = self.instance_slices(i).map(|(s, r)| (&volumes[s], r));
                self.rebuild_from(i, pixels, &metadata)
            })
            .collect()
    }

    /// Restores instance `index`, decoding only the tiles that hold its
    /// slices.
    ///
    /// # Errors
    ///
    /// As [`Archive::restore_all`].
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of range.
    pub fn restore(&self, index: usize, registry: &dyn ModelRegistry) -> Result<Vec<u8>, Error> {
        let options = self.decode_options(registry);
        let slices = match self.instance_slices(index) {
            Some((stack, range)) => {
                if stack >= self.stacks.len() {
                    return Err(Error::Corrupt("stack index"));
                }
                Some(tomoz_codec::decode_slices(self.stack(stack), &options, range)?)
            }
            None => None,
        };
        let metadata = if slices.is_some() { self.metadata()? } else { Vec::new() };
        self.rebuild(index, slices.as_ref(), &metadata)
    }

    /// Rebuilds instance `index` from the decoded slices of its pixel data
    /// (exactly [`Archive::instance_slices`], `None` for stored instances)
    /// and the archive [`metadata`](Archive::metadata).
    ///
    /// # Errors
    ///
    /// Consistency errors and [`Error::Digest`].
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of range.
    pub fn rebuild(&self, index: usize, slices: Option<&Volume>, metadata: &[u8]) -> Result<Vec<u8>, Error> {
        let pixels = slices.map(|v| (v, 0..v.depth()));
        self.rebuild_from(index, pixels, metadata)
    }

    fn rebuild_from(
        &self,
        index: usize,
        pixels: Option<(&Volume, std::ops::Range<usize>)>,
        metadata: &[u8],
    ) -> Result<Vec<u8>, Error> {
        let entry = &self.instances[index];
        let out = match &entry.kind {
            InstanceKind::Stored { offset, length } => {
                let frame = section(section(self.bytes, self.stored)?, (*offset, *length))?;
                unzstd(frame, entry.size)?
            }
            InstanceKind::Coded { layout, convention, meta_offset, prefix, suffix, trailing, .. } => {
                let (volume, range) = pixels.ok_or(Error::Corrupt("coded instance without pixel data"))?;
                let frame = layout.frame_samples();
                if volume.height() != layout.rows as usize || volume.width() != layout.columns as usize {
                    return Err(Error::Corrupt("instance geometry does not match its stack"));
                }
                if range.len() != layout.frames as usize || range.end > volume.depth() {
                    return Err(Error::Corrupt("instance slices outside its stack"));
                }
                let values = volume.samples()[range.start * frame..range.end * frame].to_vec();
                let m = section(metadata, (*meta_offset, entry.meta_len()))?;
                let (pre, rest) = m.split_at(*prefix as usize);
                let (suf, rest) = rest.split_at(*suffix as usize);
                let (trail, irr) = rest.split_at(*trailing as usize);
                let irregular = (*convention == BitConvention::Irregular)
                    .then(|| irr.as_chunks::<2>().0.iter().map(|&c| u16::from_le_bytes(c)).collect());
                let samples = Samples { values, convention: *convention, irregular, trailing: trail.to_vec() };
                let (lo, hi) = layout.value_range();
                if samples.values.iter().any(|v| *v < lo || *v > hi) {
                    return Err(Error::Corrupt("pixel values outside the stored range"));
                }
                let pixel_data = layout.reconstruct(&samples);
                let mut file = Vec::with_capacity(pre.len() + pixel_data.len() + suf.len());
                file.extend_from_slice(pre);
                file.extend_from_slice(&pixel_data);
                file.extend_from_slice(suf);
                file
            }
        };
        if out.len() as u64 != entry.size || <[u8; 32]>::from(Sha256::digest(&out)) != entry.sha256 {
            return Err(Error::Digest(index));
        }
        Ok(out)
    }
}

pub(crate) struct Reader<'a> {
    pub bytes: &'a [u8],
    pub pos: usize,
}

impl<'a> Reader<'a> {
    pub fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.pos.checked_add(n).ok_or(Error::Corrupt("truncated"))?;
        let s = self.bytes.get(self.pos..end).ok_or(Error::Corrupt("truncated"))?;
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, Error> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().map_err(|_| Error::Corrupt("truncated"))?))
    }

    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().map_err(|_| Error::Corrupt("truncated"))?))
    }

    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().map_err(|_| Error::Corrupt("truncated"))?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zstd_sizes_are_not_trusted() {
        let data = vec![7u8; 100_000];
        let frame = zstd::bulk::compress(&data, 3).unwrap();
        assert_eq!(unzstd(&frame, data.len() as u64).unwrap(), data);
        assert!(matches!(unzstd(&frame, data.len() as u64 + 1), Err(Error::Corrupt(_))));
        assert!(matches!(unzstd(&frame, 1 << 40), Err(Error::Corrupt(_))));
        assert!(matches!(unzstd(&frame, u64::MAX), Err(Error::Corrupt(_))));
    }
}
