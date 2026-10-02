//! NIfTI-1 images (`.nii`, `.nii.gz`) of 8- and 16-bit integers.
//!
//! The header and extensions (everything before the voxel data) are kept
//! verbatim as container metadata, so an uncompressed `.nii` file is restored
//! byte for byte.

use std::io::Read;

use anyhow::{Context, Result, bail};
use tomoz_codec::Volume;

/// Prefix of the metadata that marks a NIfTI header.
pub const METADATA_TAG: &[u8; 8] = b"nifti1\0\0";

/// A NIfTI image split into its header bytes and its voxels.
pub struct Nifti {
    /// Header and extensions, up to the voxel offset.
    pub header: Vec<u8>,
    /// Voxels as a volume of `dim[3] * dim[4]` slices.
    pub volume: Volume,
    big_endian: bool,
}

/// Decompresses gzip input; returns other input unchanged.
pub fn maybe_gunzip(bytes: Vec<u8>) -> Result<Vec<u8>> {
    if bytes.starts_with(&[0x1f, 0x8b]) {
        let mut out = Vec::new();
        flate2::read::MultiGzDecoder::new(&bytes[..]).read_to_end(&mut out).context("gzip")?;
        Ok(out)
    } else {
        Ok(bytes)
    }
}

pub fn read(bytes: &[u8]) -> Result<Nifti> {
    if bytes.len() < 352 {
        bail!("too short for a NIfTI-1 file");
    }
    let big_endian = match (i32::from_le_bytes(bytes[0..4].try_into()?), i32::from_be_bytes(bytes[0..4].try_into()?)) {
        (348, _) => false,
        (_, 348) => true,
        _ => bail!("not a NIfTI-1 file (sizeof_hdr is not 348)"),
    };
    let i16_at = |o: usize| {
        let b = [bytes[o], bytes[o + 1]];
        if big_endian { i16::from_be_bytes(b) } else { i16::from_le_bytes(b) }
    };
    let f32_at = |o: usize| {
        let b = [bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]];
        if big_endian { f32::from_be_bytes(b) } else { f32::from_le_bytes(b) }
    };
    if &bytes[344..347] != b"n+1" {
        bail!("only single-file NIfTI-1 images (magic n+1) are supported");
    }
    let dims: Vec<usize> = (0..8).map(|i| i16_at(40 + 2 * i).max(0) as usize).collect();
    if dims[0] < 2 || dims[0] > 4 {
        bail!("NIfTI images with {} dimensions are not supported", dims[0]);
    }
    let (w, h) = (dims[1].max(1), dims[2].max(1));
    let depth = (if dims[0] >= 3 { dims[3].max(1) } else { 1 }) * (if dims[0] >= 4 { dims[4].max(1) } else { 1 });
    let datatype = i16_at(70);
    let offset = f32_at(108);
    if !(offset >= 352.0 && offset.fract() == 0.0) {
        bail!("invalid voxel offset {offset}");
    }
    let offset = offset as usize;
    let n = w * h * depth;
    let data = bytes.get(offset..).context("voxel data missing")?;
    let read16 =
        |c: &[u8]| if big_endian { u16::from_be_bytes([c[0], c[1]]) } else { u16::from_le_bytes([c[0], c[1]]) };
    let (samples, bits, signed): (Vec<i32>, u8, bool) = match datatype {
        2 => (data.get(..n).context("voxel data truncated")?.iter().map(|&v| i32::from(v)).collect(), 8, false),
        256 => (data.get(..n).context("voxel data truncated")?.iter().map(|&v| i32::from(v as i8)).collect(), 8, true),
        4 => (
            data.get(..2 * n)
                .context("voxel data truncated")?
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| i32::from(read16(c) as i16))
                .collect(),
            16,
            true,
        ),
        512 => (
            data.get(..2 * n)
                .context("voxel data truncated")?
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| i32::from(read16(c)))
                .collect(),
            16,
            false,
        ),
        t => bail!("NIfTI datatype {t} is not supported: Tomoz codes 8- and 16-bit integers"),
    };
    if data.len() > n * usize::from(bits / 8) {
        bail!("trailing bytes after the voxel data are not supported");
    }
    let volume = Volume::new(depth, h, w, bits, signed, samples)?;
    Ok(Nifti { header: bytes[..offset].to_vec(), volume, big_endian })
}

impl Nifti {
    /// Container metadata: tag, endianness flag, header bytes.
    pub fn metadata(&self) -> Vec<u8> {
        let mut m = METADATA_TAG.to_vec();
        m.push(u8::from(self.big_endian));
        m.extend_from_slice(&self.header);
        m
    }

    /// Rebuilds the file from container metadata and decoded voxels.
    pub fn restore(metadata: &[u8], volume: &Volume) -> Result<Vec<u8>> {
        let rest = metadata.strip_prefix(METADATA_TAG.as_slice()).context("metadata is not a NIfTI header")?;
        let (&big, header) = rest.split_first().context("truncated NIfTI metadata")?;
        let mut out = header.to_vec();
        for &s in volume.samples() {
            if volume.bits() <= 8 {
                out.push(s as u8);
            } else if big == 1 {
                out.extend_from_slice(&(s as u16).to_be_bytes());
            } else {
                out.extend_from_slice(&(s as u16).to_le_bytes());
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(datatype: i16, dims: [i16; 4], data: &[u8]) -> Vec<u8> {
        let mut h = vec![0u8; 352];
        h[0..4].copy_from_slice(&348i32.to_le_bytes());
        h[40..42].copy_from_slice(&dims[0].to_le_bytes());
        for (i, d) in dims[1..].iter().enumerate() {
            h[42 + 2 * i..44 + 2 * i].copy_from_slice(&d.to_le_bytes());
        }
        h[70..72].copy_from_slice(&datatype.to_le_bytes());
        h[108..112].copy_from_slice(&352f32.to_le_bytes());
        h[344..348].copy_from_slice(b"n+1\0");
        h.extend_from_slice(data);
        h
    }

    #[test]
    fn int16_roundtrip_is_byte_exact() {
        let data: Vec<u8> = (0..2 * 3 * 4i16).flat_map(|v| (v * 300 - 900).to_le_bytes()).collect();
        let bytes = file(4, [3, 4, 3, 2], &data);
        let n = read(&bytes).unwrap();
        assert_eq!((n.volume.depth(), n.volume.height(), n.volume.width()), (2, 3, 4));
        assert_eq!(n.volume.samples()[1], -600);
        assert_eq!(Nifti::restore(&n.metadata(), &n.volume).unwrap(), bytes);
    }

    #[test]
    fn floats_are_rejected() {
        assert!(read(&file(16, [3, 2, 2, 1], &[0; 16])).is_err());
    }
}
