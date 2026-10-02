//! Volumes and the mapping of their values to the coded range.

use sha2::{Digest, Sha256};
use tomoz_entropy::{BitModel, Decoder, Encoder};

use crate::Error;

/// Largest supported bit depth.
pub const MAX_BITS: u8 = 16;

/// A stack of equally sized grayscale slices with integer samples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Volume {
    depth: usize,
    height: usize,
    width: usize,
    bits: u8,
    signed: bool,
    samples: Vec<i32>,
}

impl Volume {
    /// A volume of `depth` slices of `height × width` samples in raster order
    /// (slice, row, column). Values must be representable in `bits` bits,
    /// two's complement if `signed`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidVolume`] if a dimension is zero, the bit depth is
    /// outside `1..=16`, the sample count does not match or a value is out of
    /// range.
    pub fn new(
        depth: usize,
        height: usize,
        width: usize,
        bits: u8,
        signed: bool,
        samples: Vec<i32>,
    ) -> Result<Self, Error> {
        if depth == 0 || height == 0 || width == 0 {
            return Err(Error::InvalidVolume("empty dimension".into()));
        }
        if bits == 0 || bits > MAX_BITS {
            return Err(Error::InvalidVolume(format!("{bits} bits per sample; 1 to {MAX_BITS} are supported")));
        }
        let n = depth.checked_mul(height).and_then(|v| v.checked_mul(width));
        if n != Some(samples.len()) {
            return Err(Error::InvalidVolume(format!("{} samples for {depth}×{height}×{width}", samples.len())));
        }
        let (lo, hi) = value_range(bits, signed);
        if let Some(v) = samples.iter().find(|&&v| v < lo || v > hi) {
            return Err(Error::InvalidVolume(format!("value {v} outside [{lo}, {hi}]")));
        }
        Ok(Self { depth, height, width, bits, signed, samples })
    }

    /// A volume of unsigned 16-bit samples with `bits` significant bits.
    ///
    /// # Errors
    ///
    /// As [`Volume::new`].
    pub fn from_u16(depth: usize, height: usize, width: usize, bits: u8, samples: &[u16]) -> Result<Self, Error> {
        Self::new(depth, height, width, bits, false, samples.iter().map(|&v| i32::from(v)).collect())
    }

    /// A volume of signed 16-bit samples with `bits` significant bits.
    ///
    /// # Errors
    ///
    /// As [`Volume::new`].
    pub fn from_i16(depth: usize, height: usize, width: usize, bits: u8, samples: &[i16]) -> Result<Self, Error> {
        Self::new(depth, height, width, bits, true, samples.iter().map(|&v| i32::from(v)).collect())
    }

    /// Number of slices.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Rows per slice.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Columns per row.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Bits per sample.
    #[must_use]
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Whether samples are signed.
    #[must_use]
    pub fn signed(&self) -> bool {
        self.signed
    }

    /// The samples in raster order.
    #[must_use]
    pub fn samples(&self) -> &[i32] {
        &self.samples
    }

    /// Consumes the volume and returns its samples.
    #[must_use]
    pub fn into_samples(self) -> Vec<i32> {
        self.samples
    }

    /// Size of the samples in their natural 16-bit representation.
    #[must_use]
    pub fn raw_bytes(&self) -> usize {
        self.samples.len() * 2
    }

    /// SHA-256 of the samples, each as a 16-bit little-endian integer (two's
    /// complement if signed): the digest of `array.astype('<i2' or
    /// '<u2').tobytes()` in NumPy terms.
    #[must_use]
    pub fn sha256(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        for chunk in self.samples.chunks(4096) {
            let bytes: Vec<u8> = chunk.iter().flat_map(|&v| (v as u16).to_le_bytes()).collect();
            h.update(&bytes);
        }
        h.finalize().into()
    }
}

/// Smallest and largest value of `bits`-bit samples.
#[must_use]
pub fn value_range(bits: u8, signed: bool) -> (i32, i32) {
    let b = u32::from(bits);
    if signed { (-(1 << (b - 1)), (1 << (b - 1)) - 1) } else { (0, (1 << b) - 1) }
}

/// Maps sample values to the non-negative range the codec works in.
///
/// Values are shifted so that the smallest is zero. When less than three
/// quarters of the resulting range is used (images whose values are all
/// even, or with a padding value far from the rest), values are also replaced
/// by their rank among the values present ("histogram packing"), which keeps
/// their order and removes the gaps; the table of values present is stored
/// in the header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Mapping {
    /// Value of mapped zero.
    pub offset: i32,
    /// Largest mapped value.
    pub max_mapped: i32,
    /// For packed volumes, `values[rank] - offset` for every rank.
    pub table: Option<Vec<i32>>,
}

impl Mapping {
    pub(crate) fn analyze(samples: &[i32], allow_packing: bool) -> Self {
        let lo = samples.iter().copied().min().unwrap_or(0);
        let hi = samples.iter().copied().max().unwrap_or(0);
        let span = (hi - lo) as usize;
        let mut used = vec![false; span + 1];
        for &v in samples {
            used[(v - lo) as usize] = true;
        }
        let distinct = used.iter().filter(|&&u| u).count();
        if allow_packing && span >= 64 && distinct * 4 < (span + 1) * 3 {
            let table: Vec<i32> = used.iter().enumerate().filter(|(_, u)| **u).map(|(i, _)| i as i32).collect();
            Self { offset: lo, max_mapped: distinct as i32 - 1, table: Some(table) }
        } else {
            Self { offset: lo, max_mapped: span as i32, table: None }
        }
    }

    /// Mapped values of `samples`.
    pub(crate) fn map(&self, samples: &[i32]) -> Vec<i32> {
        match &self.table {
            None => samples.iter().map(|&v| v - self.offset).collect(),
            Some(t) => {
                let span = t.last().map_or(0, |&v| v as usize);
                let mut rank = vec![0i32; span + 1];
                for (r, &v) in t.iter().enumerate() {
                    rank[v as usize] = r as i32;
                }
                samples.iter().map(|&v| rank[(v - self.offset) as usize]).collect()
            }
        }
    }

    /// Restores sample values from mapped values (which the decoder has
    /// checked to be within `0..=max_mapped`).
    pub(crate) fn unmap_in_place(&self, mapped: &mut [i32]) {
        match &self.table {
            None => mapped.iter_mut().for_each(|v| *v += self.offset),
            Some(t) => mapped.iter_mut().for_each(|v| *v = t[*v as usize] + self.offset),
        }
    }

    /// Serialises the table of a packed mapping as a range-coded bitmap.
    pub(crate) fn encode_table(&self) -> Vec<u8> {
        let Some(t) = &self.table else { return Vec::new() };
        let span = t.last().map_or(0, |&v| v as usize);
        let mut used = vec![false; span + 1];
        for &v in t {
            used[v as usize] = true;
        }
        let mut enc = Encoder::new();
        enc.encode_bypass(span as u32, 17);
        let mut models = [BitModel::new(); 4];
        let mut ctx = 0usize;
        for &u in &used {
            enc.encode_bit(&mut models[ctx], u);
            ctx = ((ctx << 1) | usize::from(u)) & 3;
        }
        enc.finish()
    }

    /// Reads a table written by [`Mapping::encode_table`].
    pub(crate) fn decode_table(bytes: &[u8], offset: i32, max_mapped: i32) -> Result<Self, Error> {
        let mut dec = Decoder::new(bytes);
        let span = dec.decode_bypass(17) as usize;
        if span > 1 << 16 {
            return Err(Error::Corrupt("packing table span out of range"));
        }
        let mut models = [BitModel::new(); 4];
        let mut ctx = 0usize;
        let mut table = Vec::new();
        for i in 0..=span {
            let u = dec.decode_bit(&mut models[ctx]);
            if u {
                table.push(i as i32);
            }
            ctx = ((ctx << 1) | usize::from(u)) & 3;
        }
        dec.finish().map_err(|_| Error::Corrupt("packing table stream"))?;
        if table.len() != max_mapped as usize + 1 || table.last() != Some(&(span as i32)) || table.first() != Some(&0) {
            return Err(Error::Corrupt("packing table does not match the header"));
        }
        Ok(Self { offset, max_mapped, table: Some(table) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_triggers_on_sparse_values() {
        let samples: Vec<i32> = (0..1000).map(|i| (i % 300) * 2 - 1000).collect();
        let m = Mapping::analyze(&samples, true);
        assert_eq!(m.offset, -1000);
        assert_eq!(m.max_mapped, 299);
        let mut mapped = m.map(&samples);
        assert_eq!(mapped[..3], [0, 1, 2]);
        let table = m.encode_table();
        let back = Mapping::decode_table(&table, m.offset, m.max_mapped).unwrap();
        assert_eq!(back, m);
        back.unmap_in_place(&mut mapped);
        assert_eq!(mapped, samples);
    }

    #[test]
    fn dense_values_are_only_shifted() {
        let samples: Vec<i32> = (0..5000).map(|i| (i * 7919) % 4096 - 2048).collect();
        let m = Mapping::analyze(&samples, true);
        assert!(m.table.is_none());
        assert_eq!((m.offset, m.max_mapped), (-2048, 4095));
    }

    #[test]
    fn volume_validation() {
        assert!(Volume::new(1, 2, 2, 12, false, vec![0, 4095, 1, 2]).is_ok());
        assert!(Volume::new(1, 2, 2, 12, false, vec![0, 4096, 1, 2]).is_err());
        assert!(Volume::new(1, 2, 2, 12, true, vec![-2048, 2047, 0, 0]).is_ok());
        assert!(Volume::new(1, 2, 2, 17, false, vec![0; 4]).is_err());
        assert!(Volume::new(1, 2, 3, 8, false, vec![0; 4]).is_err());
    }
}
