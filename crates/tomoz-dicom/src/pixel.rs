//! Native pixel data: conversion to integer samples and back.

use crate::attrs::Attributes;
use crate::parse::DatasetEncoding;

/// Why pixel data cannot be converted to samples.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LayoutError {
    /// A required attribute is missing.
    #[error("missing attribute {0}")]
    Missing(&'static str),
    /// The pixel data uses a layout Tomoz does not convert.
    #[error("unsupported pixel data: {0}")]
    Unsupported(&'static str),
    /// The pixel data is shorter than its attributes imply.
    #[error("pixel data has {actual} bytes, {expected} expected")]
    Length {
        /// Bytes implied by rows, columns, frames and bits allocated.
        expected: usize,
        /// Bytes present.
        actual: usize,
    },
}

/// How the bits of each pixel cell outside the stored value are filled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitConvention {
    /// All other bits are zero.
    Zero,
    /// Bits above the high bit replicate the sign; bits below are zero.
    SignExtend,
    /// Other bits carry data (old overlays, or garbage); the difference from
    /// [`BitConvention::Zero`] is kept per sample.
    Irregular,
}

/// Geometry and bit layout of native (uncompressed) grayscale pixel data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelLayout {
    /// Rows per frame.
    pub rows: u32,
    /// Columns per frame.
    pub columns: u32,
    /// Number of frames.
    pub frames: u32,
    /// Bits per pixel cell: 8 or 16.
    pub bits_allocated: u8,
    /// Bits of the stored value: 1 to `bits_allocated`.
    pub bits_stored: u8,
    /// Most significant bit of the stored value.
    pub high_bit: u8,
    /// Whether values are two's complement.
    pub signed: bool,
    /// Whether 16-bit cells are big endian.
    pub big_endian: bool,
}

/// Pixel data split into values and the information needed to restore the
/// exact bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Samples {
    /// Stored values, frame by frame, row by row.
    pub values: Vec<i32>,
    /// How the remaining bits of each cell are filled.
    pub convention: BitConvention,
    /// For [`BitConvention::Irregular`]: each cell XOR its
    /// [`BitConvention::Zero`] reconstruction.
    pub irregular: Option<Vec<u16>>,
    /// Bytes of the value after the last cell (the padding byte of an odd
    /// length, or anything else that was there).
    pub trailing: Vec<u8>,
}

impl PixelLayout {
    /// The layout described by `a`, for a data set encoded with `encoding`.
    ///
    /// # Errors
    ///
    /// [`LayoutError::Missing`] if an attribute without a default is absent and
    /// [`LayoutError::Unsupported`] for layouts other than one-sample 8- or
    /// 16-bit cells.
    pub fn from_attributes(a: &Attributes, encoding: DatasetEncoding) -> Result<Self, LayoutError> {
        if a.samples_per_pixel.unwrap_or(1) != 1 {
            return Err(LayoutError::Unsupported("more than one sample per pixel"));
        }
        let rows = u32::from(a.rows.ok_or(LayoutError::Missing("Rows"))?);
        let columns = u32::from(a.columns.ok_or(LayoutError::Missing("Columns"))?);
        let bits_allocated = a.bits_allocated.ok_or(LayoutError::Missing("BitsAllocated"))?;
        if bits_allocated != 8 && bits_allocated != 16 {
            return Err(LayoutError::Unsupported("bits allocated other than 8 or 16"));
        }
        let bits_stored = a.bits_stored.unwrap_or(bits_allocated);
        let high_bit = a.high_bit.unwrap_or(bits_stored.saturating_sub(1));
        if bits_stored == 0 || bits_stored > bits_allocated || high_bit >= bits_allocated || high_bit + 1 < bits_stored
        {
            return Err(LayoutError::Unsupported("inconsistent bits stored and high bit"));
        }
        let frames = a.number_of_frames.unwrap_or(1).max(1);
        if rows == 0 || columns == 0 {
            return Err(LayoutError::Unsupported("empty frame"));
        }
        // Declared sizes are untrusted: the byte count must be addressable
        // (usize is 32 bits on WebAssembly) so that it cannot overflow later.
        let bytes = u64::from(rows * columns)
            .checked_mul(u64::from(frames))
            .and_then(|n| n.checked_mul(u64::from(bits_allocated / 8)))
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| isize::try_from(n).is_ok());
        if bytes.is_none() {
            return Err(LayoutError::Unsupported("pixel data larger than addressable memory"));
        }
        Ok(Self {
            rows,
            columns,
            frames,
            bits_allocated: bits_allocated as u8,
            bits_stored: bits_stored as u8,
            high_bit: high_bit as u8,
            signed: a.pixel_representation == Some(1),
            big_endian: encoding == DatasetEncoding::ExplicitBig,
        })
    }

    /// Number of samples per frame.
    #[must_use]
    pub fn frame_samples(&self) -> usize {
        self.rows as usize * self.columns as usize
    }

    /// Number of samples in all frames.
    #[must_use]
    pub fn samples(&self) -> usize {
        self.frame_samples() * self.frames as usize
    }

    /// Bytes per cell.
    #[must_use]
    pub fn cell_bytes(&self) -> usize {
        usize::from(self.bits_allocated / 8)
    }

    /// Bytes of all cells, without padding.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.samples() * self.cell_bytes()
    }

    /// Smallest and largest representable value.
    #[must_use]
    pub fn value_range(&self) -> (i32, i32) {
        let b = u32::from(self.bits_stored);
        if self.signed { (-(1 << (b - 1)), (1 << (b - 1)) - 1) } else { (0, (1 << b) - 1) }
    }

    fn shift(&self) -> u32 {
        u32::from(self.high_bit + 1 - self.bits_stored)
    }

    fn mask(&self) -> u32 {
        (1u32 << self.bits_stored) - 1
    }

    #[inline]
    fn read_cell(&self, value: &[u8], i: usize) -> u16 {
        if self.bits_allocated == 8 {
            u16::from(value[i])
        } else {
            let b = [value[2 * i], value[2 * i + 1]];
            if self.big_endian { u16::from_be_bytes(b) } else { u16::from_le_bytes(b) }
        }
    }

    #[inline]
    fn decode(&self, cell: u16) -> i32 {
        let raw = (u32::from(cell) >> self.shift()) & self.mask();
        let b = u32::from(self.bits_stored);
        if self.signed && raw >> (b - 1) != 0 { raw as i32 - (1 << b) } else { raw as i32 }
    }

    #[inline]
    fn encode_zero(&self, v: i32) -> u16 {
        (((v as u32) & self.mask()) << self.shift()) as u16
    }

    #[inline]
    fn encode_sign(&self, v: i32) -> u16 {
        let cell = self.encode_zero(v);
        if v < 0 {
            let width = u32::from(self.bits_allocated);
            let above = ((1u32 << width) - 1) & !((1u32 << (u32::from(self.high_bit) + 1)) - 1);
            cell | above as u16
        } else {
            cell
        }
    }

    /// Splits the value of a native Pixel Data element into samples.
    ///
    /// # Errors
    ///
    /// [`LayoutError::Length`] if the value is shorter than the cells.
    pub fn extract(&self, value: &[u8]) -> Result<Samples, LayoutError> {
        let n = self.samples();
        let len = self.byte_len();
        if value.len() < len {
            return Err(LayoutError::Length { expected: len, actual: value.len() });
        }
        let mut values = Vec::with_capacity(n);
        let mut zero = true;
        let mut sign = self.signed;
        for i in 0..n {
            let cell = self.read_cell(value, i);
            let v = self.decode(cell);
            zero &= cell == self.encode_zero(v);
            sign &= cell == self.encode_sign(v);
            values.push(v);
        }
        let (convention, irregular) = if zero {
            (BitConvention::Zero, None)
        } else if sign {
            (BitConvention::SignExtend, None)
        } else {
            let plane =
                values.iter().enumerate().map(|(i, &v)| self.read_cell(value, i) ^ self.encode_zero(v)).collect();
            (BitConvention::Irregular, Some(plane))
        };
        Ok(Samples { values, convention, irregular, trailing: value[len..].to_vec() })
    }

    /// Rebuilds the exact bytes of a Pixel Data value from its samples.
    ///
    /// # Panics
    ///
    /// Panics if the number of values (or of irregular cells) does not match
    /// the layout, or if an irregular convention comes without its plane.
    #[must_use]
    pub fn reconstruct(&self, samples: &Samples) -> Vec<u8> {
        assert_eq!(samples.values.len(), self.samples(), "sample count does not match the layout");
        let mut out = Vec::with_capacity(self.byte_len() + samples.trailing.len());
        let plane = match samples.convention {
            BitConvention::Irregular => {
                let p = samples.irregular.as_deref().expect("irregular convention without its plane");
                assert_eq!(p.len(), samples.values.len(), "irregular plane does not match the layout");
                Some(p)
            }
            _ => None,
        };
        for (i, &v) in samples.values.iter().enumerate() {
            let cell = match samples.convention {
                BitConvention::Zero => self.encode_zero(v),
                BitConvention::SignExtend => self.encode_sign(v),
                BitConvention::Irregular => self.encode_zero(v) ^ plane.map_or(0, |p| p[i]),
            };
            if self.bits_allocated == 8 {
                out.push(cell as u8);
            } else if self.big_endian {
                out.extend_from_slice(&cell.to_be_bytes());
            } else {
                out.extend_from_slice(&cell.to_le_bytes());
            }
        }
        out.extend_from_slice(&samples.trailing);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn layout(bits_allocated: u8, bits_stored: u8, high_bit: u8, signed: bool, big_endian: bool) -> PixelLayout {
        PixelLayout { rows: 3, columns: 5, frames: 2, bits_allocated, bits_stored, high_bit, signed, big_endian }
    }

    fn cells_to_bytes(l: &PixelLayout, cells: &[u16]) -> Vec<u8> {
        cells
            .iter()
            .flat_map(|&c| {
                if l.bits_allocated == 8 {
                    vec![c as u8]
                } else if l.big_endian {
                    c.to_be_bytes().to_vec()
                } else {
                    c.to_le_bytes().to_vec()
                }
            })
            .collect()
    }

    #[test]
    fn signed_12_bit_in_16_with_sign_extension() {
        let l = layout(16, 12, 11, true, false);
        let values: Vec<i32> = (0..30).map(|i| i * 137 - 2048).collect();
        let cells: Vec<u16> = values.iter().map(|&v| v as i16 as u16).collect();
        let s = l.extract(&cells_to_bytes(&l, &cells)).unwrap();
        assert_eq!(s.values, values);
        assert_eq!(s.convention, BitConvention::SignExtend);
        assert_eq!(l.reconstruct(&s), cells_to_bytes(&l, &cells));
    }

    #[test]
    fn overlay_bits_are_preserved() {
        let l = layout(16, 12, 11, false, false);
        let cells: Vec<u16> = (0..30u16).map(|i| (i * 100) | if i % 7 == 0 { 0x8000 } else { 0 }).collect();
        let bytes = cells_to_bytes(&l, &cells);
        let s = l.extract(&bytes).unwrap();
        assert_eq!(s.convention, BitConvention::Irregular);
        assert_eq!(s.values[7], 700);
        assert_eq!(l.reconstruct(&s), bytes);
    }

    #[test]
    fn odd_length_padding_is_kept() {
        let l = PixelLayout { rows: 3, columns: 3, frames: 1, ..layout(8, 8, 7, false, false) };
        let mut bytes: Vec<u8> = (0..9).collect();
        bytes.push(0);
        let s = l.extract(&bytes).unwrap();
        assert_eq!(s.trailing, vec![0]);
        assert_eq!(l.reconstruct(&s), bytes);
        assert_eq!(l.extract(&bytes[..8]), Err(LayoutError::Length { expected: 9, actual: 8 }));
    }

    proptest! {
        #[test]
        fn any_cells_roundtrip(
            bits_allocated in prop::sample::select(vec![8u8, 16]),
            stored_frac in 0.0f64..1.0,
            hb_frac in 0.0f64..1.0,
            signed in any::<bool>(),
            big_endian in any::<bool>(),
            cells in prop::collection::vec(any::<u16>(), 30),
            trailing in prop::collection::vec(any::<u8>(), 0..3),
        ) {
            let bits_stored = 1 + ((f64::from(bits_allocated) - 1.0) * stored_frac).round() as u8;
            let high_bit = bits_stored - 1 + ((f64::from(bits_allocated - bits_stored)) * hb_frac).floor() as u8;
            let l = layout(bits_allocated, bits_stored, high_bit, signed, big_endian);
            let cells: Vec<u16> = cells.iter().map(|&c| if bits_allocated == 8 { c & 0xFF } else { c }).collect();
            let mut bytes = cells_to_bytes(&l, &cells);
            bytes.extend_from_slice(&trailing);
            let s = l.extract(&bytes).unwrap();
            let (lo, hi) = l.value_range();
            prop_assert!(s.values.iter().all(|&v| (lo..=hi).contains(&v)));
            prop_assert_eq!(l.reconstruct(&s), bytes);
        }
    }
}
