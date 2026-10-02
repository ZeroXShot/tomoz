//! The attributes Tomoz reads from a data set.

use crate::tag::{self, Tag};

/// Top-level attributes that describe the pixel data of an instance and its
/// place in a series. Absent or unparsable values are `None`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Attributes {
    /// Transfer Syntax UID from the file meta information.
    pub transfer_syntax_uid: Option<String>,
    /// SOP Class UID.
    pub sop_class_uid: Option<String>,
    /// SOP Instance UID.
    pub sop_instance_uid: Option<String>,
    /// Modality.
    pub modality: Option<String>,
    /// Image Type values.
    pub image_type: Vec<String>,
    /// Study Instance UID.
    pub study_instance_uid: Option<String>,
    /// Series Instance UID.
    pub series_instance_uid: Option<String>,
    /// Instance Number.
    pub instance_number: Option<i64>,
    /// Acquisition Number.
    pub acquisition_number: Option<i64>,
    /// First value of Echo Numbers.
    pub echo_number: Option<i64>,
    /// Temporal Position Identifier.
    pub temporal_position: Option<i64>,
    /// Image Position (Patient), in millimetres.
    pub image_position: Option<[f64; 3]>,
    /// Image Orientation (Patient): row then column direction cosines.
    pub image_orientation: Option<[f64; 6]>,
    /// Pixel Spacing (row spacing, column spacing), in millimetres.
    pub pixel_spacing: Option<[f64; 2]>,
    /// Slice Thickness, in millimetres.
    pub slice_thickness: Option<f64>,
    /// Spacing Between Slices, in millimetres.
    pub spacing_between_slices: Option<f64>,
    /// Samples per Pixel.
    pub samples_per_pixel: Option<u16>,
    /// Photometric Interpretation.
    pub photometric_interpretation: Option<String>,
    /// Planar Configuration.
    pub planar_configuration: Option<u16>,
    /// Number of Frames.
    pub number_of_frames: Option<u32>,
    /// Rows.
    pub rows: Option<u16>,
    /// Columns.
    pub columns: Option<u16>,
    /// Bits Allocated.
    pub bits_allocated: Option<u16>,
    /// Bits Stored.
    pub bits_stored: Option<u16>,
    /// High Bit.
    pub high_bit: Option<u16>,
    /// Pixel Representation (0 unsigned, 1 two's complement).
    pub pixel_representation: Option<u16>,
    /// Rescale Intercept.
    pub rescale_intercept: Option<f64>,
    /// Rescale Slope.
    pub rescale_slope: Option<f64>,
}

impl Attributes {
    /// Records the value of `tag` if it is one Tomoz reads. `big_endian`
    /// applies to binary values.
    pub(crate) fn set(&mut self, tag: Tag, value: &[u8], big_endian: bool) {
        match tag {
            tag::SOP_CLASS_UID => self.sop_class_uid = text(value),
            tag::SOP_INSTANCE_UID => self.sop_instance_uid = text(value),
            tag::MODALITY => self.modality = text(value),
            tag::IMAGE_TYPE => self.image_type = texts(value),
            tag::STUDY_INSTANCE_UID => self.study_instance_uid = text(value),
            tag::SERIES_INSTANCE_UID => self.series_instance_uid = text(value),
            tag::INSTANCE_NUMBER => self.instance_number = integer(value),
            tag::ACQUISITION_NUMBER => self.acquisition_number = integer(value),
            tag::ECHO_NUMBERS => self.echo_number = integer(value),
            tag::TEMPORAL_POSITION_IDENTIFIER => self.temporal_position = integer(value),
            tag::IMAGE_POSITION_PATIENT => self.image_position = decimals(value),
            tag::IMAGE_ORIENTATION_PATIENT => self.image_orientation = decimals(value),
            tag::PIXEL_SPACING => self.pixel_spacing = decimals(value),
            tag::SLICE_THICKNESS => self.slice_thickness = decimals::<1>(value).map(|v| v[0]),
            tag::SPACING_BETWEEN_SLICES => self.spacing_between_slices = decimals::<1>(value).map(|v| v[0]),
            tag::SAMPLES_PER_PIXEL => self.samples_per_pixel = unsigned(value, big_endian),
            tag::PHOTOMETRIC_INTERPRETATION => self.photometric_interpretation = text(value),
            tag::PLANAR_CONFIGURATION => self.planar_configuration = unsigned(value, big_endian),
            tag::NUMBER_OF_FRAMES => {
                self.number_of_frames = integer(value).and_then(|v| u32::try_from(v).ok());
            }
            tag::ROWS => self.rows = unsigned(value, big_endian),
            tag::COLUMNS => self.columns = unsigned(value, big_endian),
            tag::BITS_ALLOCATED => self.bits_allocated = unsigned(value, big_endian),
            tag::BITS_STORED => self.bits_stored = unsigned(value, big_endian),
            tag::HIGH_BIT => self.high_bit = unsigned(value, big_endian),
            tag::PIXEL_REPRESENTATION => self.pixel_representation = unsigned(value, big_endian),
            tag::RESCALE_INTERCEPT => self.rescale_intercept = decimals::<1>(value).map(|v| v[0]),
            tag::RESCALE_SLOPE => self.rescale_slope = decimals::<1>(value).map(|v| v[0]),
            _ => {}
        }
    }

    /// Whether `tag` is read by [`Attributes::set`].
    pub(crate) fn wants(tag: Tag) -> bool {
        matches!(
            tag,
            tag::SOP_CLASS_UID
                | tag::SOP_INSTANCE_UID
                | tag::MODALITY
                | tag::IMAGE_TYPE
                | tag::STUDY_INSTANCE_UID
                | tag::SERIES_INSTANCE_UID
                | tag::INSTANCE_NUMBER
                | tag::ACQUISITION_NUMBER
                | tag::ECHO_NUMBERS
                | tag::TEMPORAL_POSITION_IDENTIFIER
                | tag::IMAGE_POSITION_PATIENT
                | tag::IMAGE_ORIENTATION_PATIENT
                | tag::PIXEL_SPACING
                | tag::SLICE_THICKNESS
                | tag::SPACING_BETWEEN_SLICES
                | tag::SAMPLES_PER_PIXEL
                | tag::PHOTOMETRIC_INTERPRETATION
                | tag::PLANAR_CONFIGURATION
                | tag::NUMBER_OF_FRAMES
                | tag::ROWS
                | tag::COLUMNS
                | tag::BITS_ALLOCATED
                | tag::BITS_STORED
                | tag::HIGH_BIT
                | tag::PIXEL_REPRESENTATION
                | tag::RESCALE_INTERCEPT
                | tag::RESCALE_SLOPE
        )
    }
}

/// A string value without the trailing padding (space or NUL) and without
/// leading spaces. Empty values are `None`.
pub(crate) fn text(value: &[u8]) -> Option<String> {
    let s = String::from_utf8_lossy(value);
    let s = s.trim_end_matches(['\0', ' ']).trim_start_matches(' ');
    (!s.is_empty()).then(|| s.to_owned())
}

fn texts(value: &[u8]) -> Vec<String> {
    String::from_utf8_lossy(value)
        .trim_end_matches(['\0', ' '])
        .split('\\')
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect()
}

fn integer(value: &[u8]) -> Option<i64> {
    let s = core::str::from_utf8(value).ok()?;
    s.split('\\').next()?.trim_matches(|c: char| c == ' ' || c == '\0').parse().ok()
}

fn decimals<const N: usize>(value: &[u8]) -> Option<[f64; N]> {
    let s = core::str::from_utf8(value).ok()?;
    let mut out = [0.0; N];
    let mut parts = s.trim_end_matches(['\0', ' ']).split('\\');
    for v in &mut out {
        let x: f64 = parts.next()?.trim().parse().ok()?;
        if !x.is_finite() {
            return None;
        }
        *v = x;
    }
    Some(out)
}

fn unsigned(value: &[u8], big_endian: bool) -> Option<u16> {
    let b: [u8; 2] = value.get(..2)?.try_into().ok()?;
    Some(if big_endian { u16::from_be_bytes(b) } else { u16::from_le_bytes(b) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_decoding() {
        assert_eq!(text(b"1.2.3\0"), Some("1.2.3".into()));
        assert_eq!(text(b"  CT "), Some("CT".into()));
        assert_eq!(text(b"   "), None);
        assert_eq!(texts(b"ORIGINAL\\PRIMARY\\AXIAL "), vec!["ORIGINAL", "PRIMARY", "AXIAL"]);
        assert_eq!(integer(b" 12 "), Some(12));
        assert_eq!(integer(b"3\\4"), Some(3));
        assert_eq!(integer(b"x"), None);
        assert_eq!(decimals::<3>(b"-1.5\\2e1\\ 3 "), Some([-1.5, 20.0, 3.0]));
        assert_eq!(decimals::<3>(b"1\\2"), None);
        assert_eq!(decimals::<1>(b"nan"), None);
        assert_eq!(unsigned(&[0x00, 0x02], false), Some(512));
        assert_eq!(unsigned(&[0x02, 0x00], true), Some(512));
        assert_eq!(unsigned(&[0x02], true), None);
    }
}
