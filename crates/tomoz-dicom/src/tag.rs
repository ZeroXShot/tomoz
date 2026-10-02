//! Data element tags used by Tomoz.

use core::fmt;

/// A data element tag: (group, element).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Tag(pub u16, pub u16);

impl Tag {
    /// The group number.
    #[must_use]
    pub const fn group(self) -> u16 {
        self.0
    }

    /// The element number.
    #[must_use]
    pub const fn element(self) -> u16 {
        self.1
    }
}

impl fmt::Debug for Tag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:04X},{:04X})", self.0, self.1)
    }
}

impl fmt::Display for Tag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// Item.
pub const ITEM: Tag = Tag(0xFFFE, 0xE000);
/// Item Delimitation Item.
pub const ITEM_DELIMITATION: Tag = Tag(0xFFFE, 0xE00D);
/// Sequence Delimitation Item.
pub const SEQUENCE_DELIMITATION: Tag = Tag(0xFFFE, 0xE0DD);

/// Transfer Syntax UID (file meta information).
pub const TRANSFER_SYNTAX_UID: Tag = Tag(0x0002, 0x0010);
/// Media Storage SOP Class UID.
pub const MEDIA_STORAGE_SOP_CLASS_UID: Tag = Tag(0x0002, 0x0002);
/// Media Storage SOP Instance UID.
pub const MEDIA_STORAGE_SOP_INSTANCE_UID: Tag = Tag(0x0002, 0x0003);

/// Image Type.
pub const IMAGE_TYPE: Tag = Tag(0x0008, 0x0008);
/// SOP Class UID.
pub const SOP_CLASS_UID: Tag = Tag(0x0008, 0x0016);
/// SOP Instance UID.
pub const SOP_INSTANCE_UID: Tag = Tag(0x0008, 0x0018);
/// Modality.
pub const MODALITY: Tag = Tag(0x0008, 0x0060);
/// Slice Thickness.
pub const SLICE_THICKNESS: Tag = Tag(0x0018, 0x0050);
/// Echo Numbers.
pub const ECHO_NUMBERS: Tag = Tag(0x0018, 0x0086);
/// Spacing Between Slices.
pub const SPACING_BETWEEN_SLICES: Tag = Tag(0x0018, 0x0088);
/// Study Instance UID.
pub const STUDY_INSTANCE_UID: Tag = Tag(0x0020, 0x000D);
/// Series Instance UID.
pub const SERIES_INSTANCE_UID: Tag = Tag(0x0020, 0x000E);
/// Acquisition Number.
pub const ACQUISITION_NUMBER: Tag = Tag(0x0020, 0x0012);
/// Instance Number.
pub const INSTANCE_NUMBER: Tag = Tag(0x0020, 0x0013);
/// Image Position (Patient).
pub const IMAGE_POSITION_PATIENT: Tag = Tag(0x0020, 0x0032);
/// Image Orientation (Patient).
pub const IMAGE_ORIENTATION_PATIENT: Tag = Tag(0x0020, 0x0037);
/// Temporal Position Identifier.
pub const TEMPORAL_POSITION_IDENTIFIER: Tag = Tag(0x0020, 0x0100);
/// Samples per Pixel.
pub const SAMPLES_PER_PIXEL: Tag = Tag(0x0028, 0x0002);
/// Photometric Interpretation.
pub const PHOTOMETRIC_INTERPRETATION: Tag = Tag(0x0028, 0x0004);
/// Planar Configuration.
pub const PLANAR_CONFIGURATION: Tag = Tag(0x0028, 0x0006);
/// Number of Frames.
pub const NUMBER_OF_FRAMES: Tag = Tag(0x0028, 0x0008);
/// Rows.
pub const ROWS: Tag = Tag(0x0028, 0x0010);
/// Columns.
pub const COLUMNS: Tag = Tag(0x0028, 0x0011);
/// Pixel Spacing.
pub const PIXEL_SPACING: Tag = Tag(0x0028, 0x0030);
/// Bits Allocated.
pub const BITS_ALLOCATED: Tag = Tag(0x0028, 0x0100);
/// Bits Stored.
pub const BITS_STORED: Tag = Tag(0x0028, 0x0101);
/// High Bit.
pub const HIGH_BIT: Tag = Tag(0x0028, 0x0102);
/// Pixel Representation.
pub const PIXEL_REPRESENTATION: Tag = Tag(0x0028, 0x0103);
/// Rescale Intercept.
pub const RESCALE_INTERCEPT: Tag = Tag(0x0028, 0x1052);
/// Rescale Slope.
pub const RESCALE_SLOPE: Tag = Tag(0x0028, 0x1053);
/// Float Pixel Data.
pub const FLOAT_PIXEL_DATA: Tag = Tag(0x7FE0, 0x0008);
/// Double Float Pixel Data.
pub const DOUBLE_FLOAT_PIXEL_DATA: Tag = Tag(0x7FE0, 0x0009);
/// Pixel Data.
pub const PIXEL_DATA: Tag = Tag(0x7FE0, 0x0010);
