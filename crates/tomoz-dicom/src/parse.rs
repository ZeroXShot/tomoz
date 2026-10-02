//! Data set walker.

use crate::attrs::{Attributes, text};
use crate::tag::{self, Tag};
use crate::transfer_syntax as ts;

/// Deepest sequence nesting accepted. Real files rarely exceed five levels;
/// the bound protects the recursive walk from crafted input.
const MAX_DEPTH: usize = 32;

/// Length value meaning "undefined length, delimited by an item".
const UNDEFINED: u32 = 0xFFFF_FFFF;

/// Errors raised for input that is not a well-formed DICOM data set.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The data ends inside an element.
    #[error("unexpected end of data at offset {offset}")]
    Truncated {
        /// Offset at which more data was needed.
        offset: usize,
    },
    /// An element violates the encoding rules.
    #[error("malformed element {tag} at offset {offset}: {what}")]
    Malformed {
        /// Tag of the offending element.
        tag: Tag,
        /// Offset of the element.
        offset: usize,
        /// What is wrong with it.
        what: &'static str,
    },
    /// Sequences are nested deeper than the parser accepts.
    #[error("sequences nested deeper than {MAX_DEPTH} levels at offset {offset}")]
    TooDeep {
        /// Offset of the innermost item.
        offset: usize,
    },
}

/// How the data set following the file meta information is encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DatasetEncoding {
    /// Implicit VR, little endian.
    ImplicitLittle,
    /// Explicit VR, little endian (also every compressed transfer syntax).
    ExplicitLittle,
    /// Explicit VR, big endian.
    ExplicitBig,
    /// Deflate-compressed explicit VR little endian; the data set is not
    /// walked.
    Deflated,
}

impl DatasetEncoding {
    fn explicit(self) -> bool {
        !matches!(self, Self::ImplicitLittle)
    }

    fn big_endian(self) -> bool {
        matches!(self, Self::ExplicitBig)
    }
}

/// Location of the top-level Pixel Data element in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelSpan {
    /// Offset of the element's tag.
    pub element_start: usize,
    /// Offset of the first value byte.
    pub value_start: usize,
    /// Offset just past the value (for encapsulated pixel data, past the
    /// sequence delimitation item).
    pub value_end: usize,
    /// Whether the pixel data is encapsulated (compressed fragments).
    pub encapsulated: bool,
}

impl PixelSpan {
    /// Length of the value in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.value_end - self.value_start
    }

    /// Whether the value is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.value_end == self.value_start
    }
}

/// Result of [`parse`].
#[derive(Clone, Debug, PartialEq)]
pub struct ParsedFile {
    /// Whether the file starts with the 128-byte preamble and `DICM`.
    pub has_preamble: bool,
    /// Offset of the first data set element after the file meta information.
    pub dataset_start: usize,
    /// Encoding of the data set.
    pub encoding: DatasetEncoding,
    /// The attributes Tomoz reads.
    pub attributes: Attributes,
    /// The top-level Pixel Data element, if any.
    pub pixel: Option<PixelSpan>,
    /// Whether the data set has Float or Double Float Pixel Data, which
    /// Tomoz stores unchanged.
    pub has_float_pixels: bool,
}

/// Parses a DICOM Part 10 file (or a bare data set in implicit VR little
/// endian, as written by old ACR-NEMA style tools).
///
/// # Errors
///
/// Returns an [`Error`] if an element runs past the end of the data, violates
/// the encoding rules, or sequences are nested too deeply.
pub fn parse(bytes: &[u8]) -> Result<ParsedFile, Error> {
    let has_preamble = bytes.get(128..132) == Some(b"DICM");
    let meta_start = if has_preamble {
        132
    } else if bytes.starts_with(&[0x02, 0x00]) {
        0
    } else {
        usize::MAX
    };

    let mut attributes = Attributes::default();
    let (dataset_start, encoding) = if meta_start == usize::MAX {
        (0, guess_encoding(bytes, 0))
    } else {
        let end = parse_meta(bytes, meta_start, &mut attributes)?;
        let encoding = match attributes.transfer_syntax_uid.as_deref() {
            Some(ts::IMPLICIT_VR_LITTLE_ENDIAN) => DatasetEncoding::ImplicitLittle,
            Some(ts::EXPLICIT_VR_BIG_ENDIAN) => DatasetEncoding::ExplicitBig,
            Some(ts::DEFLATED_EXPLICIT_VR_LITTLE_ENDIAN) => DatasetEncoding::Deflated,
            Some(_) => DatasetEncoding::ExplicitLittle,
            None => guess_encoding(bytes, end),
        };
        (end, encoding)
    };

    let mut parsed =
        ParsedFile { has_preamble, dataset_start, encoding, attributes, pixel: None, has_float_pixels: false };
    if encoding != DatasetEncoding::Deflated {
        walk_top_level(bytes, dataset_start, encoding, &mut parsed)?;
    }
    Ok(parsed)
}

/// Parses the group 0002 elements starting at `pos`; returns the offset of the
/// first element of another group. The file meta information is explicit VR
/// little endian; a few broken writers use implicit VR, which is accepted.
fn parse_meta(bytes: &[u8], mut pos: usize, attributes: &mut Attributes) -> Result<usize, Error> {
    let encoding = guess_encoding(bytes, pos);
    while pos < bytes.len() {
        if bytes.get(pos..pos + 2).map(|g| u16::from_le_bytes([g[0], g[1]])) != Some(0x0002) {
            break;
        }
        let h = Header::read(bytes, pos, encoding)?;
        if h.length == UNDEFINED {
            return Err(Error::Malformed {
                tag: h.tag,
                offset: pos,
                what: "undefined length in file meta information",
            });
        }
        let value = value(bytes, &h)?;
        match h.tag {
            tag::TRANSFER_SYNTAX_UID => attributes.transfer_syntax_uid = text(value),
            tag::MEDIA_STORAGE_SOP_CLASS_UID => {
                attributes.sop_class_uid = attributes.sop_class_uid.take().or_else(|| text(value));
            }
            tag::MEDIA_STORAGE_SOP_INSTANCE_UID => {
                attributes.sop_instance_uid = attributes.sop_instance_uid.take().or_else(|| text(value));
            }
            _ => {}
        }
        pos = h.value_start() + value.len();
    }
    Ok(pos)
}

/// Explicit VR if the element at `pos` carries two upper-case letters where
/// an explicit VR would be, implicit VR otherwise.
fn guess_encoding(bytes: &[u8], pos: usize) -> DatasetEncoding {
    match bytes.get(pos + 4..pos + 6) {
        Some(vr) if is_vr(vr) => DatasetEncoding::ExplicitLittle,
        _ => DatasetEncoding::ImplicitLittle,
    }
}

fn is_vr(vr: &[u8]) -> bool {
    vr.len() == 2 && vr[0].is_ascii_uppercase() && vr[1].is_ascii_uppercase()
}

/// VRs whose explicit encoding has two reserved bytes and a 32-bit length.
fn has_long_length(vr: [u8; 2]) -> bool {
    matches!(&vr, b"OB" | b"OD" | b"OF" | b"OL" | b"OV" | b"OW" | b"SQ" | b"UC" | b"UR" | b"UT" | b"UN")
}

/// A decoded element header.
#[derive(Clone, Copy, Debug)]
struct Header {
    tag: Tag,
    vr: Option<[u8; 2]>,
    length: u32,
    start: usize,
    header_len: usize,
}

impl Header {
    fn read(bytes: &[u8], pos: usize, encoding: DatasetEncoding) -> Result<Self, Error> {
        let big = encoding.big_endian();
        let b = bytes.get(pos..pos + 8).ok_or(Error::Truncated { offset: pos })?;
        let u16_at =
            |i: usize| if big { u16::from_be_bytes([b[i], b[i + 1]]) } else { u16::from_le_bytes([b[i], b[i + 1]]) };
        let u32_of = |x: &[u8]| {
            let a = [x[0], x[1], x[2], x[3]];
            if big { u32::from_be_bytes(a) } else { u32::from_le_bytes(a) }
        };
        let tag = Tag(u16_at(0), u16_at(2));
        // Items and delimiters never carry a VR.
        if tag.group() == 0xFFFE || !encoding.explicit() {
            return Ok(Self { tag, vr: None, length: u32_of(&b[4..8]), start: pos, header_len: 8 });
        }
        let vr = [b[4], b[5]];
        if !is_vr(&vr) {
            return Err(Error::Malformed { tag, offset: pos, what: "invalid value representation" });
        }
        if has_long_length(vr) {
            let l = bytes.get(pos + 8..pos + 12).ok_or(Error::Truncated { offset: pos + 8 })?;
            Ok(Self { tag, vr: Some(vr), length: u32_of(l), start: pos, header_len: 12 })
        } else {
            Ok(Self { tag, vr: Some(vr), length: u32::from(u16_at(6)), start: pos, header_len: 8 })
        }
    }

    fn value_start(&self) -> usize {
        self.start + self.header_len
    }

    /// End of the value; `None` if it does not fit in the address space.
    fn value_end(&self) -> Option<usize> {
        self.value_start().checked_add(self.length as usize)
    }

    /// Encoding of the items of an undefined-length sequence opened by this
    /// element: UN contents are always implicit VR little endian.
    fn nested(&self, encoding: DatasetEncoding) -> DatasetEncoding {
        if self.vr == Some(*b"UN") { DatasetEncoding::ImplicitLittle } else { encoding }
    }
}

fn value<'a>(bytes: &'a [u8], h: &Header) -> Result<&'a [u8], Error> {
    h.value_end().and_then(|end| bytes.get(h.value_start()..end)).ok_or(Error::Truncated { offset: h.value_start() })
}

/// Offset after a defined-length element whose value lies within `bytes`.
fn skip(bytes: &[u8], h: &Header) -> Result<usize, Error> {
    value(bytes, h).map(|v| h.value_start() + v.len())
}

fn walk_top_level(bytes: &[u8], mut pos: usize, encoding: DatasetEncoding, out: &mut ParsedFile) -> Result<(), Error> {
    let big = encoding.big_endian();
    while pos < bytes.len() {
        let h = Header::read(bytes, pos, encoding)?;
        if h.tag.group() == 0xFFFE {
            return Err(Error::Malformed { tag: h.tag, offset: pos, what: "delimiter outside a sequence" });
        }
        let is_pixels = h.tag == tag::PIXEL_DATA;
        if matches!(h.tag, tag::FLOAT_PIXEL_DATA | tag::DOUBLE_FLOAT_PIXEL_DATA) {
            out.has_float_pixels = true;
        }
        if h.length == UNDEFINED {
            let end = if is_pixels {
                skip_fragments(bytes, h.value_start(), encoding)?
            } else {
                skip_sequence(bytes, h.value_start(), h.nested(encoding), 1)?
            };
            if is_pixels && out.pixel.is_none() {
                out.pixel = Some(PixelSpan {
                    element_start: h.start,
                    value_start: h.value_start(),
                    value_end: end,
                    encapsulated: true,
                });
            }
            pos = end;
        } else {
            let v = value(bytes, &h)?;
            if Attributes::wants(h.tag) {
                out.attributes.set(h.tag, v, big);
            }
            let end = h.value_start() + v.len();
            if is_pixels && out.pixel.is_none() {
                out.pixel = Some(PixelSpan {
                    element_start: h.start,
                    value_start: h.value_start(),
                    value_end: end,
                    encapsulated: false,
                });
            }
            pos = end;
        }
    }
    Ok(())
}

/// Skips the items of an undefined-length sequence starting at `pos`; returns
/// the offset after its sequence delimitation item.
fn skip_sequence(bytes: &[u8], mut pos: usize, encoding: DatasetEncoding, depth: usize) -> Result<usize, Error> {
    if depth > MAX_DEPTH {
        return Err(Error::TooDeep { offset: pos });
    }
    loop {
        let h = Header::read(bytes, pos, encoding)?;
        match h.tag {
            tag::SEQUENCE_DELIMITATION => return Ok(h.value_start()),
            tag::ITEM if h.length == UNDEFINED => pos = skip_item(bytes, h.value_start(), encoding, depth + 1)?,
            tag::ITEM => pos = skip(bytes, &h)?,
            _ => return Err(Error::Malformed { tag: h.tag, offset: pos, what: "expected an item in a sequence" }),
        }
    }
}

/// Skips the elements of an undefined-length item; returns the offset after
/// its item delimitation item.
fn skip_item(bytes: &[u8], mut pos: usize, encoding: DatasetEncoding, depth: usize) -> Result<usize, Error> {
    if depth > MAX_DEPTH {
        return Err(Error::TooDeep { offset: pos });
    }
    loop {
        let h = Header::read(bytes, pos, encoding)?;
        if h.tag == tag::ITEM_DELIMITATION {
            return Ok(h.value_start());
        }
        if h.tag.group() == 0xFFFE {
            return Err(Error::Malformed { tag: h.tag, offset: pos, what: "unexpected delimiter in an item" });
        }
        pos = if h.length == UNDEFINED {
            skip_sequence(bytes, h.value_start(), h.nested(encoding), depth + 1)?
        } else {
            skip(bytes, &h)?
        };
    }
}

/// Skips the fragments of encapsulated pixel data; returns the offset after
/// the sequence delimitation item.
fn skip_fragments(bytes: &[u8], mut pos: usize, encoding: DatasetEncoding) -> Result<usize, Error> {
    loop {
        let h = Header::read(bytes, pos, encoding)?;
        match h.tag {
            tag::SEQUENCE_DELIMITATION => return Ok(h.value_start()),
            tag::ITEM if h.length != UNDEFINED => pos = skip(bytes, &h)?,
            _ => return Err(Error::Malformed { tag: h.tag, offset: pos, what: "expected a pixel data fragment" }),
        }
    }
}
