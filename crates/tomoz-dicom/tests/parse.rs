//! Parsing synthetic files in every supported encoding.

use proptest::prelude::*;
use tomoz_dicom::{BitConvention, DatasetEncoding, Error, PixelLayout, parse, tag, transfer_syntax as ts};

/// Builds data sets element by element.
struct Builder {
    explicit: bool,
    big: bool,
    out: Vec<u8>,
}

impl Builder {
    fn new(explicit: bool, big: bool) -> Self {
        Self { explicit, big, out: Vec::new() }
    }

    fn u16(&mut self, v: u16) {
        self.out.extend_from_slice(&if self.big { v.to_be_bytes() } else { v.to_le_bytes() });
    }

    fn u32(&mut self, v: u32) {
        self.out.extend_from_slice(&if self.big { v.to_be_bytes() } else { v.to_le_bytes() });
    }

    fn header(&mut self, t: tomoz_dicom::Tag, vr: &[u8; 2], len: u32) {
        self.u16(t.0);
        self.u16(t.1);
        if t.0 == 0xFFFE || !self.explicit {
            self.u32(len);
        } else if matches!(vr, b"OB" | b"OW" | b"SQ" | b"UN" | b"UT") {
            self.out.extend_from_slice(vr);
            self.out.extend_from_slice(&[0, 0]);
            self.u32(len);
        } else {
            self.out.extend_from_slice(vr);
            self.u16(len as u16);
        }
    }

    fn element(&mut self, t: tomoz_dicom::Tag, vr: &[u8; 2], value: &[u8]) -> &mut Self {
        let mut v = value.to_vec();
        if v.len() % 2 == 1 {
            v.push(if matches!(vr, b"UI" | b"OB") { 0 } else { b' ' });
        }
        self.header(t, vr, v.len() as u32);
        self.out.extend_from_slice(&v);
        self
    }

    fn us(&mut self, t: tomoz_dicom::Tag, v: u16) -> &mut Self {
        let b = if self.big { v.to_be_bytes() } else { v.to_le_bytes() };
        self.element(t, b"US", &b)
    }

    fn raw(&mut self, bytes: &[u8]) -> &mut Self {
        self.out.extend_from_slice(bytes);
        self
    }

    fn delimiter(&mut self, t: tomoz_dicom::Tag) -> &mut Self {
        self.header(t, b"  ", 0);
        self
    }
}

fn meta(transfer_syntax: &str) -> Vec<u8> {
    let mut m = Builder::new(true, false);
    m.element(tomoz_dicom::Tag(0x0002, 0x0001), b"OB", &[0, 1])
        .element(tag::MEDIA_STORAGE_SOP_CLASS_UID, b"UI", b"1.2.840.10008.5.1.4.1.1.2")
        .element(tag::MEDIA_STORAGE_SOP_INSTANCE_UID, b"UI", b"1.2.3.4.5")
        .element(tag::TRANSFER_SYNTAX_UID, b"UI", transfer_syntax.as_bytes());
    let mut out = vec![0u8; 128];
    out.extend_from_slice(b"DICM");
    let mut g = Builder::new(true, false);
    g.element(tomoz_dicom::Tag(0x0002, 0x0000), b"UL", &(m.out.len() as u32).to_le_bytes());
    out.extend_from_slice(&g.out);
    out.extend_from_slice(&m.out);
    out
}

/// A 4x3 CT-like image with a nested sequence before the pixel data and
/// trailing padding after it.
fn dataset(explicit: bool, big: bool, encapsulated: bool) -> (Vec<u8>, Vec<u8>) {
    let mut b = Builder::new(explicit, big);
    b.element(tag::SOP_CLASS_UID, b"UI", b"1.2.840.10008.5.1.4.1.1.2")
        .element(tag::SOP_INSTANCE_UID, b"UI", b"1.2.3.4.5")
        .element(tag::MODALITY, b"CS", b"CT")
        .element(tag::IMAGE_TYPE, b"CS", b"ORIGINAL\\PRIMARY\\AXIAL");
    // An undefined-length sequence with one undefined-length item holding a
    // defined-length sequence, and a second defined-length item.
    b.header(tomoz_dicom::Tag(0x0008, 0x1140), b"SQ", 0xFFFF_FFFF);
    b.header(tag::ITEM, b"  ", 0xFFFF_FFFF);
    b.element(tomoz_dicom::Tag(0x0008, 0x1150), b"UI", b"1.2.3");
    let mut inner = Builder::new(explicit, big);
    inner.element(tomoz_dicom::Tag(0x0008, 0x0100), b"SH", b"CODE");
    let mut item = Builder::new(explicit, big);
    item.header(tag::ITEM, b"  ", inner.out.len() as u32);
    item.raw(&inner.out.clone());
    b.header(tomoz_dicom::Tag(0x0040, 0xA730), b"SQ", item.out.len() as u32);
    b.raw(&item.out);
    b.delimiter(tag::ITEM_DELIMITATION);
    let mut item2 = Builder::new(explicit, big);
    item2.element(tomoz_dicom::Tag(0x0008, 0x1155), b"UI", b"1.2.4");
    b.header(tag::ITEM, b"  ", item2.out.len() as u32);
    b.raw(&item2.out);
    b.delimiter(tag::SEQUENCE_DELIMITATION);
    if explicit && !big {
        // A private UN element with undefined length: its items are implicit VR.
        b.header(tomoz_dicom::Tag(0x0029, 0x1010), b"UN", 0xFFFF_FFFF);
        let mut un = Builder::new(false, false);
        un.header(tag::ITEM, b"  ", 0xFFFF_FFFF);
        un.element(tomoz_dicom::Tag(0x0029, 0x1001), b"LO", b"PRIVATE");
        un.delimiter(tag::ITEM_DELIMITATION);
        un.delimiter(tag::SEQUENCE_DELIMITATION);
        b.raw(&un.out);
    }
    b.element(tag::SERIES_INSTANCE_UID, b"UI", b"1.2.3.4")
        .element(tag::INSTANCE_NUMBER, b"IS", b"7")
        .element(tag::IMAGE_POSITION_PATIENT, b"DS", b"-125.5\\-100\\42.25")
        .element(tag::IMAGE_ORIENTATION_PATIENT, b"DS", b"1\\0\\0\\0\\1\\0")
        .element(tag::PIXEL_SPACING, b"DS", b"0.7\\0.7")
        .us(tag::SAMPLES_PER_PIXEL, 1)
        .element(tag::PHOTOMETRIC_INTERPRETATION, b"CS", b"MONOCHROME2")
        .us(tag::ROWS, 3)
        .us(tag::COLUMNS, 4)
        .us(tag::BITS_ALLOCATED, 16)
        .us(tag::BITS_STORED, 12)
        .us(tag::HIGH_BIT, 11)
        .us(tag::PIXEL_REPRESENTATION, 1)
        .element(tag::RESCALE_INTERCEPT, b"DS", b"-1024")
        .element(tag::RESCALE_SLOPE, b"DS", b"1");
    let values: Vec<i16> = vec![-2000, -1000, 0, 5, 40, 2047, -2048, 1, 2, 3, 100, -7];
    let pixels: Vec<u8> = values.iter().flat_map(|&v| if big { v.to_be_bytes() } else { v.to_le_bytes() }).collect();
    if encapsulated {
        b.header(tag::PIXEL_DATA, b"OB", 0xFFFF_FFFF);
        b.header(tag::ITEM, b"  ", 0);
        b.header(tag::ITEM, b"  ", 8);
        b.raw(&[1, 2, 3, 4, 5, 6, 7, 8]);
        b.delimiter(tag::SEQUENCE_DELIMITATION);
    } else {
        b.element(tag::PIXEL_DATA, b"OW", &pixels);
    }
    b.element(tomoz_dicom::Tag(0xFFFC, 0xFFFC), b"OB", &[0; 6]);
    (b.out, pixels)
}

fn file(transfer_syntax: &str, explicit: bool, big: bool, encapsulated: bool) -> (Vec<u8>, Vec<u8>) {
    let (ds, pixels) = dataset(explicit, big, encapsulated);
    let mut out = meta(transfer_syntax);
    out.extend_from_slice(&ds);
    (out, pixels)
}

#[test]
fn every_native_encoding() {
    for (syntax, explicit, big, encoding) in [
        (ts::IMPLICIT_VR_LITTLE_ENDIAN, false, false, DatasetEncoding::ImplicitLittle),
        (ts::EXPLICIT_VR_LITTLE_ENDIAN, true, false, DatasetEncoding::ExplicitLittle),
        (ts::EXPLICIT_VR_BIG_ENDIAN, true, true, DatasetEncoding::ExplicitBig),
    ] {
        let (bytes, pixels) = file(syntax, explicit, big, false);
        let p = parse(&bytes).unwrap_or_else(|e| panic!("{syntax}: {e}"));
        assert!(p.has_preamble);
        assert_eq!(p.encoding, encoding);
        let a = &p.attributes;
        assert_eq!(a.transfer_syntax_uid.as_deref(), Some(syntax));
        assert_eq!(a.sop_instance_uid.as_deref(), Some("1.2.3.4.5"));
        assert_eq!(a.series_instance_uid.as_deref(), Some("1.2.3.4"));
        assert_eq!(a.modality.as_deref(), Some("CT"));
        assert_eq!(a.image_type, vec!["ORIGINAL", "PRIMARY", "AXIAL"]);
        assert_eq!(a.instance_number, Some(7));
        assert_eq!(a.image_position, Some([-125.5, -100.0, 42.25]));
        assert_eq!(a.image_orientation, Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]));
        assert_eq!((a.rows, a.columns, a.bits_allocated, a.bits_stored), (Some(3), Some(4), Some(16), Some(12)));
        assert_eq!(a.pixel_representation, Some(1));
        assert_eq!(a.rescale_intercept, Some(-1024.0));
        let span = p.pixel.expect("pixel data");
        assert!(!span.encapsulated);
        assert_eq!(&bytes[span.value_start..span.value_end], &pixels[..]);
        let trailer = if explicit { 12 + 6 } else { 8 + 6 };
        assert_eq!(bytes.len() - span.value_end, trailer, "trailing padding element follows the pixel data");

        let layout = PixelLayout::from_attributes(a, p.encoding).unwrap();
        let samples = layout.extract(&bytes[span.value_start..span.value_end]).unwrap();
        assert_eq!(samples.values[0], -2000);
        assert_eq!(samples.convention, BitConvention::SignExtend);
        assert_eq!(layout.reconstruct(&samples), pixels);
    }
}

#[test]
fn encapsulated_pixel_data() {
    let (bytes, _) = file("1.2.840.10008.1.2.4.80", true, false, true);
    let p = parse(&bytes).unwrap();
    assert_eq!(p.encoding, DatasetEncoding::ExplicitLittle);
    let span = p.pixel.unwrap();
    assert!(span.encapsulated);
    // Offset table (empty item), one 8-byte fragment, delimiter.
    assert_eq!(span.len(), 8 + 16 + 8);
}

#[test]
fn deflated_data_sets_are_not_walked() {
    let mut bytes = meta(ts::DEFLATED_EXPLICIT_VR_LITTLE_ENDIAN);
    bytes.extend_from_slice(&[0x78, 0x9c, 1, 2, 3]);
    let p = parse(&bytes).unwrap();
    assert_eq!(p.encoding, DatasetEncoding::Deflated);
    assert!(p.pixel.is_none());
}

#[test]
fn bare_data_set_without_meta() {
    let (ds, pixels) = dataset(false, false, false);
    let p = parse(&ds).unwrap();
    assert!(!p.has_preamble);
    assert_eq!(p.encoding, DatasetEncoding::ImplicitLittle);
    let span = p.pixel.unwrap();
    assert_eq!(&ds[span.value_start..span.value_end], &pixels[..]);
}

#[test]
fn truncation_is_an_error() {
    let (bytes, _) = file(ts::EXPLICIT_VR_LITTLE_ENDIAN, true, false, false);
    // Cuts inside an element (a cut between elements is a valid, shorter data
    // set; archives detect it with the file digest).
    for cut in [140, bytes.len() - 30, bytes.len() - 1] {
        let err = parse(&bytes[..cut]).unwrap_err();
        assert!(matches!(err, Error::Truncated { .. } | Error::Malformed { .. }), "cut at {cut}: {err}");
    }
}

#[test]
fn deep_nesting_is_rejected() {
    let mut b = Builder::new(true, false);
    for _ in 0..40 {
        b.header(tomoz_dicom::Tag(0x0008, 0x1140), b"SQ", 0xFFFF_FFFF);
        b.header(tag::ITEM, b"  ", 0xFFFF_FFFF);
    }
    let mut bytes = meta(ts::EXPLICIT_VR_LITTLE_ENDIAN);
    bytes.extend_from_slice(&b.out);
    assert!(matches!(parse(&bytes), Err(Error::TooDeep { .. })));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// Corrupted files are rejected or parsed, never a panic, and any span
    /// reported lies inside the buffer.
    #[test]
    fn corruption_never_panics(seed in 0usize..3, flips in prop::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..8)) {
        let syntaxes = [(ts::IMPLICIT_VR_LITTLE_ENDIAN, false, false), (ts::EXPLICIT_VR_LITTLE_ENDIAN, true, false), (ts::EXPLICIT_VR_BIG_ENDIAN, true, true)];
        let (syntax, explicit, big) = syntaxes[seed];
        let (mut bytes, _) = file(syntax, explicit, big, seed == 1);
        for (i, v) in flips {
            let i = i.index(bytes.len());
            bytes[i] = v;
        }
        if let Ok(p) = parse(&bytes)
            && let Some(s) = p.pixel
        {
            prop_assert!(s.element_start < s.value_start && s.value_start <= s.value_end && s.value_end <= bytes.len());
        }
    }

    #[test]
    fn random_bytes_never_panic(bytes in prop::collection::vec(any::<u8>(), 0..600)) {
        let _ = parse(&bytes);
    }
}
