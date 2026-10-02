//! Lossless round trips and robustness of the decoder.

#![allow(clippy::unwrap_used)] // Helpers of tests panic on unexpected errors.

use proptest::prelude::*;
use tomoz_codec::{
    DecodeOptions, EncodeOptions, Error, ModelSet, Volume, decode, decode_slices, encode, inspect, value_range,
};

fn models() -> &'static ModelSet {
    static MODELS: std::sync::OnceLock<ModelSet> = std::sync::OnceLock::new();
    MODELS.get_or_init(ModelSet::untrained)
}

fn roundtrip(volume: &Volume, slab: u16, stripe: u16) -> Vec<u8> {
    let options = EncodeOptions { slab, stripe, ..EncodeOptions::with_models(models().clone()) };
    let bytes = encode(volume, &options).unwrap();
    let back = decode(&bytes, &DecodeOptions::with_registry(models())).unwrap();
    assert_eq!(&back, volume);
    bytes
}

/// Values with structure (gradients, flat regions, edges) and noise.
fn structured(depth: usize, height: usize, width: usize, bits: u8, signed: bool, seed: u64) -> Volume {
    let (lo, hi) = value_range(bits, signed);
    let mut x = seed | 1;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let span = i64::from(hi) - i64::from(lo);
    let mut samples = Vec::with_capacity(depth * height * width);
    for z in 0..depth {
        for y in 0..height {
            for xx in 0..width {
                let base =
                    (span / 2 + (y as i64 * 37 + xx as i64 * 11 + z as i64 * 5) % (span / 4 + 1)) + i64::from(lo);
                let v = match next() % 10 {
                    0..=2 if xx > width / 2 => i64::from(lo),
                    3 => i64::from(lo) + (next() as i64).rem_euclid(span + 1),
                    _ => base + (next() % 7) as i64 - 3,
                };
                samples.push(v.clamp(i64::from(lo), i64::from(hi)) as i32);
            }
        }
    }
    Volume::new(depth, height, width, bits, signed, samples).unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn lossless(
        depth in 1usize..7,
        height in 1usize..40,
        width in 1usize..40,
        bits in 1u8..=16,
        signed in any::<bool>(),
        slab in 1u16..5,
        stripe in 1u16..50,
        seed in any::<u64>(),
    ) {
        let v = structured(depth, height, width, bits, signed, seed);
        roundtrip(&v, slab, stripe);
    }
}

#[test]
fn slice_ranges_decode_alone() {
    let v = structured(11, 13, 9, 12, true, 21);
    let bytes = roundtrip(&v, 3, 5);
    let plane = 13 * 9;
    let options = DecodeOptions::with_registry(models());
    for (a, b) in [(0, 1), (0, 11), (2, 3), (3, 6), (5, 11), (10, 11)] {
        let part = decode_slices(&bytes, &options, a..b).unwrap();
        assert_eq!(part.depth(), b - a);
        assert_eq!(part.samples(), &v.samples()[a * plane..b * plane], "slices {a}..{b}");
    }
    assert!(decode_slices(&bytes, &options, 3..3).is_err());
    assert!(decode_slices(&bytes, &options, 5..12).is_err());
}

#[test]
fn constant_and_extreme_volumes() {
    for (bits, signed, value) in
        [(16, true, -32768), (16, true, 32767), (16, false, 65535), (1, false, 1), (8, false, 0)]
    {
        let v = Volume::new(3, 17, 9, bits, signed, vec![value; 3 * 17 * 9]).unwrap();
        roundtrip(&v, 2, 5);
    }
    // Full-range noise in every bit depth.
    for bits in [1u8, 7, 12, 16] {
        roundtrip(&structured(2, 20, 33, bits, true, 99), 32, 512);
    }
}

#[test]
fn sparse_values_are_packed() {
    let samples: Vec<i32> = (0..4 * 30 * 30).map(|i| ((i * 31) % 400) * 4 - 800).collect();
    let v = Volume::new(4, 30, 30, 12, true, samples).unwrap();
    let bytes = roundtrip(&v, 32, 512);
    assert!(inspect(&bytes).unwrap().packed());
}

#[test]
fn header_reports_geometry() {
    let v = structured(5, 21, 13, 12, false, 3);
    let bytes = roundtrip(&v, 2, 8);
    let h = inspect(&bytes).unwrap();
    assert_eq!((h.depth, h.height, h.width, h.bits, h.signed), (5, 21, 13, 12, false));
    assert_eq!((h.slabs(), h.stripes(), h.tile_count()), (3, 3, 9));
    assert_eq!(h.sha256, v.sha256());
}

#[test]
fn corruption_is_detected() {
    let v = structured(3, 24, 24, 12, true, 5);
    let bytes = roundtrip(&v, 2, 12);
    let header_len = {
        let h = inspect(&bytes).unwrap();
        h.header_len
    };
    let registry = models();
    let options = DecodeOptions::with_registry(registry);
    // Flipping any byte must give an error, never a wrong volume or a panic.
    for i in (0..bytes.len()).step_by(7) {
        let mut b = bytes.clone();
        b[i] ^= 0x5a;
        match decode(&b, &options) {
            Ok(back) => panic!("corruption at {i} (header {header_len}) decoded to a volume equal: {}", back == v),
            Err(
                Error::NotTomoz
                | Error::Corrupt(_)
                | Error::Checksum(_)
                | Error::Truncated
                | Error::Unsupported(_)
                | Error::UnknownModel(_),
            ) => {}
            Err(e) => panic!("unexpected error kind at {i}: {e}"),
        }
    }
    for cut in [0, 3, 50, header_len, bytes.len() - 1] {
        assert!(decode(&bytes[..cut], &options).is_err(), "cut at {cut}");
    }
}

#[test]
fn huge_dimensions_are_rejected() {
    // Found by fuzzing: depth × height × width overflowed a u64 and slipped
    // under the sample limit.
    let volume = structured(2, 8, 8, 12, false, 1);
    let mut bytes = roundtrip(&volume, 32, 512);
    bytes[8..20].fill(0xFF);
    assert!(matches!(inspect(&bytes), Err(Error::Corrupt("volume dimensions"))));
    assert!(matches!(decode(&bytes, &DecodeOptions::with_registry(models())), Err(Error::Corrupt(_))));
}

#[test]
fn builtin_models_are_the_released_ones() {
    // Changing a model file changes every container: it must be deliberate.
    let m = ModelSet::builtin();
    assert_eq!(m.two_d.id().to_string(), "78f954e342dbbd3eef09fc52ab780e42");
    assert_eq!(m.three_d.id().to_string(), "fe411d9822c41fac88240c959e0e48ea");
    assert_eq!((m.two_d.name(), m.three_d.name()), ("tz1-2d-h48x48", "tz1-3d-h48x48"));
}

#[test]
fn unknown_models_are_reported() {
    let v = structured(2, 8, 8, 8, false, 1);
    let bytes = roundtrip(&v, 32, 512);
    let empty: Vec<tomoz_codec::Model> = Vec::new();
    let err = decode(&bytes, &DecodeOptions::with_registry(&empty)).unwrap_err();
    assert!(matches!(err, Error::UnknownModel(_)));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn garbage_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..300)) {
        let mut b = vec![0x89, b'T', b'M', b'Z', 1, 0];
        b.extend_from_slice(&bytes);
        let _ = decode(&b, &DecodeOptions::with_registry(models()));
    }
}
