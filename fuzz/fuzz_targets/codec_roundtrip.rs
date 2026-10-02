//! Every volume survives a round trip, whatever its shape, format and tiling.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use tomoz_codec::{DecodeOptions, EncodeOptions, ModelSet, Volume, decode, encode, value_range};

#[derive(Debug)]
struct Case {
    volume: Volume,
    slab: u16,
    stripe: u16,
    packing: bool,
}

impl<'a> Arbitrary<'a> for Case {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let depth = u.int_in_range(1..=6)?;
        let height = u.int_in_range(1..=24)?;
        let width = u.int_in_range(1..=24)?;
        let bits = u.int_in_range(1..=16)?;
        let signed = u.arbitrary()?;
        let (lo, hi) = value_range(bits, signed);
        let mut samples = Vec::with_capacity(depth * height * width);
        let mut previous = lo;
        for _ in 0..depth * height * width {
            // Mostly small steps (like images), sometimes anything.
            let v = if u.ratio(1, 8)? {
                u.int_in_range(lo..=hi)?
            } else {
                (previous + i32::from(u.int_in_range(-3i8..=3)?)).clamp(lo, hi)
            };
            samples.push(v);
            previous = v;
        }
        let volume =
            Volume::new(depth, height, width, bits, signed, samples).map_err(|_| arbitrary::Error::IncorrectFormat)?;
        Ok(Self { volume, slab: u.int_in_range(1..=4)?, stripe: u.int_in_range(1..=16)?, packing: u.arbitrary()? })
    }
}

fuzz_target!(|case: Case| {
    let models = ModelSet::builtin();
    let options = EncodeOptions {
        slab: case.slab,
        stripe: case.stripe,
        packing: case.packing,
        threads: Some(1),
        ..EncodeOptions::with_models(models.clone())
    };
    let bytes = encode(&case.volume, &options).expect("valid volumes always encode");
    let back = decode(&bytes, &DecodeOptions::with_registry(&models)).expect("own output always decodes");
    assert_eq!(back, case.volume);
});
