//! The range decoder reads any byte string without panicking, and coding a
//! decoded symbol sequence again reproduces it.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tomoz_entropy::{BitModel, Cdf, Decoder, Encoder};

fuzz_target!(|data: &[u8]| {
    let Some((&plan, stream)) = data.split_first() else { return };
    let mut dec = Decoder::new(stream);
    let (mut cdf, mut bit) = (Cdf::<16>::uniform(), BitModel::new());
    let mut ops = Vec::new();
    for i in 0..256u32 {
        match (u32::from(plan) + i) % 3 {
            0 => ops.push((0, dec.decode_symbol(&mut cdf) as u32)),
            1 => ops.push((1, u32::from(dec.decode_bit(&mut bit)))),
            _ => ops.push((2, dec.decode_bypass(7))),
        }
    }
    let _ = dec.finish();
    let mut enc = Encoder::with_capacity(64);
    let (mut cdf, mut bit) = (Cdf::<16>::uniform(), BitModel::new());
    for &(kind, v) in &ops {
        match kind {
            0 => enc.encode_symbol(&mut cdf, v as usize),
            1 => enc.encode_bit(&mut bit, v == 1),
            _ => enc.encode_bypass(v, 7),
        }
    }
    let bytes = enc.finish();
    let mut dec = Decoder::new(&bytes);
    let (mut cdf, mut bit) = (Cdf::<16>::uniform(), BitModel::new());
    for &(kind, v) in &ops {
        let got = match kind {
            0 => dec.decode_symbol(&mut cdf) as u32,
            1 => u32::from(dec.decode_bit(&mut bit)),
            _ => dec.decode_bypass(7),
        };
        assert_eq!(got, v);
    }
});
