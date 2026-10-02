//! Round trips, determinism, efficiency and robustness of the range coder.

use proptest::prelude::*;
use tomoz_entropy::{BitModel, Cdf, Decoder, Encoder, Error};

/// One coding operation of a test script.
#[derive(Clone, Debug)]
enum Op {
    Sym4(usize, usize),
    Sym16(usize, usize),
    Bit(usize, bool),
    Bypass(u32, u32),
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        (0usize..3, 0usize..4).prop_map(|(m, s)| Op::Sym4(m, s)),
        (0usize..3, 0usize..16).prop_map(|(m, s)| Op::Sym16(m, s)),
        (0usize..3, any::<bool>()).prop_map(|(m, b)| Op::Bit(m, b)),
        (0u32..=32).prop_flat_map(|n| (Just(n), any::<u32>())).prop_map(|(n, v)| Op::Bypass(v & mask(n), n)),
    ]
}

fn mask(bits: u32) -> u32 {
    if bits == 32 { u32::MAX } else { (1 << bits) - 1 }
}

#[derive(Default)]
struct Models {
    m4: [Cdf<4>; 3],
    m16: [Cdf<16>; 3],
    bits: [BitModel; 3],
}

fn encode(ops: &[Op]) -> Vec<u8> {
    let mut m = Models::default();
    let mut enc = Encoder::new();
    for op in ops {
        match *op {
            Op::Sym4(i, s) => enc.encode_symbol(&mut m.m4[i], s),
            Op::Sym16(i, s) => enc.encode_symbol(&mut m.m16[i], s),
            Op::Bit(i, b) => enc.encode_bit(&mut m.bits[i], b),
            Op::Bypass(v, n) => enc.encode_bypass(v, n),
        }
    }
    enc.finish()
}

fn decode(ops: &[Op], bytes: &[u8]) -> Result<(), Error> {
    let mut m = Models::default();
    let mut dec = Decoder::new(bytes);
    for op in ops {
        match *op {
            Op::Sym4(i, s) => assert_eq!(dec.decode_symbol(&mut m.m4[i]), s),
            Op::Sym16(i, s) => assert_eq!(dec.decode_symbol(&mut m.m16[i]), s),
            Op::Bit(i, b) => assert_eq!(dec.decode_bit(&mut m.bits[i]), b),
            Op::Bypass(v, n) => assert_eq!(dec.decode_bypass(n), v),
        }
    }
    dec.finish()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn roundtrip(ops in prop::collection::vec(op(), 0..2000)) {
        let bytes = encode(&ops);
        prop_assert_eq!(decode(&ops, &bytes), Ok(()));
    }

    #[test]
    fn skewed_roundtrip(seed in any::<u64>(), n in 1usize..20_000) {
        // Long runs of the same symbol push the models to their limits.
        let mut rng = Lcg(seed);
        let ops: Vec<Op> = (0..n).map(|_| {
            let r = rng.next();
            if r.is_multiple_of(1000) { Op::Sym16(0, (r >> 20) as usize % 16) } else if r.is_multiple_of(3) { Op::Bit(1, r.is_multiple_of(997)) } else { Op::Sym16(0, 3) }
        }).collect();
        let bytes = encode(&ops);
        prop_assert_eq!(decode(&ops, &bytes), Ok(()));
    }

    #[test]
    fn garbage_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..512), ops in prop::collection::vec(op(), 0..300)) {
        let mut m = Models::default();
        let mut dec = Decoder::new(&bytes);
        for op in &ops {
            match *op {
                Op::Sym4(i, _) => { prop_assert!(dec.decode_symbol(&mut m.m4[i]) < 4); }
                Op::Sym16(i, _) => { prop_assert!(dec.decode_symbol(&mut m.m16[i]) < 16); }
                Op::Bit(i, _) => { dec.decode_bit(&mut m.bits[i]); }
                Op::Bypass(_, n) => { prop_assert!(dec.decode_bypass(n) <= mask(n)); }
            }
        }
        let _ = dec.finish();
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 16
    }
}

#[test]
fn empty_stream() {
    let bytes = Encoder::new().finish();
    assert!(bytes.len() <= 4, "{bytes:?}");
    assert_eq!(Decoder::new(&bytes).finish(), Ok(()));
}

#[test]
fn truncation_and_trailing_bytes_are_reported() {
    let mut rng = Lcg(7);
    let ops: Vec<Op> = (0..5000).map(|_| Op::Sym16(0, (rng.next() % 16) as usize)).collect();
    let bytes = encode(&ops);
    let mut long = bytes.clone();
    long.push(0);
    let mut m = Cdf::<16>::uniform();
    let mut dec = Decoder::new(&long);
    for _ in &ops {
        dec.decode_symbol(&mut m);
    }
    assert_eq!(dec.finish(), Err(Error::TrailingBytes { unread: 1 }));

    let short = &bytes[..bytes.len() - 3];
    let mut m = Cdf::<16>::uniform();
    let mut dec = Decoder::new(short);
    for _ in &ops {
        dec.decode_symbol(&mut m);
    }
    assert!(matches!(dec.finish(), Err(Error::Truncated { .. })));
}

/// The coded size approaches the entropy of the source once the model has
/// adapted.
#[test]
fn efficiency_close_to_entropy() {
    let p = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.03125];
    let entropy: f64 = p.iter().map(|&q: &f64| -q * q.log2()).sum();
    let mut rng = Lcg(42);
    let n = 400_000;
    let mut enc = Encoder::new();
    let mut m = Cdf::<6>::uniform();
    for _ in 0..n {
        let u = (rng.next() % 1_000_000) as f64 / 1e6;
        let mut acc = 0.0;
        let mut s = p.len() - 1;
        for (i, &q) in p.iter().enumerate() {
            acc += q;
            if u < acc {
                s = i;
                break;
            }
        }
        enc.encode_symbol(&mut m, s);
    }
    let bits = enc.finish().len() as f64 * 8.0 / n as f64;
    assert!(bits < entropy * 1.02, "{bits:.4} bits/symbol vs entropy {entropy:.4}");

    // A heavily skewed binary source.
    let mut enc = Encoder::new();
    let mut m = BitModel::new();
    let n = 1_000_000;
    for i in 0..n {
        enc.encode_bit(&mut m, i % 2000 == 0);
    }
    let bits = enc.finish().len() as f64 * 8.0 / n as f64;
    assert!(bits < 0.01, "{bits:.5} bits/flag");
}

/// The byte stream for a fixed input never changes: decoders deployed today
/// must read archives written by any later version.
#[test]
fn format_is_stable() {
    let mut rng = Lcg(2026);
    let ops: Vec<Op> = (0..10_000)
        .map(|i| {
            let r = rng.next();
            match i % 4 {
                0 => Op::Sym16(i % 3, (r % 16) as usize),
                1 => Op::Sym4(i % 3, (r % 4) as usize),
                2 => Op::Bit(i % 3, r.is_multiple_of(5)),
                _ => Op::Bypass((r as u32) & mask((r % 20) as u32), (r % 20) as u32),
            }
        })
        .collect();
    let bytes = encode(&ops);
    let digest = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3));
    assert_eq!((bytes.len(), digest), (GOLDEN_LEN, GOLDEN_FNV));
}

const GOLDEN_LEN: usize = 5058;
const GOLDEN_FNV: u64 = 8_568_829_283_792_806_208;
