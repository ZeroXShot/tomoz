//! Throughput of symbol, bit and bypass coding.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tomoz_entropy::{BitModel, Cdf, Decoder, Encoder};

fn symbols(n: usize) -> Vec<usize> {
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            // Geometric-like: small symbols dominate, as for prediction residuals.
            (x.trailing_zeros() as usize).min(15)
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let syms = symbols(1 << 20);
    let mut g = c.benchmark_group("cdf16");
    g.throughput(Throughput::Elements(syms.len() as u64));
    g.bench_function("encode", |b| {
        b.iter(|| {
            let mut enc = Encoder::with_capacity(syms.len());
            let mut m = Cdf::<16>::uniform();
            for &s in &syms {
                enc.encode_symbol(&mut m, s);
            }
            black_box(enc.finish())
        });
    });
    let mut enc = Encoder::new();
    let mut m = Cdf::<16>::uniform();
    for &s in &syms {
        enc.encode_symbol(&mut m, s);
    }
    let bytes = enc.finish();
    g.bench_function("decode", |b| {
        b.iter(|| {
            let mut dec = Decoder::new(&bytes);
            let mut m = Cdf::<16>::uniform();
            let mut acc = 0usize;
            for _ in 0..syms.len() {
                acc += dec.decode_symbol(&mut m);
            }
            black_box(acc)
        });
    });
    g.finish();

    let mut g = c.benchmark_group("bit");
    g.throughput(Throughput::Elements(syms.len() as u64));
    g.bench_function("decode", |b| {
        let mut enc = Encoder::new();
        let mut m = BitModel::new();
        for &s in &syms {
            enc.encode_bit(&mut m, s == 0);
        }
        let bytes = enc.finish();
        b.iter(|| {
            let mut dec = Decoder::new(&bytes);
            let mut m = BitModel::new();
            let mut acc = 0usize;
            for _ in 0..syms.len() {
                acc += usize::from(dec.decode_bit(&mut m));
            }
            black_box(acc)
        });
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
