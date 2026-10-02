//! Encode and decode throughput on a synthetic CT-like volume.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tomoz_codec::{DecodeOptions, EncodeOptions, ModelSet, Volume, decode, encode};

/// A smooth phantom with noise and a circular field of view.
fn phantom(depth: usize, size: usize) -> Volume {
    let mut x = 0x2545_f491_4f6c_dd1du64;
    let mut noise = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % 41) as i32 - 20
    };
    let mut samples = Vec::with_capacity(depth * size * size);
    let c = size as f64 / 2.0;
    for z in 0..depth {
        for y in 0..size {
            for xx in 0..size {
                let r = ((y as f64 - c).powi(2) + (xx as f64 - c).powi(2)).sqrt();
                let v = if r > c * 0.95 {
                    -2000
                } else {
                    (40.0 * ((xx as f64 + z as f64) / 20.0).sin() + 1000.0 * (1.0 - r / c)) as i32 + noise()
                };
                samples.push(v);
            }
        }
    }
    Volume::new(depth, size, size, 16, true, samples).expect("valid phantom")
}

fn bench(c: &mut Criterion) {
    let volume = phantom(8, 256);
    let models = ModelSet::untrained();
    let options = EncodeOptions::with_models(models.clone());
    let bytes = encode(&volume, &options).expect("encode");
    let mut g = c.benchmark_group("phantom-8x256x256");
    g.throughput(Throughput::Bytes(volume.raw_bytes() as u64));
    g.sample_size(10);
    g.bench_function("encode", |b| b.iter(|| black_box(encode(&volume, &options).expect("encode"))));
    g.bench_function("decode", |b| {
        b.iter(|| black_box(decode(&bytes, &DecodeOptions::with_registry(&models)).expect("decode")));
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
