//! Forward-pass throughput of a predictor-sized network on every kernel.

#![allow(clippy::unwrap_used)] // Helpers of tests panic on unexpected errors.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tomoz_nn::{Activation, Kernel, Layer, Network, Scratch};

fn layer(inputs: usize, outputs: usize, last: bool) -> Layer {
    let weights = (0..inputs * outputs).map(|i| ((i * 37) % 255) as i8).collect();
    Layer {
        inputs,
        outputs,
        weights,
        bias: vec![100; outputs],
        shift: 9,
        activation: if last { Activation::Linear } else { Activation::ClippedRelu },
    }
}

fn bench(c: &mut Criterion) {
    let batch = 512;
    let layers = vec![layer(32, 48, false), layer(48, 48, false), layer(48, 6, true)];
    let mut g = c.benchmark_group("predictor-32-48-48-6");
    g.throughput(Throughput::Elements(batch as u64));
    for kernel in Kernel::available() {
        let net = Network::with_kernel(layers.clone(), kernel).unwrap();
        let inputs: Vec<i8> = (0..batch * net.input_stride()).map(|i| ((i * 13) % 200) as i8).collect();
        let mut out = vec![0; batch * net.outputs()];
        let mut scratch = Scratch::default();
        g.bench_function(format!("{kernel:?}"), |b| {
            b.iter(|| {
                net.forward(black_box(&inputs), batch, &mut scratch, &mut out);
                black_box(out[0])
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
