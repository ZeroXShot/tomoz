//! Every kernel computes the scalar reference exactly.

use proptest::prelude::*;
use tomoz_nn::{Activation, Error, Kernel, Layer, Network, Scratch, padded};

fn layer(inputs: usize, outputs: usize, shift: u8, last: bool, seed: &mut u64) -> Layer {
    let mut next = || {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    };
    let weights = (0..inputs * outputs).map(|_| (next() % 256) as u8 as i8).collect();
    let bias = (0..outputs).map(|_| (next() % 200_000) as i32 - 100_000).collect();
    Layer {
        inputs,
        outputs,
        weights,
        bias,
        shift,
        activation: if last { Activation::Linear } else { Activation::ClippedRelu },
    }
}

fn network(widths: &[usize], shifts: &[u8], seed: u64) -> Vec<Layer> {
    let mut s = seed | 1;
    (0..widths.len() - 1)
        .map(|i| layer(widths[i], widths[i + 1], shifts[i % shifts.len()], i + 2 == widths.len(), &mut s))
        .collect()
}

fn run(net: &Network, inputs: &[i8], batch: usize) -> Vec<i32> {
    let mut out = vec![0; batch * net.outputs()];
    net.forward(inputs, batch, &mut Scratch::default(), &mut out);
    out
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn kernels_match_scalar(
        widths in prop::collection::vec(1usize..70, 2..5),
        shifts in prop::collection::vec(0u8..14, 1..4),
        seed in any::<u64>(),
        batch in 1usize..9,
        raw in prop::collection::vec(any::<i8>(), 9 * 80),
    ) {
        let layers = network(&widths, &shifts, seed);
        let reference = Network::with_kernel(layers.clone(), Kernel::Scalar).unwrap();
        let stride = reference.input_stride();
        let mut inputs = vec![0i8; batch * stride];
        for s in 0..batch {
            for i in 0..widths[0] {
                inputs[s * stride + i] = raw[(s * 80 + i) % raw.len()];
            }
        }
        let expected = run(&reference, &inputs, batch);
        for kernel in Kernel::available() {
            let net = Network::with_kernel(layers.clone(), kernel).unwrap();
            prop_assert_eq!(net.kernel(), kernel);
            prop_assert_eq!(&run(&net, &inputs, batch), &expected, "kernel {:?}", kernel);
        }
    }

    #[test]
    fn serialization_roundtrip(widths in prop::collection::vec(1usize..40, 2..5), seed in any::<u64>()) {
        let net = Network::new(network(&widths, &[7], seed)).unwrap();
        let bytes = net.to_bytes();
        let (back, used) = Network::from_bytes(&bytes).unwrap();
        prop_assert_eq!(used, bytes.len());
        prop_assert_eq!(back.layers(), net.layers());
    }

    #[test]
    fn decoding_garbage_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..400)) {
        let _ = Network::from_bytes(&bytes);
    }
}

#[test]
fn extreme_values_are_exact() {
    // Largest weights and inputs everywhere: the accumulator bound is tight
    // but valid, and every kernel must agree.
    let inputs_n = 255;
    let l1 = Layer {
        inputs: inputs_n,
        outputs: 20,
        weights: vec![-128; inputs_n * 20],
        bias: vec![0; 20],
        shift: 8,
        activation: Activation::ClippedRelu,
    };
    let l2 = Layer {
        inputs: 20,
        outputs: 3,
        weights: vec![127; 60],
        bias: vec![-5; 3],
        shift: 0,
        activation: Activation::Linear,
    };
    let reference = Network::with_kernel(vec![l1.clone(), l2.clone()], Kernel::Scalar).unwrap();
    let stride = padded(inputs_n);
    for fill in [-128i8, 127, 0] {
        let mut x = vec![0i8; stride * 2];
        for v in x.iter_mut().take(inputs_n) {
            *v = fill;
        }
        let expected = run(&reference, &x, 2);
        for kernel in Kernel::available() {
            let net = Network::with_kernel(vec![l1.clone(), l2.clone()], kernel).unwrap();
            assert_eq!(run(&net, &x, 2), expected, "{kernel:?}");
        }
    }
}

#[test]
fn validation() {
    let ok = network(&[4, 8, 2], &[5], 1);
    assert!(Network::new(ok.clone()).is_ok());
    assert_eq!(Network::new(vec![]).unwrap_err(), Error::LayerCount(0));
    let mut bad = ok.clone();
    bad[1].inputs = 7;
    bad[1].weights.truncate(14);
    assert!(matches!(Network::new(bad).unwrap_err(), Error::Shape { layer: 1, .. }));
    let mut bad = ok.clone();
    bad[0].activation = Activation::Linear;
    assert_eq!(Network::new(bad).unwrap_err(), Error::Activation(0));
    let mut bad = ok.clone();
    bad[0].bias[0] = i32::MAX;
    assert_eq!(Network::new(bad).unwrap_err(), Error::Overflow(0));
    let mut bad = ok;
    bad[0].shift = 30;
    assert!(matches!(Network::new(bad).unwrap_err(), Error::Shift { .. }));
}
