//! Conformance: fixed volumes must compress to exactly the same bytes on every
//! platform and build (x86-64, AArch64, WebAssembly; scalar or SIMD kernels;
//! one thread or many).
//!
//! The volumes are synthetic phantoms generated from a seed, so the cases fit
//! in `tests/conformance/cases.json`, which pins the SHA-256 of the samples
//! and of the container produced with the built-in models. The JavaScript
//! test of the WebAssembly build (`crates/tomoz-wasm/js/test`) runs the same
//! cases. When the format or the built-in models change on purpose, refresh
//! the hashes with `TOMOZ_BLESS=1 cargo test -p tomoz-codec --test conformance`.

#![allow(clippy::unwrap_used)] // Helpers of tests panic on unexpected errors.

use sha2::{Digest, Sha256};
use tomoz_codec::{DecodeOptions, EncodeOptions, ModelSet, Volume, decode, decode_slices, encode, value_range};

const CASES: &str = "tests/conformance/cases.json";

/// A disc phantom: flat background, a bright rim (sharp edges), and tissue
/// with a gradient and uniform noise, quantised to multiples of `step`.
/// `crates/tomoz-wasm/js/test/phantom.mjs` implements the same algorithm.
#[allow(clippy::too_many_arguments)]
fn phantom(
    depth: usize,
    height: usize,
    width: usize,
    bits: u8,
    signed: bool,
    seed: u32,
    noise: i64,
    step: i64,
) -> Volume {
    let (lo, hi) = value_range(bits, signed);
    let range = i64::from(hi) - i64::from(lo) + 1;
    let mut state = seed;
    let mut rand = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        i64::from(state >> 16)
    };
    let (cy, cx) = ((height / 2) as i64, (width / 2) as i64);
    let r = (height.min(width) * 2 / 5) as i64;
    let mut samples = Vec::with_capacity(depth * height * width);
    for z in 0..depth as i64 {
        let rz = r - z % 4;
        for y in 0..height as i64 {
            for x in 0..width as i64 {
                let d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                let v = if d2 > rz * rz {
                    0
                } else if d2 > (rz - 2) * (rz - 2) {
                    range * 3 / 4
                } else {
                    range / 4 + (x * 7 + y * 3 + z * 11) % (range / 8 + 1) + rand() % (noise + 1)
                };
                samples.push((i64::from(lo) + v.min(range - 1) / step * step) as i32);
            }
        }
    }
    Volume::new(depth, height, width, bits, signed, samples).unwrap()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sample_bytes(v: &Volume) -> Vec<u8> {
    v.samples().iter().flat_map(|&s| (s as u16).to_le_bytes()).collect()
}

#[test]
fn containers_are_identical_everywhere() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(CASES);
    let mut doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let bless = std::env::var_os("TOMOZ_BLESS").is_some();
    let models = ModelSet::builtin();
    let ids = [models.two_d.id().to_string(), models.three_d.id().to_string()];
    let mut failures = Vec::new();
    for case in doc["cases"].as_array_mut().unwrap() {
        let n = |k: &str| case[k].as_u64().unwrap_or_else(|| panic!("{k}"));
        let name = case["name"].as_str().unwrap().to_owned();
        let volume = phantom(
            n("depth") as usize,
            n("height") as usize,
            n("width") as usize,
            n("bits") as u8,
            case["signed"].as_bool().unwrap(),
            n("seed") as u32,
            n("noise") as i64,
            n("step") as i64,
        );
        let options = EncodeOptions {
            slab: n("slab") as u16,
            stripe: n("stripe") as u16,
            packing: case["packing"].as_bool().unwrap(),
            ..EncodeOptions::with_models(models.clone())
        };
        let samples_sha = hex(&Sha256::digest(sample_bytes(&volume)));
        assert_eq!(samples_sha, hex(&volume.sha256()), "{name}: Volume::sha256 is the SHA-256 of 16-bit LE samples");
        // Every thread count produces the same container.
        let bytes = encode(&volume, &options).unwrap();
        for threads in [Some(1), Some(3)] {
            assert_eq!(
                encode(&volume, &EncodeOptions { threads, ..options.clone() }).unwrap(),
                bytes,
                "{name}: {threads:?} threads"
            );
        }
        let decode_options = DecodeOptions::with_registry(&models);
        assert_eq!(decode(&bytes, &decode_options).unwrap(), volume, "{name}: round trip");
        let depth = volume.depth();
        let part = decode_slices(&bytes, &decode_options, depth / 2..depth).unwrap();
        let plane = volume.height() * volume.width();
        assert_eq!(part.samples(), &volume.samples()[depth / 2 * plane..], "{name}: slices");

        let container_sha = hex(&Sha256::digest(&bytes));
        if bless {
            case["samples_sha256"] = samples_sha.into();
            case["container_sha256"] = container_sha.into();
            case["container_bytes"] = bytes.len().into();
            continue;
        }
        if case["samples_sha256"].as_str() != Some(samples_sha.as_str()) {
            failures.push(format!("{name}: phantom samples {samples_sha} (generator changed?)"));
        }
        if case["container_sha256"].as_str() != Some(container_sha.as_str()) {
            failures.push(format!("{name}: container {container_sha} ({} bytes)", bytes.len()));
        }
    }
    if bless {
        doc["model_2d"] = ids[0].clone().into();
        doc["model_3d"] = ids[1].clone().into();
        std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap() + "\n").unwrap();
        return;
    }
    let pinned = [doc["model_2d"].as_str().unwrap_or(""), doc["model_3d"].as_str().unwrap_or("")];
    assert!(
        failures.is_empty(),
        "conformance hashes differ:\n  {}\n(pinned with models {pinned:?}, built-in models are {ids:?}; \
         if the change is intended, run with TOMOZ_BLESS=1)",
        failures.join("\n  ")
    );
}
