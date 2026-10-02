//! The codec reproduces the predictions of the Python reference
//! (`lab/src/tomoz_lab/golden.py`) exactly.

#![allow(clippy::unwrap_used)] // Helpers of tests panic on unexpected errors.

use tomoz_codec::{DecodeOptions, EncodeOptions, Model, ModelSet, Volume, decode, encode, trace_predictions};

fn hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

#[test]
fn predictions_match_the_reference() {
    let text = include_str!("golden/predictions.json");
    let doc: serde_json::Value = serde_json::from_str(text).unwrap();
    for (n, case) in doc["cases"].as_array().unwrap().iter().enumerate() {
        let shape: Vec<usize> =
            case["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let samples: Vec<i32> =
            case["samples"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect();
        let bits = case["bits"].as_u64().unwrap() as u8;
        let volume = Volume::new(shape[0], shape[1], shape[2], bits, false, samples).unwrap();
        let models = ModelSet {
            two_d: Model::from_bytes(&hex(case["model_2d"].as_str().unwrap())).unwrap(),
            three_d: Model::from_bytes(&hex(case["model_3d"].as_str().unwrap())).unwrap(),
        };
        let options =
            EncodeOptions { slab: 255, stripe: 4096, packing: false, ..EncodeOptions::with_models(models.clone()) };
        let got = trace_predictions(&volume, &options);
        let want = case["predictions"].as_array().unwrap();
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            let w: Vec<i64> = w.as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
            let expected = if w[3] == 1 { None } else { Some((w[4], w[5])) };
            assert_eq!(*g, expected, "case {n}, sample (z, y, x) = ({}, {}, {})", w[0], w[1], w[2]);
        }
        // And the volume survives a round trip with these models.
        let bytes = encode(&volume, &options).unwrap();
        assert_eq!(decode(&bytes, &DecodeOptions::with_registry(&models)).unwrap(), volume);
    }
}
