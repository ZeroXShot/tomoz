//! Model files are validated: parsing never panics, and a model that loads
//! can code a volume without overflowing its integer arithmetic.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tomoz_codec::{DecodeOptions, EncodeOptions, Model, ModelKind, ModelSet, Volume, decode, encode};

fuzz_target!(|data: &[u8]| {
    let Ok(model) = Model::from_bytes(data) else { return };
    let builtin = ModelSet::builtin();
    let models = match model.kind() {
        ModelKind::TwoD => ModelSet { two_d: model, three_d: builtin.three_d },
        ModelKind::ThreeD => ModelSet { two_d: builtin.two_d, three_d: model },
    };
    let samples: Vec<i32> = (0..3 * 9 * 11).map(|i| (i * 7919) % 4096).collect();
    let volume = Volume::new(3, 9, 11, 12, false, samples).expect("valid volume");
    let options = EncodeOptions { threads: Some(1), ..EncodeOptions::with_models(models.clone()) };
    let bytes = encode(&volume, &options).expect("valid volumes always encode");
    assert_eq!(decode(&bytes, &DecodeOptions::with_registry(&models)).expect("own output decodes"), volume);
});
