//! Decoding arbitrary bytes fails cleanly: no panic, no unbounded allocation.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tomoz_codec::{DecodeOptions, ModelSet, decode, decode_slices, inspect};

fuzz_target!(|input: &[u8]| {
    let Some((&flags, rest)) = input.split_first() else { return };
    let models = ModelSet::builtin();
    // Half of the inputs name the built-in models, whatever they named.
    let mut owned = rest.to_vec();
    if flags & 1 == 1 {
        tomoz_fuzz::retarget(&mut owned, &models);
    }
    let data = owned.as_slice();
    let registry = vec![models.two_d, models.three_d];
    let options = DecodeOptions { max_samples: 1 << 20, threads: Some(1), ..DecodeOptions::with_registry(&registry) };
    let header = inspect(data);
    let full = decode(data, &options);
    if let Ok(h) = header {
        let _ = decode_slices(data, &options, 0..1);
        let _ = decode_slices(data, &options, h.depth.saturating_sub(1) as usize..h.depth as usize);
        if let Ok(v) = full {
            // A container that decodes is one whose samples match its digest.
            assert_eq!(v.sha256(), h.sha256);
        }
    }
});
