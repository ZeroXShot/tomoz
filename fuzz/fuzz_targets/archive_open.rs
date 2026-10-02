//! Opening and restoring arbitrary archives fails cleanly.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tomoz_archive::Archive;
use tomoz_codec::ModelSet;

fuzz_target!(|input: &[u8]| {
    let Some((&flags, rest)) = input.split_first() else { return };
    let models = ModelSet::builtin();
    let mut owned = rest.to_vec();
    if flags & 1 == 1 {
        // Point every stack at the built-in models (see `retarget`).
        let spans: Vec<(usize, usize)> = match Archive::open(rest) {
            Ok(a) => (0..a.stack_count())
                .map(|i| (a.stack(i).as_ptr() as usize - rest.as_ptr() as usize, a.stack(i).len()))
                .collect(),
            Err(_) => Vec::new(),
        };
        for (offset, len) in spans {
            tomoz_fuzz::retarget(&mut owned[offset..offset + len], &models);
        }
    }
    let data = owned.as_slice();
    let Ok(archive) = Archive::open(data) else { return };
    let archive = archive.with_max_samples(1 << 20);
    let _ = archive.metadata();
    for i in 0..archive.instances().len().min(8) {
        let _ = archive.restore(i, &models);
    }
    let _ = archive.restore_all(&models);
});
