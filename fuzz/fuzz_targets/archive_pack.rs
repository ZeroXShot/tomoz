//! Packing any set of files (DICOM or not, valid or not) and restoring the
//! archive gives back exactly the same bytes. The gateway packs whatever
//! clients upload, so this is the property it relies on.
//!
//! The input is split into files at each `\xffSPLIT\xff`, so that seeds can
//! be several real DICOM files joined together.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tomoz_archive::{Archive, PackOptions, pack};
use tomoz_codec::{EncodeOptions, ModelSet};

const SEPARATOR: &[u8] = b"\xffSPLIT\xff";

fn split(mut data: &[u8]) -> Vec<&[u8]> {
    let mut files = Vec::new();
    while let Some(i) = data.windows(SEPARATOR.len()).position(|w| w == SEPARATOR) {
        files.push(&data[..i]);
        data = &data[i + SEPARATOR.len()..];
    }
    files.push(data);
    files
}

fuzz_target!(|data: &[u8]| {
    let files: Vec<(String, &[u8])> =
        split(data).into_iter().take(8).enumerate().map(|(i, b)| (format!("file{i}.dcm"), b)).collect();
    let models = ModelSet::builtin();
    let options =
        PackOptions::new(EncodeOptions { threads: Some(1), slab: 2, ..EncodeOptions::with_models(models.clone()) });
    let Ok((bytes, _)) = pack(&files, &options) else { return };
    let archive = Archive::open(&bytes).expect("an archive just written opens");
    let restored = archive.restore_all(&models).expect("an archive just written restores");
    assert_eq!(restored.len(), files.len());
    for (back, (name, original)) in restored.iter().zip(&files) {
        assert_eq!(back.as_slice(), *original, "{name}");
    }
    for (i, (_, original)) in files.iter().enumerate() {
        assert_eq!(archive.restore(i, &models).expect("single restore").as_slice(), *original);
    }
});
