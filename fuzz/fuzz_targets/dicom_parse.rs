//! The DICOM parser never panics, and what it reports lies inside the file.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(parsed) = tomoz_dicom::parse(data) {
        assert!(parsed.dataset_start <= data.len());
        if let Some(p) = parsed.pixel {
            assert!(p.element_start < p.value_start && p.value_start <= p.value_end && p.value_end <= data.len());
        }
    }
});
