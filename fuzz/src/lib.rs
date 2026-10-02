//! Helpers shared by the fuzz targets.

use tomoz_codec::ModelSet;

/// Rewrites the model identifiers of a Tomoz container in place to those of
/// `models` and fixes the header checksum, so that mutated inputs reach tile
/// decoding instead of stopping at "unknown model" (a fuzzer cannot forge a
/// 16-byte hash). Inputs that are not containers are left alone.
pub fn retarget(container: &mut [u8], models: &ModelSet) {
    if container.len() < 112 || container[..4] != [0x89, b'T', b'M', b'Z'] {
        return;
    }
    let variable = u32::from_le_bytes([container[40], container[41], container[42], container[43]]) as usize;
    let Some(end) = variable.checked_add(44).filter(|&e| e >= 112 && e <= container.len()) else { return };
    container[44..60].copy_from_slice(&models.two_d.id().0);
    container[60..76].copy_from_slice(&models.three_d.id().0);
    let crc = crc32c::crc32c(&container[..end - 4]);
    container[end - 4..end].copy_from_slice(&crc.to_le_bytes());
}
