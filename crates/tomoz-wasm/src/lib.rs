//! Tomoz for WebAssembly hosts, through a small C ABI.
//!
//! No bindings generator is needed: the host allocates input buffers with
//! [`tomoz_alloc`], calls a function, reads the result buffer (whose length
//! [`tomoz_result_len`] returns) and frees it with [`tomoz_free`]. On error
//! a function returns null and [`tomoz_error`] gives the message.
//!
//! `js/tomoz.mjs` wraps this ABI for browsers and Node.js. The decoder runs
//! the same integer arithmetic as on every other platform (with the
//! WebAssembly SIMD kernel when the module is built with `+simd128`), so a
//! volume decodes to the same samples everywhere.

#![allow(unsafe_code)] // The C ABI exchanges raw pointers with the host.

use std::cell::RefCell;

use tomoz_codec::{DecodeOptions, EncodeOptions, ModelSet, Volume};

thread_local! {
    static RESULT_LEN: RefCell<usize> = const { RefCell::new(0) };
    static ERROR: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static MODELS: ModelSet = ModelSet::builtin();
}

/// Hands a byte buffer to the host; it must be released with [`tomoz_free`].
fn give(mut v: Vec<u8>) -> *mut u8 {
    v.shrink_to_fit();
    RESULT_LEN.with(|l| *l.borrow_mut() = v.len());
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

fn fail(message: impl std::fmt::Display) -> *mut u8 {
    ERROR.with(|e| *e.borrow_mut() = message.to_string().into_bytes());
    RESULT_LEN.with(|l| *l.borrow_mut() = 0);
    std::ptr::null_mut()
}

/// Allocates `len` bytes for the host to fill.
#[unsafe(no_mangle)]
pub extern "C" fn tomoz_alloc(len: usize) -> *mut u8 {
    let mut v = vec![0u8; len];
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// Frees a buffer from [`tomoz_alloc`] or a result.
///
/// # Safety
///
/// `ptr` must come from [`tomoz_alloc`] (with the same `len`) or be a result
/// whose length was `len`, and must not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tomoz_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        // SAFETY: guaranteed by the caller; results are exactly `len` long
        // (shrunk before being handed out).
        drop(unsafe { Vec::from_raw_parts(ptr, len, len) });
    }
}

/// Length of the last result.
#[unsafe(no_mangle)]
pub extern "C" fn tomoz_result_len() -> usize {
    RESULT_LEN.with(|l| *l.borrow())
}

/// The last error message (UTF-8), valid until the next call; its length is
/// returned by [`tomoz_error_len`].
#[unsafe(no_mangle)]
pub extern "C" fn tomoz_error() -> *const u8 {
    ERROR.with(|e| e.borrow().as_ptr())
}

/// Length of the last error message.
#[unsafe(no_mangle)]
pub extern "C" fn tomoz_error_len() -> usize {
    ERROR.with(|e| e.borrow().len())
}

/// # Safety
///
/// `ptr` must point to `len` readable bytes (or `len` must be zero).
unsafe fn input<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    if len == 0 || ptr.is_null() {
        &[]
    } else {
        // SAFETY: guaranteed by the caller.
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

/// Serialises a decoded volume: depth, height, width, bits, signed (u32 LE
/// each), then the samples as 16-bit little-endian integers.
pub fn volume_bytes(v: &Volume) -> Vec<u8> {
    let mut out = Vec::with_capacity(20 + 2 * v.samples().len());
    for x in [v.depth() as u32, v.height() as u32, v.width() as u32, u32::from(v.bits()), u32::from(v.signed())] {
        out.extend_from_slice(&x.to_le_bytes());
    }
    for &s in v.samples() {
        out.extend_from_slice(&(s as u16).to_le_bytes());
    }
    out
}

/// Decodes slices `z0..z1` of a container (all slices when `z1 == 0`).
/// The result is laid out as described by [`volume_bytes`].
///
/// # Safety
///
/// `ptr` must point to `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tomoz_decode(ptr: *const u8, len: usize, z0: u32, z1: u32) -> *mut u8 {
    // SAFETY: guaranteed by the caller.
    let data = unsafe { input(ptr, len) };
    MODELS.with(|models| {
        let options = DecodeOptions::with_registry(models);
        let result = if z1 == 0 {
            tomoz_codec::decode(data, &options)
        } else {
            tomoz_codec::decode_slices(data, &options, z0 as usize..z1 as usize)
        };
        match result {
            Ok(v) => give(volume_bytes(&v)),
            Err(e) => fail(e),
        }
    })
}

/// The header of a container as JSON.
///
/// # Safety
///
/// `ptr` must point to `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tomoz_info(ptr: *const u8, len: usize) -> *mut u8 {
    // SAFETY: guaranteed by the caller.
    let data = unsafe { input(ptr, len) };
    match tomoz_codec::inspect(data) {
        Ok(h) => {
            let sha: String = h.sha256.iter().map(|b| format!("{b:02x}")).collect();
            give(
                format!(
                    "{{\"depth\":{},\"height\":{},\"width\":{},\"bits\":{},\"signed\":{},\"slab\":{},\"stripe\":{},\"tiles\":{},\"packed\":{},\"sha256\":\"{sha}\",\"model2d\":\"{}\",\"model3d\":\"{}\"}}",
                    h.depth, h.height, h.width, h.bits, h.signed, h.slab, h.stripe, h.tile_count(), h.packed(), h.model_2d, h.model_3d
                )
                .into_bytes(),
            )
        }
        Err(e) => fail(e),
    }
}

/// Encodes a volume of 16-bit little-endian samples (`depth × height ×
/// width` of them, two's complement when `signed`) into tiles of `slab`
/// slices × `stripe` rows (0: the defaults, 32 × 512); `packing` (0 or 1)
/// allows histogram packing.
///
/// # Safety
///
/// `ptr` must point to `len` readable bytes.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn tomoz_encode(
    ptr: *const u8,
    len: usize,
    depth: u32,
    height: u32,
    width: u32,
    bits: u32,
    signed: u32,
    slab: u32,
    stripe: u32,
    packing: u32,
) -> *mut u8 {
    // SAFETY: guaranteed by the caller.
    let data = unsafe { input(ptr, len) };
    if data.len() % 2 != 0 {
        return fail("odd number of bytes for 16-bit samples");
    }
    let samples: Vec<i32> = data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&c| {
            let v = u16::from_le_bytes(c);
            if signed != 0 { i32::from(v as i16) } else { i32::from(v) }
        })
        .collect();
    let volume =
        match Volume::new(depth as usize, height as usize, width as usize, bits.min(255) as u8, signed != 0, samples) {
            Ok(v) => v,
            Err(e) => return fail(e),
        };
    let (Ok(slab), Ok(stripe)) = (u16::try_from(slab), u16::try_from(stripe)) else {
        return fail("slab and stripe must be below 65536");
    };
    MODELS.with(|models| {
        let defaults = EncodeOptions::with_models(models.clone());
        let options = EncodeOptions {
            slab: if slab == 0 { defaults.slab } else { slab },
            stripe: if stripe == 0 { defaults.stripe } else { stripe },
            packing: packing != 0,
            ..defaults
        };
        match tomoz_codec::encode(&volume, &options) {
            Ok(b) => give(b),
            Err(e) => fail(e),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_roundtrip() {
        let samples: Vec<u8> =
            (0..2 * 8 * 8).flat_map(|i: i32| ((i * 37) % 1000 - 500).to_le_bytes()[..2].to_vec()).collect();
        // SAFETY: valid buffers throughout.
        unsafe {
            let enc = tomoz_encode(samples.as_ptr(), samples.len(), 2, 8, 8, 16, 1, 0, 0, 1);
            assert!(!enc.is_null());
            let enc_len = tomoz_result_len();
            let dec = tomoz_decode(enc, enc_len, 0, 0);
            assert!(!dec.is_null());
            let dec_len = tomoz_result_len();
            let out = std::slice::from_raw_parts(dec, dec_len);
            assert_eq!(&out[20..], &samples[..]);
            assert_eq!(&out[..4], &2u32.to_le_bytes());
            let one = tomoz_decode(enc, enc_len, 1, 2);
            let one_len = tomoz_result_len();
            assert_eq!(&std::slice::from_raw_parts(one, one_len)[20..], &samples[128..]);
            tomoz_free(one, one_len);
            tomoz_free(dec, dec_len);
            tomoz_free(enc, enc_len);
            let error = || {
                std::str::from_utf8(std::slice::from_raw_parts(tomoz_error(), tomoz_error_len())).unwrap().to_owned()
            };
            assert!(tomoz_decode(b"nope".as_ptr(), 4, 0, 0).is_null());
            assert!(error().contains("not a Tomoz"), "{}", error());
            assert!(tomoz_encode(samples.as_ptr(), 3, 1, 1, 1, 16, 0, 0, 0, 1).is_null());
            assert!(error().contains("odd number"), "{}", error());
        }
    }
}
