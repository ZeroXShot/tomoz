//! x86-64 AVX2 kernel.
//!
//! Inputs are widened to 16 bits; `vpmaddwd` multiplies a broadcast pair of
//! inputs with the weights of eight outputs and adds each pair of products
//! into a 32-bit lane. Products are at most 2^14 in magnitude, so the pair
//! sums are exact.
#![allow(unsafe_code)]

use core::arch::x86_64::*;

use super::Packed;

/// Accumulators of the 8 outputs starting at `o8`; `x16` holds the inputs
/// widened to 16 bits.
#[inline]
#[target_feature(enable = "avx2")]
fn block(p: &Packed, o8: usize, x16: &[i16]) -> __m256i {
    let pairs = p.in_pad / 2;
    let w = &p.w16[o8 * p.in_pad..(o8 + 8) * p.in_pad];
    let bias = &p.bias[o8..o8 + 8];
    // SAFETY: `bias` is exactly 8 values; each weight load reads 16 values
    // within `w`, whose length is `pairs * 16`.
    unsafe {
        let mut acc = _mm256_loadu_si256(bias.as_ptr().cast());
        for k in 0..pairs {
            let pair = (u32::from(x16[2 * k] as u16) | (u32::from(x16[2 * k + 1] as u16) << 16)) as i32;
            let wv = _mm256_loadu_si256(w[k * 16..k * 16 + 16].as_ptr().cast());
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(wv, _mm256_set1_epi32(pair)));
        }
        acc
    }
}

#[inline]
fn widen(x: &[i8], out: &mut Vec<i16>) {
    out.clear();
    out.extend(x.iter().map(|&v| i16::from(v)));
}

/// # Safety
///
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn hidden(p: &Packed, shift: u8, src: &[i8], batch: usize, dst: &mut [i8]) {
    let mut x16 = Vec::with_capacity(p.in_pad);
    let round = if shift == 0 { 0 } else { 1 << (shift - 1) };
    for s in 0..batch {
        widen(&src[s * p.in_pad..(s + 1) * p.in_pad], &mut x16);
        let y = &mut dst[s * p.out_pad..(s + 1) * p.out_pad];
        for o8 in (0..p.out_pad).step_by(8) {
            let acc = block(p, o8, &x16);
            let mut lanes = [0i32; 8];
            // SAFETY: `lanes` holds 8 values.
            unsafe {
                let v = _mm256_add_epi32(acc, _mm256_set1_epi32(round));
                let v = _mm256_sra_epi32(v, _mm_cvtsi32_si128(i32::from(shift)));
                let v = _mm256_min_epi32(_mm256_max_epi32(v, _mm256_setzero_si256()), _mm256_set1_epi32(127));
                _mm256_storeu_si256(lanes.as_mut_ptr().cast(), v);
            }
            for (d, &v) in y[o8..o8 + 8].iter_mut().zip(&lanes) {
                *d = v as i8;
            }
        }
    }
}

/// # Safety
///
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn linear(p: &Packed, outputs: usize, src: &[i8], batch: usize, out: &mut [i32]) {
    let mut x16 = Vec::with_capacity(p.in_pad);
    for s in 0..batch {
        widen(&src[s * p.in_pad..(s + 1) * p.in_pad], &mut x16);
        let y = &mut out[s * outputs..(s + 1) * outputs];
        for o8 in (0..outputs).step_by(8) {
            let acc = block(p, o8, &x16);
            let mut lanes = [0i32; 8];
            // SAFETY: `lanes` holds 8 values.
            unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), acc) };
            let n = (outputs - o8).min(8);
            y[o8..o8 + n].copy_from_slice(&lanes[..n]);
        }
    }
}
