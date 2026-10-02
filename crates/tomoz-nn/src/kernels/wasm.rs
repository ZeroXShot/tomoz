//! WebAssembly SIMD kernel.
//!
//! `i32x4.dot_i16x8_s` multiplies a broadcast pair of 16-bit inputs with the
//! weights of four outputs and adds each pair of products into a lane; the
//! products are small enough for the sums to be exact.

use core::arch::wasm32::*;

use super::Packed;

#[inline]
fn block(p: &Packed, o4: usize, x16: &[i16]) -> v128 {
    let pairs = p.in_pad / 2;
    let w = &p.w16[o4 * p.in_pad..(o4 + 4) * p.in_pad];
    let mut acc = i32x4(p.bias[o4], p.bias[o4 + 1], p.bias[o4 + 2], p.bias[o4 + 3]);
    for k in 0..pairs {
        let c = &w[k * 8..k * 8 + 8];
        let wv = i16x8(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
        let pair = (u32::from(x16[2 * k] as u16) | (u32::from(x16[2 * k + 1] as u16) << 16)) as i32;
        acc = i32x4_add(acc, i32x4_dot_i16x8(wv, i32x4_splat(pair)));
    }
    acc
}

fn lanes(v: v128) -> [i32; 4] {
    [i32x4_extract_lane::<0>(v), i32x4_extract_lane::<1>(v), i32x4_extract_lane::<2>(v), i32x4_extract_lane::<3>(v)]
}

pub(super) fn hidden(p: &Packed, shift: u8, src: &[i8], batch: usize, dst: &mut [i8]) {
    let round = if shift == 0 { 0 } else { 1 << (shift - 1) };
    let mut x16 = Vec::with_capacity(p.in_pad);
    for s in 0..batch {
        x16.clear();
        x16.extend(src[s * p.in_pad..(s + 1) * p.in_pad].iter().map(|&v| i16::from(v)));
        let y = &mut dst[s * p.out_pad..(s + 1) * p.out_pad];
        for o4 in (0..p.out_pad).step_by(4) {
            let v = i32x4_shr(i32x4_add(block(p, o4, &x16), i32x4_splat(round)), u32::from(shift));
            let v = i32x4_min(i32x4_max(v, i32x4_splat(0)), i32x4_splat(127));
            for (d, l) in y[o4..o4 + 4].iter_mut().zip(lanes(v)) {
                *d = l as i8;
            }
        }
    }
}

pub(super) fn linear(p: &Packed, outputs: usize, src: &[i8], batch: usize, out: &mut [i32]) {
    let mut x16 = Vec::with_capacity(p.in_pad);
    for s in 0..batch {
        x16.clear();
        x16.extend(src[s * p.in_pad..(s + 1) * p.in_pad].iter().map(|&v| i16::from(v)));
        let y = &mut out[s * outputs..(s + 1) * outputs];
        for o4 in (0..outputs).step_by(4) {
            let l = lanes(block(p, o4, &x16));
            let n = (outputs - o4).min(4);
            y[o4..o4 + n].copy_from_slice(&l[..n]);
        }
    }
}
