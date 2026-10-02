//! AArch64 Advanced SIMD kernels.
//!
//! `*_dot` use the 8-bit dot-product extension (SDOT): one instruction adds
//! the dot products of four 4-input groups to four accumulators. The others
//! widen to 16 bits and multiply-accumulate one input at a time.
#![allow(unsafe_code)]

use core::arch::aarch64::*;

use super::Packed;

/// Accumulators of the 4 outputs starting at `o4` for the inputs at `x`.
///
/// # Safety
///
/// `x` must point to `p.in_pad` readable bytes and the CPU must support the
/// dot-product extension.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn block_dot(p: &Packed, o4: usize, x: *const i8) -> int32x4_t {
    let groups = p.in_pad / 4;
    // SAFETY: `o4 + 4 <= out_pad` and the blocks of this output block span
    // `groups * 16` bytes from its start, within `w8`; `x` is valid for
    // `in_pad` bytes, read 16 at a time.
    unsafe {
        let mut acc = vld1q_s32(p.bias.as_ptr().add(o4));
        let mut w = p.w8.as_ptr().add(o4 * groups * 4);
        for g16 in 0..p.in_pad / 16 {
            let xv = vld1q_s8(x.add(g16 * 16));
            acc = vdotq_laneq_s32::<0>(acc, vld1q_s8(w), xv);
            acc = vdotq_laneq_s32::<1>(acc, vld1q_s8(w.add(16)), xv);
            acc = vdotq_laneq_s32::<2>(acc, vld1q_s8(w.add(32)), xv);
            acc = vdotq_laneq_s32::<3>(acc, vld1q_s8(w.add(48)), xv);
            w = w.add(64);
        }
        acc
    }
}

/// Accumulators of the 4 outputs starting at `o4` for 4 samples at once:
/// each weight block is loaded once and used for the four samples, and the
/// four independent accumulator chains hide the latency of SDOT.
///
/// # Safety
///
/// Each `x[i]` must point to `p.in_pad` readable bytes and the CPU must
/// support the dot-product extension.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn block_dot4(p: &Packed, o4: usize, x: [*const i8; 4]) -> [int32x4_t; 4] {
    let groups = p.in_pad / 4;
    // SAFETY: as in `block_dot`, for each of the four samples.
    unsafe {
        // Two partial sums per sample (even and odd input groups) give eight
        // independent dependency chains, enough to keep both SIMD pipes busy
        // despite the latency of SDOT.
        let b = vld1q_s32(p.bias.as_ptr().add(o4));
        let z = vdupq_n_s32(0);
        let (mut a0, mut a1, mut a2, mut a3) = (b, b, b, b);
        let (mut c0, mut c1, mut c2, mut c3) = (z, z, z, z);
        let mut w = p.w8.as_ptr().add(o4 * groups * 4);
        for g16 in 0..p.in_pad / 16 {
            let x0 = vld1q_s8(x[0].add(g16 * 16));
            let x1 = vld1q_s8(x[1].add(g16 * 16));
            let x2 = vld1q_s8(x[2].add(g16 * 16));
            let x3 = vld1q_s8(x[3].add(g16 * 16));
            let w0 = vld1q_s8(w);
            let w1 = vld1q_s8(w.add(16));
            let w2 = vld1q_s8(w.add(32));
            let w3 = vld1q_s8(w.add(48));
            a0 = vdotq_laneq_s32::<0>(a0, w0, x0);
            a1 = vdotq_laneq_s32::<0>(a1, w0, x1);
            a2 = vdotq_laneq_s32::<0>(a2, w0, x2);
            a3 = vdotq_laneq_s32::<0>(a3, w0, x3);
            c0 = vdotq_laneq_s32::<1>(c0, w1, x0);
            c1 = vdotq_laneq_s32::<1>(c1, w1, x1);
            c2 = vdotq_laneq_s32::<1>(c2, w1, x2);
            c3 = vdotq_laneq_s32::<1>(c3, w1, x3);
            a0 = vdotq_laneq_s32::<2>(a0, w2, x0);
            a1 = vdotq_laneq_s32::<2>(a1, w2, x1);
            a2 = vdotq_laneq_s32::<2>(a2, w2, x2);
            a3 = vdotq_laneq_s32::<2>(a3, w2, x3);
            c0 = vdotq_laneq_s32::<3>(c0, w3, x0);
            c1 = vdotq_laneq_s32::<3>(c1, w3, x1);
            c2 = vdotq_laneq_s32::<3>(c2, w3, x2);
            c3 = vdotq_laneq_s32::<3>(c3, w3, x3);
            w = w.add(64);
        }
        [vaddq_s32(a0, c0), vaddq_s32(a1, c1), vaddq_s32(a2, c2), vaddq_s32(a3, c3)]
    }
}

/// Accumulators of the 16 outputs starting at `o16` for 4 samples: `[sample][block of 4 outputs]`.
///
/// # Safety
///
/// Each `x[i]` must point to `p.in_pad` readable bytes, `o16 + 16 <= out_pad`
/// and the CPU must support the dot-product extension.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn block16_dot4(p: &Packed, o16: usize, x: [*const i8; 4]) -> [[int32x4_t; 4]; 4] {
    let groups = p.in_pad / 4;
    // SAFETY: the four output blocks of 4 starting at `o16` lie within `w8`
    // and `bias`; each sample row has `in_pad` bytes.
    unsafe {
        let mut acc = [[vdupq_n_s32(0); 4]; 4];
        for (b, row) in acc[0].iter_mut().enumerate() {
            *row = vld1q_s32(p.bias.as_ptr().add(o16 + 4 * b));
        }
        acc[1] = acc[0];
        acc[2] = acc[0];
        acc[3] = acc[0];
        let wbase: [*const i8; 4] = core::array::from_fn(|b| p.w8.as_ptr().add((o16 + 4 * b) * groups * 4));
        for g16 in 0..p.in_pad / 16 {
            let xv = [
                vld1q_s8(x[0].add(g16 * 16)),
                vld1q_s8(x[1].add(g16 * 16)),
                vld1q_s8(x[2].add(g16 * 16)),
                vld1q_s8(x[3].add(g16 * 16)),
            ];
            macro_rules! lanes {
                ($b:literal) => {
                    let w = wbase[$b].add(g16 * 64);
                    let (w0, w1, w2, w3) = (vld1q_s8(w), vld1q_s8(w.add(16)), vld1q_s8(w.add(32)), vld1q_s8(w.add(48)));
                    for s in 0..4 {
                        let mut a = acc[s][$b];
                        a = vdotq_laneq_s32::<0>(a, w0, xv[s]);
                        a = vdotq_laneq_s32::<1>(a, w1, xv[s]);
                        a = vdotq_laneq_s32::<2>(a, w2, xv[s]);
                        a = vdotq_laneq_s32::<3>(a, w3, xv[s]);
                        acc[s][$b] = a;
                    }
                };
            }
            lanes!(0);
            lanes!(1);
            lanes!(2);
            lanes!(3);
        }
        acc
    }
}

/// Requantises 16 accumulators of one sample into 16 bytes.
#[inline]
#[target_feature(enable = "neon")]
fn requantize16(acc: [int32x4_t; 4], shift: u8, out: &mut [i8]) {
    let sh = vdupq_n_s32(-i32::from(shift));
    let lo = vcombine_s16(vqmovn_s32(vrshlq_s32(acc[0], sh)), vqmovn_s32(vrshlq_s32(acc[1], sh)));
    let hi = vcombine_s16(vqmovn_s32(vrshlq_s32(acc[2], sh)), vqmovn_s32(vrshlq_s32(acc[3], sh)));
    let v = vmaxq_s8(vcombine_s8(vqmovn_s16(lo), vqmovn_s16(hi)), vdupq_n_s8(0));
    let out: &mut [i8; 16] = (&mut out[..16]).try_into().expect("16 outputs");
    // SAFETY: `out` is exactly 16 bytes.
    unsafe { vst1q_s8(out.as_mut_ptr(), v) };
}

/// Accumulators of the 4 outputs starting at `o4`, 16-bit arithmetic.
///
/// # Safety
///
/// `x` must point to `p.in_pad` readable bytes.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn block(p: &Packed, o4: usize, x: *const i8) -> int32x4_t {
    // SAFETY: the 16-bit weights of this output block span `in_pad * 4`
    // values from `o4 * in_pad`; `x` is valid for `in_pad` bytes.
    unsafe {
        let mut acc = vld1q_s32(p.bias.as_ptr().add(o4));
        let w = p.w16.as_ptr().add(o4 * p.in_pad);
        for i in 0..p.in_pad {
            acc = vmlal_n_s16(acc, vld1_s16(w.add(i * 4)), i16::from(*x.add(i)));
        }
        acc
    }
}

/// `clamp((acc + 2^(shift-1)) >> shift, 0, 127)` on four lanes, stored as
/// bytes.
///
/// A negative count makes VQRSHL a rounding right shift; the saturating
/// narrowings clamp to the i16 and then the i8 range, and the final maximum
/// with zero completes the clamp to `[0, 127]`. Accumulators are bounded by
/// 2^30, so no step saturates before the narrowing and the result equals the
/// scalar definition.
#[inline]
#[target_feature(enable = "neon")]
fn requantize4(acc: int32x4_t, shift: u8, out: &mut [i8]) {
    let v = vqrshlq_s32(acc, vdupq_n_s32(-i32::from(shift)));
    let n16 = vqmovn_s32(v);
    let n8 = vmax_s8(vqmovn_s16(vcombine_s16(n16, n16)), vdup_n_s8(0));
    let word = vget_lane_u32::<0>(vreinterpret_u32_s8(n8));
    out[..4].copy_from_slice(&word.to_le_bytes().map(|b| b as i8));
}

#[inline]
#[target_feature(enable = "neon")]
fn store_linear(acc: int32x4_t, out: &mut [i32]) {
    let lanes =
        [vgetq_lane_s32::<0>(acc), vgetq_lane_s32::<1>(acc), vgetq_lane_s32::<2>(acc), vgetq_lane_s32::<3>(acc)];
    let n = out.len().min(4);
    out[..n].copy_from_slice(&lanes[..n]);
}

/// # Safety
///
/// The CPU must support the dot-product extension; buffers must hold
/// `batch` rows.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn hidden_dot(p: &Packed, shift: u8, src: &[i8], batch: usize, dst: &mut [i8]) {
    assert!(src.len() >= batch * p.in_pad && dst.len() >= batch * p.out_pad);
    let full = batch / 4 * 4;
    for s in (0..full).step_by(4) {
        let x = core::array::from_fn(|i| src[(s + i) * p.in_pad..].as_ptr());
        for o16 in (0..p.out_pad).step_by(16) {
            // SAFETY: each sample has `in_pad` bytes left (asserted above) and
            // `out_pad` is a multiple of 16.
            let acc = unsafe { block16_dot4(p, o16, x) };
            for (i, a) in acc.into_iter().enumerate() {
                let row = (s + i) * p.out_pad + o16;
                requantize16(a, shift, &mut dst[row..row + 16]);
            }
        }
    }
    for s in full..batch {
        let x = src[s * p.in_pad..].as_ptr();
        let y = &mut dst[s * p.out_pad..(s + 1) * p.out_pad];
        for o4 in (0..p.out_pad).step_by(4) {
            // SAFETY: `x` has `in_pad` bytes left (asserted above).
            let acc = unsafe { block_dot(p, o4, x) };
            requantize4(acc, shift, &mut y[o4..o4 + 4]);
        }
    }
}

/// # Safety
///
/// The CPU must support the dot-product extension; buffers must hold
/// `batch` rows.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn linear_dot(p: &Packed, outputs: usize, src: &[i8], batch: usize, out: &mut [i32]) {
    assert!(src.len() >= batch * p.in_pad && out.len() >= batch * outputs);
    let full = batch / 4 * 4;
    for s in (0..full).step_by(4) {
        let x = core::array::from_fn(|i| src[(s + i) * p.in_pad..].as_ptr());
        for o4 in (0..outputs).step_by(4) {
            // SAFETY: each sample has `in_pad` bytes left (asserted above).
            let acc = unsafe { block_dot4(p, o4, x) };
            for (i, a) in acc.into_iter().enumerate() {
                let row = (s + i) * outputs;
                store_linear(a, &mut out[row + o4..row + outputs]);
            }
        }
    }
    for s in full..batch {
        let x = src[s * p.in_pad..].as_ptr();
        let y = &mut out[s * outputs..(s + 1) * outputs];
        for o4 in (0..outputs).step_by(4) {
            // SAFETY: `x` has `in_pad` bytes left (asserted above).
            let acc = unsafe { block_dot(p, o4, x) };
            store_linear(acc, &mut y[o4..]);
        }
    }
}

/// # Safety
///
/// Buffers must hold `batch` rows.
#[target_feature(enable = "neon")]
pub(super) unsafe fn hidden(p: &Packed, shift: u8, src: &[i8], batch: usize, dst: &mut [i8]) {
    assert!(src.len() >= batch * p.in_pad && dst.len() >= batch * p.out_pad);
    for s in 0..batch {
        let x = src[s * p.in_pad..].as_ptr();
        let y = &mut dst[s * p.out_pad..(s + 1) * p.out_pad];
        for o4 in (0..p.out_pad).step_by(4) {
            // SAFETY: `x` has `in_pad` bytes left (asserted above).
            let acc = unsafe { block(p, o4, x) };
            requantize4(acc, shift, &mut y[o4..o4 + 4]);
        }
    }
}

/// # Safety
///
/// Buffers must hold `batch` rows.
#[target_feature(enable = "neon")]
pub(super) unsafe fn linear(p: &Packed, outputs: usize, src: &[i8], batch: usize, out: &mut [i32]) {
    assert!(src.len() >= batch * p.in_pad && out.len() >= batch * outputs);
    for s in 0..batch {
        let x = src[s * p.in_pad..].as_ptr();
        let y = &mut out[s * outputs..(s + 1) * outputs];
        for o4 in (0..outputs).step_by(4) {
            // SAFETY: `x` has `in_pad` bytes left (asserted above).
            let acc = unsafe { block(p, o4, x) };
            store_linear(acc, &mut y[o4..]);
        }
    }
}
