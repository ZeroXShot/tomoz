//! Portable reference kernel.

use super::Packed;
use crate::requantize;

#[inline]
fn accumulate(p: &Packed, x: &[i8], o: usize) -> i32 {
    let w = &p.w8[o * p.in_pad..(o + 1) * p.in_pad];
    p.bias[o] + w.iter().zip(x).map(|(&w, &x)| i32::from(w) * i32::from(x)).sum::<i32>()
}

pub(super) fn hidden(p: &Packed, shift: u8, src: &[i8], batch: usize, dst: &mut [i8]) {
    for s in 0..batch {
        let x = &src[s * p.in_pad..(s + 1) * p.in_pad];
        let y = &mut dst[s * p.out_pad..(s + 1) * p.out_pad];
        for (o, v) in y.iter_mut().enumerate() {
            *v = requantize(accumulate(p, x, o), shift);
        }
    }
}

pub(super) fn linear(p: &Packed, outputs: usize, src: &[i8], batch: usize, out: &mut [i32]) {
    for s in 0..batch {
        let x = &src[s * p.in_pad..(s + 1) * p.in_pad];
        for (o, v) in out[s * outputs..(s + 1) * outputs].iter_mut().enumerate() {
            *v = accumulate(p, x, o);
        }
    }
}
