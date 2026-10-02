//! Causal neighbourhoods and network inputs (model family TZ1).
//!
//! This is the Rust twin of `lab/src/tomoz_lab/features.py`; both define the
//! same integers and golden vectors check that they agree.

#[cfg(test)]
use crate::math::pwl;
use crate::math::{ilog2_8, median3};
use crate::model::ModelKind;

/// Padding around every slice buffer.
pub(crate) const PAD: usize = 3;

/// Neighbours in the rows above, (dy, dx).
const FAR_2D: [(isize, isize); 11] =
    [(-1, 0), (-2, 0), (-1, -1), (-1, 1), (-2, -1), (-2, 1), (-1, -2), (-1, 2), (-3, 0), (-2, -2), (-2, 2)];
/// Neighbours in the previous slice.
const FAR_P1: [(isize, isize); 13] =
    [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1), (0, -2), (0, 2), (-2, 0), (2, 0)];
/// Neighbours two slices back.
const FAR_P2: [(isize, isize); 5] = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)];

/// A slice of a tile with `PAD` samples of padding on every side.
///
/// While a slice is being coded the padding is causal: rows above hold the
/// tile constant `t0`, the columns left of row `y` hold the first sample of
/// row `y - 1` and the columns right of row `y` its last sample. Once the
/// slice is complete it is re-padded by replication and serves as context
/// for the next slices.
#[derive(Clone, Debug)]
pub(crate) struct Plane {
    width: usize,
    height: usize,
    stride: usize,
    data: Vec<i32>,
}

impl Plane {
    pub(crate) fn new(width: usize, height: usize) -> Self {
        let stride = width + 2 * PAD;
        Self { width, height, stride, data: vec![0; stride * (height + 2 * PAD)] }
    }

    #[inline]
    fn index(&self, y: isize, x: isize) -> usize {
        (y + PAD as isize) as usize * self.stride + (x + PAD as isize) as usize
    }

    /// Sample at tile coordinates (padding included).
    #[inline]
    pub(crate) fn at(&self, y: isize, x: isize) -> i32 {
        self.data[self.index(y, x)]
    }

    #[inline]
    pub(crate) fn set(&mut self, y: usize, x: usize, v: i32) {
        let i = self.index(y as isize, x as isize);
        self.data[i] = v;
    }

    /// Prepares causal padding for coding a new slice.
    pub(crate) fn begin(&mut self, t0: i32) {
        let top = PAD * self.stride;
        self.data[..top].fill(t0);
    }

    /// Row `y` (padding rows included) starting at column `dx`, `width` long.
    #[inline]
    pub(crate) fn row_slice(&self, y: isize, dx: isize) -> &[i32] {
        let i = self.index(y, dx);
        &self.data[i..i + self.width]
    }

    /// Sets the left padding of row `y` before coding it.
    pub(crate) fn begin_row(&mut self, y: usize, t0: i32) {
        let v = if y == 0 { t0 } else { self.at(y as isize - 1, 0) };
        let i = self.index(y as isize, -(PAD as isize));
        self.data[i..i + PAD].fill(v);
    }

    /// Sets the right padding of row `y` once it is complete.
    pub(crate) fn end_row(&mut self, y: usize) {
        let v = self.at(y as isize, self.width as isize - 1);
        let i = self.index(y as isize, self.width as isize);
        self.data[i..i + PAD].fill(v);
    }

    /// Re-pads a complete slice by replication.
    pub(crate) fn replicate(&mut self) {
        let (w, h, stride) = (self.width, self.height, self.stride);
        for y in 0..h {
            let row = (y + PAD) * stride;
            let first = self.data[row + PAD];
            let last = self.data[row + PAD + w - 1];
            self.data[row..row + PAD].fill(first);
            self.data[row + PAD + w..row + stride].fill(last);
        }
        let first_row = PAD * stride;
        let last_row = (PAD + h - 1) * stride;
        for p in 0..PAD {
            self.data.copy_within(first_row..first_row + stride, p * stride);
            self.data.copy_within(last_row..last_row + stride, (PAD + h + p) * stride);
        }
    }
}

/// Far context of every sample of a row, as columns over the row.
#[derive(Clone, Debug, Default)]
pub(crate) struct RowContext {
    /// Network inputs, `stride` per sample (zero padded).
    pub inputs: Vec<i8>,
    /// Reference prediction.
    pub r: Vec<i32>,
    /// Activity in units of 1/4.
    pub a4: Vec<i32>,
    /// `2^22 / a4`.
    pub inv: Vec<i32>,
    /// High and low 16 bits of `inv`, for exact 32-bit products.
    inv_hi: Vec<u32>,
    inv_lo: Vec<u32>,
    /// One quantised column before it is spread into the inputs.
    column: Vec<i8>,
    /// `v >= 0` when the rows above (and the previous slice) agree on `v`
    /// around the sample: the sample is in a flat region if its left
    /// neighbour (and, in 2-D, the one before) also equals `v`. `-1`
    /// otherwise (samples are never negative).
    pub flat: Vec<i32>,
    stride: usize,
}

impl RowContext {
    fn prepare(&mut self, width: usize, stride: usize) {
        if self.stride != stride || self.r.len() != width {
            // Padding lanes of the inputs are never written: start from zero.
            self.inputs.clear();
            self.inputs.resize(width * stride, 0);
            self.r.resize(width, 0);
            self.a4.resize(width, 0);
            self.inv.resize(width, 0);
            self.inv_hi.resize(width, 0);
            self.inv_lo.resize(width, 0);
            self.column.resize(width, 0);
            self.flat.resize(width, 0);
            self.stride = stride;
        }
    }
}

/// Quantises one neighbour column, `sign(d) * pwl((|d| * inv) >> 16)` with
/// `d = v - r`, and spreads it into lane `k` of the network inputs.
///
/// The product is formed from the 16-bit halves of `inv` so that every
/// intermediate fits in 32 bits (`|d| < 2^16`), which lets the loop run on
/// 32-bit SIMD lanes; the companding is the minimum of its three affine
/// segments, which equals [`pwl`] because the curve is concave.
#[inline]
fn quantize_column(values: &[i32], ctx: &mut RowContext, stride: usize, k: usize) {
    for ((((q, &v), &r), &hi), &lo) in ctx.column.iter_mut().zip(values).zip(&ctx.r).zip(&ctx.inv_hi).zip(&ctx.inv_lo) {
        let d = v - r;
        let a = d.unsigned_abs();
        let m = (a * hi + ((a * lo) >> 16)) as i32;
        let m = m.min(32 + ((m - 32) >> 2)).min(64 + ((m - 160) >> 5)).min(127);
        *q = (if d < 0 { -m } else { m }) as i8;
    }
    assert!(k < stride);
    for (row, &q) in ctx.inputs.chunks_exact_mut(stride).zip(&ctx.column) {
        row[k] = q;
    }
}

/// Computes the far context of row `y` of `cur`.
pub(crate) fn row_context(
    kind: ModelKind,
    cur: &Plane,
    prev: Option<(&Plane, &Plane)>,
    y: usize,
    stride: usize,
    out: &mut RowContext,
) {
    let w = cur.width;
    out.prepare(w, stride);
    let y = y as isize;
    let c = |dy: isize, dx: isize| cur.row_slice(y + dy, dx);
    let (n, nn, nw, ne) = (c(-1, 0), c(-2, 0), c(-1, -1), c(-1, 1));
    let prev = if kind == ModelKind::ThreeD { prev } else { None };
    if let Some((p1, _)) = prev {
        let (p, pn) = (p1.row_slice(y, 0), p1.row_slice(y - 1, 0));
        for x in 0..w {
            let (n, nn, nw, ne, p, pn) = (n[x], nn[x], nw[x], ne[x], p[x], pn[x]);
            out.r[x] = median3(n, p, n + p - pn);
            let act = (n - nw).abs() + (n - ne).abs() + (nn - n).abs() + (p - pn).abs() + (n - pn).abs();
            out.a4[x] = (4 * act / 5).max(2) + 2;
            out.flat[x] = if n == nw && n == ne && n == p { n } else { -1 };
        }
    } else {
        for x in 0..w {
            let (n, nn, nw, ne) = (n[x], nn[x], nw[x], ne[x]);
            out.r[x] = n;
            let act = (n - nw).abs() + (n - ne).abs() + (nn - n).abs();
            out.a4[x] = (4 * act / 3).max(2) + 2;
            out.flat[x] = if n == nw && n == ne { n } else { -1 };
        }
    }
    for x in 0..w {
        let inv = (1 << 22) / out.a4[x];
        out.inv[x] = inv;
        out.inv_hi[x] = (inv >> 16) as u32;
        out.inv_lo[x] = (inv & 0xFFFF) as u32;
    }
    let mut k = 0;
    for &(dy, dx) in &FAR_2D {
        quantize_column(c(dy, dx), out, stride, k);
        k += 1;
    }
    if let Some((p1, p2)) = prev {
        for &(dy, dx) in &FAR_P1 {
            quantize_column(p1.row_slice(y + dy, dx), out, stride, k);
            k += 1;
        }
        for &(dy, dx) in &FAR_P2 {
            quantize_column(p2.row_slice(y + dy, dx), out, stride, k);
            k += 1;
        }
    }
    assert!(k < stride);
    for (row, &a4) in out.inputs.chunks_exact_mut(stride).zip(&out.a4) {
        row[k] = (ilog2_8(a4 as u64) - 16).clamp(-127, 127) as i8;
    }
}

/// Left neighbours relative to `r` in units of A/256, within ±4096.
#[inline]
pub(crate) fn near(values: [i32; 3], r: i32, inv: i32) -> [i64; 3] {
    values.map(|v| ((i64::from(v - r) * i64::from(inv)) >> 12).clamp(-4096, 4096))
}

/// Predicted mean (1/16 units, clamped to `[0, 16 max]`) and log2 of the
/// scale in 1/8 octaves.
#[inline]
pub(crate) fn head(out: &[i32], r: i32, a4: i32, near: [i64; 3], out_shift: u8, max_value: i64) -> (i64, i64) {
    let (r, a4) = (i64::from(r), i64::from(a4));
    let o = |i: usize| i64::from(out[i]);
    let s = u32::from(out_shift);
    let m = ((o(0) << 8) + o(1) * near[0] + o(2) * near[1] + o(3) * near[2]).clamp(-(1 << 40), 1 << 40);
    let mu16 = (16 * r + ((a4 * m + (1 << (s + 3))) >> (s + 4))).clamp(0, 16 * max_value);
    let lg_w8 = ilog2_8(256 + near[0].unsigned_abs()) - 64;
    let ls8 = (8 * o(4) + o(5) * lg_w8 + (1 << (s - 1))) >> s;
    (mu16, ilog2_8(a4 as u64) - 16 + ls8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_minimum_equals_pwl() {
        for m in 0..(1 << 21) {
            let q = m.min(32 + ((m - 32) >> 2)).min(64 + ((m - 160) >> 5)).min(127);
            assert_eq!(i64::from(q), pwl(i64::from(m)), "m = {m}");
        }
    }
}
