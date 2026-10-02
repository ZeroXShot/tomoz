//! Coding of one tile: a block of consecutive slices and rows coded
//! independently of the rest of the volume.

use tomoz_entropy::{Decoder, Encoder};
use tomoz_nn::Scratch;

use crate::Error;
use crate::context::{Plane, RowContext, head, near, row_context};
use crate::model::{Model, ModelKind, ModelSet, OUTPUTS};
use crate::residual::{self, CoderState, Prediction};

/// Position and size of a tile in the volume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TileGeom {
    pub z0: usize,
    pub y0: usize,
    pub depth: usize,
    pub height: usize,
    pub width: usize,
}

impl TileGeom {
    pub(crate) fn samples(&self) -> usize {
        self.depth * self.height * self.width
    }
}

/// Per-thread working memory.
#[derive(Default)]
struct Work {
    ctx: RowContext,
    scratch: Scratch,
    out: Vec<i32>,
}

/// What a tile pass does with each sample.
trait Pass {
    /// The value of the first sample of the tile.
    fn t0(&mut self, bits: u32) -> Result<i32, Error>;
    /// Codes the flat-region flag; returns whether the sample equals its left
    /// neighbour.
    fn flat(&mut self, state: &mut CoderState, ctx: usize, z: usize, y: usize, x: usize, west: i32) -> bool;
    /// Codes a sample given its prediction; returns its value.
    fn sample(&mut self, state: &mut CoderState, p: Prediction, z: usize, y: usize, x: usize) -> Result<i32, Error>;
}

struct EncodePass<'a> {
    enc: Encoder,
    tile: &'a [i32],
    geom: TileGeom,
}

impl EncodePass<'_> {
    fn value(&self, z: usize, y: usize, x: usize) -> i32 {
        self.tile[(z * self.geom.height + y) * self.geom.width + x]
    }
}

impl Pass for EncodePass<'_> {
    fn t0(&mut self, bits: u32) -> Result<i32, Error> {
        let t0 = self.value(0, 0, 0);
        self.enc.encode_bypass(t0 as u32, bits);
        Ok(t0)
    }

    fn flat(&mut self, state: &mut CoderState, ctx: usize, z: usize, y: usize, x: usize, west: i32) -> bool {
        let equal = self.value(z, y, x) == west;
        residual::encode_flat(&mut self.enc, state, ctx, equal);
        equal
    }

    fn sample(&mut self, state: &mut CoderState, p: Prediction, z: usize, y: usize, x: usize) -> Result<i32, Error> {
        let v = self.value(z, y, x);
        residual::encode(&mut self.enc, state, p, i64::from(v));
        Ok(v)
    }
}

struct DecodePass<'a> {
    dec: Decoder<'a>,
    max_value: i64,
}

impl Pass for DecodePass<'_> {
    fn t0(&mut self, bits: u32) -> Result<i32, Error> {
        let t0 = i64::from(self.dec.decode_bypass(bits));
        if t0 > self.max_value {
            return Err(Error::Corrupt("first sample outside the value range"));
        }
        Ok(t0 as i32)
    }

    fn flat(&mut self, state: &mut CoderState, ctx: usize, _z: usize, _y: usize, _x: usize, _west: i32) -> bool {
        residual::decode_flat(&mut self.dec, state, ctx)
    }

    fn sample(&mut self, state: &mut CoderState, p: Prediction, _z: usize, _y: usize, _x: usize) -> Result<i32, Error> {
        residual::decode(&mut self.dec, state, p, self.max_value).map(|v| v as i32)
    }
}

/// Records predictions instead of coding (golden tests and diagnostics).
struct TracePass<'a> {
    tile: &'a [i32],
    geom: TileGeom,
    trace: Vec<Option<(i64, i64)>>,
}

impl Pass for TracePass<'_> {
    fn t0(&mut self, _bits: u32) -> Result<i32, Error> {
        Ok(self.tile[0])
    }

    fn flat(&mut self, _state: &mut CoderState, _ctx: usize, z: usize, y: usize, x: usize, west: i32) -> bool {
        let equal = self.tile[(z * self.geom.height + y) * self.geom.width + x] == west;
        if equal {
            self.trace.push(None);
        }
        equal
    }

    fn sample(&mut self, _state: &mut CoderState, p: Prediction, z: usize, y: usize, x: usize) -> Result<i32, Error> {
        self.trace.push(Some((p.mu16, p.log_scale8)));
        Ok(self.tile[(z * self.geom.height + y) * self.geom.width + x])
    }
}

/// Predictions of every sample of a tile in raster order: `None` where the
/// flat-region flag alone codes the sample, else `(mu16, log_scale8)`.
pub(crate) fn trace(tile: &[i32], geom: TileGeom, models: &ModelSet, max_value: i64) -> Vec<Option<(i64, i64)>> {
    let mut pass = TracePass { tile, geom, trace: Vec::with_capacity(geom.samples()) };
    let _ = run(&mut pass, geom, models, max_value, |_| {});
    pass.trace
}

/// Bits needed for values in `0..=max_value`.
fn value_bits(max_value: i64) -> u32 {
    (64 - max_value.leading_zeros()).max(1)
}

/// Runs a pass over the tile; `sink` receives every sample in raster order.
fn run<P: Pass>(
    pass: &mut P,
    geom: TileGeom,
    models: &ModelSet,
    max_value: i64,
    mut sink: impl FnMut(i32),
) -> Result<(), Error> {
    let (w, h) = (geom.width, geom.height);
    let mut cur = Plane::new(w, h);
    let mut p1 = Plane::new(w, h);
    let mut p2 = Plane::new(w, h);
    let mut state_2d = CoderState::new(models.two_d.priors());
    let mut state_3d = CoderState::new(models.three_d.priors());
    let mut work = Work::default();
    let t0 = pass.t0(value_bits(max_value))?;
    for z in 0..geom.depth {
        let (model, kind, state): (&Model, ModelKind, &mut CoderState) = if z == 0 {
            (&models.two_d, ModelKind::TwoD, &mut state_2d)
        } else {
            (&models.three_d, ModelKind::ThreeD, &mut state_3d)
        };
        let three_d = kind == ModelKind::ThreeD;
        let net = model.network();
        let stride = net.input_stride();
        let out_shift = model.out_shift();
        cur.begin(t0);
        for y in 0..h {
            cur.begin_row(y, t0);
            let prev = (z > 0).then_some((&p1, if z >= 2 { &p2 } else { &p1 }));
            row_context(kind, &cur, prev, y, stride, &mut work.ctx);
            work.out.clear();
            work.out.resize(w * OUTPUTS, 0);
            net.forward(&work.ctx.inputs, w, &mut work.scratch, &mut work.out);
            let mut previous_was_flat = false;
            for x in 0..w {
                let (yi, xi) = (y as isize, x as isize);
                let west = cur.at(yi, xi - 1);
                let ww = cur.at(yi, xi - 2);
                if work.ctx.flat[x] == west && (three_d || ww == west) {
                    let ctx = residual::flat_context(three_d, previous_was_flat);
                    if pass.flat(state, ctx, z, y, x, west) {
                        cur.set(y, x, west);
                        sink(west);
                        previous_was_flat = true;
                        continue;
                    }
                }
                previous_was_flat = false;
                let r = work.ctx.r[x];
                let nr = near([west, ww, cur.at(yi, xi - 3)], r, work.ctx.inv[x]);
                let (mu16, log_scale8) =
                    head(&work.out[x * OUTPUTS..(x + 1) * OUTPUTS], r, work.ctx.a4[x], nr, out_shift, max_value);
                let v = pass.sample(state, Prediction { mu16, log_scale8 }, z, y, x)?;
                cur.set(y, x, v);
                sink(v);
            }
            cur.end_row(y);
        }
        cur.replicate();
        std::mem::swap(&mut p2, &mut p1);
        std::mem::swap(&mut p1, &mut cur);
    }
    Ok(())
}

/// Encodes the samples of a tile (raster order, values in `0..=max_value`).
pub(crate) fn encode(tile: &[i32], geom: TileGeom, models: &ModelSet, max_value: i64) -> Vec<u8> {
    debug_assert_eq!(tile.len(), geom.samples());
    let mut pass = EncodePass { enc: Encoder::with_capacity(geom.samples()), tile, geom };
    // Encoding cannot fail: every value is in range by construction.
    let _ = run(&mut pass, geom, models, max_value, |_| {});
    pass.enc.finish()
}

/// Decodes a tile into `out` (raster order).
pub(crate) fn decode(
    bytes: &[u8],
    geom: TileGeom,
    models: &ModelSet,
    max_value: i64,
    out: &mut Vec<i32>,
) -> Result<(), Error> {
    out.clear();
    out.try_reserve_exact(geom.samples()).map_err(|_| Error::TooLarge(geom.samples()))?;
    let mut pass = DecodePass { dec: Decoder::new(bytes), max_value };
    run(&mut pass, geom, models, max_value, |v| out.push(v))?;
    pass.dec.finish().map_err(|_| Error::Corrupt("tile stream does not end where expected"))
}
