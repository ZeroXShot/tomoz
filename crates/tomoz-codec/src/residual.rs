//! Coding of one sample given its prediction.
//!
//! The predicted scale selects the context of a 16-symbol adaptive model and
//! how many low-order bits of the residual magnitude bypass it: residuals are
//! split as `|e| = q * 2^k + low` with `k = floor(log2 scale) - 1`, so that `q`
//! has a small alphabet whatever the scale. Large `q` escape to bypass bits.
//! The sign is coded with a binary model whose context includes the
//! fractional part of the predicted mean.

use tomoz_entropy::{BitModel, Cdf, Decoder, Encoder};

use crate::Error;
use crate::model::{FLAT_CONTEXTS, Priors, SIGN_CONTEXTS, TOKEN_CONTEXTS, TOKENS};

/// Tokens below this value are the quotient itself.
const DIRECT: u64 = 14;
/// Token 14 covers quotients `14..30` with 4 extra bits; token 15 escapes.
const ESCAPE: u64 = DIRECT + 16;
/// Largest number of low-order bits that bypass the token model.
const MAX_K: i64 = 14;

/// Adaptive models of one tile.
#[derive(Clone, Debug)]
pub(crate) struct CoderState {
    token: [Cdf<TOKENS>; TOKEN_CONTEXTS],
    sign: [BitModel; SIGN_CONTEXTS],
    flat: [BitModel; FLAT_CONTEXTS],
}

impl CoderState {
    pub(crate) fn new(priors: &Priors) -> Self {
        Self {
            token: priors.token.map(|f| Cdf::from_cumulative(&f).unwrap_or_default()),
            sign: priors.sign.map(BitModel::with_p0),
            flat: priors.flat.map(BitModel::with_p0),
        }
    }
}

/// Context of a sample: the predicted mean in 1/16 units and the log2 scale
/// in 1/8 octaves.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Prediction {
    pub mu16: i64,
    pub log_scale8: i64,
}

struct Plan {
    predicted: i64,
    k: u32,
    token_ctx: usize,
    sign_ctx: usize,
}

impl Prediction {
    fn plan(self) -> Plan {
        let predicted = (self.mu16 + 8) >> 4;
        let frac = self.mu16 - 16 * predicted;
        let l = self.log_scale8;
        let k = ((l >> 3) - 1).clamp(0, MAX_K);
        let token_ctx = if k == 0 { ((l + 16) >> 1).clamp(0, 15) as usize } else { 16 + ((l & 7) >> 1) as usize };
        let sign_ctx = ((frac + 8) >> 2) as usize * 17 + if k == 0 { token_ctx } else { 16 };
        Plan { predicted, k: k as u32, token_ctx, sign_ctx }
    }
}

/// Context of the flat-region flag.
#[inline]
pub(crate) fn flat_context(three_d: bool, previous_was_flat: bool) -> usize {
    usize::from(three_d) * 2 + usize::from(previous_was_flat)
}

pub(crate) fn encode_flat(enc: &mut Encoder, s: &mut CoderState, ctx: usize, equal: bool) {
    enc.encode_bit(&mut s.flat[ctx], !equal);
}

pub(crate) fn decode_flat(dec: &mut Decoder<'_>, s: &mut CoderState, ctx: usize) -> bool {
    !dec.decode_bit(&mut s.flat[ctx])
}

pub(crate) fn encode(enc: &mut Encoder, s: &mut CoderState, p: Prediction, x: i64) {
    let plan = p.plan();
    let e = x - plan.predicted;
    let m = e.unsigned_abs();
    let q = m >> plan.k;
    let token = q.min(DIRECT) as usize + usize::from(q >= ESCAPE);
    enc.encode_symbol(&mut s.token[plan.token_ctx], token);
    if q >= ESCAPE {
        let v = q - ESCAPE + 1;
        let n = 63 - v.leading_zeros();
        enc.encode_bypass(n, 5);
        enc.encode_bypass((v - (1 << n)) as u32, n);
    } else if q >= DIRECT {
        enc.encode_bypass((q - DIRECT) as u32, 4);
    }
    if plan.k > 0 {
        enc.encode_bypass((m & ((1 << plan.k) - 1)) as u32, plan.k);
    }
    if m != 0 {
        enc.encode_bit(&mut s.sign[plan.sign_ctx], e < 0);
    }
}

pub(crate) fn decode(dec: &mut Decoder<'_>, s: &mut CoderState, p: Prediction, max_value: i64) -> Result<i64, Error> {
    let plan = p.plan();
    let token = dec.decode_symbol(&mut s.token[plan.token_ctx]) as u64;
    let q = match token {
        15 => {
            let n = dec.decode_bypass(5);
            if n > 20 {
                return Err(Error::Corrupt("residual escape out of range"));
            }
            (1u64 << n) + u64::from(dec.decode_bypass(n)) - 1 + ESCAPE
        }
        14 => DIRECT + u64::from(dec.decode_bypass(4)),
        t => t,
    };
    let mut m = q << plan.k;
    if plan.k > 0 {
        m |= u64::from(dec.decode_bypass(plan.k));
    }
    let e = if m != 0 && dec.decode_bit(&mut s.sign[plan.sign_ctx]) { -(m as i64) } else { m as i64 };
    let x = plan.predicted + e;
    if (0..=max_value).contains(&x) { Ok(x) } else { Err(Error::Corrupt("decoded sample outside the value range")) }
}
