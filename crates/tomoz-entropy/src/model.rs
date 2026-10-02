//! Adaptive probability models.

use crate::{PROB_BITS, PROB_ONE};

/// Largest alphabet supported by [`Cdf`].
pub const MAX_SYMBOLS: usize = 16;

/// Adaptive cumulative distribution over `N` symbols (`2 <= N <= 16`).
///
/// `f[i]` is the probability that the symbol is smaller than `i`, in units of
/// 2<sup>-15</sup>; `f[0]` is always zero and `f[N]` (implicit) is one. The
/// sequence is monotone but not necessarily strictly increasing: a symbol
/// that was never seen can see its probability decay to zero. The coder
/// therefore codes with [`Cdf::boundary`], which reserves one unit for every
/// symbol, so that any symbol stays codable.
///
/// After each symbol, every boundary moves towards its target (zero below the
/// symbol, one above it) by a fraction 2<sup>-rate</sup>. The rate starts low,
/// so that a fresh model adapts in a few symbols, and grows to its final value
/// after 32 symbols. Both update rules are monotone, so the order of the
/// boundaries is preserved exactly by integer arithmetic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cdf<const N: usize> {
    f: [u16; N],
    count: u8,
}

impl<const N: usize> Default for Cdf<N> {
    fn default() -> Self {
        Self::uniform()
    }
}

impl<const N: usize> Cdf<N> {
    const VALID: () = assert!(N >= 2 && N <= MAX_SYMBOLS, "Cdf supports 2 to 16 symbols");

    /// Rate shift of a fresh model; the final rate is two more.
    const RATE0: u32 = 3 + if N > 8 {
        2
    } else if N > 2 {
        1
    } else {
        0
    };

    /// A uniform distribution.
    #[must_use]
    pub fn uniform() -> Self {
        let () = Self::VALID;
        let mut f = [0u16; N];
        for (i, v) in f.iter_mut().enumerate() {
            *v = ((i as u32 * PROB_ONE) / N as u32) as u16;
        }
        Self { f, count: 0 }
    }

    /// A model starting from the given cumulative frequencies.
    ///
    /// `freq[i]` are non-negative relative frequencies of each symbol; they are
    /// scaled to the model's precision. A zero total yields a uniform model.
    #[must_use]
    pub fn from_frequencies(freq: &[u32; N]) -> Self {
        let () = Self::VALID;
        let total: u64 = freq.iter().map(|&v| u64::from(v)).sum();
        if total == 0 {
            return Self::uniform();
        }
        let mut f = [0u16; N];
        let mut acc = 0u64;
        for i in 0..N {
            f[i] = ((acc * u64::from(PROB_ONE)) / total) as u16;
            acc += u64::from(freq[i]);
        }
        Self { f, count: 0 }
    }

    /// A model starting from cumulative values `f[0..N]` in units of
    /// 2<sup>-15</sup>; returns `None` unless `f[0] == 0` and the values are
    /// non-decreasing and at most 2<sup>15</sup>.
    #[must_use]
    pub fn from_cumulative(f: &[u16; N]) -> Option<Self> {
        let () = Self::VALID;
        let valid = f[0] == 0 && f.windows(2).all(|w| w[0] <= w[1]) && u32::from(f[N - 1]) <= PROB_ONE;
        valid.then_some(Self { f: *f, count: 0 })
    }

    /// The raw cumulative values `f[0..N]` (`f[0]` is zero).
    #[must_use]
    pub fn cumulative(&self) -> &[u16; N] {
        &self.f
    }

    /// Coding boundary of symbol `i` in `0..=N`, in units of 2<sup>-15</sup>.
    ///
    /// `boundary(0) == 0`, `boundary(N) == 2^15`, and consecutive boundaries
    /// differ by at least one.
    #[inline]
    #[must_use]
    pub fn boundary(&self, i: usize) -> u32 {
        if i >= N {
            return PROB_ONE;
        }
        i as u32 + ((u32::from(self.f[i]) * (PROB_ONE - N as u32)) >> PROB_BITS)
    }

    /// Moves the distribution towards `symbol`.
    #[inline]
    pub fn update(&mut self, symbol: usize) {
        let rate = Self::RATE0 + u32::from(self.count >= 16) + u32::from(self.count >= 32);
        self.count = self.count.saturating_add(1).min(32);
        // Branch-free so that the loop vectorises.
        for (i, f) in self.f.iter_mut().enumerate().skip(1) {
            let v = u32::from(*f);
            let up = v + ((PROB_ONE - v) >> rate);
            let down = v - (v >> rate);
            *f = if i > symbol { up } else { down } as u16;
        }
    }
}

/// Adaptive probability of a binary event.
///
/// Two estimates of the probability of a zero, with different speeds, are
/// averaged: the fast one tracks local changes, the slow one gives a precise
/// estimate in stationary regions. The estimates keep 24 bits of precision so
/// that a shift-based update can approach certainty closely (with 15 bits it
/// would stall 2<sup>rate</sup> units away from it); the coder sees the
/// average rounded to 15 bits and clamped to `[1, 2^15 - 1]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitModel {
    fast: u32,
    slow: u32,
}

impl Default for BitModel {
    fn default() -> Self {
        Self::new()
    }
}

impl BitModel {
    const FAST: u32 = 4;
    const SLOW: u32 = 7;
    const STATE_BITS: u32 = 24;
    const STATE_ONE: u32 = 1 << Self::STATE_BITS;

    /// Probability one half.
    #[must_use]
    pub const fn new() -> Self {
        Self::with_p0((PROB_ONE / 2) as u16)
    }

    /// A model starting at probability `p0 / 2^15` of a zero (clamped to the
    /// valid range).
    #[must_use]
    pub const fn with_p0(p0: u16) -> Self {
        let p = if p0 < 1 {
            1
        } else if p0 as u32 > PROB_ONE - 1 {
            PROB_ONE - 1
        } else {
            p0 as u32
        };
        let s = p << (Self::STATE_BITS - PROB_BITS);
        Self { fast: s, slow: s }
    }

    /// Probability of a zero, in units of 2<sup>-15</sup>, in `[1, 2^15 - 1]`.
    #[inline]
    #[must_use]
    pub fn p0(&self) -> u32 {
        ((self.fast + self.slow) >> (Self::STATE_BITS - PROB_BITS + 1)).clamp(1, PROB_ONE - 1)
    }

    /// Updates the model with an observed bit.
    #[inline]
    pub fn update(&mut self, bit: bool) {
        self.fast = step(self.fast, bit, Self::FAST);
        self.slow = step(self.slow, bit, Self::SLOW);
    }
}

#[inline]
fn step(p: u32, bit: bool, rate: u32) -> u32 {
    if bit { p - (p >> rate) } else { p + ((BitModel::STATE_ONE - p) >> rate) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_invariants<const N: usize>(m: &Cdf<N>) {
        assert_eq!(m.boundary(0), 0);
        assert_eq!(m.boundary(N), PROB_ONE);
        for i in 0..N {
            assert!(m.boundary(i + 1) > m.boundary(i), "symbol {i} has no room: {m:?}");
        }
        for w in m.f.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn uniform_is_valid() {
        check_invariants(&Cdf::<2>::uniform());
        check_invariants(&Cdf::<7>::uniform());
        check_invariants(&Cdf::<16>::uniform());
    }

    #[test]
    fn repeated_symbol_keeps_others_codable() {
        let mut m = Cdf::<16>::uniform();
        for _ in 0..10_000 {
            m.update(5);
            check_invariants(&m);
        }
        let p5 = m.boundary(6) - m.boundary(5);
        assert!(p5 > PROB_ONE * 95 / 100, "p5 = {p5}");
        let mut m = Cdf::<16>::uniform();
        for _ in 0..10_000 {
            m.update(0);
        }
        check_invariants(&m);
        let mut m = Cdf::<16>::uniform();
        for _ in 0..10_000 {
            m.update(15);
        }
        check_invariants(&m);
    }

    #[test]
    fn from_frequencies_scales() {
        let m = Cdf::<4>::from_frequencies(&[1, 1, 2, 0]);
        assert_eq!(m.cumulative(), &[0, 8192, 16384, 32768]);
        check_invariants(&m);
        assert_eq!(Cdf::<4>::from_frequencies(&[0; 4]), Cdf::<4>::uniform());
    }

    #[test]
    fn bit_model_stays_in_range() {
        let mut m = BitModel::new();
        for _ in 0..100_000 {
            m.update(false);
        }
        assert_eq!(m.p0(), PROB_ONE - 1);
        for _ in 0..100_000 {
            m.update(true);
        }
        assert_eq!(m.p0(), 1);
        assert_eq!(BitModel::with_p0(0).p0(), 1);
        assert_eq!(BitModel::with_p0(u16::MAX).p0(), PROB_ONE - 1);
    }
}
