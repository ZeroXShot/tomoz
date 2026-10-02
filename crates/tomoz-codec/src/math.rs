//! Integer functions shared by the context model and the head.
//!
//! They are part of the bitstream definition: changing any of them changes
//! the predictions and therefore the format.

const LOG8_MANTISSA: [i64; 8] = [0, 1, 3, 4, 5, 6, 6, 7];

/// `8 * log2(x)` for `x >= 1`, from the position of the leading one and the
/// three bits after it. Returns 0 for `x == 0`.
#[inline]
#[must_use]
pub fn ilog2_8(x: u64) -> i64 {
    if x == 0 {
        return 0;
    }
    let e = 63 - i64::from(x.leading_zeros());
    let m = if e >= 3 { (x >> (e - 3)) & 7 } else { (x << (3 - e)) & 7 };
    8 * e + LOG8_MANTISSA[m as usize]
}

/// Piecewise-linear companding of a magnitude in 1/16 units to `[0, 127]`:
/// slope 1 up to 2, 1/4 up to 10, 1/32 above (an approximation of
/// `16 * log2(1 + v / 16)`). Reference definition: the codec computes the
/// equivalent minimum of the three segments (see `context.rs`).
#[cfg(test)]
pub fn pwl(v: i64) -> i64 {
    if v < 32 {
        v
    } else if v < 160 {
        32 + ((v - 32) >> 2)
    } else {
        (64 + ((v - 160) >> 5)).min(127)
    }
}

/// Median of three values.
#[inline]
#[must_use]
pub fn median3(a: i32, b: i32, c: i32) -> i32 {
    a.min(b).max(a.max(b).min(c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ilog2_8_values() {
        let cases =
            [(1, 0), (2, 8), (3, 13), (4, 16), (7, 22), (8, 24), (9, 25), (15, 31), (16, 32), (1024, 80), (65535, 127)];
        for (x, want) in cases {
            assert_eq!(ilog2_8(x), want, "x = {x}");
        }
        assert_eq!(ilog2_8(1 << 40), 320);
    }

    #[test]
    fn pwl_is_monotone_and_bounded() {
        let mut last = -1;
        for v in 0..5000 {
            let q = pwl(v);
            assert!(q >= last && q <= 127);
            last = q;
        }
        assert_eq!((pwl(31), pwl(32), pwl(159), pwl(160), pwl(2175), pwl(2176)), (31, 32, 63, 64, 126, 127));
    }

    #[test]
    fn median() {
        assert_eq!(median3(1, 2, 3), 2);
        assert_eq!(median3(3, 1, 2), 2);
        assert_eq!(median3(5, 5, -1), 5);
        assert_eq!(median3(-4, 9, 100), 9);
    }
}
