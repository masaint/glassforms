//! Curves: integrating a flow, placing a curve in the volume, cutting it into equal steps.
//!
//! Everything here gives, bit for bit, the numbers the web version gives (`orbit`, `placement` and
//! `prepCurve` in forms.js). Several of the flows are chaotic: an orbit that differs in the last bit
//! after one step is a different shape a few thousand steps later. So the arithmetic is written in the
//! same order as there, and sines, logarithms and the rest come from [`jsmath`], which gives the bits the
//! browser's `Math` gives.

use crate::presets::Flow;
use crate::FIT;
use jsmath::{hypot2, hypot3};

/// The browser's `Math` functions, to the last bit.
///
/// A sine from the C runtime, or from the `libm` crate, is right to within one step of the last place,
/// and so is the browser's, but they are not always the same step. For most uses that is nothing; a
/// chaotic orbit doubles it every few dozen steps until it is the whole shape. So each function here is
/// the one Chrome has:
///
///   sin, cos, exp,      correctly rounded: the nearest 64-bit number to the true value, always. (Chrome
///   log, atan2          takes these from LLVM's C library. Node still has fdlibm's, which are a step off
///                       between three and thirteen times in a hundred; they are kept here as `fdlibm_sin`
///                       and so on.)
///   pow                 correctly rounded too, which is as near as can be got: see `pow`
///   sqrt                exact everywhere
///   round, hypot        the browser's own rules: see below
///
/// `tests/curve_parity.rs` checks each against exact arithmetic and pins its fingerprint over 100,000
/// inputs to the one Chrome prints (`tools/probe_math.js`).
// (the constants are written digit for digit as fdlibm has them)
#[allow(clippy::excessive_precision, clippy::approx_constant)]
pub mod jsmath {
    /// `Math.exp` as Chrome gives it: the nearest 64-bit number to the true value. (Node's is fdlibm's,
    /// [`fdlibm_exp`], one step of the last place off about one time in ten.)
    pub fn exp(x: f64) -> f64 {
        rounded::exp(x)
    }

    /// `Math.log` as Chrome gives it: the natural logarithm, correctly rounded. Worked out the slow way
    /// (microseconds): the shapes ask for a few dozen at most. Node's is [`fdlibm_log`].
    pub fn log(x: f64) -> f64 {
        whole::log(x)
    }

    /// `Math.atan2` as Chrome gives it: correctly rounded, and the slow way too. Node's is
    /// [`fdlibm_atan2`]. (Where the answer is below 2^-1022 this falls back on that one.)
    pub fn atan2(y: f64, x: f64) -> f64 {
        whole::atan2(y, x)
    }

    /// `Math.pow`: the nearest 64-bit number to the true power, after the cases the language spells out.
    ///
    /// This is the one function that cannot be made the browser's own. Chrome (like node) hands
    /// `Math.pow` to the C library it happened to be built with, which is neither this nor any other
    /// that can be linked here; its answer is the nearest one all but about four times in ten thousand,
    /// and then one step away. So an answer from here is the same as Chrome's except, that rarely, by
    /// one step: of no account in a random loop's harmonics, and in an easing curve only if the eased
    /// number drives a chaotic shape and the two are compared frame for frame.
    pub fn pow(x: f64, y: f64) -> f64 {
        whole::pow(x, y)
    }

    pub fn sqrt(x: f64) -> f64 {
        x.sqrt()
    }

    /// `Math.round`: the nearest whole number, halves going up (so -2.5 gives -2).
    pub fn round(x: f64) -> f64 {
        let c = x.ceil();
        if c - 0.5 > x { c - 1.0 } else { c }
    }

    /// `Math.hypot` of two numbers. See [`hypot3`].
    #[inline]
    pub fn hypot2(a: f64, b: f64) -> f64 {
        let v = [a.abs(), b.abs()];
        let mut max = 0.0;
        for x in v {
            if x > max { max = x; }
        }
        if max == f64::INFINITY { return f64::INFINITY; }
        if a.is_nan() || b.is_nan() { return f64::NAN; }
        if max == 0.0 { return 0.0; }
        let (mut sum, mut lost) = (0.0, 0.0);
        for x in v {
            let n = x / max;
            let add = n * n - lost;
            let next = sum + add;
            lost = (next - sum) - add;
            sum = next;
        }
        sum.sqrt() * max
    }

    /// `Math.hypot` of three numbers, as the browser works it out: divide by the largest, add the squares
    /// with a running correction for the rounding of each addition, take the root, scale back up.
    /// (Not the same as `f64::hypot` or a plain root of a sum of squares in the last bit.)
    #[inline]
    pub fn hypot3(a: f64, b: f64, c: f64) -> f64 {
        let v = [a.abs(), b.abs(), c.abs()];
        let mut max = 0.0;
        for x in v {
            if x > max { max = x; }
        }
        // an infinite side makes an infinite length whatever the others are; otherwise anything that is not a number spoils it
        if max == f64::INFINITY { return f64::INFINITY; }
        if a.is_nan() || b.is_nan() || c.is_nan() { return f64::NAN; }
        if max == 0.0 { return 0.0; }
        // The running correction, written out for three: the first square goes in as it is (nothing to
        // correct yet), the second leaves behind what its addition rounded away, the third makes that up.
        let (x, y, z) = (v[0] / max, v[1] / max, v[2] / max);
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let two = xx + yy;
        let lost = (two - xx) - yy;
        (two + (zz - lost)).sqrt() * max
    }

    /// fdlibm's exponential: `Math.exp` as node gives it. (`libm` still follows that code to the letter, but for
    /// one input: asked for exp(1), V8 answers e outright, where the routine itself gets the last bit wrong.)
    pub fn fdlibm_exp(x: f64) -> f64 {
        if x == 1.0 { std::f64::consts::E } else { libm::exp(x) }
    }

    /// fdlibm's arctangent of y/x by quadrant: `Math.atan2` as node gives it (`libm` has not parted from it).
    pub fn fdlibm_atan2(y: f64, x: f64) -> f64 {
        libm::atan2(y, x)
    }

    // fdlibm_sin, fdlibm_cos and fdlibm_log follow fdlibm's k_sin.c, k_cos.c, e_rem_pio2.c, s_sin.c, s_cos.c and e_log.c:
    //
    // ====================================================
    // Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
    //
    // Developed at SunSoft, a Sun Microsystems, Inc. business.
    // Permission to use, copy, modify, and distribute this
    // software is freely granted, provided that this notice
    // is preserved.
    // ====================================================

    /// The top half of a number's 64 bits: its sign, its exponent and the first 20 bits of its fraction.
    #[inline]
    fn high(x: f64) -> i32 {
        (x.to_bits() >> 32) as i32
    }

    /// sin(x + y) for |x| up to pi/4, where y is the small part of the angle that x could not hold
    /// (`tail`: there is one).
    #[inline]
    fn sin_near(x: f64, y: f64, tail: bool) -> f64 {
        const S1: f64 = -1.66666666666666324348e-01;
        const S2: f64 = 8.33333333332248946124e-03;
        const S3: f64 = -1.98412698298579493134e-04;
        const S4: f64 = 2.75573137070700676789e-06;
        const S5: f64 = -2.50507602534068634195e-08;
        const S6: f64 = 1.58969099521155010221e-10;
        if high(x) & 0x7fffffff < 0x3e400000 { return x; } // below 2^-27 the sine is the angle
        let z = x * x;
        let v = z * x;
        let r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
        if tail { x - ((z * (0.5 * y - v * r) - y) - v * S1) } else { x + v * (S1 + z * r) }
    }

    /// cos(x + y) for |x| up to pi/4, y as for `sin_near`.
    #[inline]
    fn cos_near(x: f64, y: f64) -> f64 {
        const C1: f64 = 4.16666666666666019037e-02;
        const C2: f64 = -1.38888888888741095749e-03;
        const C3: f64 = 2.48015872894767294178e-05;
        const C4: f64 = -2.75573143513906633035e-07;
        const C5: f64 = 2.08757232129817482790e-09;
        const C6: f64 = -1.13596475577881948265e-11;
        let ix = high(x) & 0x7fffffff;
        if ix < 0x3e400000 { return 1.0; }
        let z = x * x;
        let r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
        if ix < 0x3fd33333 { return 1.0 - (0.5 * z - (z * r - x * y)); } // |x| < 0.3
        // take about x/4 off both sides first, so that the subtraction from 1 loses nothing
        let qx = if ix > 0x3fe90000 { 0.28125 } else { f64::from_bits(((ix - 0x00200000) as u64) << 32) };
        let hz = 0.5 * z - qx;
        let a = 1.0 - qx;
        a - (hz - (z * r - x * y))
    }

    /// Angles up to this (2^19 quarter-turns) are brought into range the way the browser does it.
    const MEDIUM: i32 = 0x413921fb;

    /// Take whole quarter-turns off an angle: returns how many (their last two bits, at least) and what
    /// is left as a pair, a number within pi/4 of zero and the small part it could not hold.
    /// Only for pi/4 < |x| <= 2^19 pi/2.
    fn quarter_turns(x: f64) -> (i32, f64, f64) {
        // the top halves of the multiples of pi/2: an angle that starts with the same bits is too close to one for the quick way
        const NEAR: [i32; 32] = [
            0x3FF921FB, 0x400921FB, 0x4012D97C, 0x401921FB, 0x401F6A7A, 0x4022D97C, 0x4025FDBB, 0x402921FB, 0x402C463A, 0x402F6A7A, 0x4031475C,
            0x4032D97C, 0x40346B9C, 0x4035FDBB, 0x40378FDB, 0x403921FB, 0x403AB41B, 0x403C463A, 0x403DD85A, 0x403F6A7A, 0x40407E4C, 0x4041475C,
            0x4042106C, 0x4042D97C, 0x4043A28C, 0x40446B9C, 0x404534AC, 0x4045FDBB, 0x4046C6CB, 0x40478FDB, 0x404858EB, 0x404921FB,
        ];
        const INVPIO2: f64 = 6.36619772367581382433e-01; // 2/pi
        // pi/2 in pieces: the first 33 bits, the next 33, the next 33, and what is left after each
        const PIO2_1: f64 = 1.57079632673412561417e+00;
        const PIO2_1T: f64 = 6.07710050650619224932e-11;
        const PIO2_2: f64 = 6.07710050630396597660e-11;
        const PIO2_2T: f64 = 2.02226624879595063154e-21;
        const PIO2_3: f64 = 2.02226624871116645580e-21;
        const PIO2_3T: f64 = 8.47842766036889956997e-32;
        let hx = high(x);
        let ix = hx & 0x7fffffff;
        if ix < 0x4002d97c {
            // |x| < 3pi/4: one quarter-turn either way
            let near = ix == 0x3ff921fb; // close to pi/2: 33+53 bits of pi are not enough, use 33+33+53
            if hx > 0 {
                let mut z = x - PIO2_1;
                let (a, b) = if near {
                    z -= PIO2_2;
                    let a = z - PIO2_2T;
                    (a, (z - a) - PIO2_2T)
                } else {
                    let a = z - PIO2_1T;
                    (a, (z - a) - PIO2_1T)
                };
                return (1, a, b);
            }
            let mut z = x + PIO2_1;
            let (a, b) = if near {
                z += PIO2_2;
                let a = z + PIO2_2T;
                (a, (z - a) + PIO2_2T)
            } else {
                let a = z + PIO2_1T;
                (a, (z - a) + PIO2_1T)
            };
            return (-1, a, b);
        }
        let t = x.abs();
        let n = (t * INVPIO2 + 0.5) as i32;
        let f = n as f64;
        let mut r = t - f * PIO2_1;
        let mut w = f * PIO2_1T; // first round, good to 85 bits
        let mut a = r - w;
        if !(n < 32 && ix != NEAR[n as usize - 1]) {
            // how many bits cancelled: if too many, go round again with more of pi
            let j = ix >> 20;
            let lost = |a: f64| j - ((high(a) >> 20) & 0x7ff);
            if lost(a) > 16 {
                let t = r;
                w = f * PIO2_2;
                r = t - w;
                w = f * PIO2_2T - ((t - r) - w);
                a = r - w; // good to 118 bits
                if lost(a) > 49 {
                    let t = r;
                    w = f * PIO2_3;
                    r = t - w;
                    w = f * PIO2_3T - ((t - r) - w);
                    a = r - w; // 151 bits: covers every case
                }
            }
        }
        let b = (r - a) - w;
        if hx < 0 { (-n, -a, -b) } else { (n, a, b) }
    }

    /// fdlibm's sine: `Math.sin` as node gives it, and as Chrome gave it before it changed to a correctly
    /// rounded one. (Beyond 2^19 quarter-turns, about 8e5, this falls back on `libm`, which may differ
    /// from node in the last place.)
    pub fn fdlibm_sin(x: f64) -> f64 {
        let ix = high(x) & 0x7fffffff;
        if ix <= 0x3fe921fb { return sin_near(x, 0.0, false); } // |x| <= pi/4
        if ix > MEDIUM { return libm::sin(x); }
        let (n, a, b) = quarter_turns(x);
        match n & 3 {
            0 => sin_near(a, b, true),
            1 => cos_near(a, b),
            2 => -sin_near(a, b, true),
            _ => -cos_near(a, b),
        }
    }

    /// fdlibm's cosine: see [`fdlibm_sin`].
    pub fn fdlibm_cos(x: f64) -> f64 {
        let ix = high(x) & 0x7fffffff;
        if ix <= 0x3fe921fb { return cos_near(x, 0.0); }
        if ix > MEDIUM { return libm::cos(x); }
        let (n, a, b) = quarter_turns(x);
        match n & 3 {
            0 => cos_near(a, b),
            1 => -sin_near(a, b, true),
            2 => -cos_near(a, b),
            _ => sin_near(a, b, true),
        }
    }

    /// fdlibm's natural logarithm: `Math.log` as node gives it. (`libm` follows a later revision of the same code
    /// that sums the same series in another order: about one result in a hundred differs.)
    pub fn fdlibm_log(x: f64) -> f64 {
        const LN2_HI: f64 = 6.93147180369123816490e-01;
        const LN2_LO: f64 = 1.90821492927058770002e-10;
        const TWO54: f64 = 1.80143985094819840000e+16;
        const LG1: f64 = 6.666666666666735130e-01;
        const LG2: f64 = 3.999999999940941908e-01;
        const LG3: f64 = 2.857142874366239149e-01;
        const LG4: f64 = 2.222219843214978396e-01;
        const LG5: f64 = 1.818357216161805012e-01;
        const LG6: f64 = 1.531383769920937332e-01;
        const LG7: f64 = 1.479819860511658591e-01;
        let mut x = x;
        let mut hx = high(x);
        let mut k = 0i32;
        if hx < 0x00100000 {
            // below 2^-1022, zero or negative
            if x == 0.0 { return f64::NEG_INFINITY; }
            if hx < 0 { return f64::NAN; }
            k -= 54;
            x *= TWO54;
            hx = high(x);
        }
        if hx >= 0x7ff00000 { return x + x; }
        // x = 2^k (1 + f), with 1 + f between sqrt(1/2) and sqrt(2)
        k += (hx >> 20) - 1023;
        hx &= 0x000fffff;
        let i = (hx + 0x95f64) & 0x100000;
        x = f64::from_bits((x.to_bits() & 0xffff_ffff) | (((hx | (i ^ 0x3ff00000)) as u64) << 32));
        k += i >> 20;
        let f = x - 1.0;
        let dk = k as f64;
        if (0x000fffff & (2 + hx)) < 3 {
            // |f| < 2^-20: a short series is enough
            if f == 0.0 {
                return if k == 0 { 0.0 } else { dk * LN2_HI + dk * LN2_LO };
            }
            let r = f * f * (0.5 - 0.33333333333333333 * f);
            return if k == 0 { f - r } else { dk * LN2_HI - ((r - dk * LN2_LO) - f) };
        }
        let s = f / (2.0 + f);
        let z = s * s;
        let w = z * z;
        let t1 = w * (LG2 + w * (LG4 + w * LG6));
        let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
        let r = t2 + t1;
        if (hx - 0x6147a) | (0x6b851 - hx) > 0 {
            let hfsq = 0.5 * f * f;
            if k == 0 { f - (hfsq - s * (hfsq + r)) } else { dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f) }
        } else if k == 0 {
            f - s * (f - r)
        } else {
            dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f)
        }
    }

    /// `Math.sin` as Chrome gives it: the nearest 64-bit number to the true sine, every time. (Node's is
    /// fdlibm's, [`fdlibm_sin`], one step of the last place off about three times in a hundred: enough
    /// to send a chaotic orbit somewhere else.)
    ///
    /// Up to 2^52 the answer is exact; beyond, far outside anything a shape asks for, this falls back
    /// on [`fdlibm_sin`].
    pub fn sin(x: f64) -> f64 {
        rounded::sin(x)
    }

    /// `Math.cos` as Chrome gives it. See [`sin`].
    pub fn cos(x: f64) -> f64 {
        rounded::cos(x)
    }

    /// The same answers as [`sin`], [`cos`] and [`exp`] worked out the slow way throughout, in whole-number
    /// arithmetic with 256 binary places: what the quick ways are checked against. (Angles below 2^52.)
    pub fn sin_slowly(x: f64) -> f64 {
        rounded::sin_slowly(x)
    }
    pub fn cos_slowly(x: f64) -> f64 {
        rounded::cos_slowly(x)
    }
    pub fn exp_slowly(x: f64) -> f64 {
        whole::exp(x)
    }

    /// Correctly rounded sine, cosine and exponential, the quick way: the three that are asked for often.
    ///
    /// For a sine: take off whole steps of pi/128, look up the sine and cosine of the steps taken (kept
    /// to twice the usual precision), and put back the little that is left with a short series. All of it
    /// is done in pairs of numbers, a value and the rounding error of that value, so that the answer is
    /// known to about 70 binary places, with a bound on how far out it can be. If every number within
    /// that bound rounds to the same 64-bit number, that is the answer. Otherwise (about one time in
    /// 70,000) the sum is done again exactly, in whole numbers with 256 binary places. The exponential
    /// goes the same way, in steps of ln2/64.
    mod rounded {
        use super::whole::{exp_parts, in_two, nearest, shr, sin_cos, times, LN2, PI};
        #[cfg(test)]
        use super::whole::{add, big, plus};
        use std::sync::OnceLock;

        // ------------------------------------------------------------ sums and products without rounding error
        /// a + b as the rounded sum and exactly what the rounding lost.
        #[inline(always)]
        fn two_sum(a: f64, b: f64) -> (f64, f64) {
            let s = a + b;
            let bb = s - a;
            (s, (a - (s - bb)) + (b - bb))
        }
        /// a in two halves of 26 bits each.
        #[inline(always)]
        fn split(a: f64) -> (f64, f64) {
            let c = 134217729.0 * a; // 2^27 + 1
            let h = c - (c - a);
            (h, a - h)
        }
        /// a * b as the rounded product and exactly what the rounding lost.
        #[inline(always)]
        fn two_prod(a: f64, b: f64) -> (f64, f64) {
            let p = a * b;
            let ((ah, al), (bh, bl)) = (split(a), split(b));
            (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
        }

        // ------------------------------------------------------------ the quick way
        /// 128/pi
        const STEPS: f64 = f64::from_bits(0x40445f306dc9c883);
        /// pi/128 in four pieces: three of 30 bits, so that a whole number of them (below 2^22) is exact,
        /// then the rest. Together they are pi/128 to 143 bits.
        const P1: f64 = f64::from_bits(0x3f9921fb54000000);
        const P2: f64 = f64::from_bits(0x3db10b4611800000);
        const P3: f64 = f64::from_bits(0x3bc313198a000000);
        const P4: f64 = f64::from_bits(0x39d701b839a25205);
        /// Angles below this are within reach of the quick way: k stays below 2^22.
        const NEAR: f64 = 65536.0;
        /// Adding 1.5 * 2^52 and taking it away again leaves the nearest whole number.
        const WHOLE: f64 = 6755399441055744.0;

        /// x = k (pi/128) + y, with y within pi/256 of zero (a shade more at worst): k, and y as a pair.
        #[inline(always)]
        fn reduce(x: f64) -> (i64, f64, f64) {
            let k = (x * STEPS + WHOLE) - WHOLE;
            // k P1 is exact and so close to x that the difference is exact too
            let a = x - k * P1;
            let (s, e1) = two_sum(a, -(k * P2));
            let (s, e2) = two_sum(s, -(k * P3));
            let (yh, yl) = two_sum(s, (e1 + e2) - k * P4);
            (k as i64, yh, yl)
        }

        /// sin(k pi/128 + y) as a pair (value, what it lacks) and how far the pair may be from the truth.
        #[inline(always)]
        fn pair(k: i64, yh: f64, yl: f64) -> (f64, f64, f64) {
            let t = table();
            // S, C: the sine and cosine of the steps taken
            let ([sh, sl], [ch, cl]) = (t[(k & 255) as usize], t[((k + 64) & 255) as usize]);
            let (q, ql) = two_prod(yh, yh);
            // sin y = yh + sy:  the rest of y, then -y^3/6 + y^5/120 - y^7/5040 + y^9/362880
            let sy = yl + (yh * q * (-1.0 / 6.0 + q * (1.0 / 120.0 + q * (-1.0 / 5040.0 + q * (1.0 / 362880.0)))) - 0.5 * q * yl);
            // cos y = 1 + cy + cl2:  -y^2/2 exactly, then y^4/24 - y^6/720 + y^8/40320
            let cy = -0.5 * q;
            let cy2 = -0.5 * (ql + 2.0 * yh * yl) + q * q * (1.0 / 24.0 + q * (-1.0 / 720.0 + q * (1.0 / 40320.0)));
            // S cos y + C sin y: the three large terms exactly, the small ones as they come
            let (p, pl) = two_prod(ch, yh);
            let (m, ml) = two_prod(sh, cy);
            let (a, ae) = two_sum(sh, p);
            let (b, be) = two_sum(a, m);
            let low = ((ae + be) + (pl + ml)) + ((sl + cl * yh) + sl * cy) + (sh * cy2 + ch * sy);
            let (rh, rl) = two_sum(b, low);
            // What can be wrong: the series for sin y (about 2^-53 of y^3), the one for cos y (2^-80 of S),
            // the roundings of the small sums (2^-104 of the whole), and the last bits of pi (2^-111).
            // Each is allowed for several times over.
            let err = 4.44e-16 * (yh.abs() * q) + 1.4e-23 * sh.abs() + 1.6e-30 * b.abs() + 3.1e-33;
            (rh, rl, err)
        }

        /// The pair rounded, if everything within its error rounds the same way.
        #[inline(always)]
        fn settled((rh, rl, err): (f64, f64, f64)) -> Option<f64> {
            let (up, down) = (rh + (rl + err), rh + (rl - err));
            (up == down).then_some(up)
        }

        pub fn sin(x: f64) -> f64 {
            let ax = x.abs();
            if ax < 1.4901161193847656e-8 { return x; } // below 2^-26 the sine is the angle, to the last place
            // out of the quick way's reach: exactly up to 2^52, as fdlibm has it beyond (and for what is not a number)
            if ax.is_nan() || ax >= NEAR { return if ax < 4503599627370496.0 { sin_slowly(x) } else { super::fdlibm_sin(x) }; }
            let (k, yh, yl) = reduce(x);
            settled(pair(k, yh, yl)).unwrap_or_else(|| sin_slowly(x))
        }

        pub fn cos(x: f64) -> f64 {
            let ax = x.abs();
            if ax < 7.450580596923828e-9 { return 1.0; } // below 2^-27 the cosine is 1, to the last place
            if ax.is_nan() || ax >= NEAR { return if ax < 4503599627370496.0 { cos_slowly(x) } else { super::fdlibm_cos(x) }; }
            // the cosine is the sine a quarter turn on: 64 steps
            let (k, yh, yl) = reduce(x);
            settled(pair(k + 64, yh, yl)).unwrap_or_else(|| cos_slowly(x))
        }

        // ------------------------------------------------------------ the exponential, the quick way
        /// 64/ln 2
        const LOG_STEPS: f64 = f64::from_bits(0x40571547652b82fe);
        /// ln2/64 in three pieces: two short enough that a whole number of them (below 2^16) is exact, then the rest.
        const L1: f64 = f64::from_bits(0x3f862e42fefa0000);
        const L2: f64 = f64::from_bits(0x3d1cf79abc900000);
        const L3: f64 = f64::from_bits(0x3b0c7673007e5ed6);

        /// 2^(j/64) for j from 0 to 63, each as the nearest 64-bit number and then the nearest to what
        /// that left over. Worked out the slow way, once.
        fn powers() -> &'static [[f64; 2]; 64] {
            static TABLE: OnceLock<[[f64; 2]; 64]> = OnceLock::new();
            TABLE.get_or_init(|| std::array::from_fn(|j| if j == 0 { [1.0, 0.0] } else { in_two(&(false, exp_parts(&(false, shr(&times(&LN2, j as u64), 6))).0)) }))
        }

        /// e^x as a pair with how far the pair may be from the truth, and the power of two it is all to be
        /// multiplied by. For x within 700 of zero: x = n (ln2/64) + r, and e^x = 2^(n/64) e^r, the first
        /// looked up, the second a short series.
        #[inline(always)]
        fn exp_pair(x: f64) -> ((f64, f64, f64), i64) {
            let n = (x * LOG_STEPS + WHOLE) - WHOLE;
            // n L1 is exact and so close to x that the difference is exact too
            let a = x - n * L1;
            let (s, e1) = two_sum(a, -(n * L2));
            let (rh, rl) = two_sum(s, e1 - n * L3);
            let n = n as i64;
            let [th, tl] = powers()[(n & 63) as usize];
            // e^r = 1 + rh + half + low:  half of rh^2 exactly, then the rest of r and r^3/6 + r^4/24 ...
            let (q, ql) = two_prod(rh, rh);
            let half = 0.5 * q;
            let low = rl + (0.5 * ql + rh * rl) + rh * q * (1.0 / 6.0 + rh * (1.0 / 24.0 + rh * (1.0 / 120.0 + rh * (1.0 / 720.0 + rh * (1.0 / 5040.0 + rh * (1.0 / 40320.0))))));
            // T e^r for T = 2^(j/64): the three large terms exactly, the small ones as they come
            let (p, pl) = two_prod(th, rh);
            let (m, ml) = two_prod(th, half);
            let (a, ae) = two_sum(th, p);
            let (b, be) = two_sum(a, m);
            let small = ((ae + be) + (pl + ml)) + (tl + tl * (rh + half)) + th * low;
            let (hi, lo) = two_sum(b, small);
            // What can be wrong: the series from r^3 on (about 2^-53 of r^3), the roundings of the small
            // sums, the last bits of ln 2. Each is allowed for several times over.
            let err = 9e-16 * (rh.abs() * q) * th + 1.6e-30 * b + 3.1e-33;
            ((hi, lo, err), n >> 6)
        }

        pub fn exp(x: f64) -> f64 {
            if x.is_nan() { return x; }
            let ax = x.abs();
            if ax < 2.7755575615628914e-17 { return 1.0; } // below 2^-55, e^x is 1 to the last place
            // far out, where the answer may be too large or too small to keep its 53 bits: the slow way
            if ax >= 700.0 { return super::whole::exp(x); }
            let (pair, k) = exp_pair(x);
            match settled(pair) {
                Some(v) => v * f64::from_bits(((k + 1023) as u64) << 52),
                None => super::whole::exp(x),
            }
        }

        // ------------------------------------------------------------ the slow way, and the table it fills
        pub fn sin_slowly(x: f64) -> f64 {
            if x == 0.0 { return x; } // keeping the sign of a zero
            let ((neg, s), _) = sin_cos(&super::whole::big(x.abs()));
            if neg != (x < 0.0) { -nearest(&s) } else { nearest(&s) }
        }
        pub fn cos_slowly(x: f64) -> f64 {
            let (_, (neg, c)) = sin_cos(&super::whole::big(x.abs()));
            if neg { -nearest(&c) } else { nearest(&c) }
        }

        /// sin(k pi/128) for k from 0 to 255, each as the nearest 64-bit number and then the nearest to
        /// what that left over. Worked out the slow way, once, the first time a sine is asked for.
        fn table() -> &'static [[f64; 2]; 256] {
            static TABLE: OnceLock<[[f64; 2]; 256]> = OnceLock::new();
            TABLE.get_or_init(|| {
                std::array::from_fn(|k| {
                    // the quarter turns are exact
                    if k % 64 == 0 { return [[0.0, 1.0, 0.0, -1.0][k / 64], 0.0]; }
                    in_two(&sin_cos(&shr(&times(&PI, k as u64), 7)).0)
                })
            })
        }

        #[cfg(test)]
        mod tests {
            use super::*;

            /// How far the quick way's pair for sin x (or cos x) was from the truth, as a fraction of the
            /// error it allowed itself (anything up to 1 is as it should be), and whether it settled.
            fn strain(x: f64, cosine: bool) -> (f64, bool) {
                let (k, yh, yl) = reduce(x);
                let (rh, rl, err) = pair(if cosine { k + 64 } else { k }, yh, yl);
                let (sine, cos) = sin_cos(&big(x.abs()));
                let truth = if cosine { cos } else { (sine.0 != (x < 0.0), sine.1) };
                // truth - rh - rl, in whole numbers
                let left = [-rh, -rl].into_iter().fold(truth, |sum, part| plus(sum, (part < 0.0, big(part.abs()))));
                (nearest(&left.1) / err, settled((rh, rl, err)).is_some())
            }

            /// Angles of every kind the quick way takes: spread over a few turns, over thousands of turns,
            /// up against its limit, small, and sitting on (or one step off) whole steps of pi/128, where the
            /// sine or the cosine is close to nothing or to a value in the table.
            fn angles(n: usize) -> impl Iterator<Item = f64> {
                let mut s = 0x9E3779B97F4A7C15u64;
                let mut unit = move || {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (s >> 11) as f64 / (1u64 << 53) as f64
                };
                (0..n).map(move |i| {
                    let sign = if unit() < 0.5 { -1.0 } else { 1.0 };
                    sign * match i % 8 {
                        0 | 1 => unit() * 8.0,
                        2 => unit() * 1000.0,
                        3 => unit() * 65535.9,
                        4 => 65535.0 + unit(),
                        5 => libm::exp(-18.0 * unit()),
                        6 => (unit() * 3000.0).floor() * std::f64::consts::PI / 128.0,
                        _ => f64::from_bits(((unit() * 800.0).floor() * std::f64::consts::PI / 128.0 + 1.0).to_bits() + (unit() * 5.0) as u64) - 1.0,
                    }
                })
            }

            #[test]
            fn the_quick_way_gives_what_the_slow_way_gives() {
                for x in angles(60_000) {
                    assert_eq!(sin(x).to_bits(), sin_slowly(x).to_bits(), "sin {x:e}");
                    assert_eq!(cos(x).to_bits(), cos_slowly(x).to_bits(), "cos {x:e}");
                }
                // the ends of the range, and beyond the quick way's reach
                for x in [0.0, -0.0, 5e-324, 1e-300, 1.4e-8, 1.5e-8, 7.4e-9, 7.5e-9, 65535.999, 65536.0, 1e9, -3e12, 4e15] {
                    assert_eq!(sin(x).to_bits(), if x.abs() < 1e-8 { x.to_bits() } else { sin_slowly(x).to_bits() }, "sin {x:e}");
                    assert_eq!(cos(x).to_bits(), if x.abs() < 7e-9 { 1f64.to_bits() } else { cos_slowly(x).to_bits() }, "cos {x:e}");
                }
                // just under the sizes below which the sine is taken to be the angle and the cosine to be 1
                for i in 0..2000 {
                    let under = 1.0 - (i as f64 + 0.5) / 4000.0;
                    let (x, y) = (1.4901161193847656e-8 * under, 7.450580596923828e-9 * under);
                    assert_eq!(sin(x).to_bits(), sin_slowly(x).to_bits(), "sin {x:e}");
                    assert_eq!(cos(y).to_bits(), cos_slowly(y).to_bits(), "cos {y:e}");
                }
                assert!(sin(f64::INFINITY).is_nan() && cos(f64::NEG_INFINITY).is_nan() && sin(f64::NAN).is_nan());
            }

            /// The bound the quick way puts on its own error is what makes its answers certain: the true
            /// value must never be further from its pair than that. Measured here against the truth.
            #[test]
            fn the_quick_way_is_never_further_out_than_it_allows_for() {
                let (mut worst, mut at, mut unsettled, mut n) = (0.0f64, 0.0, 0, 0);
                for x in angles(40_000).filter(|x| x.abs() >= 1.5e-8) {
                    for cosine in [false, true] {
                        let (strain, settled) = strain(x, cosine);
                        if strain > worst { (worst, at) = (strain, x); }
                        unsettled += !settled as usize;
                        n += 1;
                    }
                }
                println!("worst: {worst:.3} of the error allowed (at {at:e}); {unsettled} of {n} had to be done the slow way");
                assert!(worst < 0.7, "the pair for {at:e} was {worst} of its allowed error from the truth");
                assert!(unsettled * 200 < n, "the slow way was needed {unsettled} times in {n}");
            }

            /// The same two checks over tens of millions of angles: minutes, so only when asked for.
            ///     cargo test --release -p gf-core rounded -- --ignored --nocapture
            #[test]
            #[ignore]
            fn the_quick_way_over_many_more_angles() {
                let (mut worst, mut at, mut unsettled, mut n) = (0.0f64, 0.0, 0u64, 0u64);
                for (i, x) in angles(20_000_000).enumerate() {
                    assert_eq!(sin(x).to_bits(), sin_slowly(x).to_bits(), "sin {x:e}");
                    assert_eq!(cos(x).to_bits(), cos_slowly(x).to_bits(), "cos {x:e}");
                    if i % 4 == 0 && x.abs() >= 1.5e-8 {
                        for cosine in [false, true] {
                            let (strain, settled) = strain(x, cosine);
                            if strain > worst { (worst, at) = (strain, x); }
                            unsettled += !settled as u64;
                            n += 1;
                        }
                    }
                }
                println!("worst: {worst:.3} of the error allowed (at {at:e}); {unsettled} of {n} had to be done the slow way");
                assert!(worst < 0.7);
            }

            /// The same for the exponential: how far its pair was from the truth as a fraction of the error it
            /// allowed itself, and whether it settled.
            fn exp_strain(x: f64) -> (f64, bool) {
                let ((hi, lo, err), k) = exp_pair(x);
                // the truth as m 2^to; brought to the pair's own power of two (one doubling or halving at most)
                let (m, to) = exp_parts(&(x < 0.0, big(x.abs())));
                let truth = match to as i64 - k {
                    0 => m,
                    1 => add(&m, &m),
                    -1 => shr(&m, 1),
                    by => panic!("e^{x}: powers of two {by} apart"),
                };
                let left = [-hi, -lo].into_iter().fold((false, truth), |sum, part| plus(sum, (part < 0.0, big(part.abs()))));
                (nearest(&left.1) / err, settled((hi, lo, err)).is_some())
            }

            /// Exponents of every kind the quick way takes: small, middling, out to its limit, and sitting on
            /// (or one step off) whole steps of ln2/64.
            fn exponents(n: usize) -> impl Iterator<Item = f64> {
                let mut s = 0xD1B54A32D192ED03u64;
                let mut unit = move || {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (s >> 11) as f64 / (1u64 << 53) as f64
                };
                (0..n).map(move |i| {
                    let sign = if unit() < 0.5 { -1.0 } else { 1.0 };
                    sign * match i % 6 {
                        0 | 1 => unit() * 20.0,
                        2 => unit() * 699.9,
                        3 => libm::exp(-36.0 * unit()),
                        4 => (unit() * 60000.0).floor() * std::f64::consts::LN_2 / 64.0,
                        _ => f64::from_bits(((unit() * 6000.0).floor() * std::f64::consts::LN_2 / 64.0 + 1.0).to_bits() + (unit() * 5.0) as u64) - 1.0,
                    }
                })
            }

            #[test]
            fn the_quick_exponential_gives_what_the_slow_one_gives() {
                for x in exponents(60_000) {
                    assert_eq!(exp(x).to_bits(), super::super::whole::exp(x).to_bits(), "exp {x:e}");
                }
                // the ends: where 1 is the answer, where the slow way takes over, and where the numbers run out
                for x in [0.0, -0.0, 5e-324, -1e-300, 2.7e-17, -2.8e-17, 1.1e-16, -1.2e-16, 1.0, -1.0, 699.999, 700.0, -700.5, 709.782712893384, 709.7827128933841, -708.4, -745.13, -745.14, -746.0, 1e300, -1e300] {
                    assert_eq!(exp(x).to_bits(), super::super::whole::exp(x).to_bits(), "exp {x:e}");
                }
                assert_eq!(exp(0.0), 1.0);
                assert_eq!(exp(1.0), std::f64::consts::E);
                assert_eq!(exp(709.782712893384), 1.7976931348622732e308);
                assert_eq!(exp(709.7827128933841), f64::INFINITY);
                assert_eq!(exp(-745.13), 5e-324);
                assert_eq!(exp(-745.14), 0.0);
                assert!(exp(f64::NAN).is_nan() && exp(f64::INFINITY) == f64::INFINITY && exp(f64::NEG_INFINITY) == 0.0);
            }

            #[test]
            fn the_quick_exponential_is_never_further_out_than_it_allows_for() {
                let (mut worst, mut at, mut unsettled, mut n) = (0.0f64, 0.0, 0, 0);
                for x in exponents(60_000).filter(|x| x.abs() >= 2.8e-17) {
                    let (strain, settled) = exp_strain(x);
                    if strain > worst { (worst, at) = (strain, x); }
                    unsettled += !settled as usize;
                    n += 1;
                }
                println!("exp worst: {worst:.3} of the error allowed (at {at:e}); {unsettled} of {n} had to be done the slow way");
                assert!(worst < 0.7, "the pair for e^{at:e} was {worst} of its allowed error from the truth");
                assert!(unsettled * 200 < n, "the slow way was needed {unsettled} times in {n}");
            }

            /// The same over millions of exponents: a minute or so, only when asked for.
            #[test]
            #[ignore]
            fn the_quick_exponential_over_many_more() {
                let (mut worst, mut at, mut unsettled, mut n) = (0.0f64, 0.0, 0u64, 0u64);
                for x in exponents(12_000_000) {
                    assert_eq!(exp(x).to_bits(), super::super::whole::exp(x).to_bits(), "exp {x:e}");
                    if x.abs() >= 2.8e-17 {
                        let (strain, settled) = exp_strain(x);
                        if strain > worst { (worst, at) = (strain, x); }
                        unsettled += !settled as u64;
                        n += 1;
                    }
                }
                println!("exp worst: {worst:.3} of the error allowed (at {at:e}); {unsettled} of {n} had to be done the slow way");
                assert!(worst < 0.7);
            }
        }
    }

    /// Arithmetic in whole numbers, for the answers that have to be certain.
    ///
    /// A number here has 256 binary places and up to 64 bits before the point, in five 64-bit pieces.
    /// Sines, exponentials, logarithms and arctangents are summed as plain series to about 240 places,
    /// far more than it takes to tell which 64-bit number is the nearest. That costs microseconds, so it
    /// is the fallback of the quick ways in `rounded`, and the whole of the functions that a shape calls
    /// only a few times.
    mod whole {
        use std::sync::OnceLock;

        /// A number below 2^64 with 256 binary places, in five 64-bit pieces from the least significant.
        pub type Big = [u64; 5];
        /// The same with a sign: (negative, size).
        pub type Signed = (bool, Big);
        pub const PI: Big = [0x082efa98ec4e6c89, 0xa4093822299f31d0, 0x13198a2e03707344, 0x243f6a8885a308d3, 3];
        pub const LN2: Big = [0x8a0d175b8baafa2b, 0x40f343267298b62d, 0xc9e3b39803f2f6af, 0xb17217f7d1cf79ab, 0];
        pub const ONE: Big = [0, 0, 0, 0, 1];
        const NOTHING: Big = [0; 5];

        pub fn less(a: &Big, b: &Big) -> bool {
            for i in (0..5).rev() {
                if a[i] != b[i] { return a[i] < b[i]; }
            }
            false
        }
        pub fn add(a: &Big, b: &Big) -> Big {
            let (mut out, mut carry) = ([0; 5], false);
            for i in 0..5 {
                let (s, c1) = a[i].overflowing_add(b[i]);
                let (s, c2) = s.overflowing_add(carry as u64);
                (out[i], carry) = (s, c1 || c2);
            }
            out
        }
        /// a - b, for a not less than b.
        pub fn sub(a: &Big, b: &Big) -> Big {
            let (mut out, mut borrow) = ([0; 5], false);
            for i in 0..5 {
                let (d, b1) = a[i].overflowing_sub(b[i]);
                let (d, b2) = d.overflowing_sub(borrow as u64);
                (out[i], borrow) = (d, b1 || b2);
            }
            out
        }
        /// Two numbers with signs, added.
        pub fn plus(a: Signed, b: Signed) -> Signed {
            if a.0 == b.0 { (a.0, add(&a.1, &b.1)) } else if less(&a.1, &b.1) { (b.0, sub(&b.1, &a.1)) } else { (a.0, sub(&a.1, &b.1)) }
        }
        /// a shifted down by n bits, whatever n.
        pub fn shr(a: &Big, n: u32) -> Big {
            let (words, bits) = ((n / 64) as usize, n % 64);
            std::array::from_fn(|i| {
                let at = |j: usize| if j < 5 { a[j] } else { 0 };
                (at(i + words) >> bits) | if bits > 0 { at(i + words + 1) << (64 - bits) } else { 0 }
            })
        }
        /// a doubled.
        fn twice(a: &Big) -> Big {
            add(a, a)
        }
        pub fn times(a: &Big, q: u64) -> Big {
            let (mut out, mut carry) = ([0; 5], 0u128);
            for i in 0..5 {
                let t = a[i] as u128 * q as u128 + carry;
                (out[i], carry) = (t as u64, t >> 64);
            }
            out
        }
        pub fn over(a: &Big, d: u64) -> Big {
            let (mut out, mut rest) = ([0; 5], 0u128);
            for i in (0..5).rev() {
                let t = (rest << 64) | a[i] as u128;
                (out[i], rest) = ((t / d as u128) as u64, t % d as u128);
            }
            out
        }
        /// a b, cut off (not rounded) at the 256th place.
        pub fn mul(a: &Big, b: &Big) -> Big {
            let mut p = [0u64; 10];
            for i in 0..5 {
                if a[i] == 0 { continue; }
                let mut carry = 0u128;
                for j in 0..5 {
                    let t = a[i] as u128 * b[j] as u128 + p[i + j] as u128 + carry;
                    (p[i + j], carry) = (t as u64, t >> 64);
                }
                p[i + 5] = carry as u64;
            }
            [p[4], p[5], p[6], p[7], p[8]]
        }
        /// 1/a, for a from about a half up: a first guess in 53 bits, doubled in precision three times over.
        fn recip(a: &Big) -> Big {
            let mut y = big(1.0 / nearest(a));
            for _ in 0..3 {
                y = mul(&y, &sub(&twice(&ONE), &mul(a, &y)));
            }
            y
        }

        /// A 64-bit number's size (it must be below 2^64) as it is, exactly; anything below the 256th place is dropped.
        pub fn big(x: f64) -> Big {
            let bits = x.to_bits();
            let (e, frac) = (((bits >> 52) & 0x7ff) as i32, bits & ((1 << 52) - 1));
            if e == 0 && frac == 0 { return NOTHING; }
            // |x| = m 2^(e - 1075)
            let (m, place) = (if e == 0 { frac } else { frac | 1 << 52 }, e.max(1) - 1075 + 256);
            assert!(place <= 267, "{x} does not fit");
            if place < 0 { return [if place > -64 { m >> -place } else { 0 }, 0, 0, 0, 0]; }
            let (word, bit) = (place as usize / 64, place as u32 % 64);
            let wide = (m as u128) << bit;
            let mut out = NOTHING;
            out[word] = wide as u64;
            if word < 4 { out[word + 1] = (wide >> 64) as u64; }
            out
        }
        /// m 2^e for a whole number m (below 2^54) and any e: exact whenever the answer can be held.
        fn scaled(m: u64, e: i32) -> f64 {
            let two = |e: i32| f64::from_bits(((e + 1023) as u64) << 52);
            // in two steps, so that neither factor is out of range on its own
            let first = e.clamp(-1000, 1000);
            m as f64 * two(first) * two((e - first).clamp(-1000, 1000))
        }
        /// The nearest 64-bit number to a 2^by: 53 bits are kept, fewer once the answer is too small to
        /// have that many, and what is dropped is rounded (an exact half goes to the even neighbour).
        pub fn nearest_scaled(a: &Big, by: i32) -> f64 {
            let Some(top) = (0..5).rev().find(|&i| a[i] != 0) else { return 0.0 };
            // the leading 1 is bit `lead`, worth 2^(lead - 256 + by); bits below `drop` are not kept
            let lead = (top as u32 * 64 + 63 - a[top].leading_zeros()) as i32;
            let drop = (lead - 52).max(-1074 - by + 256);
            if drop <= 0 { return scaled(a[0], by - 256); }
            if drop > lead + 1 { return 0.0; } // less than half the smallest number there is
            let bit = |n: i32| if n < 320 { a[n as usize / 64] >> (n % 64) & 1 } else { 0 };
            let kept = (0..53).fold(0u64, |m, i| m | bit(drop + i) << i);
            let below = (0..drop - 1).any(|n| bit(n) == 1);
            let up = bit(drop - 1) == 1 && (below || kept & 1 == 1);
            scaled(kept + up as u64, drop - 256 + by)
        }
        pub fn nearest(a: &Big) -> f64 {
            nearest_scaled(a, 0)
        }
        /// The nearest 64-bit number to a signed one.
        pub fn nearest_signed(a: &Signed) -> f64 {
            if a.0 { -nearest(&a.1) } else { nearest(&a.1) }
        }
        /// A value as the nearest 64-bit number and then the nearest to what that leaves over.
        pub fn in_two(a: &Signed) -> [f64; 2] {
            let hi = nearest(&a.1);
            let lo = nearest_signed(&plus((false, a.1), (true, big(hi))));
            if a.0 { [-hi, -lo] } else { [hi, lo] }
        }

        // ------------------------------------------------------------ sine and cosine
        /// The sine and cosine of a number given exactly (not negative).
        pub fn sin_cos(x: &Big) -> (Signed, Signed) {
            let (half_pi, quarter_pi) = (shr(&PI, 1), shr(&PI, 2));
            // x = q (pi/2) + r with r within pi/4 of zero: q guessed, then put right
            let mut q = (nearest(x) * std::f64::consts::FRAC_2_PI + 0.5) as u64;
            let (neg, r) = loop {
                let turns = times(&half_pi, q);
                let (neg, r) = if less(x, &turns) { (true, sub(&turns, x)) } else { (false, sub(x, &turns)) };
                if !less(&quarter_pi, &r) { break (neg, r); }
                if neg { q -= 1 } else { q += 1 }
            };
            // the two series, the terms that add and the terms that take away summed apart
            let r2 = mul(&r, &r);
            let series = |first: Big, mut n: u64| {
                let (mut gains, mut losses, mut term, mut losing) = (first, NOTHING, first, true);
                loop {
                    term = over(&mul(&term, &r2), n * (n + 1));
                    if term == NOTHING { break sub(&gains, &losses); }
                    if losing { losses = add(&losses, &term) } else { gains = add(&gains, &term) }
                    (losing, n) = (!losing, n + 2);
                }
            };
            let (s, c) = (series(r, 2), series(ONE, 1));
            match q & 3 {
                0 => ((neg, s), (false, c)),
                1 => ((false, c), (!neg, s)),
                2 => ((!neg, s), (true, c)),
                _ => ((true, c), (neg, s)),
            }
        }

        // ------------------------------------------------------------ the exponential
        /// e^x for a number given exactly, as m 2^k with m from 1 to 2. For x within a few thousand of zero.
        pub fn exp_parts(x: &Signed) -> (Big, i32) {
            // x = k ln 2 + r with r from 0 up to ln 2: k guessed, then put right
            let mut k = (nearest_signed(x) / std::f64::consts::LN_2).floor() as i64;
            let r = loop {
                let r = plus(*x, (k > 0, times(&LN2, k.unsigned_abs())));
                if r.0 && r.1 != NOTHING {
                    k -= 1
                } else if !less(&r.1, &LN2) {
                    k += 1
                } else {
                    break r.1;
                }
            };
            // e^r is the sixteenth power of e^(r/16), whose series is short
            let small = shr(&r, 4);
            let (mut sum, mut term, mut n) = (add(&ONE, &small), small, 2);
            loop {
                term = over(&mul(&term, &small), n);
                if term == NOTHING { break; }
                (sum, n) = (add(&sum, &term), n + 1);
            }
            for _ in 0..4 {
                sum = mul(&sum, &sum);
            }
            (sum, k as i32)
        }
        pub fn exp(x: f64) -> f64 {
            if x.is_nan() { return x; }
            // beyond these the answer is past the largest number, or less than half the smallest
            if x > 709.782712893384 { return f64::INFINITY; }
            if x < -745.2 { return 0.0; }
            if x.abs() < 8.673617379884035e-19 { return 1.0; } // below 2^-60, e^x is 1 to the last place
            let (m, k) = exp_parts(&(x < 0.0, big(x.abs())));
            nearest_scaled(&m, k)
        }

        // ------------------------------------------------------------ the logarithm
        /// ln x for x above nothing and below infinity.
        pub fn ln(x: f64) -> Signed {
            // x = m 2^e with m between the square roots of a half and of two
            let bits = x.to_bits();
            let (mut e, mut frac) = (((bits >> 52) & 0x7ff) as i32 - 1023, bits & ((1 << 52) - 1));
            if e == -1023 {
                // too small to have its leading 1 kept for it: bring it up
                let up = frac.leading_zeros() - 11;
                (e, frac) = (-1022 - up as i32, (frac << up) & ((1 << 52) - 1));
            }
            let mut m = f64::from_bits(frac | 1023 << 52);
            if m > std::f64::consts::SQRT_2 { (m, e) = (m / 2.0, e + 1); }
            // ln m = 2 (z + z^3/3 + z^5/5 ...) with z = (m - 1)/(m + 1), no more than 0.172 either way
            let mb = big(m);
            let (neg, top) = if m < 1.0 { (true, sub(&ONE, &mb)) } else { (false, sub(&mb, &ONE)) };
            let z = mul(&top, &recip(&add(&mb, &ONE)));
            let z2 = mul(&z, &z);
            let (mut sum, mut power, mut n) = (z, z, 3);
            loop {
                power = mul(&power, &z2);
                let term = over(&power, n);
                if term == NOTHING { break; }
                (sum, n) = (add(&sum, &term), n + 2);
            }
            plus((neg, twice(&sum)), (e < 0, times(&LN2, e.unsigned_abs() as u64)))
        }
        pub fn log(x: f64) -> f64 {
            if x.is_nan() || x < 0.0 { return f64::NAN; }
            if x == 0.0 { return f64::NEG_INFINITY; }
            if x == f64::INFINITY { return x; }
            if x == 1.0 { return 0.0; }
            nearest_signed(&ln(x))
        }

        // ------------------------------------------------------------ the arctangent
        /// u - u^3/3 + u^5/5 ..., for u below 1.
        fn atan_series(u: &Big) -> Big {
            let u2 = mul(u, u);
            let (mut gains, mut losses, mut power, mut n, mut losing) = (*u, NOTHING, *u, 3, true);
            loop {
                power = mul(&power, &u2);
                let term = over(&power, n);
                if term == NOTHING { break sub(&gains, &losses); }
                if losing { losses = add(&losses, &term) } else { gains = add(&gains, &term) }
                (losing, n) = (!losing, n + 2);
            }
        }
        /// k/8 for k from 0 to 8.
        fn eighth(k: usize) -> Big {
            [0, 0, 0, (k as u64) << 61, k as u64 >> 3]
        }
        /// atan(k/8) for k from 0 to 8, worked out once.
        fn eighths() -> &'static [Big; 9] {
            static TABLE: OnceLock<[Big; 9]> = OnceLock::new();
            TABLE.get_or_init(|| std::array::from_fn(|k| if k == 8 { shr(&PI, 2) } else { atan_series(&eighth(k)) }))
        }
        /// atan q for q from 0 to 1: from the nearest eighth c, atan q = atan c + atan((q - c)/(1 + q c)),
        /// and that last angle is small enough for its series to be short.
        fn atan(q: &Big) -> Big {
            let k = (nearest(q) * 8.0 + 0.5) as usize;
            let c = eighth(k);
            let (under, gap) = if less(q, &c) { (true, sub(&c, q)) } else { (false, sub(q, &c)) };
            let rest = atan_series(&mul(&gap, &recip(&add(&ONE, &mul(q, &c)))));
            if under { sub(&eighths()[k], &rest) } else { add(&eighths()[k], &rest) }
        }
        /// A positive number as m 2^e with m a whole number of exactly 53 bits.
        fn mantissa(x: f64) -> (u64, i32) {
            let bits = x.to_bits();
            let (e, frac) = (((bits >> 52) & 0x7ff) as i32, bits & ((1 << 52) - 1));
            if e > 0 { return (frac | 1 << 52, e - 1075); }
            let up = frac.leading_zeros() - 11;
            (frac << up, -1074 - up as i32)
        }
        /// The angle of the point (x, y) from the x axis, as `Math.atan2(y, x)` has it.
        pub fn atan2(y: f64, x: f64) -> f64 {
            if x.is_nan() || y.is_nan() { return f64::NAN; }
            // a point endlessly far off lies along an axis or a diagonal
            let (y, x) = match (y.is_infinite(), x.is_infinite()) {
                (true, true) => (1f64.copysign(y), 1f64.copysign(x)),
                (true, false) => (1f64.copysign(y), 0.0),
                (false, true) => (0f64.copysign(y), 1f64.copysign(x)),
                _ => (y, x),
            };
            let (ay, ax, left) = (y.abs(), x.abs(), x.is_sign_negative());
            let signed = |a: f64| a.copysign(y);
            if ay == 0.0 { return if left { signed(nearest(&PI)) } else { y }; }
            if ax == 0.0 { return signed(nearest(&shr(&PI, 1))); }
            // the angle from the nearer axis has the shorter side over the longer for its tangent
            let (short, long, steep) = if ay > ax { (ax, ay, true) } else { (ay, ax, false) };
            if !steep && !left && short < long * 7.888609052210118e-31 {
                // So shallow (a slope below 2^-100) that the angle is the slope less a part in 2^200 of it:
                // the nearest number to that is the nearest to the slope, which is what dividing gives.
                // (Not so if the answer is too small to keep its 53 bits; then fdlibm's will do.)
                let slope = y / x;
                return if slope.abs() >= f64::MIN_POSITIVE { slope } else { libm::atan2(y, x) };
            }
            let ((ms, es), (ml, el)) = (mantissa(short), mantissa(long));
            let slope = shr(&over(&[0, 0, 0, 0, ms], ml), (el - es) as u32);
            let mut angle = atan(&slope);
            if steep { angle = sub(&shr(&PI, 1), &angle); }
            if left { angle = sub(&PI, &angle); }
            signed(nearest(&angle))
        }

        // ------------------------------------------------------------ powers
        /// x^y as `Math.pow` has it: first the cases the language spells out, then the nearest 64-bit
        /// number to the true power.
        pub fn pow(x: f64, y: f64) -> f64 {
            if y == 0.0 { return 1.0; }
            if x.is_nan() || y.is_nan() { return f64::NAN; }
            if y.is_infinite() {
                let size = x.abs();
                return if size == 1.0 { f64::NAN } else if (size > 1.0) == (y > 0.0) { f64::INFINITY } else { 0.0 };
            }
            let whole = y == y.trunc();
            let odd = whole && y.abs() < 9007199254740992.0 && y.abs() % 2.0 == 1.0;
            if x.is_infinite() || x == 0.0 {
                // an endless or an empty base gives everything or nothing, and keeps its sign only through an odd power
                let all = if x.is_infinite() == (y > 0.0) { f64::INFINITY } else { 0.0 };
                return if odd && x.is_sign_negative() { -all } else { all };
            }
            if x < 0.0 && !whole { return f64::NAN; }
            let size = power(x.abs(), y);
            if x < 0.0 && odd { -size } else { size }
        }
        /// x^y for x above nothing and y not nothing, both finite.
        fn power(x: f64, y: f64) -> f64 {
            if x == 1.0 { return 1.0; }
            if let Some(exactly) = dyadic_power(x, y) { return exactly; }
            // roughly y ln x: far enough out and the answer is past the largest number, or less than half
            // the smallest; close enough to nothing and it is 1 to the last place
            let rough = y * super::fdlibm_log(x);
            if rough > 710.0 { return f64::INFINITY; }
            if rough < -746.0 { return 0.0; }
            if rough.abs() < 8.673617379884035e-19 { return 1.0; }
            let l = ln(x);
            let (m, k) = exp_parts(&(l.0 != (y < 0.0), mul(&l.1, &big(y.abs()))));
            nearest_scaled(&m, k)
        }
        /// The powers that come out as a whole number times a power of two, where the series would only
        /// come close and an exact half could go either way: a power of two to a power that makes its
        /// exponent whole, or a whole power (or one in halves, quarters ... of a base that has those
        /// roots) of a base with few enough bits. Done exactly. None: not one of these.
        fn dyadic_power(x: f64, y: f64) -> Option<f64> {
            // x = m 2^e with m odd
            let (m, e) = mantissa(x);
            let (m, e) = (m >> m.trailing_zeros(), e + m.trailing_zeros() as i32);
            if m == 1 {
                // 2^(e y): whole when y's last binary place is no finer than e's factors of two allow
                if e == 0 { return Some(1.0); }
                let (my, ey) = mantissa(y.abs());
                let places = -(ey + my.trailing_zeros() as i32);
                if places > e.trailing_zeros() as i32 { return None; }
                let to = e as f64 * y; // a whole number, so exact (or far too large to matter)
                return Some(if to > 1023.0 { f64::INFINITY } else if to < -1074.0 { 0.0 } else { scaled(1, to as i32) });
            }
            // y = n / 2^j with j at most 5 (a base with more roots than that would not fit in 53 bits)
            let in_32nds = y * 32.0;
            if !(in_32nds > 0.0 && in_32nds < 1e12 && in_32nds == in_32nds.trunc()) { return None; }
            let in_32nds = in_32nds as u64;
            let j = 5 - in_32nds.trailing_zeros().min(5);
            let n = in_32nds >> (5 - j);
            if e.rem_euclid(1 << j) != 0 { return None; }
            // the 2^j-th root of m, if it has one
            let mut root = m;
            for _ in 0..j {
                let r = (root as f64).sqrt().round() as u64;
                if r * r != root { return None; }
                root = r;
            }
            // root^n exactly, if it stays below 2^120
            let mut p = 1u128;
            for _ in 0..n {
                p = p.checked_mul(root as u128).filter(|&p| p < 1 << 120)?;
            }
            let to = (e as i64 >> j) * n as i64;
            Some(nearest_scaled(&[p as u64, (p >> 64) as u64, 0, 0, 0], to.clamp(-5000, 5000) as i32 + 256))
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI as PI_NEAREST};
            const INF: f64 = f64::INFINITY;

            fn same(a: f64, b: f64) -> bool {
                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
            }

            #[test]
            fn the_constants_are_what_they_say() {
                // pi and ln 2 to 64-bit precision, and through the functions that lean on them
                assert_eq!(nearest(&PI), PI_NEAREST);
                assert_eq!(nearest(&LN2), std::f64::consts::LN_2);
                assert_eq!(nearest_scaled(&exp_parts(&(false, LN2)).0, exp_parts(&(false, LN2)).1), 2.0);
                assert_eq!(log(2.0), std::f64::consts::LN_2);
                assert_eq!(atan2(1.0, 1.0), FRAC_PI_4);
                // a reciprocal times what it is the reciprocal of is 1, to within a few of the last places
                let three = [0, 0, 0, 0, 3];
                let product = mul(&recip(&three), &three);
                let gap = if less(&product, &ONE) { sub(&ONE, &product) } else { sub(&product, &ONE) };
                assert!(gap[1..] == [0; 4] && gap[0] < 1 << 16, "{gap:x?}");
            }

            #[test]
            fn rounding_at_the_edges() {
                // a whole number of 54 bits ending in 1 is an exact half: it goes to the even neighbour
                assert_eq!(nearest_scaled(&[(1 << 53) + 1, 0, 0, 0, 0], 256), 9007199254740992.0);
                assert_eq!(nearest_scaled(&[(1 << 53) + 3, 0, 0, 0, 0], 256), 9007199254740996.0);
                // the smallest numbers: halves of the smallest step go to the even side, more than half goes up
                assert_eq!(nearest_scaled(&ONE, -1074), 5e-324);
                assert_eq!(nearest_scaled(&ONE, -1075), 0.0);
                assert_eq!(nearest_scaled(&[1, 0, 0, 0, 1], -1075), 5e-324);
                assert_eq!(nearest_scaled(&[0, 0, 0, 0, 3], -1075), 1e-323);
                assert_eq!(nearest_scaled(&ONE, -1080), 0.0);
                // and the largest
                assert_eq!(nearest_scaled(&ONE, 1023), 8.98846567431158e307);
                assert_eq!(nearest_scaled(&ONE, 1024), INF);
            }

            #[test]
            fn logarithms_at_the_edges() {
                assert!(log(-1.0).is_nan() && log(f64::NAN).is_nan());
                assert_eq!(log(0.0), -INF);
                assert_eq!(log(-0.0), -INF);
                assert_eq!(log(INF), INF);
                assert!(same(log(1.0), 0.0));
                assert_eq!(log(5e-324), -744.4400719213812);
                assert_eq!(log(f64::MAX), 709.782712893384);
                assert_eq!(log(std::f64::consts::E), 1.0);
                // either side of 1, where the answer is all but the gap itself
                assert_eq!(log(1.0000000000000002), 2.2204460492503128e-16);
                assert_eq!(log(0.9999999999999999), -1.1102230246251565e-16);
            }

            #[test]
            fn angles_at_the_edges() {
                // the cases the language spells out: zeros keep their signs, the far ends lie on axes and diagonals
                let cases = [
                    (0.0, 1.0, 0.0), (-0.0, 1.0, -0.0), (0.0, 0.0, 0.0), (-0.0, 0.0, -0.0),
                    (0.0, -1.0, PI_NEAREST), (-0.0, -1.0, -PI_NEAREST), (0.0, -0.0, PI_NEAREST), (-0.0, -0.0, -PI_NEAREST),
                    (1.0, 0.0, FRAC_PI_2), (-1.0, 0.0, -FRAC_PI_2), (1.0, -0.0, FRAC_PI_2), (-3.0, -0.0, -FRAC_PI_2),
                    (INF, 5.0, FRAC_PI_2), (-INF, -5.0, -FRAC_PI_2), (INF, INF, FRAC_PI_4), (-INF, INF, -FRAC_PI_4),
                    (INF, -INF, 2.356194490192345), (-INF, -INF, -2.356194490192345),
                    (5.0, INF, 0.0), (-5.0, INF, -0.0), (5.0, -INF, PI_NEAREST), (-5.0, -INF, -PI_NEAREST),
                    (1.0, 1.0, FRAC_PI_4), (1.0, -1.0, 2.356194490192345), (-1.0, -1.0, -2.356194490192345), (-1.0, 1.0, -FRAC_PI_4),
                    // shallow, steep, and both the other way about
                    (1e-200, 1.0, 1e-200), (-1e-200, 1.0, -1e-200), (1.0, 1e-200, FRAC_PI_2), (1e-200, -1.0, PI_NEAREST), (3e-310, 2.0, 3e-310 / 2.0),
                    (1e-20, 1.0, 1e-20), (1e-10, 1.0, 1e-10), (1e-5, 1.0, 9.999999999666668e-6),
                ];
                for (y, x, angle) in cases {
                    assert!(same(atan2(y, x), angle), "atan2({y:e}, {x:e}) = {:e}, not {angle:e}", atan2(y, x));
                }
                assert!(atan2(f64::NAN, 1.0).is_nan() && atan2(1.0, f64::NAN).is_nan());
            }

            #[test]
            fn powers_at_the_edges() {
                let nan = f64::NAN;
                let cases = [
                    // the cases the language spells out
                    (nan, 0.0, 1.0), (nan, -0.0, 1.0), (2.0, nan, nan), (nan, 2.0, nan),
                    (2.0, INF, INF), (0.5, INF, 0.0), (2.0, -INF, 0.0), (0.5, -INF, INF), (1.0, INF, nan), (-1.0, -INF, nan), (-2.0, INF, INF),
                    (INF, 2.0, INF), (INF, -2.0, 0.0), (-INF, 3.0, -INF), (-INF, 2.0, INF), (-INF, -3.0, -0.0), (-INF, -2.0, 0.0), (-INF, 0.5, INF),
                    (0.0, 2.0, 0.0), (0.0, -2.0, INF), (-0.0, 3.0, -0.0), (-0.0, 2.0, 0.0), (-0.0, -3.0, -INF), (-0.0, -2.0, INF),
                    (-8.0, 1.0 / 3.0, nan), (-2.0, 3.0, -8.0), (-2.0, 2.0, 4.0), (-2.0, -1.0, -0.5), (1.0, 1e300, 1.0), (1.0, -5e-324, 1.0),
                    // exact answers
                    (3.0, 2.0, 9.0), (10.0, 2.0, 100.0), (4.0, -1.5, 0.125), (9.0, 0.5, 3.0), (2.0, 10.0, 1024.0), (0.25, 0.5, 0.5), (1.5, 2.0, 2.25),
                    (2.0, -1074.0, 5e-324), (2.0, 1023.0, 8.98846567431158e307), (2.0, 1024.0, INF), (0.5, 1075.0, 0.0), (2.0, -1080.0, 0.0),
                    // exact halves, which go to the even neighbour: 3^34 and (2^27 - 1)^2 have 54 bits, and
                    // (81/2^860)^(5/4) is 243 halves of the smallest number there is
                    (3.0, 34.0, 16677181699666568.0), (134217727.0, 2.0, 18014398241046528.0), (81.0 * 2f64.powi(-860), 1.25, 122.0 * 5e-324),
                    // and ordinary ones
                    (2.0, 0.5, std::f64::consts::SQRT_2), (2.0, -2.131929115066305, 0.22815258183914391), (10.0, -2.0, 0.01), (0.1, 3.0, 0.0010000000000000002),
                    (7.0, -1.0, 0.14285714285714285), (10.0, 308.0, 1e308), (10.0, 309.0, INF), (10.0, -323.0, 1e-323), (10.0, -324.0, 0.0),
                    (1.0000000000000002, 4503599627370496.0, std::f64::consts::E),
                ];
                for (x, y, power) in cases {
                    assert!(same(pow(x, y), power), "pow({x:e}, {y:e}) = {:e}, not {power:e}", pow(x, y));
                }
            }
        }
    }
}

/// Runge-Kutta 4 through a flow. Returns x,y,z triples; stops early if it flies off (forms.js `orbit`).
pub fn orbit(flow: &Flow, v: &[f64], steps: usize, settle: usize) -> Vec<f64> {
    let [mut x, mut y, mut z] = flow.start;
    let (f, h) = (flow.f, flow.dt);
    let mut step = || {
        let a = f(x, y, z, v);
        let b = f(x + h / 2.0 * a[0], y + h / 2.0 * a[1], z + h / 2.0 * a[2], v);
        let c = f(x + h / 2.0 * b[0], y + h / 2.0 * b[1], z + h / 2.0 * b[2], v);
        let d = f(x + h * c[0], y + h * c[1], z + h * c[2], v);
        x += h / 6.0 * (a[0] + 2.0 * b[0] + 2.0 * c[0] + d[0]);
        y += h / 6.0 * (a[1] + 2.0 * b[1] + 2.0 * c[1] + d[1]);
        z += h / 6.0 * (a[2] + 2.0 * b[2] + 2.0 * c[2] + d[2]);
        [x, y, z]
    };
    for _ in 0..settle {
        step();
    }
    let mut out = Vec::with_capacity(steps * 3);
    for _ in 0..steps {
        let p = step();
        // it flew off: keep what we have
        if !(p[0].abs() < 1e6 && p[1].abs() < 1e6 && p[2].abs() < 1e6) { break; }
        out.extend_from_slice(&p);
    }
    out
}

/// The `flow()` helper of scenes/attractors.js: `steps` points of the orbit after it has settled.
/// quick: settle less (a slider is being dragged or animated).
pub fn orbit_line(flow: &Flow, v: &[f64], steps: usize, quick: bool) -> Vec<f64> {
    orbit(flow, v, steps, if quick { flow.settle.min(20000) } else { flow.settle })
}

/// Where a curve sits in the volume: its centre, the scale that brings its farthest point to FIT, which
/// of its axes go where, and (for shapes that ask to be turned face-on) a rotation.
#[derive(Clone, Debug, PartialEq)]
pub struct Pose {
    pub c: [f64; 3],
    pub sc: f64,
    pub ax: [usize; 3],
    pub e: Option<[[f64; 3]; 3]>,
}

/// What `placement` is asked for. `centre`: false = keep the origin where it is. `fit`: the distance to
/// bring the farthest point to (None = FIT). `align`: turn the shape face-on. `e`: a rotation to use
/// instead of working one out.
#[derive(Clone, Debug, Default)]
pub struct PlaceOpts {
    pub centre: bool,
    pub fit: Option<f64>,
    pub axes: Option<[usize; 3]>,
    pub align: bool,
    pub e: Option<[[f64; 3]; 3]>,
}

/// The points of a line, three numbers each (anything left over at the end is not a point).
#[inline]
fn points(line: &[f64]) -> &[[f64; 3]] {
    line.as_chunks().0
}

/// Work out where a curve sits (forms.js `placement`). Kept separate from `prep_curve` so that an
/// animation can hold a placement still while the curve itself changes; refitting every frame would
/// make the shape pulse and swing.
// (the loops over 0..3 are the indices of the mathematics)
#[allow(clippy::needless_range_loop)]
pub fn placement(lines: &[&[f64]], o: &PlaceOpts) -> Pose {
    let (mut lo, mut hi) = ([1e300; 3], [-1e300; 3]);
    for p in lines.iter().copied().flat_map(points) {
        for k in 0..3 {
            if p[k] < lo[k] { lo[k] = p[k]; }
            if p[k] > hi[k] { hi[k] = p[k]; }
        }
    }
    let c = if o.centre { [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0] } else { [0.0; 3] };
    let mut r2 = 0.0;
    for p in lines.iter().copied().flat_map(points) {
        let (dx, dy, dz) = (p[0] - c[0], p[1] - c[1], p[2] - c[2]);
        let d = dx * dx + dy * dy + dz * dz;
        if d > r2 { r2 = d; }
    }
    // a fit of nothing, like a curve of no size, falls back on the usual
    let fit = match o.fit { Some(f) if f != 0.0 && !f.is_nan() => f, _ => FIT };
    let r = r2.sqrt();
    let sc = fit / if r != 0.0 && !r.is_nan() { r } else { 1.0 };
    let ax = o.axes.unwrap_or([0, 1, 2]);
    let mut e = o.e;
    // Turn the shape to face the camera: widest direction across, next widest up, flattest toward the viewer.
    // (Principal axes of the point cloud, by Jacobi rotations of its 3x3 covariance.)
    if o.align && e.is_none() {
        let mut cv = [[0.0f64; 3]; 3];
        let mut cnt = 0.0;
        for p in lines.iter().copied().flat_map(points) {
            let q = [(p[ax[0]] - c[ax[0]]) * sc, (p[ax[1]] - c[ax[1]]) * sc, (p[ax[2]] - c[ax[2]]) * sc];
            for a in 0..3 {
                for b in a..3 { cv[a][b] += q[a] * q[b]; }
            }
            cnt += 1.0;
        }
        for a in 0..3 {
            for b in a..3 {
                cv[a][b] /= cnt;
                cv[b][a] = cv[a][b];
            }
        }
        let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for _sweep in 0..12 {
            for (p, q) in [(0, 1), (0, 2), (1, 2)] {
                if cv[p][q].abs() < 1e-14 { continue; }
                let th = 0.5 * jsmath::atan2(2.0 * cv[p][q], cv[q][q] - cv[p][p]);
                let (cs, sn) = (jsmath::cos(th), jsmath::sin(th));
                for k in 0..3 {
                    let (a, b) = (cv[k][p], cv[k][q]);
                    cv[k][p] = cs * a - sn * b;
                    cv[k][q] = sn * a + cs * b;
                }
                for k in 0..3 {
                    let (a, b) = (cv[p][k], cv[q][k]);
                    cv[p][k] = cs * a - sn * b;
                    cv[q][k] = sn * a + cs * b;
                }
                for k in 0..3 {
                    let (a, b) = (v[k][p], v[k][q]);
                    v[k][p] = cs * a - sn * b;
                    v[k][q] = sn * a + cs * b;
                }
            }
        }
        // widest first; directions that are equally wide stay in the order x, y, z
        let mut order = [0usize, 1, 2];
        for i in 1..3 {
            let mut j = i;
            while j > 0 && cv[order[j]][order[j]] - cv[order[j - 1]][order[j - 1]] > 0.0 {
                order.swap(j, j - 1);
                j -= 1;
            }
        }
        let mut rows = [[0.0f64; 3]; 3];
        for (row, &j) in rows.iter_mut().zip(&order) {
            let ev = [v[0][j], v[1][j], v[2][j]];
            // each axis points the way its largest part points
            let mut big = 0.0f64;
            for x in ev {
                if x.abs() > big.abs() { big = x; }
            }
            *row = if big < 0.0 { [-ev[0], -ev[1], -ev[2]] } else { ev };
        }
        // and the three keep the handedness of x, y, z
        let det = rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1]) - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
            + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0]);
        if det < 0.0 { rows[2] = [-rows[2][0], -rows[2][1], -rows[2][2]]; }
        e = Some(rows);
    }
    Pose { c, sc, ax, e }
}

/// What `prep_curve` is asked for.
pub struct PrepOpts<'a> {
    /// the tube's radius
    pub tube: f64,
    /// strands closer than about this flow into each other (0 = hard union)
    pub melt: f64,
    /// the length of the steps the curve is cut into
    pub ds: f64,
    /// each line is a loop whose last point repeats its first
    pub closed: bool,
    /// radius factor from (fraction along, distance along, whole length); None = 1
    pub radius: Option<&'a (dyn Fn(f64, f64, f64) -> f64 + Sync)>,
    pub place: PlaceOpts,
    /// a placement worked out earlier, used instead of working one out
    pub pose: Option<Pose>,
}

/// A curve ready to be baked into the volume.
#[derive(Clone, Debug)]
pub struct Curve {
    /// x,y,z,radius per point
    pub pts: Vec<f32>,
    /// 1 = no segment from this point to the next
    pub brk: Vec<u8>,
    pub n: usize,
    pub rmax: f64,
    pub ds: f64,
    pub length: f64,
    /// with a melt: what every segment's contribution is weighted by, so that a strand alone keeps its radius
    pub amp: f64,
    pub pose: Pose,
    /// The radius differs a great deal from strand to strand (a tree: trunk and twigs). The melt's weight
    /// is then worked out for each segment's own radius rather than once for the tube, and the far field
    /// uses each point's own radius rather than the thickest.
    pub vary: bool,
    /// Nothing below this height (y, in the volume) is kept: a flat cut (`RawCurve::ground`).
    pub cut: Option<f64>,
}

/// How far it is from a to b.
#[inline]
fn apart(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    hypot3(b[0] - a[0], b[1] - a[1], b[2] - a[2])
}

/// lines: x,y,z... each. Place them (centre, scale, perhaps turn: o.pose, or worked out afresh), then cut
/// each line into equal steps of length ds, so the tube is equally smooth everywhere whatever the spacing
/// (forms.js `prepCurve`).
///
/// This runs every frame while a shape animates, so it does each sum once: the lengths of the chords are
/// worked out once and kept for the passes that need them again, and nothing is allocated per point.
/// None of that changes a single bit of the result: every number still comes from the same operations
/// on the same values, in the same order, as in the web version.
pub fn prep_curve(lines: &[&[f64]], o: &PrepOpts) -> Curve {
    let pose = match &o.pose { Some(p) => p.clone(), None => placement(lines, &o.place) };
    let (c, sc, ax) = (pose.c, pose.sc, pose.ax);

    // Every line placed in the volume, end to end in one list; `spans` says where each starts and how many
    // points it has. chord[i] is the distance to point i from the one before it on its line.
    let count: usize = lines.iter().map(|l| l.len() / 3).sum();
    let mut placed: Vec<[f64; 3]> = Vec::with_capacity(count);
    let mut chord: Vec<f64> = Vec::with_capacity(count);
    let mut spans: Vec<(usize, usize)> = Vec::with_capacity(lines.len());
    let mut length = 0.0;
    for l in lines {
        let first = placed.len();
        for p in points(l) {
            let (x, y, z) = ((p[ax[0]] - c[ax[0]]) * sc, (p[ax[1]] - c[ax[1]]) * sc, (p[ax[2]] - c[ax[2]]) * sc);
            placed.push(match &pose.e {
                Some(e) => [x * e[0][0] + y * e[0][1] + z * e[0][2], x * e[1][0] + y * e[1][1] + z * e[1][2], x * e[2][0] + y * e[2][1] + z * e[2][2]],
                None => [x, y, z],
            });
        }
        let t = &placed[first..];
        if !t.is_empty() { chord.push(0.0); }
        for w in t.windows(2) {
            let d = apart(&w[0], &w[1]);
            chord.push(d);
            length += d;
        }
        spans.push((first, t.len()));
    }
    let ds = o.ds.max(length / 240000.0);

    // With a melt, a segment's neighbours along its own strand all add to it. On a straight run of radius r
    // the sum at the surface is `sum`; weighting every contribution by 1/sum puts the surface back at r.
    let mut amp = 1.0;
    if o.melt > 0.0 {
        let (k, r) = (1.0 / o.melt, o.tube);
        let mut sum = 1.0; // the segment under the point, then its neighbours either side
        for m in 1..6000 {
            let t = jsmath::exp(-k * (hypot2(r, (m as f64 - 0.5) * ds) - r));
            sum += 2.0 * t;
            if t < 1e-8 { break; }
        }
        amp = 1.0 / sum;
    }

    // about how many points will come out (a step is never shorter than a 240000th of the whole)
    let steps = if length.is_finite() { ((length / ds) as usize).min(240000) } else { 0 };
    let room = steps + steps / 50 + 2 * spans.len() + 16;
    let (mut pts, mut brk): (Vec<f32>, Vec<u8>) = (Vec::with_capacity(room * 4), Vec::with_capacity(room));
    let mut rmax = 0.0;
    // one line at a time: `fine` is the smooth curve through its points, `seg` the lengths of that curve's chords
    let (mut fine, mut seg): (Vec<[f64; 3]>, Vec<f64>) = (Vec::new(), Vec::new());
    let half = 0.5 * ds;
    for &(first, m) in &spans {
        if m < 2 { continue; }
        let t = &placed[first..first + m];
        // The lines arrive as points that may be several steps apart (an integrator's time steps, say). Joined by
        // straight chords they make a faintly polygonal tube, and glass shows every bend. So first run a smooth
        // (Catmull-Rom) curve through them, fine enough that the chords left are far smaller than a step.
        let line: &[[f64; 3]] = if m < 3 {
            t
        } else {
            fine.clear();
            let along: f64 = chord[first + 1..first + m].iter().sum();
            if along.is_finite() { fine.reserve(((along / half) as usize).min(480000) + m + 1); }
            for i in 0..m - 1 {
                // the points before, at either end of, and after this chord. A closed loop repeats its first point
                // at the end: there the neighbours are taken round the join, so the curve leaves no kink.
                let (a, b, c, d) = if o.closed {
                    (t[(i + m - 2) % (m - 1)], t[i], t[(i + 1) % (m - 1)], t[(i + 2) % (m - 1)])
                } else {
                    (t[i.saturating_sub(1)], t[i], t[i + 1], t[(i + 2).min(m - 1)])
                };
                // the chord's length is already known, unless its far end was taken from the other side of the join
                let far = if o.closed && i + 2 == m { apart(&b, &c) } else { chord[first + i + 1] };
                let n = (far / half).ceil();
                // (a length that is not a number gives no points at all, as in the web version)
                let n = if n.is_nan() { 0.0 } else if n < 1.0 { 1.0 } else { n };
                // between b and c the curve is  0.5 (p0 + p1 u + p2 u^2 + p3 u^3)  for u from 0 to 1
                let p: [[f64; 4]; 3] = std::array::from_fn(|k| [2.0 * b[k], c[k] - a[k], 2.0 * a[k] - 5.0 * b[k] + 4.0 * c[k] - d[k], 3.0 * b[k] - a[k] - 3.0 * c[k] + d[k]]);
                let mut j = 0.0;
                while j < n {
                    let u = j / n;
                    let u2 = u * u;
                    let u3 = u2 * u;
                    fine.push(p.map(|p| 0.5 * (p[0] + p[1] * u + p[2] * u2 + p[3] * u3)));
                    j += 1.0;
                }
            }
            fine.push(t[m - 1]);
            &fine
        };

        // Now walk along it, putting down a point every ds.
        seg.clear();
        let mut total = 0.0;
        for w in line.windows(2) {
            let d = apart(&w[0], &w[1]);
            seg.push(d);
            total += d;
        }
        let whole = if total != 0.0 && !total.is_nan() { total } else { 1.0 };
        let mut emit = |p: [f64; 3], s: f64| {
            // the radius may vary along the tube: it is told the fraction along, the distance along, the whole length
            let r = o.tube * match o.radius { Some(f) => f(s / whole, s, total), None => 1.0 };
            if r > rmax { rmax = r; }
            pts.extend_from_slice(&[p[0] as f32, p[1] as f32, p[2] as f32, r as f32]);
            brk.push(0u8);
        };
        emit(line[0], 0.0);
        let (mut acc, mut next) = (0.0, ds);
        for (w, &len) in line.windows(2).zip(&seg) {
            if acc + len >= next {
                let (a, b) = (w[0], w[1]);
                let (dx, dy, dz) = (b[0] - a[0], b[1] - a[1], b[2] - a[2]);
                while acc + len >= next {
                    let u = (next - acc) / len;
                    emit([a[0] + dx * u, a[1] + dy * u, a[2] + dz * u], next);
                    next += ds;
                }
            }
            acc += len;
        }
        emit(line[line.len() - 1], total);
        *brk.last_mut().unwrap() = 1;
    }
    let n = brk.len();
    Curve { pts, brk, n, rmax, ds, length, amp, pose, vary: false, cut: None }
}

/// The thinnest a blade is drawn, in steps of the curve either side of its middle (a step is 0.75 of a
/// voxel). (GF_BLADE_FLOOR overrides it, for finding out what the grid holds.)
pub fn blade_floor() -> f64 {
    std::env::var("GF_BLADE_FLOOR").ok().and_then(|v| v.parse::<f64>().ok()).filter(|v| *v > 0.0).unwrap_or(1.5)
}

/// What `prep_strands` is asked for.
pub struct StrandOpts {
    /// every radius is multiplied by this (the thickness slider)
    pub thick: f64,
    /// the length of the steps the strands are cut into
    pub ds: f64,
    pub place: PlaceOpts,
    /// a placement worked out earlier, used instead of working one out from the lines
    pub pose: Option<Pose>,
    /// a level in the lines' own units below which nothing is kept (`RawCurve::ground`)
    pub ground: Option<f64>,
}

/// Strands that each carry their own radius at every point (`radii`, in the lines' own units): a tree's
/// branches. Placed as `prep_curve` places lines, run through the same smooth curve, and cut into equal
/// steps of `ds`; the radius is carried along the smooth curve with the points.
///
/// No radius comes out thinner than two steps (one and a half voxels of the volume being built): a
/// thinner strand would not be held by the grid and would break up into beads.
///
/// `blades` (leaves) follow the strands as three points each, told apart by a radius below zero: the base
/// (its radius the blade's half-thickness, negated), the tip (its radius the blade's half-width), and a
/// point one unit across the blade from the base (its radius the cup). The baker draws them as blades
/// (`bake_curve.wgsl`). A blade is no thinner than `BLADE_FLOOR` steps either side of its middle.
pub fn prep_strands(lines: &[&[f64]], radii: &[&[f64]], blades: &[crate::presets::Blade], o: &StrandOpts) -> Curve {
    let pose = match &o.pose { Some(p) => p.clone(), None => placement(lines, &o.place) };
    let (c, sc, ax) = (pose.c, pose.sc, pose.ax);
    let place = |p: &[f64; 3]| -> [f64; 3] {
        let (x, y, z) = ((p[ax[0]] - c[ax[0]]) * sc, (p[ax[1]] - c[ax[1]]) * sc, (p[ax[2]] - c[ax[2]]) * sc);
        match &pose.e {
            Some(e) => [x * e[0][0] + y * e[0][1] + z * e[0][2], x * e[1][0] + y * e[1][1] + z * e[1][2], x * e[2][0] + y * e[2][1] + z * e[2][2]],
            None => [x, y, z],
        }
    };
    // every strand placed: points and radii
    let strands: Vec<(Vec<[f64; 3]>, Vec<f64>)> = lines
        .iter()
        .zip(radii.iter().copied().chain(std::iter::repeat(&[][..])))
        .map(|(l, r)| {
            let pts: Vec<[f64; 3]> = points(l).iter().map(place).collect();
            let rs: Vec<f64> = (0..pts.len()).map(|i| r.get(i).copied().unwrap_or(0.0) * sc * o.thick).collect();
            (pts, rs)
        })
        .collect();
    let length: f64 = strands.iter().map(|(p, _)| p.windows(2).map(|w| apart(&w[0], &w[1])).sum::<f64>()).sum();
    let ds = o.ds.max(length / 240000.0);
    let floor = 2.0 * o.ds;
    let half = 0.5 * ds;

    let (mut pts, mut brk): (Vec<f32>, Vec<u8>) = (Vec::new(), Vec::new());
    let mut rmax = 0.0f64;
    let (mut fine, mut fr): (Vec<[f64; 3]>, Vec<f64>) = (Vec::new(), Vec::new());
    for (t, tr) in &strands {
        let m = t.len();
        if m < 2 { continue; }
        // a smooth (Catmull-Rom) curve through the points, cut finer than a step; the radius goes straight
        // from one point's to the next
        fine.clear();
        fr.clear();
        for i in 0..m - 1 {
            let (a, b, cc, d) = (t[i.saturating_sub(1)], t[i], t[i + 1], t[(i + 2).min(m - 1)]);
            let n = (apart(&b, &cc) / half).ceil().max(1.0);
            let n = if n.is_finite() { n as usize } else { 1 };
            let p: [[f64; 4]; 3] = std::array::from_fn(|k| [2.0 * b[k], cc[k] - a[k], 2.0 * a[k] - 5.0 * b[k] + 4.0 * cc[k] - d[k], 3.0 * b[k] - a[k] - 3.0 * cc[k] + d[k]]);
            for j in 0..n {
                let u = j as f64 / n as f64;
                let (u2, u3) = (u * u, u * u * u);
                fine.push(p.map(|p| 0.5 * (p[0] + p[1] * u + p[2] * u2 + p[3] * u3)));
                fr.push(tr[i] + (tr[i + 1] - tr[i]) * u);
            }
        }
        fine.push(t[m - 1]);
        fr.push(tr[m - 1]);

        // walk along it, putting down a point every ds
        let mut emit = |p: [f64; 3], r: f64| {
            let r = r.max(floor);
            if r > rmax { rmax = r; }
            pts.extend_from_slice(&[p[0] as f32, p[1] as f32, p[2] as f32, r as f32]);
            brk.push(0u8);
        };
        emit(fine[0], fr[0]);
        let (mut acc, mut next) = (0.0, ds);
        for (w, wr) in fine.windows(2).zip(fr.windows(2)) {
            let len = apart(&w[0], &w[1]);
            while acc + len >= next && len > 0.0 {
                let u = (next - acc) / len;
                emit(std::array::from_fn(|k| w[0][k] + (w[1][k] - w[0][k]) * u), wr[0] + (wr[1] - wr[0]) * u);
                next += ds;
            }
            acc += len;
        }
        emit(fine[fine.len() - 1], fr[fr.len() - 1]);
        *brk.last_mut().unwrap() = 1;
    }
    let thin = blade_floor() * o.ds;
    for b in blades {
        let (base, tip) = (place(&b.base), place(&b.tip));
        let side = place(&[b.base[0] + b.side[0], b.base[1] + b.side[1], b.base[2] + b.side[2]]);
        let side = [(side[0] - base[0]) / sc, (side[1] - base[1]) / sc, (side[2] - base[2]) / sc];
        let width = b.width * sc;
        if !(width > 0.0) || apart(&base, &tip) <= 0.0 { continue; }
        let t = (b.thick * sc * o.thick).max(thin);
        if width > rmax { rmax = width; }
        pts.extend_from_slice(&[base[0] as f32, base[1] as f32, base[2] as f32, -t as f32]);
        pts.extend_from_slice(&[tip[0] as f32, tip[1] as f32, tip[2] as f32, width as f32]);
        pts.extend_from_slice(&[(base[0] + side[0]) as f32, (base[1] + side[1]) as f32, (base[2] + side[2]) as f32, b.cup as f32]);
        brk.extend_from_slice(&[0, 1, 1]);
    }
    let n = brk.len();
    // the ground's level as placed (only for a shape that is not turned: a cut is level in the volume)
    let cut = o.ground.filter(|_| pose.e.is_none() && ax[1] == 1).map(|g| (g - c[1]) * sc);
    Curve { pts, brk, n, rmax, ds, length, amp: 1.0, pose, vary: true, cut }
}

/// A piece's first point carries this, less the number of points of data that follow its second, in
/// place of a radius (`add_pieces`). Anything below `PIECE_BELOW` is a piece's.
pub const PIECE_MARK: f32 = -1.0e6;
pub const PIECE_BELOW: f32 = -5.0e5;

/// How many points the piece starting at point i takes up, all told (0: no piece starts there).
pub fn piece_len(pts: &[f32], i: usize) -> usize {
    let w = pts[i * 4 + 3];
    if w < PIECE_BELOW { 2 + (PIECE_MARK - w).round() as usize } else { 0 }
}

/// Solid pieces (`presets::Piece`) added to a curve, placed as its strands are (`c.pose`).
///
/// Each piece is a run of points, told apart from a strand's by its first radius (`PIECE_MARK`):
///   the middle of the box it fills, placed (radius: PIECE_MARK less the count of data points)
///   the box's half-size (radius 0)
///   then its terms, each a tag (kind, and up to three numbers) and the term's own points:
///     0 plane      tag (0)            (n, d)                              n.p - d
///     1 ball       tag (1)            (c, r)                              |p - c| - r
///     2 hollow     tag (2)            (c, r)                              r - |p - c|
///     3 shell      tag (3, t)         (c, r)                              ||p - c| - r| - t
///     4 slab       tag (4, t)         (n, d)                              |n.p - d| - t
///     5 hex column tag (5, r, h)      (c), (axis), (u)
///     6 outline    tag (6, edges, depth, round)  (o, scale), (ux), (uy), (band y0, y1, margin), then
///                  (x0, y0, x1, y1) per edge
///     7 sweep      tag (7, round)     (a, half-width at a), (c, half-width at c), (side at a, half-height
///                  at a), (side at c, half-height at c), (cut at a, 1 if an end), (cut at c, 1 if an end)
/// The piece is everything inside all of its terms. The baker draws only the segment from its first
/// point to its second (`bake_curve.wgsl` `piece_dist`), into its box.
/// `thick` (the thickness slider) scales the walls of shells and slabs; `ds` is the curve's step: no wall
/// is drawn thinner than a blade is (`blade_floor`).
pub fn add_pieces(c: &mut Curve, pieces: &[crate::presets::Piece], thick: f64, ds: f64) {
    use crate::presets::Term;
    if pieces.is_empty() { return; }
    let pose = c.pose.clone();
    let (cc, sc, ax) = (pose.c, pose.sc, pose.ax);
    let dir = |d: &[f64; 3]| -> [f64; 3] {
        let (x, y, z) = (d[ax[0]], d[ax[1]], d[ax[2]]);
        match &pose.e {
            Some(e) => [x * e[0][0] + y * e[0][1] + z * e[0][2], x * e[1][0] + y * e[1][1] + z * e[1][2], x * e[2][0] + y * e[2][1] + z * e[2][2]],
            None => [x, y, z],
        }
    };
    let place = |p: &[f64; 3]| -> [f64; 3] {
        let q = dir(&[p[0] - cc[0], p[1] - cc[1], p[2] - cc[2]]);
        [q[0] * sc, q[1] * sc, q[2] * sc]
    };
    let thin = blade_floor() * ds;
    // (a film seen face on through its neighbours shows rings below some two voxels: walls hold that)
    let wall = thin.max(2.7 * ds);
    let f = |v: f64| v as f32;
    for pc in pieces {
        let mut data: Vec<f32> = Vec::new();
        let mut put = |a: f64, b: f64, cz: f64, d: f64| data.extend_from_slice(&[f(a), f(b), f(cz), f(d)]);
        let off = |n: &[f64; 3], d: f64| sc * (d - (n[0] * cc[0] + n[1] * cc[1] + n[2] * cc[2]));
        for t in &pc.terms {
            match t {
                Term::Plane { n, d } => {
                    let m = dir(n);
                    put(0.0, 0.0, 0.0, 0.0);
                    put(m[0], m[1], m[2], off(n, *d));
                }
                Term::Ball { c: o, r } | Term::Hollow { c: o, r } => {
                    let q = place(o);
                    put(if matches!(t, Term::Ball { .. }) { 1.0 } else { 2.0 }, 0.0, 0.0, 0.0);
                    put(q[0], q[1], q[2], r * sc);
                }
                Term::Shell { c: o, r, t } => {
                    let q = place(o);
                    put(3.0, (t * sc * thick).max(wall), 0.0, 0.0);
                    put(q[0], q[1], q[2], r * sc);
                }
                Term::Slab { n, d, t } => {
                    let m = dir(n);
                    put(4.0, (t * sc * thick).max(wall), 0.0, 0.0);
                    put(m[0], m[1], m[2], off(n, *d));
                }
                Term::Hex { c: o, axis, u, r, h } => {
                    let (q, a, uu) = (place(o), dir(axis), dir(u));
                    put(5.0, r * sc, (h * sc).max(thin), 0.0);
                    put(q[0], q[1], q[2], 0.0);
                    put(a[0], a[1], a[2], 0.0);
                    put(uu[0], uu[1], uu[2], 0.0);
                }
                Term::Glyph { o, ux, uy, depth, round, band, edges } => {
                    let (q, x, y) = (place(o), dir(ux), dir(uy));
                    let s = (ux[0] * ux[0] + ux[1] * ux[1] + ux[2] * ux[2]).sqrt().max(1e-300);
                    put(6.0, edges.len() as f64, (depth * sc).max(thin), (round * sc).max(0.0));
                    put(q[0], q[1], q[2], s * sc);
                    put(x[0] / s, x[1] / s, x[2] / s, 0.0);
                    put(y[0] / s, y[1] / s, y[2] / s, 0.0);
                    put(band[0], band[1], band[2], 0.0);
                    for e in edges { put(e[0], e[1], e[2], e[3]); }
                }
                Term::Sweep { a, c, side_a, side_c, size_a, size_c, round, n0, n1, ends } => {
                    // (the sizes are the thickness slider's to scale, and never thinner than a wall holds)
                    let k = sc * thick;
                    let s = |v: f64| (v * k).max(thin);
                    let (qa, qc, sa, scd, m0, m1) = (place(a), place(c), dir(side_a), dir(side_c), dir(n0), dir(n1));
                    put(7.0, round * k, 0.0, 0.0);
                    put(qa[0], qa[1], qa[2], s(size_a[0]));
                    put(qc[0], qc[1], qc[2], s(size_c[0]));
                    put(sa[0], sa[1], sa[2], s(size_a[1]));
                    put(scd[0], scd[1], scd[2], s(size_c[1]));
                    put(m0[0], m0[1], m0[2], if ends[0] { 1.0 } else { 0.0 });
                    put(m1[0], m1[1], m1[2], if ends[1] { 1.0 } else { 0.0 });
                }
            }
        }
        // the box, placed: the box round its eight corners placed
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for k in 0..8 {
            let q = place(&[if k & 1 == 0 { pc.lo[0] } else { pc.hi[0] }, if k & 2 == 0 { pc.lo[1] } else { pc.hi[1] }, if k & 4 == 0 { pc.lo[2] } else { pc.hi[2] }]);
            for a in 0..3 {
                lo[a] = lo[a].min(q[a]);
                hi[a] = hi[a].max(q[a]);
            }
        }
        if !(lo[0] <= hi[0]) || data.is_empty() { continue; }
        let n = data.len() / 4;
        c.pts.extend_from_slice(&[f(0.5 * (lo[0] + hi[0])), f(0.5 * (lo[1] + hi[1])), f(0.5 * (lo[2] + hi[2])), PIECE_MARK - n as f32]);
        c.pts.extend_from_slice(&[f(0.5 * (hi[0] - lo[0])), f(0.5 * (hi[1] - lo[1])), f(0.5 * (hi[2] - lo[2])), 0.0]);
        c.pts.extend_from_slice(&data);
        c.brk.push(0);
        c.brk.extend(std::iter::repeat_n(1u8, n + 1));
    }
    c.n = c.brk.len();
    c.vary = true;
    c.rmax = c.rmax.max(2.0 * ds);
}

impl Curve {
    /// Turn the curve by the rotation m and move it by t: every point of every strand and blade,
    /// and every piece's points as what each of them is (a position, a direction, a plane).
    pub fn moved(&mut self, m: &crate::math::Mat3, t: [f64; 3]) {
        let r = [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]];
        let rot = |q: &[f32]| -> [f64; 3] {
            let v = [q[0] as f64, q[1] as f64, q[2] as f64];
            [r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2], r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2], r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2]]
        };
        fn set(q: &mut [f32], v: [f64; 3]) {
            for a in 0..3 { q[a] = v[a] as f32; }
        }
        let moved = |v: [f64; 3]| [v[0] + t[0], v[1] + t[1], v[2] + t[2]];
        let n = self.pts.len() / 4;
        let mut i = 0;
        while i < n {
            let m = piece_len(&self.pts, i);
            if m == 0 {
                let w = rot(&self.pts[i * 4..]);
                set(&mut self.pts[i * 4..], moved(w));
                i += 1;
                continue;
            }
            // the box: its middle moves; its half-size becomes that of the box round the turned box
            let mid = rot(&self.pts[i * 4..]);
            set(&mut self.pts[i * 4..], moved(mid));
            let h = [self.pts[i * 4 + 4] as f64, self.pts[i * 4 + 5] as f64, self.pts[i * 4 + 6] as f64];
            let ext: [f64; 3] = std::array::from_fn(|a| (0..3).map(|b| (r[a][b] * h[b]).abs()).sum());
            set(&mut self.pts[i * 4 + 4..], ext);
            let (mut j, end) = (i + 2, i + m);
            while j < end {
                let kind = self.pts[j * 4].round() as i32;
                let at = (j + 1) * 4;
                match kind {
                    0 | 4 => {
                        // a plane n.p = d: n turns, and d gains n'.t
                        let nn = rot(&self.pts[at..]);
                        let d = self.pts[at + 3] as f64 + nn[0] * t[0] + nn[1] * t[1] + nn[2] * t[2];
                        set(&mut self.pts[at..], nn);
                        self.pts[at + 3] = d as f32;
                        j += 2;
                    }
                    1..=3 => {
                        let w = rot(&self.pts[at..]);
                        set(&mut self.pts[at..], moved(w));
                        j += 2;
                    }
                    7 => {
                        // a swept stretch: two positions, then four directions
                        for k in [1, 2] {
                            let w = rot(&self.pts[(j + k) * 4..]);
                            set(&mut self.pts[(j + k) * 4..], moved(w));
                        }
                        for k in 3..=6 {
                            let w = rot(&self.pts[(j + k) * 4..]);
                            set(&mut self.pts[(j + k) * 4..], w);
                        }
                        j += 7;
                    }
                    _ => {
                        // a hex column or an outline: a position, then two directions
                        let w = rot(&self.pts[at..]);
                        set(&mut self.pts[at..], moved(w));
                        for k in [2, 3] {
                            let w = rot(&self.pts[(j + k) * 4..]);
                            set(&mut self.pts[(j + k) * 4..], w);
                        }
                        let edges = if kind == 6 { self.pts[j * 4 + 1].round() as usize } else { 0 };
                        j += if kind == 6 { 5 } else { 4 } + edges;
                    }
                }
            }
            i += m;
        }
    }
}
