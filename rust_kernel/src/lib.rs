//! Rust FRSR kernel — same RNE semantics as dayval.minifloat, many times faster.
//!
//! Exposes two operations via PyO3:
//!   - `peak_error_single(...)`: peak relative error for one (K, c0, c1) config.
//!     Returns `(eps, x_star_bits)`. Used for unit tests against the Python kernel.
//!   - `k_sweep(...)`: for K in [k_lo, k_hi), compute peak ε and witness x*_bits.
//!     Returns three parallel arrays (one entry per K). Used by B2 K-only sweeps.
//!
//! Intermediate arithmetic is f64 with explicit RNE-to-(E, M, bias) at each op.
//! This is bit-identical to the Python kernel for E <= 11, M <= 23.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Copy, Clone, Debug)]
struct Fmt {
    e_bits: u32,
    m_bits: u32,
    bias: i32,
}

impl Fmt {
    fn new(e_bits: u32, m_bits: u32, bias: i32) -> Self {
        Self { e_bits, m_bits, bias }
    }
    #[inline]
    fn all_mask(&self) -> u32 {
        let w = 1 + self.e_bits + self.m_bits;
        if w >= 32 { u32::MAX } else { (1u32 << w) - 1 }
    }
    #[inline]
    fn sign_shift(&self) -> u32 {
        self.e_bits + self.m_bits
    }
    #[inline]
    fn exp_max_biased(&self) -> u32 {
        (1u32 << self.e_bits) - 1
    }
    #[inline]
    fn mant_mask(&self) -> u32 {
        (1u32 << self.m_bits) - 1
    }
}

/// Decode bit pattern → f64 per IEEE-style (E, M, bias).
#[inline]
fn bits_to_f64(bits: u32, fmt: Fmt) -> f64 {
    let sign = ((bits >> fmt.sign_shift()) & 1) as u64;
    let biased_e = ((bits >> fmt.m_bits) & ((1u32 << fmt.e_bits) - 1)) as i32;
    let mant = (bits & fmt.mant_mask()) as u64;
    let mant_frac = (mant as f64) / ((1u64 << fmt.m_bits) as f64);
    let v = if biased_e == 0 {
        // subnormal / zero
        mant_frac * 2f64.powi(1 - fmt.bias)
    } else if biased_e == fmt.exp_max_biased() as i32 {
        if mant == 0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    } else {
        (1.0 + mant_frac) * 2f64.powi(biased_e - fmt.bias)
    };
    if sign == 1 { -v } else { v }
}

/// RNE-round f64 to the nearest representable (E, M, bias) value, returned as f64.
/// Ties go to the even mantissa (banker's rounding). Pure bit manipulation —
/// no powi/frexp/round calls in the hot normal-range path.
#[inline]
fn quantize(v: f64, fmt: Fmt) -> f64 {
    if v.is_nan() {
        return f64::NAN;
    }
    if v == 0.0 {
        return v;
    }
    let bits = v.to_bits();
    let sign = bits >> 63;
    let biased_e_f64 = ((bits >> 52) & 0x7ff) as i32;
    let mant_f64 = bits & ((1u64 << 52) - 1);

    if biased_e_f64 == 0x7ff {
        // inf or NaN in f64; NaN handled above, so inf.
        return v;
    }
    if biased_e_f64 == 0 {
        // Subnormal f64 input (rare). Fall back to a slow path.
        return quantize_slow(v, fmt);
    }

    let unbiased_e = biased_e_f64 - 1023;
    let target_biased_e = unbiased_e + fmt.bias;

    // We want to keep the top M bits of the f64 mantissa, rounding RNE using
    // the next bit + sticky.
    let shift = 52 - fmt.m_bits as i32;
    let keep = mant_f64 >> shift;
    let low_mask = (1u64 << shift) - 1;
    let low = mant_f64 & low_mask;
    let half = 1u64 << (shift - 1);
    let round_up = low > half || (low == half && (keep & 1) != 0);
    let mut new_mant = keep + (if round_up { 1 } else { 0 });
    let mut new_biased_e = target_biased_e;
    if new_mant == (1u64 << fmt.m_bits) {
        new_mant = 0;
        new_biased_e += 1;
    }

    let exp_max_biased = fmt.exp_max_biased() as i32;
    if new_biased_e >= exp_max_biased {
        return if sign == 1 { f64::NEG_INFINITY } else { f64::INFINITY };
    }
    if new_biased_e < 1 {
        // Subnormal / underflow — less common. Handle with the slow path which
        // re-computes rounding from |v|.
        return quantize_subnormal(v, fmt, sign == 1);
    }

    // Reconstruct f64 bits from the (sign, target_biased_e, new_mant).
    let f64_biased_e = (new_biased_e - fmt.bias + 1023) as u64;
    let f64_mant = new_mant << shift;
    f64::from_bits((sign << 63) | (f64_biased_e << 52) | f64_mant)
}

#[cold]
fn quantize_subnormal(v: f64, fmt: Fmt, negative: bool) -> f64 {
    // Value in the target subnormal range (or zero).
    let a = v.abs();
    let scale = 2f64.powi(fmt.m_bits as i32 + fmt.bias - 1);
    let mant_sub = rne(a * scale) as i64;
    if mant_sub == 0 {
        return if negative { -0.0 } else { 0.0 };
    }
    if mant_sub == (1i64 << fmt.m_bits) {
        let result = 2f64.powi(1 - fmt.bias);
        return if negative { -result } else { result };
    }
    let result = (mant_sub as f64) * 2f64.powi(1 - fmt.bias - fmt.m_bits as i32);
    if negative { -result } else { result }
}

#[cold]
fn quantize_slow(v: f64, fmt: Fmt) -> f64 {
    // Unusual path: f64 subnormal input. Promote and retry via quantize.
    // f64 subnormals are in [2^-1074, 2^-1022), way below any target format's
    // normal range, so they always quantize to target subnormal or zero.
    let a = v.abs();
    quantize_subnormal(v, fmt, v.is_sign_negative() && a > 0.0)
}

/// Encode f64 → bit pattern for (E, M, bias) per RNE.
#[inline]
fn f64_to_bits(v: f64, fmt: Fmt) -> u32 {
    let sign_bit: u32 = if v.is_sign_negative() { 1 } else { 0 };
    let sign_in_place = sign_bit << fmt.sign_shift();
    if v.is_nan() {
        let exp = fmt.exp_max_biased() << fmt.m_bits;
        return sign_in_place | exp | (1u32 << (fmt.m_bits - 1));
    }
    if v == 0.0 {
        return sign_in_place;
    }
    let a = v.abs();
    if a.is_infinite() {
        let exp = fmt.exp_max_biased() << fmt.m_bits;
        return sign_in_place | exp;
    }
    let (mantissa, exp) = frexp(a);
    let unbiased_e = exp - 1;
    let f = 2.0 * mantissa - 1.0;
    let biased_e = unbiased_e + fmt.bias;
    let mant_scale = (1u64 << fmt.m_bits) as f64;
    let mut mant_int = rne(f * mant_scale) as i64;
    let mut final_biased_e = biased_e;
    if mant_int == (1i64 << fmt.m_bits) {
        mant_int = 0;
        final_biased_e += 1;
    }
    let exp_max_biased = fmt.exp_max_biased() as i32;
    if final_biased_e >= exp_max_biased {
        return sign_in_place | (fmt.exp_max_biased() << fmt.m_bits);
    }
    if final_biased_e < 1 {
        let scale = 2f64.powi(fmt.m_bits as i32 + fmt.bias - 1);
        let mant_sub = rne(a * scale) as i64;
        if mant_sub == (1i64 << fmt.m_bits) {
            return sign_in_place | (1u32 << fmt.m_bits);
        }
        if mant_sub == 0 {
            return sign_in_place;
        }
        return sign_in_place | (mant_sub as u32);
    }
    sign_in_place | ((final_biased_e as u32) << fmt.m_bits) | (mant_int as u32)
}

/// Round-to-nearest-even for f64. Rust 1.80+ ships `round_ties_even()` in std;
/// we emulate it here for 1.75 compatibility.
#[inline]
fn rne(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let r = x.round(); // rounds half away from zero
    let diff = (x - r).abs();
    if diff == 0.5 {
        // tie — adjust toward even
        if (r as i64) % 2 == 0 {
            r
        } else {
            r - r.signum()
        }
    } else {
        r
    }
}

/// Standard frexp for f64 that matches libc behaviour (no std::f64 exposure).
#[inline]
fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() || x.is_nan() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let biased_e = ((bits >> 52) & 0x7ff) as i32;
    if biased_e == 0 {
        // Subnormal f64. Normalise.
        let (m, e) = frexp(x * 2f64.powi(64));
        return (m, e - 64);
    }
    let exp = biased_e - 1022;
    let mantissa_bits = (bits & !(0x7ffu64 << 52)) | (1022u64 << 52);
    let mantissa = f64::from_bits(mantissa_bits);
    (mantissa, exp)
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum CoarseKind {
    ShiftThenSub,       // Y = K - (X >> 1)     [rsqrt, FRSR]
    SubThenShift2K,     // Y = (2K - X) >> 1    [rsqrt, §9.2]
    SubThenShift2K1,    // Y = (2K+1 - X) >> 1  [rsqrt, §9.2]
    NoShift,            // Y = K - X            [reciprocal, FRCP]
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum RefineKind {
    C1XYY,
    C1YXY,
    C1YYX,
    XYC1Y,
    XYYC1,
    YYC1X,
    YYXC1,
    C1XPairYY,
    C1YPairXY,
}

/// Target function x^(-a/b). Rsqrt = (1, 2), Recip = (1, 1).
#[derive(Copy, Clone, Debug, PartialEq)]
enum TargetFn {
    Rsqrt,
    Recip,
}

impl TargetFn {
    fn exp_ab(&self) -> f64 {
        match self {
            TargetFn::Rsqrt => 0.5,
            TargetFn::Recip => 1.0,
        }
    }
}

fn parse_target(s: &str) -> Option<TargetFn> {
    match s {
        "rsqrt" => Some(TargetFn::Rsqrt),
        "recip" => Some(TargetFn::Recip),
        _ => None,
    }
}

/// Algorithm tier per FRGR-PLAN.md §B2 (reference arm).
///
/// Each variant specifies: number of free coefficients and the functional
/// form of the refinement. `z` = `x * y^b` in the target format (b=2 for
/// rsqrt, b=1 for reciprocal), computed with canonical left-associative
/// ordering.
#[derive(Copy, Clone, Debug, PartialEq)]
enum Tier {
    T0Monic,        // y_final = y                         coefs: []
    T0Scale,        // y_final = y * k                     coefs: [k]
    T1Monic,        // y_final = y * (a - z)               coefs: [a]
    T1Gen,          // y_final = y * (c0 + c1*z)           coefs: [c0, c1]
    T2MonicHorner,  // y_final = y * (a + z*(z - b))       coefs: [a, b]
    T2GenHorner,    // y_final = y * (c0 + z*(c1 + z*c2))  coefs: [c0, c1, c2]
}

fn parse_tier(s: &str) -> Option<Tier> {
    match s {
        "T0_monic" => Some(Tier::T0Monic),
        "T0_scale" => Some(Tier::T0Scale),
        "T1_monic" => Some(Tier::T1Monic),
        "T1_gen" => Some(Tier::T1Gen),
        "T2_monic_horner" => Some(Tier::T2MonicHorner),
        "T2_gen_horner" => Some(Tier::T2GenHorner),
        _ => None,
    }
}

/// Compute z = x * y^b with canonical left-associative ordering and per-op RNE.
#[inline]
fn compute_z(y: f64, x: f64, fmt: Fmt, target: TargetFn) -> f64 {
    match target {
        TargetFn::Rsqrt => {
            // z = x * y * y, evaluated as (x * y) * y.
            let t = quantize(x * y, fmt);
            quantize(t * y, fmt)
        }
        TargetFn::Recip => quantize(x * y, fmt),
    }
}

/// Evaluate one tier's refinement given pre-quantized coefficients.
#[inline]
fn apply_tier(
    y: f64,
    x: f64,
    coefs_q: &[f64],
    fmt: Fmt,
    tier: Tier,
    target: TargetFn,
) -> f64 {
    match tier {
        Tier::T0Monic => y,
        Tier::T0Scale => {
            let k = coefs_q[0];
            quantize(y * k, fmt)
        }
        Tier::T1Monic => {
            let a = coefs_q[0];
            let z = compute_z(y, x, fmt, target);
            let t = quantize(a - z, fmt);
            quantize(y * t, fmt)
        }
        Tier::T1Gen => {
            // y * (c0 + c1 * z).
            let c0 = coefs_q[0];
            let c1 = coefs_q[1];
            let z = compute_z(y, x, fmt, target);
            let t = quantize(c1 * z, fmt);
            let s = quantize(c0 + t, fmt);
            quantize(y * s, fmt)
        }
        Tier::T2MonicHorner => {
            // y * (a + z*(z - b))
            let a = coefs_q[0];
            let b = coefs_q[1];
            let z = compute_z(y, x, fmt, target);
            let t = quantize(z - b, fmt);
            let t = quantize(t * z, fmt);
            let s = quantize(a + t, fmt);
            quantize(y * s, fmt)
        }
        Tier::T2GenHorner => {
            // y * (c0 + z*(c1 + z*c2))
            let c0 = coefs_q[0];
            let c1 = coefs_q[1];
            let c2 = coefs_q[2];
            let z = compute_z(y, x, fmt, target);
            let t = quantize(z * c2, fmt);
            let t = quantize(c1 + t, fmt);
            let t = quantize(t * z, fmt);
            let s = quantize(c0 + t, fmt);
            quantize(y * s, fmt)
        }
    }
}

fn num_coefs(tier: Tier) -> usize {
    match tier {
        Tier::T0Monic => 0,
        Tier::T0Scale | Tier::T1Monic => 1,
        Tier::T1Gen | Tier::T2MonicHorner => 2,
        Tier::T2GenHorner => 3,
    }
}

fn parse_coarse(s: &str) -> Option<CoarseKind> {
    match s {
        "shift_then_sub" => Some(CoarseKind::ShiftThenSub),
        "sub_then_shift_2K" => Some(CoarseKind::SubThenShift2K),
        "sub_then_shift_2K1" => Some(CoarseKind::SubThenShift2K1),
        "no_shift" => Some(CoarseKind::NoShift),
        _ => None,
    }
}

fn parse_refine(s: &str) -> Option<RefineKind> {
    match s {
        "c1xyy" => Some(RefineKind::C1XYY),
        "c1yxy" => Some(RefineKind::C1YXY),
        "c1yyx" => Some(RefineKind::C1YYX),
        "xyc1y" => Some(RefineKind::XYC1Y),
        "xyyc1" => Some(RefineKind::XYYC1),
        "yyc1x" => Some(RefineKind::YYC1X),
        "yyxc1" => Some(RefineKind::YYXC1),
        "c1x_yy" => Some(RefineKind::C1XPairYY),
        "c1y_xy" => Some(RefineKind::C1YPairXY),
        _ => None,
    }
}

#[inline]
fn apply_coarse(k: u32, x_bits: u32, fmt: Fmt, kind: CoarseKind) -> u32 {
    let mask = fmt.all_mask();
    match kind {
        CoarseKind::ShiftThenSub => (k as i64 - ((x_bits as i64) >> 1)) as u32 & mask,
        CoarseKind::SubThenShift2K => (((2i64 * k as i64) - x_bits as i64) >> 1) as u32 & mask,
        CoarseKind::SubThenShift2K1 => (((2i64 * k as i64 + 1) - x_bits as i64) >> 1) as u32 & mask,
        CoarseKind::NoShift => (k as i64 - x_bits as i64) as u32 & mask,
    }
}

#[inline]
fn apply_refine(
    y: f64, x: f64, c0q: f64, c1q: f64, fmt: Fmt, kind: RefineKind,
) -> f64 {
    let t = match kind {
        RefineKind::C1XYY => {
            let t = quantize(c1q * x, fmt);
            let t = quantize(t * y, fmt);
            quantize(t * y, fmt)
        }
        RefineKind::C1YXY => {
            let t = quantize(c1q * y, fmt);
            let t = quantize(t * x, fmt);
            quantize(t * y, fmt)
        }
        RefineKind::C1YYX => {
            let t = quantize(c1q * y, fmt);
            let t = quantize(t * y, fmt);
            quantize(t * x, fmt)
        }
        RefineKind::XYC1Y => {
            let t = quantize(x * y, fmt);
            let t = quantize(t * c1q, fmt);
            quantize(t * y, fmt)
        }
        RefineKind::XYYC1 => {
            let t = quantize(x * y, fmt);
            let t = quantize(t * y, fmt);
            quantize(t * c1q, fmt)
        }
        RefineKind::YYC1X => {
            let t = quantize(y * y, fmt);
            let t = quantize(t * c1q, fmt);
            quantize(t * x, fmt)
        }
        RefineKind::YYXC1 => {
            let t = quantize(y * y, fmt);
            let t = quantize(t * x, fmt);
            quantize(t * c1q, fmt)
        }
        RefineKind::C1XPairYY => {
            let a = quantize(c1q * x, fmt);
            let b = quantize(y * y, fmt);
            quantize(a * b, fmt)
        }
        RefineKind::C1YPairXY => {
            let a = quantize(c1q * y, fmt);
            let b = quantize(x * y, fmt);
            quantize(a * b, fmt)
        }
    };
    let s = quantize(c0q + t, fmt);
    quantize(y * s, fmt)
}

/// Compute peak relative error for one config over the supplied x_bits.
/// Returns (eps, x_star_bits). Non-finite errors are treated as f64::INFINITY
/// so overflow/NaN configurations are flagged.
///
/// For sweeps where x_bits is constant, prefer `peak_single_precomputed`:
/// it takes the already-decoded x values and x^(a/b) so the inner loop is
/// cheaper by ~30%.
#[inline]
fn peak_single(
    x_bits: &[u32],
    k: u32,
    c0: f64,
    c1: f64,
    fmt: Fmt,
    coarse: CoarseKind,
    refine: RefineKind,
    a: f64,
    b: f64,
) -> (f64, u32) {
    // Streaming path: decode each x on the fly. Avoids the 16 GB precompute
    // arrays that peak_single_precomputed would need at fp32. A single-K call
    // doesn't benefit from precomputation anyway.
    let c0q = quantize(c0, fmt);
    let c1q = quantize(c1, fmt);
    let exponent_ab = a / b;
    let mut peak = 0.0f64;
    let mut xstar = x_bits[0];
    for &xb in x_bits {
        let x = bits_to_f64(xb, fmt);
        let yb = apply_coarse(k, xb, fmt, coarse);
        let y = bits_to_f64(yb, fmt);
        let yfinal = apply_refine(y, x, c0q, c1q, fmt, refine);
        let err = (1.0 - x.powf(exponent_ab) * yfinal).abs();
        let err_safe = if err.is_finite() { err } else { f64::INFINITY };
        if err_safe > peak {
            peak = err_safe;
            xstar = xb;
        }
    }
    (peak, xstar)
}

#[inline]
fn peak_single_precomputed(
    x_bits: &[u32],
    x_f64: &[f64],
    target_pow: &[f64],
    k: u32,
    c0: f64,
    c1: f64,
    fmt: Fmt,
    coarse: CoarseKind,
    refine: RefineKind,
) -> (f64, u32) {
    let c0q = quantize(c0, fmt);
    let c1q = quantize(c1, fmt);
    let mut peak = 0.0f64;
    let mut xstar = x_bits[0];
    for i in 0..x_bits.len() {
        let xb = x_bits[i];
        let x = x_f64[i];
        let yb = apply_coarse(k, xb, fmt, coarse);
        let y = bits_to_f64(yb, fmt);
        let yfinal = apply_refine(y, x, c0q, c1q, fmt, refine);
        let err = (1.0 - target_pow[i] * yfinal).abs();
        let err_safe = if err.is_finite() { err } else { f64::INFINITY };
        if err_safe > peak {
            peak = err_safe;
            xstar = xb;
        }
    }
    (peak, xstar)
}

#[pyfunction]
#[pyo3(signature = (x_bits, k, c0, c1, e_bits, m_bits, bias,
                    coarse_ordering=String::from("shift_then_sub"),
                    refine_ordering=String::from("c1x_yy"),
                    a=1.0, b=2.0))]
fn peak_error_single(
    py: Python<'_>,
    x_bits: PyReadonlyArray1<u32>,
    k: u32,
    c0: f64,
    c1: f64,
    e_bits: u32,
    m_bits: u32,
    bias: i32,
    coarse_ordering: String,
    refine_ordering: String,
    a: f64,
    b: f64,
) -> PyResult<(f64, u32)> {
    let x = x_bits.as_slice()?;
    let fmt = Fmt::new(e_bits, m_bits, bias);
    let coarse = parse_coarse(&coarse_ordering)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bad coarse ordering"))?;
    let refine = parse_refine(&refine_ordering)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bad refine ordering"))?;
    let result = py.allow_threads(|| peak_single(x, k, c0, c1, fmt, coarse, refine, a, b));
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (x_bits, k_lo, k_hi, c0, c1, e_bits, m_bits, bias,
                    coarse_ordering=String::from("shift_then_sub"),
                    refine_ordering=String::from("c1x_yy"),
                    a=1.0, b=2.0))]
#[allow(clippy::too_many_arguments)]
fn k_sweep<'py>(
    py: Python<'py>,
    x_bits: PyReadonlyArray1<u32>,
    k_lo: u32,
    k_hi: u32,
    c0: f64,
    c1: f64,
    e_bits: u32,
    m_bits: u32,
    bias: i32,
    coarse_ordering: String,
    refine_ordering: String,
    a: f64,
    b: f64,
) -> PyResult<(&'py PyArray1<u32>, &'py PyArray1<f64>, &'py PyArray1<u32>)> {
    let x = x_bits.as_slice()?.to_vec();
    let fmt = Fmt::new(e_bits, m_bits, bias);
    let coarse = parse_coarse(&coarse_ordering)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bad coarse ordering"))?;
    let refine = parse_refine(&refine_ordering)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bad refine ordering"))?;
    if k_hi <= k_lo {
        return Err(pyo3::exceptions::PyValueError::new_err("k_hi must be > k_lo"));
    }
    let n = (k_hi - k_lo) as usize;
    let exponent_ab = a / b;
    let x_f64: Vec<f64> = x.iter().map(|&b| bits_to_f64(b, fmt)).collect();
    let target_pow: Vec<f64> = x_f64.iter().map(|&v| v.powf(exponent_ab)).collect();
    let (ks, eps, xs): (Vec<u32>, Vec<f64>, Vec<u32>) = py.allow_threads(|| {
        let results: Vec<(u32, f64, u32)> = (0..n as u32)
            .into_par_iter()
            .map(|i| {
                let k = k_lo + i;
                let (e, xs) = peak_single_precomputed(
                    &x, &x_f64, &target_pow, k, c0, c1, fmt, coarse, refine,
                );
                (k, e, xs)
            })
            .collect();
        let mut ks = Vec::with_capacity(n);
        let mut eps = Vec::with_capacity(n);
        let mut xs_arr = Vec::with_capacity(n);
        for (k, e, xs) in results {
            ks.push(k);
            eps.push(e);
            xs_arr.push(xs);
        }
        (ks, eps, xs_arr)
    });
    Ok((
        ks.into_pyarray(py),
        eps.into_pyarray(py),
        xs.into_pyarray(py),
    ))
}

/// Utility: quantize a single f64 to (E, M, bias) — used by tests.
#[pyfunction]
fn quantize_f64(v: f64, e_bits: u32, m_bits: u32, bias: i32) -> f64 {
    quantize(v, Fmt::new(e_bits, m_bits, bias))
}

/// Peak error for a tier configuration (one (K, coefs)).
/// coefs length must match num_coefs(tier).
#[pyfunction]
#[pyo3(signature = (x_bits, k, coefs, e_bits, m_bits, bias,
                    tier=String::from("T1_gen"),
                    coarse_ordering=String::from("shift_then_sub"),
                    target=String::from("rsqrt")))]
#[allow(clippy::too_many_arguments)]
fn tier_peak(
    py: Python<'_>,
    x_bits: PyReadonlyArray1<u32>,
    k: u32,
    coefs: Vec<f64>,
    e_bits: u32,
    m_bits: u32,
    bias: i32,
    tier: String,
    coarse_ordering: String,
    target: String,
) -> PyResult<(f64, u32)> {
    let fmt = Fmt::new(e_bits, m_bits, bias);
    let t = parse_tier(&tier)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bad tier"))?;
    let coarse = parse_coarse(&coarse_ordering)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bad coarse"))?;
    let tgt = parse_target(&target)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bad target"))?;
    if coefs.len() != num_coefs(t) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("tier {:?} expects {} coefs, got {}", t, num_coefs(t), coefs.len())
        ));
    }
    let x = x_bits.as_slice()?;
    let coefs_q: Vec<f64> = coefs.iter().map(|&c| quantize(c, fmt)).collect();
    let exp_ab = tgt.exp_ab();
    let result = py.allow_threads(|| {
        let mut peak = 0.0f64;
        let mut xstar = x[0];
        for &xb in x {
            let xv = bits_to_f64(xb, fmt);
            let yb = apply_coarse(k, xb, fmt, coarse);
            let y = bits_to_f64(yb, fmt);
            let y_final = apply_tier(y, xv, &coefs_q, fmt, t, tgt);
            let err = (1.0 - xv.powf(exp_ab) * y_final).abs();
            let err_safe = if err.is_finite() { err } else { f64::INFINITY };
            if err_safe > peak {
                peak = err_safe;
                xstar = xb;
            }
        }
        (peak, xstar)
    });
    Ok(result)
}

/// Exhaustive (K × coef_grid) sweep for a tier. `coef_candidates` is a list
/// of f64 values; the sweep tries every length-num_coefs(tier) combination
/// from it (Cartesian product, with repetition).
///
/// Returns (best_K, best_coefs, best_eps, best_x_star).
/// Parallelizes across (K, c0_idx) pairs with rayon.
#[pyfunction]
#[pyo3(signature = (x_bits, k_lo, k_hi, coef_candidates, e_bits, m_bits, bias,
                    tier=String::from("T1_gen"),
                    coarse_ordering=String::from("shift_then_sub"),
                    target=String::from("rsqrt")))]
#[allow(clippy::too_many_arguments)]
fn tier_exhaustive(
    py: Python<'_>,
    x_bits: PyReadonlyArray1<u32>,
    k_lo: u32,
    k_hi: u32,
    coef_candidates: Vec<f64>,
    e_bits: u32,
    m_bits: u32,
    bias: i32,
    tier: String,
    coarse_ordering: String,
    target: String,
) -> PyResult<(u32, Vec<f64>, f64, u32)> {
    let fmt = Fmt::new(e_bits, m_bits, bias);
    let t = parse_tier(&tier)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bad tier"))?;
    let coarse = parse_coarse(&coarse_ordering)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bad coarse"))?;
    let tgt = parse_target(&target)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bad target"))?;
    let x: Vec<u32> = x_bits.as_slice()?.to_vec();
    let exp_ab = tgt.exp_ab();
    let nc = num_coefs(t);
    let cand_q: Vec<f64> = coef_candidates.iter().map(|&c| quantize(c, fmt)).collect();
    let n_cand = cand_q.len();
    let total_k = k_hi.saturating_sub(k_lo) as usize;

    // Precompute x_f64 and target_pow for the full x set. These are shared
    // across all (K, coefs) and amortise heavily at larger formats.
    let x_f64: Vec<f64> = x.iter().map(|&b| bits_to_f64(b, fmt)).collect();
    let target_pow: Vec<f64> = x_f64.iter().map(|&v| v.powf(exp_ab)).collect();

    // Outer parallelism: flatten (K, coef[0]) for coarse-grained work
    // distribution. Inner serial loops over remaining coefs.
    let (best_k, best_coefs, best_eps, best_xstar) = py.allow_threads(|| {
        let iter_space: Vec<(u32, usize)> = (0..total_k as u32)
            .flat_map(|ki| (0..n_cand.max(1)).map(move |ci0| (k_lo + ki, ci0)))
            .collect();
        let res: Option<(f64, u32, Vec<f64>, u32)> = iter_space
            .into_par_iter()
            .map(|(k, ci0)| {
                let mut coefs_q = vec![0.0f64; nc];
                let mut best_local = (f64::INFINITY, 0u32, vec![0.0f64; nc], 0u32);
                // nc == 0: single iteration, no coefs.
                let iter_rest = if nc <= 1 { 1 } else { n_cand.pow((nc - 1) as u32) };
                for r in 0..iter_rest {
                    // Decode r into (c1_idx, c2_idx, ...).
                    let mut rr = r;
                    if nc >= 1 {
                        coefs_q[0] = cand_q[ci0];
                    }
                    for ci in 1..nc {
                        coefs_q[ci] = cand_q[rr % n_cand];
                        rr /= n_cand;
                    }
                    // Evaluate peak over all x.
                    let mut peak = 0.0f64;
                    let mut xstar = x[0];
                    for i in 0..x.len() {
                        let xb = x[i];
                        let xv = x_f64[i];
                        let yb = apply_coarse(k, xb, fmt, coarse);
                        let y = bits_to_f64(yb, fmt);
                        let y_final = apply_tier(y, xv, &coefs_q, fmt, t, tgt);
                        let err = (1.0 - target_pow[i] * y_final).abs();
                        let err_safe = if err.is_finite() { err } else { f64::INFINITY };
                        if err_safe > peak {
                            peak = err_safe;
                            xstar = xb;
                        }
                    }
                    if peak < best_local.0 {
                        best_local = (peak, k, coefs_q.clone(), xstar);
                    }
                }
                best_local
            })
            .reduce_with(|a, b| if a.0 <= b.0 { a } else { b });
        let (eps, k, coefs, xs) = res.unwrap_or((f64::INFINITY, 0, vec![0.0; nc], 0));
        (k, coefs, eps, xs)
    });
    Ok((best_k, best_coefs, best_eps, best_xstar))
}

/// Format-intrinsic floor: for each positive normal input, compute f(x) in
/// high precision (f64 is plenty at these widths), round to fmt, measure
/// peak relative error per Day eq (10). Returns (eps_floor, x_star_bits).
#[pyfunction]
fn format_floor(
    x_bits: PyReadonlyArray1<u32>,
    e_bits: u32,
    m_bits: u32,
    bias: i32,
    target: String,
) -> PyResult<(f64, u32)> {
    let fmt = Fmt::new(e_bits, m_bits, bias);
    let tgt = parse_target(&target)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bad target"))?;
    let x = x_bits.as_slice()?;
    let exp_ab = tgt.exp_ab();
    let mut peak = 0.0f64;
    let mut xstar = x[0];
    for &xb in x {
        let xv = bits_to_f64(xb, fmt);
        // f(x) = x^(-a/b).
        let fx = xv.powf(-exp_ab);
        // Round f(x) to fmt.
        let fx_q = quantize(fx, fmt);
        // Relative error per eq (10): 1 - x^(a/b) * y_tilde.
        let err = (1.0 - xv.powf(exp_ab) * fx_q).abs();
        let err_safe = if err.is_finite() { err } else { f64::INFINITY };
        if err_safe > peak {
            peak = err_safe;
            xstar = xb;
        }
    }
    Ok((peak, xstar))
}

#[pymodule]
fn dayval_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(peak_error_single, m)?)?;
    m.add_function(wrap_pyfunction!(k_sweep, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_f64, m)?)?;
    m.add_function(wrap_pyfunction!(tier_peak, m)?)?;
    m.add_function(wrap_pyfunction!(tier_exhaustive, m)?)?;
    m.add_function(wrap_pyfunction!(format_floor, m)?)?;
    Ok(())
}
