"""Parametric (E, M, bias) minifloat arithmetic with IEEE-style round-to-nearest-even.

All values are represented as numpy unsigned-integer bit patterns of width
1 + E + M. Intermediate arithmetic is performed in float64 and rounded back
to the target format. This is correct whenever the intermediate fits in
float64 with room for the RNE decision — true for all (E, M) we target
(M <= 23, E <= 8).

No FMA. Each binary op independently rounds. Callers wanting a particular
product ordering must express it as a sequence of binary ops.

The "harness" in the plan's §A4 is this module: software arithmetic with
bit-exact behaviour independent of host ISA.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Format:
    E: int  # exponent bits
    M: int  # mantissa (fraction) bits
    bias: int

    @property
    def width(self) -> int:
        return 1 + self.E + self.M

    @property
    def sign_shift(self) -> int:
        return self.E + self.M

    @property
    def exp_max_biased(self) -> int:
        """Largest biased exponent that is NOT inf/NaN."""
        return (1 << self.E) - 2

    @property
    def mantissa_mask(self) -> int:
        return (1 << self.M) - 1

    @property
    def exp_mask(self) -> int:
        return ((1 << self.E) - 1) << self.M

    @property
    def sign_mask(self) -> int:
        return 1 << self.sign_shift

    @property
    def all_mask(self) -> int:
        return (1 << self.width) - 1

    def __repr__(self) -> str:
        return f"fp{self.width}(E={self.E},M={self.M},bias={self.bias})"


# Validation-arm formats (sign-exp-mantissa). FP6 and FP8 aliases below
# point at the validation-arm layouts for backward compatibility; the
# OCP-MX production formats used by the reference arm have their own
# named entries.
FP4 = Format(E=2, M=1, bias=1)               # MXFP4 / OCP E2M1
FP6 = Format(E=3, M=2, bias=3)               # alias for FP6_E3M2
FP8 = Format(E=4, M=3, bias=7)               # alias for FP8_E4M3
FP16 = Format(E=5, M=10, bias=15)
FP18 = Format(E=6, M=11, bias=31)
FP20 = Format(E=6, M=13, bias=31)
FP24 = Format(E=7, M=16, bias=63)
FP32 = Format(E=8, M=23, bias=127)

BF16 = Format(E=8, M=7, bias=127)  # phase 2 but handy to declare

# OCP Microscaling production formats for the reference arm (FRGR-PLAN.md).
# FP8 E4M3 and E5M2 follow the Micikevicius et al. [2022] / OCP MX spec,
# which diverges from strict IEEE at E4M3 (no ±inf, single-pattern NaN).
# For FRSR/FRCP characterization those specials don't affect positive-
# normal enumeration — they only change what's reserved at the exponent
# boundaries, not the set of positive normals. Our enumerator iterates the
# positive-normal set defined by biased exp in [1, 2^E - 2], matching any
# spec that reserves the all-ones exponent for inf/NaN.
FP4_E2M1 = Format(E=2, M=1, bias=1)          # 4 pos normals
FP6_E2M3 = Format(E=2, M=3, bias=1)          # 12 pos normals
FP6_E3M2 = Format(E=3, M=2, bias=3)          # 24 pos normals
FP8_E4M3 = Format(E=4, M=3, bias=7)          # 112 pos normals
FP8_E5M2 = Format(E=5, M=2, bias=15)         # 112 pos normals

PHASE1 = (FP4, FP6, FP8, FP16, FP18, FP20, FP24, FP32)
LOWBIT = (FP4_E2M1, FP6_E2M3, FP6_E3M2, FP8_E4M3, FP8_E5M2)


def positive_normals_bits(fmt: Format) -> np.ndarray:
    """All positive-normal bit patterns for the format, ascending magnitude."""
    num_exps = fmt.exp_max_biased  # biased 1..2^E - 2
    num_mants = 1 << fmt.M
    total = num_exps * num_mants
    idx = np.arange(total, dtype=np.uint32)
    exps = (idx >> fmt.M) + 1
    mants = idx & fmt.mantissa_mask
    return ((exps << fmt.M) | mants).astype(np.uint32)


def bits_to_float(bits, fmt: Format) -> np.ndarray:
    """Decode uint bit patterns to float64 values. Handles subnormals, zero,
    inf, NaN per IEEE-style interpretation of (E, M, bias)."""
    b = np.asarray(bits, dtype=np.uint32).astype(np.uint64)
    sign = ((b >> fmt.sign_shift) & 1).astype(np.int64)
    biased_e = ((b >> fmt.M) & ((1 << fmt.E) - 1)).astype(np.int64)
    mant = (b & fmt.mantissa_mask).astype(np.int64)

    exp_max_biased = (1 << fmt.E) - 1
    is_zero_or_sub = biased_e == 0
    is_inf_or_nan = biased_e == exp_max_biased

    mant_frac = mant.astype(np.float64) / (1 << fmt.M)
    exp_val = biased_e - fmt.bias
    scale_normal = np.power(2.0, exp_val.astype(np.float64))
    val_normal = (1.0 + mant_frac) * scale_normal

    scale_sub = 2.0 ** (1 - fmt.bias)
    val_sub = mant_frac * scale_sub

    val = np.where(is_zero_or_sub, val_sub, val_normal)
    val = np.where(is_inf_or_nan & (mant == 0), np.inf, val)
    val = np.where(is_inf_or_nan & (mant != 0), np.nan, val)

    sign_mult = np.where(sign == 1, -1.0, 1.0)
    return (sign_mult * val).astype(np.float64)


def float_to_bits(v, fmt: Format) -> np.ndarray:
    """Encode float64 values to uint bit patterns with round-to-nearest-even.

    Ties (exact half) round to even mantissa. Overflow → inf with input sign.
    Underflow → subnormal or signed zero. NaN → a canonical quiet NaN pattern
    with the input sign preserved."""
    v = np.asarray(v, dtype=np.float64)
    # Shape-broadcast to an array so np.where works.
    v = np.broadcast_to(v, np.shape(v)).copy()

    sign_bit = np.signbit(v).astype(np.uint64)
    is_nan = np.isnan(v)
    is_inf = np.isinf(v)
    is_zero = v == 0.0
    finite_nonzero = np.isfinite(v) & (~is_zero)

    abs_v = np.abs(v)

    # Default: use frexp on abs_v for finite nonzero values. Force finite_nonzero
    # values through even though numpy will also process the rest; we will mask
    # at the end.
    safe_abs = np.where(finite_nonzero, abs_v, np.full_like(abs_v, 1.0))
    mantissa_frexp, exp_frexp = np.frexp(safe_abs)
    unbiased_e = (exp_frexp - 1).astype(np.int64)
    f = (2.0 * mantissa_frexp - 1.0).astype(np.float64)  # in [0, 1)

    biased_e = unbiased_e + fmt.bias  # int64

    # Normal-path rounding: mant_int = RNE(f * 2^M).
    mant_scaled = f * float(1 << fmt.M)
    mant_int = np.rint(mant_scaled).astype(np.int64)  # RNE
    rollover = mant_int == (1 << fmt.M)
    mant_int = np.where(rollover, 0, mant_int)
    biased_e = np.where(rollover, biased_e + 1, biased_e)

    # Underflow path: value represents subnormal or rounds up into smallest normal.
    # Subnormal integer mantissa = RNE(abs_v * 2^(M + bias - 1)). Compute only
    # for finite inputs to avoid rint(inf) cast warnings; the result is masked
    # out by the is_inf/is_nan handling below anyway.
    abs_v_safe = np.where(np.isfinite(abs_v), abs_v, 0.0)
    mant_sub_scaled = abs_v_safe * (2.0 ** (fmt.M + fmt.bias - 1))
    # Clip to a safe int64-representable range before cast. Values outside this
    # window round to irrelevant integers because the underflow/overflow masks
    # below will discard them.
    INT_CLAMP = 1 << 62
    mant_sub_scaled = np.clip(mant_sub_scaled, -INT_CLAMP, INT_CLAMP)
    mant_sub = np.rint(mant_sub_scaled).astype(np.int64)
    sub_rollover = mant_sub == (1 << fmt.M)

    exp_max_biased = (1 << fmt.E) - 1
    underflow = biased_e < 1
    overflow = biased_e >= exp_max_biased

    final_biased_e = np.where(
        underflow,
        np.where(sub_rollover, 1, 0),
        np.where(overflow, exp_max_biased, biased_e),
    ).astype(np.int64)
    final_mant = np.where(
        underflow,
        np.where(sub_rollover, 0, mant_sub),
        np.where(overflow, 0, mant_int),
    ).astype(np.int64)

    bits = ((sign_bit << fmt.sign_shift)
            | (final_biased_e.astype(np.uint64) << fmt.M)
            | final_mant.astype(np.uint64))

    # Zero
    bits = np.where(is_zero, sign_bit << fmt.sign_shift, bits)
    # Inf
    inf_pattern = np.uint64(exp_max_biased) << np.uint64(fmt.M)
    bits = np.where(is_inf, (sign_bit << fmt.sign_shift) | inf_pattern, bits)
    # NaN
    nan_pattern = inf_pattern | np.uint64(1 << (fmt.M - 1))
    bits = np.where(is_nan, (sign_bit << fmt.sign_shift) | nan_pattern, bits)

    return bits.astype(np.uint32)


# ---------------------------------------------------------------------------
# Arithmetic ops. Each takes/returns bits in the given format.
# Every op does exactly one RNE at the end.
# ---------------------------------------------------------------------------

def quantize(v, fmt: Format) -> np.ndarray:
    """Round a float64 value to the nearest representable value in `fmt`, but
    return the result as float64. Equivalent to
      bits_to_float(float_to_bits(v, fmt), fmt)
    and correct for (E <= 11, M <= 23) on standard IEEE-754 float64 hardware."""
    return bits_to_float(float_to_bits(v, fmt), fmt)


def fmul(a, b, fmt: Format) -> np.ndarray:
    return float_to_bits(bits_to_float(a, fmt) * bits_to_float(b, fmt), fmt)


def fadd(a, b, fmt: Format) -> np.ndarray:
    return float_to_bits(bits_to_float(a, fmt) + bits_to_float(b, fmt), fmt)


def fsub(a, b, fmt: Format) -> np.ndarray:
    return float_to_bits(bits_to_float(a, fmt) - bits_to_float(b, fmt), fmt)


def fneg(a, fmt: Format) -> np.ndarray:
    # Flip sign bit — correct for all finite and inf; NaN stays NaN.
    a = np.asarray(a, dtype=np.uint32)
    return a ^ np.uint32(fmt.sign_mask)
