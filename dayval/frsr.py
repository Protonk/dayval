"""Parametric FRSR kernel for Day (2023) validation.

Implements the two coarse-approximation orderings from §9.2 and the nine
multiplicative orderings of `c1 * x * y * y` from the same section. Kadlec-
style factorings (w = c1' * y with 7 variants) are declared but not yet
implemented — Phase 1 can produce ablation-table columns 1, 2, and a partial
step 3 without them.

All arithmetic is done by the software harness in `minifloat.py`: values are
decoded to float64, operated upon, and RNE-rounded to the target format after
each binary op. No FMA.

Input and output are numpy arrays of uint32 bit patterns in the target format.
Vectorization is over `x` (all positive normals) by default; batch over `K`
by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import minifloat as mf


COARSE_ORDERINGS = ("shift_then_sub", "sub_then_shift_2K", "sub_then_shift_2K1")

# 9 refinement orderings from §9.2. Each computes the product c1*x*y*y as a
# left-to-right sequence of three binary multiplies (plus one "pair" variant).
# Labels follow the paper row/column order.
REFINE_ORDERINGS = (
    "c1xyy",   # c1 * x * y * y
    "c1yxy",   # c1 * y * x * y
    "c1yyx",   # c1 * y * y * x
    "xyc1y",   # x * y * c1 * y
    "xyyc1",   # x * y * y * c1
    "yyc1x",   # y * y * c1 * x
    "yyxc1",   # y * y * x * c1
    "c1x_yy",  # (c1 * x) * (y * y)
    "c1y_xy",  # (c1 * y) * (x * y)
)


def coarse(K: int, X_bits: np.ndarray, fmt: mf.Format,
           ordering: str = "shift_then_sub") -> np.ndarray:
    """Coarse-approximation integer stage.

    Shift-then-subtract (Listing 1): Y = K - (X >> 1).
    Subtract-then-shift (§9.2):      Y = (C' - X) >> 1, C' in {2K, 2K+1}.

    K (or C') is an int; X_bits is a uint32 array of target-format bit
    patterns. Returns a uint32 array of target-format bit patterns, masked to
    the format width. The masking matters for formats narrower than 32 bits
    (fp4..fp24) — values outside the format just wrap, which is the same
    behaviour the hardware would exhibit on a native-width integer.
    """
    X_bits = np.asarray(X_bits, dtype=np.uint32)
    mask = np.uint32(fmt.all_mask)
    K = int(K)
    if ordering == "shift_then_sub":
        Y = (np.int64(K) - (X_bits.astype(np.int64) >> 1)).astype(np.uint32) & mask
    elif ordering == "sub_then_shift_2K":
        Cp = np.int64(2 * K)
        Y = (((Cp - X_bits.astype(np.int64)) >> 1).astype(np.uint32)) & mask
    elif ordering == "sub_then_shift_2K1":
        Cp = np.int64(2 * K + 1)
        Y = (((Cp - X_bits.astype(np.int64)) >> 1).astype(np.uint32)) & mask
    else:
        raise ValueError(f"unknown coarse ordering {ordering!r}")
    return Y


def _q(v, fmt):
    return mf.quantize(v, fmt)


def refine(y: np.ndarray, x: np.ndarray, c0: float, c1: float,
           fmt: mf.Format, ordering: str = "c1x_yy") -> np.ndarray:
    """Compute y * (c0 + <c1*x*y*y ordered by `ordering`>), one RNE per op.

    All arguments are already-quantized float64 arrays (output of
    bits_to_float). c0 and c1 are scalars that will be pre-quantized to the
    target format before use. Returns a float64 array of target-format values.
    """
    c0q = float(mf.quantize(np.float64(c0), fmt))
    c1q = float(mf.quantize(np.float64(c1), fmt))

    if ordering == "c1xyy":
        t = _q(c1q * x, fmt)
        t = _q(t * y, fmt)
        t = _q(t * y, fmt)
    elif ordering == "c1yxy":
        t = _q(c1q * y, fmt)
        t = _q(t * x, fmt)
        t = _q(t * y, fmt)
    elif ordering == "c1yyx":
        t = _q(c1q * y, fmt)
        t = _q(t * y, fmt)
        t = _q(t * x, fmt)
    elif ordering == "xyc1y":
        t = _q(x * y, fmt)
        t = _q(t * c1q, fmt)
        t = _q(t * y, fmt)
    elif ordering == "xyyc1":
        t = _q(x * y, fmt)
        t = _q(t * y, fmt)
        t = _q(t * c1q, fmt)
    elif ordering == "yyc1x":
        t = _q(y * y, fmt)
        t = _q(t * c1q, fmt)
        t = _q(t * x, fmt)
    elif ordering == "yyxc1":
        t = _q(y * y, fmt)
        t = _q(t * x, fmt)
        t = _q(t * c1q, fmt)
    elif ordering == "c1x_yy":
        a_ = _q(c1q * x, fmt)
        b_ = _q(y * y, fmt)
        t = _q(a_ * b_, fmt)
    elif ordering == "c1y_xy":
        a_ = _q(c1q * y, fmt)
        b_ = _q(x * y, fmt)
        t = _q(a_ * b_, fmt)
    else:
        raise ValueError(f"unknown refine ordering {ordering!r}")

    s = _q(c0q + t, fmt)
    return _q(y * s, fmt)


def frsr(x_bits: np.ndarray, K: int, c0: float, c1: float, fmt: mf.Format,
         coarse_ordering: str = "shift_then_sub",
         refine_ordering: str = "c1x_yy") -> np.ndarray:
    """One-shot FRSR. Given x (bits), returns y_final (float64, quantized).

    Use this for correctness checks. For large sweeps, prefer `peak_error`
    which avoids unnecessary decodings and materializations.
    """
    x = mf.bits_to_float(x_bits, fmt)
    Y = coarse(K, x_bits, fmt, coarse_ordering)
    y = mf.bits_to_float(Y, fmt)
    return refine(y, x, c0, c1, fmt, refine_ordering)


def relative_error(x_bits: np.ndarray, K: int, c0: float, c1: float,
                   fmt: mf.Format, coarse_ordering: str = "shift_then_sub",
                   refine_ordering: str = "c1x_yy",
                   a: int = 1, b: int = 2) -> np.ndarray:
    """Relative error |1 - x^(a/b) * y_final| over the input bit patterns.

    For FRSR (a=1, b=2), the target is x^(-1/2) and the check is
    |1 - sqrt(x) * y_final|. Returned array has the same shape as x_bits.
    """
    x = mf.bits_to_float(x_bits, fmt)
    y_final = frsr(x_bits, K, c0, c1, fmt, coarse_ordering, refine_ordering)
    # Target x^(-a/b). Compute "truth" in fp64; target is x^(a/b) * y_final == 1.
    target_inv = np.power(x, np.float64(a) / b)  # x^(a/b)
    return np.abs(1.0 - target_inv * y_final)


def peak_error(x_bits: np.ndarray, K: int, c0: float, c1: float,
               fmt: mf.Format, coarse_ordering: str = "shift_then_sub",
               refine_ordering: str = "c1x_yy",
               a: int = 1, b: int = 2) -> tuple[float, int]:
    """Peak |relative error| over x_bits, with a witness bit pattern.

    Returns (eps, x_star_bits) where eps is the peak and x_star_bits is the
    input bit pattern that realises it. Non-finite errors (NaN/inf) are
    treated as np.inf for the purposes of argmax.
    """
    err = relative_error(x_bits, K, c0, c1, fmt, coarse_ordering,
                         refine_ordering, a=a, b=b)
    err_safe = np.where(np.isfinite(err), err, np.inf)
    idx = int(np.argmax(err_safe))
    return float(err_safe[idx]), int(x_bits[idx])
