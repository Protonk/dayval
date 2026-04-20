"""LOW-BIT-FRGR-REFERENCE-PLAN drivers.

M1: format-intrinsic floors per (format, target function).
M2: exhaustive tier optima per (format, target function, tier).
M3: implementation-variant ablation on top of the best M2 tier.

Uses the Rust extension `dayval_rust` for the exhaustive sweeps. Python
orchestration is thin.

Every tier result records: K, coefficients (format-quantized), peak ε,
worst-case witness x*. Tie-set and near-optimal band require a secondary
pass and are computed for the best tier per format + function.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import minifloat as mf

try:
    import dayval_rust
    _HAVE_RUST = True
except ImportError:
    dayval_rust = None
    _HAVE_RUST = False


TARGETS = ("rsqrt", "recip")

RSQRT_TIERS = ("T0_monic", "T0_scale", "T1_monic", "T1_gen",
               "T2_monic_horner", "T2_gen_horner")
# Reciprocal shares the same tier names; the kernel dispatches based on target.
RECIP_TIERS = RSQRT_TIERS

TIER_OPS = {
    # (rsqrt, recip) op-count pairs per the plan's tier tables.
    "T0_monic": (0, 0),
    "T0_scale": (1, 1),
    "T1_monic": (3, 2),
    "T1_gen":   (4, 3),
    "T2_monic_horner": (5, 4),
    "T2_gen_horner":   (6, 5),
}

# rsqrt uses shift_then_sub coarse as canonical (Listing 1). Reciprocal uses
# no_shift (K - X) since there is no half-shift in the FRCP seed.
DEFAULT_COARSE = {"rsqrt": "shift_then_sub", "recip": "no_shift"}


@dataclass
class TierResult:
    target: str
    tier: str
    format_name: str
    K: int
    coefs_bits: tuple = ()            # format-quantized bit patterns of coefs
    coefs_values: tuple = ()          # corresponding float values
    eps: float = float("nan")
    x_star: int = 0
    ops: int = 0


@dataclass
class FormatFloor:
    target: str
    format_name: str
    eps_floor: float
    x_star: int


def format_floor(fmt: mf.Format, target: str) -> FormatFloor:
    if not _HAVE_RUST:
        raise RuntimeError("format_floor requires dayval_rust")
    x_bits = mf.positive_normals_bits(fmt)
    eps, xs = dayval_rust.format_floor(x_bits, fmt.E, fmt.M, fmt.bias, target)
    return FormatFloor(target=target, format_name=str(fmt),
                       eps_floor=float(eps), x_star=int(xs))


def _signed_positive_normals(fmt: mf.Format) -> np.ndarray:
    """Coefficient candidate set: every positive normal, both signs.

    The sign bit is the MSB of the format; flipping it negates the value.
    Returns f64 values (the Rust kernel re-quantizes internally)."""
    pn = mf.positive_normals_bits(fmt)
    pos_vals = mf.bits_to_float(pn, fmt)
    neg_vals = -pos_vals
    return np.concatenate([pos_vals, neg_vals]).astype(np.float64)


def tier_exhaustive(
    fmt: mf.Format, target: str, tier: str,
    coarse: Optional[str] = None,
    k_lo: Optional[int] = None, k_hi: Optional[int] = None,
    coef_candidates: Optional[np.ndarray] = None,
) -> TierResult:
    """Exhaustive (K × coef_grid) sweep. Returns the global optimum under
    the canonical coarse ordering for the target function."""
    if not _HAVE_RUST:
        raise RuntimeError("tier_exhaustive requires dayval_rust")
    coarse = coarse or DEFAULT_COARSE[target]
    if k_lo is None:
        k_lo = 0
    if k_hi is None:
        k_hi = fmt.all_mask + 1
    if coef_candidates is None:
        coef_candidates = _signed_positive_normals(fmt)
    x_bits = mf.positive_normals_bits(fmt)
    K, coefs, eps, xs = dayval_rust.tier_exhaustive(
        x_bits, k_lo, k_hi,
        coef_candidates.tolist(),
        fmt.E, fmt.M, fmt.bias,
        tier, coarse, target,
    )
    coefs_bits = tuple(int(mf.float_to_bits(np.float64(c), fmt)) for c in coefs)
    ops = TIER_OPS[tier][0 if target == "rsqrt" else 1]
    return TierResult(
        target=target, tier=tier, format_name=str(fmt),
        K=int(K) & fmt.all_mask,
        coefs_bits=coefs_bits,
        coefs_values=tuple(float(c) for c in coefs),
        eps=float(eps), x_star=int(xs), ops=ops,
    )


def tier_peak_single(
    fmt: mf.Format, target: str, tier: str, K: int, coefs: tuple,
    coarse: Optional[str] = None,
) -> tuple[float, int]:
    """Peak relative error for one (K, coefs) configuration."""
    coarse = coarse or DEFAULT_COARSE[target]
    x_bits = mf.positive_normals_bits(fmt)
    if _HAVE_RUST:
        eps, xs = dayval_rust.tier_peak(
            x_bits, K & fmt.all_mask,
            list(coefs),
            fmt.E, fmt.M, fmt.bias,
            tier, coarse, target,
        )
        return float(eps), int(xs)
    raise RuntimeError("tier_peak_single requires dayval_rust (for now)")
