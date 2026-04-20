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


# ---------------------------------------------------------------------------
# M3 implementation-variant ablation (Day §9 levers).
#
# Applied to the chosen tier's best (K, coefs). Each lever is measured as a
# Δε from the baseline (canonical coarse + canonical refine ordering).
# ---------------------------------------------------------------------------

# The "canonical" T1_gen refine ordering per the LOW-BIT plan's assumption:
# xyyc1 matches Day Listing 5 at rsqrt; c1xy is the one-fewer-factor
# analogue for reciprocal (there is no y² in the reciprocal refinement).
CANONICAL_REFINE = {"rsqrt": "xyyc1", "recip": "c1xyy"}

# The 9 refine orderings that apply to rsqrt (3 factors: x, y, y in the
# c1 product). For reciprocal there are only 3! = 6 permutations of
# (c1, x, y), but my kernel uses the same ordering names — only the ones
# whose product-as-written makes sense for z=x*y matter. For reciprocal
# I run a subset that differs only in left-to-right associativity.
REFINE_ORDERINGS_RSQRT = (
    "c1xyy", "c1yxy", "c1yyx",
    "xyc1y", "xyyc1", "yyc1x", "yyxc1",
    "c1x_yy", "c1y_xy",
)
# For reciprocal, the "y²" factor collapses to "y"; the 9-way enumeration
# partially collapses (several orderings become equivalent). We still try
# all 9 — at T1_gen with z=x*y the kernel just treats one y as a
# multiplicand. Effective distinct outcomes may number fewer than 9.
REFINE_ORDERINGS_RECIP = REFINE_ORDERINGS_RSQRT


@dataclass
class M3Row:
    """One lever's contribution, relative to the M3 baseline."""
    name: str
    eps: float
    delta: float          # eps - baseline
    detail: str = ""


@dataclass
class M3Result:
    target: str
    tier: str
    baseline_K: int
    baseline_coefs: tuple
    baseline_eps: float
    baseline_coarse: str
    baseline_refine: str
    levers: list = field(default_factory=list)


def _eval_with(fmt, target, tier, K, coefs, coarse, refine):
    """Evaluate tier_peak, but through refine_kernel path when we need a
    non-canonical refine ordering. For T1_gen, refine ordering is part of
    the polynomial evaluation; tier_peak uses a canonical ordering per
    target, so for ordering sweeps we need to fall back to the FRSR
    refine (only applicable for T1_gen / rsqrt). Reciprocal refinement
    orderings would require a similar enumerator in Rust; current
    implementation uses only the canonical ordering for recip. Returns
    None to signal 'not applicable' when the kernel can't evaluate the
    requested ordering."""
    from . import frsr  # circular-avoid import
    if tier != "T1_gen":
        return None
    x_bits = mf.positive_normals_bits(fmt)
    if target == "rsqrt":
        c0 = float(mf.bits_to_float(np.uint32(int(mf.float_to_bits(np.float64(coefs[0]), fmt))), fmt))
        c1 = float(mf.bits_to_float(np.uint32(int(mf.float_to_bits(np.float64(coefs[1]), fmt))), fmt))
        if _HAVE_RUST:
            eps, xs = dayval_rust.peak_error_single(
                x_bits, K & fmt.all_mask, c0, c1,
                fmt.E, fmt.M, fmt.bias, coarse, refine,
            )
        else:
            eps, xs = frsr.peak_error(
                x_bits, K & fmt.all_mask, c0, c1, fmt, coarse, refine,
            )
        return float(eps)
    return None


def m3_ablation(
    fmt: mf.Format, target: str, base: TierResult,
) -> M3Result:
    """Measure Day §9 levers applied to the canonical baseline.

    Baseline = T1_gen best with canonical coarse and canonical refine
    ordering (xyyc1 for rsqrt, c1xyy for recip). Levers reported (each
    Δε is relative to baseline, negative = improvement):

      - §9.2 C' extra bit (rsqrt only): sub_then_shift_2K, _2K+1 at same
        (K, coefs).
      - §9.2 best alternative ordering: scan the 9 simple orderings at
        the baseline (K, coefs); report the best Δε.
      - §9.3 parity switch: evaluate the quantized analytic parity-
        representative K+coefs for the NON-optimal parity and report
        Δε from baseline. Answers "how much worse would picking the
        wrong parity make us?"
    """
    if base.tier != "T1_gen":
        # M3 is defined for T1_gen in the plan. Return empty result for
        # other tiers.
        return M3Result(target=target, tier=base.tier,
                        baseline_K=base.K, baseline_coefs=base.coefs_values,
                        baseline_eps=base.eps,
                        baseline_coarse=DEFAULT_COARSE[target],
                        baseline_refine=CANONICAL_REFINE[target])

    coarse_canonical = DEFAULT_COARSE[target]
    refine_canonical = CANONICAL_REFINE[target]

    result = M3Result(
        target=target, tier=base.tier,
        baseline_K=base.K, baseline_coefs=base.coefs_values,
        baseline_eps=base.eps,
        baseline_coarse=coarse_canonical,
        baseline_refine=refine_canonical,
    )

    # Lever 1: C' extra bit (rsqrt only).
    if target == "rsqrt":
        best_cp_eps = base.eps
        best_cp_name = "shift_then_sub"
        for coarse in ("sub_then_shift_2K", "sub_then_shift_2K1"):
            eps = _eval_with(fmt, target, base.tier, base.K,
                             base.coefs_values, coarse, refine_canonical)
            if eps is not None and eps < best_cp_eps:
                best_cp_eps = eps
                best_cp_name = coarse
        result.levers.append(M3Row(
            name="C' extra bit (§9.2)",
            eps=best_cp_eps,
            delta=best_cp_eps - base.eps,
            detail=f"best coarse = {best_cp_name}",
        ))
    else:
        result.levers.append(M3Row(
            name="C' extra bit (§9.2)",
            eps=base.eps, delta=0.0,
            detail="N/A for reciprocal (no shift)",
        ))

    # Lever 2: best alternative refine ordering.
    orderings = (REFINE_ORDERINGS_RSQRT if target == "rsqrt"
                 else REFINE_ORDERINGS_RECIP)
    best_ord_eps = base.eps
    best_ord_name = refine_canonical
    for ordering in orderings:
        if ordering == refine_canonical:
            continue
        eps = _eval_with(fmt, target, base.tier, base.K,
                         base.coefs_values, coarse_canonical, ordering)
        if eps is not None and eps < best_ord_eps:
            best_ord_eps = eps
            best_ord_name = ordering
    result.levers.append(M3Row(
        name="best refine ordering (§9.2)",
        eps=best_ord_eps,
        delta=best_ord_eps - base.eps,
        detail=f"best = {best_ord_name}",
    ))

    # Lever 3: parity switch. Compute both analytic parity reps;
    # evaluate each at canonical coarse/refine with format-quantized
    # analytic coefficients; report the Δε of the non-winner relative to
    # the baseline. The baseline may not use the winning-parity K (it's
    # exhaustive), so the "parity switch" value here is "what does the
    # wrong-parity analytic give you" — a lower bound on damage from
    # picking the wrong s.
    from . import algorithm3, magic
    import mpmath as mp
    with mp.workdps(max(50, 4 * fmt.M)):
        a_num, b_den = (1, 2) if target == "rsqrt" else (1, 1)
        eps_by_parity = {}
        for s in (-1, 0):
            r = algorithm3.run(a=a_num, b=b_den, n=1, s=s)
            K = magic.magic_C_int(a=a_num, b=b_den, M=fmt.M,
                                  bias=fmt.bias, c=r.c) & fmt.all_mask
            c0q = float(mf.bits_to_float(np.uint32(int(
                mf.float_to_bits(np.float64(float(r.coeffs[0])), fmt))), fmt))
            c1q = float(mf.bits_to_float(np.uint32(int(
                mf.float_to_bits(np.float64(float(r.coeffs[1])), fmt))), fmt))
            eps = _eval_with(fmt, target, "T1_gen", K,
                             (c0q, c1q), coarse_canonical, refine_canonical)
            eps_by_parity[s] = (float(eps) if eps is not None
                                 else float("inf"))
    s_best = min(eps_by_parity, key=lambda s: eps_by_parity[s])
    s_other = 0 if s_best == -1 else -1
    parity_penalty = eps_by_parity[s_other] - eps_by_parity[s_best]
    result.levers.append(M3Row(
        name="parity switch (§9.3)",
        eps=eps_by_parity[s_best],
        delta=parity_penalty,
        detail=(f"s_best={s_best} eps={eps_by_parity[s_best]:.3e}, "
                f"s_other={s_other} eps={eps_by_parity[s_other]:.3e}; "
                f"Δε is the penalty for choosing the wrong parity"),
    ))

    return result
