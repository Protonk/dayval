"""Phase B data collection (validation arm): B1 analytic family, B2 optima,
B3 ablation, B4 witnesses. See FRGR-PLAN.md for the primary-table schema.

Design:
- Every sweep function takes a `Format` and returns a list of `RowPartial`
  records. Records accumulate monotonically (B1 → B2 → B3) with ablation
  columns filled in as the ladder proceeds.
- fp4/fp6/fp8 route through `lowbit.tier_exhaustive` (Rust-parallelised) for
  the full (K, c0_bits, c1_bits) enumeration — true global optimum per plan
  §B2. `eps_opt_kind` on the SweepRow is tagged "exhaustive" for these rows.
- fp16..fp24 go through `k_only_plus_local`: one K-only pass per parity with
  coefficients pinned at the format-quantised analytic values, plus an 8-seed
  joint local search per parity. The K-only pass uses the Rust kernel
  `k_sweep`; local search is Python (few hundred steps, not a bottleneck).
  Tagged "local" — results are not globally optimal.
- fp32 is a replication check: evaluate Day Listing 5 at (K=0x5F5FFF00,
  c0=1.1893165, c1=-0.24889956) and assert eps == 6.501791e-4. Tagged
  "replication" — the stored eps_opt is the paper's tuned value, not a
  pipeline-derived optimum.

Witness records (B4) are attached to every optimum stored: worst-case x,
exact minimizer set (K's tying the best peak eps), near-optimal band (K's
within 1 ULP of the best peak eps), and second-worst relative-error input.

Memory: `k_sweep` returns three arrays of length (k_hi - k_lo). At fp24 with
2^24 K, that's 16M * (4+8+4) = 256 MB — fits comfortably in the existing-sim
environment. For fp20 it's 16 MB.

Determinism: per-format RNG seed is derived from format width + bias so the
8-seed joint local search is repeatable.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional

import mpmath as mp
import numpy as np

from . import algorithm3, frsr, magic
from . import minifloat as mf

try:
    import dayval_rust
    _HAVE_RUST = True
except ImportError:
    dayval_rust = None
    _HAVE_RUST = False


@dataclass
class AnalyticPoint:
    """B1: format-quantised analytic representative for one parity class."""
    s: int                 # parity parameter (-1 or 0 for FRSR)
    c: float               # exact-arithmetic c
    c0_exact: float
    c1_exact: float
    K: int                 # format-quantised magic constant
    c0_bits: int           # format-quantised c0 bit pattern
    c1_bits: int           # format-quantised c1 bit pattern
    eps_theory: float
    eps_realized: float    # peak eps of the quantised triple under shift_then_sub
    x_star: int            # witness


@dataclass
class WitnessRecord:
    """B4: degeneracy structure at a stored optimum."""
    x_star: int
    exact_minimizer_count: int     # # K values achieving this exact peak eps
    near_optimal_band_count: int   # # K values whose peak eps is within 1 ULP
    second_worst_x: int
    second_worst_eps: float


@dataclass
class SweepRow:
    """One primary-table row, keyed on (E, M, bias). Filled in across
    B1 -> B2 -> B3 -> B4."""
    format_name: str
    E: int
    M: int
    bias: int
    width: int
    positive_normals: int
    K_s_minus_one: int = 0
    K_s_zero: int = 0
    K_opt: int = 0
    winning_s: int = 0
    eps_theory: float = float("nan")
    eps_real_analytic_winning: float = float("nan")
    eps_real_plus_Cprime: float = float("nan")
    eps_real_plus_orderings: float = float("nan")
    eps_real_plus_coef_tune: float = float("nan")
    eps_opt: float = float("nan")
    eps_opt_kind: str = ""         # "exhaustive" | "local" | "replication"
    tie_set_size: int = 0
    near_optimal_band_count: int = 0
    second_worst_x: int = 0
    second_worst_eps: float = 0.0
    x_star_opt: int = 0
    notes: list = field(default_factory=list)


def _signed_finite_encodings(fmt: mf.Format) -> np.ndarray:
    """Every finite representable value (normals + subnormals + zero) as
    f64, both signs. Excludes only inf and NaN. Used by the validation-arm
    exhaustive path at tiny formats so the candidate set matches what the
    fp16+ joint_local_search effectively searches via bit-ULP neighborhood
    steps. Reference-arm spec sheets keep the plan's strict positive-
    normal-only set via `lowbit._signed_positive_normals`."""
    exp_max_biased = (1 << fmt.E) - 1  # all-ones exp → inf/NaN
    all_bits = np.arange(1 << fmt.width, dtype=np.uint32)
    biased_e = (all_bits >> fmt.M) & ((1 << fmt.E) - 1)
    finite = all_bits[biased_e != exp_max_biased]
    return mf.bits_to_float(finite, fmt).astype(np.float64)


def _format_seed(fmt: mf.Format) -> int:
    """Deterministic per-format RNG seed."""
    h = hashlib.blake2s(
        f"{fmt.E}-{fmt.M}-{fmt.bias}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(h, "little")


def _precision_dps(fmt: mf.Format) -> int:
    """Working precision for Algorithm 3 — generous margin above format M."""
    return max(50, 4 * fmt.M)


def analytic_point(a: int, b: int, n: int, s: int, fmt: mf.Format) -> AnalyticPoint:
    """B1 entry. Returns the quantised magic constant + coefficients and the
    peak relative error these attain under the shift-then-sub, c1x_yy
    baseline kernel."""
    with mp.workdps(_precision_dps(fmt)):
        r = algorithm3.run(a=a, b=b, n=n, s=s)
        C_int = magic.magic_C_int(a=a, b=b, M=fmt.M, bias=fmt.bias, c=r.c)
        # Format-quantise the coefficients — round to nearest representable.
        c0q_bits = int(mf.float_to_bits(np.float64(float(r.coeffs[0])), fmt))
        c1q_bits = int(mf.float_to_bits(np.float64(float(r.coeffs[1])), fmt))
        c0q = float(mf.bits_to_float(np.uint32(c0q_bits), fmt))
        c1q = float(mf.bits_to_float(np.uint32(c1q_bits), fmt))
        eps_theory = float(r.eps_theory)

    x_bits = mf.positive_normals_bits(fmt)
    if _HAVE_RUST:
        eps, xs = dayval_rust.peak_error_single(
            x_bits, C_int & fmt.all_mask, c0q, c1q,
            fmt.E, fmt.M, fmt.bias,
            "shift_then_sub", "xyyc1",
        )
    else:
        eps, xs = frsr.peak_error(
            x_bits, C_int & fmt.all_mask, c0q, c1q, fmt,
            "shift_then_sub", "xyyc1",
        )

    return AnalyticPoint(
        s=s, c=float(r.c),
        c0_exact=float(r.coeffs[0]), c1_exact=float(r.coeffs[1]),
        K=C_int & fmt.all_mask,
        c0_bits=c0q_bits, c1_bits=c1q_bits,
        eps_theory=eps_theory, eps_realized=eps, x_star=int(xs),
    )


def k_only_sweep(fmt: mf.Format, c0q: float, c1q: float,
                 coarse_ordering: str = "shift_then_sub",
                 refine_ordering: str = "xyyc1",
                 k_lo: int = 0, k_hi: Optional[int] = None,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scan all K in [k_lo, k_hi) with coefficients pinned. Returns
    (K_array, eps_array, x_star_array). Requires the Rust kernel for
    formats at fp20+ (Python path would take hours)."""
    if k_hi is None:
        k_hi = fmt.all_mask + 1
    if not _HAVE_RUST:
        raise RuntimeError("k_only_sweep requires the compiled dayval_rust "
                           "extension for tractable throughput")
    x_bits = mf.positive_normals_bits(fmt)
    return dayval_rust.k_sweep(
        x_bits, k_lo, k_hi, c0q, c1q,
        fmt.E, fmt.M, fmt.bias,
        coarse_ordering, refine_ordering,
    )


def _ulp_at(v: float) -> float:
    """Approximate absolute ULP at positive value v under float64."""
    if v <= 0 or not np.isfinite(v):
        return float("inf")
    return np.nextafter(v, v + 1) - v


def witness(eps_arr: np.ndarray, ks: np.ndarray, xs: np.ndarray
            ) -> tuple[int, WitnessRecord]:
    """B4: from a K-sweep result, extract the optimal K plus degeneracy
    structure."""
    # Ignore non-finite (overflow / NaN) configurations.
    finite = np.isfinite(eps_arr)
    if not finite.any():
        raise ValueError("no finite-epsilon K in sweep — entire format overflowed")
    best = eps_arr[finite].min()
    best_mask = finite & (eps_arr == best)
    best_idx = np.flatnonzero(best_mask)
    k_opt = int(ks[best_idx[0]])
    x_star = int(xs[best_idx[0]])

    # Near-optimal band: within 1 ULP of the best eps.
    tol = _ulp_at(best)
    near_mask = finite & (eps_arr <= best + tol)

    # Second-worst x over sweeps at the best K: requires re-running at best K.
    # Skip for now — the K-sweep captured only one witness per K.
    return k_opt, WitnessRecord(
        x_star=x_star,
        exact_minimizer_count=int(best_mask.sum()),
        near_optimal_band_count=int(near_mask.sum()),
        second_worst_x=0,       # populated by a follow-up single-K call
        second_worst_eps=0.0,
    )


def joint_local_search(fmt: mf.Format, seed_K: int, seed_c0_bits: int,
                       seed_c1_bits: int,
                       coarse_ordering: str = "shift_then_sub",
                       refine_ordering: str = "xyyc1",
                       rng: Optional[np.random.Generator] = None,
                       extra_seeds: int = 8,
                       box_radius_exp: Optional[int] = None,
                       max_no_improve: int = 32,
                       ) -> tuple[int, int, int, float, int]:
    """Local descent over (K, c0_bits, c1_bits) starting from the seed plus
    `extra_seeds` sampled uniformly in a box of radius 2^ceil(M/2) ULPs.
    Best step is chosen by steepest improvement; ties broken
    lexicographically. Stop after `max_no_improve` consecutive non-improvers.

    Returns (best_K, best_c0_bits, best_c1_bits, best_eps, best_x_star).
    """
    if rng is None:
        rng = np.random.default_rng(_format_seed(fmt))
    if box_radius_exp is None:
        box_radius_exp = (fmt.M + 1) // 2
    box = 1 << box_radius_exp

    x_bits = mf.positive_normals_bits(fmt)
    mask = fmt.all_mask

    def eval_point(k: int, c0b: int, c1b: int) -> tuple[float, int]:
        c0 = float(mf.bits_to_float(np.uint32(c0b & mask), fmt))
        c1 = float(mf.bits_to_float(np.uint32(c1b & mask), fmt))
        if _HAVE_RUST:
            eps, xs = dayval_rust.peak_error_single(
                x_bits, k & mask, c0, c1,
                fmt.E, fmt.M, fmt.bias,
                coarse_ordering, refine_ordering,
            )
        else:
            eps, xs = frsr.peak_error(
                x_bits, k & mask, c0, c1, fmt,
                coarse_ordering, refine_ordering,
            )
        return float(eps), int(xs)

    seeds = [(seed_K, seed_c0_bits, seed_c1_bits)]
    for _ in range(extra_seeds):
        dk = int(rng.integers(-box, box + 1))
        dc0 = int(rng.integers(-box, box + 1))
        dc1 = int(rng.integers(-box, box + 1))
        seeds.append((
            (seed_K + dk) & mask,
            (seed_c0_bits + dc0) & mask,
            (seed_c1_bits + dc1) & mask,
        ))

    best = None
    for k, c0b, c1b in seeds:
        eps, xs = eval_point(k, c0b, c1b)
        k_cur, c0_cur, c1_cur = k, c0b, c1b
        eps_cur, xs_cur = eps, xs
        no_improve = 0
        while no_improve < max_no_improve:
            # 6 neighbours: +/- 1 ULP in each coordinate.
            best_step = None
            for dk, dc0, dc1 in [(1, 0, 0), (-1, 0, 0),
                                 (0, 1, 0), (0, -1, 0),
                                 (0, 0, 1), (0, 0, -1)]:
                nk = (k_cur + dk) & mask
                nc0 = (c0_cur + dc0) & mask
                nc1 = (c1_cur + dc1) & mask
                e, xs2 = eval_point(nk, nc0, nc1)
                if e < eps_cur - 1e-18 and (best_step is None or
                                            (e, dk, dc0, dc1) < best_step[0]):
                    best_step = ((e, dk, dc0, dc1), nk, nc0, nc1, xs2)
            if best_step is None:
                no_improve += 1
                # No improvement — since every neighbour was evaluated and none
                # beat current, the search from this seed has converged.
                break
            (e_new, _, _, _), nk, nc0, nc1, xs_new = best_step
            k_cur, c0_cur, c1_cur = nk, nc0, nc1
            eps_cur, xs_cur = e_new, xs_new
            no_improve = 0
        if best is None or eps_cur < best[3]:
            best = (k_cur, c0_cur, c1_cur, eps_cur, xs_cur)
    return best  # type: ignore[return-value]


def exhaustive_tiny(fmt: mf.Format,
                    coarse_ordering: str = "shift_then_sub",
                    refine_ordering: str = "xyyc1",
                    ) -> tuple[int, int, int, float, int]:
    """Full (K, c0_bits, c1_bits) enumeration over positive-normal coefficient
    candidates. Tractable only for fp4/6/8."""
    pn = mf.positive_normals_bits(fmt)
    n_k = fmt.all_mask + 1
    n_coef = len(pn)
    total = n_k * n_coef * n_coef
    if total > 10_000_000:
        raise ValueError(f"{fmt} has {total:,} configurations — too large for "
                         "exhaustive_tiny")

    x_bits = pn
    best_eps = float("inf")
    best = (0, 0, 0, best_eps, 0)
    for K in range(n_k):
        for c0b in pn:
            for c1b in pn:
                c0 = float(mf.bits_to_float(np.uint32(c0b), fmt))
                # c1 is negative in Day's formulation — flip sign for the
                # kernel contract (refine computes c0 + c1*x*y*y).
                c1 = -float(mf.bits_to_float(np.uint32(c1b), fmt))
                if _HAVE_RUST:
                    eps, xs = dayval_rust.peak_error_single(
                        x_bits, K, c0, c1,
                        fmt.E, fmt.M, fmt.bias,
                        coarse_ordering, refine_ordering,
                    )
                else:
                    eps, xs = frsr.peak_error(
                        x_bits, K, c0, c1, fmt,
                        coarse_ordering, refine_ordering,
                    )
                if eps < best_eps:
                    best_eps = eps
                    best = (K, int(c0b), int(c1b), eps, int(xs))
    return best


def phase1_row(fmt: mf.Format, *, format_name: str = "") -> SweepRow:
    """Assemble one Phase-1 row by running B1, B2, B3 in order."""
    name = format_name or str(fmt)
    pn = len(mf.positive_normals_bits(fmt))
    row = SweepRow(
        format_name=name,
        E=fmt.E, M=fmt.M, bias=fmt.bias, width=fmt.width,
        positive_normals=pn,
    )

    # B1: analytic family.
    a1m = analytic_point(1, 2, 1, s=-1, fmt=fmt)
    a0 = analytic_point(1, 2, 1, s=0, fmt=fmt)
    row.K_s_minus_one = a1m.K
    row.K_s_zero = a0.K
    row.eps_theory = a1m.eps_theory
    if a1m.eps_realized <= a0.eps_realized:
        winning, other = a1m, a0
        row.winning_s = -1
    else:
        winning, other = a0, a1m
        row.winning_s = 0
    row.eps_real_analytic_winning = winning.eps_realized
    row.notes.append(
        f"analytic s=-1: K=0x{a1m.K:x} eps={a1m.eps_realized:.6e}; "
        f"analytic s= 0: K=0x{a0.K:x} eps={a0.eps_realized:.6e}"
    )

    # B3 ladder step 2: add C' extra bit (sub-then-shift with 2K or 2K+1).
    best_cp = winning.eps_realized
    for coarse in ("sub_then_shift_2K", "sub_then_shift_2K1"):
        c0 = float(mf.bits_to_float(np.uint32(winning.c0_bits), fmt))
        c1 = float(mf.bits_to_float(np.uint32(winning.c1_bits), fmt))
        if _HAVE_RUST:
            eps, _ = dayval_rust.peak_error_single(
                mf.positive_normals_bits(fmt), winning.K, c0, c1,
                fmt.E, fmt.M, fmt.bias, coarse, "xyyc1",
            )
        else:
            eps, _ = frsr.peak_error(
                mf.positive_normals_bits(fmt), winning.K, c0, c1, fmt,
                coarse, "xyyc1",
            )
        if eps < best_cp:
            best_cp = eps
    row.eps_real_plus_Cprime = best_cp

    # B3 ladder step 3: best of the 9 refinement orderings under the
    # Newton-style refinement form with the (now-possibly-improved)
    # coarse ordering. Pick the coarse that won step 2.
    best_ord = best_cp
    x_bits_full = mf.positive_normals_bits(fmt)
    for refine in frsr.REFINE_ORDERINGS:
        for coarse in ("shift_then_sub", "sub_then_shift_2K", "sub_then_shift_2K1"):
            c0 = float(mf.bits_to_float(np.uint32(winning.c0_bits), fmt))
            c1 = float(mf.bits_to_float(np.uint32(winning.c1_bits), fmt))
            if _HAVE_RUST:
                eps, _ = dayval_rust.peak_error_single(
                    x_bits_full, winning.K, c0, c1,
                    fmt.E, fmt.M, fmt.bias, coarse, refine,
                )
            else:
                eps, _ = frsr.peak_error(
                    x_bits_full, winning.K, c0, c1, fmt, coarse, refine,
                )
            if eps < best_ord:
                best_ord = eps
    row.eps_real_plus_orderings = best_ord

    # B2 / B3 ladder step 4: eps_opt depends on format width per plan §B2.
    #  - fp4/fp6/fp8: exhaustive (K, c0, c1) via lowbit.tier_exhaustive.
    #    Validation arm expands the plan's "positive-normal encodings"
    #    candidate set to "all finite encodings" (normals + subnormals +
    #    zero, both signs). This matches what joint_local_search reaches
    #    at fp16+ via bit-ULP stepping and gives a consistent eps_opt
    #    semantics across widths. Reference-arm keeps the strict set.
    #  - fp16+: joint local search from the analytic seed, labeled "local".
    if fmt.width <= 8 and _HAVE_RUST:
        from . import lowbit
        coef_candidates = _signed_finite_encodings(fmt)
        r = lowbit.tier_exhaustive(
            fmt, "rsqrt", "T1_gen",
            coef_candidates=coef_candidates,
        )
        k_opt = r.K
        c0_opt_bits, c1_opt_bits = r.coefs_bits
        eps_opt = r.eps
        xstar = r.x_star
        row.eps_opt_kind = "exhaustive"
    else:
        k_opt, c0_opt_bits, c1_opt_bits, eps_opt, xstar = joint_local_search(
            fmt,
            seed_K=winning.K,
            seed_c0_bits=winning.c0_bits,
            seed_c1_bits=winning.c1_bits,
        )
        row.eps_opt_kind = "local"
    row.eps_real_plus_coef_tune = eps_opt
    row.K_opt = k_opt
    row.eps_opt = eps_opt
    row.x_star_opt = xstar

    # B4: tie-set size, near-optimal band, second-worst witness. K-sweep
    # pins (c0, c1) at the step-4 optimum; tie / near-band are conditional
    # on those coefficients (documented in the CSV via eps_opt_kind).
    if fmt.width <= 20 and _HAVE_RUST:
        c0 = float(mf.bits_to_float(np.uint32(c0_opt_bits), fmt))
        c1 = float(mf.bits_to_float(np.uint32(c1_opt_bits), fmt))
        ks, eps_arr, xs_arr = k_only_sweep(fmt, c0, c1)
        try:
            _, wr = witness(eps_arr, ks, xs_arr)
            row.tie_set_size = wr.exact_minimizer_count
            row.near_optimal_band_count = wr.near_optimal_band_count
        except ValueError:
            pass
        # Second-worst: re-evaluate per-input error at K_opt (one pass over
        # positive normals) and pick rank 2.
        x_bits_all = mf.positive_normals_bits(fmt)
        err = frsr.relative_error(
            x_bits_all, k_opt, c0, c1, fmt, "shift_then_sub", "xyyc1",
        )
        err_safe = np.where(np.isfinite(err), err, -np.inf)
        if err_safe.size >= 2:
            order = np.argsort(-err_safe)
            row.second_worst_x = int(x_bits_all[order[1]])
            row.second_worst_eps = float(err_safe[order[1]])

    return row
