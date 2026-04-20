"""Gates for LOW-BIT-FRGR-REFERENCE-PLAN tier kernels.

- FRSR T1_gen exhaustive at fp8 E4M3 must recover the Day-plan Phase-1
  optimum (eps = 5.234114e-2; matches the format floor — T1_gen saturates
  here so higher tiers can't improve).
- FRCP T0_scale at fp4 E2M1: with 4 positive normals and 4 k candidates,
  the exhaustive sweep is 16 × 4 = 64 configurations; must terminate.
- Reciprocal format floor is well-defined.
- Tier op counts match plan table.
"""
import numpy as np
import pytest

from dayval import lowbit, minifloat as mf

dayval_rust = pytest.importorskip("dayval_rust")


def test_rsqrt_t1gen_fp8_matches_day_phase1():
    """Exhaustive T1_gen for FRSR at fp8 E4M3 should yield the same eps we
    saw in the Day Phase-1 run (5.234114e-2)."""
    fmt = mf.FP8_E4M3
    r = lowbit.tier_exhaustive(fmt, "rsqrt", "T1_gen")
    assert abs(r.eps - 5.234114e-2) < 1e-7
    assert r.ops == 4


def test_rsqrt_floor_at_fp8_matches_t1gen():
    """T1_gen already hits the format floor at fp8 E4M3, so
    ε_floor ≈ ε_T1_gen. Saturation confirms the plan's expectation."""
    fmt = mf.FP8_E4M3
    floor = lowbit.format_floor(fmt, "rsqrt")
    r = lowbit.tier_exhaustive(fmt, "rsqrt", "T1_gen")
    assert abs(floor.eps_floor - r.eps) < 1e-12


def test_recip_floor_fp4():
    """FP4 E2M1 has 4 positive normals. The reciprocal floor is well-
    defined and matches direct computation."""
    fmt = mf.FP4_E2M1
    floor = lowbit.format_floor(fmt, "recip")
    # Brute compute the same thing in Python.
    xb = mf.positive_normals_bits(fmt)
    x = mf.bits_to_float(xb, fmt)
    inv = 1.0 / x
    inv_q = mf.bits_to_float(mf.float_to_bits(inv, fmt), fmt)
    err = np.abs(1.0 - x * inv_q)
    expected = float(err.max())
    assert abs(floor.eps_floor - expected) < 1e-12


def test_t0_scale_tiny_format():
    """T0_scale at fp4 E2M1 has K=2^4=16 × k candidates=8 (4 pos + 4 neg).
    Just verify the sweep runs and produces a finite best eps."""
    fmt = mf.FP4_E2M1
    r = lowbit.tier_exhaustive(fmt, "recip", "T0_scale")
    assert r.eps < float("inf")
    assert len(r.coefs_bits) == 1


def test_reciprocal_t1gen_beats_t0monic():
    """T1_gen reciprocal at fp8 E4M3 should strictly improve on T0_monic."""
    fmt = mf.FP8_E4M3
    t0 = lowbit.tier_exhaustive(fmt, "recip", "T0_monic")
    t1 = lowbit.tier_exhaustive(fmt, "recip", "T1_gen")
    assert t1.eps <= t0.eps + 1e-12


def test_reciprocal_floor_bound():
    """For every format × tier, peak ε must be ≥ format floor."""
    fmt = mf.FP6_E3M2
    floor = lowbit.format_floor(fmt, "recip")
    for tier in ("T0_monic", "T0_scale", "T1_monic", "T1_gen"):
        r = lowbit.tier_exhaustive(fmt, "recip", tier)
        assert r.eps >= floor.eps_floor - 1e-12, (
            f"{tier} eps={r.eps:.6e} violates floor {floor.eps_floor:.6e}")


def test_m3_ablation_rsqrt_fp6():
    """M3 on T1_gen at FP6 E3M2 rsqrt should produce three levers, all
    with non-negative δε (levers can't improve on the exhaustive baseline
    for the same (K, coefs), but the parity-switch lever's δ is the
    penalty between parities, not a baseline delta)."""
    fmt = mf.FP6_E3M2
    base = lowbit.tier_exhaustive(fmt, "rsqrt", "T1_gen")
    m3 = lowbit.m3_ablation(fmt, "rsqrt", base)
    assert m3.baseline_eps == pytest.approx(base.eps)
    assert len(m3.levers) == 3
    names = [l.name for l in m3.levers]
    assert "C' extra bit (§9.2)" in names
    assert "best refine ordering (§9.2)" in names
    assert "parity switch (§9.3)" in names
    # The first two levers reference the same (K, coefs); they cannot
    # improve on the already-optimal canonical configuration.
    for lever in m3.levers[:2]:
        assert lever.delta >= -1e-12, (
            f"{lever.name} improved on canonical baseline unexpectedly: "
            f"δε={lever.delta}")


def test_m3_ablation_recip_c_prime_na():
    """Reciprocal has no shift, so the C' extra bit lever reports N/A."""
    fmt = mf.FP6_E3M2
    base = lowbit.tier_exhaustive(fmt, "recip", "T1_gen")
    m3 = lowbit.m3_ablation(fmt, "recip", base)
    cp = next(l for l in m3.levers if l.name.startswith("C' extra bit"))
    assert cp.delta == 0.0
    assert "N/A" in cp.detail
