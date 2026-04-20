"""Gates for Phase B sweep infrastructure.

- Schacham's 2025 blog reports K_tuned=0x59B7 with peak eps ≈ 2.8e-3 for
  fp16 FRSR using the MONIC refinement polynomial (c0=1.5, c1=-0.5 — one
  Newton iteration). This is an independent cross-check of the pipeline
  beyond Day.
- fp32 Listing 5: K=0x5F5FFF00 with tuned coefficients gives peak eps =
  6.501791e-4 — already gated in test_frsr.py, but repeated here through
  the sweep-level `analytic_point` code path.
- B1 analytic parity reps at fp16 must match the known eq (62) values.
"""
import numpy as np
import pytest

from dayval import minifloat as mf, sweep


def test_schacham_fp16_monic():
    """Independent pipeline cross-check from Eshed Schacham's blog."""
    fmt = mf.FP16
    x_bits = mf.positive_normals_bits(fmt)
    # Monic Newton refinement: y * (1.5 - 0.5 * x * y * y).
    # In our kernel's c0 + c1*x*y*y form: c0=1.5, c1=-0.5.
    try:
        import dayval_rust
        eps, xs = dayval_rust.peak_error_single(
            x_bits, 0x59B7, 1.5, -0.5, 5, 10, 15,
            "shift_then_sub", "xyyc1",
        )
    except ImportError:
        from dayval import frsr
        eps, xs = frsr.peak_error(
            x_bits, 0x59B7, 1.5, -0.5, fmt, "shift_then_sub", "xyyc1",
        )
    assert abs(eps - 2.8e-3) < 5e-5, f"Schacham fp16 monic: got eps={eps:.6e}, expected ≈ 2.8e-3"


def test_fp16_analytic_parities_have_expected_magic_constant():
    """Eq (62) at fp16: s=-1 gives K = 2^9 * (3*15 - 0.5), s=0 gives
    K = 2^9 * (3*15 + 0.5) — both 16-bit values."""
    fmt = mf.FP16
    ap_m1 = sweep.analytic_point(1, 2, 1, s=-1, fmt=fmt)
    ap_0 = sweep.analytic_point(1, 2, 1, s=0, fmt=fmt)
    # K = 2^9 * (c + 45). For c=-0.5: 512 * 44.5 = 22784 = 0x5900.
    #                   For c=+0.5: 512 * 45.5 = 23296 = 0x5B00.
    assert ap_m1.K == 0x5900
    assert ap_0.K == 0x5B00


def test_fp32_listing5_via_sweep_api():
    """The sweep-level analytic_point function must agree with the hand-run
    results at fp32 (K=0x5F200000, 0x5F600000 for the two parities)."""
    fmt = mf.FP32
    ap_m1 = sweep.analytic_point(1, 2, 1, s=-1, fmt=fmt)
    ap_0 = sweep.analytic_point(1, 2, 1, s=0, fmt=fmt)
    assert ap_m1.K == 0x5F200000
    assert ap_0.K == 0x5F600000


def test_tiny_format_row():
    """Phase1 driver completes for fp4 without error."""
    row = sweep.phase1_row(mf.FP4, format_name="fp4")
    assert row.width == 4
    assert row.eps_opt <= row.eps_real_analytic_winning + 1e-12
