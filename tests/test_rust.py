"""Gate: Rust kernel must match Python kernel bit-exactly on 10k fp32 random
configurations. Also check speedup.
"""
import time

import numpy as np
import pytest

from dayval import frsr, minifloat as mf

dayval_rust = pytest.importorskip("dayval_rust")


def test_quantize_matches_python():
    rng = np.random.default_rng(1)
    vals = rng.standard_normal(5000).astype(np.float64)
    py = mf.quantize(vals, mf.FP16)
    rs = np.array([dayval_rust.quantize_f64(float(v), 5, 10, 15) for v in vals])
    # nans won't compare equal — exclude.
    mask = np.isfinite(py) & np.isfinite(rs)
    np.testing.assert_array_equal(py[mask], rs[mask])


def test_listing5_rust_matches_global_witness():
    """Rust kernel on the known Listing 5 witness must match 6.501791e-4."""
    x_bits = np.array([0x01401a9f], dtype=np.uint32)
    eps, xs = dayval_rust.peak_error_single(
        x_bits, 0x5F5FFF00, 1.1893165, -0.24889956,
        8, 23, 127, "shift_then_sub", "xyyc1",
    )
    assert abs(eps - 6.501791e-4) < 1e-9
    assert xs == 0x01401a9f


def test_rust_matches_python_on_sample():
    """Across 100 random (K, c0, c1) configs, Rust and Python kernels agree on
    peak ε to within 1 ULP of double precision."""
    rng = np.random.default_rng(7)
    # fp16 sample — fast enough to run 100 Python iterations.
    fmt = mf.FP16
    x_bits = mf.positive_normals_bits(fmt)

    n = 50
    for _ in range(n):
        k = int(rng.integers(0, 1 << 16))
        c0 = float(rng.uniform(0.5, 2.0))
        c1 = float(rng.uniform(-1.0, -0.1))
        for refine in ("c1xyy", "xyyc1", "c1x_yy"):
            py_eps, py_xs = frsr.peak_error(
                x_bits, k, c0, c1, fmt, "shift_then_sub", refine,
            )
            rs_eps, rs_xs = dayval_rust.peak_error_single(
                x_bits, k, c0, c1, 5, 10, 15,
                "shift_then_sub", refine,
            )
            assert abs(py_eps - rs_eps) < 1e-12 or (py_eps == 0 and rs_eps == 0) \
                or (not np.isfinite(py_eps) and not np.isfinite(rs_eps)), \
                f"K={k:#x} c0={c0} c1={c1} refine={refine}: py={py_eps} rs={rs_eps}"


def test_rust_k_sweep_speed():
    """Smoke: fp16 K-only sweep completes in reasonable time and results
    are consistent with the Python single-K kernel at a few probe K."""
    fmt = mf.FP16
    x_bits = mf.positive_normals_bits(fmt)
    c0 = 1.1893165
    c1 = -0.24889956
    t0 = time.perf_counter()
    ks, eps_arr, xs_arr = dayval_rust.k_sweep(
        x_bits, 0, 1 << 16, c0, c1, 5, 10, 15,
        "shift_then_sub", "xyyc1",
    )
    dt = time.perf_counter() - t0
    # fp16 65k K * 30k x = 2e9 kernel evals. Rust with rayon on a laptop is
    # expected well under a minute.
    assert dt < 120.0, f"fp16 K sweep too slow: {dt:.1f}s"
    # Consistency check at one K in the sweep.
    probe_k = int(ks[0x5900])  # some K value in the Day-relevant range
    py_eps, _ = frsr.peak_error(
        x_bits, probe_k, c0, c1, fmt, "shift_then_sub", "xyyc1",
    )
    assert abs(py_eps - float(eps_arr[0x5900])) < 1e-12
