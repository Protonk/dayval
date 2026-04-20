"""Round-trip and arithmetic tests for the parametric minifloat harness.

Gate: fp32 round-trip must agree with host struct.pack for a random sample
and at all boundary values; fp16 must agree with numpy's float16."""
import struct

import numpy as np
import pytest

from dayval import minifloat as mf


def _host_fp32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def test_fp32_roundtrip_against_host():
    # Sample values spanning normals, subnormals, zero, inf, NaN, boundaries.
    vals = np.array([
        1.0, -1.0, 2.0, 0.5, 3.14159, 1e-10, 1e10,
        0.0, -0.0, float("inf"), -float("inf"),
        2.0 ** -126,                       # smallest normal
        2.0 ** -149,                       # smallest subnormal
        (2 - 2 ** -23) * 2 ** 127,         # largest normal
        2.0 ** -126 - 2.0 ** -149,         # largest subnormal
    ], dtype=np.float64)
    got = mf.float_to_bits(vals, mf.FP32)
    want = np.array([_host_fp32_bits(float(v)) for v in vals], dtype=np.uint32)
    # NaN patterns are allowed to differ in payload but both must be NaN; we
    # skip that here since no NaN in the list.
    np.testing.assert_array_equal(got, want)


def test_fp32_random_roundtrip():
    rng = np.random.default_rng(20260419)
    # log-uniform magnitudes spanning normal range.
    mags = 10.0 ** rng.uniform(-30, 30, size=1000)
    signs = rng.choice([-1.0, 1.0], size=1000)
    vals = (signs * mags).astype(np.float64)
    got = mf.float_to_bits(vals, mf.FP32)
    want = np.array([_host_fp32_bits(float(v)) for v in vals], dtype=np.uint32)
    np.testing.assert_array_equal(got, want)


def test_fp32_encode_decode_identity_on_normals():
    # For every float32 positive normal, encode(decode(bits)) == bits.
    rng = np.random.default_rng(42)
    # Sample 10000 random positive-normal fp32 bit patterns.
    biased_e = rng.integers(1, 255, size=10000, endpoint=False, dtype=np.uint32)
    mant = rng.integers(0, 1 << 23, size=10000, endpoint=False, dtype=np.uint32)
    bits = (biased_e << 23) | mant
    vals = mf.bits_to_float(bits, mf.FP32)
    bits2 = mf.float_to_bits(vals, mf.FP32)
    np.testing.assert_array_equal(bits2, bits)


def test_fp16_against_numpy():
    # Use numpy's float16 to cross-check fp16 encode/decode.
    rng = np.random.default_rng(7)
    vals = rng.standard_normal(5000).astype(np.float64)
    vals = np.concatenate([vals, np.array([0.0, 1.0, -1.0, 65504.0, 6.1035e-5])])
    fp16 = vals.astype(np.float16)
    ref_bits = fp16.view(np.uint16).astype(np.uint32)
    got_bits = mf.float_to_bits(vals, mf.FP16)
    np.testing.assert_array_equal(got_bits, ref_bits)


def test_positive_normals_count():
    # Sanity check the enumerator cardinalities from the plan's table.
    assert len(mf.positive_normals_bits(mf.FP4)) == 4
    assert len(mf.positive_normals_bits(mf.FP6)) == 24
    assert len(mf.positive_normals_bits(mf.FP8)) == 112
    assert len(mf.positive_normals_bits(mf.FP16)) == 30720
    assert len(mf.positive_normals_bits(mf.FP18)) == 126976
    assert len(mf.positive_normals_bits(mf.FP20)) == 507904
    assert len(mf.positive_normals_bits(mf.FP24)) == 8257536
    assert len(mf.positive_normals_bits(mf.FP32)) == 2130706432


def test_lowbit_format_cardinalities():
    # IEEE-style enumeration: biased exp in [1, 2^E - 2], all mantissas.
    # Matches LOW-BIT-FRGR-REFERENCE-PLAN for FP4 E2M1 / FP6 E3M2 / FP8 E4M3.
    # Plan also lists FP6 E2M3 = 12 and FP8 E5M2 = 112, which appear to be
    # typos (IEEE gives 16 and 120 respectively; OCP MX semantics would give
    # even more because the all-ones exponent encodes normals too). We use
    # the IEEE-style counts throughout; the discrepancy is flagged in the
    # plan-update conversation.
    assert len(mf.positive_normals_bits(mf.FP4_E2M1)) == 4
    assert len(mf.positive_normals_bits(mf.FP6_E2M3)) == 16
    assert len(mf.positive_normals_bits(mf.FP6_E3M2)) == 24
    assert len(mf.positive_normals_bits(mf.FP8_E4M3)) == 112
    assert len(mf.positive_normals_bits(mf.FP8_E5M2)) == 120


def test_fmul_fp32_matches_host():
    rng = np.random.default_rng(3)
    a = rng.standard_normal(200).astype(np.float32)
    b = rng.standard_normal(200).astype(np.float32)
    ref = (a * b).view(np.uint32)  # fp32 mul under IEEE RNE
    a_bits = a.view(np.uint32)
    b_bits = b.view(np.uint32)
    got = mf.fmul(a_bits, b_bits, mf.FP32)
    np.testing.assert_array_equal(got, ref)


def test_fadd_fsub_fp32_matches_host():
    rng = np.random.default_rng(5)
    a = rng.standard_normal(200).astype(np.float32)
    b = rng.standard_normal(200).astype(np.float32)
    a_bits = a.view(np.uint32)
    b_bits = b.view(np.uint32)
    np.testing.assert_array_equal(mf.fadd(a_bits, b_bits, mf.FP32), (a + b).view(np.uint32))
    np.testing.assert_array_equal(mf.fsub(a_bits, b_bits, mf.FP32), (a - b).view(np.uint32))
