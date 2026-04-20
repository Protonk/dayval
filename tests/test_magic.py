"""Gate: the two FRSR parity magic constants at fp32 are 0x5F200000 (s=-1)
and 0x5F600000 (s=0), per the plan's table and Day §9.1."""
import mpmath as mp
import pytest

from dayval import algorithm3, magic


@pytest.fixture(autouse=True)
def _precision():
    old = mp.mp.dps
    mp.mp.dps = 50
    yield
    mp.mp.dps = old


def test_fp32_frsr_s_minus_one_magic():
    r = algorithm3.run(a=1, b=2, n=1, s=-1)
    C = magic.magic_C_int(a=1, b=2, M=23, bias=127, c=r.c)
    assert C == 0x5F200000


def test_fp32_frsr_s_zero_magic():
    r = algorithm3.run(a=1, b=2, n=1, s=0)
    C = magic.magic_C_int(a=1, b=2, M=23, bias=127, c=r.c)
    assert C == 0x5F600000


def test_fp32_reciprocal_cuberoot_magic_lomont_note():
    """Day Listing 10 fn 11: reciprocal cube root (a=1, b=3) under the
    Moroz/corrected Algorithm 3. The plan flags this as deferred but we test
    the formula shape here so Phase 2 can reuse it."""
    # Just exercise the function for a=1, b=3 without asserting a specific
    # hex; the gate for the correction is in Phase 2.
    r = algorithm3.run(a=1, b=3, n=1, s=-1)
    C = magic.magic_C_int(a=1, b=3, M=23, bias=127, c=r.c)
    assert 0 < C < (1 << 32)
