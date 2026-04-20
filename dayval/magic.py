"""Magic constant C from Algorithm 3's c via Day eq (62).

  C = (2^M / b) * (c + bias * (a + b))

For FRSR (a=1, b=2, M=23, bias=127) this is 2^22 * (c + 381).

The returned C is a rational/mpmath value; it must be rounded to the target
integer width by the caller (round-to-nearest, since C is always positive for
our use cases and small compared to 2^W).
"""
from __future__ import annotations

import mpmath as mp


def magic_C(a: int, b: int, M: int, bias: int, c) -> mp.mpf:
    """Exact-arithmetic magic constant. c should be an mpmath value."""
    return (mp.mpf(1 << M) / b) * (c + bias * (a + b))


def magic_C_int(a: int, b: int, M: int, bias: int, c) -> int:
    """Round-to-nearest integer magic constant, as a Python int."""
    C = magic_C(a, b, M, bias, c)
    # mp.nint rounds to nearest integer with banker's rounding for halves.
    return int(mp.nint(C))
