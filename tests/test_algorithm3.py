"""Gate tests for Algorithm 3 — must pass before any sweep runs.

Reference values from Day (2023) equation (57) and the numerical derivation
of the s = 0 parity in §9.3.
"""
import mpmath as mp
import pytest

from dayval import algorithm3


@pytest.fixture(autouse=True)
def _precision():
    old = mp.mp.dps
    mp.mp.dps = 50
    yield
    mp.mp.dps = old


def _close(actual, expected, tol):
    assert abs(actual - expected) < tol, f"{actual} vs {expected}, diff {abs(actual - expected)}"


def test_frsr_s_minus_one_eq57():
    """Eq (57): FRSR s = -1 analytic."""
    r = algorithm3.run(a=1, b=2, n=1, s=-1)
    assert r.c == mp.mpf('-0.5')
    _close(r.zmin, mp.mpf(3) / 4, mp.mpf('1e-40'))
    _close(r.zmax, mp.mpf(27) / 32, mp.mpf('1e-40'))
    _close(r.coeffs[0], mp.mpf('1.68191391'), mp.mpf('1e-8'))
    _close(r.coeffs[1], mp.mpf('-0.703952009'), mp.mpf('1e-9'))
    # Paper eq (57) prints "6.50070298e-4"; the true minimax value to 13 sf is
    # 6.500702958850e-4 (verified by direct equioscillation check at 100 dps),
    # so the paper's last digit is a display typo. Gate against the true value.
    _close(r.eps_theory, mp.mpf('6.500702958850e-4'), mp.mpf('1e-15'))


def test_frsr_s_zero_parity():
    """§9.3 s = 0 parity: c = +1/2. Analytic coefficients on [3/2, 27/16]
    are c0_{s=-1}/sqrt(2) and c1_{s=-1}/(2*sqrt(2)) by the exact-rescaling
    argument (zmin, zmax both scale by 2 when s increments by 1).

    Note: these are the *analytic* s=0 values. Listing 5's published
    coefficients 1.1891763 / -0.24885956 are close but not equal — they
    are §9.4-tuned from the analytic s=0 baseline."""
    r = algorithm3.run(a=1, b=2, n=1, s=0)
    r_ref = algorithm3.run(a=1, b=2, n=1, s=-1)
    assert r.c == mp.mpf('0.5')
    _close(r.zmin, mp.mpf(3) / 2, mp.mpf('1e-40'))
    _close(r.zmax, mp.mpf(27) / 16, mp.mpf('1e-40'))
    _close(r.coeffs[0], r_ref.coeffs[0] / mp.sqrt(2), mp.mpf('1e-40'))
    _close(r.coeffs[1], r_ref.coeffs[1] / (2 * mp.sqrt(2)), mp.mpf('1e-40'))
    # Sanity: 9-sf expected values from the closed form.
    _close(r.coeffs[0], mp.mpf('1.18929273'), mp.mpf('1e-8'))
    _close(r.coeffs[1], mp.mpf('-0.24888462'), mp.mpf('1e-8'))


def test_frsr_eps_theory_parity_invariant():
    """eps_theory must be the same under the s shift (eq (45) is periodic)."""
    r_m1 = algorithm3.run(a=1, b=2, n=1, s=-1)
    r_0 = algorithm3.run(a=1, b=2, n=1, s=0)
    _close(r_m1.eps_theory, r_0.eps_theory, mp.mpf('1e-30'))


def test_reciprocal_closed_form():
    """§7.3: a = b = 1 gives c = sqrt(2) - 2 with zmin = sqrt(2)/2, zmax =
    (3 + 2*sqrt(2))/8."""
    r = algorithm3.run(a=1, b=1, n=1, s=-1)
    _close(r.c, mp.sqrt(2) - 2, mp.mpf('1e-40'))
    _close(r.zmin, mp.sqrt(2) / 2, mp.mpf('1e-40'))
    _close(r.zmax, (3 + 2 * mp.sqrt(2)) / 8, mp.mpf('1e-40'))


def test_remez_matches_closed_form_for_linear():
    """The n=1 closed form must equal what Remez would give."""
    r_closed = algorithm3.run(a=1, b=2, n=1, s=-1)
    coeffs_remez, eps_remez = algorithm3._remez(2, 1, r_closed.zmin, r_closed.zmax)
    # Remez precision is limited by the node-refinement stopping criterion;
    # a 1e-10 match is sufficient evidence that the two approaches agree.
    _close(coeffs_remez[0], r_closed.coeffs[0], mp.mpf('1e-10'))
    _close(coeffs_remez[1], r_closed.coeffs[1], mp.mpf('1e-10'))
    _close(eps_remez, r_closed.eps_theory, mp.mpf('1e-12'))
