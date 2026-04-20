"""Algorithm 3 from Day (2023) — exact-arithmetic port.

Given coprime (a, b) in Z+ and polynomial degree n in N, plus integer free
parameter s in Z, returns the FRGR constants (c, c_0, ..., c_n, zmin, zmax,
eps_theory) that minimise the worst-case relative error of the refined
approximation y*p(z) to x^(-a/b) over the positive normals.

For n = 1 (FRSR), uses the closed form of eq (56). For n >= 2 uses Remez
iteration on [zmin, zmax] approximating z^(-1/b).

All arithmetic is mpmath at a working precision set from mpmath.mp.dps;
callers that care about determinism should set mp.dps explicitly before
calling. The returned values are mpf/mpc scalars, not native floats.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import mpmath as mp


@dataclass(frozen=True)
class Algo3Result:
    a: int
    b: int
    n: int
    s: int
    c: mp.mpf
    coeffs: tuple  # (c_0, c_1, ..., c_n)
    zmin: mp.mpf
    zmax: mp.mpf
    eps_theory: mp.mpf


def _tstar(a: int, b: int) -> mp.mpf:
    """t* from Algorithm 3, independent of s."""
    alpha = min(a, b)
    beta = max(a, b)
    gamma = a + b

    if alpha == 1:
        t0 = 1 / mp.log(2) - 1
    else:
        t0 = (mp.mpf(alpha) - 1) / (mp.power(2, 1 - mp.mpf(1) / alpha) - 1) - alpha
    phi = 1 / (mp.power(2, mp.mpf(1) / gamma) - 1) - gamma + 1
    rbar = int(mp.floor(phi))
    t1 = phi - rbar
    if alpha == 1:
        lo = mp.mpf(rbar - 1) / beta
        hi = mp.mpf(rbar) / beta
        if t1 < lo:
            return lo
        if t1 > hi:
            return hi
        return t1
    return t0


def _zmin_zmax(a: int, b: int, s: int) -> tuple[mp.mpf, mp.mpf]:
    alpha = min(a, b)
    gamma = a + b
    tstar = _tstar(a, b)

    if alpha == 1:
        t0 = 1 / mp.log(2) - 1
    else:
        t0 = (mp.mpf(alpha) - 1) / (mp.power(2, 1 - mp.mpf(1) / alpha) - 1) - alpha
    phi = 1 / (mp.power(2, mp.mpf(1) / gamma) - 1) - gamma + 1
    rbar = int(mp.floor(phi))
    t1 = phi - rbar

    r_alpha = 0 if tstar < t0 else alpha - 1
    r_gamma = rbar if tstar < t1 else rbar - 1

    zmin = mp.power(2, s - r_alpha) * mp.power(1 + (r_alpha + tstar) / alpha, alpha)
    zmax = mp.power(2, s - r_gamma) * mp.power(1 + (r_gamma + tstar) / gamma, gamma)
    return zmin, zmax


def _linear_minimax(b: int, zmin: mp.mpf, zmax: mp.mpf) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    """Closed form eq (56) for the degree-1 minimax of z^(-1/b) on [zmin, zmax]."""
    inv_b = mp.mpf(1) / b
    zmin_pb = mp.power(zmin, inv_b)
    zmax_pb = mp.power(zmax, inv_b)
    zmin_1p = mp.power(zmin, 1 + inv_b)
    zmax_1p = mp.power(zmax, 1 + inv_b)

    T = (zmax_1p - zmin_1p) / (zmax_pb - zmin_pb)
    U = b * mp.power(T / (b + 1), 1 + inv_b)
    V = (mp.power(zmin * zmax, inv_b) * (zmax - zmin)) / (zmax_pb - zmin_pb)
    c0 = 2 * T / (U + V)
    c1 = mp.mpf(-2) / (U + V)
    eps = (U - V) / (U + V)
    return c0, c1, eps


def _remez(b: int, n: int, zmin: mp.mpf, zmax: mp.mpf,
           max_iter: int = 60) -> tuple[tuple, mp.mpf]:
    """Remez exchange for degree-n minimax of z^(-1/b) on [zmin, zmax] under
    relative-error norm (sign convention matches eq (10): e(z) = 1 - z^(1/b) p(z)).

    Returns (coefficients c0..cn, peak relative error eps).
    """
    inv_b = mp.mpf(1) / b

    def f(z):  # target
        return mp.power(z, -inv_b)

    # Initial Chebyshev-like nodes on [zmin, zmax].
    nodes = [zmin + (zmax - zmin) * (1 - mp.cos(mp.pi * k / (n + 1))) / 2
             for k in range(n + 2)]

    for _ in range(max_iter):
        # Solve the interpolation system: p(z_k) + (-1)^k * eps * f(z_k) = f(z_k)
        # i.e. sum_j c_j z_k^j + sigma_k eps f(z_k) = f(z_k), sigma_k = (-1)^k.
        # Relative error: e = 1 - z^(1/b) p(z) = +/- eps at nodes,
        # so p(z_k) = (1 -/+ eps) z_k^(-1/b). Signs alternate.
        A = mp.matrix(n + 2, n + 2)
        rhs = mp.matrix(n + 2, 1)
        for k, zk in enumerate(nodes):
            for j in range(n + 1):
                A[k, j] = mp.power(zk, j)
            A[k, n + 1] = -(-1) ** k * f(zk)  # coefficient of eps
            rhs[k, 0] = f(zk)
        sol = mp.lu_solve(A, rhs)
        coeffs = tuple(sol[j, 0] for j in range(n + 1))
        eps = sol[n + 1, 0]

        # Locate new extrema of the error e(z) = 1 - z^(1/b) * p(z) on [zmin, zmax]
        # by scanning and then local-refining. With only n+2 alternation points,
        # a dense scan between consecutive nodes locates each extremum.
        def err(z):
            p = sum(coeffs[j] * mp.power(z, j) for j in range(n + 1))
            return 1 - mp.power(z, inv_b) * p

        # Always include endpoints.
        candidates = [zmin, zmax]
        scan = 200
        for i in range(scan + 1):
            z = zmin + (zmax - zmin) * mp.mpf(i) / scan
            candidates.append(z)
        # Find (n+2) extrema by local search (golden-section-like bracket) on the
        # scan maxima/minima.
        vals = [(z, err(z)) for z in candidates]
        vals.sort(key=lambda p: p[0])
        # Keep turning points: sign changes of slope.
        extrema = [vals[0]]
        for i in range(1, len(vals) - 1):
            if ((vals[i][1] >= vals[i - 1][1] and vals[i][1] >= vals[i + 1][1]) or
                (vals[i][1] <= vals[i - 1][1] and vals[i][1] <= vals[i + 1][1])):
                extrema.append(vals[i])
        extrema.append(vals[-1])

        # Refine each extremum with a few bisection-on-derivative steps.
        refined = []
        for i, (z, _) in enumerate(extrema):
            # Choose a small bracket around z, clamped to [zmin, zmax].
            dz = (zmax - zmin) / (2 * scan)
            lo = max(zmin, z - dz)
            hi = min(zmax, z + dz)
            # Ternary search on |err(z)|.
            for _t in range(40):
                m1 = lo + (hi - lo) / 3
                m2 = hi - (hi - lo) / 3
                if abs(err(m1)) < abs(err(m2)):
                    lo = m1
                else:
                    hi = m2
            refined.append((lo + hi) / 2)

        # Select n+2 extrema with alternating signs and largest |err|.
        refined.sort()
        # Deduplicate close extrema.
        dedup = [refined[0]]
        tol = (zmax - zmin) * mp.mpf('1e-20')
        for z in refined[1:]:
            if z - dedup[-1] > tol:
                dedup.append(z)
        # Take every extremum; if too many, keep the n+2 with largest |err|.
        if len(dedup) > n + 2:
            dedup.sort(key=lambda z: -abs(err(z)))
            dedup = sorted(dedup[:n + 2])
        elif len(dedup) < n + 2:
            # Pad with endpoints if degenerate.
            if dedup[0] > zmin:
                dedup.insert(0, zmin)
            while len(dedup) < n + 2 and dedup[-1] < zmax:
                dedup.append(zmax)

        new_nodes = dedup[:n + 2]
        # Convergence check.
        max_shift = max(abs(a - b) for a, b in zip(new_nodes, nodes))
        nodes = new_nodes
        if max_shift < (zmax - zmin) * mp.mpf('1e-25'):
            break

    eps = abs(eps)
    return coeffs, eps


def run(a: int, b: int, n: int, s: int) -> Algo3Result:
    if gcd(a, b) != 1:
        raise ValueError(f"a={a}, b={b} must be coprime")
    if a < 1 or b < 1:
        raise ValueError("a, b must be positive")
    if n < 0:
        raise ValueError("n must be nonnegative")

    tstar = _tstar(a, b)
    c = s + tstar
    zmin, zmax = _zmin_zmax(a, b, s)

    if n == 1:
        c0, c1, eps = _linear_minimax(b, zmin, zmax)
        coeffs = (c0, c1)
    elif n == 0:
        # Degree-0 minimax of z^(-1/b) on [zmin, zmax]: constant that
        # equioscillates between endpoints. The minimax constant p0 is the one
        # making relative errors at zmin and zmax equal in magnitude with
        # opposite signs: 1 - zmin^(1/b) p0 = -(1 - zmax^(1/b) p0) => p0 = 2 /
        # (zmin^(1/b) + zmax^(1/b)).
        inv_b = mp.mpf(1) / b
        zmin_pb = mp.power(zmin, inv_b)
        zmax_pb = mp.power(zmax, inv_b)
        p0 = 2 / (zmin_pb + zmax_pb)
        eps = (zmax_pb - zmin_pb) / (zmax_pb + zmin_pb)
        coeffs = (p0,)
    else:
        coeffs, eps = _remez(b, n, zmin, zmax)

    return Algo3Result(a=a, b=b, n=n, s=s, c=c,
                       coeffs=tuple(coeffs), zmin=zmin, zmax=zmax,
                       eps_theory=eps)
