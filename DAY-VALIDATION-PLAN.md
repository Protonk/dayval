# Day 2023 Cross-Validation Plan (v2)

## Background

Day [2023] generalises FRSR to FRGR: compute `x^(-a/b)` for any coprime
`a, b ∈ ℤ⁺`, with polynomial refinement of arbitrary degree `n`.
Algorithm 3 produces, in closed-ish form, the magic constant `c` and
the refinement coefficients `c₀…cₙ`, explicitly **under exact
arithmetic** ("all calculations will be considered exact", §4.1).
Finite-precision issues are deferred to §9, which gives tuning
heuristics and worked fp32 implementations (Listings 2–13).

Two things the initial plan got wrong and this revision corrects.

**(i) Analytic output is a family, not a point.** Algorithm 3 has one
free integer `s` (§5, §7.2). Day takes `s = -1` only to match prior
literature. §9.3 observes that for FRSR the practical choice reduces
to two parity classes (`s` even, `s` odd), and that the better parity
is not the traditionally-used one. Via equation (62), the fp32
representatives of the two parities are:

| parity | `c` | `C = 2^(M-1)·(c + 3·bias)` | hex |
|---|---|---|---|
| odd  (`s = -1`) | `-½` | `2²²·380.5` | `0x5F200000` |
| even (`s = 0`)  | `+½` | `2²²·381.5` | `0x5F600000` |

Day's best tuned FRSR (Listing 5) is `0x5F5FFF00` — `0x100` below the
even-parity analytic, not substantially away from either analytic
as I earlier wrote. The phenomenon to validate is therefore
**analytic family (parity-indexed) versus finite-format optimum**, not
"analytic K vs tuned K."

**(ii) Theoretical error and realized error are distinct.** Equation
(57) gives `ε_theory ≈ 6.50070298 × 10⁻⁴` for FRSR under exact
arithmetic. Listing 5 achieves `ε_realized ≈ 6.501791 × 10⁻⁴` at
fp32 after quantising constants and applying §9 tuning. These are
different quantities and must not share the name `ε` in any table.

## Research questions

- **Q1 (family tracking)**. For each format, does exactly one of Day's
  parity-conditioned analytic representatives track the finite-format
  optimum? Does the winning parity depend on `(E, M, bias)`?
- **Q2 (heuristic portability)**. Which §9 levers port across formats,
  and which are fp32-shaped? Ablation ladder (below) separates them.
- **Q3 (grid dominance)**. At what `(E, M)` does the mantissa raster
  dominate — brute-force optimum `K` becomes non-unique (large tie
  sets), and §9 heuristics stop helping?

## Scope

Primary key is `(E, M, bias)`, not width. Total width `W = 1 + E + M`
is a convenience column only. FRSR: `(a, b, n) = (1, 2, 1)`.

**In-scope formats (Phase 1)**:

| name | layout | bias | `#` pos. normals | `(K, c₀, c₁)` exhaustive? |
|---|---|---|---|---|
| fp4  | 1.2.1   | 1   | 4         | trivial |
| fp6  | 1.3.2   | 3   | 24        | trivial |
| fp8  | 1.4.3   | 7   | 112       | laptop-feasible |
| fp16 | 1.5.10  | 15  | 30,720    | `K`-only exhaustive |
| fp18 | 1.6.11  | 31  | 126,976   | `K`-only exhaustive |
| fp20 | 1.6.13  | 31  | 507,904   | `K`-only with care |
| fp24 | 1.7.16  | 63  | 8,257,536 | `K`-only overnight |
| fp32 | 1.8.23  | 127 | 2,130,706,432 | Day's ground truth — replication only |

fp32 is in Phase 1 as the ground-truth sanity check: Algorithm 3 must
reproduce equation (57) coefficients and Listing 5 must reproduce
`0x5F5FFF00` / `ε ≈ 6.501791 × 10⁻⁴` before anything else runs.

**Deferred to Phase 2**: fp64, bf16, `(a,b) ∈ {(1,1), (1,3), (2,3)}`,
`n ≥ 2`, the 2-iteration FRSR (§10.5), monic vs general polynomial
tradeoff (§6 Figure 8).

## Phase A — implementation

### A1. Algorithm 3 reference port

Port §5 verbatim. For any `(a, b, n)` return `(c, c₀…cₙ, zmin, zmax,
ε_theory)` as exact-arithmetic quantities (rationals / high-precision
reals, not target-format floats). Validate against §7.2 equation (57):
FRSR `(1, 2, 1)` must yield `c₀ ≈ 1.68191391`, `c₁ ≈ -0.703952009`,
`ε_theory ≈ 6.50070298 × 10⁻⁴`.

### A2. Parametric minifloat arithmetic

Bit-cast helpers parameterised on `(E, M, bias)`. Correct handling of:
positive normals (the FRSR domain per §3 fn 2), subnormals, zero,
infinity, NaN. IEEE round-to-nearest-even throughout. Enumerator that
yields all positive normals for a given format.

### A3. Parametric FRSR kernel

`frsr(x, format, K, c₀, c₁)` with format-native arithmetic. Both
instruction orderings from §9.2:

- Shift-then-subtract (Listing 1): `Y = K - (X >> 1)`
- Subtract-then-shift (§9.2): `Y = (C' - X) >> 1`, `C' ∈ {2K, 2K+1}`

The second gives one extra bit of effective `K` precision and must be
a separate ablation dimension, not folded into "tuning."

### A4. Execution semantics

All cross-width results come from a software IEEE-style arithmetic
harness implementing round-to-nearest-even with the target format's
exact bit layout. Hardware math is not used, so numbers are
independent of FMA availability, ISA, or compiler instruction
selection — the concerns Day raises in §2 fn 1. For fp32 we
additionally reproduce Listing 5 against a host-compiled C kernel as
a second sanity check, but reported cross-width numbers all come
from the software harness. No FMA anywhere.

## Phase B — data collection

Per format:

### B1. Analytic family

Run Algorithm 3. For each `s ∈ {-1, 0}` (the two parity
representatives for FRSR; more `s` values would be redundant modulo
exact power-of-two scaling per §9.3):

- Exact-arithmetic `(c, c₀, c₁)` and `ε_theory`
- Format-quantised `(K_s, ĉ₀_s, ĉ₁_s)` via equation (62) and
  round-to-nearest for coefficients
- Realized peak error `ε_realized(analytic, s)` over all positive
  normals, using shift-then-subtract kernel

### B2. Finite-format optima

Regime determined by format, not by an imported fp32 window:

**Search domain (all regimes)**. `K` ranges over all `W`-bit integers
for format width `W`. Candidate coefficients `ĉ₀, ĉ₁` range over the
positive normal encodings of the format; subnormals, zero, infinity,
and NaN are excluded as *candidates*, though the kernel must still
evaluate correctly when intermediate values land in those classes on
specific inputs `x`.

- **fp4, fp6, fp8** — exhaustive over all `(K, ĉ₀, ĉ₁)` in the above
  domain. Gold standard.
- **fp16, fp18, fp20, fp24** — two passes:
  - `K`-only exhaustive sweep with `(ĉ₀, ĉ₁)` pinned to quantised
    analytic values, one sweep per parity. This is the clean Q1 probe.
  - Joint local search over `(K, ĉ₀, ĉ₁)` for the Q2/Q3 probe.
    **Seeds**: both parity analytic points, plus 8 starts sampled
    uniformly from a box of radius `2^⌈M/2⌉` ULPs around each parity
    seed in each coordinate. **Neighborhood**: one ULP step in any
    single coordinate (6 neighbors per point). **Descent**: steepest
    improvement in peak `ε` per step, breaking ties by lexicographic
    coordinate order for reproducibility. **Stopping**: no improvement
    in 32 consecutive steps. All results labeled **local**, not
    global.
- **fp32** — replicate Day's Listing 5 and verify our pipeline against
  his published error.

### B3. Heuristic ablation ladder

Applied on top of B2's best finite-format point, stepwise. Each step
gets an independent column in the table:

1. Winning parity from B1.
2. `C'`-extra-bit via subtract-then-shift (§9.2).
3. The 16 product-ordering variants (§9.2).
4. Coefficient fine-tuning (§9.4 search, within format).

The table must show how much `ε_realized` moves at each step so we
can identify which mechanism is portable and which is fp32-shaped.

### B4. Witness-preserving records

At every recorded optimum (or near-optimum), store:

- The input `x*` realising the peak error (worst-case witness).
- Exact minimizer set: the `K` values that achieve the minimum peak
  `ε` exactly. Well-defined in the software harness where arithmetic
  is bit-exact.
- Near-optimal band: count of `K` values whose peak `ε` is within one
  ULP of the minimum.
- Second-worst error and number of inputs within one ULP of the
  worst.

This is what Q3 actually needs — degeneracy structure, not just a
scalar `ε`.

## Phase C — analysis

Primary table keyed on `(E, M, bias)`:

| E | M | bias | `K_(s=-1)` | `K_(s=0)` | `K_opt` | winning `s` | `ε_theory` | `ε_real(analytic, winning s)` | `ε_real(+C' bit)` | `ε_real(+orderings)` | `ε_real(+coef tune)` | `ε_opt` | tie-set | `x*` |

Derived views:

- **Q1 plot**: `ε_real(analytic, winning s) / ε_opt` vs `M` at fixed
  `E/bias` ratio, and vs `E` at fixed `M`. Our heuristic expectation
  is that this ratio trends toward 1 as the mantissa grid refines —
  the finite-precision kernel should become increasingly well-modelled
  by Day's exact-arithmetic construction. This is an expectation we're
  testing, not a cross-width convergence theorem from the paper; what
  matters is the shape of any approach or the pattern of any
  failure to approach.
- **Q2 ladder**: for each ablation step, the gain `Δε` as a function
  of `(E, M)`. A step that helps at fp32 but flatlines at fp16 is
  fp32-shaped.
- **Q3 degeneracy**: exact-minimizer set size and near-optimal band
  count, each vs `(E, M)`. Look for growth as `M` shrinks; full
  degeneracy (all `K` in some range tying exactly) is a candidate
  grid-dominance threshold. Exponent-layout effects may introduce
  non-monotone structure.

## Deliverables

1. Reference Algorithm 3 port + parametric kernel + harness. Runs on
   a laptop for formats up to fp20; fp24 overnight.
2. Primary table as CSV plus the three derived views.
3. Short writeup: which of Q1/Q2/Q3 admit clean answers, and for any
   that don't, what the failure-to-answer structure is.

## Phase 2 — deferred work

- fp28, fp64.
- bf16 (1.8.7) — same `E` as fp32 with much smaller `M`. Useful
  precisely because it holds `E` fixed and varies `M`.
- Reciprocal `(a=b=1)` across widths.
- Reciprocal cube root `(a=1, b=3)` across widths. Day notes in
  Listing 10 fn 11 that Moroz et al.'s magic constant was suboptimal
  due to a clamp-omission in Algorithm 3 line 9; cross-width check
  of his correction.
- `(a=2, b=3)` (§10.4).
- `n ≥ 2` (§10.5).
- Monic vs general polynomial tradeoff (§6, Figure 8) as a function
  of `(E, M)`.

## Citations beyond Day [2023]

One optional external anchor. The existing data point at fp16 is
Eshed Schacham, "16-bit Fast Inverse Square Root," personal blog,
[https://ashdnazg.github.io/articles/25/16-bit-Fast-Inverse-Square-Root](https://ashdnazg.github.io/articles/25/16-bit-Fast-Inverse-Square-Root),
published 2025-08-23, reporting exhaustive brute-force
`K_tuned = 0x59B7` with `ε ≈ 2.8 × 10⁻³`. Useful as a pipeline
self-check at fp16. Not structurally required; A1/A2/A3 validated
against fp32 is sufficient pipeline evidence. **Approve or drop?**
