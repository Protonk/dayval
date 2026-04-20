# Results — Day cross-validation and low-bit FRGR reference

Two plans drive this codebase: `DAY-VALIDATION-PLAN.md` (cross-width
validation of Day [2023]'s parity-indexed analytic family against
finite-format optima) and `LOW-BIT-FRGR-REFERENCE-PLAN.md` (exhaustive
spec sheets for `1/√x` and `1/x` at the five production low-bit float
formats). They share a single software harness — a parametric
`(E, M, bias)` IEEE-style floating-point implementation with explicit
round-to-nearest-even at every op — and a PyO3 Rust kernel that runs
at ~580 million kernel evaluations per second per core.

## What was built

The Python package `dayval/` and the Rust extension `dayval_rust`
together provide: an exact-arithmetic port of Day's Algorithm 3
(eq 56 closed form for degree 1, Remez for higher degree), a
parametric minifloat harness with bit-exact RNE, an FRSR kernel
parameterised over three coarse orderings and nine refinement
orderings from Day §9.2, the eq (62) magic constant, and both plans'
drivers.
Every gate passes: 35 tests including a bit-exact match to a compiled
C copy of Day Listing 5 and reproduction of `ε = 6.501791 × 10⁻⁴` at
witness `x* = 0x01401a9f` over all 2.13 × 10⁹ positive fp32 normals.

## Day cross-validation, Phase 1

The primary table is in `results/primary.csv`, derived views in
`results/q1_ratio.csv`, `results/q2_ladder.csv`, and
`results/q3_degeneracy.csv`. Phase 1 covers fp4, fp6 (E3M2), fp8
(E4M3), fp16, fp18, fp20, and fp32 (replication-only); fp24 is
deferred to a long-running job estimated at about 36 h on 32 cores
with the current Rust kernel.

**Q1 — family tracking.** The expectation was that as the mantissa
grid refines, the format-quantised analytic-family representative
should track the brute-force optimum. The data confirms this clearly:
the ratio `ε_real(analytic, winning s) / ε_opt` falls monotonically
from 2.86 at fp4 to 1.21 at fp16 to 1.04 at fp20 to 1.0001 at fp32.
There is no non-monotonicity in the Phase 1 range, and nothing in
the fp4–fp20 results suggests the ratio would deviate from unity at
wider widths. Phase 2's fp64 / bf16 / fp28 points will extend this.

**Q2 — heuristic portability.** The plan's ablation-ladder columns
(`eps_real_plus_Cprime`, `eps_real_plus_orderings`,
`eps_real_plus_coef_tune`) are populated for fp4 through fp20. The
largest gain attributable to §9.2 orderings alone is at fp18 (delta
`1.07 × 10⁻⁴`, a 8% reduction over the plus-C' baseline); the gain
collapses to zero at fp16 and fp20 — fp32-shaped rather than
portable, in the plan's vocabulary. The C' extra-bit lever (§9.2
subtract-then-shift with `C' ∈ {2K, 2K+1}`) contributes at fp16 (a
`1.5 × 10⁻⁴` reduction) and fp8 but nowhere else. Coefficient
fine-tuning via joint local search contributes at every format,
shrinking monotonically from fp4 to fp20.

**Q3 — grid dominance.** The tie-set size is 1 at every Phase 1
format; no wide near-optimal band was observed in the K-only sweeps.
fp24 is the format where Day's paper would lead one to expect this to
change, so the question is genuinely unanswered until that run
completes.

## Low-bit spec sheets

The per-format sheets in `results/lowbit/` cover both rsqrt (`1/√x`)
and reciprocal (`1/x`) at FP4 E2M1, FP6 E2M3, FP6 E3M2, FP8 E4M3,
and FP8 E5M2. Each sheet reports the format-intrinsic floor
`ε_floor(f)`, the global optimum at each of six algorithm tiers
(T0_monic, T0_scale, T1_monic, T1_gen, T2_monic_horner,
T2_gen_horner), and a §9 ablation ladder on the T1_gen baseline.

**T1_gen already saturates the floor at FP8 E4M3** for both rsqrt
(5.23 × 10⁻²) and reciprocal (1.88 × 10⁻¹). Higher tiers — T2_monic
and T2_gen — cannot improve on T1_gen at this format; the four-op
Quake-style refinement is the operational sweet spot, and spending
additional ops buys nothing. This is a strong operational
implication: for FP8 E4M3 there is no point in a quadratic-refinement
kernel.

**Reciprocal saturates even earlier at T0_scale** — a single post-
multiply `y * k` with `k ≈ 0.375` hits the floor at FP8 E4M3, FP8
E5M2, and FP6 E3M2. At FP6 E2M3 reciprocal, T0_scale halves the
T1_gen gap. At FP4 E2M1 reciprocal, no tier reaches the floor (every
tier pegs at 1.0 peak error because the coefficient grid has only
four positive normals and the optimum cannot be represented). The
takeaway for kernel authors is that FP4 reciprocal is LUT-only at
the format level — there is no useful FRCP kernel.

**At FP8 E5M2 rsqrt, the §9.2 C' extra-bit lever closes the T1_gen
gap to the floor exactly.** Canonical shift-then-subtract gives
`ε = 1.18 × 10⁻¹` at `K = 0x43`, `c0 = 64`, `c1 = −28672` — short of
the floor by `3.55 × 10⁻²`. Switching the coarse step to
`Y = (2K − X) >> 1` at the same `(K, c0, c1)` yields exactly
`ε = 8.25 × 10⁻² = ε_floor`. This is the "one extra bit of K
precision" Day describes qualitatively in §9.2, made concrete at a
production format: it moves E5M2 rsqrt from 1.4× the floor to
floor-saturated, for the same op count. The same lever produces no
improvement at FP8 E4M3 (where canonical already saturates), which
is the clearest evidence the lever pays off only when the parity
interaction with the mantissa raster leaves room for it.

**Analytic parity choice matters a lot at low precision.** The
parity-switch column of the ablation table shows penalties as large
as `1.07 × 10⁻¹` at FP6 E3M2 and FP8 E5M2 for picking the wrong
parity analytically. For FRCP at FP4/FP6 and FP8 E4M3, both
analytic parity representatives overflow intermediate values and
produce `ε = ∞`, which is a negative-result data point: Day's §9.1
magic-constant formula is not directly usable at these widths
without tuning.

## Paper and plan corrections surfaced by the implementation

Two mechanical corrections came out of reproducing the paper's
numbers:

**Day eq (57) ε has a last-digit typo.** The paper prints
`ε ≈ 6.50070298 × 10⁻⁴`; the true minimax value to 13 significant
figures is `6.500702958850 × 10⁻⁴`, verified by direct
equioscillation check at 100 decimal-place precision and by two
independent methods (the eq (56) closed form and a Remez iteration
implemented in mpmath). My rounded-to-9sf value is `6.50070296e-4`,
one unit in the last digit below the paper's printed value. The
test `test_frsr_s_minus_one_eq57` gates on the true value with a
tolerance of `1e-15`.

**Listing 5's coefficients in the rendered PDF of the paper are
unreadable on screen at typical DPI.** I initially OCRed
`1.1891763f` and `0.24885956f`, which reproduce peak error
`7.484 × 10⁻⁴` — close to but not Day's published `6.501791 × 10⁻⁴`.
The arxiv LaTeX source (downloaded and kept in `sources/`) has
`1.1893165f` and `0.24889956f`, which reproduce the published peak
exactly at witness `x* = 0x01401a9f` (verified against a compiled C
kernel with `-mno-fma -fno-fast-math`).

**LOW-BIT plan target-formats table** listed FP6 E2M3 = 12 and FP8
E5M2 = 112 positive normals, which disagree with both IEEE-style
enumeration (16 / 120) and OCP MX semantics (24 / 120). Commit
`89e8db3` corrects these and adds a paragraph clarifying the
enumeration convention.

## Outstanding

Two items remain before the full deliverable set of either plan is
complete:

- **fp24** for the Day plan. Estimated 36 h on 32 cores with the
  current Rust kernel; memory-flat single-K streaming path already
  verified at fp32. Ready to kick off whenever the machine can spare
  the CPU.
- **B4 second-worst witness**. The K-sweep currently records one
  witness per K; the second-worst x requires a follow-up single-K
  re-evaluation. Cheap to add once the analysis phase needs it.

## Reproducing

All results were produced from a clean checkout with Python 3.10,
mpmath 1.4.1, numpy 2.2.6, and Rust 1.75. `.venv/bin/maturin develop
--release` builds `dayval_rust`; `pytest tests/` runs all 35 gates
(approximately 4 minutes, dominated by the fp32 kernel-replay test);
`python scripts/run_phase1.py` regenerates the Day-plan primary
table; `python scripts/run_lowbit.py` regenerates the low-bit spec
sheets. The LaTeX source of Day [2023] is kept in `sources/` for
anyone who wants to verify the numerical checks against the paper's
typeset equations.
