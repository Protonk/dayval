# Results — Day cross-validation and low-bit FRGR reference

The project (`FRGR-PLAN.md`) has two arms sharing one software
harness: a **validation arm** that cross-checks Day [2023]'s parity-
indexed analytic family against finite-format optima across widths,
and a **reference arm** that produces exhaustive spec sheets for
`1/√x` and `1/x` at the five production low-bit float formats. The
harness is a parametric `(E, M, bias)` IEEE-style floating-point
implementation with explicit round-to-nearest-even at every op,
backed by a PyO3 Rust kernel that runs at ~580 million kernel
evaluations per second per core.

## What was built

The Python package `dayval/` and the Rust extension `dayval_rust`
together provide: an exact-arithmetic port of Day's Algorithm 3
(eq 56 closed form for degree 1, Remez for higher degree), a
parametric minifloat harness with bit-exact RNE, an FRSR kernel
parameterised over three coarse orderings and nine refinement
orderings from Day §9.2, the eq (62) magic constant, and both plans'
drivers.
Every gate passes: 35 tests including a bit-exact kernel-agreement check
against a compiled C copy of Day Listing 5 over 200k stratified fp32
samples, and a witness replay at `x* = 0x01401a9f` reproducing
`ε = 6.501791 × 10⁻⁴`. The global-peak claim across all 2.13 × 10⁹
positive fp32 normals is from an out-of-band exhaustive C scan documented
in `DAY-2023-ERRATA.md`; the pytest gate does not enumerate the full set.

The `results/` directory is tracked (commit `67b44ba`), so the numbers
below can be cross-referenced against the committed CSVs and spec
sheets directly. Regenerate from the drivers in `scripts/` to refresh.

## Validation arm, Phase 1

The primary table is in `results/primary.csv`, derived views in
`results/q1_ratio.csv`, `results/q2_ladder.csv`, and
`results/q3_degeneracy.csv`. Phase 1 covers fp4, fp6 (E3M2), fp8
(E4M3), fp16, fp18, fp20, and fp32 (replication-only); fp24 is
deferred to a long-running job estimated at about 36 h on 32 cores
with the current Rust kernel.

**Q1 — family tracking.** The expectation was that as the mantissa
grid refines, the format-quantised analytic-family representative
should track the brute-force optimum. The data from `results/
q1_ratio.csv` shows a broadly downward trend but not strict
monotonicity in the Phase 1 range: ratios are 2.86 at fp4, 1.00 at
fp6, 1.88 at fp8, 1.21 at fp16, 1.18 at fp18, 1.04 at fp20, and
1.0001 at fp32. The fp6→fp8 excursion is the interesting feature —
fp6 E3M2 has exactly one positive normal between `zmin` and `zmax`
for the winning-parity baseline, so the quantised analytic and the
exhaustive optimum coincide at ratio 1; at fp8 the raster opens up
enough that the analytic coefficients diverge from the best
quantised `(c₀, c₁, K)` again before the trend resumes toward 1
from fp16 onward. Phase 2's fp64 / bf16 / fp28 points will extend
the tail.

`eps_opt` is tagged in the primary CSV via `eps_opt_kind`:
**exhaustive** for fp4/fp6/fp8 (true global optimum via
`lowbit.tier_exhaustive`), **local** for fp16/fp18/fp20/fp24 (joint
local search from the analytic seed — not guaranteed global), and
**replication** for fp32 (Day Listing 5's `6.501791 × 10⁻⁴`, not a
pipeline-derived optimum). The fp32 ratio should therefore be read as
"how close the format-quantised analytic family gets to the paper's
published kernel," not "how close to the fp32 global optimum."

**Q2 — heuristic portability.** The plan's ablation-ladder columns
(`eps_real_plus_Cprime`, `eps_real_plus_orderings`,
`eps_real_plus_coef_tune`) are populated for fp4 through fp20. The
§9.2 orderings lever delivers its only non-zero gain at fp18
(`Δε = 1.07 × 10⁻⁴`, about 8% over the plus-C' baseline); it is zero
everywhere else, which reads as fp32-shaped rather than portable.
The C' extra-bit lever (§9.2 subtract-then-shift with `C' ∈ {2K,
2K+1}`) contributes materially at fp4 (`Δε = 3.37 × 10⁻¹`, the
largest lever in the table) and modestly at fp16 (`Δε = 1.51 ×
10⁻⁴`); it is zero at fp6, fp8, fp18, and fp20. Coefficient
fine-tuning via exhaustive-or-local search contributes at every
format *except* fp6 (where the exhaustive optimum already coincides
with the analytic point under the winning parity's C' baseline): fp4
`Δε = 2.07 × 10⁻¹`, fp8 `Δε = 4.63 × 10⁻²`, fp16 `Δε = 1.43 × 10⁻⁴`,
fp18 `Δε = 8.99 × 10⁻⁵`, fp20 `Δε = 3.14 × 10⁻⁵`. The trend shrinks
monotonically from fp8 onward; the fp4→fp8 step is not monotone.

**Q3 — grid dominance.** The exact-minimizer count is 1 at every
Phase 1 format under the step-4 `(c₀, c₁)` values, and the near-optimal
band (K values within one ULP of the minimum at those coefficients) is
also 1. Both are emitted in the primary CSV and `q3_degeneracy.csv`,
alongside the second-worst witness. Degeneracy is conditional on the
specific `(c₀, c₁)` used — at fp16+ that's the local-search result, not
a global optimum — so the absence of a wide band there is only weak
evidence of grid non-dominance. fp24 is the format where Day's paper
would lead one to expect this to change, so the question is genuinely
unanswered until that run completes.

## Reference arm: low-bit spec sheets

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
E5M2, and FP6 E3M2. At FP6 E2M3 reciprocal, T1_gen (ε = 0.625) cuts
T0_scale's gap-to-floor roughly in half — floor 0.1875, T0_scale
1.0625 (gap 0.875), T1_gen 0.625 (gap 0.4375). At FP4 E2M1
reciprocal, no tier reaches the floor (every tier pegs at 1.0 peak
error because the coefficient grid has only four positive normals
and the optimum cannot be represented). The takeaway for kernel
authors is that FP4 reciprocal is LUT-only at the format level —
there is no useful FRCP kernel.

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

**Analytic parity choice matters a lot at low precision (rsqrt).** The
parity-switch column of the ablation table shows penalties as large as
`1.07 × 10⁻¹` at FP6 E3M2 and FP8 E5M2 for picking the wrong parity
analytically. The reciprocal parity-switch lever is now wired to the
T1_gen FRCP kernel (canonical ordering only — the 9-way refine
enumeration collapses for FRCP per plan §9.2); numbers for FRCP parity
switch at FP4/FP6/FP8 will repopulate on the next
`scripts/run_lowbit.py` run. Earlier readouts of `ε = ∞` for those
formats were an artefact of the recip path not being evaluated in the
lever, not a genuine intermediate-overflow — replace with the
regenerated numbers before citing.

## Plan correction surfaced by the implementation

The standalone low-bit plan's target-formats table listed FP6 E2M3
= 12 and FP8 E5M2 = 112 positive normals, which disagree with both
IEEE-style enumeration (16 / 120) and OCP MX semantics (24 / 120).
Commit `89e8db3` corrected these and added a paragraph clarifying
the enumeration convention; that paragraph now lives in
`FRGR-PLAN.md`.

Day [2023]'s own errata (eq (57) ε last-digit typo, Listing 5
coefficients unreadable in the rendered PDF) are catalogued
separately in `DAY-2023-ERRATA.md`.

## Outstanding

- **fp24** for the validation arm. Estimated 36 h on 32 cores with the
  current Rust kernel; memory-flat single-K streaming path already
  verified at fp32. Ready to kick off whenever the machine can spare
  the CPU. Will emit with `eps_opt_kind = "local"` like fp16/18/20.
- **FRCP §9.2 refine-ordering enumeration**. The 9-way enumeration
  collapses for reciprocal but the effectively-distinct subset isn't
  separately ported into the Rust kernel yet, so the reference-arm
  "best refine ordering" lever reports a delta of 0 for recip. Cheap
  to add; useful for completeness of the recip spec sheet.

## Reproducing

All results were produced from a clean checkout with Python 3.10,
mpmath 1.4.1, numpy 2.2.6, and Rust 1.75. `.venv/bin/maturin develop
--release` builds `dayval_rust`; `pytest tests/` runs all 35 gates
(approximately 4 minutes, dominated by the fp32 kernel-replay test);
`python scripts/run_phase1.py` regenerates the validation-arm primary
table; `python scripts/run_lowbit.py` regenerates the reference-arm
spec sheets. The LaTeX source of Day [2023] is kept in `sources/` for
anyone who wants to verify the numerical checks against the paper's
typeset equations.
