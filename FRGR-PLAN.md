# FRGR Cross-Format Validation and Reference Plan

Two arms of one project, sharing one software harness and one
Algorithm 3 reference port.

**Validation arm**: does Day [2023]'s parity-indexed analytic family
track the finite-format optimum as format width varies for FRSR
(`(a, b, n) = (1, 2, 1)`), and which §9 levers port across widths
vs. are fp32-shaped?

**Reference arm**: exhaustive spec sheets for `1/√x` and `1/x` at the
five production low-bit formats (FP4 E2M1, FP6 E2M3, FP6 E3M2, FP8
E4M3, FP8 E5M2), covering every algorithm tier from the bare seed
through degree-2 general polynomial refinement.

Day [2023] generalises FRSR to FRGR: compute `x^(−a/b)` for coprime
`a, b ∈ ℤ⁺` with polynomial refinement of degree `n`. Algorithm 3
gives the magic constant `c` and coefficients `c₀…cₙ` under exact
arithmetic; §9 covers finite-precision tuning. The two arms are
complementary: the reference arm's exhaustive low-bit data is the
B2 gold standard for the validation arm, and the validation arm's
analytic family is the seed for the reference arm's §9.3 parity
lever.

## Audience

- **Kernel author**: opens the per-format spec sheet, picks the
  tier matching their op budget, copies `K` and coefficients.
- **Hardware designer**: opens the cross-format summary, compares
  the format-intrinsic floor to the achievable tier curve, decides
  LUT-vs-FISR/FRCP per primitive.
- **Numerical library author**: reports the format-intrinsic floor
  alongside the documented kernel error, flags gaps.
- **Numerical-methods researcher**: reads the validation arm's
  primary CSV and Q1/Q2/Q3 findings.

## Scope

Primary key is `(E, M, bias)`, not width. Positive-normal enumeration
is IEEE-style: biased exp in `[1, 2^E − 2]`, reserving the all-ones
exponent for inf/NaN. OCP MX semantics for FP4 E2M1, FP6 E2M3, FP6
E3M2, and FP8 E4M3 encode normals in the all-ones exponent too; we
use IEEE-style throughout, and the kernel evaluates correctly when
intermediate results land in the reserved region.

| Format   | Layout  | Bias | Pos. normals   | Arm |
|---|---|---|---|---|
| FP4 E2M1 | 1.2.1   | 1    | 4              | both |
| FP6 E2M3 | 1.2.3   | 1    | 16             | reference |
| FP6 E3M2 | 1.3.2   | 3    | 24             | both |
| FP8 E4M3 | 1.4.3   | 7    | 112            | both |
| FP8 E5M2 | 1.5.2   | 15   | 120            | reference |
| fp16     | 1.5.10  | 15   | 30,720         | validation |
| fp18     | 1.6.11  | 31   | 126,976        | validation |
| fp20     | 1.6.13  | 31   | 507,904        | validation |
| fp24     | 1.7.16  | 63   | 8,257,536      | validation |
| fp32     | 1.8.23  | 127  | 2,130,706,432  | validation (replication-only) |

Target functions: rsqrt (`(a, b) = (1, 2)`) for both arms; reciprocal
(`(a, b) = (1, 1)`) for the reference arm. Reciprocal runs one op
cheaper than rsqrt at every tier (consequence of `a = b = 1`); Day's
§7.3 reciprocal accuracy is a consequence of that cheaper column.

## Research questions (validation arm)

- **Q1 (family tracking)**. For each format, does exactly one of
  Day's parity-conditioned analytic representatives track the
  finite-format optimum? Does the winning parity depend on
  `(E, M, bias)`?
- **Q2 (heuristic portability)**. Which §9 levers port across
  formats, and which are fp32-shaped? The B3 ablation ladder
  separates them.
- **Q3 (grid dominance)**. At what `(E, M)` does the mantissa raster
  dominate — brute-force `K` becomes non-unique (large tie sets),
  and §9 heuristics stop helping?

## Phase A — implementation

### A1. Algorithm 3 reference port

Port §5 verbatim. For any `(a, b, n)` return `(c, c₀…cₙ, zmin, zmax,
ε_theory)` as exact-arithmetic quantities. Validate against §7.2
eq (57): FRSR `(1, 2, 1)` must yield `c₀ ≈ 1.68191391`,
`c₁ ≈ −0.703952009`, `ε_theory = 6.500702958850 × 10⁻⁴`.

### A2. Parametric minifloat arithmetic

Bit-cast helpers parameterised on `(E, M, bias)` with RNE throughout.
Correct handling of positive normals (the FRSR domain per §3 fn 2),
subnormals, zero, infinity, NaN. Enumerator yielding all positive
normals for a given format.

### A3. Parametric FRGR kernel

`frgr(x, format, target, tier, K, coefs)` in format-native arithmetic
covering both target functions and both coarse-approximation
orderings from §9.2:

- Shift-then-subtract (Listing 1): `Y = K − (X >> 1)` (rsqrt)
- Subtract-then-shift (§9.2): `Y = (C' − X) >> 1`, `C' ∈ {2K, 2K+1}`
  (rsqrt; the extra effective bit of `K` precision is a separate
  ablation dimension, not folded into "tuning")
- Full-width subtract (reciprocal): `Y = K − X`

### A4. Execution semantics

All results come from a software IEEE-style harness. Hardware math is
not used, so numbers are independent of FMA, ISA, or compiler
instruction selection — Day §2 fn 1. For fp32, Listing 5 is
replicated against a host-compiled C kernel as a second sanity check,
but reported cross-width numbers come from the software harness.
No FMA anywhere.

## Phase B — data collection

### B0. Format-intrinsic floor per target function

For each format × function, for every positive normal `x`, compute
`f(x)` exactly and round to the nearest representable format value;
peak relative error per Day eq (10) is the **format floor**
`ε_floor(f)`. This is what any full LUT achieves. Record the
worst-case witness. `ε_floor(1/x)` and `ε_floor(1/√x)` generally
differ — different rounding geometries.

### B1. Analytic family (validation arm)

Run Algorithm 3. For each `s ∈ {−1, 0}` (the two parity
representatives for FRSR; more `s` values are redundant modulo
exact power-of-two scaling per §9.3):

- Exact-arithmetic `(c, c₀, c₁)` and `ε_theory`.
- Format-quantised `(K_s, ĉ₀_s, ĉ₁_s)` via eq (62) and RNE on
  coefficients.
- Realized peak error `ε_realized(analytic, s)` over positive normals.

### B2. Finite-format optima at each algorithm tier

Tiers are ordered by op count; `z = x · y^b` (b=2 rsqrt, b=1 recip)
with canonical left-associative ordering:

| Tier             | Form                                 | Ops rsqrt / recip |
|---|---|---|
| T0_monic         | `y`                                  | 0 / 0 |
| T0_scale         | `y · k`                              | 1 / 1 |
| T1_monic         | `y · (a − z)`                        | 3 / 2 |
| T1_gen           | `y · (c₀ + c₁ · z)`                  | 4 / 3 |
| T2_monic Horner  | `y · (a + z · (z − b))`              | 5 / 4 |
| T2_gen Horner    | `y · (c₀ + z · (c₁ + z · c₂))`       | 6 / 5 |

Search domain: `K` over all `W`-bit integers; candidate coefficients
over the positive-normal encodings (both signs). Subnormals, zero,
inf, NaN are excluded as candidates but the kernel must still
evaluate correctly when intermediate values land there.

- **FP4, FP6, FP8 (both arms)**: exhaustive over `(K, c₀, …, cₙ)`
  for every tier. Gold standard; feeds the validation arm's B2 at
  these widths directly.
- **fp16, fp18, fp20, fp24 (validation arm)**: T1_gen only, two
  passes. `K`-only exhaustive with `(c₀, c₁)` pinned to quantised
  analytic values, one sweep per parity — clean Q1 probe. Joint
  local search over `(K, c₀, c₁)` for the Q2/Q3 probe. Seeds: both
  parity analytic points, plus 8 starts sampled uniformly from a box
  of radius `2^⌈M/2⌉` ULPs around each parity seed in each
  coordinate. Neighborhood: one ULP step in any single coordinate
  (6 neighbors per point). Descent: steepest improvement in peak
  `ε` per step, ties broken by lexicographic coordinate order.
  Stopping: no improvement in 32 consecutive steps. Results labeled
  **local**, not global.
- **fp32 (validation arm)**: replicate Day's Listing 5 and verify the
  pipeline against his published error.

Stopping rule: when a tier's peak `ε` equals `ε_floor(f)`, higher
tiers cannot improve; report the tier as saturated. Early saturation
at low bit widths is itself an operational result.

### B3. Implementation-variant ablation (§9 levers)

Applied on top of B2's T1_gen best, stepwise. Each step gets an
independent column in the validation arm's primary table and a line
in the reference arm's spec sheet:

1. Winning parity from B1.
2. `C'` extra bit via subtract-then-shift (§9.2, rsqrt only).
3. The 9 algebraically-equivalent orderings of `c₁ · x · y · y`
   under the Newton-style refinement form (§9.2).
4. Coefficient fine-tuning (§9.4 search, within format).

The table must show how much `ε_realized` moves at each step so we
can identify which mechanism is portable and which is fp32-shaped.

### B4. Witness-preserving records

At every recorded optimum store: the input `x*` realising the peak
error; exact minimizer set (`K` values achieving the minimum peak
exactly — well-defined in the bit-exact software harness); near-
optimal band count (`K` values within one ULP of the minimum);
second-worst error and number of inputs within one ULP of the worst.
This is what Q3 actually needs — degeneracy structure, not just a
scalar `ε`.

## Phase C — analysis and deliverables

Primary table keyed on `(E, M, bias)` (validation arm, rsqrt):

| E | M | bias | `K_{s=−1}` | `K_{s=0}` | `K_opt` | winning `s` | `ε_theory` | `ε_real(analytic)` | `ε_real(+C')` | `ε_real(+ord)` | `ε_real(+coef)` | `ε_opt` | tie-set | `x*` |

Derived views:

- **Q1**: `ε_real(analytic, winning s) / ε_opt` vs `M` at fixed
  `E/bias` ratio, and vs `E` at fixed `M`. Expectation: the ratio
  trends toward 1 as the mantissa grid refines. The shape of any
  approach — or any failure to approach — is the result.
- **Q2**: Δε per ablation-ladder step vs `(E, M)`. A step that helps
  at fp32 but flatlines at fp16 is fp32-shaped.
- **Q3**: exact-minimizer set size and near-optimal band count vs
  `(E, M)`. Growth as `M` shrinks is the grid-dominance signal;
  exponent-layout effects may introduce non-monotone structure.

Per-format spec sheet (reference arm): floor, tier table (T0..T2
monic + gen, both functions), §9 ablation block on T1_gen baseline,
format-peculiarity notes.

Cross-format summary (reference arm): one table per function with
`ε_floor`, `ε_T0_monic`, `ε_T1_gen_best`, recommended `K`, gap to
floor, LUT entries for floor.

**Deliverables**:
1. Reference Algorithm 3 port + parametric kernel + harness. Runs on
   a laptop for formats up to fp20; fp24 overnight on 32 cores.
2. Primary CSV + Q1/Q2/Q3 derived views.
3. Per-format spec sheets and cross-format summary CSVs.
4. Short writeup (`RESULTS.md`): Q1/Q2/Q3 findings, reference-arm
   saturation / lever-payoff findings, paper and plan corrections
   surfaced during implementation.

## Out of scope

- Kadlec non-Newton factorings (Day §9.2).
- Host/hardware-dependent kernels. No FMA.
- Phase 2: fp28 / fp64 / bf16; reciprocal across wide widths;
  reciprocal cube root `(a, b) = (1, 3)` (see Day Listing 10 fn 11
  on Moroz et al.'s suboptimal magic constant); `(a, b) = (2, 3)`
  (§10.4); `n ≥ 2` at wide formats; the 2-iteration FRSR (§10.5);
  monic vs general polynomial tradeoff (§6 Figure 8); format-
  peculiar edge cases (E4M3's truncated top-of-range, behavior on
  subnormal inputs); block-scaling interactions.

## Citations

- Day [2023], *Generalising the Fast Reciprocal Square Root
  Algorithm*, arXiv:2307.15600 — parametric FRGR, §9 levers.
- OCP Microscaling Formats (MX) Specification v1.0, September 2023
  — FP4/FP6/FP8 bit layouts, biases, and special-value handling.
- Micikevicius et al. [2022], *FP8 Formats for Deep Learning*,
  arXiv:2209.05433 — E4M3 deviations from IEEE (no ±∞, one NaN).
- Schacham, *16-bit Fast Inverse Square Root* (2025-08-23,
  [ashdnazg.github.io/articles/25/16-bit-Fast-Inverse-Square-Root](https://ashdnazg.github.io/articles/25/16-bit-Fast-Inverse-Square-Root))
  — fp16 monic K-only brute-force, `K = 0x59B7` / `ε ≈ 2.8 × 10⁻³`.
  Pipeline self-check at fp16 (`test_schacham_fp16_monic`).
