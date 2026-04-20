# Low-Bit FRGR Reference Plan

Covering reciprocal square root (`1/√x`) and reciprocal (`1/x`) at the
five production low-bit float formats. "FRGR" is Day's own term for
the generalized algorithm family that subsumes both — `(a,b) = (1,2)`
for rsqrt, `(a,b) = (1,1)` for reciprocal.

## What this produces

A numerical reference for the two division-free primitives most used
in ML kernels at low precision — `1/√x` and `1/x` — on all five
production float formats at 8 bits and below. Per target function, per
format: exhaustively-optimal magic constants, coefficients, worst-case
errors, worst-case witnesses, and cost-versus-accuracy frontiers. A
drop-in spec sheet per format covering both functions. Across formats,
a single cross-table for hardware designers deciding what LUT budget
buys for each primitive.

Reciprocal is first-class here, not deferred: attention denominators,
softmax normalization, and any 1/d scaling hit reciprocal far more
often than rsqrt. At fp32 Day shows reciprocal is naturally ~6× more
accurate than FRSR per op (§7.3), and the same geometry should favor
it at low bit widths.

This is a reference-production effort, not a hypothesis test. It shares
infrastructure with the Day cross-validation plan but has different
success criteria: correctness, completeness, and presentation, not
confirmation or refutation.

## Audience and deliverables

**Kernel author** ("I need `1/x` on FP8 E4M3 inside my softmax
kernel, or rsqrt on FP6 E3M2 for a layer-norm variant"): opens the
per-format spec sheet, picks the target-function section, copies `K`
and coefficients for the algorithm tier matching their op budget,
reads the worst-case error and worst-case witness, ships.

**Hardware designer** ("I'm deciding whether to spend gate area on a
division or rsqrt LUT"): opens the cross-format summary, reads the
zero-LUT error floor for each format × function, compares to the
LUT-of-size-N curve, makes the call per primitive.

**Numerical library author** ("I need to document what my rsqrt and
reciprocal primitives guarantee"): opens the per-format sheet, reports
the
format-intrinsic floor alongside the achievable algorithm error, flags
any gap.

## Target formats

Operationally-live plus near-live research formats, all standalone
(block-scaling framework ignored — rsqrt on the element format itself
is what we characterize):

| Name | Layout | Bias | Pos. normals | Status |
|---|---|---|---|---|
| FP4 E2M1 | 1.2.1 | 1 | 4 | OCP MXFP4, NVFP4 — production (Blackwell) |
| FP6 E2M3 | 1.2.3 | 1 | 16 | OCP MXFP6 — production (MI355, B200) |
| FP6 E3M2 | 1.3.2 | 3 | 24 | OCP MXFP6 — production |
| FP8 E4M3 | 1.4.3 | 7 | 112 | OCP/NVIDIA/Arm/Intel — production (H100+) |
| FP8 E5M2 | 1.5.2 | 15 | 120 | OCP/NVIDIA/Arm/Intel — production (H100+) |

Positive-normal counts are IEEE-style: biased exponent in `[1, 2^E − 2]`,
all mantissas, reserving `2^E − 1` for inf/NaN. OCP MX semantics for FP4
E2M1, FP6 E2M3, FP6 E3M2, and FP8 E4M3 encode normals in the all-ones
exponent too (no ±∞; NaN reserved only at the top pattern of E4M3),
which would give counts of 6 / 24 / 28 / 119 respectively. We use IEEE-
style enumeration throughout — the all-ones exponent is excluded as a
coefficient/input candidate but the kernel still evaluates correctly
when intermediate results land there.

## Methodology

Every quantity below is computed by exhaustive enumeration, because at
these widths it is tractable. Software IEEE-style harness, round-to-
nearest-even, no FMA, bit-exact reproducibility.

### M1. Format-intrinsic floors per target function

For each format and each target function `f ∈ {1/√x, 1/x}`, for every
positive normal input `x`, compute the exact `f(x)` and round to the
nearest representable format value. The peak relative error over
inputs is the **format floor** `ε_floor(f)`: no implementation of `f`
at any computational budget can do better. Also record the worst-case
witness `x_floor*(f)`.

These numbers exist independent of any FISR/FRCP algorithm. They are
what a full lookup table achieves for each primitive. Note that
`ε_floor(1/x)` and `ε_floor(1/√x)` are generally different even for
the same format — different functions have different rounding
geometries.

### M2. Algorithm tiers

For each format and each target function, find the globally optimal
`(K, c₀, …, cₙ)` for each algorithm tier by exhaustive enumeration.
Tiers are ordered by arithmetic op count (post bit-cast, excluding any
input precomputation shared by all NR variants).

**Rsqrt tiers** (`(a,b) = (1,2)`; `xhalf = x/2` precomputed):

| Tier | Form | Ops | Description |
|---|---|---|---|
| T0_monic | return `L⁻¹(K - (X>>1))` | 0 | Bare seed |
| T0_scale | `y₀ × k` | 1 | Seed with post-scale |
| T1_monic | `y₀ × (a - xhalf·y₀²)` | 3 | Monic degree-1 poly |
| T1_gen | `y₀ × (c₀ - c₁·xhalf·y₀²)` | 4 | Standard Quake FISR |
| T2_monic | `y₀ × (a + z(z - b))`, `z = xhalf·y₀²` | 5 | Monic degree-2 |
| T2_gen | degree-2 general | 6 | Full flexibility |

**Reciprocal tiers** (`(a,b) = (1,1)`; no `xhalf` needed):

| Tier | Form | Ops | Description |
|---|---|---|---|
| T0_monic | return `L⁻¹(K - X)` | 0 | Bare seed |
| T0_scale | `y₀ × k` | 1 | Seed with post-scale |
| T1_monic | `y₀ × (a - x·y₀)` | 2 | Monic degree-1 |
| T1_gen | `y₀ × (c₀ + c₁·x·y₀)` | 3 | Full degree-1 NR |
| T2_monic | `y₀ × (a + z(z - b))`, `z = x·y₀` | 4 | Monic degree-2 |
| T2_gen | degree-2 general | 5 | Full flexibility |

Reciprocal tiers run one op cheaper than rsqrt counterparts —
consequence of `a=b=1` simplifying the iteration. This is the op-count
reason Day's §7.3 reciprocal is more accurate per op than FRSR: it's
in a cheaper column, not just a better row.

Note also that `K - X` with no shift means the reciprocal magic
constant is a full-width subtraction, not the half-width subtract-
then-shift of FRSR. This changes what "extra bit of precision" means
in the implementation-variant ablation (§9.2 below): the subtract-then-
shift trick doesn't apply, and the `C'` lever drops out for reciprocal.

For each format × function × tier, record: `K`, coefficients, peak `ε`,
worst-case witness `x*`, exact-minimizer count, near-optimal band count
(within one ULP of peak `ε`).

Stopping rule: when a tier's peak `ε` equals the format floor
`ε_floor(f)`, higher tiers cannot improve and we report the tier as
saturated. At fp4 and fp6 this is likely to happen early for both
target functions; confirming saturation and at which tier is itself a
useful operational result.

### M3. Implementation variants

For the tier that hits the operational sweet spot at each format
(which tier this is depends on the format), apply the Day §9 levers
as an ablation ladder. Each lever gets its own line in the spec sheet
so practitioners can choose which ones are worth the implementation
complexity:

- **§9.2 shift order (rsqrt only)**: shift-then-subtract `K - (X>>1)`
  versus subtract-then-shift `(C' - X) >> 1` with `C' ∈ {2K, 2K+1}`.
  Does not apply to reciprocal since there's no shift.
- **§9.2 product orderings**: 16 algebraically-equivalent orderings
  of `c₁·x·y·y` and Kadlec's factored form.
- **§9.3 parity choice**: for the FRSR magic-constant derivation,
  `s` even versus `s` odd.

Record: the best combination, and the individual `Δε` attributable to
each lever. Levers whose `Δε` is zero at every format are noise and
can be dropped from kernel-author guidance.

### M4. Cost-versus-accuracy frontier

Per format × function, plot `log₁₀ ε` against op count across tiers.
Overlay the format floor as a horizontal line. Two plots per format.
This visualization tells hardware designers at a glance where each
format × function saturates — some may be fully solved by a single
integer subtraction, others have a meaningful curve.

## Cross-format summary tables

Two reference tables for hardware designers, one per target function.

**Rsqrt summary**:

| Format | `ε_floor` | `ε_T0_monic` | `ε_T1_gen_best` | `K_recommended` | Gap to floor | LUT entries for floor |
|---|---|---|---|---|---|---|

**Reciprocal summary**:

| Format | `ε_floor` | `ε_T0_monic` | `ε_T1_gen_best` | `K_recommended` | Gap to floor | LUT entries for floor |
|---|---|---|---|---|---|---|

"LUT entries needed for floor" is the positive normals count — the
size of the table-based implementation that guarantees `ε_floor`. For
FP4 with 4 entries, FISR/FRCP arguments are academic; the designer
picks the table. For FP8 variants with 112 entries, the tradeoff is
real.

## Per-format spec sheet template

```
FORMAT: FP8 E4M3
Layout: 1.4.3, bias 7
Positive normals: 112
Special values: no ±∞, single NaN bit-pattern

==== RSQRT (1/√x) ====
Format-intrinsic floor:
  ε_floor       = <value>
  worst input   = 0x<hex> = <decimal>

Algorithm tiers (global optima by exhaustive search):
  Tier     | ops | K    | c₀    | c₁    | ε        | x*     | exact min | near-opt band
  T0_monic | 0   | 0x54 | —     | —     | <value>  | 0x??   | <n>       | <n>
  T0_scale | 1   | …    | …     | —     | …        | …      | …         | …
  T1_monic | 3   | …    | …     | —     | …        | …      | …         | …
  T1_gen   | 4   | …    | …     | …     | …        | …      | …         | …
  T2_monic | 5   | …    | …     | …     | …        | …      | …         | …
  T2_gen   | 6   | …    | …     | …     | …        | …      | …         | …

Implementation variants (applied to T1_gen best):
  Baseline (shift-then-subtract, s=-1, canonical ordering): ε = <value>
  + §9.2 C'-extra-bit:    Δε = <value>
  + §9.2 best ordering:   Δε = <value>
  + §9.3 parity switch:   Δε = <value>
  Combined best:          ε = <value>

==== RECIPROCAL (1/x) ====
Format-intrinsic floor:
  ε_floor       = <value>
  worst input   = 0x<hex> = <decimal>

Algorithm tiers (global optima by exhaustive search):
  Tier     | ops | K    | c₀    | c₁    | ε        | x*     | exact min | near-opt band
  T0_monic | 0   | 0x?? | —     | —     | <value>  | 0x??   | <n>       | <n>
  T0_scale | 1   | …    | …     | —     | …        | …      | …         | …
  T1_monic | 2   | …    | …     | —     | …        | …      | …         | …
  T1_gen   | 3   | …    | …     | …     | …        | …      | …         | …
  T2_monic | 4   | …    | …     | …     | …        | …      | …         | …
  T2_gen   | 5   | …    | …     | …     | …        | …      | …         | …

Implementation variants (applied to T1_gen best):
  Baseline (full-width subtract, s=-1, canonical ordering): ε = <value>
  + §9.2 best ordering:   Δε = <value>
  + §9.3 parity switch:   Δε = <value>
  Combined best:          ε = <value>
  (§9.2 C'-extra-bit does not apply — no shift in reciprocal)

==== Notes on format peculiarities ====
  E4M3 drops ±∞ and reserves one NaN pattern, so the top of the
  exponent range has one fewer finite value than a strict IEEE 1.4.3
  format would. FISR/FRCP behavior near the max normal is
  [cleanly / degenerately].
```

## Computational budget

Per format × function, worst case (T2_gen, all coefficient pairs, all
inputs, all shift/order/parity variants). Budget roughly doubles from
the single-function case since both functions share the harness but
need separate sweeps:

| Format | `K` search | coef pair search | inputs | variant multiplier | evals per function |
|---|---|---|---|---|---|
| FP4 E2M1 | 2⁴ | 2⁸ | 4 | ~32 | ~130K |
| FP6 (both) | 2⁶ | 2¹² | 12 or 24 | ~32 | ~100–200M |
| FP8 E4M3, E5M2 | 2⁸ | 2¹⁶ | 112 | ~32 | ~60G |

Total across five formats and two target functions is ~600G
evaluations worst case. Vectorized on a laptop, hours; on a GPU,
minutes. Still CI-job scale.

## Interaction with the Day plan

Complementary, not redundant. The Day plan asks "does the analytic
framework track brute-force optima across widths." This plan asks
"what are the brute-force optima at each operational format." The
brute-force arm of the Day plan and the enumeration arm of this plan
can share a single harness and run once. Outputs serve different
audiences: the Day plan produces a cross-width analysis for
numerical-methods readers; this plan produces spec sheets for kernel
authors.

Practical sequencing: build the harness for this plan first. It is
simpler (no analytic-family parity bookkeeping, no cross-width
normalization), produces operational value immediately, and its
brute-force data feeds the Day plan's B2 stage directly.

## Phase 2 — deferred

- Reciprocal cube root (`a=1, b=3`) at these formats. Less demand at
  ML precisions than rsqrt or reciprocal; Day §10.4 covers the fp32
  case. Same methodology would apply.
- Format-peculiar edge cases: E4M3's truncated top-of-range, what
  happens when FISR/FRCP intermediates underflow into subnormals,
  behavior of each tier on the subnormal domain that Day excludes.
- Two-iteration variants at these formats — likely saturated at T1
  for fp6 and below, so this mostly matters for fp8.
- Interaction with block-scaling: rsqrt or reciprocal of a full MX
  block, where the shared E8M0 exponent simplifies part of the
  computation.

## Citations

- Day [2023]: methodology for parametric FISR, algorithm tiers,
  implementation variants (§9 levers).
- OCP Microscaling Formats (MX) Specification v1.0, September 2023:
  authoritative definition for FP4 E2M1, FP6 E2M3, FP6 E3M2, FP8 E4M3,
  FP8 E5M2 bit layouts, biases, and special-value handling.
- Micikevicius et al. [2022], "FP8 Formats for Deep Learning"
  (arXiv:2209.05433): joint NVIDIA/Arm/Intel FP8 proposal; authoritative
  for E4M3's deviations from IEEE (no ±∞, one NaN).
