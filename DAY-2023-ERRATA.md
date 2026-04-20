# Day [2023] errata

Two mechanical errata in Mike Day, *Generalising the Fast Reciprocal
Square Root Algorithm* (arXiv:2307.15600v1, 28 Jul 2023), surfaced
while reproducing the paper's numbers. Gated by tests in this repo.

## Eq (57): ε last-digit typo

The paper prints

```
ε ≈ 6.50070298 × 10⁻⁴
```

The true minimax error for the FRSR linear case to 13 significant
figures is

```
ε = 6.500702958850 × 10⁻⁴
```

which rounds to `6.50070296 × 10⁻⁴` at 9 sf, not `…298`. Verified by:

- The eq (56) closed form evaluated at 100 decimal-place precision
  in mpmath.
- An independent Remez iteration at the same precision.
- Direct equioscillation check: at the three extrema `zmin = 3/4`,
  `z_mid = −c₀/(3 c₁)`, and `zmax = 27/32`, the residual
  `1 − √z · (c₀ + c₁ z)` has magnitude
  `6.500702958850 × 10⁻⁴` at all three, with alternating signs.

The `c₀` and `c₁` values in eq (57) are correct to their stated 9 sf;
only the printed ε is off by one unit in the last digit.

Gated by `tests/test_algorithm3.py::test_frsr_s_minus_one_eq57` with
a `1e-15` tolerance on the true value.

## Listing 5: PDF-rendered coefficients unreadable on screen

The PDF of Listing 5 is unreadable at typical on-screen DPI. The
glyph sequence that rounds through OCR to

```
return y * (1.1891763f - x*y*y*0.24885956f);
```

produces peak relative error `7.484 × 10⁻⁴` over all positive fp32
normals, not Day's reported `6.501791 × 10⁻⁴`.

The arxiv LaTeX source (`results.tex`) has

```
return y * (1.1893165f - x*y*y*0.24889956f);
```

which reproduces `ε = 6.501791 × 10⁻⁴` exactly at witness
`x* = 0x01401a9f = 3.642883…×10⁻³⁸`, verified by exhaustive C scan
over all 2,130,706,432 positive fp32 normals (`gcc -O2 -fno-fast-math
-mno-fma`). The corresponding listings in the paper's §10 text box
are similarly sensitive to rendering quality; use the LaTeX source
as the authoritative reference.

Gated by `tests/test_frsr.py::test_listing5_exact_witness` and
`tests/test_frsr.py::test_listing5_matches_host_c_kernel`.
