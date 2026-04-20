# dayval

Code and results for the FRGR cross-format validation and low-bit
reference project; plan in `FRGR-PLAN.md`, findings in `RESULTS.md`.
Build with `pip install -e . && (cd rust_kernel && maturin develop
--release)`; regenerate data with `scripts/run_phase1.py` (validation
arm) or `scripts/run_lowbit.py` (reference arm). Gates in `tests/`.

```
FRGR-PLAN.md         plan
RESULTS.md           findings writeup
dayval/              Python package (algorithm3, minifloat, frsr,
                     magic, sweep, lowbit, specsheet, tables)
rust_kernel/         PyO3 extension (Rust)
scripts/             drivers
tests/               pytest gates
results/             emitted CSVs and spec sheets
sources/             paper PDF + gitignored LaTeX source
```
