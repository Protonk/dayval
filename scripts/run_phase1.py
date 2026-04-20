"""Run Phase 1 sweeps and emit the primary CSV plus derived views.

Usage:
    python scripts/run_phase1.py [--formats fp4,fp6,fp8,fp16,fp18,fp20,fp24,fp32]
                                 [--output-dir results/]

Defaults to all formats except fp24 (which needs a long-running job). To
include fp24, pass `--formats ...,fp24` explicitly.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dayval import minifloat as mf
from dayval import sweep, tables


FORMATS = {
    "fp4": mf.FP4,
    "fp6": mf.FP6,
    "fp8": mf.FP8,
    "fp16": mf.FP16,
    "fp18": mf.FP18,
    "fp20": mf.FP20,
    "fp24": mf.FP24,
    "fp32": mf.FP32,
}

DEFAULT_FORMATS = "fp4,fp6,fp8,fp16,fp18,fp20,fp32"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", default=DEFAULT_FORMATS,
                    help=f"comma-separated; default {DEFAULT_FORMATS}")
    ap.add_argument("--output-dir", default="results",
                    help="directory for CSV output")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen = [s.strip() for s in args.formats.split(",") if s.strip()]
    unknown = set(chosen) - set(FORMATS)
    if unknown:
        print(f"unknown formats: {unknown}", file=sys.stderr)
        return 2

    rows = []
    for name in chosen:
        fmt = FORMATS[name]
        t0 = time.perf_counter()
        print(f"[{name}] {fmt}...", flush=True)
        if name == "fp32":
            # Replication-only: record the analytic parity reps and Day's
            # Listing 5 tuned result; skip the joint local search (intractable
            # at fp32 without a much larger compute budget).
            row = sweep.SweepRow(
                format_name=name,
                E=fmt.E, M=fmt.M, bias=fmt.bias, width=fmt.width,
                positive_normals=len(mf.positive_normals_bits(fmt)),
            )
            a_m1 = sweep.analytic_point(1, 2, 1, s=-1, fmt=fmt)
            a_0 = sweep.analytic_point(1, 2, 1, s=0, fmt=fmt)
            row.K_s_minus_one = a_m1.K
            row.K_s_zero = a_0.K
            row.eps_theory = a_m1.eps_theory
            row.winning_s = -1 if a_m1.eps_realized <= a_0.eps_realized else 0
            winning = a_m1 if row.winning_s == -1 else a_0
            row.eps_real_analytic_winning = winning.eps_realized
            # Day Listing 5 as the fp32 "optimum" reference.
            row.K_opt = 0x5F5FFF00
            row.eps_opt = 6.501791e-4
            row.x_star_opt = 0x01401a9f
            row.notes.append("fp32 is replication-only; K_opt and eps_opt from Day Listing 5")
            rows.append(row)
        else:
            row = sweep.phase1_row(fmt, format_name=name)
            rows.append(row)
        dt = time.perf_counter() - t0
        print(f"  eps_analytic_winning={row.eps_real_analytic_winning:.6e}  "
              f"eps_opt={row.eps_opt:.6e}  "
              f"K_opt=0x{row.K_opt:0{(fmt.width+3)//4}x}  "
              f"elapsed={dt:.1f}s",
              flush=True)

    tables.write_primary(rows, out_dir / "primary.csv")
    tables.write_q1_ratio(rows, out_dir / "q1_ratio.csv")
    tables.write_q2_ladder(rows, out_dir / "q2_ladder.csv")
    tables.write_q3_degeneracy(rows, out_dir / "q3_degeneracy.csv")
    print(f"\nwrote {out_dir}/primary.csv and 3 derived views")
    return 0


if __name__ == "__main__":
    sys.exit(main())
