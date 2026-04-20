"""Run LOW-BIT-FRGR-REFERENCE-PLAN sweeps and emit spec sheets + cross-format
summary CSVs.

Usage:
    python scripts/run_lowbit.py [--formats fp4,fp6e2m3,fp6e3m2,fp8e4m3,fp8e5m2]
                                 [--output-dir results/lowbit]
                                 [--tiers T0_monic,T0_scale,T1_monic,T1_gen,T2_monic_horner,T2_gen_horner]

By default runs every tier for every format. At fp8 T2_gen_horner this is
~320G evaluations per function — tens of minutes with Rayon across all cores.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from dayval import minifloat as mf
from dayval import specsheet


FORMATS = {
    "fp4":         ("FP4 E2M1",  mf.FP4_E2M1),
    "fp6e2m3":     ("FP6 E2M3",  mf.FP6_E2M3),
    "fp6e3m2":     ("FP6 E3M2",  mf.FP6_E3M2),
    "fp8e4m3":     ("FP8 E4M3",  mf.FP8_E4M3),
    "fp8e5m2":     ("FP8 E5M2",  mf.FP8_E5M2),
}

DEFAULT_FORMATS = ",".join(FORMATS)
DEFAULT_TIERS = ",".join((
    "T0_monic", "T0_scale", "T1_monic", "T1_gen",
    "T2_monic_horner", "T2_gen_horner",
))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", default=DEFAULT_FORMATS)
    ap.add_argument("--tiers", default=DEFAULT_TIERS)
    ap.add_argument("--output-dir", default="results/lowbit")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen = [s.strip() for s in args.formats.split(",") if s.strip()]
    unknown = set(chosen) - set(FORMATS)
    if unknown:
        print(f"unknown formats: {unknown}", file=sys.stderr)
        return 2

    tiers = [s.strip() for s in args.tiers.split(",") if s.strip()]

    sheets = []
    for name in chosen:
        pretty, fmt = FORMATS[name]
        t0 = time.perf_counter()
        print(f"[{pretty}] building spec sheet...", flush=True)
        sheet = specsheet.build_sheet(fmt, format_name=pretty, tiers=tiers)
        dt = time.perf_counter() - t0
        print(f"  done in {dt:.1f}s", flush=True)
        # Write the per-format sheet.
        out_path = out_dir / f"{name}.txt"
        with out_path.open("w") as f:
            f.write(specsheet.format_sheet(sheet))
        sheets.append(sheet)

    # Cross-format summary CSVs: one row per (format, target).
    rows = specsheet.cross_summary_rows(sheets)
    # Emit a rsqrt table and a recip table separately per the plan.
    for target in ("rsqrt", "recip"):
        path = out_dir / f"summary_{target}.csv"
        target_rows = [r for r in rows if r["target"] == target]
        if not target_rows:
            continue
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(target_rows[0].keys()))
            w.writeheader()
            for r in target_rows:
                w.writerow(r)

    print(f"\nwrote {out_dir} ({len(sheets)} spec sheets + 2 summary CSVs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
