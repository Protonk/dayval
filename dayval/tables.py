"""Phase C: emit the primary CSV plus derived views.

The primary table is keyed on (E, M, bias) and has the columns specified in
FRGR-PLAN.md (Phase C). Derived views:

- `q1_ratio`: ε_real(analytic, winning s) / ε_opt as a function of (E, M).
- `q2_ladder`: Δε per ablation-ladder step, per format.
- `q3_degeneracy`: exact-minimizer set size and near-optimal band count.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .sweep import SweepRow


PRIMARY_COLUMNS = [
    "format_name", "E", "M", "bias", "width", "positive_normals",
    "K_s_minus_one", "K_s_zero", "K_opt", "winning_s",
    "eps_theory",
    "eps_real_analytic_winning",
    "eps_real_plus_Cprime",
    "eps_real_plus_orderings",
    "eps_real_plus_coef_tune",
    "eps_opt",
    "tie_set_size", "x_star_opt",
]


def write_primary(rows: Iterable[SweepRow], path: Path | str) -> None:
    path = Path(path)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PRIMARY_COLUMNS)
        for r in rows:
            d = asdict(r)
            # Render K columns as hex for easy cross-reference with the paper.
            row = [
                d["format_name"], d["E"], d["M"], d["bias"], d["width"],
                d["positive_normals"],
                f"0x{d['K_s_minus_one']:0{(d['width']+3)//4}x}",
                f"0x{d['K_s_zero']:0{(d['width']+3)//4}x}",
                f"0x{d['K_opt']:0{(d['width']+3)//4}x}",
                d["winning_s"],
                f"{d['eps_theory']:.9e}",
                f"{d['eps_real_analytic_winning']:.9e}",
                f"{d['eps_real_plus_Cprime']:.9e}",
                f"{d['eps_real_plus_orderings']:.9e}",
                f"{d['eps_real_plus_coef_tune']:.9e}",
                f"{d['eps_opt']:.9e}",
                d["tie_set_size"],
                f"0x{d['x_star_opt']:0{(d['width']+3)//4}x}",
            ]
            w.writerow(row)


def write_q1_ratio(rows: Iterable[SweepRow], path: Path | str) -> None:
    path = Path(path)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["format_name", "E", "M", "bias",
                    "eps_analytic", "eps_opt", "ratio"])
        for r in rows:
            if r.eps_opt > 0:
                ratio = r.eps_real_analytic_winning / r.eps_opt
            else:
                ratio = float("nan")
            w.writerow([r.format_name, r.E, r.M, r.bias,
                        f"{r.eps_real_analytic_winning:.9e}",
                        f"{r.eps_opt:.9e}",
                        f"{ratio:.6f}"])


def write_q2_ladder(rows: Iterable[SweepRow], path: Path | str) -> None:
    path = Path(path)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["format_name", "E", "M",
                    "analytic", "plus_Cprime", "plus_orderings",
                    "plus_coef_tune",
                    "delta_Cprime", "delta_orderings", "delta_coef_tune"])
        for r in rows:
            a = r.eps_real_analytic_winning
            b = r.eps_real_plus_Cprime
            c = r.eps_real_plus_orderings
            d = r.eps_real_plus_coef_tune
            w.writerow([r.format_name, r.E, r.M,
                        f"{a:.6e}", f"{b:.6e}", f"{c:.6e}", f"{d:.6e}",
                        f"{a-b:.6e}", f"{b-c:.6e}", f"{c-d:.6e}"])


def write_q3_degeneracy(rows: Iterable[SweepRow], path: Path | str) -> None:
    path = Path(path)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["format_name", "E", "M", "bias",
                    "K_opt_hex", "tie_set_size", "x_star_opt_hex"])
        for r in rows:
            w.writerow([r.format_name, r.E, r.M, r.bias,
                        f"0x{r.K_opt:0{(r.width+3)//4}x}",
                        r.tie_set_size,
                        f"0x{r.x_star_opt:0{(r.width+3)//4}x}"])
