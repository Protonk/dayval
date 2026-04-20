"""Per-format spec sheet emitter for LOW-BIT-FRGR-REFERENCE-PLAN.

Formats one sheet per format covering both rsqrt (1/√x) and reciprocal
(1/x), per the plan's template. §9 implementation-variant ablation
(M3) is stubbed — it runs on the best tier and reports deltas if the
caller populates `ablation` in the sheet; the spec-sheet framework is
here to be filled in as §9 variants are added.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from typing import Optional

from . import lowbit, minifloat as mf


@dataclass
class SheetSection:
    target: str                  # "rsqrt" | "recip"
    floor: lowbit.FormatFloor
    tiers: list[lowbit.TierResult] = field(default_factory=list)
    ablation: dict = field(default_factory=dict)  # lever name -> Δε (TODO)


@dataclass
class Sheet:
    format_name: str
    fmt: mf.Format
    sections: list[SheetSection] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _hex_width(fmt: mf.Format) -> int:
    return max(1, (fmt.width + 3) // 4)


def _fmt_coefs(tier: lowbit.TierResult, fmt: mf.Format) -> str:
    if not tier.coefs_values:
        return "—"
    hw = _hex_width(fmt)
    pieces = []
    for v, b in zip(tier.coefs_values, tier.coefs_bits):
        pieces.append(f"{v:+.6g}(0x{b:0{hw}x})")
    return " ".join(pieces)


def format_sheet(sheet: Sheet) -> str:
    buf = StringIO()
    fmt = sheet.fmt
    pn = len(mf.positive_normals_bits(fmt))
    hw = _hex_width(fmt)

    buf.write(f"FORMAT: {sheet.format_name}\n")
    buf.write(f"Layout: 1.{fmt.E}.{fmt.M}, bias {fmt.bias}\n")
    buf.write(f"Positive normals (IEEE-style): {pn}\n")
    buf.write("\n")

    for section in sheet.sections:
        if section.target == "rsqrt":
            buf.write("==== RSQRT (1/sqrt(x)) ====\n")
        else:
            buf.write("==== RECIPROCAL (1/x) ====\n")
        buf.write(f"Format-intrinsic floor:\n")
        buf.write(f"  eps_floor   = {section.floor.eps_floor:.9e}\n")
        buf.write(f"  worst input = 0x{section.floor.x_star:0{hw}x}\n")
        buf.write("\n")

        if section.tiers:
            buf.write("Algorithm tiers (global optima by exhaustive search):\n")
            header = f"  {'Tier':<16} | ops | {'K':<{hw+2}} | {'coefs':<40} | eps         | x*\n"
            buf.write(header)
            buf.write("  " + "-" * (len(header) - 3) + "\n")
            for t in section.tiers:
                buf.write(
                    f"  {t.tier:<16} | {t.ops:<3} | "
                    f"0x{t.K:0{hw}x} | {_fmt_coefs(t, fmt):<40} | "
                    f"{t.eps:.4e} | 0x{t.x_star:0{hw}x}\n"
                )
            buf.write("\n")

        if section.ablation:
            buf.write("Implementation variants (M3):\n")
            for name, delta in section.ablation.items():
                buf.write(f"  {name:<30}: Δε = {delta:+.3e}\n")
            buf.write("\n")
        else:
            buf.write("Implementation variants (M3): not yet populated.\n\n")

    if sheet.notes:
        buf.write("==== Notes ====\n")
        for n in sheet.notes:
            buf.write(f"  - {n}\n")

    return buf.getvalue()


def build_sheet(fmt: mf.Format, format_name: str,
                tiers: list[str] = lowbit.RSQRT_TIERS,
                notes: Optional[list[str]] = None) -> Sheet:
    """Run M1 (floor) + M2 (all tiers) for both rsqrt and recip at `fmt`."""
    sheet = Sheet(format_name=format_name, fmt=fmt, notes=notes or [])
    for target in ("rsqrt", "recip"):
        floor = lowbit.format_floor(fmt, target)
        section = SheetSection(target=target, floor=floor)
        for tier in tiers:
            r = lowbit.tier_exhaustive(fmt, target, tier)
            section.tiers.append(r)
        sheet.sections.append(section)
    return sheet


def cross_summary_rows(sheets: list[Sheet]) -> list[dict]:
    """Cross-format summary rows (one per sheet per target)."""
    rows = []
    for sheet in sheets:
        for section in sheet.sections:
            t1 = next((t for t in section.tiers if t.tier == "T1_gen"), None)
            t0 = next((t for t in section.tiers if t.tier == "T0_monic"), None)
            rows.append({
                "format_name": sheet.format_name,
                "target": section.target,
                "eps_floor": section.floor.eps_floor,
                "eps_T0_monic": t0.eps if t0 else float("nan"),
                "eps_T1_gen": t1.eps if t1 else float("nan"),
                "K_T1_gen_hex": (f"0x{t1.K:0{_hex_width(sheet.fmt)}x}"
                                if t1 else ""),
                "gap_to_floor": ((t1.eps - section.floor.eps_floor) if t1
                                else float("nan")),
                "lut_entries_for_floor": len(
                    mf.positive_normals_bits(sheet.fmt)
                ),
            })
    return rows
