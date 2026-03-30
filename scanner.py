"""
SPX Top-50 Weekly MACD 2nd-Entry Scanner
=========================================
Finds all 2nd (and subsequent) MACD histogram upturn entries for the 50
largest S&P 500 stocks on weekly bars using standard MACD (12, 26, 9).

Usage:
    python scanner.py                        # full scan, last 5 years
    python scanner.py --weeks 104            # last 2 years only
    python scanner.py --recent               # only signals in last 8 weeks
    python scanner.py --refresh-universe     # force re-fetch market caps
"""

import argparse
import os
import sys

import pandas as pd
from tabulate import tabulate

from universe import load_top50
from macd_signals import scan_universe


def parse_args():
    p = argparse.ArgumentParser(description="SPX Top-50 Weekly MACD 2nd-Entry Scanner")
    p.add_argument("--weeks", type=int, default=260, help="Lookback in weeks (default 260 = 5 years)")
    p.add_argument("--recent", action="store_true", help="Print only signals from the last 8 weeks")
    p.add_argument("--refresh-universe", action="store_true", help="Delete cached universe and re-fetch")
    p.add_argument("--output", type=str, default=None, help="Save results to CSV file")
    return p.parse_args()


def main():
    args = parse_args()

    cache = "top50.csv"
    if args.refresh_universe and os.path.exists(cache):
        os.remove(cache)
        print("Universe cache cleared.")

    # ── 1. Load universe ─────────────────────────────────────────────────────
    top50 = load_top50(cache)
    print(f"\nUniverse: {len(top50)} stocks")
    print("  " + ", ".join(top50[:10]) + " ...")

    # ── 2. Scan ───────────────────────────────────────────────────────────────
    print(f"\nScanning {len(top50)} tickers for weekly MACD 2nd-entry signals "
          f"(lookback={args.weeks} weeks) ...\n")

    signals = scan_universe(top50, lookback_weeks=args.weeks)

    if signals.empty:
        print("No signals found.")
        sys.exit(0)

    # ── 3. Filter to recent if requested ─────────────────────────────────────
    if args.recent:
        cutoff = pd.Timestamp.today() - pd.Timedelta(weeks=8)
        signals = signals[signals["date"] >= cutoff]
        print(f"Filtering to last 8 weeks (>= {cutoff.date()}) ...\n")

    # ── 4. Display ────────────────────────────────────────────────────────────
    display = signals.copy()
    display["date"] = display["date"].dt.strftime("%Y-%m-%d")
    display.rename(
        columns={
            "date": "Week",
            "ticker": "Ticker",
            "entry_n": "Entry#",
            "hist": "Hist",
            "prev_hist": "PrevHist",
        },
        inplace=True,
    )

    print(tabulate(display, headers="keys", tablefmt="github", showindex=False))
    print(f"\nTotal signals: {len(signals)}")

    # ── 5. Recent summary ─────────────────────────────────────────────────────
    last_8w = pd.Timestamp.today() - pd.Timedelta(weeks=8)
    recent = signals[signals["date"] >= last_8w]
    if not recent.empty and not args.recent:
        print(f"\n--- Signals in last 8 weeks ({len(recent)}) ---")
        disp2 = recent.copy()
        disp2["date"] = disp2["date"].dt.strftime("%Y-%m-%d")
        disp2.rename(
            columns={
                "date": "Week",
                "ticker": "Ticker",
                "entry_n": "Entry#",
                "hist": "Hist",
                "prev_hist": "PrevHist",
            },
            inplace=True,
        )
        print(tabulate(disp2, headers="keys", tablefmt="github", showindex=False))

    # ── 6. Save to CSV ────────────────────────────────────────────────────────
    if args.output:
        signals.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
