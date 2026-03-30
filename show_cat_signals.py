"""
show_cat_signals.py
--------------------
Run this locally to see all 2nd+ MACD histogram upturn entries for CAT
on weekly bars over the last 10 years.

    pip install yfinance pandas tabulate
    python show_cat_signals.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from tabulate import tabulate

FAST, SLOW, SIGNAL_PERIOD = 12, 26, 9
TICKER = "CAT"
PERIOD = "10y"


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def macd_histogram(close: pd.Series) -> pd.Series:
    macd_line = ema(close, FAST) - ema(close, SLOW)
    return macd_line - ema(macd_line, SIGNAL_PERIOD)


def find_signals(close: pd.Series):
    hist = macd_histogram(close)

    rows = []
    in_sub_zero = False
    upturn_count = 0
    in_upturn = False
    prev_h = np.nan

    for date, h in hist.items():
        if np.isnan(h) or np.isnan(prev_h):
            prev_h = h
            continue

        # enter sub-zero zone
        if h < 0 and not in_sub_zero:
            in_sub_zero = True
            upturn_count = 0
            in_upturn = False
        # leave sub-zero zone — reset
        elif h >= 0 and in_sub_zero:
            in_sub_zero = False
            upturn_count = 0
            in_upturn = False

        if in_sub_zero:
            if h > prev_h:
                if not in_upturn:
                    upturn_count += 1
                    in_upturn = True
                    if upturn_count >= 2:
                        rows.append({
                            "Week (Mon)": date.strftime("%Y-%m-%d"),
                            "Entry #": upturn_count,
                            "Hist": round(h, 4),
                            "Prev Hist": round(prev_h, 4),
                            "Change": round(h - prev_h, 4),
                            "Close": round(float(close.loc[date]), 2),
                        })
            else:
                in_upturn = False

        prev_h = h

    return hist, rows


def main():
    print(f"Downloading {TICKER} weekly data ({PERIOD}) from Yahoo Finance...")
    raw = yf.download(TICKER, period=PERIOD, interval="1wk",
                      progress=False, auto_adjust=True)

    if raw.empty:
        print("Download failed — check your internet connection.")
        return

    close = raw["Close"].squeeze()
    print(f"  {len(close)} weekly bars  ({close.index[0].date()} → {close.index[-1].date()})\n")

    hist, signals = find_signals(close)

    if not signals:
        print("No 2nd+ entry signals found.")
        return

    df = pd.DataFrame(signals)
    print(f"=== {TICKER} Weekly MACD (12,26,9) — 2nd+ Histogram Upturn Entries ===\n")
    print(tabulate(df, headers="keys", tablefmt="github", showindex=True))
    print(f"\nTotal signals: {len(df)}")

    # Summary by entry number
    print("\nBreakdown by entry number:")
    print(df.groupby("Entry #").size().rename("Count").to_string())

    # Current status
    last_hist = hist.dropna().iloc[-1]
    prev_hist = hist.dropna().iloc[-2]
    last_date = hist.dropna().index[-1]
    print(f"\nCurrent histogram ({last_date.strftime('%Y-%m-%d')}): {last_hist:.4f}  "
          f"(prev: {prev_hist:.4f})  {'↑ rising' if last_hist > prev_hist else '↓ falling'}")
    if last_hist < 0:
        print("  → Currently in sub-zero zone")
    else:
        print("  → Histogram is above zero")


if __name__ == "__main__":
    main()
