"""
Weekly MACD Histogram signal detector.

Strategy rules:
  - Timeframe  : Weekly bars
  - MACD params: fast=12, slow=26, signal=9 (standard)
  - Entry cond : histogram < 0  AND  hist[i] > hist[i-1]   (turning up while negative)
  - Sequence   : We only flag the 2nd and subsequent entries within a single
                 sub-zero "swing" (i.e. after at least one prior upturn that
                 failed to push histogram above 0 and then rolled back down).
  - Reset      : A new sub-zero swing begins when histogram crosses above 0
                 (even briefly) and then falls back below again.

Definition of "entry" and "2nd+ entry":
  - entry_n=1 : first bar in a sub-zero region where hist turns up
  - entry_n=2+: every subsequent upturn in the same sub-zero region
                (histogram dipped back below the prior upturn bar before
                 turning up again = a new local attempt)

We track state per ticker:
  prev_hist         : histogram value of bar i-1
  in_sub_zero       : bool, are we in the sub-zero region?
  upturn_count      : how many upturns have occurred in current sub-zero region
  last_upturn_high  : highest histogram value reached during the current upturn
  in_upturn         : bool, are we currently climbing (upturn in progress)?
"""

import yfinance as yf
import pandas as pd
import numpy as np


FAST = 12
SLOW = 26
SIGNAL = 9


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_macd_histogram(close: pd.Series) -> pd.Series:
    """Return MACD histogram series."""
    fast_ema = ema(close, FAST)
    slow_ema = ema(close, SLOW)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, SIGNAL)
    return macd_line - signal_line


def detect_signals(ticker: str, lookback_weeks: int = 260) -> pd.DataFrame:
    """
    Download weekly data and return a DataFrame of 2nd+ MACD entry signals.

    Returns columns: date, ticker, entry_n, hist_value, prev_hist_value
    """
    raw = yf.download(
        ticker,
        period=f"{lookback_weeks}w",
        interval="1wk",
        progress=False,
        auto_adjust=True,
    )
    if raw.empty or len(raw) < SLOW + SIGNAL + 5:
        return pd.DataFrame()

    close = raw["Close"].squeeze()
    hist = compute_macd_histogram(close)

    signals = []

    # State machine
    in_sub_zero = False
    upturn_count = 0       # upturns completed (or in progress) in current sub-zero swing
    in_upturn = False      # currently climbing within sub-zero
    prev_hist = np.nan

    for date, h in hist.items():
        if np.isnan(h) or np.isnan(prev_hist):
            prev_hist = h
            continue

        # --- transition: entered sub-zero zone ---
        if h < 0 and not in_sub_zero:
            in_sub_zero = True
            upturn_count = 0
            in_upturn = False

        # --- transition: left sub-zero zone (reset) ---
        elif h >= 0 and in_sub_zero:
            in_sub_zero = False
            upturn_count = 0
            in_upturn = False

        # --- within sub-zero zone ---
        if in_sub_zero:
            if h > prev_hist:
                # Histogram is rising
                if not in_upturn:
                    # A new upturn just started
                    upturn_count += 1
                    in_upturn = True

                    # Only record if this is the 2nd or later upturn
                    if upturn_count >= 2:
                        signals.append(
                            {
                                "date": date,
                                "ticker": ticker,
                                "entry_n": upturn_count,
                                "hist": round(float(h), 6),
                                "prev_hist": round(float(prev_hist), 6),
                            }
                        )
            else:
                # Histogram falling or flat — upturn ended
                in_upturn = False

        prev_hist = h

    if not signals:
        return pd.DataFrame()

    df = pd.DataFrame(signals)
    df["date"] = pd.to_datetime(df["date"])
    return df


def scan_universe(tickers: list[str], lookback_weeks: int = 260) -> pd.DataFrame:
    """Scan all tickers and return combined signal DataFrame."""
    all_signals = []
    for ticker in tickers:
        try:
            df = detect_signals(ticker, lookback_weeks)
            if not df.empty:
                all_signals.append(df)
        except Exception as e:
            print(f"  Warning: {ticker} failed — {e}")

    if not all_signals:
        return pd.DataFrame()

    combined = pd.concat(all_signals, ignore_index=True)
    combined.sort_values("date", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined
