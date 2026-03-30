"""
Unit tests for the MACD 2nd-entry signal detector using synthetic price data.
Run with: python test_signals.py
"""

import numpy as np
import pandas as pd
import sys

from macd_signals import compute_macd_histogram, ema


def synthetic_close(base: float, changes: list[float]) -> pd.Series:
    """Build a close price series from a list of daily % changes."""
    prices = [base]
    for c in changes:
        prices.append(prices[-1] * (1 + c))
    dates = pd.date_range("2020-01-06", periods=len(prices), freq="W-MON")
    return pd.Series(prices, index=dates)


def apply_state_machine(hist: pd.Series):
    """
    Replicate the state machine from macd_signals.detect_signals
    and return (date, entry_n) pairs for all 2nd+ entries.
    """
    signals = []
    in_sub_zero = False
    upturn_count = 0
    in_upturn = False
    prev_h = np.nan

    for date, h in hist.items():
        if np.isnan(h) or np.isnan(prev_h):
            prev_h = h
            continue

        if h < 0 and not in_sub_zero:
            in_sub_zero = True
            upturn_count = 0
            in_upturn = False
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
                        signals.append((date, upturn_count))
            else:
                in_upturn = False

        prev_h = h

    return signals


# ──────────────────────────────────────────────────────────────────────────────
def make_hist(values: list[float]) -> pd.Series:
    """Build a histogram Series from explicit values for direct state-machine testing."""
    dates = pd.date_range("2020-01-06", periods=len(values) + 1, freq="W-MON")
    # Prepend NaN so state machine has a valid prev_hist on 2nd bar
    return pd.Series([np.nan] + values, index=dates)


def test_no_signal_on_first_upturn():
    """1st upturn should NOT produce a signal (state machine direct test)."""
    # Sub-zero: one upturn only (rises then stays down or flat — no second upturn)
    vals = [-5, -4, -3, -4, -5]   # one upturn (-5→-4→-3), then falls again — no 2nd upturn
    hist = make_hist(vals)
    sigs = apply_state_machine(hist)
    assert len(sigs) == 0, f"Expected 0 signals, got {len(sigs)}: {sigs}"
    print("PASS  test_no_signal_on_first_upturn")


def test_second_upturn_triggers_signal():
    """2nd upturn after staying sub-zero must produce exactly one signal."""
    # down, 1st upturn, back down, 2nd upturn
    vals = [-5, -4, -5, -4]
    hist = make_hist(vals)
    sigs = apply_state_machine(hist)
    assert len(sigs) == 1, f"Expected 1 signal, got {len(sigs)}: {sigs}"
    assert sigs[0][1] == 2, f"Expected entry_n=2, got {sigs[0][1]}"
    print(f"PASS  test_second_upturn_triggers_signal  (signals={len(sigs)})")


def test_third_upturn_produces_entry3():
    """3rd upturn must be recorded as entry_n=3."""
    vals = [-5, -4, -5, -4, -5, -4]
    hist = make_hist(vals)
    sigs = apply_state_machine(hist)
    assert len(sigs) == 2, f"Expected 2 signals, got {len(sigs)}: {sigs}"
    assert sigs[0][1] == 2
    assert sigs[1][1] == 3
    print(f"PASS  test_third_upturn_produces_entry3")


def test_reset_after_crossing_above_zero():
    """After histogram crosses above 0, counter resets; 1st upturn in new swing = no signal."""
    # Sub-zero: 1st upturn, 2nd upturn (signal), cross above zero, new sub-zero, 1st upturn only
    vals = [-5, -4, -5, -4,   # 2 upturns in first swing
             1, 2,             # cross above 0 — reset
            -3, -2]            # new sub-zero swing, 1st upturn only — no signal
    hist = make_hist(vals)
    sigs = apply_state_machine(hist)
    # Exactly 1 signal from the first swing (entry_n=2)
    assert len(sigs) == 1, f"Expected 1 signal, got {len(sigs)}: {sigs}"
    assert sigs[0][1] == 2
    print(f"PASS  test_reset_after_crossing_above_zero  (signals={len(sigs)})")


def test_macd_ema_convergence():
    """EMA of a constant series should equal the constant."""
    s = pd.Series([5.0] * 100)
    result = ema(s, 12)
    assert abs(result.iloc[-1] - 5.0) < 1e-9, f"EMA did not converge: {result.iloc[-1]}"
    print("PASS  test_macd_ema_convergence")


def test_macd_histogram_flat_series_is_zero():
    """Flat price -> MACD line = 0 -> signal line = 0 -> histogram = 0."""
    close = pd.Series([100.0] * 100)
    hist = compute_macd_histogram(close)
    assert abs(hist.iloc[-1]) < 1e-9, f"Histogram not zero for flat series: {hist.iloc[-1]}"
    print("PASS  test_macd_histogram_flat_series_is_zero")


if __name__ == "__main__":
    tests = [
        test_macd_ema_convergence,
        test_macd_histogram_flat_series_is_zero,
        test_no_signal_on_first_upturn,
        test_second_upturn_triggers_signal,
        test_third_upturn_produces_entry3,
        test_reset_after_crossing_above_zero,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {e}")
            failed += 1

    print(f"\n{'All tests passed.' if failed == 0 else f'{failed} test(s) FAILED.'}")
    sys.exit(0 if failed == 0 else 1)
