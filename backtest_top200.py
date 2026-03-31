"""
backtest_top200.py
==================
For each calendar year, identify the 200 largest stocks (by approx market cap =
year-start close × current shares outstanding), detect weekly MACD (12,26,9)
2nd+ histogram-upturn entries, and simulate trades with:

  Entry      : close of signal candle
  Stop-loss  : low of signal candle
  Exit       : close 8 weeks later
  Re-entry   : if stopped out and price later closes above signal candle high
               → re-enter at that close, new stop = low of re-entry candle,
                 new 8-week clock; skipped if a new MACD signal fires first

Filters applied:
  SPY filter : SPY weekly MACD histogram must be > 0 on signal date

Data fetched via AWS Lambda (yfinance inside Lambda has unrestricted internet).
Results saved to trades_top200.csv.

Usage:
    python backtest_top200.py
"""

import os, json, time, math, concurrent.futures
import boto3
import numpy as np
import pandas as pd

# ── AWS / Lambda config ────────────────────────────────────────────────────────
REGION        = 'eu-north-1'
FUNC_NAME     = 'yfinance-data-fetcher'
BATCH_SIZE    = 15          # tickers per Lambda invocation
MAX_WORKERS   = 20          # parallel Lambda invocations
TOP_N         = 200         # universe size per year
UNIVERSE_FILE = '/tmp/sp500_universe.json'
DATA_CACHE    = '/tmp/ohlcv_cache.json'
SPY_CACHE     = '/tmp/spy_cache.json'
OUTPUT_CSV    = 'trades_top200.csv'

from botocore.config import Config as BotoConfig
_boto_cfg = BotoConfig(read_timeout=300, connect_timeout=10, retries={'max_attempts': 1})

lam = boto3.client(
    'lambda', region_name=REGION,
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    config=_boto_cfg,
)


# ── MACD helpers ───────────────────────────────────────────────────────────────

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def macd_histogram(close: pd.Series) -> pd.Series:
    macd = ema(close, 12) - ema(close, 26)
    return macd - ema(macd, 9)


def detect_signals(close: pd.Series) -> list:
    """Return list of (date, entry_n) for all 2nd+ histogram upturns."""
    hist = macd_histogram(close)
    signals = []
    in_sub  = False
    n_up    = 0
    in_up   = False
    prev    = np.nan

    for date, h in hist.items():
        if np.isnan(h) or np.isnan(prev):
            prev = h; continue
        if h < 0 and not in_sub:
            in_sub = True;  n_up = 0;  in_up = False
        elif h >= 0 and in_sub:
            in_sub = False; n_up = 0;  in_up = False
        if in_sub:
            if h > prev:
                if not in_up:
                    n_up += 1; in_up = True
                    if n_up >= 2:
                        signals.append((date, n_up))
            else:
                in_up = False
        prev = h
    return signals


# ── Trade simulation ───────────────────────────────────────────────────────────

def simulate(signal_date, entry_px, stop_px, candle_high,
             all_dates, close_s, low_s, high_s, signal_date_set):
    """Simulate one trade + optional re-entry. Returns dict."""

    future_idx = all_dates.index(signal_date) + 1
    future     = all_dates[future_idx: future_idx + 8]

    t = dict(
        entry_date=signal_date, entry_price=round(entry_px, 4),
        stop=round(stop_px, 4), candle_high=round(candle_high, 4),
        exit_date=None, exit_price=None, exit_reason=None, pnl_pct=None,
        re_entry_date=None, re_entry_price=None, re_stop=None,
        re_exit_date=None, re_exit_price=None, re_exit_reason=None, re_pnl_pct=None,
    )

    stopped_idx = None
    for i, d in enumerate(future):
        if low_s[d] <= stop_px:                         # stopped out
            t.update(exit_date=d, exit_price=round(stop_px, 4),
                     exit_reason='STOP',
                     pnl_pct=round((stop_px / entry_px - 1) * 100, 2))
            stopped_idx = all_dates.index(d)
            break
        if i == len(future) - 1:                        # 8-week time exit
            cp = close_s[d]
            t.update(exit_date=d, exit_price=round(cp, 4),
                     exit_reason='8W',
                     pnl_pct=round((cp / entry_px - 1) * 100, 2))

    # ── re-entry ───────────────────────────────────────────────────────────────
    if stopped_idx is not None:
        for d in all_dates[stopped_idx + 1:]:
            # New MACD signal fires → skip re-entry
            if d in signal_date_set and d != signal_date:
                break
            if close_s[d] > candle_high:
                re_px   = close_s[d]
                re_stop = low_s[d]
                re_fut  = all_dates[all_dates.index(d) + 1: all_dates.index(d) + 9]
                re_exit_d = re_exit_p = re_reason = re_pnl = None
                for j, d2 in enumerate(re_fut):
                    if low_s[d2] <= re_stop:
                        re_exit_d, re_exit_p = d2, round(re_stop, 4)
                        re_reason = 'STOP'
                        re_pnl    = round((re_stop / re_px - 1) * 100, 2)
                        break
                    if j == len(re_fut) - 1:
                        re_exit_d, re_exit_p = d2, round(close_s[d2], 4)
                        re_reason = '8W'
                        re_pnl    = round((close_s[d2] / re_px - 1) * 100, 2)
                t.update(
                    re_entry_date=d,  re_entry_price=round(re_px, 4),
                    re_stop=round(re_stop, 4),
                    re_exit_date=re_exit_d, re_exit_price=re_exit_p,
                    re_exit_reason=re_reason, re_pnl_pct=re_pnl,
                )
                break
    return t


# ── Lambda data fetching ───────────────────────────────────────────────────────

def invoke_batch(tickers):
    resp = lam.invoke(
        FunctionName=FUNC_NAME,
        InvocationType='RequestResponse',
        Payload=json.dumps({'op': 'ohlcv_batch', 'tickers': tickers, 'period': '11y'}),
    )
    return json.loads(resp['Payload'].read())


def fetch_all_ohlcv(tickers):
    """Fetch weekly OHLCV for all tickers via parallel Lambda calls. Uses cache."""
    if os.path.exists(DATA_CACHE):
        print(f'  Loading OHLCV from cache: {DATA_CACHE}')
        with open(DATA_CACHE) as f:
            return json.load(f)

    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    print(f'  Fetching {len(tickers)} tickers in {len(batches)} batches '
          f'({BATCH_SIZE}/batch, {MAX_WORKERS} parallel) ...')

    all_data = {}
    done = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(invoke_batch, b): b for b in batches}
        for fut in concurrent.futures.as_completed(futs):
            batch = futs[fut]
            try:
                result = fut.result()
                if 'error' in result:
                    print(f'  Batch error: {result["error"][:80]}')
                else:
                    for t, rows in result.items():
                        if isinstance(rows, list) and rows:
                            all_data[t] = rows
                        # None or error → skip
            except Exception as e:
                print(f'  Batch exception: {e}')
            done += 1
            if done % 5 == 0 or done == len(batches):
                print(f'  {done}/{len(batches)} batches done, {len(all_data)} tickers OK')

    with open(DATA_CACHE, 'w') as f:
        json.dump(all_data, f)
    print(f'  Cached to {DATA_CACHE}')
    return all_data


# ── Per-year universe ──────────────────────────────────────────────────────────

def build_yearly_universes(all_data, shares_map):
    """
    For each year Y, rank tickers by (Jan-Y close) × shares_outstanding.
    Return dict: year -> set of top-N tickers.
    """
    years = list(range(2015, 2027))
    universe = {}

    for year in years:
        jan_start = f'{year}-01'
        caps = {}
        for ticker, rows in all_data.items():
            shares = shares_map.get(ticker)
            if not shares:
                continue
            # Find first available close on or after Jan 1 of year
            for row in rows:
                if row['date'].startswith(str(year)):
                    caps[ticker] = row['close'] * shares
                    break
        # Sort descending, take top N
        top = sorted(caps, key=lambda t: caps[t], reverse=True)[:TOP_N]
        universe[year] = set(top)
        print(f'  {year}: {len(top)} stocks in universe')

    return universe


# ── SPY MACD filter ────────────────────────────────────────────────────────────

def fetch_spy_histogram():
    """Fetch SPY weekly OHLCV via Lambda, return MACD histogram Series."""
    if os.path.exists(SPY_CACHE):
        print(f'  Loading SPY from cache')
        with open(SPY_CACHE) as f:
            rows = json.load(f)
    else:
        print('  Fetching SPY data via Lambda...')
        resp = lam.invoke(
            FunctionName=FUNC_NAME, InvocationType='RequestResponse',
            Payload=json.dumps({'op': 'ohlcv_batch', 'tickers': ['SPY'], 'period': '11y'}),
        )
        rows = json.loads(resp['Payload'].read())['SPY']
        with open(SPY_CACHE, 'w') as f:
            json.dump(rows, f)

    close = pd.Series(
        {pd.Timestamp(r['date']): r['close'] for r in rows}
    ).sort_index()
    hist = macd_histogram(close)
    print(f'  SPY MACD histogram: {len(hist)} bars, '
          f'last={hist.index[-1].date()} h={hist.iloc[-1]:.3f}')
    return hist


# ── Main backtest ──────────────────────────────────────────────────────────────

def run_backtest(all_data, yearly_universe, spy_hist):
    """
    Collect all candidate signals, sort chronologically, apply:
      1. SPY MACD > 0 filter
      2. MAX_CONCURRENT (20) open-trade cap
    then simulate accepted trades.
    """
    tickers = list(all_data.keys())
    print(f'Collecting candidate signals from {len(tickers)} tickers...')

    # Build per-ticker data frames and detect signals
    ticker_data  = {}   # ticker -> (dfk, close_s, high_s, low_s, all_dates, signal_date_set)
    all_candidates = [] # (sig_date, ticker, entry_n)

    for ticker in tickers:
        rows = all_data[ticker]
        if not rows or len(rows) < 40:
            continue
        dfk = pd.DataFrame(rows)
        dfk['date'] = pd.to_datetime(dfk['date'])
        dfk = dfk.set_index('date').sort_index()

        close_s   = dfk['close']
        high_s    = dfk['high']
        low_s     = dfk['low']
        all_dates = dfk.index.tolist()
        signals   = detect_signals(close_s)
        sig_set   = {s[0] for s in signals}

        ticker_data[ticker] = (dfk, close_s, high_s, low_s, all_dates, sig_set)

        for (sig_date, entry_n) in signals:
            year = sig_date.year
            if ticker not in yearly_universe.get(year, set()):
                continue
            if sig_date not in dfk.index:
                continue
            all_candidates.append((sig_date, ticker, entry_n))

    # Sort chronologically
    all_candidates.sort(key=lambda x: x[0])
    print(f'  {len(all_candidates)} candidates before filters')

    # Apply filters chronologically
    records     = []
    spy_skipped = 0

    for sig_date, ticker, entry_n in all_candidates:
        # ── Filter: SPY MACD must be > 0 ──────────────────────────────────────
        spy_dates = spy_hist.index[spy_hist.index <= sig_date]
        if len(spy_dates) == 0:
            spy_skipped += 1; continue
        if spy_hist[spy_dates[-1]] <= 0:
            spy_skipped += 1; continue

        # ── Simulate ───────────────────────────────────────────────────────────
        dfk, close_s, high_s, low_s, all_dates, sig_set = ticker_data[ticker]

        trade = simulate(
            sig_date,
            float(close_s[sig_date]),
            float(low_s[sig_date]),
            float(high_s[sig_date]),
            all_dates, close_s, low_s, high_s, sig_set,
        )
        trade['ticker']  = ticker
        trade['entry_n'] = entry_n
        trade['year']    = sig_date.year
        records.append(trade)

    print(f'  SPY-filtered out : {spy_skipped}')
    print(f'  Trades accepted  : {len(records)}')
    return records


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    # 1. Load universe
    print('=== Step 1: Load S&P500 universe ===')
    if not os.path.exists(UNIVERSE_FILE):
        # Get tickers
        r = json.loads(lam.invoke(FunctionName=FUNC_NAME, InvocationType='RequestResponse',
            Payload=json.dumps({'op': 'tickers'}))['Payload'].read())
        all_tickers = r['tickers']
        print(f'  {len(all_tickers)} tickers. Fetching shares in batches...')

        # Get shares in parallel batches of 25
        share_batches = [all_tickers[i:i+25] for i in range(0, len(all_tickers), 25)]
        shares_map = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {pool.submit(lambda b: json.loads(lam.invoke(
                FunctionName=FUNC_NAME, InvocationType='RequestResponse',
                Payload=json.dumps({'op': 'shares_batch', 'tickers': b}))['Payload'].read()), b): b
                for b in share_batches}
            for fut in concurrent.futures.as_completed(futs):
                try:
                    res = fut.result()
                    shares_map.update({t: s for t, s in res.items() if s})
                except Exception as e:
                    print(f'  shares batch error: {e}')
        print(f'  {len(shares_map)} tickers with shares data')
        with open(UNIVERSE_FILE, 'w') as f:
            json.dump({'tickers': all_tickers, 'shares': shares_map}, f)
    else:
        with open(UNIVERSE_FILE) as f:
            universe_data = json.load(f)
        all_tickers = universe_data['tickers']
        shares_map  = {t: s for t, s in universe_data['shares'].items() if s}
        print(f'  {len(all_tickers)} tickers, {len(shares_map)} with shares data')

    # 2. Fetch OHLCV
    print('\n=== Step 2: Fetch weekly OHLCV (10+ years) ===')
    all_data = fetch_all_ohlcv(all_tickers)
    print(f'  {len(all_data)} tickers with data')

    # 3. Build per-year universes
    print('\n=== Step 3: Build per-year top-200 universes ===')
    yearly_universe = build_yearly_universes(all_data, shares_map)

    # 4. Fetch SPY MACD
    print('\n=== Step 4: SPY regime filter ===')
    spy_hist = fetch_spy_histogram()

    # 5. Run backtest
    print('\n=== Step 5: Backtest ===')
    records = run_backtest(all_data, yearly_universe, spy_hist)
    print(f'  Total trades: {len(records)}')

    # 6. Save CSV
    print(f'\n=== Step 6: Save to {OUTPUT_CSV} ===')
    cols = [
        'ticker', 'year', 'entry_n',
        'entry_date', 'entry_price', 'stop', 'candle_high',
        'exit_date', 'exit_price', 'exit_reason', 'pnl_pct',
        're_entry_date', 're_entry_price', 're_stop',
        're_exit_date', 're_exit_price', 're_exit_reason', 're_pnl_pct',
    ]
    df_out = pd.DataFrame(records)[cols]
    df_out = df_out.sort_values(['entry_date', 'ticker']).reset_index(drop=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f'  Saved {len(df_out)} rows to {OUTPUT_CSV}')

    # 7. Quick summary
    closed = df_out[df_out['pnl_pct'].notna()]
    wins   = closed[closed['pnl_pct'] > 0]
    stops  = closed[closed['exit_reason'] == 'STOP']
    tw     = closed[closed['exit_reason'] == '8W']

    re_closed = df_out[df_out['re_pnl_pct'].notna()]
    re_wins   = re_closed[re_closed['re_pnl_pct'] > 0]

    print(f"""
══ Summary ══════════════════════════════════════════
  Trades          : {len(closed)}
  Win rate        : {100*len(wins)/len(closed):.1f}%
  Stopped out     : {len(stops)}  ({100*len(stops)/len(closed):.1f}%)
  8-week exits    : {len(tw)}   ({100*len(tw)/len(closed):.1f}%)
  Avg P&L         : {closed['pnl_pct'].mean():+.2f}%
  Avg winner      : {wins['pnl_pct'].mean():+.2f}%
  Avg loser       : {closed[closed['pnl_pct']<=0]['pnl_pct'].mean():+.2f}%
  Best trade      : {closed['pnl_pct'].max():+.2f}%
  Worst trade     : {closed['pnl_pct'].min():+.2f}%

  Re-entries taken: {len(re_closed)}
  Re-entry WR     : {100*len(re_wins)/max(len(re_closed),1):.1f}%
  Avg re-entry P&L: {re_closed['re_pnl_pct'].mean():+.2f}%
══════════════════════════════════════════════════════
""")


if __name__ == '__main__':
    main()
