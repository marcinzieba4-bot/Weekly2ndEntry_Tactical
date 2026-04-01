"""
options_premiums.py
===================
Downloads real ATM call-option premiums from S3 (optionsDataCall/),
builds a per-ticker / per-date premium-pct lookup, and returns the
8-week equivalent premium for any (ticker, date) pair.

Methodology
-----------
  premium_pct  = entry_premium / stock_price * 100   (raw, ~30-day DTE)
  premium_8w   = premium_pct * 1.17                  (vol-curve scale-up)

For tickers not in the options dataset the closest sector/peer ticker is used.
For dates before the dataset starts (Sep-2020) the earliest available row
for that (mapped) ticker is used.

Cache: /tmp/options_premiums_cache.json  (dict: ticker -> {date_str: pct})
"""

import os, io, json, boto3
import pandas as pd
from botocore.config import Config

# ── AWS ────────────────────────────────────────────────────────────────────────
REGION    = 'eu-north-1'
BUCKET    = 's3bucketmz'
PREFIX    = 'optionsDataCall/'
CACHE     = '/tmp/options_premiums_cache.json'
SCALE_8W  = 1.17          # 30-day → 8-week vol-curve adjustment

AWS_KEY    = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET = os.environ['AWS_SECRET_ACCESS_KEY']

_s3 = None
def _get_s3():
    global _s3
    if _s3 is None:
        _s3 = boto3.client(
            's3', region_name=REGION,
            aws_access_key_id=AWS_KEY,
            aws_secret_access_key=AWS_SECRET,
            config=Config(connect_timeout=10, read_timeout=60),
        )
    return _s3


# ── Sector / peer mapping ──────────────────────────────────────────────────────
# For every ticker NOT in the 42-ticker options dataset, map to the best proxy.
# Grouped by GICS sector / sub-industry.

PEER_MAP = {
    # ── Technology ────────────────────────────────────────────────────────────
    'ADBE': 'CRM',   'AKAM': 'IBM',   'AMD':  'NVDA',  'AMAT': 'NVDA',
    'AME':  'IBM',   'ANSS': 'MSFT',  'APH':  'IBM',   'CDNS': 'MSFT',
    'CSCO': 'IBM',   'CDW':  'IBM',   'CTSH': 'ACN',   'DELL': 'IBM',
    'EA':   'MSFT',  'ENPH': 'NVDA',  'EQIX': 'CRM',   'FTNT': 'CRM',
    'GEN':  'IBM',   'GLW':  'IBM',   'HPE':  'IBM',   'HPQ':  'IBM',
    'INTC': 'QCOM',  'IT':   'ACN',   'JNPR': 'IBM',   'KEYS': 'IBM',
    'KLAC': 'NVDA',  'LRCX': 'NVDA',  'MCHP': 'QCOM',  'MPWR': 'NVDA',
    'MSI':  'IBM',   'MU':   'NVDA',  'NTAP': 'IBM',   'NXPI': 'QCOM',
    'PANW': 'CRM',   'PAYX': 'ACN',   'POOL': 'MSFT',  'PTC':  'CRM',
    'QRVO': 'QCOM',  'ROP':  'MSFT',  'SNPS': 'MSFT',  'STX':  'IBM',
    'TDY':  'IBM',   'TEL':  'IBM',   'TXN':  'QCOM',  'VRSN': 'MSFT',
    'WDC':  'IBM',   'ZBRA': 'IBM',
    # SaaS / cloud
    'DDOG': 'CRM',   'MDB':  'CRM',   'SNOW': 'CRM',   'TEAM': 'CRM',
    'WDAY': 'CRM',   'ZS':   'CRM',
    # ── Communication / Media ─────────────────────────────────────────────────
    'CHTR': 'CMCSA', 'FOX':  'DIS',   'FOXA': 'DIS',   'IAC':  'META',
    'LYFT': 'TSLA',  'MTCH': 'META',  'NDAQ': 'MSFT',  'PARA': 'DIS',
    'PINS': 'META',  'SNAP': 'META',  'T':    'CMCSA', 'TMUS': 'CMCSA',
    'TTWO': 'MSFT',  'UBER': 'TSLA',  'VZ':   'CMCSA', 'WBD':  'DIS',
    # ── Consumer Discretionary ────────────────────────────────────────────────
    'ABNB': 'AMZN',  'APTV': 'TSLA',  'BBY':  'WMT',   'BKNG': 'AMZN',
    'BURL': 'WMT',   'CCL':  'DIS',   'CMG':  'WMT',   'CPRT': 'WMT',
    'DAL':  'TSLA',  'DE':   'HD',    'DG':   'WMT',   'DHI':  'HD',
    'DLTR': 'WMT',   'DPZ':  'WMT',   'EXPE': 'AMZN',  'F':    'TSLA',
    'GM':   'TSLA',  'HLT':  'AMZN',  'JBHT': 'WMT',   'KBH':  'HD',
    'KR':   'WMT',   'LEN':  'HD',    'LUV':  'TSLA',  'LVS':  'DIS',
    'LOW':  'HD',    'MAR':  'AMZN',  'MCD':  'WMT',   'MGM':  'DIS',
    'NCLH': 'DIS',   'NVR':  'HD',    'ODFL': 'WMT',   'ORLY': 'WMT',
    'PHM':  'HD',    'RCL':  'DIS',   'RL':   'NKE',   'ROST': 'WMT',
    'SBUX': 'WMT',   'SPG':  'WMT',   'TGT':  'WMT',   'TJX':  'WMT',
    'TOL':  'HD',    'TPR':  'NKE',   'TSN':  'WMT',   'UAL':  'TSLA',
    'URI':  'HD',    'VICI': 'DIS',   'WAB':  'HD',    'XPO':  'WMT',
    'YUM':  'WMT',
    # ── Consumer Staples ──────────────────────────────────────────────────────
    'ADM':  'KO',    'CAG':  'KO',    'CHD':  'PG',    'CL':   'PG',
    'CLX':  'PG',    'EL':   'PG',    'FAST': 'WMT',   'GIS':  'KO',
    'HSY':  'KO',    'KDP':  'KO',    'KMB':  'PG',    'L':    'PG',
    'MKC':  'KO',    'MO':   'KO',    'MNST': 'KO',    'PM':   'KO',
    'SYY':  'WMT',
    # ── Financials ────────────────────────────────────────────────────────────
    'AFL':  'JNJ',   'AIG':  'JPM',   'AJG':  'JPM',   'ALL':  'JPM',
    'AMG':  'BX',    'AON':  'JPM',   'ARES': 'BX',    'AXP':  'MA',
    'BLK':  'BX',    'BR':   'ACN',   'C':    'BAC',   'CB':   'JPM',
    'CFG':  'BAC',   'CINF': 'JPM',   'CME':  'MSFT',  'COF':  'MA',
    'CG':   'BX',    'DFS':  'MA',    'EFX':  'ACN',   'FDS':  'ACN',
    'FI':   'V',     'FIS':  'V',     'GPN':  'V',     'GS':   'JPM',
    'ICE':  'MSFT',  'IVZ':  'BX',    'KKR':  'BX',    'MCO':  'ACN',
    'MET':  'JPM',   'MMC':  'JPM',   'MS':   'JPM',   'MSCI': 'ACN',
    'NDAQ': 'MSFT',  'PGR':  'JPM',   'PNC':  'JPM',   'PRU':  'JPM',
    'PYPL': 'V',     'RF':   'BAC',   'RHI':  'ACN',   'SCHW': 'JPM',
    'SPGI': 'ACN',   'SYF':  'MA',    'TFC':  'BAC',   'TROW': 'BX',
    'TRV':  'JPM',   'USB':  'JPM',   'VRSK': 'ACN',
    # ── Healthcare ────────────────────────────────────────────────────────────
    'A':    'TMO',   'AMGN': 'LLY',   'BAX':  'ABT',   'BDX':  'ABT',
    'BIIB': 'LLY',   'BMY':  'ABBV',  'BSX':  'ABT',   'CI':   'UNH',
    'CNC':  'UNH',   'CRL':  'TMO',   'CTLT': 'ABT',   'CVS':  'UNH',
    'DGX':  'TMO',   'DHR':  'TMO',   'EW':   'ABT',   'GILD': 'ABBV',
    'HCA':  'UNH',   'HOLX': 'ABT',   'HUM':  'UNH',   'IDXX': 'TMO',
    'IQV':  'TMO',   'ISRG': 'ABT',   'LH':   'TMO',   'MCK':  'UNH',
    'MDT':  'ABT',   'MOS':  'XOM',   'MRNA': 'LLY',   'MTD':  'TMO',
    'PFE':  'MRK',   'REGN': 'LLY',   'STE':  'ABT',   'SYK':  'ABT',
    'TDG':  'ABT',   'UHS':  'UNH',   'VRTX': 'LLY',   'WAT':  'TMO',
    'ZBH':  'ABT',   'ZTS':  'ABT',   'ELV':  'UNH',
    # ── Energy ────────────────────────────────────────────────────────────────
    'APA':  'XOM',   'BKR':  'XOM',   'COP':  'CVX',   'DVN':  'XOM',
    'EOG':  'XOM',   'EPD':  'XOM',   'EQT':  'XOM',   'ET':   'XOM',
    'FANG': 'XOM',   'HAL':  'XOM',   'HES':  'CVX',   'KMI':  'XOM',
    'MPC':  'CVX',   'MPLX': 'XOM',   'MRO':  'XOM',   'OKE':  'XOM',
    'OXY':  'CVX',   'PSX':  'CVX',   'PXD':  'CVX',   'RRC':  'XOM',
    'RS':   'XOM',   'SLB':  'XOM',   'VLO':  'CVX',   'WMB':  'XOM',
    # ── Industrials ───────────────────────────────────────────────────────────
    'CARR': 'IBM',   'CAT':  'HD',    'CHRW': 'WMT',   'CMC':  'HD',
    'CTAS': 'ACN',   'CSGP': 'CRM',   'CSX':  'SPY',   'DD':   'SPY',
    'DOW':  'SPY',   'EMR':  'IBM',   'ETN':  'IBM',   'EXPD': 'WMT',
    'FDX':  'WMT',   'FTV':  'IBM',   'GD':   'SPY',   'GE':   'SPY',
    'GNRC': 'IBM',   'GWW':  'HD',    'HII':  'SPY',   'HON':  'IBM',
    'HWM':  'SPY',   'IEX':  'IBM',   'IFF':  'PG',    'IP':   'SPY',
    'ITW':  'IBM',   'JBHT': 'WMT',   'JCI':  'IBM',   'LHX':  'SPY',
    'LIN':  'SPY',   'LNT':  'SPY',   'LYB':  'SPY',   'MMM':  'PG',
    'MSC':  'HD',    'NOC':  'SPY',   'NSC':  'SPY',   'NUE':  'SPY',
    'OTIS': 'IBM',   'PH':   'IBM',   'PKG':  'SPY',   'PNR':  'IBM',
    'PPG':  'PG',    'PCAR': 'HD',    'PWR':  'HD',    'ROK':  'IBM',
    'RTX':  'SPY',   'SHW':  'PG',    'STLD': 'SPY',   'SWK':  'HD',
    'TT':   'IBM',   'TXT':  'SPY',   'UNP':  'SPY',   'UPS':  'WMT',
    'WM':   'WMT',   'WY':   'SPY',   'XEL':  'SPY',   'XYL':  'IBM',
    # ── Utilities / REITs ─────────────────────────────────────────────────────
    'AEP':  'SPY',   'AMT':  'SPY',   'APD':  'SPY',   'AVB':  'SPY',
    'BALL': 'SPY',   'CCI':  'SPY',   'CF':   'SPY',   'D':    'SPY',
    'DLR':  'CRM',   'DTE':  'SPY',   'DUK':  'SPY',   'ECL':  'PG',
    'ESS':  'SPY',   'ETR':  'SPY',   'EXC':  'SPY',   'EXR':  'SPY',
    'FCX':  'SPY',   'FE':   'SPY',   'IRM':  'SPY',   'MAA':  'SPY',
    'NEE':  'SPY',   'NEM':  'SPY',   'O':    'SPY',   'PCG':  'SPY',
    'PEG':  'SPY',   'PLD':  'SPY',   'PSA':  'SPY',   'AMCR': 'SPY',
    'AFMD': 'SPY',   'SEE':  'SPY',   'SO':   'SPY',   'SRE':  'SPY',
    'WELL': 'SPY',   'WEC':  'SPY',   'ALB':  'SPY',   'CE':   'SPY',
    'BRK-B':'JPM',
}

# Options tickers available in S3
OPTIONS_TICKERS = frozenset([
    'AAPL','ABBV','ABT','ACN','AMZN','AVGO','BAC','BX','CMCSA','COST',
    'CRM','CVX','DIS','GOOG','GOOGL','HD','IBM','INTU','JNJ','JPM',
    'KO','LLY','MA','META','MRK','MSFT','NFLX','NKE','NOW','NVDA',
    'ORCL','PEP','PG','QCOM','SPY','TMO','TSLA','UNH','V','WFC',
    'WMT','XOM',
])


def resolve_ticker(ticker: str) -> str:
    """Return the options-dataset ticker to use for `ticker`."""
    if ticker in OPTIONS_TICKERS:
        return ticker
    mapped = PEER_MAP.get(ticker)
    if mapped and mapped in OPTIONS_TICKERS:
        return mapped
    return 'SPY'   # ultimate fallback


# ── Download & build premium table ────────────────────────────────────────────

def _download_all() -> dict:
    """
    Return dict:  options_ticker → {obs_date_str → premium_pct_float}
    premium_pct = entry_premium / stock_price * 100  (raw ~30-day)
    """
    s3 = _get_s3()
    paginator = s3.get_paginator('list_objects_v2')

    # Collect all CSV keys
    csv_keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for obj in page.get('Contents', []):
            k = obj['Key']
            if k.endswith('.csv'):
                csv_keys.append(k)

    print(f'  Downloading {len(csv_keys)} option CSV files from S3...')

    data = {t: {} for t in OPTIONS_TICKERS}
    errors = 0
    for i, key in enumerate(csv_keys):
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            raw = obj['Body'].read().decode()
            df  = pd.read_csv(io.StringIO(raw))
            if df.empty:
                continue
            row = df.iloc[0]
            ticker  = str(row['ticker'])
            obs_d   = str(row['observation_date'])[:10]
            prem    = float(row['entry_premium'])
            spx     = float(row['stock_price'])
            if spx > 0:
                data[ticker][obs_d] = round(prem / spx * 100, 4)
        except Exception:
            errors += 1
        if (i + 1) % 500 == 0:
            print(f'    {i+1}/{len(csv_keys)} ...')

    if errors:
        print(f'  {errors} files skipped (parse error)')
    return data


def load_premiums(force_refresh: bool = False) -> dict:
    """Load premium table, using cache if available."""
    if not force_refresh and os.path.exists(CACHE):
        print(f'  Loading options premiums from cache: {CACHE}')
        with open(CACHE) as f:
            return json.load(f)

    print('  Building options premiums table from S3...')
    data = _download_all()
    # Sort dates for each ticker
    data = {t: dict(sorted(v.items())) for t, v in data.items()}
    with open(CACHE, 'w') as f:
        json.dump(data, f)
    print(f'  Cached to {CACHE}  ({sum(len(v) for v in data.values())} observations)')
    return data


# ── Lookup function ────────────────────────────────────────────────────────────

class PremiumLookup:
    """
    Callable: get_premium(ticker, signal_date) -> 8-week premium %
    Uses nearest observation date (not future-looking — takes the most
    recent observation on or before signal_date; if none, takes earliest).
    """

    def __init__(self, data: dict):
        # Convert to {ticker: sorted list of (date_str, prem_pct)}
        self._series: dict[str, list] = {}
        for t, obs in data.items():
            self._series[t] = sorted(obs.items())   # [(date_str, pct), ...]

    def get_premium(self, ticker: str, signal_date) -> float:
        opt_ticker = resolve_ticker(ticker)
        series     = self._series.get(opt_ticker, [])
        if not series:
            return 5.0 * SCALE_8W   # hard fallback

        sig_str = str(signal_date)[:10]

        # Binary-search for most recent obs <= signal_date
        lo, hi = 0, len(series) - 1
        best   = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if series[mid][0] <= sig_str:
                best = mid
                lo   = mid + 1
            else:
                hi   = mid - 1

        raw_pct = series[best][1]
        return round(raw_pct * SCALE_8W, 4)
