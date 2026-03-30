"""
Fetch the 50 largest S&P 500 stocks by market cap.
Source: Wikipedia S&P 500 constituents list + yfinance market cap.
"""

import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd


SPX_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_spx_tickers() -> list[str]:
    """Scrape all S&P 500 tickers from Wikipedia."""
    resp = requests.get(SPX_WIKI_URL, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    tickers = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if cells:
            ticker = cells[0].text.strip().replace(".", "-")
            tickers.append(ticker)
    return tickers


def get_top50_by_market_cap(tickers: list[str]) -> list[str]:
    """Return the 50 tickers with the highest market cap."""
    caps = {}
    # Batch download info via yfinance
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).fast_info
            mc = getattr(info, "market_cap", None)
            if mc:
                caps[ticker] = mc
        except Exception:
            pass

    sorted_tickers = sorted(caps, key=lambda t: caps[t], reverse=True)
    return sorted_tickers[:50]


def load_top50(cache_file: str = "top50.csv") -> list[str]:
    """Load top-50 from cache or fetch fresh."""
    import os
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        return df["ticker"].tolist()

    print("Fetching S&P 500 tickers from Wikipedia...")
    all_tickers = get_spx_tickers()
    print(f"  Found {len(all_tickers)} tickers. Fetching market caps...")
    top50 = get_top50_by_market_cap(all_tickers)
    pd.DataFrame({"ticker": top50}).to_csv(cache_file, index=False)
    print(f"  Top-50 cached to {cache_file}")
    return top50


if __name__ == "__main__":
    top50 = load_top50()
    print("Top 50 SPX stocks by market cap:")
    for i, t in enumerate(top50, 1):
        print(f"  {i:2d}. {t}")
