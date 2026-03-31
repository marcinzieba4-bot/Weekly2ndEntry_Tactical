"""
portfolio_analysis.py
=====================
Builds a weekly equal-weight portfolio from backtest trades.
  - 200 slots, each weighted 1/200
  - Slot is "in" during an active trade (incl. re-entry gap = cash)
  - Unoccupied slots earn 0 (cash)
  - Uses OHLCV weekly closes for intra-trade weekly P&L

Outputs:
  - Console: risk stats + annual returns table
  - portfolio_analysis.png: equity curve / annual returns / drawdown chart
"""

import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

TRADES_CSV   = 'trades_top200.csv'
OHLCV_CACHE  = '/tmp/ohlcv_cache.json'
OUTPUT_PNG   = 'portfolio_analysis.png'
N_SLOTS      = 200
WEIGHT       = 1 / N_SLOTS
RISK_FREE    = 0.04   # annual, for Sharpe/Sortino

# ── 1. Load data ──────────────────────────────────────────────────────────────
print('Loading data...')
trades = pd.read_csv(TRADES_CSV, parse_dates=[
    'entry_date','exit_date','re_entry_date','re_exit_date'])

with open(OHLCV_CACHE) as f:
    raw_ohlcv = json.load(f)

# Build per-ticker weekly close series
price = {}
for ticker, rows in raw_ohlcv.items():
    if not rows or not isinstance(rows, list):
        continue
    s = pd.Series(
        {pd.Timestamp(r['date']): r['close'] for r in rows},
        name=ticker
    ).sort_index()
    price[ticker] = s

# Master weekly date index (Mon close dates)
all_dates = sorted(set(d for s in price.values() for d in s.index))
all_dates = pd.DatetimeIndex(all_dates)
print(f'  Date range: {all_dates[0].date()} → {all_dates[-1].date()}  ({len(all_dates)} weeks)')

# ── 2. Build per-stock weekly return series during active trades ──────────────
print('Building per-stock active-period returns...')

# For each stock, collect (week_date -> weekly_return) while in a trade
stock_weekly_ret = {d: {} for d in all_dates}  # date -> {ticker: ret}

def apply_trade(ticker, entry_date, entry_price, exit_date, exit_price, exit_reason):
    """Fill in weekly returns for one trade leg."""
    if ticker not in price:
        return
    s = price[ticker]
    # Find all weekly dates strictly between entry and exit (inclusive of exit)
    mask = (s.index > entry_date) & (s.index <= exit_date)
    trade_dates = s.index[mask]
    if len(trade_dates) == 0:
        return

    prev_price = entry_price
    for i, d in enumerate(trade_dates):
        is_last = (i == len(trade_dates) - 1)
        if is_last:
            curr = exit_price  # stop or 8W close
        else:
            curr = s.get(d, np.nan)
            if np.isnan(curr):
                curr = prev_price  # flat if missing

        ret = (curr / prev_price - 1) if prev_price > 0 else 0.0
        if d in stock_weekly_ret:
            stock_weekly_ret[d][ticker] = stock_weekly_ret[d].get(ticker, 0) + ret
        prev_price = curr

for _, row in trades.iterrows():
    ticker = row['ticker']
    if pd.isna(row['exit_date']):
        continue
    apply_trade(ticker, row['entry_date'], row['entry_price'],
                row['exit_date'], row['exit_price'], row['exit_reason'])
    # Re-entry leg
    if pd.notna(row['re_entry_date']) and pd.notna(row['re_exit_date']):
        apply_trade(ticker, row['re_entry_date'], row['re_entry_price'],
                    row['re_exit_date'], row['re_exit_price'], row['re_exit_reason'])

# ── 3. Portfolio weekly returns ───────────────────────────────────────────────
print('Aggregating portfolio returns...')

port_rets   = []
deployed_rets = []   # return on capital actually deployed that week
n_active_list = []

for d in all_dates:
    active = stock_weekly_ret.get(d, {})
    n      = len(active)
    # Total portfolio (cash slots = 0, each active = WEIGHT)
    pr = sum(WEIGHT * r for r in active.values())
    # Deployed-only: average return of active positions (0 if none active)
    dr = sum(active.values()) / n if n > 0 else 0.0
    port_rets.append(pr)
    deployed_rets.append(dr)
    n_active_list.append(n)

port_rets     = pd.Series(port_rets,     index=all_dates, name='portfolio')
deployed_rets = pd.Series(deployed_rets, index=all_dates, name='deployed')
n_active      = pd.Series(n_active_list, index=all_dates, name='n_active')

port_rets     = port_rets    [port_rets.index     >= '2015-01-01']
deployed_rets = deployed_rets[deployed_rets.index >= '2015-01-01']
n_active      = n_active     [n_active.index      >= '2015-01-01']

# ── 4. Equity curves & drawdown ───────────────────────────────────────────────
equity    = (1 + port_rets).cumprod()
eq_dep    = (1 + deployed_rets).cumprod()   # deployed-capital equity
roll_max  = equity.cummax()
drawdown  = (equity / roll_max - 1)
dd_dep    = (eq_dep / eq_dep.cummax() - 1)
util_pct  = (n_active / N_SLOTS * 100).rolling(4).mean()   # 4-week smoothed utilisation

# ── 5. Risk statistics ────────────────────────────────────────────────────────
weeks_per_year = 52
rf_weekly      = (1 + RISK_FREE) ** (1 / weeks_per_year) - 1

total_weeks    = len(port_rets)
years          = total_weeks / weeks_per_year

# ── full portfolio stats ──────────────────────────────────────────────────────
total_return   = equity.iloc[-1] - 1
cagr           = (equity.iloc[-1]) ** (1 / years) - 1
vol_ann        = port_rets.std() * np.sqrt(weeks_per_year)
sharpe         = (cagr - RISK_FREE) / vol_ann
downside       = port_rets[port_rets < rf_weekly] - rf_weekly
sortino_denom  = downside.std() * np.sqrt(weeks_per_year)
sortino        = (cagr - RISK_FREE) / sortino_denom if sortino_denom > 0 else np.nan
max_dd         = drawdown.min()
calmar         = cagr / abs(max_dd) if max_dd != 0 else np.nan

# ── deployed-capital stats ────────────────────────────────────────────────────
dep_active     = deployed_rets[n_active > 0]
dep_tr         = eq_dep.iloc[-1] - 1
dep_cagr       = (eq_dep.iloc[-1]) ** (1 / years) - 1
dep_vol        = dep_active.std() * np.sqrt(weeks_per_year) if len(dep_active) > 1 else np.nan
dep_sharpe     = (dep_cagr - RISK_FREE) / dep_vol if dep_vol else np.nan
dep_max_dd     = dd_dep.min()
avg_util       = n_active.mean() / N_SLOTS

# Max drawdown duration
in_dd = drawdown < 0
dd_dur = 0; cur_dur = 0; max_dd_dur = 0
for v in in_dd:
    if v: cur_dur += 1; max_dd_dur = max(max_dd_dur, cur_dur)
    else: cur_dur = 0

win_trades  = trades[trades['pnl_pct'] > 0]
lose_trades = trades[trades['pnl_pct'] <= 0]
win_rate    = len(win_trades) / len(trades[trades['pnl_pct'].notna()])
avg_win     = win_trades['pnl_pct'].mean()
avg_loss    = lose_trades['pnl_pct'].mean()

# ── 6. Annual returns ─────────────────────────────────────────────────────────
annual_ret = port_rets.resample('YE').apply(lambda x: (1+x).prod() - 1)
annual_ret.index = annual_ret.index.year

# ── 7. Print stats ────────────────────────────────────────────────────────────
print(f"""
══════════════════════════════════════════════════════════════
  PORTFOLIO RISK STATISTICS  (equal weight 1/200, SPY+Cap filtered)
══════════════════════════════════════════════════════════════
  Period              : {port_rets.index[0].date()} → {port_rets.index[-1].date()}
  Avg capital deployed: {avg_util:.1%}  (avg {n_active.mean():.1f} / {N_SLOTS} slots)

  ── Full portfolio (cash for idle slots) ──────────────────
  Total return        : {total_return:+.1%}
  CAGR                : {cagr:+.1%}
  Ann. volatility     : {vol_ann:.1%}
  Sharpe ratio        : {sharpe:.2f}  (rf={RISK_FREE:.0%})
  Sortino ratio       : {sortino:.2f}
  Max drawdown        : {max_dd:.1%}
  Max DD duration     : {max_dd_dur} weeks
  Calmar ratio        : {calmar:.2f}

  ── Deployed capital only (active trades) ─────────────────
  Total return        : {dep_tr:+.1%}
  CAGR                : {dep_cagr:+.1%}
  Ann. volatility     : {dep_vol:.1%}
  Sharpe ratio        : {dep_sharpe:.2f}
  Max drawdown        : {dep_max_dd:.1%}

  ── Trade-level stats ──────────────────────────────────────
  Trades              : {len(trades[trades['pnl_pct'].notna()])}
  Win rate            : {win_rate:.1%}
  Avg winner          : +{avg_win:.2f}%
  Avg loser           : {avg_loss:.2f}%
  Profit factor       : {abs(avg_win * len(win_trades)) / abs(avg_loss * len(lose_trades)):.2f}
══════════════════════════════════════════════════════════════

Annual returns:
""")
for yr, ret in annual_ret.items():
    bar = '█' * int(abs(ret) * 200)
    sign = '+' if ret >= 0 else '-'
    print(f"  {yr}  {sign}{abs(ret):5.1%}  {bar}")

# ── 8. Chart ──────────────────────────────────────────────────────────────────
print(f'\nGenerating chart → {OUTPUT_PNG}')
fig = plt.figure(figsize=(14, 13), facecolor='#0d1117')
fig.suptitle('SPX Top-200  |  Weekly MACD 2nd-Entry  |  SPY Filter + 20-Trade Cap  |  1/200 Weight',
             color='white', fontsize=13, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1.8, 1.8, 1.2], hspace=0.08,
                       left=0.07, right=0.97, top=0.94, bottom=0.05)

ax1 = fig.add_subplot(gs[0])   # equity curves
ax2 = fig.add_subplot(gs[1])   # annual returns
ax3 = fig.add_subplot(gs[2])   # drawdown
ax4 = fig.add_subplot(gs[3])   # capital utilisation

DARK  = '#0d1117'
GREEN = '#26a641'
RED   = '#f85149'
GOLD  = '#d29922'
BLUE  = '#58a6ff'
GREY  = '#8b949e'

for ax in (ax1, ax2, ax3, ax4):
    ax.set_facecolor(DARK)
    ax.tick_params(colors=GREY, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

# ── Equity curves ─────────────────────────────────────────────────────────────
ax1.plot(equity.index,  equity.values,  color=BLUE,  linewidth=1.4, label=f'Full portfolio  CAGR {cagr:+.1%}')
ax1.plot(eq_dep.index,  eq_dep.values,  color=GOLD,  linewidth=1.2, linestyle='--',
         label=f'Deployed capital CAGR {dep_cagr:+.1%}')
ax1.fill_between(equity.index, 1, equity.values,
                 where=(equity.values >= 1), alpha=0.10, color=GREEN)
ax1.fill_between(equity.index, 1, equity.values,
                 where=(equity.values <  1), alpha=0.10, color=RED)
ax1.axhline(1, color=GREY, linewidth=0.5, linestyle='--')

ax1.legend(loc='upper left', fontsize=8.5, facecolor='#161b22',
           edgecolor='#30363d', labelcolor='white')
ax1.set_ylabel('Value (start=1)', color=GREY, fontsize=9)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
ax1.grid(axis='y', color='#21262d', linewidth=0.5)
ax1.set_xticklabels([])

stats_txt = (f'Full port: Sharpe {sharpe:.2f}  Sortino {sortino:.2f}  Max DD {max_dd:.1%}  Calmar {calmar:.2f}  Vol {vol_ann:.1%} | '
             f'Deployed: Sharpe {dep_sharpe:.2f}  Max DD {dep_max_dd:.1%}  Avg util {avg_util:.1%}')
ax1.text(0.01, 0.03, stats_txt, transform=ax1.transAxes,
         color=GREY, fontsize=7.8,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#161b22', edgecolor='#30363d'))

# ── Annual returns ─────────────────────────────────────────────────────────────
colors = [GREEN if r >= 0 else RED for r in annual_ret.values]
bars   = ax2.bar(annual_ret.index, annual_ret.values * 100,
                 color=colors, width=0.6, alpha=0.85, zorder=3)
ax2.axhline(0, color=GREY, linewidth=0.6)
ax2.set_ylabel('Annual Return %', color=GREY, fontsize=9)
ax2.grid(axis='y', color='#21262d', linewidth=0.5, zorder=0)
ax2.set_xticklabels([])
ax2.set_xlim(annual_ret.index[0] - 0.6, annual_ret.index[-1] + 0.6)

for bar, val in zip(bars, annual_ret.values):
    ypos = val * 100 + (0.3 if val >= 0 else -0.8)
    ax2.text(bar.get_x() + bar.get_width()/2, ypos,
             f'{val:+.1%}', ha='center', va='bottom' if val >= 0 else 'top',
             color='white', fontsize=7.5, fontweight='bold')

# ── Drawdown ──────────────────────────────────────────────────────────────────
ax3.fill_between(drawdown.index, drawdown.values * 100, 0,
                 color=RED, alpha=0.5, linewidth=0, label='Full port')
ax3.plot(dd_dep.index, dd_dep.values * 100, color=GOLD, linewidth=0.8,
         linestyle='--', label='Deployed')
ax3.axhline(0, color=GREY, linewidth=0.5)
ax3.annotate(f'{max_dd:.1%}', xy=(drawdown.idxmin(), max_dd * 100),
             xytext=(15, -8), textcoords='offset points',
             color=RED, fontsize=8,
             arrowprops=dict(arrowstyle='->', color=RED, lw=0.7))
ax3.legend(loc='lower right', fontsize=7.5, facecolor='#161b22',
           edgecolor='#30363d', labelcolor='white')
ax3.set_ylabel('Drawdown %', color=GREY, fontsize=9)
ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax3.grid(axis='y', color='#21262d', linewidth=0.5)
ax3.set_xticklabels([])

# ── Capital utilisation ────────────────────────────────────────────────────────
ax4.fill_between(util_pct.index, util_pct.values, 0,
                 color=BLUE, alpha=0.4, linewidth=0)
ax4.plot(util_pct.index, util_pct.values, color=BLUE, linewidth=0.8)
ax4.axhline(n_active.mean() / N_SLOTS * 100, color=GREY, linewidth=0.6,
            linestyle=':', label=f'avg {avg_util:.1%}')
ax4.set_ylabel('Utilisation %', color=GREY, fontsize=9)
ax4.set_ylim(0, 12)
ax4.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax4.legend(loc='upper right', fontsize=7.5, facecolor='#161b22',
           edgecolor='#30363d', labelcolor='white')
ax4.grid(axis='y', color='#21262d', linewidth=0.5)
ax4.tick_params(axis='x', colors=GREY, labelsize=8)

plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print(f'Saved {OUTPUT_PNG}')
