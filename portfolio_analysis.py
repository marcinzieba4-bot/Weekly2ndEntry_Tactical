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
    """Fill in weekly returns for one stock trade leg (mark-to-market each week)."""
    if ticker not in price:
        return
    s = price[ticker]
    mask = (s.index > entry_date) & (s.index <= exit_date)
    trade_dates = s.index[mask]
    if len(trade_dates) == 0:
        return

    prev_price = entry_price
    for i, d in enumerate(trade_dates):
        is_last = (i == len(trade_dates) - 1)
        if is_last:
            curr = exit_price
        else:
            curr = s.get(d, np.nan)
            if np.isnan(curr):
                curr = prev_price

        ret = (curr / prev_price - 1) if prev_price > 0 else 0.0
        if d in stock_weekly_ret:
            stock_weekly_ret[d][ticker] = stock_weekly_ret[d].get(ticker, 0) + ret
        prev_price = curr


def apply_trade_option(ticker, entry_date, exit_date, pnl_pct_val):
    """
    Fill in weekly returns for one option leg.
    P&L is realised only at expiry (all intermediate weeks = 0).
    """
    if ticker not in price or pd.isna(exit_date) or pd.isna(pnl_pct_val):
        return
    s = price[ticker]
    mask = (s.index > entry_date) & (s.index <= exit_date)
    trade_dates = s.index[mask]
    if len(trade_dates) == 0:
        return
    last_d = trade_dates[-1]
    if last_d in stock_weekly_ret:
        stock_weekly_ret[last_d][ticker] = (
            stock_weekly_ret[last_d].get(ticker, 0) + pnl_pct_val / 100
        )


IS_OPTION = 'trade_mode' in trades.columns and (trades['trade_mode'] == 'option').any()

for _, row in trades.iterrows():
    ticker = row['ticker']
    if pd.isna(row['exit_date']):
        continue
    if IS_OPTION:
        apply_trade_option(ticker, row['entry_date'], row['exit_date'], row['pnl_pct'])
        if pd.notna(row['re_entry_date']) and pd.notna(row['re_exit_date']):
            apply_trade_option(ticker, row['re_entry_date'], row['re_exit_date'], row['re_pnl_pct'])
    else:
        apply_trade(ticker, row['entry_date'], row['entry_price'],
                    row['exit_date'], row['exit_price'], row['exit_reason'])
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
annual_ret     = port_rets.resample('YE').apply(lambda x: (1+x).prod() - 1)
annual_ret.index = annual_ret.index.year

dep_annual_ret = deployed_rets.resample('YE').apply(lambda x: (1+x).prod() - 1)
dep_annual_ret.index = dep_annual_ret.index.year

avg_annual_dep = dep_annual_ret.mean()
avg_annual_full = annual_ret.mean()

# Max DD duration (deployed)
in_dd_dep = dd_dep < 0
cur_dur = 0; dep_max_dd_dur = 0
for v in in_dd_dep:
    if v: cur_dur += 1; dep_max_dd_dur = max(dep_max_dd_dur, cur_dur)
    else: cur_dur = 0

# ── 7. Print stats ────────────────────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════════════╗
  STRATEGY: SPX Top-200 | Weekly MACD 2nd-Entry | SPY Filter
  OPTIONS:  ATM call, real IV premiums ×1.17, 8-week hold
  PERIOD:   {port_rets.index[0].date()} → {port_rets.index[-1].date()}
╚══════════════════════════════════════════════════════════════╝

  ── Deployed Capital (active option positions) ────────────
  Total return        : {dep_tr:+.1%}
  CAGR                : {dep_cagr:+.1%}
  Avg annual return   : {avg_annual_dep:+.1%}  (simple mean of yearly returns)
  Ann. volatility     : {dep_vol:.1%}
  Sharpe ratio        : {dep_sharpe:.2f}  (rf={RISK_FREE:.0%})
  Max drawdown        : {dep_max_dd:.1%}
  Max DD duration     : {dep_max_dd_dur} weeks
  Calmar ratio        : {dep_cagr/abs(dep_max_dd) if dep_max_dd != 0 else float('nan'):.2f}

  ── Full Portfolio (1/200 weight, cash for idle slots) ────
  Avg capital deployed: {avg_util:.1%}  (avg {n_active.mean():.1f} / {N_SLOTS} slots)
  CAGR                : {cagr:+.1%}
  Sharpe ratio        : {sharpe:.2f}
  Max drawdown        : {max_dd:.1%}

  ── Trade-level stats ──────────────────────────────────────
  Total trades        : {len(trades[trades['pnl_pct'].notna()])}
  ITM rate            : {win_rate:.1%}
  Avg winner          : +{avg_win:.2f}%
  Avg loser           : {avg_loss:.2f}%
  Profit factor       : {abs(avg_win * len(win_trades)) / abs(avg_loss * len(lose_trades)):.2f}
  Avg 8W premium      : {'N/A' if 'premium_pct' not in trades.columns else f"{trades['premium_pct'].mean():.2f}%"}
══════════════════════════════════════════════════════════════

  Year   Full port   Deployed    # trades
  ────   ─────────   ────────    ────────""")

trades['entry_date'] = pd.to_datetime(trades['entry_date'])
for yr in sorted(annual_ret.index):
    fp  = annual_ret.get(yr, float('nan'))
    dep = dep_annual_ret.get(yr, float('nan'))
    n   = (trades['entry_date'].dt.year == yr).sum()
    fp_s  = f'{fp:+.1%}' if not pd.isna(fp)  else '  N/A'
    dep_s = f'{dep:+.1%}' if not pd.isna(dep) else '  N/A'
    mark  = ' ◄' if not pd.isna(dep) and dep < -0.10 else ''
    print(f"  {yr}   {fp_s:>9}   {dep_s:>8}    {n:>4}{mark}")

print()

# ── 8. Chart ──────────────────────────────────────────────────────────────────
print(f'\nGenerating chart → {OUTPUT_PNG}')

DARK  = '#0d1117'
GREEN = '#26a641'
RED   = '#f85149'
GOLD  = '#d29922'
BLUE  = '#58a6ff'
GREY  = '#8b949e'
WHITE = '#e6edf3'

if IS_OPTION and 'premium_pct' in trades.columns:
    avg_p = trades['premium_pct'].mean()
    mode_label = f'Call Options  |  Real IV  |  Avg {avg_p:.1f}% 8W premium'
elif IS_OPTION:
    mode_label = 'Call Options  |  Real IV  |  8W hold'
else:
    mode_label = 'Stock positions'

fig = plt.figure(figsize=(14, 15), facecolor=DARK)
fig.suptitle(
    f'SPX Top-200  ·  Weekly MACD 2nd-Entry  ·  SPY Filter  ·  {mode_label}',
    color=WHITE, fontsize=13, fontweight='bold', y=0.99,
)

gs = gridspec.GridSpec(4, 2,
    height_ratios=[2.8, 1.8, 1.6, 1.6],
    width_ratios=[3, 1],
    hspace=0.10, wspace=0.08,
    left=0.07, right=0.97, top=0.96, bottom=0.05,
)

ax1  = fig.add_subplot(gs[0, 0])   # equity curves (deployed)
ax1r = fig.add_subplot(gs[0, 1])   # stats table
ax2  = fig.add_subplot(gs[1, :])   # annual returns (deployed, side-by-side full)
ax3  = fig.add_subplot(gs[2, 0])   # drawdown (deployed)
ax4  = fig.add_subplot(gs[2, 1])   # trade P&L histogram
ax5  = fig.add_subplot(gs[3, :])   # cumulative per-trade P&L

for ax in (ax1, ax2, ax3, ax4, ax5):
    ax.set_facecolor(DARK)
    ax.tick_params(colors=GREY, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
ax1r.set_facecolor(DARK)
ax1r.axis('off')

# ── Panel 1: Equity curves ────────────────────────────────────────────────────
ax1.plot(eq_dep.index, eq_dep.values, color=GOLD, linewidth=1.6,
         label=f'Deployed capital  CAGR {dep_cagr:+.1%}')
ax1.plot(equity.index, equity.values, color=BLUE, linewidth=1.0, alpha=0.6,
         linestyle=':', label=f'Full portfolio  CAGR {cagr:+.1%}')
ax1.fill_between(eq_dep.index, 1, eq_dep.values,
                 where=(eq_dep.values >= 1), alpha=0.12, color=GREEN)
ax1.fill_between(eq_dep.index, 1, eq_dep.values,
                 where=(eq_dep.values <  1), alpha=0.12, color=RED)
ax1.axhline(1, color=GREY, linewidth=0.5, linestyle='--')
ax1.legend(loc='upper left', fontsize=8, facecolor='#161b22',
           edgecolor='#30363d', labelcolor=WHITE)
ax1.set_ylabel('Growth of $1 (deployed)', color=GREY, fontsize=8.5)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:.1f}'))
ax1.grid(axis='y', color='#21262d', linewidth=0.4)
ax1.set_xticklabels([])
ax1.set_title('Cumulative P&L — Deployed Capital', color=GREY, fontsize=9, pad=4)

# ── Panel 1R: Stats table ─────────────────────────────────────────────────────
profit_factor = abs(avg_win * len(win_trades)) / abs(avg_loss * len(lose_trades))
stats_lines = [
    ('DEPLOYED CAPITAL', ''),
    ('CAGR',             f'{dep_cagr:+.1%}'),
    ('Avg annual ret',   f'{avg_annual_dep:+.1%}'),
    ('Ann. volatility',  f'{dep_vol:.1%}'),
    ('Sharpe ratio',     f'{dep_sharpe:.2f}'),
    ('Max drawdown',     f'{dep_max_dd:.1%}'),
    ('Max DD duration',  f'{dep_max_dd_dur}w'),
    ('Calmar ratio',     f'{dep_cagr/abs(dep_max_dd) if dep_max_dd!=0 else 0:.2f}'),
    ('', ''),
    ('TRADES', ''),
    ('Total',            f'{len(trades[trades["pnl_pct"].notna()])}'),
    ('ITM rate',         f'{win_rate:.1%}'),
    ('Avg winner',       f'+{avg_win:.1f}%'),
    ('Avg loser',        f'{avg_loss:.1f}%'),
    ('Profit factor',    f'{profit_factor:.2f}'),
]
y0 = 0.98
for label, val in stats_lines:
    if val == '':
        col = GOLD if label else GREY
        ax1r.text(0.05, y0, label, transform=ax1r.transAxes,
                  color=col, fontsize=8.5, fontweight='bold')
    else:
        ax1r.text(0.05, y0, label, transform=ax1r.transAxes,
                  color=GREY, fontsize=8)
        color_val = WHITE
        if label == 'Max drawdown': color_val = RED
        elif label in ('CAGR', 'Avg annual ret', 'Sharpe ratio', 'Profit factor', 'Calmar ratio'):
            color_val = GREEN if (val.startswith('+') or (val[0].isdigit() and float(val.split()[0]) > 0)) else RED
        ax1r.text(0.65, y0, val, transform=ax1r.transAxes,
                  color=color_val, fontsize=8, fontweight='bold', ha='right')
    y0 -= 0.065

# ── Panel 2: Annual returns (deployed vs full, side-by-side) ─────────────────
years_idx = sorted(set(dep_annual_ret.index) | set(annual_ret.index))
x    = np.arange(len(years_idx))
w    = 0.38
dep_vals  = [dep_annual_ret.get(yr, 0) * 100 for yr in years_idx]
full_vals = [annual_ret.get(yr, 0) * 100 for yr in years_idx]

b1 = ax2.bar(x - w/2, dep_vals,  width=w, zorder=3,
             color=[GREEN if v >= 0 else RED for v in dep_vals],  alpha=0.9,
             label='Deployed capital')
b2 = ax2.bar(x + w/2, full_vals, width=w, zorder=3,
             color=[BLUE if v >= 0 else '#6e40c9' for v in full_vals], alpha=0.6,
             label='Full portfolio')

ax2.axhline(0, color=GREY, linewidth=0.6)
ax2.set_ylabel('Annual Return %', color=GREY, fontsize=8.5)
ax2.set_xticks(x)
ax2.set_xticklabels(years_idx, color=GREY, fontsize=8.5)
ax2.grid(axis='y', color='#21262d', linewidth=0.4, zorder=0)
ax2.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor=WHITE, loc='upper left')
ax2.set_title('Annual Returns', color=GREY, fontsize=9, pad=4)

for bar, val in zip(b1, dep_vals):
    if abs(val) < 0.5: continue
    ax2.text(bar.get_x() + bar.get_width()/2,
             val + (0.5 if val >= 0 else -0.5),
             f'{val:+.0f}%', ha='center',
             va='bottom' if val >= 0 else 'top',
             color=WHITE, fontsize=7, fontweight='bold')

# ── Panel 3: Drawdown (deployed) ──────────────────────────────────────────────
ax3.fill_between(dd_dep.index, dd_dep.values * 100, 0,
                 color=RED, alpha=0.55, linewidth=0)
ax3.plot(dd_dep.index, dd_dep.values * 100, color=RED, linewidth=0.8)
ax3.axhline(0, color=GREY, linewidth=0.4)
worst_dd_date = dd_dep.idxmin()
ax3.annotate(f'{dep_max_dd:.1%}',
             xy=(worst_dd_date, dep_max_dd * 100),
             xytext=(20, 10), textcoords='offset points',
             color=RED, fontsize=8.5, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=RED, lw=0.8))
ax3.set_ylabel('Drawdown %', color=GREY, fontsize=8.5)
ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax3.grid(axis='y', color='#21262d', linewidth=0.4)
ax3.set_title('Drawdown — Deployed Capital', color=GREY, fontsize=9, pad=4)
ax3.tick_params(axis='x', colors=GREY, labelsize=8)

# ── Panel 4: Trade P&L distribution ───────────────────────────────────────────
all_pnls = pd.concat([
    trades['pnl_pct'].dropna(),
    trades['re_pnl_pct'].dropna(),
])
bins = np.linspace(all_pnls.min() - 1, min(all_pnls.max() + 1, 50), 50)
n_itm = (all_pnls > 0).sum()
n_otm = (all_pnls <= 0).sum()
ax4.hist(all_pnls[all_pnls > 0],  bins=bins, color=GREEN, alpha=0.75, label=f'ITM {n_itm}')
ax4.hist(all_pnls[all_pnls <= 0], bins=bins, color=RED,   alpha=0.75, label=f'OTM {n_otm}')
ax4.axvline(all_pnls.mean(), color=GOLD, linewidth=1.2, linestyle='--',
            label=f'mean {all_pnls.mean():+.1f}%')
ax4.axvline(0, color=GREY, linewidth=0.6)
ax4.set_xlabel('P&L %', color=GREY, fontsize=8.5)
ax4.set_ylabel('# trades', color=GREY, fontsize=8.5)
ax4.legend(fontsize=7.5, facecolor='#161b22', edgecolor='#30363d', labelcolor=WHITE)
ax4.grid(axis='y', color='#21262d', linewidth=0.4)
ax4.set_title('Trade P&L Distribution', color=GREY, fontsize=9, pad=4)

# ── Panel 5: Cumulative per-trade P&L (waterfall-style) ──────────────────────
trade_seq = trades[trades['pnl_pct'].notna()].sort_values('entry_date').copy()
cum_pnl   = trade_seq['pnl_pct'].cumsum().values
trade_x   = np.arange(len(cum_pnl))

ax5.plot(trade_x, cum_pnl, color=GOLD, linewidth=1.0, zorder=3)
ax5.fill_between(trade_x, 0, cum_pnl,
                 where=(cum_pnl >= 0), alpha=0.15, color=GREEN, zorder=2)
ax5.fill_between(trade_x, 0, cum_pnl,
                 where=(cum_pnl <  0), alpha=0.15, color=RED,   zorder=2)
ax5.axhline(0, color=GREY, linewidth=0.5, linestyle='--')
ax5.set_xlabel('Trade #', color=GREY, fontsize=8.5)
ax5.set_ylabel('Cumulative P&L %', color=GREY, fontsize=8.5)
ax5.grid(axis='y', color='#21262d', linewidth=0.4)
ax5.set_title(f'Cumulative Trade P&L  (avg per trade {all_pnls.mean():+.2f}%,  total {cum_pnl[-1]:+.0f}%)',
              color=GREY, fontsize=9, pad=4)
ax5.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:+.0f}%'))

plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print(f'Saved {OUTPUT_PNG}')
