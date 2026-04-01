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

port_rets     = []
n_active_list = []

for d in all_dates:
    active = stock_weekly_ret.get(d, {})
    n      = len(active)
    pr     = sum(WEIGHT * r for r in active.values())
    port_rets.append(pr)
    n_active_list.append(n)

port_rets = pd.Series(port_rets,     index=all_dates, name='portfolio')
n_active  = pd.Series(n_active_list, index=all_dates, name='n_active')

port_rets = port_rets[port_rets.index >= '2015-01-01']
n_active  = n_active [n_active.index  >= '2015-01-01']

# ── 4. Equity curve & drawdown ────────────────────────────────────────────────
equity   = (1 + port_rets).cumprod()
roll_max = equity.cummax()
drawdown = (equity / roll_max - 1)
avg_util = n_active.mean() / N_SLOTS
util_pct = (n_active / N_SLOTS * 100).rolling(4).mean()

# ── 5. Risk statistics ────────────────────────────────────────────────────────
weeks_per_year = 52
rf_weekly      = (1 + RISK_FREE) ** (1 / weeks_per_year) - 1
total_weeks    = len(port_rets)
years          = total_weeks / weeks_per_year

total_return  = equity.iloc[-1] - 1
cagr          = (equity.iloc[-1]) ** (1 / years) - 1
vol_ann       = port_rets.std() * np.sqrt(weeks_per_year)
sharpe        = (cagr - RISK_FREE) / vol_ann
downside      = port_rets[port_rets < rf_weekly] - rf_weekly
sortino_denom = downside.std() * np.sqrt(weeks_per_year)
sortino       = (cagr - RISK_FREE) / sortino_denom if sortino_denom > 0 else np.nan
max_dd        = drawdown.min()
calmar        = cagr / abs(max_dd) if max_dd != 0 else np.nan

in_dd = drawdown < 0
cur_dur = 0; max_dd_dur = 0
for v in in_dd:
    if v: cur_dur += 1; max_dd_dur = max(max_dd_dur, cur_dur)
    else: cur_dur = 0

win_trades  = trades[trades['pnl_pct'] > 0]
lose_trades = trades[trades['pnl_pct'] <= 0]
win_rate    = len(win_trades) / len(trades[trades['pnl_pct'].notna()])
avg_win     = win_trades['pnl_pct'].mean()
avg_loss    = lose_trades['pnl_pct'].mean()
profit_factor = abs(avg_win * len(win_trades)) / abs(avg_loss * len(lose_trades))

# ── 6. Annual returns ─────────────────────────────────────────────────────────
annual_ret       = port_rets.resample('YE').apply(lambda x: (1+x).prod() - 1)
annual_ret.index = annual_ret.index.year
avg_annual       = annual_ret.mean()

# ── 7. Print stats ────────────────────────────────────────────────────────────
avg_prem_str = f"{trades['premium_pct'].mean():.2f}%" if 'premium_pct' in trades.columns else 'N/A'

print(f"""
╔══════════════════════════════════════════════════════════════╗
  STRATEGY: SPX Top-200 | Weekly MACD 2nd-Entry | SPY Filter
  OPTIONS:  ATM call, real IV×1.17, 8W hold | 1/200 equal weight
  PERIOD:   {port_rets.index[0].date()} → {port_rets.index[-1].date()}
╚══════════════════════════════════════════════════════════════╝

  ── Portfolio (equal weight 1/200, cash for idle slots) ───
  Avg slots active    : {n_active.mean():.1f} / {N_SLOTS}  ({avg_util:.1%} utilisation)
  Total return        : {total_return:+.1%}
  CAGR                : {cagr:+.1%}
  Avg annual return   : {avg_annual:+.1%}
  Ann. volatility     : {vol_ann:.1%}
  Sharpe ratio        : {sharpe:.2f}  (rf={RISK_FREE:.0%})
  Sortino ratio       : {sortino:.2f}
  Max drawdown        : {max_dd:.1%}
  Max DD duration     : {max_dd_dur} weeks
  Calmar ratio        : {calmar:.2f}

  ── Trade-level stats ──────────────────────────────────────
  Total trades        : {len(trades[trades['pnl_pct'].notna()])}
  ITM rate            : {win_rate:.1%}
  Avg winner          : +{avg_win:.2f}%
  Avg loser           : {avg_loss:.2f}%
  Profit factor       : {profit_factor:.2f}
  Avg 8W premium      : {avg_prem_str}
══════════════════════════════════════════════════════════════

  Year   Return    # trades
  ────   ──────    ────────""")

trades['entry_date'] = pd.to_datetime(trades['entry_date'])
for yr in sorted(annual_ret.index):
    ret = annual_ret.get(yr, float('nan'))
    n   = (trades['entry_date'].dt.year == yr).sum()
    ret_s = f'{ret:+.2%}' if not pd.isna(ret) else '  N/A'
    bar   = ('█' * int(abs(ret) * 400)) if not pd.isna(ret) else ''
    sign  = '+' if (not pd.isna(ret) and ret >= 0) else '-'
    print(f"  {yr}   {ret_s:>8}    {n:>4}   {bar}")

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

avg_p      = trades['premium_pct'].mean() if 'premium_pct' in trades.columns else None
mode_label = (f'Call Options  ·  Real IV  ·  Avg {avg_p:.1f}% 8W premium' if avg_p
              else ('Call Options' if IS_OPTION else 'Stock'))

fig = plt.figure(figsize=(14, 14), facecolor=DARK)
fig.suptitle(
    f'SPX Top-200  ·  Weekly MACD 2nd-Entry  ·  SPY Filter  ·  {mode_label}  ·  1/200 Weight',
    color=WHITE, fontsize=13, fontweight='bold', y=0.99,
)

gs = gridspec.GridSpec(4, 2,
    height_ratios=[2.6, 1.8, 1.6, 1.6],
    width_ratios=[3, 1],
    hspace=0.12, wspace=0.08,
    left=0.07, right=0.97, top=0.96, bottom=0.05,
)

ax1  = fig.add_subplot(gs[0, 0])   # equity curve
ax1r = fig.add_subplot(gs[0, 1])   # stats table
ax2  = fig.add_subplot(gs[1, :])   # annual returns
ax3  = fig.add_subplot(gs[2, 0])   # drawdown
ax4  = fig.add_subplot(gs[2, 1])   # trade P&L histogram
ax5  = fig.add_subplot(gs[3, :])   # cumulative per-trade P&L

for ax in (ax1, ax2, ax3, ax4, ax5):
    ax.set_facecolor(DARK)
    ax.tick_params(colors=GREY, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
ax1r.set_facecolor(DARK)
ax1r.axis('off')

# ── Panel 1: Equity curve ────────────────────────────────────────────────────
ax1.plot(equity.index, equity.values, color=BLUE, linewidth=1.6,
         label=f'Portfolio  CAGR {cagr:+.1%}')
ax1.fill_between(equity.index, 1, equity.values,
                 where=(equity.values >= 1), alpha=0.12, color=GREEN)
ax1.fill_between(equity.index, 1, equity.values,
                 where=(equity.values <  1), alpha=0.12, color=RED)
ax1.axhline(1, color=GREY, linewidth=0.5, linestyle='--')
ax1.legend(loc='upper left', fontsize=8.5, facecolor='#161b22',
           edgecolor='#30363d', labelcolor=WHITE)
ax1.set_ylabel('Growth of $1  (1/200 weight)', color=GREY, fontsize=8.5)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:.3f}'))
ax1.grid(axis='y', color='#21262d', linewidth=0.4)
ax1.set_xticklabels([])
ax1.set_title('Portfolio Equity Curve  (equal weight, cash for idle slots)', color=GREY, fontsize=9, pad=4)

# ── Panel 1R: Stats table ─────────────────────────────────────────────────────
stats_lines = [
    ('PORTFOLIO  1/200', ''),
    ('CAGR',            f'{cagr:+.1%}'),
    ('Avg annual ret',  f'{avg_annual:+.1%}'),
    ('Ann. volatility', f'{vol_ann:.1%}'),
    ('Sharpe ratio',    f'{sharpe:.2f}'),
    ('Sortino ratio',   f'{sortino:.2f}'),
    ('Max drawdown',    f'{max_dd:.1%}'),
    ('Max DD dur',      f'{max_dd_dur}w'),
    ('Calmar ratio',    f'{calmar:.2f}'),
    ('Avg utilisation', f'{avg_util:.1%}'),
    ('', ''),
    ('TRADES', ''),
    ('Total',           f'{len(trades[trades["pnl_pct"].notna()])}'),
    ('ITM rate',        f'{win_rate:.1%}'),
    ('Avg winner',      f'+{avg_win:.1f}%'),
    ('Avg loser',       f'{avg_loss:.1f}%'),
    ('Profit factor',   f'{profit_factor:.2f}'),
]
y0 = 0.98
for label, val in stats_lines:
    if val == '':
        ax1r.text(0.05, y0, label, transform=ax1r.transAxes,
                  color=GOLD if label else GREY, fontsize=8.5, fontweight='bold')
    else:
        ax1r.text(0.05, y0, label, transform=ax1r.transAxes, color=GREY, fontsize=8)
        if label == 'Max drawdown':
            cv = RED
        elif label in ('CAGR', 'Avg annual ret', 'Sharpe ratio', 'Sortino ratio',
                       'Profit factor', 'Calmar ratio'):
            cv = GREEN if val.startswith('+') or (val[0].isdigit() and float(val) > 0) else RED
        else:
            cv = WHITE
        ax1r.text(0.68, y0, val, transform=ax1r.transAxes,
                  color=cv, fontsize=8, fontweight='bold', ha='right')
    y0 -= 0.058

# ── Panel 2: Annual returns ────────────────────────────────────────────────────
yrs   = sorted(annual_ret.index)
x     = np.arange(len(yrs))
vals  = [annual_ret.get(yr, 0) * 100 for yr in yrs]
bars  = ax2.bar(x, vals, width=0.6, zorder=3,
                color=[GREEN if v >= 0 else RED for v in vals], alpha=0.88)
ax2.axhline(0, color=GREY, linewidth=0.6)
ax2.set_ylabel('Annual Return %', color=GREY, fontsize=8.5)
ax2.set_xticks(x)
ax2.set_xticklabels(yrs, color=GREY, fontsize=8.5)
ax2.grid(axis='y', color='#21262d', linewidth=0.4, zorder=0)
ax2.set_title(f'Annual Returns  (avg {avg_annual:+.2%}/yr)', color=GREY, fontsize=9, pad=4)
for bar, val in zip(bars, vals):
    if abs(val) < 0.05: continue
    ax2.text(bar.get_x() + bar.get_width()/2,
             val + (0.02 if val >= 0 else -0.02),
             f'{val:+.2f}%', ha='center',
             va='bottom' if val >= 0 else 'top',
             color=WHITE, fontsize=7.5, fontweight='bold')

# ── Panel 3: Drawdown ─────────────────────────────────────────────────────────
ax3.fill_between(drawdown.index, drawdown.values * 100, 0,
                 color=RED, alpha=0.55, linewidth=0)
ax3.plot(drawdown.index, drawdown.values * 100, color=RED, linewidth=0.8)
ax3.axhline(0, color=GREY, linewidth=0.4)
worst_dd_date = drawdown.idxmin()
ax3.annotate(f'{max_dd:.2%}',
             xy=(worst_dd_date, max_dd * 100),
             xytext=(20, 10), textcoords='offset points',
             color=RED, fontsize=8.5, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=RED, lw=0.8))
ax3.set_ylabel('Drawdown %', color=GREY, fontsize=8.5)
ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}%'))
ax3.grid(axis='y', color='#21262d', linewidth=0.4)
ax3.set_title('Portfolio Drawdown', color=GREY, fontsize=9, pad=4)
ax3.tick_params(axis='x', colors=GREY, labelsize=8)

# ── Panel 4: Trade P&L distribution ───────────────────────────────────────────
all_pnls = pd.concat([trades['pnl_pct'].dropna(), trades['re_pnl_pct'].dropna()])
bins  = np.linspace(all_pnls.min() - 0.5, min(all_pnls.max() + 0.5, 50), 50)
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

# ── Panel 5: Cumulative per-trade P&L ────────────────────────────────────────
trade_seq = trades[trades['pnl_pct'].notna()].sort_values('entry_date').copy()
cum_pnl   = trade_seq['pnl_pct'].cumsum().values
trade_x   = np.arange(len(cum_pnl))
ax5.plot(trade_x, cum_pnl, color=GOLD, linewidth=1.0, zorder=3)
ax5.fill_between(trade_x, 0, cum_pnl,
                 where=(cum_pnl >= 0), alpha=0.15, color=GREEN, zorder=2)
ax5.fill_between(trade_x, 0, cum_pnl,
                 where=(cum_pnl < 0),  alpha=0.15, color=RED,   zorder=2)
ax5.axhline(0, color=GREY, linewidth=0.5, linestyle='--')
ax5.set_xlabel('Trade #', color=GREY, fontsize=8.5)
ax5.set_ylabel('Cumulative P&L %', color=GREY, fontsize=8.5)
ax5.grid(axis='y', color='#21262d', linewidth=0.4)
ax5.set_title(
    f'Cumulative Trade P&L  (avg/trade {all_pnls.mean():+.2f}%  ·  total {cum_pnl[-1]:+.0f}%)',
    color=GREY, fontsize=9, pad=4,
)
ax5.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:+.0f}%'))

plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print(f'Saved {OUTPUT_PNG}')
