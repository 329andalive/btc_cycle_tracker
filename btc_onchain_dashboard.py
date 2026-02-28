import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

plt.style.use('dark_background')
BG   = '#0d0d0d'
GRID = '#2a2a2a'
TEXT = '#e0e0e0'
GOLD = '#FFD700'

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. LOAD ALL DATASETS ──────────────────────────────────

def load_json(filename):
    with open(os.path.join(OUTPUT_DIR, filename)) as f:
        return json.load(f)

# MVRV Z-Score
mvrv_raw = load_json('mvrv_zscore.json')
mvrv = pd.DataFrame(mvrv_raw)
mvrv['date'] = pd.to_datetime(mvrv['d'])
mvrv['value'] = pd.to_numeric(mvrv['mvrvZscore'], errors='coerce')
mvrv = mvrv[['date', 'value']].dropna().sort_values('date')

# Realized Price
rp_raw = load_json('realized_price.json')
rp = pd.DataFrame(rp_raw)
rp['date'] = pd.to_datetime(rp['theDay'])   # note different field name
rp['value'] = pd.to_numeric(rp['realizedPrice'], errors='coerce')
rp = rp[['date', 'value']].dropna().sort_values('date')

# LTH Supply
lth_raw = load_json('lth_supply.json')
lth = pd.DataFrame(lth_raw)
lth['date'] = pd.to_datetime(lth['d'])
lth['value'] = pd.to_numeric(lth['longTermHodlerSupplyBtc'],
                              errors='coerce')
lth = lth[['date', 'value']].dropna().sort_values('date')
# Remove obviously bad values (negative supply makes no sense)
lth = lth[lth['value'] > 0]

# Exchange Netflow
nf_raw = load_json('exchange_netflow_btc.json')
nf = pd.DataFrame(nf_raw)
nf['date'] = pd.to_datetime(nf['d'])
nf['value'] = pd.to_numeric(nf['exchangeNetflowBtc'], errors='coerce')
nf = nf[['date', 'value']].dropna().sort_values('date')
# Smooth with 30-day rolling mean — daily netflow is very noisy
nf['smooth'] = nf['value'].rolling(30, center=True).mean()

# Exchange Outflow USD (Jan-Feb 2026 snapshot only)
of_raw = load_json('exchange_outflow_usd.json')
of = pd.DataFrame(of_raw)
of['date'] = pd.to_datetime(of['d'])
of['value'] = pd.to_numeric(of['exchangeOutflowUsd'], errors='coerce')
of = of[['date', 'value']].dropna().sort_values('date')

# ── 2. LOAD BTC PRICE for overlay ────────────────────────
btc = pd.read_csv(os.path.join(OUTPUT_DIR, 'btc-usd-max.csv'))
btc['date'] = pd.to_datetime(
    btc['snapped_at']).dt.tz_localize(None).dt.normalize()
btc = btc.rename(columns={'price': 'btc_price'})
btc = btc[['date', 'btc_price']].sort_values('date')

# ── 3. KEY DATES ──────────────────────────────────────────
HALVINGS = [
    pd.Timestamp('2012-11-28'),
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-19'),
]
CYCLE_TOPS = [
    pd.Timestamp('2013-12-04'),
    pd.Timestamp('2017-12-17'),
    pd.Timestamp('2021-11-10'),
    pd.Timestamp('2025-10-06'),
]
CYCLE_BOTTOMS = [
    pd.Timestamp('2015-01-14'),
    pd.Timestamp('2018-12-15'),
    pd.Timestamp('2022-11-21'),
]
TODAY = pd.Timestamp('2026-02-25')

# ── 4. CONSOLE SUMMARY ────────────────────────────────────
print("=" * 58)
print("ON-CHAIN DASHBOARD — CURRENT READINGS")
print("=" * 58)

latest_mvrv = float(mvrv['value'].iloc[-1])
latest_rp   = float(rp['value'].iloc[-1])
latest_btc  = float(btc['btc_price'].iloc[-1])
latest_lth  = float(lth['value'].iloc[-1])
latest_nf   = float(nf['smooth'].dropna().iloc[-1])
latest_of   = float(of['value'].iloc[-1])

print(f"\n  MVRV Z-Score:        {latest_mvrv:.2f}")
print(f"  Realized Price:      ${latest_rp:,.0f}")
print(f"  BTC Market Price:    ${latest_btc:,.0f}")
print(f"  Premium over RP:     "
      f"{((latest_btc - latest_rp) / latest_rp * 100):.1f}%")
print(f"\n  LTH Supply:          {latest_lth/1e6:.3f}M BTC")
print(f"  Exchange Netflow     "
      f"(30d avg): {latest_nf:+.0f} BTC/day")
print(f"  Exchange Outflow     "
      f"(latest): ${latest_of/1e9:.2f}B/day")

# MVRV historical context
mvrv_cycle_tops    = []
mvrv_cycle_bottoms = []
for t in CYCLE_TOPS:
    row = mvrv[mvrv['date'] <= t].iloc[-1]
    mvrv_cycle_tops.append(float(row['value']))
for b in CYCLE_BOTTOMS:
    row = mvrv[mvrv['date'] <= b].iloc[-1]
    mvrv_cycle_bottoms.append(float(row['value']))

print(f"\n  MVRV at cycle tops:    "
      f"{[f'{v:.1f}' for v in mvrv_cycle_tops]}")
print(f"  MVRV at cycle bottoms: "
      f"{[f'{v:.2f}' for v in mvrv_cycle_bottoms]}")
print(f"  MVRV today:            {latest_mvrv:.2f}")
print(f"\n  Accumulation support:  $53,923")
print(f"  Realized price:        ${latest_rp:,.0f}")
print(f"  Gap between them:      "
      f"${abs(latest_rp - 53923):,.0f}")
print("=" * 58)


# ── 5. BUILD DASHBOARD ────────────────────────────────────
# 4 panels sharing x-axis, price overlay on each

fig, axes = plt.subplots(4, 1, figsize=(16, 20),
                         facecolor=BG, sharex=False)
fig.suptitle(
    'Bitcoin On-Chain Dashboard — Testing the Accumulation Thesis\n'
    'MVRV Z-Score  |  Realized Price  |  '
    'LTH Supply  |  Exchange Netflow',
    fontsize=14, fontweight='bold', color=TEXT
)

def add_cycle_markers(ax, y_top_pct=0.92, y_bot_pct=0.08):
    """Add halving, top, bottom vertical lines to any axis."""
    ylim = ax.get_ylim()
    yrange = ylim[1] - ylim[0]
    for h in HALVINGS:
        ax.axvline(h, color='#888888', linewidth=0.8,
                   linestyle=':', alpha=0.5, zorder=2)
        ax.text(h, ylim[0] + yrange * 0.02, 'H',
                fontsize=7, color='#888888',
                alpha=0.6, ha='center')
    for t in CYCLE_TOPS:
        ax.axvline(t, color=GOLD, linewidth=0.8,
                   linestyle='--', alpha=0.4, zorder=2)
    for b in CYCLE_BOTTOMS:
        ax.axvline(b, color='#ff4d4d', linewidth=0.8,
                   linestyle='--', alpha=0.4, zorder=2)
    # Today
    ax.axvline(TODAY, color='white', linewidth=1.2,
               linestyle='-', alpha=0.5, zorder=3)

def format_ax(ax, title, ylabel, color):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=11,
                 fontweight='bold', color=color, pad=6)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.grid(linestyle='--', alpha=0.15, color=GRID)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.setp(ax.xaxis.get_majorticklabels(),
             rotation=45, ha='right', fontsize=8)

def add_price_twin(ax, btc_df, start_date=None):
    """Add BTC price as right-axis overlay."""
    if start_date:
        btc_slice = btc_df[btc_df['date'] >= start_date]
    else:
        btc_slice = btc_df
    ax2 = ax.twinx()
    ax2.plot(btc_slice['date'], btc_slice['btc_price'],
             color='white', linewidth=0.8,
             alpha=0.25, zorder=1)
    ax2.set_ylabel('BTC Price', fontsize=7,
                   color='white', alpha=0.4)
    ax2.tick_params(colors=TEXT, labelsize=7)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
    return ax2


# ── PANEL 1: MVRV Z-Score ─────────────────────────────────
ax1 = axes[0]

# Color zones
ax1.axhspan(7,   12,  color='#8b0000', alpha=0.15,
            label='Extreme overvalue (top zone)')
ax1.axhspan(3.5, 7,   color='#ff4500', alpha=0.10,
            label='Overvalued')
ax1.axhspan(1,   3.5, color='#ffd700', alpha=0.07,
            label='Fair to elevated')
ax1.axhspan(-1,  1,   color='#2ecc71', alpha=0.08,
            label='Accumulation zone')
ax1.axhspan(-3, -1,   color='#1a6b3c', alpha=0.12,
            label='Extreme undervalue (bottom zone)')

ax1.plot(mvrv['date'], mvrv['value'],
         color='#00d4ff', linewidth=1.5, zorder=3)
ax1.axhline(0, color='#2ecc71', linewidth=1.0,
            linestyle='--', alpha=0.7)
ax1.axhline(latest_mvrv, color='white', linewidth=1,
            linestyle=':', alpha=0.5)

# Annotate current value
ax1.scatter(TODAY, latest_mvrv,
            color='white', s=120, zorder=6, marker='*')
ax1.text(TODAY, latest_mvrv + 0.3,
         f' Today: {latest_mvrv:.2f}',
         fontsize=9, color='white', fontweight='bold')

# Zone labels
for y, label, col in [
    (9,    'Extreme Overvalue', '#ff4444'),
    (5,    'Overvalued',        '#ff8c00'),
    (2,    'Fair Value',        GOLD),
    (0,    'Accumulation',      '#2ecc71'),
    (-1.5, 'Extreme Fear',      '#1a9b5c'),
]:
    ax1.text(pd.Timestamp('2010-01-01'), y, label,
             fontsize=7.5, color=col, alpha=0.7, va='center')

add_price_twin(ax1, btc, '2012-01-01')
format_ax(ax1, 'MVRV Z-Score — Are Holders Overextended or Accumulating?',
          'Z-Score', '#00d4ff')
ax1.set_ylim(-3, 12)
add_cycle_markers(ax1)
ax1.legend(fontsize=7, facecolor='#1a1a1a',
           labelcolor=TEXT, loc='upper left',
           ncol=2)


# ── PANEL 2: Realized Price vs Market Price ───────────────
ax2 = axes[1]

# Merge price and realized price on date
merged = pd.merge(btc, rp.rename(columns={'value': 'rp'}),
                  on='date', how='inner')
merged = merged[merged['date'] >= pd.Timestamp('2012-01-01')]

ax2.plot(merged['date'], merged['btc_price'],
         color='#F7931A', linewidth=1.5,
         label='BTC Market Price', zorder=3)
ax2.plot(merged['date'], merged['rp'],
         color='#2ecc71', linewidth=1.5,
         linestyle='--', label='Realized Price (avg cost basis)',
         zorder=3)

# Shade gap — green when price above RP, red when below
ax2.fill_between(merged['date'],
                 merged['btc_price'], merged['rp'],
                 where=merged['btc_price'] >= merged['rp'],
                 alpha=0.12, color='#2ecc71',
                 label='Price above cost basis')
ax2.fill_between(merged['date'],
                 merged['btc_price'], merged['rp'],
                 where=merged['btc_price'] < merged['rp'],
                 alpha=0.20, color='#ff4d4d',
                 label='Price below cost basis (capitulation)')

# Mark accumulation support and realized price today
ax2.axhline(53923, color=GOLD, linewidth=1.2,
            linestyle=':', alpha=0.8,
            label=f'Accum. support $53,923')
ax2.scatter(TODAY, latest_btc,
            color='white', s=120, zorder=6, marker='*')
ax2.text(TODAY, latest_btc * 1.05,
         f' ${latest_btc:,.0f}',
         fontsize=8, color='white', fontweight='bold')
ax2.scatter(TODAY, latest_rp,
            color='#2ecc71', s=80, zorder=6, marker='D')
ax2.text(TODAY, latest_rp * 0.88,
         f' RP: ${latest_rp:,.0f}',
         fontsize=8, color='#2ecc71')

format_ax(ax2,
          'Realized Price vs Market Price — '
          'How Close Are We to Aggregate Cost Basis?',
          'Price (USD)', '#F7931A')
ax2.set_yscale('log')
ax2.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
add_cycle_markers(ax2)
ax2.legend(fontsize=7.5, facecolor='#1a1a1a',
           labelcolor=TEXT, loc='upper left')


# ── PANEL 3: LTH Supply ───────────────────────────────────
ax3 = axes[2]

lth_plot = lth[lth['date'] >= pd.Timestamp('2012-01-01')]

ax3.plot(lth_plot['date'], lth_plot['value'] / 1e6,
         color='#9b59b6', linewidth=1.5, zorder=3)
ax3.fill_between(lth_plot['date'],
                 lth_plot['value'] / 1e6,
                 alpha=0.10, color='#9b59b6')

# Latest value annotation
latest_lth_plot = float(lth_plot['value'].iloc[-1])
ax3.scatter(TODAY, latest_lth_plot / 1e6,
            color='white', s=120, zorder=6, marker='*')
ax3.text(TODAY, latest_lth_plot / 1e6 * 1.01,
         f' {latest_lth_plot/1e6:.2f}M BTC',
         fontsize=9, color='white', fontweight='bold')

# What thesis says: LTH supply should be RISING during bear
# Find LTH at each cycle bottom for context
print("\nLTH Supply at cycle events:")
for label, dates in [('Tops', CYCLE_TOPS),
                      ('Bottoms', CYCLE_BOTTOMS)]:
    for d in dates:
        row = lth[lth['date'] <= d]
        if len(row) > 0:
            val = float(row['value'].iloc[-1])
            print(f"  {label} {d.date()}: {val/1e6:.3f}M BTC")

add_price_twin(ax3, btc, '2012-01-01')
format_ax(ax3,
          'Long-Term Holder Supply — Are Accumulators Absorbing Supply?',
          'LTH Supply (Million BTC)', '#9b59b6')
add_cycle_markers(ax3)


# ── PANEL 4: Exchange Netflow ─────────────────────────────
ax4 = axes[3]

nf_plot = nf[nf['date'] >= pd.Timestamp('2012-01-01')]

# Color bars by direction
pos_mask = nf_plot['smooth'] >= 0
neg_mask = nf_plot['smooth'] < 0

ax4.fill_between(nf_plot['date'],
                 nf_plot['smooth'],
                 where=pos_mask,
                 color='#ff4d4d', alpha=0.6,
                 label='Inflow > Outflow (sell pressure)')
ax4.fill_between(nf_plot['date'],
                 nf_plot['smooth'],
                 where=neg_mask,
                 color='#2ecc71', alpha=0.6,
                 label='Outflow > Inflow (accumulation)')

ax4.plot(nf_plot['date'], nf_plot['smooth'],
         color=TEXT, linewidth=0.8, alpha=0.5, zorder=3)
ax4.axhline(0, color=TEXT, linewidth=0.8,
            alpha=0.4, linestyle='--')

# Latest reading
ax4.scatter(TODAY, latest_nf,
            color='white', s=120, zorder=6, marker='*')
ax4.text(TODAY, latest_nf,
         f'  {latest_nf:+.0f} BTC/day',
         fontsize=9, color='white', fontweight='bold')

add_price_twin(ax4, btc, '2012-01-01')
format_ax(ax4,
          'Exchange Netflow (30d avg) — '
          'Are Coins Leaving or Entering Exchanges?',
          'BTC/day', '#2ecc71')
ax4.legend(fontsize=8, facecolor='#1a1a1a',
           labelcolor=TEXT, loc='upper left')
add_cycle_markers(ax4)


# ── PANEL 5: Exchange Outflow USD — Jan/Feb 2026 ──────────
# Add as inset on panel 4 — recent snapshot
ax4_inset = ax4.inset_axes([0.02, 0.55, 0.25, 0.38])
ax4_inset.set_facecolor('#1a1a1a')
ax4_inset.bar(range(len(of)), of['value'] / 1e9,
              color='#2ecc71', alpha=0.8, width=0.7)
ax4_inset.set_xticks(range(len(of)))
ax4_inset.set_xticklabels(
    [d.strftime('%m/%d') for d in of['date']],
    fontsize=5, rotation=45, ha='right', color=TEXT
)
ax4_inset.tick_params(colors=TEXT, labelsize=5)
ax4_inset.set_title('Outflow USD\nJan-Feb 2026',
                    fontsize=6.5, color='#2ecc71',
                    fontweight='bold')
ax4_inset.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'${x:.1f}B'))
ax4_inset.grid(linestyle='--', alpha=0.2, color=GRID)


# ── SHARED LEGEND FOR CYCLE MARKERS ──────────────────────
from matplotlib.lines import Line2D
legend_lines = [
    Line2D([0], [0], color='#888888', linewidth=1,
           linestyle=':', label='Halving (H)'),
    Line2D([0], [0], color=GOLD, linewidth=1,
           linestyle='--', label='Cycle Top'),
    Line2D([0], [0], color='#ff4d4d', linewidth=1,
           linestyle='--', label='Cycle Bottom'),
    Line2D([0], [0], color='white', linewidth=1.2,
           linestyle='-', label='Today'),
]
fig.legend(handles=legend_lines,
           loc='lower center', ncol=4,
           fontsize=9, facecolor='#1a1a1a',
           labelcolor=TEXT,
           bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.04, 1, 1])
path = os.path.join(OUTPUT_DIR, 'btc_onchain_dashboard.png')
fig.savefig(path, dpi=150, facecolor=BG)
plt.close()
print(f"\nSaved: {path}")
print("\nAll done.")