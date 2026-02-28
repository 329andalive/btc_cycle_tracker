import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('dark_background')
BG   = '#0d0d0d'
GRID = '#2a2a2a'
TEXT = '#e0e0e0'
GOLD = '#FFD700'

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

STOCKS = {
    'AAPL': {'name': 'Apple',  'color': '#a8d8ea'},
    'AMZN': {'name': 'Amazon', 'color': '#f39c12'},
    'TSLA': {'name': 'Tesla',  'color': '#e74c3c'},
}

# ── Define the major bottoms — start of accumulation phases
# These are the confirmed cycle troughs from our analysis
ACCUMULATION_STARTS = {
    'AAPL': [
        {'label': 'Post dot-com',  'bottom': '2003-04-14', 'color': '#e74c3c'},
        {'label': 'Post GFC',      'bottom': '2009-01-20', 'color': '#f39c12'},
        {'label': 'Post 2022',     'bottom': '2023-01-03', 'color': '#2ecc71'},
    ],
    'AMZN': [
        {'label': 'Post dot-com',  'bottom': '2001-09-28', 'color': '#e74c3c'},
        {'label': 'Post GFC',      'bottom': '2008-11-21', 'color': '#f39c12'},
        {'label': 'Post 2022',     'bottom': '2022-12-28', 'color': '#2ecc71'},
    ],
    'TSLA': [
        {'label': 'Post 2022',     'bottom': '2022-12-27', 'color': '#e74c3c'},
    ],
}

TODAY     = pd.Timestamp('2026-02-26')
PLOT_DAYS = 504   # ~2 trading years

print("Downloading stock data...")
frames = {}
for ticker in STOCKS:
    df = yf.download(ticker, start='1995-01-01',
                     auto_adjust=True, progress=False)
    df = df[['Close']].copy()
    df.columns = ['price']
    df.index   = pd.to_datetime(df.index)
    frames[ticker] = df
    print(f"  {ticker}: {len(df)} trading days")

# ── ANALYSIS ──────────────────────────────────────────────
print("\nACCUMULATION SUPPORT ANALYSIS")
print("=" * 60)

fig, axes = plt.subplots(3, 1, figsize=(16, 18), facecolor=BG)
fig.suptitle(
    'Apple  |  Amazon  |  Tesla\n'
    'Does the First 200-Day Accumulation Low Hold as Support?',
    fontsize=14, fontweight='bold', color=TEXT
)

for ax, (ticker, meta) in zip(axes, STOCKS.items()):
    ax.set_facecolor(BG)
    df     = frames[ticker]
    phases = ACCUMULATION_STARTS[ticker]

    print(f"\n{meta['name']} ({ticker})")

    for phase in phases:
        bottom_dt = pd.Timestamp(phase['bottom'])
        end_dt    = min(bottom_dt + pd.Timedelta(days=PLOT_DAYS * 2), TODAY)
        col       = phase['color']

        # Slice from bottom onward
        mask  = (df.index >= bottom_dt) & (df.index <= end_dt)
        chunk = df[mask].copy()
        if len(chunk) < 10:
            continue

        chunk['td']  = range(len(chunk))   # trading days from bottom
        bottom_price = float(chunk['price'].iloc[0])

        # Normalize to % from bottom price
        chunk['pct'] = ((chunk['price'] - bottom_price)
                        / bottom_price) * 100

        # First 200 TRADING days low
        first200  = chunk[chunk['td'] <= 200]
        low_idx   = first200['pct'].idxmin()
        low_pct   = float(first200.loc[low_idx, 'pct'])
        low_td    = int(first200.loc[low_idx, 'td'])
        low_price = float(first200.loc[low_idx, 'price'])

        # Post-200 low
        after200    = chunk[chunk['td'] > 200]
        if len(after200) > 0:
            after_low     = float(after200['pct'].min())
            after_low_td  = int(after200.loc[
                after200['pct'].idxmin(), 'td'])
            after_low_px  = float(after200.loc[
                after200['pct'].idxmin(), 'price'])
            held = after_low > low_pct
        else:
            after_low    = None
            after_low_td = None
            held         = None

        print(f"\n  {phase['label']}")
        print(f"    Bottom date:      {bottom_dt.date()}  "
              f"${bottom_price:,.2f}")
        print(f"    200d accum low:   {low_pct:+.1f}%  "
              f"day {low_td}  ${low_price:,.2f}")
        if after_low is not None:
            print(f"    Post-200d low:    {after_low:+.1f}%  "
                  f"day {after_low_td}  ${after_low_px:,.2f}")
            print(f"    Support held:     "
                  f"{'✅ YES' if held else '❌ NO'}")

        # ── Plot ──────────────────────────────────────────
        ax.plot(chunk['td'], chunk['pct'],
                color=col, linewidth=1.8,
                label=phase['label'], zorder=3, alpha=0.9)
        ax.fill_between(chunk['td'], chunk['pct'], -120,
                        alpha=0.04, color=col)

        # Shade accumulation zone
        ax.axvspan(0, 200, alpha=0.06, color='white', zorder=1)

        # Support line
        ax.axhline(low_pct, color=GOLD, linewidth=1.5,
                   linestyle='--', alpha=0.7, zorder=4)

        # Mark accumulation low
        ax.scatter(low_td, low_pct, color=GOLD,
                   s=100, zorder=6, marker='o',
                   edgecolors='white', linewidths=0.6)
        ax.annotate(
            f"{phase['label']}\n200d low: {low_pct:+.1f}%\n"
            f"Day {low_td}  ${low_price:,.2f}",
            xy=(low_td, low_pct),
            xytext=(low_td + 30, low_pct - 15),
            fontsize=7.5, color=GOLD,
            arrowprops=dict(arrowstyle='->',
                            color=GOLD, lw=0.8),
        )

        # Status badge per phase
        if held is not None:
            status = '✅ Held' if held else '❌ Broken'
            color  = '#00ff88' if held else '#ff4d4d'
            ax.text(low_td + 30, low_pct + 8,
                    status, fontsize=8,
                    color=color, fontweight='bold')

    # Vertical line at day 200
    ax.axvline(200, color='white', linewidth=1,
               linestyle=':', alpha=0.35)
    ax.text(202, ax.get_ylim()[0] * 0.85 if ax.get_ylim()[0] < 0
            else 5,
            '200d\nmark', fontsize=8,
            color=TEXT, alpha=0.5)

    ax.set_title(f"{meta['name']} ({ticker}) — "
                 f"Accumulation Zone Support",
                 fontsize=11, fontweight='bold',
                 color=meta['color'])
    ax.set_xlabel('Trading Days from Cycle Bottom', fontsize=9,
                  color=TEXT)
    ax.set_ylabel('% from Cycle Bottom Price', fontsize=9,
                  color=TEXT)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x:+.0f}%'))
    ax.tick_params(colors=TEXT)
    ax.grid(linestyle='--', alpha=0.15, color=GRID)
    ax.legend(fontsize=8, facecolor='#1a1a1a',
              labelcolor=TEXT, loc='upper left')
    ax.axhline(0, color=TEXT, linewidth=0.8,
               linestyle='--', alpha=0.2)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, 'stocks_accumulation.png')
fig.savefig(path, dpi=150, facecolor=BG)
plt.close()
print(f"\nSaved: {path}")
print("\nAll done.")
