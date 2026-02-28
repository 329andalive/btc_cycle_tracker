import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# ── DARK THEME ────────────────────────────────────────────
plt.style.use('dark_background')
BACKGROUND  = '#0d0d0d'   # near-black
GRID_COLOR  = '#2a2a2a'   # subtle grid
TEXT_COLOR  = '#e0e0e0'   # off-white text
SUPPORT_COLOR = '#FFD700' # gold — reads well on black
ACCUM_SHADE   = '#ffffff'

TODAY     = pd.Timestamp('2026-02-26')
PLOT_DAYS = 1100

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

btc = pd.read_csv(os.path.join(OUTPUT_DIR, 'btc-usd-max.csv'))
btc['date'] = pd.to_datetime(btc['snapped_at']).dt.tz_localize(None).dt.normalize()
btc = btc.rename(columns={'price': 'btc_price'})
btc = btc[['date','btc_price']].sort_values('date').reset_index(drop=True)

CYCLES = [
    {
        'name':      'Cycle 1  (Halving Nov 2012)',
        'halving':   '2012-11-28',
        'top':       '2013-12-04',
        'top_price': 1150,
        'confirmed': True,
        'exclude':   True,   # Mt. Gox black swan
        'color':     '#888888',
    },
    {
        'name':      'Cycle 2  (Halving Jul 2016)',
        'halving':   '2016-07-09',
        'top':       '2017-12-17',
        'top_price': 20000,
        'confirmed': True,
        'exclude':   False,
        'color':     '#f39c12',
    },
    {
        'name':      'Cycle 3  (Halving May 2020)',
        'halving':   '2020-05-11',
        'top':       '2021-11-10',
        'top_price': 69000,
        'confirmed': True,
        'exclude':   False,
        'color':     '#2ecc71',
    },
    {
        'name':      'Cycle 4  (Halving Apr 2024)',
        'halving':   '2024-04-19',
        'top':       '2025-10-06',
        'top_price': 124774,
        'confirmed': False,
        'exclude':   False,
        'color':     '#3498db',
    },
]

TODAY     = pd.Timestamp('2026-02-25')
PLOT_DAYS = 1100

fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharey=False,
                         facecolor=BACKGROUND)
fig.suptitle(
    'Bitcoin — Early Accumulation Low as Bear Market Support\n'
    'The lowest price in the first 200 days after halving has held as support '
    'through the full bear market in every mature cycle',
    fontsize=13, fontweight='bold'
)

for ax, c in zip(axes.flat, CYCLES):
    h   = pd.Timestamp(c['halving'])
    t   = pd.Timestamp(c['top'])
    end = h + pd.Timedelta(days=PLOT_DAYS)
    if not c['confirmed']:
        end = min(end, TODAY)

    mask  = (btc['date'] >= h) & (btc['date'] <= end)
    chunk = btc[mask].copy()
    chunk['days'] = (chunk['date'] - h).dt.days
    chunk['pct']  = ((chunk['btc_price'] - c['top_price'])
                     / c['top_price']) * 100

    top_day      = (t - h).days

    # ── Find the accumulation zone low (first 200 days) ───
    accum        = chunk[chunk['days'] <= 200]
    low_idx      = accum['pct'].idxmin()
    low_pct      = accum.loc[low_idx, 'pct']
    low_day      = int(accum.loc[low_idx, 'days'])
    low_price    = accum.loc[low_idx, 'btc_price']

    # ── Find bear market low (post top) ───────────────────
    post_top     = chunk[chunk['days'] > top_day]
    bear_low_pct   = post_top['pct'].min()   if len(post_top) > 0 else None
    bear_low_day   = int(post_top.loc[post_top['pct'].idxmin(), 'days']) if len(post_top) > 0 else None
    bear_low_price = post_top.loc[post_top['pct'].idxmin(), 'btc_price'] if len(post_top) > 0 else None

    held = (bear_low_pct > low_pct) if bear_low_pct is not None else None

    # ── Print to console ──────────────────────────────────
    print(f"{c['name']}")
    print(f"  Accumulation low:  {low_pct:+.1f}%  day {low_day}  ${low_price:,.0f}")
    if bear_low_pct:
        print(f"  Bear market low:   {bear_low_pct:+.1f}%  day {bear_low_day}  ${bear_low_price:,.0f}")
        print(f"  Support held:      {'✅ YES' if held else '❌ NO'}")
        if c['exclude']:
            print(f"  Note: Excluded from pattern — Mt. Gox black swan")
    print()

    # ── PLOT ──────────────────────────────────────────────

    # Shade accumulation zone
    ax.axvspan(0, 200, alpha=0.12, color='white', zorder=1)

    # Shade post-top bear zone differently
    if top_day < PLOT_DAYS:
        ax.axvspan(top_day, min(PLOT_DAYS,
                   (TODAY - h).days if not c['confirmed'] else PLOT_DAYS),
                   alpha=0.06, color='red', zorder=1)

    # Price line — grey if excluded
    lw = 2.5 if not c['exclude'] else 1.5
    ax.plot(chunk['days'], chunk['pct'],
            color=c['color'], linewidth=lw,
            zorder=3, alpha=0.9 if not c['exclude'] else 0.5)
    ax.fill_between(chunk['days'], chunk['pct'], -115,
                    alpha=0.05, color=c['color'])

    # ── Accumulation zone label ───────────────────────────
    ax.text(100, low_pct + 12,
            'Accumulation\nZone', ha='center',
            fontsize=8, color='white', alpha=0.55)

    # ── THE KEY LINE — accumulation low extended right ────
    if not c['exclude']:
        ax.axhline(low_pct, color='yellow', linewidth=2.0,
                   linestyle='--', alpha=0.95, zorder=5,
                   label=f'Accum. low: {low_pct:.1f}%  '
                         f'(day {low_day}, ${low_price:,.0f})')
    else:
        ax.axhline(low_pct, color=SUPPORT_COLOR, linewidth=1.2,
                   
                   linestyle='--', alpha=0.5, zorder=5,
                   label=f'Accum. low: {low_pct:.1f}%  (excluded)')

    # Mark the accumulation low point
    ax.scatter(low_day, low_pct, color='yellow' if not c['exclude'] else '#888888',
               s=140, zorder=6, marker='o',
               edgecolors='white', linewidths=0.8)
    ax.annotate(
        f"Accum. low\nDay {low_day}\n${low_price:,.0f}\n({low_pct:.1f}%)",
        xy=(low_day, low_pct),
        xytext=(low_day + 80, low_pct + 12),
        fontsize=8, color='yellow' if not c['exclude'] else '#aaaaaa',
        arrowprops=dict(arrowstyle='->', color='yellow'
                        if not c['exclude'] else '#aaaaaa', lw=1.2),
    )

    # ── Cycle top marker ──────────────────────────────────
    ax.axvline(top_day, color=c['color'], linewidth=1.0,
               linestyle=':', alpha=0.5, zorder=2)
    ax.scatter(top_day, 0, color='white', s=100,
               zorder=6, marker='^')
    ax.text(top_day + 12, 3.5,
            f"▲ Top\nDay {top_day}",
            fontsize=8, color='white', alpha=0.75)

    # ── Bear market low marker ────────────────────────────
    if bear_low_pct is not None and len(post_top) > 0:
        marker_color = 'white' if c['confirmed'] else c['color']
        ax.scatter(bear_low_day, bear_low_pct,
                   color=marker_color, s=120,
                   zorder=6, marker='v')
        label = '▼ Bottom' if c['confirmed'] else '▼ Current low'
        ax.text(bear_low_day + 12, bear_low_pct - 4,
                f"{label}\nDay {bear_low_day}\n"
                f"${bear_low_price:,.0f}\n({bear_low_pct:.1f}%)",
                fontsize=8, color=c['color'], alpha=0.9)

    # ── Today marker for cycle 4 ──────────────────────────
    if not c['confirmed']:
        today_day  = (TODAY - h).days
        today_pct  = ((64074 - c['top_price']) / c['top_price']) * 100
        ax.scatter(today_day, today_pct,
                   color=c['color'], s=280, zorder=7, marker='*')
        ax.text(today_day + 15, today_pct + 5,
                f"★ Today  Day {today_day}\n"
                f"$64,074  ({today_pct:.1f}%)",
                fontsize=9, color=c['color'], fontweight='bold')

        # Distance to support
        gap_pct = ((64074 - low_price) / low_price) * 100
        ax.text(0.03, 0.08,
                f"Support: ${low_price:,.0f}\n"
                f"Current: $64,074\n"
                f"Gap: +{gap_pct:.1f}% above support",
                transform=ax.transAxes,
                fontsize=9, color='yellow', fontweight='bold',
                bbox=dict(boxstyle='round',
                          facecolor='black', alpha=0.65))

    # ── Status badge ──────────────────────────────────────
    if held is not None:
        if c['exclude']:
            badge_text  = '⚠️  Excluded — Mt. Gox black swan'
            badge_color = '#888888'
        elif held:
            badge_text  = '✅  Accumulation low held as support'
            badge_color = '#00ff88'
        else:
            badge_text  = '❌  Accumulation low broken'
            badge_color = '#ff4444'

        ax.text(0.97, 0.05, badge_text,
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=9, fontweight='bold', color=badge_color,
                bbox=dict(boxstyle='round',
                          facecolor='black', alpha=0.65))

    # ── Reference lines ───────────────────────────────────
    for pct, col, lbl in [
        (-25, '#cccccc', '−25%'),
        (-50, '#ff8c00', '−50%'),
        (-75, '#e74c3c', '−75%'),
    ]:
        ax.axhline(pct, color=col, linewidth=0.6,
                   linestyle=':', alpha=0.35)
        ax.text(PLOT_DAYS - 15, pct + 1.5,
                lbl, ha='right', fontsize=7,
                color=col, alpha=0.45)

    # ── Axis formatting ───────────────────────────────────
    ax.set_facecolor(BACKGROUND)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(linestyle='--', alpha=0.18, color=GRID_COLOR)   
    ax.set_title(c['name'], fontsize=11,
                 fontweight='bold', color=c['color'])
    ax.set_xlabel('Days from Halving', fontsize=9)
    ax.set_ylabel('% from Cycle Top', fontsize=9)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x:+.0f}%'))
    ax.set_xlim(-20, PLOT_DAYS)
    ax.set_ylim(-115, 20)
   
    ax.legend(fontsize=8.5, loc='upper right')
    ax.axhline(0, color='white', linewidth=0.8,
               linestyle='--', alpha=0.25)

# ── Footer note ───────────────────────────────────────────
fig.text(
    0.5, 0.01,
    'White shading = accumulation zone (first 200 days after halving)  |  '
    'Red shading = post-top bear market  |  '
    'Yellow dashed = accumulation low support line  |  '
    'Past patterns do not guarantee future results',
    ha='center', fontsize=8.5, style='italic', alpha=0.6
)

plt.tight_layout(rect=[0, 0.03, 1, 1])
path = os.path.join(OUTPUT_DIR, 'btc_accumulation_support.png')
fig.savefig(path, dpi=150)
plt.close()
print(f"Saved: {path}")
print("\nKey level to watch for Cycle 4:")
print(f"  Accumulation support: $53,923")
print(f"  Current price:        $64,074")
print(f"  Buffer above support: ${64074 - 53923:,.0f} ({((64074-53923)/53923)*100:.1f}%)")
print(f"\nAll done.")