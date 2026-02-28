#!/usr/bin/env python3
"""
Bitcoin On-Chain Dashboard
Fetches fresh data and generates an interactive HTML dashboard.
Run daily via cron.
"""

import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Cycle markers ──────────────────────────────────────────────────────────────
TOPS = {
    '2013 Top': '2013-12-04',
    '2017 Top': '2017-12-17',
    '2021 Top': '2021-11-10',
}
BOTTOMS = {
    '2015 Bottom': '2015-01-14',
    '2018 Bottom': '2018-12-15',
    '2022 Bottom': '2022-11-21',
}


# ── 1. FETCH DATA ──────────────────────────────────────────────────────────────
def fetch_data():
    """Pull all APIs and save to JSON files. Returns latest BTC price or None."""
    endpoints = [
        ('https://bitcoin-data.com/api/v1/mvrv-zscore',            'mvrv_zscore.json'),
        ('https://bitcoin-data.com/api/v1/realized-price',         'realized_price.json'),
        ('https://bitcoin-data.com/v1/long-term-hodler-supply-btc','lth_supply.json'),
        ('https://bitcoin-data.com/v1/exchange-netflow-btc',       'exchange_netflow_btc.json'),
    ]

    for url, filename in endpoints:
        path = os.path.join(OUTPUT_DIR, filename)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(path, 'w') as f:
                json.dump(resp.json(), f)
            print(f'  ✓ {filename}')
        except Exception as e:
            print(f'  ✗ {filename}: {e} (using existing file if available)')
        time.sleep(1)

    # CoinGecko for current price (no API key needed)
    try:
        resp = requests.get(
            'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd',
            timeout=10,
        )
        resp.raise_for_status()
        price = resp.json()['bitcoin']['usd']
        print(f'  ✓ CoinGecko: ${price:,.0f}')
        return price
    except Exception as e:
        print(f'  ✗ CoinGecko: {e}')
        return None


# ── 2. LOAD DATA ───────────────────────────────────────────────────────────────
def load_data(current_price=None):
    """Load and merge all JSON files into one DataFrame."""

    def read_json(filename):
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path) as f:
            return pd.DataFrame(json.load(f))

    # MVRV Z-Score  (fields: d, unixTs, mvrvZscore — all strings)
    mvrv = read_json('mvrv_zscore.json')
    mvrv['date'] = pd.to_datetime(mvrv['d'])
    mvrv['mvrv_zscore'] = pd.to_numeric(mvrv['mvrvZscore'], errors='coerce')
    mvrv = mvrv[['date', 'mvrv_zscore']]

    # Realized Price  (fields: theDay, unixTs, realizedPrice — all strings)
    rp = read_json('realized_price.json')
    rp['date'] = pd.to_datetime(rp['theDay'])
    rp['realized_price'] = pd.to_numeric(rp['realizedPrice'], errors='coerce')
    rp = rp[['date', 'realized_price']]

    # LTH Supply  (fields: d, unixTs int, longTermHodlerSupplyBtc float)
    lth = read_json('lth_supply.json')
    lth['date'] = pd.to_datetime(lth['d'])
    lth['lth_supply'] = pd.to_numeric(lth['longTermHodlerSupplyBtc'], errors='coerce')
    lth = lth[lth['lth_supply'] > 0][['date', 'lth_supply']]  # positive values only

    # Exchange Netflow  (fields: d, unixTs int, exchangeNetflowBtc float)
    nf = read_json('exchange_netflow_btc.json')
    nf['date'] = pd.to_datetime(nf['d'])
    nf['netflow'] = pd.to_numeric(nf['exchangeNetflowBtc'], errors='coerce')
    nf = nf[['date', 'netflow']]

    # Merge all on date
    df = mvrv.merge(rp, on='date', how='outer')
    df = df.merge(lth, on='date', how='outer')
    df = df.merge(nf, on='date', how='outer')
    df = df.sort_values('date').reset_index(drop=True)

    # BTC market price from CSV (used as base + CoinGecko fallback)
    csv_path = os.path.join(OUTPUT_DIR, 'btc-usd-max.csv')
    if os.path.exists(csv_path):
        btc = pd.read_csv(csv_path)
        btc['date'] = pd.to_datetime(btc['snapped_at']).dt.tz_localize(None).dt.normalize()
        btc = btc.rename(columns={'price': 'mkt_price'})[['date', 'mkt_price']]
        df = df.merge(btc, on='date', how='left')
    else:
        df['mkt_price'] = np.nan

    # Inject today's CoinGecko price
    if current_price:
        today = pd.Timestamp(datetime.utcnow().date())
        mask = df['date'] == today
        if mask.any():
            df.loc[mask, 'mkt_price'] = float(current_price)
        else:
            new_row = pd.DataFrame({'date': [today], 'mkt_price': [float(current_price)]})
            df = pd.concat([df, new_row], ignore_index=True).sort_values('date').reset_index(drop=True)

    # Filter to 2017-present for chart readability
    df = df[df['date'] >= '2017-01-01'].copy()

    # Derived columns
    df['realized_gap_pct'] = (df['mkt_price'] / df['realized_price'] - 1) * 100
    df['netflow_30d_ma']   = df['netflow'].rolling(30, min_periods=1).mean()
    df['lth_supply_m']     = df['lth_supply'] / 1e6

    print(f'  {len(df)} rows, {df["date"].min().date()} → {df["date"].max().date()}')
    return df


# ── 3. BUILD DASHBOARD ─────────────────────────────────────────────────────────
def build_dashboard(df):
    """Generate self-contained dark-theme HTML with 4-panel Plotly chart."""

    # ── Current values for stat cards ─────────────────────────────────────────
    latest_mvrv = df.dropna(subset=['mvrv_zscore']).iloc[-1]
    latest_mkt  = df.dropna(subset=['mkt_price']).iloc[-1]
    latest_rp   = df.dropna(subset=['realized_price']).iloc[-1]
    latest_lth  = df.dropna(subset=['lth_supply']).iloc[-1]
    latest_nf   = df.dropna(subset=['netflow_30d_ma']).iloc[-1]

    current_mvrv    = float(latest_mvrv['mvrv_zscore'])
    current_mkt     = float(latest_mkt['mkt_price'])
    current_rp      = float(latest_rp['realized_price'])
    current_gap     = (current_mkt / current_rp - 1) * 100
    current_lth     = float(latest_lth['lth_supply'])
    current_lth_m   = current_lth / 1e6
    current_lth_pct = current_lth / 21_000_000 * 100
    current_nf_ma   = float(latest_nf['netflow_30d_ma'])

    def mvrv_zone(v):
        if v < 0:   return 'EXTREME FEAR', '#00ff88'
        if v < 3.7: return 'FAIR VALUE',   '#ffd700'
        if v < 7:   return 'CAUTION',      '#ff8c00'
        return             'SELL ZONE',    '#ff4444'

    zone_label, zone_color = mvrv_zone(current_mvrv)
    updated_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')

    # ── Subplots ───────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[1, 1.2, 1, 1],
        vertical_spacing=0.03,
        subplot_titles=[
            'MVRV Z-Score',
            'Market Price vs Realized Price (Log Scale)',
            'Long-Term Holder Supply (millions BTC)',
            'Exchange Netflow — 30-Day Moving Average (BTC/day)',
        ],
    )

    # ── Panel 1: MVRV Z-Score ──────────────────────────────────────────────────
    # Shaded zone bands
    fig.add_hrect(y0=-10, y1=0,   fillcolor='rgba(0,255,136,0.06)', line_width=0, row=1, col=1)
    fig.add_hrect(y0=0,   y1=3.7, fillcolor='rgba(255,215,0,0.04)', line_width=0, row=1, col=1)
    fig.add_hrect(y0=3.7, y1=7,   fillcolor='rgba(255,140,0,0.07)', line_width=0, row=1, col=1)
    fig.add_hrect(y0=7,   y1=20,  fillcolor='rgba(255,68,68,0.09)', line_width=0, row=1, col=1)

    # Reference lines at zone boundaries
    for y_val, color in [(0, '#00ff88'), (3.7, '#ffd700'), (7, '#ff4444')]:
        fig.add_hline(y=y_val, line_dash='dot', line_color=color,
                      line_width=1, opacity=0.5, row=1, col=1)

    # Colored MVRV line — one trace per zone (NaN out-of-zone, connectgaps=False)
    mvrv_df = df.dropna(subset=['mvrv_zscore']).copy()
    zone_defs = [
        ('#00ff88', mvrv_df['mvrv_zscore'] < 0),
        ('#ffd700', (mvrv_df['mvrv_zscore'] >= 0)   & (mvrv_df['mvrv_zscore'] < 3.7)),
        ('#ff8c00', (mvrv_df['mvrv_zscore'] >= 3.7) & (mvrv_df['mvrv_zscore'] < 7)),
        ('#ff4444', mvrv_df['mvrv_zscore'] >= 7),
    ]
    for color, mask in zone_defs:
        fig.add_trace(
            go.Scatter(
                x=mvrv_df['date'],
                y=mvrv_df['mvrv_zscore'].where(mask),
                mode='lines',
                line=dict(color=color, width=1.5),
                connectgaps=False,
                showlegend=False,
                hovertemplate='%{x|%Y-%m-%d}: %{y:.2f}<extra>MVRV Z-Score</extra>',
            ),
            row=1, col=1,
        )

    # Cycle top (red) and bottom (green) vertical markers
    for date in TOPS.values():
        fig.add_vline(x=date, line_color='rgba(255,80,80,0.45)',
                      line_dash='dash', line_width=1, row=1, col=1)
    for date in BOTTOMS.values():
        fig.add_vline(x=date, line_color='rgba(0,255,136,0.45)',
                      line_dash='dash', line_width=1, row=1, col=1)

    # YOU ARE HERE annotation
    fig.add_annotation(
        x=latest_mvrv['date'], y=current_mvrv,
        text=f'  YOU ARE HERE: {current_mvrv:.2f}',
        showarrow=True, arrowhead=2, arrowwidth=1.5,
        arrowcolor=zone_color, ax=70, ay=-35,
        font=dict(color=zone_color, size=11),
        bgcolor='rgba(0,0,0,0.75)',
        bordercolor=zone_color, borderwidth=1,
        row=1, col=1,
    )

    # ── Panel 2: Market Price vs Realized Price ────────────────────────────────
    price_df = df.dropna(subset=['mkt_price', 'realized_price']).copy()

    # Green/red fill between the two lines — split at crossover points
    if len(price_df) > 1:
        above_vals = (price_df['mkt_price'] >= price_df['realized_price']).values
        n = len(above_vals)
        segments = []
        seg_start = 0
        cur_above = above_vals[0]
        for i in range(1, n):
            if above_vals[i] != cur_above:
                segments.append((price_df.iloc[seg_start:i], cur_above))
                seg_start = i
                cur_above = above_vals[i]
        segments.append((price_df.iloc[seg_start:], cur_above))

        for seg, is_above in segments:
            if len(seg) < 2:
                continue
            fill_color = 'rgba(0,220,100,0.12)' if is_above else 'rgba(255,80,80,0.12)'
            y_bot = seg['realized_price'] if is_above else seg['mkt_price']
            y_top = seg['mkt_price']      if is_above else seg['realized_price']
            # Lower boundary (invisible) then upper with fill='tonexty'
            fig.add_trace(
                go.Scatter(x=seg['date'], y=y_bot, mode='lines',
                           line=dict(color='rgba(0,0,0,0)', width=0),
                           showlegend=False, hoverinfo='skip'),
                row=2, col=1,
            )
            fig.add_trace(
                go.Scatter(x=seg['date'], y=y_top, mode='lines',
                           line=dict(color='rgba(0,0,0,0)', width=0),
                           fill='tonexty', fillcolor=fill_color,
                           showlegend=False, hoverinfo='skip'),
                row=2, col=1,
            )

    # BTC price (orange) + Realized price (cyan dashed)
    fig.add_trace(
        go.Scatter(
            x=price_df['date'], y=price_df['mkt_price'],
            mode='lines', name='BTC Price',
            line=dict(color='#f7931a', width=2),
            hovertemplate='%{x|%Y-%m-%d}: $%{y:,.0f}<extra>BTC Price</extra>',
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=price_df['date'], y=price_df['realized_price'],
            mode='lines', name='Realized Price',
            line=dict(color='#00e5ff', width=1.5, dash='dash'),
            hovertemplate='%{x|%Y-%m-%d}: $%{y:,.0f}<extra>Realized Price</extra>',
        ),
        row=2, col=1,
    )

    # Cycle bottom markers
    for date in BOTTOMS.values():
        fig.add_vline(x=date, line_color='rgba(0,255,136,0.45)',
                      line_dash='dash', line_width=1, row=2, col=1)

    # Stats annotation box (top-right of panel 2 via domain coords)
    fig.add_annotation(
        x=0.99, y=0.97,
        xref='x2 domain', yref='y2 domain',
        xanchor='right', yanchor='top',
        text=(f'BTC: <b>${current_mkt:,.0f}</b><br>'
              f'Realized: ${current_rp:,.0f}<br>'
              f'Gap: <b>{current_gap:+.1f}%</b>'),
        showarrow=False,
        font=dict(color='white', size=11),
        bgcolor='rgba(0,0,0,0.75)',
        bordercolor='rgba(255,255,255,0.25)', borderwidth=1,
        align='right',
    )

    # Log y-axis for panel 2
    fig.update_yaxes(type='log', row=2, col=1)

    # ── Panel 3: LTH Supply ────────────────────────────────────────────────────
    lth_df = df.dropna(subset=['lth_supply_m']).copy()
    fig.add_trace(
        go.Scatter(
            x=lth_df['date'], y=lth_df['lth_supply_m'],
            mode='lines', name='LTH Supply',
            line=dict(color='#9b59b6', width=1.5),
            fill='tozeroy', fillcolor='rgba(155,89,182,0.18)',
            hovertemplate='%{x|%Y-%m-%d}: %{y:.2f}M BTC<extra>LTH Supply</extra>',
        ),
        row=3, col=1,
    )

    for date in TOPS.values():
        fig.add_vline(x=date, line_color='rgba(255,80,80,0.4)',
                      line_dash='dash', line_width=1, row=3, col=1)
    for date in BOTTOMS.values():
        fig.add_vline(x=date, line_color='rgba(0,255,136,0.4)',
                      line_dash='dash', line_width=1, row=3, col=1)

    # ── Panel 4: Exchange Netflow 30d MA ───────────────────────────────────────
    nf_df = df.dropna(subset=['netflow_30d_ma']).copy()
    bar_colors = [
        'rgba(255,80,80,0.65)' if v >= 0 else 'rgba(0,220,100,0.65)'
        for v in nf_df['netflow_30d_ma']
    ]
    fig.add_trace(
        go.Bar(
            x=nf_df['date'], y=nf_df['netflow_30d_ma'],
            name='Netflow 30d MA',
            marker_color=bar_colors,
            hovertemplate='%{x|%Y-%m-%d}: %{y:+,.0f} BTC/day<extra>Netflow 30d MA</extra>',
        ),
        row=4, col=1,
    )
    fig.add_hline(y=0, line_color='rgba(255,255,255,0.25)', line_width=1, row=4, col=1)

    nf_arrow_color = '#ff5050' if current_nf_ma >= 0 else '#00dc64'
    fig.add_annotation(
        x=latest_nf['date'], y=current_nf_ma,
        text=f'  {current_nf_ma:+,.0f} BTC/day',
        showarrow=True, arrowhead=2, arrowwidth=1.5,
        arrowcolor=nf_arrow_color, ax=70, ay=-35,
        font=dict(color=nf_arrow_color, size=11),
        bgcolor='rgba(0,0,0,0.75)',
        bordercolor=nf_arrow_color, borderwidth=1,
        row=4, col=1,
    )

    # ── Global layout ──────────────────────────────────────────────────────────
    axis_style = dict(
        gridcolor='rgba(255,255,255,0.05)',
        zerolinecolor='rgba(255,255,255,0.1)',
        linecolor='rgba(255,255,255,0.15)',
        tickfont=dict(size=10),
    )

    fig.update_layout(
        height=1600,
        paper_bgcolor='#0d0d0d',
        plot_bgcolor='#111111',
        font=dict(color='#cccccc', family='Courier New, monospace', size=11),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(20,20,20,0.8)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1,
            font=dict(size=10),
            x=0.01, y=0.99,
        ),
        hovermode='x unified',
        margin=dict(l=70, r=60, t=70, b=40),
        hoverlabel=dict(bgcolor='rgba(0,0,0,0.85)', font_size=11),
    )

    for i in range(1, 5):
        fig.update_xaxes(**axis_style, row=i, col=1)
        fig.update_yaxes(**axis_style, row=i, col=1)

    # Plotly CDN so the file works offline after first load
    chart_html = fig.to_html(
        include_plotlyjs='cdn',
        full_html=False,
        config={'responsive': True, 'displayModeBar': True},
    )

    # ── HTML template ──────────────────────────────────────────────────────────
    nf_sign   = '+' if current_nf_ma >= 0 else ''
    nf_color  = '#ff5050' if current_nf_ma >= 0 else '#00dc64'
    gap_color = '#00dc64' if current_gap >= 0 else '#ff5050'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bitcoin On-Chain Dashboard</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0d0d0d;
    color: #e0e0e0;
    font-family: 'Courier New', Courier, monospace;
    padding: 20px 24px;
    max-width: 1600px;
    margin: 0 auto;
  }}
  h1 {{
    color: #f7931a;
    font-size: 1.7em;
    letter-spacing: 1px;
    margin-bottom: 4px;
  }}
  .updated {{
    color: #555;
    font-size: 0.78em;
    margin-bottom: 18px;
  }}
  .stats {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 20px;
  }}
  .card {{
    background: #161616;
    border: 1px solid #252525;
    border-radius: 8px;
    padding: 12px 18px;
    flex: 1;
    min-width: 150px;
  }}
  .card-label {{
    font-size: 0.65em;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 5px;
  }}
  .card-value {{
    font-size: 1.35em;
    font-weight: bold;
    line-height: 1.1;
  }}
  .card-sub {{
    font-size: 0.73em;
    color: #777;
    margin-top: 3px;
  }}
  .chart-wrapper {{
    width: 100%;
    border: 1px solid #1e1e1e;
    border-radius: 8px;
    overflow: hidden;
  }}
  footer {{
    text-align: center;
    color: #3a3a3a;
    font-size: 0.72em;
    margin-top: 18px;
    padding-top: 12px;
    border-top: 1px solid #1a1a1a;
  }}
  footer a {{ color: #555; text-decoration: none; }}
  footer a:hover {{ color: #888; }}
</style>
</head>
<body>

<header>
  <h1>&#8383; Bitcoin On-Chain Dashboard</h1>
  <p class="updated">Last updated: {updated_str}</p>
  <div class="stats">
    <div class="card">
      <div class="card-label">BTC Price</div>
      <div class="card-value" style="color:#f7931a">${current_mkt:,.0f}</div>
    </div>
    <div class="card">
      <div class="card-label">MVRV Z-Score</div>
      <div class="card-value" style="color:{zone_color}">{current_mvrv:.2f}</div>
      <div class="card-sub" style="color:{zone_color}">{zone_label}</div>
    </div>
    <div class="card">
      <div class="card-label">vs Realized Price</div>
      <div class="card-value" style="color:{gap_color}">{current_gap:+.1f}%</div>
      <div class="card-sub">Realized: ${current_rp:,.0f}</div>
    </div>
    <div class="card">
      <div class="card-label">LTH Supply</div>
      <div class="card-value" style="color:#9b59b6">{current_lth_m:.2f}M BTC</div>
      <div class="card-sub">{current_lth_pct:.1f}% of 21M supply</div>
    </div>
    <div class="card">
      <div class="card-label">Netflow 30d MA</div>
      <div class="card-value" style="color:{nf_color}">{nf_sign}{current_nf_ma:,.0f}</div>
      <div class="card-sub">{'Sell pressure ↑' if current_nf_ma >= 0 else 'Accumulation ↓'}</div>
    </div>
  </div>
</header>

<div class="chart-wrapper">
{chart_html}
</div>

<footer>
  Auto-updated daily &nbsp;&bull;&nbsp;
  On-chain data: <a href="https://bitcoin-data.com" target="_blank">bitcoin-data.com</a>
  &nbsp;&bull;&nbsp;
  Price: <a href="https://coingecko.com" target="_blank">CoinGecko</a>
</footer>

</body>
</html>"""

    return html


# ── 4. SETUP CRON ──────────────────────────────────────────────────────────────
def setup_cron():
    """Print cron instructions (does not auto-install)."""
    print('\n' + '─' * 70)
    print('To auto-refresh daily at 8am, run  crontab -e  and add:')
    print()
    print('  0 8 * * * /opt/anaconda3/bin/python '
          '/Users/jeremyholt/data-analysis-projects/btc_dashboard.py '
          '>> /Users/jeremyholt/data-analysis-projects/dashboard.log 2>&1')
    print('─' * 70)


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Fetching data…')
    current_price = fetch_data()

    print('\nLoading and merging data…')
    df = load_data(current_price)

    print('\nBuilding dashboard…')
    html = build_dashboard(df)

    out_path = os.path.join(OUTPUT_DIR, 'btc_dashboard.html')
    with open(out_path, 'w') as f:
        f.write(html)
    print(f'  Dashboard saved → {out_path}')

    setup_cron()
