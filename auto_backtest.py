import os, io, math, json, pytz
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from backtest_engine import Backtester, ExecutionModel
from strategies import IntradayMeanReversion, IntradayBreakout

TICKERS = ["QQQ", "BTC-USD"]
OUT_ROOT = "outputs"
DOC_ROOT = "docs"
ASSET_DIR = os.path.join(DOC_ROOT, "assets")
DATA_DIR = "data"
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(DOC_ROOT, exist_ok=True)
os.makedirs(ASSET_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def dl_5m(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, interval="5m", period="60d", auto_adjust=False, progress=False)
    df = df.reset_index().rename(columns={
        "Datetime":"datetime", "Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"
    })
    # yfinance may return timezone-aware timestamps
    if 'datetime' in df.columns:
        # Convert to US/Eastern for equities to filter RTH; keep UTC for crypto
        if ticker != "BTC-USD":
            try:
                df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_convert("America/New_York")
            except Exception:
                df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            try:
                df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_convert("UTC")
            except Exception:
                df['datetime'] = pd.to_datetime(df['datetime'])
        # Drop tz info for engine (naive)
        df['datetime'] = df['datetime'].dt.tz_localize(None)
    df = df[['datetime','open','high','low','close','volume']].dropna().reset_index(drop=True)
    # Filter session for equities
    if ticker != "BTC-USD":
        df = df[(df['datetime'].dt.time >= pd.to_datetime("09:35").time()) & (df['datetime'].dt.time <= pd.to_datetime("16:00").time())].reset_index(drop=True)
    return df

def run_backtest(ticker: str, df: pd.DataFrame):
    # Set session according to asset
    if ticker == "BTC-USD":
        session_start, session_end = "00:00", "23:59"
    else:
        session_start, session_end = "09:35", "16:00"

    exec_model = ExecutionModel(slippage_bps=1.0, commission_per_trade=0.5)
    # Use 1000 initial equity to mirror Julien's target capital
    common = dict(initial_equity=1000.0, risk_fraction=0.01, max_leverage=2.0,
                  session_start=session_start, session_end=session_end)

    results = {}
    for strat_name, strat in [
        ("meanrev", IntradayMeanReversion()),
        ("breakout", IntradayBreakout())
    ]:
        bt = Backtester(df, exec_model, **common)
        res = bt.run(strat)
        out_folder = os.path.join(OUT_ROOT, f"{ticker}_{strat_name}")
        os.makedirs(out_folder, exist_ok=True)
        res['equity_curve'].to_csv(os.path.join(out_folder, "equity.csv"), index=False)
        res['trades'].to_csv(os.path.join(out_folder, "trades.csv"), index=False)
        results[strat_name] = res
    return results

def max_drawdown(equity: pd.Series):
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    return dd.min()

def plot_equity(equity_df: pd.DataFrame, title: str, out_path: str):
    plt.figure()
    plt.plot(equity_df['datetime'], equity_df['equity'])
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def build_report(all_results: dict):
    # Build index.html
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    rows = []
    for key, v in all_results.items():
        ticker, strat = key.split("__")
        eq = v['equity_curve']
        ret = v['return_pct']
        mdd = max_drawdown(eq['equity'])
        trades = len(v['trades'])
        img = f"{ticker}_{strat}_equity.png"
        rows.append((ticker, strat, f"{ret:.2f}%", f"{mdd*100:.2f}%", trades, img))

    # Simple HTML
    table_rows = "\\n".join([
        f"<tr><td>{t}</td><td>{s}</td><td>{r}</td><td>{m}</td><td>{tr}</td><td><img src='assets/{img}' width='380'></td></tr>"
        for (t,s,r,m,tr,img) in rows
    ])

    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Intraday Backtests</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 16px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f5f5f5; }}
</style>
</head>
<body>
<h2>Intraday Backtests (5-min) — rapport auto</h2>
<p>Généré: {now}</p>
<table>
<thead><tr><th>Ticker</th><th>Stratégie</th><th>Rendement (%)</th><th>Max Drawdown (%)</th><th>#Trades</th><th>Équity</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
<p>Données: yfinance (60 jours). Stratégies: Mean Reversion (RSI+Bollinger), Breakout (Donchian). Exécution: next bar open, slippage 1 bps, commission 0.5.</p>
</body>
</html>
"""
    with open(os.path.join(DOC_ROOT, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

def main():
    summary = {}
    for ticker in TICKERS:
        df = dl_5m(ticker)
        df.to_csv(os.path.join(DATA_DIR, f"{ticker}_5m.csv"), index=False)
        results = run_backtest(ticker, df)
        for strat_name, res in results.items():
            key = f"{ticker}__{strat_name}"
            summary[key] = res
            # Plot equity
            img_path = os.path.join(ASSET_DIR, f"{ticker}_{strat_name}_equity.png")
            plot_equity(res['equity_curve'], f"{ticker} — {strat_name}", img_path)
    build_report(summary)
    print("Done. Report at docs/index.html")

if __name__ == "__main__":
    main()