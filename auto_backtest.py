import os, time, random
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

def dl_5m(ticker: str, tries: int = 8, wait_base: int = 10):
    """Download 60d of 5m bars with retry/backoff.
    Returns a DataFrame or None. If Yahoo is rate-limited, we'll fall back to cache later."""
    last_err = None
    for attempt in range(1, tries + 1):
        try:
            df = yf.download(
                ticker,
                interval="5m",
                period="60d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df is not None and not df.empty:
                df = df.reset_index()
                for cand in ["Datetime", "Date", "index"]:
                    if cand in df.columns:
                        df = df.rename(columns={cand: "datetime"})
                        break
                df = df.rename(columns={
                    "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Adj Close": "close", "Volume": "volume"
                })
                keep = [c for c in ["datetime","open","high","low","close","volume"] if c in df.columns]
                df = df[keep].dropna().reset_index(drop=True)
                if "datetime" in df.columns:
                    s = pd.to_datetime(df["datetime"], errors="coerce")
                    try:
                        s = s.dt.tz_localize(None)
                    except Exception:
                        pass
                    df["datetime"] = s
                if ticker != "BTC-USD" and "datetime" in df.columns:
                    df = df[(df["datetime"].dt.time >= pd.to_datetime("09:35").time()) &
                            (df["datetime"].dt.time <= pd.to_datetime("16:00").time())].reset_index(drop=True)
                if not df.empty and "datetime" in df.columns:
                    return df
                last_err = RuntimeError("Empty dataframe after cleaning")
            else:
                last_err = RuntimeError("Empty dataframe from yfinance")
        except Exception as e:
            last_err = e
        sleep_s = wait_base * attempt + random.randint(0, 8)
        print(f"[INFO] Retry {attempt}/{tries} for {ticker} after {sleep_s}s due to: {last_err}")
        time.sleep(sleep_s)
    print(f"[WARN] Failed to download {ticker}: {last_err}")
    return None

def run_backtest(ticker: str, df: pd.DataFrame):
    session_start, session_end = ("00:00","23:59") if ticker == "BTC-USD" else ("09:35","16:00")
    exec_model = ExecutionModel(slippage_bps=1.0, commission_per_trade=0.5)
    common = dict(initial_equity=1000.0, risk_fraction=0.01, max_leverage=2.0,
                  session_start=session_start, session_end=session_end)
    results = {}
    for strat_name, strat in [("meanrev", IntradayMeanReversion()), ("breakout", IntradayBreakout())]:
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

def build_report(all_results: dict, notes: list[str]):
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

    notes_html = "".join([f"<li>{n}</li>" for n in notes])

    if not rows:
        html = f"""<!doctype html><html><head><meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Intraday Backtests</title></head><body style="font-family:system-ui;margin:16px">
        <h2>Intraday Backtests (5-min) - rapport auto</h2>
        <p>Genere: {now}</p>
        <ul>{notes_html}</ul>
        <p>Aucun resultat disponible (donnees indisponibles ou rate limit). Relancez le workflow dans Actions.</p>
        </body></html>"""
        with open(os.path.join(DOC_ROOT, "index.html"), "w", encoding="utf-8") as f:
            f.write(html)
        return

    table_rows = "\n".join([
        f"<tr><td>{t}</td><td>{s}</td><td>{r}</td><td>{m}</td><td>{tr}</td><td><img src='assets/{img}' width='380'></td></tr>"
        for (t,s,r,m,tr,img) in rows
    ])
    html = f"""<!doctype html>
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
<h2>Intraday Backtests (5-min) - rapport auto</h2>
<p>Genere: {now}</p>
<ul>{notes_html}</ul>
<table>
<thead><tr><th>Ticker</th><th>Strategie</th><th>Rendement (%)</th><th>Max Drawdown (%)</th><th>#Trades</th><th>Equity</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
<p>Donnees: yfinance (60 jours). Strategies: Mean Reversion (RSI+Bollinger), Breakout (Donchian). Execution: next bar open, slippage 1 bps, commission 0.5.</p>
</body>
</html>
"""
    with open(os.path.join(DOC_ROOT, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

def main():
    notes = []
    summary = {}
    for idx, ticker in enumerate(TICKERS):
        if idx > 0:
            time.sleep(12 + random.randint(0, 10))
        df = dl_5m(ticker)
        cache_path = os.path.join(DATA_DIR, f"{ticker}_5m.csv")
        if (df is None or df.empty) and os.path.exists(cache_path):
            df = pd.read_csv(cache_path, parse_dates=["datetime"])
            notes.append(f"{ticker}: donnees recuperees depuis le cache local (rate limit Yahoo).")
        if df is None or df.empty:
            notes.append(f"{ticker}: aucune donnee disponible (echec download + pas de cache).")
            continue
        df.to_csv(cache_path, index=False)
        results = run_backtest(ticker, df)
        for strat_name, res in results.items():
            key = f"{ticker}__{strat_name}"
            summary[key] = res
            img_path = os.path.join(ASSET_DIR, f"{ticker}_{strat_name}_equity.png")
            plot_equity(res['equity_curve'], f"{ticker} - {strat_name}", img_path)
    build_report(summary, notes)
    print("Done. Report at docs/index.html")

if __name__ == "__main__":
    main()
