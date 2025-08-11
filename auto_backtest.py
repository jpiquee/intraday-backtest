import os, time, random, os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import requests

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

ALPHAVANTAGE_KEY = os.environ.get("ALPHAVANTAGE_KEY", "").strip()

def dl_5m_yf(ticker: str, tries: int = 6, wait_base: int = 10):
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
        print(f"[INFO] yfinance retry {attempt}/{tries} for {ticker} after {sleep_s}s due to: {last_err}")
        time.sleep(sleep_s)
    print(f"[WARN] yfinance failed for {ticker}: {last_err}")
    return None

def dl_5m_av_equity(symbol: str, api_key: str):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": "5min",
        "outputsize": "full",
        "datatype": "json",
        "apikey": api_key,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    key = "Time Series (5min)"
    if key not in j:
        raise RuntimeError(f"Alpha Vantage equity error: {j.get('Note') or j.get('Error Message') or 'unknown'}")
    ts = j[key]
    rows = []
    for ts_str, v in ts.items():
        rows.append([ts_str, float(v["1. open"]), float(v["2. high"]), float(v["3. low"]), float(v["4. close"]), float(v["5. volume"])])
    df = pd.DataFrame(rows, columns=["datetime","open","high","low","close","volume"])
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")  # AV is in US/Eastern for equities
    df = df.sort_values("datetime").reset_index(drop=True)
    # Filter RTH 09:35-16:00
    df = df[(df["datetime"].dt.time >= pd.to_datetime("09:35").time()) &
            (df["datetime"].dt.time <= pd.to_datetime("16:00").time())].reset_index(drop=True)
    return df

def dl_5m_av_crypto(symbol: str, market: str, api_key: str):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "CRYPTO_INTRADAY",
        "symbol": symbol,
        "market": market,
        "interval": "5min",
        "outputsize": "full",
        "apikey": api_key,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    key = "Time Series Crypto (5min)"
    if key not in j:
        raise RuntimeError(f"Alpha Vantage crypto error: {j.get('Note') or j.get('Error Message') or 'unknown'}")
    ts = j[key]
    rows = []
    for ts_str, v in ts.items():
        rows.append([ts_str, float(v["1. open"]), float(v["2. high"]), float(v["3. low"]), float(v["4. close"]), float(v["5. volume"])])
    df = pd.DataFrame(rows, columns=["datetime","open","high","low","close","volume"])
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")  # AV crypto is UTC
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

def dl_5m(ticker: str):
    # Try Yahoo
    df = dl_5m_yf(ticker)
    if df is not None and not df.empty:
        return df
    # Fallback to Alpha Vantage if key present
    if ALPHAVANTAGE_KEY:
        try:
            if ticker.endswith("-USD"):  # crypto
                sym = ticker.split("-")[0]
                df = dl_5m_av_crypto(sym, "USD", ALPHAVANTAGE_KEY)
            else:
                df = dl_5m_av_equity(ticker, ALPHAVANTAGE_KEY)
            if df is not None and not df.empty:
                print(f"[INFO] Used Alpha Vantage for {ticker}")
                return df
        except Exception as e:
            print(f"[WARN] Alpha Vantage fallback failed for {ticker}: {e}")
    # else: rely on cache in main()
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
        <h2>Intraday Backtests (5-min) - auto report</h2>
        <p>Generated: {now}</p>
        <ul>{notes_html}</ul>
        <p>No results (data unavailable or rate limited). Try again later.</p>
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
<h2>Intraday Backtests (5-min) — auto report</h2>
<p>Generated: {now}</p>
<ul>{notes_html}</ul>
<table>
<thead><tr><th>Ticker</th><th>Strategy</th><th>Return (%)</th><th>Max DD (%)</th><th>#Trades</th><th>Equity</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
<p>Data sources: Yahoo Finance via yfinance (primary), Alpha Vantage fallback (5min). Execution: next bar open, 1bps slippage, 0.5 commission.</p>
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
            time.sleep(10 + random.randint(0, 10))
        df = dl_5m(ticker)
        cache_path = os.path.join(DATA_DIR, f"{ticker}_5m.csv")
        used_cache = False
        if (df is None or df.empty) and os.path.exists(cache_path):
            df = pd.read_csv(cache_path, parse_dates=["datetime"])
            used_cache = True
            notes.append(f"{ticker}: loaded from cache (API rate limited).")
        if df is None or df.empty:
            notes.append(f"{ticker}: no data available (APIs failed and no cache).")
            continue
        df.to_csv(cache_path, index=False)
        if used_cache:
            notes.append(f"{ticker}: reusing cached data for backtest.")
        results = run_backtest(ticker, df)
        for strat_name, res in results.items():
            key = f"{ticker}__{strat_name}"
            summary[key] = res
            img_path = os.path.join(ASSET_DIR, f"{ticker}_{strat_name}_equity.png")
            plot_equity(res['equity_curve'], f"{ticker} — {strat_name}", img_path)
    build_report(summary, notes)
    print("Done. Report at docs/index.html")

if __name__ == "__main__":
    main()
