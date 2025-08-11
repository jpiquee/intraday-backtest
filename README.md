# Intraday Backtest — Cloud & Mobile (GitHub Actions + Pages)

Ce dépôt permet de lancer des **backtests intraday automatiques** (5 minutes) sur **QQQ** (ETF Nasdaq) et **BTC-USD**, 
puis de publier un **rapport HTML** accessible depuis ton mobile via **GitHub Pages**.

## Vue d'ensemble
- **Planification**: toutes les heures (cron) + lancement manuel.
- **Données**: téléchargées via `yfinance` (60 derniers jours en 5-min).
- **Stratégies**: Mean Reversion (RSI+Bollinger) et Breakout (Donchian).
- **Rapport**: `docs/index.html` (tableau de métriques + graphiques d'équité).

## Mise en place (une seule fois)

1. **Créer un dépôt GitHub** (vide) et y pousser ces fichiers.
2. **Activer GitHub Actions** (Settings → Actions → General → autoriser les workflows).
3. **Activer GitHub Pages**: Settings → Pages → *Deploy from a branch* → Branch: `main` → Folder: `/docs` → Save.
4. **Lancer le workflow une première fois**: onglet *Actions* → *Backtest* → *Run workflow* (bouton vert).
5. Au bout de ~2–3 minutes, ouvre l’URL GitHub Pages affichée dans *Pages* (ex: `https://<ton_user>.github.io/<repo>/`).

## Lancer localement (optionnel)
```bash
python -m venv .venv && . .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python auto_backtest.py
# Ouvre docs/index.html dans ton navigateur
```

## Modifier les actifs / paramètres
- Dans `auto_backtest.py`, change `TICKERS = ["QQQ", "BTC-USD"]`.
- Paramètres clés du backtester accessibles dans `run_backtest()` (equity initiale, risk_fraction, etc.).