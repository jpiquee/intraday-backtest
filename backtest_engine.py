import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    direction: int  # +1 long, -1 short
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: Optional[float]
    reason: str

class ExecutionModel:
    def __init__(self, slippage_bps: float = 1.0, commission_per_trade: float = 0.5):
        self.slippage_bps = slippage_bps
        self.commission_per_trade = commission_per_trade

    def next_open_with_slippage(self, next_open: float, direction: int) -> float:
        slip = next_open * (self.slippage_bps / 10000.0)
        return next_open + slip * direction

    def commission(self) -> float:
        return self.commission_per_trade

class StopTargetEngine:
    @staticmethod
    def check_stop_target(bar_high: float, bar_low: float, stop: Optional[float], target: Optional[float], direction: int):
        if stop is None and target is None:
            return None
        if direction == +1:
            hit_stop = stop is not None and bar_low <= stop
            hit_target = target is not None and bar_high >= target
            if hit_stop and hit_target:
                return ('stop', stop)
            elif hit_stop:
                return ('stop', stop)
            elif hit_target:
                return ('target', target)
        else:
            hit_stop = stop is not None and bar_high >= stop
            hit_target = target is not None and bar_low <= target
            if hit_stop and hit_target:
                return ('stop', stop)
            elif hit_stop:
                return ('stop', stop)
            elif hit_target:
                return ('target', target)
        return None

class Backtester:
    def __init__(self, df: pd.DataFrame, execution: ExecutionModel, initial_equity: float = 1000.0,
                 risk_fraction: float = 0.01, max_leverage: float = 2.0, session_start: str = "09:35",
                 session_end: str = "16:00", atr_lookback: int = 20):
        self.df = df.copy()
        self.exec = execution
        self.equity = initial_equity
        self.initial_equity = initial_equity
        self.risk_fraction = risk_fraction
        self.max_leverage = max_leverage
        self.session_start = session_start
        self.session_end = session_end
        self.atr_lookback = atr_lookback

        self.position = 0
        self.size = 0.0
        self.entry_price = None
        self.entry_time = None
        self.stop = None
        self.target = None
        self.trades: List[Trade] = []
        self.equity_curve = []

        self._prepare_indicators()

    def _prepare_indicators(self):
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        tr1 = (high - low).abs()
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df['atr'] = tr.rolling(self.atr_lookback, min_periods=self.atr_lookback).mean()

        delta = close.diff()
        up = (delta.clip(lower=0)).rolling(14).mean()
        down = (-delta.clip(upper=0)).rolling(14).mean()
        rs = up / (down + 1e-12)
        self.df['rsi'] = 100 - (100 / (1 + rs))

        ma = close.rolling(20).mean()
        std = close.rolling(20).std(ddof=0)
        self.df['bb_mid'] = ma
        self.df['bb_up'] = ma + 2 * std
        self.df['bb_dn'] = ma - 2 * std

        self.df['don_high'] = high.rolling(20).max()
        self.df['don_low'] = low.rolling(20).min()

        if 'datetime' in self.df.columns:
            self.df = self.df.dropna().reset_index(drop=True)

    def _in_session(self, ts: pd.Timestamp) -> bool:
        t = ts.time()
        start = pd.to_datetime(self.session_start).time()
        end = pd.to_datetime(self.session_end).time()
        return start <= t <= end

    def _position_sizing(self, price: float, atr: float) -> float:
        if atr is None or np.isnan(atr) or atr == 0:
            return 0.0
        risk_capital = self.equity * self.risk_fraction
        units = risk_capital / atr
        max_units = (self.equity * self.max_leverage) / price
        return float(min(units, max_units))

    def _enter(self, i: int, direction: int, reason: str):
        next_open = self.df['open'].iloc[i]
        price = self.exec.next_open_with_slippage(next_open, direction)
        atr = self.df['atr'].iloc[i]
        units = self._position_sizing(price, atr)
        if units <= 0:
            return
        self.position = direction
        self.size = units
        self.entry_price = price
        self.entry_time = self.df['datetime'].iloc[i]
        if not np.isnan(atr) and atr > 0:
            if direction == +1:
                self.stop = price - atr
                self.target = price + 1.5 * atr
            else:
                self.stop = price + atr
                self.target = price - 1.5 * atr
        else:
            self.stop = None
            self.target = None
        self.equity -= self.exec.commission()

    def _exit(self, i: int, reason: str, price_override: float = None):
        if self.position == 0:
            return
        next_open = self.df['open'].iloc[i]
        price = price_override if price_override is not None else self.exec.next_open_with_slippage(next_open, -self.position)
        pnl = (price - self.entry_price) * self.size * self.position
        self.equity += pnl
        self.equity -= self.exec.commission()
        self.trades.append(Trade(
            entry_time=self.entry_time,
            exit_time=self.df['datetime'].iloc[i],
            direction=self.position,
            entry_price=self.entry_price,
            exit_price=price,
            size=self.size,
            pnl=pnl,
            reason=reason
        ))
        self.position = 0
        self.size = 0.0
        self.entry_price = None
        self.entry_time = None
        self.stop = None
        self.target = None

    def run(self, strategy) -> Dict[str, Any]:
        for i in range(1, len(self.df)-1):
            row = self.df.iloc[i]
            ts = row['datetime']
            if not self._in_session(ts):
                if self.position != 0:
                    self._exit(i, reason="session_end")
                self.equity_curve.append({'datetime': ts, 'equity': self.equity})
                continue

            if self.position != 0:
                fill = StopTargetEngine.check_stop_target(row['high'], row['low'], self.stop, self.target, self.position)
                if fill is not None:
                    kind, level = fill
                    self._exit(i, reason=kind, price_override=level)

            sig = strategy.signal(i, self.df)
            if sig == "enter_long" and self.position == 0:
                self._enter(i, +1, reason="enter_long")
            elif sig == "enter_short" and self.position == 0:
                self._enter(i, -1, reason="enter_short")
            elif sig == "exit" and self.position != 0:
                self._exit(i, reason="exit")

            self.equity_curve.append({'datetime': ts, 'equity': self.equity})

        self._exit(len(self.df)-1, reason="final_close")
        res = pd.DataFrame(self.equity_curve)
        return {
            'equity_curve': res,
            'trades': pd.DataFrame([t.__dict__ for t in self.trades]),
            'final_equity': self.equity,
            'return_pct': (self.equity / self.initial_equity - 1) * 100.0
        }