import pandas as pd
import numpy as np

class IntradayMeanReversion:
    """
    ~3-4 trades/day typiquement.
    Long: poke sous bande Bollinger basse + RSI<30 puis retour au-dessus.
    Short: miroir sur bande haute + RSI>70.
    Exit: croisement de la bande médiane (ou stop/target moteur).
    """
    def __init__(self, min_time="10:00", max_time="15:30", cooldown_bars=6):
        self.min_time = pd.to_datetime(min_time).time()
        self.max_time = pd.to_datetime(max_time).time()
        self.cooldown_bars = cooldown_bars
        self.cooldown = 0

    def _tod_ok(self, ts):
        t = ts.time()
        return self.min_time <= t <= self.max_time

    def signal(self, i: int, df: pd.DataFrame):
        if self.cooldown > 0:
            self.cooldown -= 1
        row = df.iloc[i]
        prev = df.iloc[i-1]
        if not self._tod_ok(row['datetime']):
            return None
        if prev['close'] < prev['bb_mid'] and row['close'] >= row['bb_mid']:
            return "exit"
        if prev['close'] > prev['bb_mid'] and row['close'] <= row['bb_mid']:
            return "exit"
        if self.cooldown == 0:
            if prev['close'] < prev['bb_dn'] and prev['rsi'] < 30 and row['close'] > row['bb_dn']:
                self.cooldown = self.cooldown_bars
                return "enter_long"
            if prev['close'] > prev['bb_up'] and prev['rsi'] > 70 and row['close'] < row['bb_up']:
                self.cooldown = self.cooldown_bars
                return "enter_short"
        return None

class IntradayBreakout:
    """
    Donchian breakout avec filtre horaire.
    Enter long: dépassement du don_high. Enter short: cassure du don_low.
    Exit: cassure opposée (ou stop/target moteur). Cooldown pour limiter le nb de trades.
    """
    def __init__(self, min_time="09:45", max_time="15:50", cooldown_bars=8):
        self.min_time = pd.to_datetime(min_time).time()
        self.max_time = pd.to_datetime(max_time).time()
        self.cooldown_bars = cooldown_bars
        self.cooldown = 0
        self.position = 0

    def _tod_ok(self, ts):
        t = ts.time()
        return self.min_time <= t <= self.max_time

    def signal(self, i: int, df: pd.DataFrame):
        if self.cooldown > 0:
            self.cooldown -= 1
        row = df.iloc[i]
        if not self._tod_ok(row['datetime']):
            return None
        long_break = row['high'] > row['don_high']
        short_break = row['low'] < row['don_low']
        if self.position == 1 and short_break:
            self.position = 0
            self.cooldown = self.cooldown_bars
            return "exit"
        if self.position == -1 and long_break:
            self.position = 0
            self.cooldown = self.cooldown_bars
            return "exit"
        if self.cooldown == 0:
            if long_break and df['atr'].iloc[i] > 0:
                self.position = 1
                self.cooldown = self.cooldown_bars
                return "enter_long"
            if short_break and df['atr'].iloc[i] > 0:
                self.position = -1
                self.cooldown = self.cooldown_bars
                return "enter_short"
        return None