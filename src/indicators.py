"""Technical indicators from first principles. No TA-Lib. Reused from Task 1."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    out = prices.copy()
    close = out["Close"]

    out["sma_50"] = sma(close, 50)
    out["sma_200"] = sma(close, 200)
    out["rsi_14"] = rsi_wilder(close, 14)
    macd_line, signal, hist = macd(close, 12, 26, 9)
    out["macd"] = macd_line
    out["macd_signal"] = signal
    out["macd_hist"] = hist
    upper, mid, lower = bollinger(close, 20, 2.0)
    out["bb_upper"] = upper
    out["bb_mid"] = mid
    out["bb_lower"] = lower
    out["momentum_bias"] = _momentum_bias(out)
    return out


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)
    return rsi.clip(0, 100)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    signal = line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    hist = line - signal
    return line, signal, hist


def bollinger(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, window)
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    return mid + num_std * std, mid, mid - num_std * std


def realized_vol_pct(close: pd.Series, window: int = 30) -> float:
    """Annualized realized volatility in percent from daily close-to-close returns."""
    returns = close.pct_change().dropna()
    if len(returns) < 2:
        raise ValueError("Not enough price history to compute volatility")
    use = min(int(window), len(returns))
    std = float(returns.tail(use).std(ddof=1))
    return std * (252 ** 0.5) * 100.0


def latest_snapshot(frame: pd.DataFrame) -> dict:
    row = frame.dropna(subset=["sma_50", "rsi_14", "macd"]).iloc[-1]
    has_sma200 = "sma_200" in frame.columns and pd.notna(row.get("sma_200"))

    snapshot = {
        "date": str(row.name.date()) if hasattr(row.name, "date") else str(row.name),
        "close": float(row["Close"]),
        "sma_50": float(row["sma_50"]),
        "sma_200": float(row["sma_200"]) if has_sma200 else None,
        "sma_cross": (
            "bullish"
            if has_sma200 and row["sma_50"] > row["sma_200"]
            else ("bearish" if has_sma200 else "unknown")
        ),
        "rsi_14": float(row["rsi_14"]),
        "macd": float(row["macd"]),
        "macd_signal": float(row["macd_signal"]),
        "macd_hist": float(row["macd_hist"]),
        "bb_upper": float(row["bb_upper"]),
        "bb_mid": float(row["bb_mid"]),
        "bb_lower": float(row["bb_lower"]),
        "bb_position": _bb_position(float(row["Close"]), float(row["bb_lower"]), float(row["bb_upper"])),
        "momentum_bias": str(row["momentum_bias"]),
    }
    return snapshot


def _bb_position(close: float, lower: float, upper: float) -> str:
    width = upper - lower
    if width <= 0:
        return "unknown"
    pct = (close - lower) / width
    if pct > 0.8:
        return "near_upper"
    if pct < 0.2:
        return "near_lower"
    return "mid_band"


def _momentum_bias(frame: pd.DataFrame) -> pd.Series:
    has_sma200 = frame["sma_200"].notna()
    bull = (
        (frame["sma_50"] > frame["sma_200"])
        & (frame["macd_hist"] > 0)
        & (frame["rsi_14"] >= 45)
        & (frame["rsi_14"] <= 70)
        & has_sma200
    )
    bear = (
        (frame["sma_50"] < frame["sma_200"])
        & (frame["macd_hist"] < 0)
        & (frame["rsi_14"] <= 55)
        & (frame["rsi_14"] >= 30)
        & has_sma200
    )
    out = pd.Series("mixed", index=frame.index, dtype="object")
    out = out.mask(bull, "bullish")
    out = out.mask(bear, "bearish")
    return out
