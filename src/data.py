"""Download and cache daily OHLCV bars. Reused from Task 1 Financial_AI."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from src.config import project_root


def fetch_prices(
    ticker: str,
    period: str = "1y",
    cache: bool = True,
) -> pd.DataFrame:
    ticker = ticker.upper()
    cache_path = project_root() / "data" / f"{ticker}_{period}.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache and cache_path.exists():
        frame = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        if not frame.empty:
            return _clean_ohlcv(frame)

    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"No price data returned for {ticker}")

    frame = _clean_ohlcv(raw)
    if cache:
        frame.to_csv(cache_path)
    return frame


def _clean_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = [str(col[0]).title() for col in frame.columns]

    rename = {col: str(col).title() for col in frame.columns}
    frame = frame.rename(columns=rename)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in needed if col not in frame.columns]
    if missing:
        raise ValueError(f"Price frame missing columns: {missing}")

    out = frame[needed].copy()
    out.index = pd.to_datetime(out.index)
    out.index.name = "Date"
    out = out.sort_index().dropna(subset=["Close"])
    return out
