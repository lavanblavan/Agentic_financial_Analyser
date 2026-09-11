import pandas as pd

from src.indicators import realized_vol_pct


def test_realized_vol_is_positive_and_annualized():
    close = pd.Series([100.0, 101.0, 99.5, 102.0, 101.5, 103.0] * 10)
    vol = realized_vol_pct(close, window=20)
    assert vol > 0
    daily = close.pct_change().dropna().tail(20).std(ddof=1)
    assert abs(vol - daily * (252 ** 0.5) * 100) < 1e-9
