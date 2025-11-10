from __future__ import annotations

import pandas as pd
from statsmodels.tsa.api import VAR


def var_forecast(df: pd.DataFrame, steps: int = 12, maxlags: int | None = None) -> pd.DataFrame:
    """Fit a VAR model on a multivariate DataFrame and forecast future values.

    If maxlags is None, use information criterion to select lag length.
    Returns a DataFrame with forecasted values for each column.
    """
    df = df.astype(float).dropna()
    model = VAR(df)
    if maxlags is None:
        res = model.fit(ic="aic")
    else:
        res = model.fit(maxlags)
    fc = res.forecast(df.values[-res.k_ar :], steps=steps)
    out = pd.DataFrame(fc, columns=df.columns)
    # Create a date index continuing the original frequency if available
    if df.index.freq is not None:
        start = df.index[-1] + df.index.freq
        out.index = pd.date_range(start=start, periods=steps, freq=df.index.freq)
    return out