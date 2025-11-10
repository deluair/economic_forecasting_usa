from __future__ import annotations

from typing import Tuple
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _ensure_series(x: pd.Series | pd.DataFrame, column: str | None = None) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        if column is None:
            raise ValueError("column must be provided when x is a DataFrame")
        return x[column].astype(float)
    return x.astype(float)


def arima_forecast(
    x: pd.Series | pd.DataFrame,
    steps: int = 12,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    column: str | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Fit SARIMAX and return forecast with confidence intervals.

    Returns a DataFrame indexed by date with columns: yhat, lower, upper.
    """
    y = _ensure_series(x, column)
    model = SARIMAX(y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=steps)
    mean = pred.predicted_mean.rename("yhat")
    ci = pred.conf_int(alpha=alpha)
    lower = ci.iloc[:, 0].rename("lower")
    upper = ci.iloc[:, 1].rename("upper")
    out = pd.concat([mean, lower, upper], axis=1)
    return out