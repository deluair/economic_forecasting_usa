import pandas as pd
from usa_econ.models.arima import arima_forecast


def test_arima_forecast_shape():
    # Simple synthetic monthly series
    idx = pd.date_range(start="2000-01-01", periods=60, freq="MS")
    s = pd.Series(range(60), index=idx)
    out = arima_forecast(s, steps=6)
    assert list(out.columns) == ["yhat", "lower", "upper"]
    assert len(out) == 6