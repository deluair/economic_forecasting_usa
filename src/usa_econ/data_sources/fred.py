import pandas as pd
from fredapi import Fred

from ..config import Config


def get_series(
    series_id: str,
    config: Config,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch a FRED series by ID and return a DataFrame indexed by date."""
    fred = Fred(api_key=config.fred_api_key)
    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    df = s.to_frame(name=series_id)
    df.index.name = "date"
    return df