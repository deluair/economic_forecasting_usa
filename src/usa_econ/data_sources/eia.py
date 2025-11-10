import requests
import pandas as pd

from ..config import Config


def get_series(series_id: str, config: Config) -> pd.DataFrame:
    """Fetch an EIA series by ID using the v1 /series endpoint.

    Note: EIA dates may be YYYYMM or YYYY; we attempt to parse monthly.
    """
    url = "https://api.eia.gov/series/"
    params = {"api_key": config.eia_api, "series_id": series_id}
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    if "series" not in data:
        raise ValueError(f"No data for EIA series {series_id}")
    points = data["series"][0]["data"]
    df = pd.DataFrame(points, columns=["date", series_id])
    # Try to parse YYYYMM; fallback to generic datetime parsing
    df["date"] = pd.to_datetime(df["date"], format="%Y%m", errors="coerce").fillna(pd.to_datetime(df["date"], errors="coerce"))
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)
    return df