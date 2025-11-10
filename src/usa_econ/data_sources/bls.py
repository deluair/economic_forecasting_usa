import requests
import pandas as pd

from ..config import Config


def get_timeseries(
    series_id: str,
    config: Config,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Fetch BLS timeseries data and return a DataFrame indexed by date.

    Note: BLS periods use M01..M13; we map them to YYYY-MM-01.
    """
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    payload = {
        "seriesid": [series_id],
        "startyear": str(start_year),
        "endyear": str(end_year),
    }
    if config.bls_api:
        payload["registrationkey"] = config.bls_api

    r = requests.post(url, json=payload)
    r.raise_for_status()
    data = r.json()

    series = data.get("Results", {}).get("series", [])
    if not series:
        raise ValueError(f"No data returned for BLS series {series_id}")

    obs = series[0].get("data", [])
    rows = []
    for item in obs:
        period = item["period"]
        year = item["year"]
        value = item["value"]
        if period.startswith("M") and period != "M13":
            month = int(period[1:])
            date_str = f"{year}-{month:02d}-01"
        else:
            date_str = f"{year}-01-01"
        rows.append({"date": date_str, series_id: float(value)})

    df = pd.DataFrame(rows).sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df