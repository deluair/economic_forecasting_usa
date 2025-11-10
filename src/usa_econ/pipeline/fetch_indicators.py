from pathlib import Path
from typing import Dict

from ..config import load_config
from ..data_sources.fred import get_series
from ..utils.io import save_df_csv


CORE_SERIES: Dict[str, str] = {
    "GDPC1": "Real GDP, quarterly",
    "CPIAUCSL": "CPI, monthly",
    "UNRATE": "Unemployment rate, monthly",
    "INDPRO": "Industrial Production Index, monthly",
    "HOUST": "Housing Starts, monthly",
    "RSXFS": "Real Retail & Food Services Sales, monthly",
}


def fetch_core_indicators(start: str | None = None, end: str | None = None) -> Dict[str, Path]:
    """Fetch core macro indicators from FRED and save to CSV under data/raw/fred."""
    cfg = load_config()
    out_paths: Dict[str, Path] = {}
    base_dir = Path("data/raw/fred")

    for series_id in CORE_SERIES.keys():
        df = get_series(series_id, cfg, start=start, end=end)
        out_path = base_dir / f"{series_id}.csv"
        save_df_csv(df, out_path)
        out_paths[series_id] = out_path

    return out_paths