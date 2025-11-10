import pandas as pd
from fredapi import Fred
import logging
from typing import Optional

from ..config import Config

logger = logging.getLogger(__name__)


def get_series(
    series_id: str,
    config: Config,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch a FRED series by ID and return a DataFrame indexed by date.

    Args:
        series_id: FRED series identifier (e.g., 'GDP', 'UNRATE')
        config: Configuration object containing API credentials
        start: Optional start date (YYYY-MM-DD format)
        end: Optional end date (YYYY-MM-DD format)

    Returns:
        DataFrame with date index and series values

    Raises:
        ValueError: If API key is missing or series_id is invalid
        ConnectionError: If unable to connect to FRED API
        RuntimeError: For other API-related errors
    """
    # Validate API key
    if not config.fred_api_key:
        raise ValueError(
            "FRED API key is missing. Please set FRED_API_KEY in your .env file. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    # Validate series_id
    if not series_id or not isinstance(series_id, str):
        raise ValueError(f"Invalid series_id: {series_id}. Must be a non-empty string.")

    try:
        fred = Fred(api_key=config.fred_api_key)
        logger.info(f"Fetching FRED series: {series_id} (start={start}, end={end})")

        s = fred.get_series(series_id, observation_start=start, observation_end=end)

        if s is None or len(s) == 0:
            raise ValueError(f"No data returned for series {series_id}")

        df = s.to_frame(name=series_id)
        df.index.name = "date"

        logger.info(f"Successfully fetched {len(df)} observations for {series_id}")
        return df

    except ValueError as e:
        # Re-raise validation errors
        raise

    except ConnectionError as e:
        logger.error(f"Network error while fetching FRED series {series_id}: {e}")
        raise ConnectionError(
            f"Unable to connect to FRED API. Please check your internet connection. "
            f"Error: {e}"
        )

    except Exception as e:
        # Catch-all for FRED API errors (invalid series, rate limits, etc.)
        error_msg = str(e).lower()

        if "bad request" in error_msg or "400" in error_msg:
            raise ValueError(
                f"Invalid FRED series ID: {series_id}. "
                f"Please verify the series exists at https://fred.stlouisfed.org/"
            )
        elif "unauthorized" in error_msg or "401" in error_msg or "403" in error_msg:
            raise ValueError(
                f"FRED API authentication failed. Please check your API key. Error: {e}"
            )
        elif "rate limit" in error_msg or "429" in error_msg:
            raise RuntimeError(
                f"FRED API rate limit exceeded. Please wait and try again. Error: {e}"
            )
        else:
            logger.error(f"Unexpected error fetching FRED series {series_id}: {e}")
            raise RuntimeError(
                f"Error fetching FRED series {series_id}: {e}"
            )