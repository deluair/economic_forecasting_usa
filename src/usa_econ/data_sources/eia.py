import requests
import pandas as pd
import logging

from ..config import Config

logger = logging.getLogger(__name__)

# Request timeout in seconds
REQUEST_TIMEOUT = 30


def get_series(series_id: str, config: Config) -> pd.DataFrame:
    """Fetch an EIA series by ID using the v1 /series endpoint.

    Note: EIA dates may be YYYYMM or YYYY; we attempt to parse monthly format first,
    then fallback to generic datetime parsing.

    Args:
        series_id: EIA series identifier (e.g., 'PET.RWTC.D' for crude oil prices)
        config: Configuration object containing API credentials

    Returns:
        DataFrame with date index and series values

    Raises:
        ValueError: If API key is missing, series_id is invalid, or no data is returned
        ConnectionError: If unable to connect to EIA API
        RuntimeError: For other API-related errors
    """
    # Validate API key
    if not config.eia_api:
        raise ValueError(
            "EIA API key is missing. Please set EIA_API in your .env file. "
            "Get a free key at https://www.eia.gov/opendata/register.php"
        )

    # Validate series_id
    if not series_id or not isinstance(series_id, str):
        raise ValueError(f"Invalid series_id: {series_id}. Must be a non-empty string.")

    try:
        url = "https://api.eia.gov/series/"
        params = {"api_key": config.eia_api, "series_id": series_id}

        logger.info(f"Fetching EIA series: {series_id}")

        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()

        # Parse JSON response
        try:
            data = r.json()
        except ValueError as e:
            raise RuntimeError(f"Invalid JSON response from EIA API: {e}")

        # Check for API errors
        if "data" in data and "error" in data["data"]:
            error_msg = data["data"]["error"]
            raise ValueError(f"EIA API error: {error_msg}")

        # Validate response structure
        if "series" not in data:
            raise ValueError(
                f"No data returned for EIA series {series_id}. "
                f"Please verify the series ID at https://www.eia.gov/opendata/"
            )

        series_list = data["series"]
        if not series_list or not isinstance(series_list, list):
            raise ValueError(f"Invalid series structure in EIA API response")

        series_data = series_list[0]
        if "data" not in series_data:
            raise ValueError(f"No data points found for EIA series {series_id}")

        points = series_data["data"]
        if not points:
            raise ValueError(f"Empty data array for EIA series {series_id}")

        # Create DataFrame
        df = pd.DataFrame(points, columns=["date", series_id])

        # Parse dates with fallback logic
        # EIA dates can be: YYYYMM (monthly), YYYY (annual), or other formats
        try:
            # First try YYYYMM format (most common)
            df["date"] = pd.to_datetime(df["date"], format="%Y%m", errors="coerce")

            # For rows where YYYYMM failed, try generic parsing
            null_dates = df["date"].isnull()
            if null_dates.any():
                df.loc[null_dates, "date"] = pd.to_datetime(
                    df.loc[null_dates, "date"].astype(str),
                    errors="coerce"
                )

            # Check if we still have null dates
            if df["date"].isnull().any():
                logger.warning(
                    f"{df['date'].isnull().sum()} dates could not be parsed for series {series_id}"
                )
                # Drop rows with null dates
                df = df.dropna(subset=["date"])

            if len(df) == 0:
                raise ValueError(f"No valid dates could be parsed for EIA series {series_id}")

        except Exception as e:
            raise RuntimeError(f"Error parsing dates for EIA series {series_id}: {e}")

        # Sort and set index
        df = df.sort_values("date")
        df = df.set_index("date")

        logger.info(f"Successfully fetched {len(df)} observations for {series_id}")
        return df

    except requests.exceptions.Timeout:
        logger.error(f"Timeout while fetching EIA series {series_id}")
        raise ConnectionError(
            f"Request to EIA API timed out after {REQUEST_TIMEOUT} seconds. "
            f"Please check your internet connection and try again."
        )

    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error while fetching EIA series {series_id}: {e}")
        raise ConnectionError(
            f"Unable to connect to EIA API. Please check your internet connection. Error: {e}"
        )

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error while fetching EIA series {series_id}: {e}")
        if e.response.status_code == 400:
            raise ValueError(
                f"Bad request to EIA API. Invalid series ID: {series_id}. "
                f"Browse available series at https://www.eia.gov/opendata/browser/"
            )
        elif e.response.status_code in (401, 403):
            raise ValueError(
                f"EIA API authentication failed. Please check your API key. Error: {e}"
            )
        elif e.response.status_code == 404:
            raise ValueError(
                f"EIA series not found: {series_id}. "
                f"Please verify the series ID at https://www.eia.gov/opendata/browser/"
            )
        elif e.response.status_code == 429:
            raise RuntimeError(
                f"EIA API rate limit exceeded. Please wait and try again. Error: {e}"
            )
        else:
            raise RuntimeError(f"HTTP error {e.response.status_code} while fetching EIA series: {e}")

    except ValueError as e:
        # Re-raise validation errors
        raise

    except Exception as e:
        logger.error(f"Unexpected error fetching EIA series {series_id}: {e}")
        raise RuntimeError(f"Error fetching EIA series {series_id}: {e}")