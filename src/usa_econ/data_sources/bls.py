import requests
import pandas as pd
import logging
from typing import Optional

from ..config import Config

logger = logging.getLogger(__name__)

# Request timeout in seconds
REQUEST_TIMEOUT = 30


def get_timeseries(
    series_id: str,
    config: Config,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Fetch BLS timeseries data and return a DataFrame indexed by date.

    Note: BLS periods use M01..M13; we map them to YYYY-MM-01.
    M13 represents annual averages and is mapped to January 1st.

    Args:
        series_id: BLS series identifier (e.g., 'LNS14000000' for unemployment)
        config: Configuration object containing API credentials
        start_year: Starting year for data (e.g., 2010)
        end_year: Ending year for data (e.g., 2023)

    Returns:
        DataFrame with date index and series values

    Raises:
        ValueError: If parameters are invalid or no data is returned
        ConnectionError: If unable to connect to BLS API
        RuntimeError: For other API-related errors
    """
    # Validate inputs
    if not series_id or not isinstance(series_id, str):
        raise ValueError(f"Invalid series_id: {series_id}. Must be a non-empty string.")

    current_year = pd.Timestamp.now().year
    if not isinstance(start_year, int) or start_year < 1900 or start_year > current_year:
        raise ValueError(f"Invalid start_year: {start_year}. Must be between 1900 and {current_year}.")

    if not isinstance(end_year, int) or end_year < start_year or end_year > current_year + 1:
        raise ValueError(f"Invalid end_year: {end_year}. Must be >= start_year and <= {current_year + 1}.")

    # Prepare API request
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    payload = {
        "seriesid": [series_id],
        "startyear": str(start_year),
        "endyear": str(end_year),
    }

    # Add API key if available (increases rate limits)
    if config.bls_api:
        payload["registrationkey"] = config.bls_api
        logger.info(f"Using BLS API key for enhanced rate limits")
    else:
        logger.warning(
            "No BLS API key found. Using public API with limited rate limits. "
            "Get a free key at https://data.bls.gov/registrationEngine/"
        )

    try:
        logger.info(f"Fetching BLS series: {series_id} ({start_year}-{end_year})")

        r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()

        # Check API response status
        status = data.get("status")
        if status != "REQUEST_SUCCEEDED":
            error_messages = data.get("message", [])
            error_text = "; ".join(error_messages) if error_messages else "Unknown error"
            raise RuntimeError(f"BLS API request failed: {error_text}")

        # Extract series data
        series = data.get("Results", {}).get("series", [])
        if not series:
            raise ValueError(
                f"No data returned for BLS series {series_id}. "
                f"Please verify the series ID at https://data.bls.gov/"
            )

        obs = series[0].get("data", [])
        if not obs:
            raise ValueError(f"No observations found for BLS series {series_id}")

        # Parse observations
        rows = []
        for item in obs:
            try:
                period = item.get("period", "")
                year = item.get("year", "")
                value = item.get("value", "")

                # Parse period to date
                if period.startswith("M") and period != "M13":
                    month = int(period[1:])
                    if not 1 <= month <= 12:
                        logger.warning(f"Invalid month {month} in period {period}, skipping")
                        continue
                    date_str = f"{year}-{month:02d}-01"
                elif period == "M13":
                    # M13 is annual average, map to January
                    date_str = f"{year}-01-01"
                else:
                    # Fallback for other period types (quarterly, annual, etc.)
                    date_str = f"{year}-01-01"

                # Convert value to float
                try:
                    float_value = float(value)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid value '{value}' for {series_id} at {date_str}, skipping")
                    continue

                rows.append({"date": date_str, series_id: float_value})

            except Exception as e:
                logger.warning(f"Error parsing observation: {item}. Error: {e}")
                continue

        if not rows:
            raise ValueError(f"No valid observations could be parsed for series {series_id}")

        # Create DataFrame
        df = pd.DataFrame(rows).sort_values("date")
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        logger.info(f"Successfully fetched {len(df)} observations for {series_id}")
        return df

    except requests.exceptions.Timeout:
        logger.error(f"Timeout while fetching BLS series {series_id}")
        raise ConnectionError(
            f"Request to BLS API timed out after {REQUEST_TIMEOUT} seconds. "
            f"Please check your internet connection and try again."
        )

    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error while fetching BLS series {series_id}: {e}")
        raise ConnectionError(
            f"Unable to connect to BLS API. Please check your internet connection. Error: {e}"
        )

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error while fetching BLS series {series_id}: {e}")
        if e.response.status_code == 429:
            raise RuntimeError(
                f"BLS API rate limit exceeded. Please wait and try again, or register for an API key at "
                f"https://data.bls.gov/registrationEngine/"
            )
        elif e.response.status_code in (401, 403):
            raise ValueError(f"BLS API authentication failed. Please check your API key. Error: {e}")
        else:
            raise RuntimeError(f"HTTP error {e.response.status_code} while fetching BLS series: {e}")

    except ValueError as e:
        # Re-raise validation errors
        raise

    except Exception as e:
        logger.error(f"Unexpected error fetching BLS series {series_id}: {e}")
        raise RuntimeError(f"Error fetching BLS series {series_id}: {e}")