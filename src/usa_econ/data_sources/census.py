import requests
import pandas as pd
import logging
from typing import Sequence, Mapping

from ..config import Config

logger = logging.getLogger(__name__)

# Request timeout in seconds
REQUEST_TIMEOUT = 30


def get_dataset(
    dataset: str,
    variables: Sequence[str],
    predicates: Mapping[str, str],
    config: Config,
) -> pd.DataFrame:
    """Generic Census API request.

    Example:
        dataset="2023/acs/acs1/subject"
        variables=["NAME","S0101_C01_001E"]
        predicates={"for":"us:1"}

    Args:
        dataset: Census dataset path (e.g., "2023/acs/acs1/subject")
        variables: List of variable names to retrieve
        predicates: Geographic/filtering predicates (e.g., {"for": "us:1"})
        config: Configuration object containing API credentials

    Returns:
        DataFrame with requested variables and geography

    Raises:
        ValueError: If parameters are invalid or no data is returned
        ConnectionError: If unable to connect to Census API
        RuntimeError: For other API-related errors
    """
    # Validate inputs
    if not dataset or not isinstance(dataset, str):
        raise ValueError(f"Invalid dataset: {dataset}. Must be a non-empty string.")

    if not variables or not isinstance(variables, (list, tuple)):
        raise ValueError(f"Invalid variables: {variables}. Must be a non-empty sequence.")

    if not predicates or not isinstance(predicates, dict):
        raise ValueError(f"Invalid predicates: {predicates}. Must be a non-empty dictionary.")

    # Prepare API request
    url = f"https://api.census.gov/data/{dataset}"
    params = {"get": ",".join(variables)}
    params.update(predicates)

    # Add API key if available
    if config.census_api:
        params["key"] = config.census_api
        logger.info(f"Using Census API key for enhanced access")
    else:
        logger.warning(
            "No Census API key found. Some datasets may be unavailable. "
            "Get a free key at https://api.census.gov/data/key_signup.html"
        )

    try:
        logger.info(f"Fetching Census dataset: {dataset} with variables: {variables}")

        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()

        # Parse JSON response
        try:
            rows = r.json()
        except ValueError as e:
            raise RuntimeError(f"Invalid JSON response from Census API: {e}")

        # Validate response structure
        if not rows or not isinstance(rows, list):
            raise ValueError(f"Invalid response structure from Census API for dataset {dataset}")

        if len(rows) < 2:
            raise ValueError(
                f"No data returned for Census dataset {dataset}. "
                f"Please verify dataset path and predicates."
            )

        # First row contains column names
        cols = rows[0]
        if not cols or not isinstance(cols, list):
            raise ValueError(f"Invalid column structure in Census API response")

        # Remaining rows contain data
        data_rows = rows[1:]
        if not data_rows:
            raise ValueError(f"No data rows returned for Census dataset {dataset}")

        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=cols)

        logger.info(f"Successfully fetched {len(df)} rows from Census dataset {dataset}")
        return df

    except requests.exceptions.Timeout:
        logger.error(f"Timeout while fetching Census dataset {dataset}")
        raise ConnectionError(
            f"Request to Census API timed out after {REQUEST_TIMEOUT} seconds. "
            f"Please check your internet connection and try again."
        )

    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error while fetching Census dataset {dataset}: {e}")
        raise ConnectionError(
            f"Unable to connect to Census API. Please check your internet connection. Error: {e}"
        )

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error while fetching Census dataset {dataset}: {e}")
        if e.response.status_code == 400:
            raise ValueError(
                f"Bad request to Census API. Please check dataset path and parameters. "
                f"Available datasets: https://api.census.gov/data.html. Error: {e}"
            )
        elif e.response.status_code == 404:
            raise ValueError(
                f"Census dataset not found: {dataset}. "
                f"Please verify the dataset path at https://api.census.gov/data.html"
            )
        elif e.response.status_code in (401, 403):
            raise ValueError(f"Census API authentication failed. Please check your API key. Error: {e}")
        elif e.response.status_code == 429:
            raise RuntimeError(
                f"Census API rate limit exceeded. Please wait and try again, or use an API key."
            )
        else:
            raise RuntimeError(f"HTTP error {e.response.status_code} while fetching Census data: {e}")

    except ValueError as e:
        # Re-raise validation errors
        raise

    except Exception as e:
        logger.error(f"Unexpected error fetching Census dataset {dataset}: {e}")
        raise RuntimeError(f"Error fetching Census dataset {dataset}: {e}")