import requests
import pandas as pd
from typing import Sequence, Mapping

from ..config import Config


def get_dataset(
    dataset: str,
    variables: Sequence[str],
    predicates: Mapping[str, str],
    config: Config,
) -> pd.DataFrame:
    """Generic Census API request.

    Example: dataset="2023/acs/acs1/subject", variables=["NAME","S0101_C01_001E"], predicates={"for":"us:1"}
    """
    url = f"https://api.census.gov/data/{dataset}"
    params = {"get": ",".join(variables)}
    params.update(predicates)
    if config.census_api:
        params["key"] = config.census_api

    r = requests.get(url, params=params)
    r.raise_for_status()
    rows = r.json()
    cols = rows[0]
    df = pd.DataFrame(rows[1:], columns=cols)
    return df