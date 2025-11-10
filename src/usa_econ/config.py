from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv


@dataclass
class Config:
    fred_api_key: str | None
    bls_api: str | None
    census_api: str | None
    eia_api: str | None
    project_root: Path


def load_config(env_path: Path | None = None) -> Config:
    """Load configuration and secrets from .env in the project root.

    If env_path is not provided, it will look for `.env` at the repository root.
    """
    # src/usa_econ/config.py -> usa_econ (0) -> src (1) -> project_root (2)
    project_root = Path(__file__).resolve().parents[2]
    if env_path is None:
        env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    return Config(
        fred_api_key=os.getenv("FRED_API_KEY"),
        bls_api=os.getenv("BLS_API"),
        census_api=os.getenv("Census_Data_API"),
        eia_api=os.getenv("EIA_API"),
        project_root=project_root,
    )