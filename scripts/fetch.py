import sys
from pathlib import Path
import typer
from rich import print

# Ensure `src` is on path when running the script directly
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from usa_econ.config import load_config
from usa_econ.data_sources.fred import get_series as fred_get_series
from usa_econ.pipeline.fetch_indicators import fetch_core_indicators
from usa_econ.utils.io import save_df_csv


app = typer.Typer(help="Economic data fetching CLI")


@app.command()
def core_indicators(
    start: str | None = typer.Option(None, help="Observation start date, e.g., 2000-01-01"),
    end: str | None = typer.Option(None, help="Observation end date, e.g., 2024-12-31"),
):
    """Fetch a curated set of core macroeconomic indicators from FRED."""
    paths = fetch_core_indicators(start=start, end=end)
    print({"saved": {k: str(v) for k, v in paths.items()}})


@app.command()
def fred_series(
    series_id: str = typer.Argument(..., help="FRED series ID, e.g., CPIAUCSL"),
    start: str | None = typer.Option(None, help="Observation start date"),
    end: str | None = typer.Option(None, help="Observation end date"),
    out: Path = typer.Option(Path("data/raw/fred"), help="Output directory"),
):
    """Fetch an arbitrary FRED series and save to CSV."""
    cfg = load_config()
    df = fred_get_series(series_id, cfg, start=start, end=end)
    out_path = out / f"{series_id}.csv"
    save_df_csv(df, out_path)
    print(f"Saved {series_id} to {out_path}")


if __name__ == "__main__":
    app()