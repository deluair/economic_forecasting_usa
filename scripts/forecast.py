import sys
from pathlib import Path
import typer
from rich import print
import pandas as pd

# Ensure `src` is on path when running the script directly
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from usa_econ.config import load_config
from usa_econ.data_sources.fred import get_series as fred_get_series
from usa_econ.models.arima import arima_forecast
from usa_econ.models.var import var_forecast
from usa_econ.models.prophet_model import prophet_forecast
from usa_econ.models.lstm_model import lstm_forecast
from usa_econ.models.ensemble import ensemble_forecast
from usa_econ.models.evaluation import compare_models, model_selection_report
from usa_econ.utils.io import save_df_csv, ensure_dir


app = typer.Typer(help="Advanced Economic Forecasting CLI")


@app.command()
def arima(
    series_id: str = typer.Argument(..., help="FRED series ID, e.g., CPIAUCSL"),
    steps: int = typer.Option(12, help="Forecast horizon in periods"),
    order: tuple[int, int, int] = typer.Option((1, 1, 1), help="ARIMA order p,d,q"),
    seasonal_order: tuple[int, int, int, int] = typer.Option((0, 1, 1, 12), help="Seasonal order P,D,Q,m"),
    start: str | None = typer.Option(None, help="Fetch start date if file is missing"),
    end: str | None = typer.Option(None, help="Fetch end date if file is missing"),
):
    """Run ARIMA forecast on a single series."""
    cfg = load_config()
    src_path = Path("data/raw/fred") / f"{series_id}.csv"
    if src_path.exists():
        df = pd.read_csv(src_path, parse_dates=["date"], index_col="date")
    else:
        print(f"[yellow]Source not found; fetching {series_id} from FRED[/yellow]")
        df = fred_get_series(series_id, cfg, start=start, end=end)
        save_df_csv(df, src_path)
    fc = arima_forecast(df[series_id], steps=steps, order=order, seasonal_order=seasonal_order)
    out_dir = Path("data/processed/forecasts")
    ensure_dir(out_dir)
    out_path = out_dir / f"arima_{series_id}.csv"
    save_df_csv(fc, out_path)
    print({"saved": str(out_path)})


@app.command()
def prophet(
    series_id: str = typer.Argument(..., help="FRED series ID, e.g., CPIAUCSL"),
    steps: int = typer.Option(12, help="Forecast horizon in periods"),
    start: str | None = typer.Option(None, help="Fetch start date if file is missing"),
    end: str | None = typer.Option(None, help="Fetch end date if file is missing"),
):
    """Run Prophet forecast on a single series."""
    cfg = load_config()
    src_path = Path("data/raw/fred") / f"{series_id}.csv"
    if src_path.exists():
        df = pd.read_csv(src_path, parse_dates=["date"], index_col="date")
    else:
        print(f"[yellow]Source not found; fetching {series_id} from FRED[/yellow]")
        df = fred_get_series(series_id, cfg, start=start, end=end)
        save_df_csv(df, src_path)
    fc = prophet_forecast(df[series_id], steps=steps)
    out_dir = Path("data/processed/forecasts")
    ensure_dir(out_dir)
    out_path = out_dir / f"prophet_{series_id}.csv"
    save_df_csv(fc, out_path)
    print({"saved": str(out_path)})


@app.command()
def lstm(
    series_id: str = typer.Argument(..., help="FRED series ID, e.g., CPIAUCSL"),
    steps: int = typer.Option(12, help="Forecast horizon in periods"),
    look_back: int = typer.Option(12, help="Number of previous periods to use as input"),
    epochs: int = typer.Option(100, help="Training epochs"),
    start: str | None = typer.Option(None, help="Fetch start date if file is missing"),
    end: str | None = typer.Option(None, help="Fetch end date if file is missing"),
):
    """Run LSTM forecast on a single series."""
    cfg = load_config()
    src_path = Path("data/raw/fred") / f"{series_id}.csv"
    if src_path.exists():
        df = pd.read_csv(src_path, parse_dates=["date"], index_col="date")
    else:
        print(f"[yellow]Source not found; fetching {series_id} from FRED[/yellow]")
        df = fred_get_series(series_id, cfg, start=start, end=end)
        save_df_csv(df, src_path)
    fc = lstm_forecast(df[series_id], steps=steps, look_back=look_back, epochs=epochs, verbose=1)
    out_dir = Path("data/processed/forecasts")
    ensure_dir(out_dir)
    out_path = out_dir / f"lstm_{series_id}.csv"
    save_df_csv(fc, out_path)
    print({"saved": str(out_path)})


@app.command()
def ensemble(
    series_id: str = typer.Argument(..., help="FRED series ID, e.g., CPIAUCSL"),
    steps: int = typer.Option(12, help="Forecast horizon in periods"),
    models: list[str] = typer.Option(["arima", "prophet", "rf", "gbm"], help="Models to include"),
    start: str | None = typer.Option(None, help="Fetch start date if file is missing"),
    end: str | None = typer.Option(None, help="Fetch end date if file is missing"),
):
    """Run ensemble forecast combining multiple models."""
    cfg = load_config()
    src_path = Path("data/raw/fred") / f"{series_id}.csv"
    if src_path.exists():
        df = pd.read_csv(src_path, parse_dates=["date"], index_col="date")
    else:
        print(f"[yellow]Source not found; fetching {series_id} from FRED[/yellow]")
        df = fred_get_series(series_id, cfg, start=start, end=end)
        save_df_csv(df, src_path)
    fc = ensemble_forecast(df[series_id], steps=steps, models=models)
    out_dir = Path("data/processed/forecasts")
    ensure_dir(out_dir)
    out_path = out_dir / f"ensemble_{series_id}.csv"
    save_df_csv(fc, out_path)
    print({"saved": str(out_path)})


@app.command()
def var(
    series_ids: list[str] = typer.Argument(..., help="List of FRED series IDs"),
    steps: int = typer.Option(12, help="Forecast horizon in periods"),
    start: str | None = typer.Option(None, help="Fetch start date if any file is missing"),
    end: str | None = typer.Option(None, help="Fetch end date if any file is missing"),
):
    """Run VAR forecast on multiple series."""
    cfg = load_config()
    frames = []
    for sid in series_ids:
        p = Path("data/raw/fred") / f"{sid}.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"], index_col="date")
        else:
            print(f"[yellow]Source not found; fetching {sid} from FRED[/yellow]")
            df = fred_get_series(sid, cfg, start=start, end=end)
            save_df_csv(df, p)
        frames.append(df)
    combined = pd.concat(frames, axis=1, join="inner").dropna()
    fc = var_forecast(combined, steps=steps)
    out_dir = Path("data/processed/forecasts")
    ensure_dir(out_dir)
    joined = "_".join(series_ids)
    out_path = out_dir / f"var_{joined}.csv"
    save_df_csv(fc, out_path)
    print({"saved": str(out_path)})


@app.command()
def compare(
    series_id: str = typer.Argument(..., help="FRED series ID to compare models on"),
    test_size: int = typer.Option(24, help="Number of periods for testing"),
    start: str | None = typer.Option(None, help="Fetch start date if file is missing"),
    end: str | None = typer.Option(None, help="Fetch end date if file is missing"),
):
    """Compare multiple models using backtesting."""
    cfg = load_config()
    src_path = Path("data/raw/fred") / f"{series_id}.csv"
    if src_path.exists():
        df = pd.read_csv(src_path, parse_dates=["date"], index_col="date")
    else:
        print(f"[yellow]Source not found; fetching {series_id} from FRED[/yellow]")
        df = fred_get_series(series_id, cfg, start=start, end=end)
        save_df_csv(df, src_path)
    
    # Define model configurations
    models = {
        'arima': {'type': 'arima', 'params': {}},
        'prophet': {'type': 'prophet', 'params': {}},
        'ensemble': {'type': 'ensemble', 'params': {'models': ['arima', 'prophet', 'rf']}}
    }
    
    try:
        comparison = compare_models(df[series_id], models, test_size=test_size)
        out_dir = Path("data/processed/comparisons")
        ensure_dir(out_dir)
        out_path = out_dir / f"comparison_{series_id}.csv"
        save_df_csv(comparison, out_path)
        print({"saved": str(out_path)})
        print(comparison.to_string())
    except Exception as e:
        print(f"[red]Model comparison failed: {e}[/red]")


@app.command()
def report(
    series_id: str = typer.Argument(..., help="FRED series ID for comprehensive report"),
    test_size: int = typer.Option(24, help="Number of periods for testing"),
    start: str | None = typer.Option(None, help="Fetch start date if file is missing"),
    end: str | None = typer.Option(None, help="Fetch end date if file is missing"),
):
    """Generate comprehensive model selection report."""
    cfg = load_config()
    src_path = Path("data/raw/fred") / f"{series_id}.csv"
    if src_path.exists():
        df = pd.read_csv(src_path, parse_dates=["date"], index_col="date")
    else:
        print(f"[yellow]Source not found; fetching {series_id} from FRED[/yellow]")
        df = fred_get_series(series_id, cfg, start=start, end=end)
        save_df_csv(df, src_path)
    
    # Define model configurations
    models = {
        'arima': {'type': 'arima', 'params': {}},
        'prophet': {'type': 'prophet', 'params': {}},
        'ensemble': {'type': 'ensemble', 'params': {'models': ['arima', 'prophet', 'rf']}}
    }
    
    try:
        out_dir = Path("data/processed/reports")
        ensure_dir(out_dir)
        plot_path = out_dir / f"report_{series_id}.png"
        
        report = model_selection_report(df[series_id], models, test_size=test_size, save_path=str(plot_path))
        
        # Save report data
        report_path = out_dir / f"report_{series_id}.csv"
        save_df_csv(report['comparison_table'], report_path)
        
        print({"report_saved": str(report_path), "plot_saved": str(plot_path)})
        print(f"Best model: {report['best_model']}")
        print(report['comparison_table'].to_string())
    except Exception as e:
        print(f"[red]Report generation failed: {e}[/red]")


if __name__ == "__main__":
    app()