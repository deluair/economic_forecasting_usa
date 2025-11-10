import sys
from pathlib import Path
import typer
from rich import print
from rich.console import Console
from rich.table import Table
import pandas as pd

# Ensure `src` is on path when running the script directly
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from usa_econ.pipeline.economic_analyzer import EconomicAnalyzer, analyze_economy


app = typer.Typer(help="US Economic Analysis CLI")
console = Console()


@app.command()
def analyze(
    indicators: list[str] = typer.Option(
        None, 
        help="Specific indicators to analyze (default: all key indicators)"
    ),
    forecast_steps: int = typer.Option(12, help="Number of periods to forecast"),
    save: bool = typer.Option(True, help="Save report to files"),
    output_dir: str = typer.Option("data/processed/reports", help="Output directory for reports")
):
    """Generate comprehensive economic analysis report."""
    
    console.print("[bold blue]🇺🇸 US Economic Analysis Report[/bold blue]")
    console.print("=" * 50)
    
    try:
        # Generate analysis
        report = analyze_economy(
            indicators=indicators, 
            forecast_steps=forecast_steps, 
            save_report=save
        )
        
        # Display report summary
        _display_report_summary(report)
        
        if save and 'saved_files' in report:
            console.print(f"\n[green]✓ Report saved to:[/green]")
            for file_type, file_path in report['saved_files'].items():
                if file_path:
                    console.print(f"  {file_type}: {file_path}")
        
    except Exception as e:
        console.print(f"[red]Error generating analysis: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def signals(
    indicators: list[str] = typer.Option(
        None, 
        help="Specific indicators to analyze"
    )
):
    """Display current economic signals only."""
    
    console.print("[bold blue]📊 Current Economic Signals[/bold blue]")
    console.print("=" * 40)
    
    try:
        analyzer = EconomicAnalyzer()
        data = analyzer.fetch_latest_data(indicators)
        signals = analyzer.calculate_economic_signals(data)
        
        # Create signals table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Indicator", style="cyan", width=20)
        table.add_column("Signal", style="green", width=20)
        table.add_column("Value", style="yellow", width=15)
        table.add_column("Trend", width=15)
        
        for indicator, signal_data in signals.items():
            signal = signal_data.get('signal', 'N/A')
            
            # Extract key value for display
            if 'growth_yoy' in signal_data:
                value = f"{signal_data['growth_yoy']:.1%}"
            elif 'yoy_change' in signal_data:
                value = f"{signal_data['yoy_change']:.1%}"
            elif 'unemployment_rate' in signal_data:
                value = f"{signal_data['unemployment_rate']:.1%}"
            elif 'fed_funds_rate' in signal_data:
                value = f"{signal_data['fed_funds_rate']:.1%}"
            elif 'confidence_index' in signal_data:
                value = f"{signal_data['confidence_index']:.1f}"
            else:
                value = "N/A"
            
            trend = signal_data.get('trend', 'N/A')
            
            table.add_row(indicator, signal, value, trend)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error generating signals: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def cycle():
    """Assess current business cycle phase."""
    
    console.print("[bold blue]🔄 Business Cycle Assessment[/bold blue]")
    console.print("=" * 40)
    
    try:
        analyzer = EconomicAnalyzer()
        data = analyzer.fetch_latest_data()
        cycle = analyzer.assess_business_cycle(data)
        
        console.print(f"\n[bold]Current Phase:[/bold] {cycle['cycle_phase']}")
        console.print(f"[bold]Recession Probability:[/bold] {cycle['recession_probability']:.1%}")
        
        console.print(f"\n[bold]Recession Indicators:[/bold]")
        indicator_names = ['GDP Contraction', 'Unemployment Rising', 'Industrial Production Decline']
        for i, (indicator, name) in enumerate(zip(cycle['indicators'], indicator_names[:len(cycle['indicators'])])):
            status = "🔴" if indicator else "🟢"
            console.print(f"  {status} {name}: {'Active' if indicator else 'Inactive'}")
        
    except Exception as e:
        console.print(f"[red]Error assessing business cycle: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def forecast(
    indicator: str = typer.Argument(..., help="Economic indicator to forecast"),
    steps: int = typer.Option(12, help="Forecast horizon in periods"),
    model: str = typer.Option("ensemble", help="Model to use (arima, prophet, lstm, ensemble)")
):
    """Generate forecast for specific economic indicator."""
    
    console.print(f"[bold blue]🔮 Economic Forecast: {indicator}[/bold blue]")
    console.print("=" * 50)
    
    try:
        analyzer = EconomicAnalyzer()
        data = analyzer.fetch_latest_data([indicator])
        
        if indicator not in data.columns:
            console.print(f"[red]Indicator '{indicator}' not available[/red]")
            raise typer.Exit(1)
        
        series_data = data[indicator].dropna()
        
        if len(series_data) < 24:
            console.print(f"[red]Insufficient data for forecasting. Need at least 24 observations, have {len(series_data)}[/red]")
            raise typer.Exit(1)
        
        # Generate forecast
        from usa_econ.models.ensemble import ensemble_forecast
        from usa_econ.models.arima import arima_forecast
        from usa_econ.models.prophet_model import prophet_forecast
        
        if model == "ensemble":
            forecast = ensemble_forecast(series_data, steps=steps)
        elif model == "arima":
            forecast = arima_forecast(series_data, steps=steps)
        elif model == "prophet":
            forecast = prophet_forecast(series_data, steps=steps)
        else:
            console.print(f"[red]Unknown model: {model}[/red]")
            raise typer.Exit(1)
        
        # Display forecast
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Date", style="cyan")
        table.add_column("Forecast", style="green")
        table.add_column("Lower Bound", style="yellow")
        table.add_column("Upper Bound", style="yellow")
        
        for date, row in forecast.iterrows():
            table.add_row(
                date.strftime('%Y-%m-%d'),
                f"{row['yhat']:.2f}",
                f"{row['lower']:.2f}",
                f"{row['upper']:.2f}"
            )
        
        console.print(table)
        
        # Calculate forecast summary
        last_actual = series_data.iloc[-1]
        forecast_end = forecast['yhat'].iloc[-1]
        total_change = (forecast_end - last_actual) / last_actual
        
        console.print(f"\n[bold]Forecast Summary:[/bold]")
        console.print(f"  Last Actual: {last_actual:.2f}")
        console.print(f"  Forecast End: {forecast_end:.2f}")
        console.print(f"  Total Change: {total_change:.1%}")
        
    except Exception as e:
        console.print(f"[red]Error generating forecast: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def compare(
    indicator: str = typer.Argument(..., help="Economic indicator to compare models on"),
    test_size: int = typer.Option(24, help="Number of periods for testing")
):
    """Compare forecasting models using backtesting."""
    
    console.print(f"[bold blue]⚖️ Model Comparison: {indicator}[/bold blue]")
    console.print("=" * 50)
    
    try:
        analyzer = EconomicAnalyzer()
        data = analyzer.fetch_latest_data([indicator])
        
        if indicator not in data.columns:
            console.print(f"[red]Indicator '{indicator}' not available[/red]")
            raise typer.Exit(1)
        
        from usa_econ.models.evaluation import compare_models
        
        models = {
            'arima': {'type': 'arima', 'params': {}},
            'prophet': {'type': 'prophet', 'params': {}},
            'ensemble': {'type': 'ensemble', 'params': {'models': ['arima', 'prophet', 'rf']}}
        }
        
        comparison = compare_models(data[indicator], models, test_size=test_size)
        
        # Display comparison table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Model", style="cyan")
        table.add_column("MAE", style="green")
        table.add_column("RMSE", style="green")
        table.add_column("MAPE", style="green")
        table.add_column("R²", style="green")
        table.add_column("Dir. Acc.", style="green")
        table.add_column("Overall Rank", style="bold")
        
        for model_name, row in comparison.iterrows():
            table.add_row(
                model_name,
                f"{row['mae']:.4f}",
                f"{row['rmse']:.4f}",
                f"{row['mape']:.4f}",
                f"{row['r2']:.4f}",
                f"{row['directional_accuracy']:.1%}",
                f"{row['overall_rank']:.1f}"
            )
        
        console.print(table)
        
        best_model = comparison.iloc[0].name
        console.print(f"\n[bold green]Best Model: {best_model}[/bold green]")
        
    except Exception as e:
        console.print(f"[red]Error comparing models: {e}[/red]")
        raise typer.Exit(1)


def _display_report_summary(report):
    """Display a formatted summary of the economic report."""
    
    # Report metadata
    console.print(f"[bold]Report Date:[/bold] {report['report_date']}")
    console.print(f"[bold]Data Period:[/bold] {report['data_period']['start']} to {report['data_period']['end']}")
    console.print(f"[bold]Observations:[/bold] {report['data_period']['observations']}")
    
    # Business cycle
    console.print(f"\n[bold]🔄 Business Cycle:[/bold] {report['business_cycle']['cycle_phase']}")
    console.print(f"[bold]Recession Probability:[/bold] {report['business_cycle']['recession_probability']:.1%}")
    
    # Key insights
    console.print(f"\n[bold]💡 Key Insights:[/bold]")
    for insight in report['key_insights']:
        console.print(f"  • {insight}")
    
    # Economic signals
    console.print(f"\n[bold]📊 Economic Signals:[/bold]")
    for category, signals in report['economic_signals'].items():
        console.print(f"  [bold]{category}:[/bold] {signals.get('signal', 'N/A')}")
    
    # Forecasts
    if report['forecasts']:
        console.print(f"\n[bold]🔮 Forecasts Generated:[/bold] {len(report['forecasts'])} indicators")
        for indicator in report['forecasts'].keys():
            console.print(f"  • {indicator}")


if __name__ == "__main__":
    app()
