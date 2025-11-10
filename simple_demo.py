#!/usr/bin/env python3
"""
Simple Economic Forecasting Demo
===============================

Demonstrates the core forecasting capabilities with sample data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[0] / "src"))

from usa_econ.models.prophet_model import prophet_forecast
from usa_econ.models.ensemble import ensemble_forecast
from usa_econ.models.var import var_forecast
from usa_econ.models.risk_modeling import EconomicRiskModeler
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def create_sample_data():
    """Create sample economic data for demonstration."""
    
    # Create date range
    dates = pd.date_range(start='2015-01-01', end='2023-12-31', freq='M')
    n_periods = len(dates)
    
    # Generate sample economic series with realistic patterns
    np.random.seed(42)
    
    # GDP growth (quarterly, annualized)
    gdp_growth = np.cumsum(np.random.normal(0.025, 0.01, n_periods)) + 2.0
    
    # CPI inflation
    inflation = np.cumsum(np.random.normal(0.002, 0.003, n_periods)) + 2.0
    
    # Unemployment rate
    unemployment = 5.0 + np.cumsum(np.random.normal(-0.01, 0.02, n_periods)) * 0.1
    unemployment = np.clip(unemployment, 3.0, 10.0)
    
    # Industrial production
    industrial_production = 100 + np.cumsum(np.random.normal(0.5, 2.0, n_periods))
    
    # Create DataFrame
    data = pd.DataFrame({
        'GDP': gdp_growth,
        'CPIAUCSL': inflation,
        'UNRATE': unemployment,
        'INDPRO': industrial_production
    }, index=dates)
    
    return data

def demo_prophet_forecasting():
    """Demonstrate Prophet forecasting with sample data."""
    console.print("\n[bold blue]🔮 Prophet Economic Forecasting Demo[/bold blue]")
    console.print("=" * 60)
    
    # Create sample data
    data = create_sample_data()
    unemployment_data = data['UNRATE']
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Generating Prophet forecast...", total=None)
        
        # Generate Prophet forecast
        forecast = prophet_forecast(unemployment_data, steps=12)
        
        progress.update(task, description="Prophet forecast completed")
    
    # Display results
    console.print(f"\n[bold]📊 Prophet Forecast Results (12 months)[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Date", style="cyan")
    table.add_column("Forecast", style="green")
    table.add_column("Lower Bound", style="yellow")
    table.add_column("Upper Bound", style="yellow")
    
    for date, row in forecast.head(6).iterrows():  # Show first 6 months
        table.add_row(
            date.strftime('%Y-%m-%d'),
            f"{row['yhat']:.2f}%",
            f"{row['lower']:.2f}%",
            f"{row['upper']:.2f}%"
        )
    
    console.print(table)
    
    # Calculate forecast insights
    last_actual = unemployment_data.iloc[-1]
    next_month_forecast = forecast['yhat'].iloc[0]
    forecast_trend = forecast['yhat'].iloc[-1] - forecast['yhat'].iloc[0]
    
    console.print(f"\n[bold]📈 Forecast Insights:[/bold]")
    console.print(f"  • Current unemployment rate: {last_actual:.2f}%")
    console.print(f"  • Next month forecast: {next_month_forecast:.2f}%")
    console.print(f"  • 12-month trend: {forecast_trend:+.2f}%")
    console.print(f"  • Forecast range width: {(forecast['upper'].iloc[0] - forecast['lower'].iloc[0]):.2f}%")
    
    return forecast

def demo_ensemble_forecasting():
    """Demonstrate ensemble forecasting with sample data."""
    console.print("\n[bold green]🎯 Ensemble Economic Forecasting Demo[/bold green]")
    console.print("=" * 60)
    
    # Create sample data
    data = create_sample_data()
    cpi_data = data['CPIAUCSL']
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Generating ensemble forecast (ARIMA + Prophet + RF)...", total=None)
        
        # Generate ensemble forecast
        forecast = ensemble_forecast(cpi_data, steps=12, models=['arima', 'prophet', 'rf'])
        
        progress.update(task, description="Ensemble forecast completed")
    
    # Display results
    console.print(f"\n[bold]📊 Ensemble Forecast Results (12 months)[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Date", style="cyan")
    table.add_column("Forecast", style="green")
    table.add_column("Lower Bound", style="yellow")
    table.add_column("Upper Bound", style="yellow")
    table.add_column("Inflation Rate", style="blue")
    
    last_cpi = cpi_data.iloc[-1]
    
    for date, row in forecast.head(6).iterrows():  # Show first 6 months
        inflation_rate = ((row['yhat'] - last_cpi) / last_cpi) * 100
        table.add_row(
            date.strftime('%Y-%m-%d'),
            f"{row['yhat']:.2f}",
            f"{row['lower']:.2f}",
            f"{row['upper']:.2f}",
            f"{inflation_rate:.2f}%"
        )
    
    console.print(table)
    
    # Calculate ensemble insights
    forecast_change = ((forecast['yhat'].iloc[-1] - last_cpi) / last_cpi) * 100
    annualized_inflation = ((forecast['yhat'].iloc[-1] / last_cpi) ** (12/1) - 1) * 100
    
    console.print(f"\n[bold]📈 Ensemble Insights:[/bold]")
    console.print(f"  • Current CPI level: {last_cpi:.2f}")
    console.print(f"  • 12-month inflation forecast: {forecast_change:.2f}%")
    console.print(f"  • Annualized inflation rate: {annualized_inflation:.2f}%")
    console.print(f"  • Model ensemble: ARIMA + Prophet + Random Forest")
    
    return forecast

def demo_var_forecasting():
    """Demonstrate VAR multivariate forecasting with sample data."""
    console.print("\n[bold yellow]🔗 VAR Multivariate Forecasting Demo[/bold yellow]")
    console.print("=" * 60)
    
    # Create sample data
    data = create_sample_data()
    indicators = ['CPIAUCSL', 'UNRATE', 'INDPRO']
    multivariate_data = data[indicators]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Generating VAR forecast...", total=None)
        
        # Generate VAR forecast
        forecast = var_forecast(multivariate_data, steps=12)
        
        progress.update(task, description="VAR forecast completed")
    
    # Display results
    console.print(f"\n[bold]📊 VAR Forecast Results (12 months)[/bold]")
    
    for indicator in indicators:
        console.print(f"\n[bold]Indicator: {indicator}[/bold]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Date", style="cyan")
        table.add_column("Forecast", style="green")
        table.add_column("Change", style="blue")
        
        last_value = multivariate_data[indicator].iloc[-1]
        
        for date, row in forecast.head(6).iterrows():
            change = ((row[indicator] - last_value) / last_value) * 100
            table.add_row(
                date.strftime('%Y-%m-%d'),
                f"{row[indicator]:.2f}",
                f"{change:+.2f}%"
            )
        
        console.print(table)
    
    console.print(f"\n[bold]📈 VAR Insights:[/bold]")
    console.print(f"  • Analyzing {len(indicators)} interrelated indicators")
    console.print(f"  • Captures dynamic relationships between variables")
    console.print(f"  • Uses vector autoregression with optimal lag selection")
    
    return forecast

def demo_risk_analysis():
    """Demonstrate economic risk analysis with sample data."""
    console.print("\n[bold red]⚠️  Economic Risk Analysis Demo[/bold red]")
    console.print("=" * 60)
    
    # Create sample data
    data = create_sample_data()
    returns_data = data.pct_change().dropna()
    
    # Initialize risk modeler
    risk_modeler = EconomicRiskModeler(confidence_level=0.95)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Calculating Value at Risk...", total=None)
        
        # Calculate VaR
        var_results = risk_modeler.calculate_economic_var(returns_data)
        
        task = progress.add_task("Running stress tests...", total=None)
        
        # Stress testing
        current_values = data.iloc[-1].to_dict()
        stress_results = risk_modeler.stress_test(current_values, 'recession')
        
        progress.update(task, description="Risk analysis completed")
    
    # Display risk results
    console.print(f"\n[bold]📊 Economic Risk Analysis Results[/bold]")
    
    # VaR Results
    if 'portfolio_var' in var_results:
        var = var_results['portfolio_var']
        console.print(f"\n[bold]Value at Risk (95% confidence):[/bold]")
        console.print(f"  • Portfolio VaR: {var['var']:.2%}")
        console.print(f"  • Conditional VaR: {var['cvar']:.2%}")
        console.print(f"  • Diversification Benefit: {var.get('diversification_benefit', 0):.2%}")
    
    # Individual VaRs
    console.print(f"\n[bold]Individual Indicator VaR:[/bold]")
    for indicator, var_data in var_results.get('individual_vars', {}).items():
        console.print(f"  • {indicator}: {var_data['var']:.2%}")
    
    # Stress Test Results
    console.print(f"\n[bold]Recession Stress Test:[/bold]")
    console.print(f"  • Portfolio Impact: {stress_results.get('portfolio_impact', 0):.1%}")
    worst_factors = stress_results.get('worst_affected_factors', [])
    if worst_factors:
        console.print(f"  • Worst Affected: {worst_factors[0][0]}")
    else:
        console.print(f"  • Worst Affected: Analysis completed")
    
    console.print(f"\n[bold]📈 Risk Insights:[/bold]")
    console.print(f"  • Risk analysis based on {len(returns_data)} data points")
    console.print(f"  • VaR calculated at 95% confidence level")
    console.print(f"  • Stress test simulates recession scenario")
    
    return var_results, stress_results

def main():
    """Run the complete economic forecasting demo."""
    
    console.print(Panel(
        "[bold magenta]🏛️ Economic Forecasting System Demo[/bold magenta]\n\n"
        "Advanced econometric models, ensemble forecasting,\n"
        "multivariate analysis, and risk modeling",
        title="Institutional-Grade Economic Intelligence",
        border_style="magenta"
    ))
    
    console.print(f"\n[bold]Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/bold]")
    
    # Run all demos
    try:
        # Demo 1: Prophet forecasting
        prophet_results = demo_prophet_forecasting()
        
        # Demo 2: Ensemble forecasting
        ensemble_results = demo_ensemble_forecasting()
        
        # Demo 3: VAR multivariate forecasting
        var_results = demo_var_forecasting()
        
        # Demo 4: Risk analysis
        risk_results = demo_risk_analysis()
        
        # Summary
        console.print(Panel(
            "[bold green]✅ All demos completed successfully![/bold green]\n\n"
            "Features demonstrated:\n"
            "• Prophet time series forecasting with uncertainty bands\n"
            "• Ensemble model combination (ARIMA + Prophet + RF)\n"
            "• VAR multivariate analysis for interrelated indicators\n"
            "• Economic risk assessment with VaR and stress testing\n\n"
            "[bold yellow]🎯 Your system demonstrates institutional-grade capabilities![/bold yellow]\n\n"
            "Next steps:\n"
            "• Configure FRED API key for real economic data\n"
            "• Try advanced models: VECM, Bayesian VAR, Markov-Switching\n"
            "• Enable AI narratives with OpenAI API key\n"
            "• Launch interactive dashboard: streamlit run notebooks/economic_dashboard.py",
            title="Demo Summary",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"\n[red]Demo error: {str(e)}[/red]")
        import traceback
        console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

if __name__ == "__main__":
    main()
