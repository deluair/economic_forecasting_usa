#!/usr/bin/env python3
"""
Institutional-Grade Economic Forecasting System Demo
===============================================

This demo showcases the advanced economic forecasting capabilities
including Prophet, Ensemble, VAR, and risk modeling features.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[0] / "src"))

from usa_econ.config import load_config
from usa_econ.pipeline.economic_analyzer import EconomicAnalyzer
from usa_econ.models.prophet_model import prophet_forecast
from usa_econ.models.ensemble import ensemble_forecast
from usa_econ.models.var import var_forecast
from usa_econ.models.risk_modeling import EconomicRiskModeler
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def demo_prophet_forecasting():
    """Demonstrate Prophet forecasting capabilities."""
    console.print("\n[bold blue]🔮 Prophet Economic Forecasting Demo[/bold blue]")
    console.print("=" * 60)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Initializing Prophet forecast...", total=None)
        
        # Initialize analyzer
        analyzer = EconomicAnalyzer()
        
        # Fetch unemployment data
        task = progress.add_task("Fetching unemployment data...", total=None)
        data = analyzer.fetch_latest_data(['UNRATE'], start_date="2010-01-01")
        
        if 'UNRATE' not in data.columns:
            console.print("[red]Error: Could not fetch unemployment data[/red]")
            return
        
        unemployment_data = data['UNRATE'].dropna()
        
        # Generate Prophet forecast
        task = progress.add_task("Generating Prophet forecast...", total=None)
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
    """Demonstrate ensemble forecasting capabilities."""
    console.print("\n[bold green]🎯 Ensemble Economic Forecasting Demo[/bold green]")
    console.print("=" * 60)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Initialize analyzer
        analyzer = EconomicAnalyzer()
        
        # Fetch CPI data
        task = progress.add_task("Fetching CPI data...", total=None)
        data = analyzer.fetch_latest_data(['CPIAUCSL'], start_date="2010-01-01")
        
        if 'CPIAUCSL' not in data.columns:
            console.print("[red]Error: Could not fetch CPI data[/red]")
            return
        
        cpi_data = data['CPIAUCSL'].dropna()
        
        # Generate ensemble forecast
        task = progress.add_task("Generating ensemble forecast (ARIMA + Prophet + RF)...", total=None)
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
    """Demonstrate VAR multivariate forecasting."""
    console.print("\n[bold yellow]🔗 VAR Multivariate Forecasting Demo[/bold yellow]")
    console.print("=" * 60)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Initialize analyzer
        analyzer = EconomicAnalyzer()
        
        # Fetch multiple indicators
        task = progress.add_task("Fetching multivariate data...", total=None)
        indicators = ['CPIAUCSL', 'UNRATE', 'INDPRO']
        data = analyzer.fetch_latest_data(indicators, start_date="2010-01-01")
        
        # Check data availability
        available_indicators = [col for col in indicators if col in data.columns]
        if len(available_indicators) < 2:
            console.print("[red]Error: Need at least 2 indicators for VAR analysis[/red]")
            return
        
        multivariate_data = data[available_indicators].dropna()
        
        # Generate VAR forecast
        task = progress.add_task("Generating VAR forecast...", total=None)
        forecast = var_forecast(multivariate_data, steps=12)
        
        progress.update(task, description="VAR forecast completed")
    
    # Display results
    console.print(f"\n[bold]📊 VAR Forecast Results (12 months)[/bold]")
    
    for indicator in available_indicators:
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
    console.print(f"  • Analyzing {len(available_indicators)} interrelated indicators")
    console.print(f"  • Captures dynamic relationships between variables")
    console.print(f"  • Uses vector autoregression with optimal lag selection")
    
    return forecast

def demo_risk_analysis():
    """Demonstrate economic risk analysis capabilities."""
    console.print("\n[bold red]⚠️  Economic Risk Analysis Demo[/bold red]")
    console.print("=" * 60)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Initialize analyzer and risk modeler
        analyzer = EconomicAnalyzer()
        risk_modeler = EconomicRiskModeler(confidence_level=0.95)
        
        # Fetch data
        task = progress.add_task("Fetching economic data for risk analysis...", total=None)
        indicators = ['CPIAUCSL', 'UNRATE', 'INDPRO']
        data = analyzer.fetch_latest_data(indicators, start_date="2015-01-01")
        
        available_indicators = [col for col in indicators if col in data.columns]
        if len(available_indicators) < 2:
            console.print("[red]Error: Insufficient data for risk analysis[/red]")
            return
        
        returns_data = data[available_indicators].pct_change().dropna()
        
        # Calculate VaR
        task = progress.add_task("Calculating Value at Risk...", total=None)
        var_results = risk_modeler.calculate_economic_var(returns_data)
        
        # Stress testing
        task = progress.add_task("Running stress tests...", total=None)
        current_values = data[available_indicators].iloc[-1].to_dict()
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
    console.print(f"  • Worst Affected: {stress_results.get('worst_affected_factors', [('', 0)])[0][0]}")
    
    console.print(f"\n[bold]📈 Risk Insights:[/bold]")
    console.print(f"  • Risk analysis based on {len(returns_data)} data points")
    console.print(f"  • VaR calculated at 95% confidence level")
    console.print(f"  • Stress test simulates recession scenario")
    
    return var_results, stress_results

def demo_economic_signals():
    """Demonstrate economic signal generation."""
    console.print("\n[bold cyan]📡 Economic Signal Generation Demo[/bold cyan]")
    console.print("=" * 60)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Initialize analyzer
        analyzer = EconomicAnalyzer()
        
        # Fetch data
        task = progress.add_task("Fetching economic indicators...", total=None)
        indicators = ['GDP', 'CPIAUCSL', 'UNRATE', 'FEDFUNDS', 'INDPRO']
        data = analyzer.fetch_latest_data(indicators, start_date="2015-01-01")
        
        # Generate signals
        task = progress.add_task("Generating economic signals...", total=None)
        signals = analyzer.calculate_economic_signals(data)
        cycle_assessment = analyzer.assess_business_cycle(data)
        
        progress.update(task, description="Signal generation completed")
    
    # Display signals
    console.print(f"\n[bold]📊 Economic Signals Dashboard[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Indicator", style="cyan", width=20)
    table.add_column("Signal", style="green", width=25)
    table.add_column("Trend", width=15)
    
    for category, signal_data in signals.items():
        signal = signal_data.get('signal', 'N/A')
        trend = signal_data.get('trend', 'N/A')
        table.add_row(category, signal, trend)
    
    console.print(table)
    
    # Business cycle assessment
    console.print(f"\n[bold]🔄 Business Cycle Assessment:[/bold]")
    console.print(f"  • Current Phase: {cycle_assessment['cycle_phase']}")
    console.print(f"  • Recession Probability: {cycle_assessment['recession_probability']:.1%}")
    
    console.print(f"\n[bold]📈 Signal Insights:[/bold]")
    console.print(f"  • Generated {len(signals)} economic signals")
    console.print(f"  • Based on real-time economic data analysis")
    console.print(f"  • Includes trend and momentum indicators")
    
    return signals, cycle_assessment

def main():
    """Run the complete economic forecasting demo."""
    
    console.print(Panel(
        "[bold magenta]🏛️ Institutional-Grade Economic Forecasting System[/bold magenta]\n\n"
        "Advanced econometric models, real-time data integration,\n"
        "AI-powered analysis, and sophisticated risk modeling",
        title="Economic Intelligence Platform",
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
        
        # Demo 5: Economic signals
        signal_results = demo_economic_signals()
        
        # Summary
        console.print(Panel(
            "[bold green]✅ Demo completed successfully![/bold green]\n\n"
            "Features demonstrated:\n"
            "• Prophet time series forecasting\n"
            "• Ensemble model combination\n"
            "• VAR multivariate analysis\n"
            "• Economic risk assessment\n"
            "• Real-time signal generation\n\n"
            "[bold]Your system is ready for institutional-grade economic analysis![/bold]",
            title="Demo Summary",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"\n[red]Demo error: {str(e)}[/red]")
        console.print("Please check your API keys and internet connection.")

if __name__ == "__main__":
    main()
