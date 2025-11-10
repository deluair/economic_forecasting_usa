import sys
from pathlib import Path
import typer
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import pandas as pd
import numpy as np

# Ensure `src` is on path when running the script directly
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from usa_econ.config import load_config
from usa_econ.pipeline.economic_analyzer import EconomicAnalyzer
from usa_econ.models.advanced_econometrics import (
    vecm_forecast, bayesian_var_forecast, markov_switching_forecast,
    dynamic_factor_forecast, unobserved_components_forecast,
    nowcast_economy, structural_break_analysis
)
from usa_econ.data_sources.realtime_data import RealTimeDataManager
from usa_econ.pipeline.ai_narrative_generator import EconomicNarrativeGenerator
from usa_econ.models.risk_modeling import EconomicRiskModeler
from usa_econ.utils.io import save_df_csv, ensure_dir


app = typer.Typer(help="Advanced US Economic Analysis CLI - Institutional Grade")
console = Console()


@app.command()
def advanced_forecast(
    indicator: str = typer.Argument(..., help="Economic indicator to forecast"),
    model: str = typer.Option(
        "ensemble", 
        help="Model type: vecm, bayesian_var, markov_switching, dynamic_factor, unobserved_components, ensemble"
    ),
    steps: int = typer.Option(12, help="Forecast horizon in periods"),
    confidence: float = typer.Option(0.95, help="Confidence level for intervals"),
    save_results: bool = typer.Option(True, help="Save forecast results")
):
    """Generate advanced economic forecasts using cutting-edge models."""
    
    console.print(f"[bold blue]🔬 Advanced Economic Forecast: {indicator}[/bold blue]")
    console.print("=" * 60)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Load data
        task = progress.add_task("Fetching economic data...", total=None)
        analyzer = EconomicAnalyzer()
        data = analyzer.fetch_latest_data([indicator])
        
        if indicator not in data.columns:
            console.print(f"[red]Indicator '{indicator}' not available[/red]")
            raise typer.Exit(1)
        
        series_data = data[indicator].dropna()
        progress.update(task, description="Data loaded successfully")
        
        # Generate forecast based on model type
        task = progress.add_task(f"Running {model} forecast...", total=None)
        
        try:
            if model == "vecm":
                # Need multiple series for VECM
                additional_indicators = ['CPI', 'UNRATE', 'INDPRO']
                multi_data = analyzer.fetch_latest_data([indicator] + additional_indicators)
                forecast = vecm_forecast(multi_data.dropna(), steps=steps)
                
            elif model == "bayesian_var":
                # Need multiple series for BVAR
                additional_indicators = ['CPI', 'UNRATE', 'FEDFUNDS']
                multi_data = analyzer.fetch_latest_data([indicator] + additional_indicators)
                forecast = bayesian_var_forecast(multi_data.dropna(), steps=steps)
                
            elif model == "markov_switching":
                forecast = markov_switching_forecast(series_data, steps=steps)
                
            elif model == "dynamic_factor":
                # Need multiple series for Dynamic Factor
                additional_indicators = ['CPI', 'UNRATE', 'HOUST']
                multi_data = analyzer.fetch_latest_data([indicator] + additional_indicators)
                forecast = dynamic_factor_forecast(multi_data.dropna(), steps=steps)
                
            elif model == "unobserved_components":
                forecast = unobserved_components_forecast(series_data, steps=steps)
                
            else:
                # Default to ensemble
                from usa_econ.models.ensemble import ensemble_forecast
                forecast = ensemble_forecast(series_data, steps=steps)
            
            progress.update(task, description="Forecast completed successfully")
            
        except Exception as e:
            console.print(f"[red]Forecast generation failed: {e}[/red]")
            raise typer.Exit(1)
    
    # Display results
    _display_advanced_forecast(forecast, indicator, model)
    
    # Save results
    if save_results:
        output_dir = Path("data/processed/advanced_forecasts")
        ensure_dir(output_dir)
        output_path = output_dir / f"{model}_{indicator}_forecast.csv"
        save_df_csv(forecast, output_path)
        console.print(f"[green]✓ Forecast saved to: {output_path}[/green]")


@app.command()
def nowcast(
    indicators: list[str] = typer.Option(
        ["GDP", "CPI", "UNRATE", "INDPRO"], 
        help="Indicators for nowcasting"
    )
):
    """Generate real-time economic nowcasts using mixed-frequency data."""
    
    console.print("[bold blue]⚡ Real-Time Economic Nowcasting[/bold blue]")
    console.print("=" * 50)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Get real-time data
        task = progress.add_task("Fetching real-time data...", total=None)
        rt_manager = RealTimeDataManager()
        realtime_data = rt_manager.get_real_time_indicators()
        progress.update(task, description="Real-time data loaded")
        
        # Get historical data
        task = progress.add_task("Loading historical data...", total=None)
        analyzer = EconomicAnalyzer()
        historical_data = analyzer.fetch_latest_data(indicators, start_date="2020-01-01")
        progress.update(task, description="Historical data loaded")
        
        # Generate nowcasts
        task = progress.add_task("Generating nowcasts...", total=None)
        nowcasts = {}
        
        for indicator in indicators:
            if indicator in historical_data.columns:
                try:
                    nowcast = nowcast_economy(
                        {indicator: historical_data[indicator]},
                        indicator
                    )
                    nowcasts[indicator] = nowcast
                except Exception as e:
                    console.print(f"[yellow]Warning: Nowcast failed for {indicator}: {e}[/yellow]")
        
        progress.update(task, description="Nowcasting completed")
    
    # Display nowcast results
    _display_nowcast_results(nowcasts, realtime_data)


@app.command()
def risk_analysis(
    indicators: list[str] = typer.Option(
        ["GDP", "CPI", "UNRATE", "FEDFUNDS"], 
        help="Indicators for risk analysis"
    ),
    confidence_level: float = typer.Option(0.95, help="Confidence level for VaR"),
    stress_scenarios: list[str] = typer.Option(
        ["recession", "stagflation", "financial_crisis"], 
        help="Stress scenarios to test"
    )
):
    """Comprehensive economic risk analysis and stress testing."""
    
    console.print("[bold red]⚠️  Economic Risk Analysis[/bold red]")
    console.print("=" * 50)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Load data
        task = progress.add_task("Loading economic data...", total=None)
        analyzer = EconomicAnalyzer()
        data = analyzer.fetch_latest_data(indicators, start_date="2010-01-01")
        returns_data = data.pct_change().dropna()
        progress.update(task, description="Data loaded successfully")
        
        # Initialize risk modeler
        task = progress.add_task("Initializing risk models...", total=None)
        risk_modeler = EconomicRiskModeler(confidence_level)
        progress.update(task, description="Risk models initialized")
        
        # Calculate VaR
        task = progress.add_task("Calculating Value at Risk...", total=None)
        var_results = risk_modeler.calculate_economic_var(returns_data)
        progress.update(task, description="VaR analysis completed")
        
        # Stress testing
        task = progress.add_task("Running stress tests...", total=None)
        current_values = data.iloc[-1].to_dict()
        stress_results = {}
        
        for scenario in stress_scenarios:
            stress_results[scenario] = risk_modeler.stress_test(current_values, scenario)
        
        progress.update(task, description="Stress testing completed")
        
        # Network analysis
        task = progress.add_task("Analyzing systemic risk...", total=None)
        correlation_matrix = returns_data.corr()
        network_analysis = risk_modeler.network_risk_analysis(correlation_matrix)
        progress.update(task, description="Network analysis completed")
    
    # Display risk results
    _display_risk_analysis(var_results, stress_results, network_analysis)
    
    # Save risk report
    output_dir = Path("data/processed/risk_reports")
    ensure_dir(output_dir)
    
    risk_report = {
        'var_results': var_results,
        'stress_results': stress_results,
        'network_analysis': network_analysis,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    report_path = output_dir / f"risk_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    import json
    with open(report_path, 'w') as f:
        json.dump(risk_report, f, indent=2, default=str)
    
    console.print(f"[green]✓ Risk report saved to: {report_path}[/green]")


@app.command()
def structural_breaks(
    indicator: str = typer.Argument(..., help="Indicator to analyze for structural breaks"),
    max_breaks: int = typer.Option(5, help="Maximum number of breaks to detect")
):
    """Detect structural breaks in economic time series."""
    
    console.print(f"[bold blue]🔍 Structural Break Analysis: {indicator}[/bold blue]")
    console.print("=" * 60)
    
    # Load data
    analyzer = EconomicAnalyzer()
    data = analyzer.fetch_latest_data([indicator], start_date="1990-01-01")
    
    if indicator not in data.columns:
        console.print(f"[red]Indicator '{indicator}' not available[/red]")
        raise typer.Exit(1)
    
    series_data = data[indicator].dropna()
    
    # Detect structural breaks
    break_analysis = structural_break_analysis(series_data, max_breaks)
    
    # Display results
    _display_structural_breaks(break_analysis, indicator)
    
    # Save results
    output_dir = Path("data/processed/structural_breaks")
    ensure_dir(output_dir)
    
    break_data = []
    for segment in break_analysis['segments']:
        break_data.append({
            'start_date': segment['start_date'],
            'end_date': segment['end_date'],
            'mean': segment['mean'],
            'growth_rate': segment['growth_rate']
        })
    
    break_df = pd.DataFrame(break_data)
    output_path = output_dir / f"breaks_{indicator}.csv"
    save_df_csv(break_df, output_path)
    
    console.print(f"[green]✓ Break analysis saved to: {output_path}[/green]")


@app.command()
def ai_narrative(
    indicators: list[str] = typer.Option(
        ["GDP", "CPI", "UNRATE", "FEDFUNDS"], 
        help="Indicators for narrative generation"
    ),
    use_openai: bool = typer.Option(False, help="Use OpenAI for narrative generation"),
    save_report: bool = typer.Option(True, help="Save narrative report")
):
    """Generate AI-powered economic narratives and insights."""
    
    console.print("[bold green]🤖 AI Economic Narrative Generator[/bold green]")
    console.print("=" * 60)
    
    # Get economic data
    analyzer = EconomicAnalyzer()
    data = analyzer.fetch_latest_data(indicators)
    
    # Calculate economic signals
    signals = analyzer.calculate_economic_signals(data)
    
    # Prepare economic data for narrative
    economic_data = {}
    for indicator in indicators:
        if indicator in data.columns:
            series_data = data[indicator].dropna()
            if len(series_data) > 0:
                economic_data[f'{indicator.lower()}_growth'] = series_data.pct_change(12).iloc[-1] if len(series_data) > 12 else series_data.pct_change().iloc[-1]
                economic_data[f'{indicator.lower()}_level'] = series_data.iloc[-1]
    
    # Add derived indicators
    economic_data.update({
        'gdp_growth': economic_data.get('gdp_growth', 0.025),
        'inflation_rate': economic_data.get('cpi_growth', 0.025),
        'unemployment_rate': economic_data.get('unemployment_level', 0.05),
        'recession_probability': 0.25  # Placeholder
    })
    
    # Initialize narrative generator
    openai_key = None  # Set in environment or config if needed
    narrative_gen = EconomicNarrativeGenerator(openai_key)
    
    # Generate comprehensive report
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Generating economic narratives...", total=None)
        report = narrative_gen.generate_comprehensive_report(economic_data)
        progress.update(task, description="Narratives generated successfully")
    
    # Display report
    _display_ai_narrative(report)
    
    # Save report
    if save_report:
        output_dir = Path("data/processed/narrative_reports")
        ensure_dir(output_dir)
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        # Save executive summary
        summary_path = output_dir / f"executive_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write(report['executive_summary'])
        
        # Save full report
        full_report_path = output_dir / f"full_report_{timestamp}.txt"
        with open(full_report_path, 'w') as f:
            f.write("EXECUTIVE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(report['executive_summary'] + "\n\n")
            
            f.write("KEY INSIGHTS\n")
            f.write("=" * 50 + "\n\n")
            for insight in report['key_insights']:
                f.write(f"• {insight}\n")
            f.write("\n")
            
            f.write("DETAILED ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            for section_name, content in report['sections'].items():
                f.write(f"{section_name.upper()}\n")
                f.write("-" * 30 + "\n\n")
                f.write(content + "\n\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("=" * 50 + "\n\n")
            f.write(report['recommendations'] + "\n\n")
            
            f.write("RISK ASSESSMENT\n")
            f.write("=" * 50 + "\n\n")
            f.write(report['risk_assessment'] + "\n")
        
        console.print(f"[green]✓ Narrative report saved to: {full_report_path}[/green]")


@app.command()
def comprehensive_analysis(
    indicators: list[str] = typer.Option(
        ["GDP", "CPI", "UNRATE", "FEDFUNDS", "INDPRO"], 
        help="Indicators for comprehensive analysis"
    ),
    forecast_steps: int = typer.Option(12, help="Forecast horizon"),
    include_risk: bool = typer.Option(True, help="Include risk analysis"),
    include_narrative: bool = typer.Option(True, help="Include AI narrative")
):
    """Generate comprehensive institutional-grade economic analysis."""
    
    console.print("[bold magenta]🏛️  Comprehensive Economic Analysis[/bold magenta]")
    console.print("=" * 70)
    
    analysis_results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # 1. Economic signals and basic analysis
        task = progress.add_task("Analyzing economic signals...", total=None)
        analyzer = EconomicAnalyzer()
        data = analyzer.fetch_latest_data(indicators)
        signals = analyzer.calculate_economic_signals(data)
        cycle_assessment = analyzer.assess_business_cycle(data)
        progress.update(task, description="Economic signals analyzed")
        
        # 2. Advanced forecasting
        task = progress.add_task("Generating advanced forecasts...", total=None)
        forecasts = {}
        
        # Ensemble forecast for main indicator
        main_indicator = indicators[0]
        if main_indicator in data.columns:
            from usa_econ.models.ensemble import ensemble_forecast
            forecasts[main_indicator] = ensemble_forecast(
                data[main_indicator].dropna(), 
                steps=forecast_steps
            )
        
        progress.update(task, description="Forecasts generated")
        
        # 3. Risk analysis (if requested)
        risk_results = {}
        if include_risk:
            task = progress.add_task("Performing risk analysis...", total=None)
            risk_modeler = EconomicRiskModeler()
            returns_data = data.pct_change().dropna()
            
            risk_results['var'] = risk_modeler.calculate_economic_var(returns_data)
            risk_results['stress'] = risk_modeler.stress_test(data.iloc[-1].to_dict(), 'recession')
            
            progress.update(task, description="Risk analysis completed")
        
        # 4. AI narrative (if requested)
        narrative_results = {}
        if include_narrative:
            task = progress.add_task("Generating AI narratives...", total=None)
            
            economic_data = {
                'gdp_growth': data['GDP'].pct_change(4).iloc[-1] if 'GDP' in data.columns else 0.025,
                'inflation_rate': data['CPI'].pct_change(12).iloc[-1] if 'CPI' in data.columns else 0.025,
                'unemployment_rate': data['UNRATE'].iloc[-1] if 'UNRATE' in data.columns else 0.05,
                'recession_probability': cycle_assessment['recession_probability']
            }
            
            narrative_gen = EconomicNarrativeGenerator()
            narrative_results = narrative_gen.generate_comprehensive_report(economic_data, forecasts)
            
            progress.update(task, description="Narratives generated")
        
        # 5. Real-time data integration
        task = progress.add_task("Integrating real-time data...", total=None)
        rt_manager = RealTimeDataManager()
        realtime_indicators = rt_manager.get_real_time_indicators()
        progress.update(task, description="Real-time data integrated")
    
    # Compile results
    analysis_results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'economic_signals': signals,
        'business_cycle': cycle_assessment,
        'forecasts': forecasts,
        'risk_analysis': risk_results,
        'narrative': narrative_results,
        'realtime_data': realtime_indicators
    }
    
    # Display comprehensive summary
    _display_comprehensive_analysis(analysis_results)
    
    # Save comprehensive report
    output_dir = Path("data/processed/comprehensive_analysis")
    ensure_dir(output_dir)
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    
    # Save as JSON
    import json
    json_path = output_dir / f"comprehensive_analysis_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    console.print(f"\n[green]✓ Comprehensive analysis saved to: {json_path}[/green]")


def _display_advanced_forecast(forecast: pd.DataFrame, indicator: str, model: str):
    """Display advanced forecast results."""
    
    console.print(f"\n[bold]📊 {model.upper()} Forecast Results for {indicator}[/bold]")
    
    # Create forecast table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Date", style="cyan")
    table.add_column("Forecast", style="green")
    table.add_column("Lower Bound", style="yellow")
    table.add_column("Upper Bound", style="yellow")
    
    for date, row in forecast.head(12).iterrows():  # Show first 12 periods
        if 'yhat' in row:
            table.add_row(
                date.strftime('%Y-%m-%d'),
                f"{row['yhat']:.2f}",
                f"{row.get('lower', 'N/A'):.2f}" if 'lower' in row else "N/A",
                f"{row.get('upper', 'N/A'):.2f}" if 'upper' in row else "N/A"
            )
        elif indicator in row:
            table.add_row(
                date.strftime('%Y-%m-%d'),
                f"{row[indicator]:.2f}",
                f"{row.get(f'{indicator}_lower', 'N/A'):.2f}" if f'{indicator}_lower' in row else "N/A",
                f"{row.get(f'{indicator}_upper', 'N/A'):.2f}" if f'{indicator}_upper' in row else "N/A"
            )
    
    console.print(table)


def _display_nowcast_results(nowcasts: Dict[str, pd.DataFrame], realtime_data: Dict[str, Any]):
    """Display nowcast results."""
    
    console.print("\n[bold]⚡ Real-Time Nowcast Results[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Indicator", style="cyan")
    table.add_column("Latest Value", style="green")
    table.add_column("Nowcast", style="yellow")
    table.add_column("Change", style="blue")
    table.add_column("Assessment", style="magenta")
    
    for indicator, nowcast in nowcasts.items():
        if not nowcast.empty:
            latest = nowcast['latest_value'].iloc[0]
            nowcast_val = nowcast['yhat'].iloc[0]
            change = nowcast['nowcast_change'].iloc[0]
            assessment = nowcast['assessment'].iloc[0]
            
            table.add_row(
                indicator,
                f"{latest:.2f}",
                f"{nowcast_val:.2f}",
                f"{change:.1%}",
                assessment
            )
    
    console.print(table)


def _display_risk_analysis(var_results: Dict, stress_results: Dict, network_analysis: Dict):
    """Display risk analysis results."""
    
    console.print("\n[bold red]⚠️  Risk Analysis Summary[/bold red]")
    
    # VaR Results
    if 'portfolio_var' in var_results:
        var = var_results['portfolio_var']
        console.print(f"\n[bold]Portfolio VaR ({var['confidence_level']:.0%} confidence):[/bold]")
        console.print(f"  Value at Risk: {var['var']:.2%}")
        console.print(f"  Conditional VaR: {var['cvar']:.2%}")
        console.print(f"  Diversification Benefit: {var.get('diversification_benefit', 0):.2%}")
    
    # Stress Test Results
    console.print(f"\n[bold]Stress Test Results:[/bold]")
    for scenario, result in stress_results.items():
        impact = result.get('portfolio_impact', 0)
        console.print(f"  {scenario.title()}: {impact:.1%} portfolio impact")
    
    # Network Analysis
    if 'network_metrics' in network_analysis:
        metrics = network_analysis['network_metrics']
        console.print(f"\n[bold]Systemic Risk Metrics:[/bold]")
        console.print(f"  Network Density: {metrics.get('density', 0):.3f}")
        console.print(f"  Systemic Risk Score: {network_analysis.get('systemic_risk_score', 0):.1f}/100")


def _display_structural_breaks(break_analysis: Dict, indicator: str):
    """Display structural break analysis."""
    
    console.print(f"\n[bold]🔍 Structural Break Analysis for {indicator}[/bold]")
    console.print(f"Number of breaks detected: {break_analysis['n_breaks']}")
    
    if break_analysis['break_detected']:
        console.print("\n[bold]Break Dates:[/bold]")
        for i, break_date in enumerate(break_analysis['break_dates']):
            console.print(f"  Break {i+1}: {break_date.strftime('%Y-%m-%d')}")
        
        console.print("\n[bold]Segment Analysis:[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Period", style="cyan")
        table.add_column("Growth Rate", style="green")
        table.add_column("Mean Value", style="yellow")
        
        for segment in break_analysis['segments']:
            period = f"{segment['start_date'].strftime('%Y-%m')} to {segment['end_date'].strftime('%Y-%m')}"
            table.add_row(
                period,
                f"{segment['growth_rate']:.1%}",
                f"{segment['mean']:.2f}"
            )
        
        console.print(table)


def _display_ai_narrative(report: Dict):
    """Display AI-generated narrative."""
    
    console.print("\n[bold green]🤖 AI-Generated Economic Narrative[/bold green]")
    
    # Executive Summary
    console.print(Panel(
        report['executive_summary'],
        title="Executive Summary",
        border_style="green"
    ))
    
    # Key Insights
    console.print("\n[bold]💡 Key Insights:[/bold]")
    for insight in report['key_insights']:
        console.print(f"  • {insight}")
    
    # Sample sections
    if 'sections' in report and len(report['sections']) > 0:
        first_section = list(report['sections'].keys())[0]
        console.print(f"\n[bold]{first_section.title()}:[/bold]")
        console.print(report['sections'][first_section][:500] + "...")


def _display_comprehensive_analysis(results: Dict):
    """Display comprehensive analysis summary."""
    
    console.print("\n[bold magenta]🏛️  Comprehensive Economic Analysis Summary[/bold magenta]")
    
    # Business Cycle
    cycle = results['business_cycle']
    console.print(f"\n[bold]Business Cycle:[/bold] {cycle['cycle_phase']}")
    console.print(f"Recession Probability: {cycle['recession_probability']:.1%}")
    
    # Economic Signals
    console.print(f"\n[bold]Key Economic Signals:[/bold]")
    signals = results['economic_signals']
    
    for category, signal_data in list(signals.items())[:3]:  # Show top 3
        signal = signal_data.get('signal', 'N/A')
        console.print(f"  {category}: {signal}")
    
    # Risk Summary
    if 'risk_analysis' in results and results['risk_analysis']:
        risk = results['risk_analysis']
        if 'portfolio_var' in risk.get('var', {}):
            var = risk['var']['portfolio_var']
            console.print(f"\n[bold]Risk Metrics:[/bold]")
            console.print(f"  Portfolio VaR: {var['var']:.2%}")
        
        if 'stress' in risk:
            stress_impact = risk['stress'].get('portfolio_impact', 0)
            console.print(f"  Recession Stress Impact: {stress_impact:.1%}")
    
    # Forecast Summary
    if 'forecasts' in results and results['forecasts']:
        console.print(f"\n[bold]Forecast Summary:[/bold]")
        for indicator, forecast in list(results['forecasts'].items())[:2]:  # Show top 2
            if not forecast.empty and 'yhat' in forecast.columns:
                next_period = forecast['yhat'].iloc[0]
                console.print(f"  {indicator} next period: {next_period:.2f}")
    
    # AI Insights
    if 'narrative' in results and results['narrative']:
        narrative = results['narrative']
        if 'key_insights' in narrative:
            console.print(f"\n[bold]AI-Generated Insights:[/bold]")
            for insight in narrative['key_insights'][:3]:  # Show top 3
                console.print(f"  • {insight}")


if __name__ == "__main__":
    app()
