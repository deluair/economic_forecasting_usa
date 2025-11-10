from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from ..config import load_config
from ..data_sources.fred import get_series as fred_get_series
from ..models.ensemble import ensemble_forecast
from ..models.evaluation import compare_models
from ..utils.io import save_df_csv, ensure_dir


class EconomicAnalyzer:
    """Comprehensive economic analysis and forecasting toolkit."""
    
    def __init__(self, config=None):
        """Initialize the economic analyzer."""
        self.config = config or load_config()
        self.indicators = {
            "GDP": "GDP",
            "CPI": "CPIAUCSL", 
            "Unemployment Rate": "UNRATE",
            "Industrial Production": "INDPRO",
            "Housing Starts": "HOUST",
            "Retail Sales": "RSXFS",
            "10-Year Treasury": "DGS10",
            "Federal Funds Rate": "FEDFUNDS",
            "Consumer Confidence": "UMCSENT",
            "PMI": "NAPMI",
            "Durable Goods": "DGORDER",
            "Personal Income": "PI"
        }
    
    def fetch_latest_data(self, indicators: List[str] = None, start_date: str = "2015-01-01") -> pd.DataFrame:
        """Fetch latest economic data for specified indicators."""
        if indicators is None:
            indicators = list(self.indicators.keys())
        
        data_dict = {}
        
        for indicator in indicators:
            if indicator not in self.indicators:
                continue
                
            series_id = self.indicators[indicator]
            try:
                df = fred_get_series(series_id, self.config, start=start_date)
                data_dict[indicator] = df[series_id]
            except Exception as e:
                print(f"Warning: Could not fetch {indicator}: {e}")
        
        return pd.DataFrame(data_dict)
    
    def calculate_economic_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate economic signals and momentum indicators."""
        signals = {}
        
        # GDP Analysis
        if 'GDP' in data.columns:
            gdp_data = data['GDP'].dropna()
            if len(gdp_data) >= 4:
                gdp_growth_qoq = gdp_data.pct_change(1).iloc[-1]
                gdp_growth_yoy = gdp_data.pct_change(4).iloc[-1]
                
                signals['GDP'] = {
                    'growth_qoq': gdp_growth_qoq,
                    'growth_yoy': gdp_growth_yoy,
                    'signal': self._get_growth_signal(gdp_growth_yoy, threshold=0.02),
                    'trend': self._calculate_trend(gdp_data)
                }
        
        # Inflation Analysis
        if 'CPI' in data.columns:
            cpi_data = data['CPI'].dropna()
            if len(cpi_data) >= 12:
                cpi_mom = cpi_data.pct_change(1).iloc[-1]
                cpi_yoy = cpi_data.pct_change(12).iloc[-1]
                
                signals['Inflation'] = {
                    'mom_change': cpi_mom,
                    'yoy_change': cpi_yoy,
                    'signal': self._get_inflation_signal(cpi_yoy),
                    'trend': self._calculate_trend(cpi_data)
                }
        
        # Labor Market Analysis
        if 'Unemployment Rate' in data.columns:
            unemployment_data = data['Unemployment Rate'].dropna()
            if len(unemployment_data) >= 3:
                unemployment_level = unemployment_data.iloc[-1]
                unemployment_change = unemployment_data.diff(1).iloc[-1]
                
                signals['Labor Market'] = {
                    'unemployment_rate': unemployment_level,
                    'monthly_change': unemployment_change,
                    'signal': self._get_employment_signal(unemployment_level),
                    'trend': self._calculate_trend(unemployment_data, inverse=True)
                }
        
        # Interest Rate Analysis
        if 'Federal Funds Rate' in data.columns:
            fed_data = data['Federal Funds Rate'].dropna()
            if len(fed_data) >= 1:
                fed_rate = fed_data.iloc[-1]
                fed_change = fed_data.diff(1).iloc[-1]
                
                signals['Monetary Policy'] = {
                    'fed_funds_rate': fed_rate,
                    'monthly_change': fed_change,
                    'signal': self._get_monetary_signal(fed_rate),
                    'trend': self._calculate_trend(fed_data)
                }
        
        # Housing Market Analysis
        if 'Housing Starts' in data.columns:
            housing_data = data['Housing Starts'].dropna()
            if len(housing_data) >= 12:
                housing_yoy = housing_data.pct_change(12).iloc[-1]
                housing_mom = housing_data.pct_change(1).iloc[-1]
                
                signals['Housing Market'] = {
                    'yoy_change': housing_yoy,
                    'mom_change': housing_mom,
                    'signal': self._get_housing_signal(housing_yoy),
                    'trend': self._calculate_trend(housing_data)
                }
        
        # Consumer Confidence Analysis
        if 'Consumer Confidence' in data.columns:
            confidence_data = data['Consumer Confidence'].dropna()
            if len(confidence_data) >= 1:
                confidence_level = confidence_data.iloc[-1]
                confidence_change = confidence_data.diff(1).iloc[-1]
                
                signals['Consumer Sentiment'] = {
                    'confidence_index': confidence_level,
                    'monthly_change': confidence_change,
                    'signal': self._get_confidence_signal(confidence_level),
                    'trend': self._calculate_trend(confidence_data)
                }
        
        return signals
    
    def _get_growth_signal(self, growth_rate: float, threshold: float = 0.02) -> str:
        """Get growth signal based on rate."""
        if growth_rate > threshold:
            return "🟢 Strong Expansion"
        elif growth_rate > 0:
            return "🟡 Moderate Growth"
        elif growth_rate > -threshold:
            return "🟠 Mild Contraction"
        else:
            return "🔴 Strong Contraction"
    
    def _get_inflation_signal(self, inflation_rate: float) -> str:
        """Get inflation signal based on rate."""
        if inflation_rate > 0.04:
            return "🔴 High Inflation"
        elif inflation_rate > 0.025:
            return "🟡 Moderate Inflation"
        elif inflation_rate > 0.015:
            return "🟢 Low Inflation"
        else:
            return "🔵 Deflation Risk"
    
    def _get_employment_signal(self, unemployment_rate: float) -> str:
        """Get employment signal based on unemployment rate."""
        if unemployment_rate < 0.04:
            return "🟢 Tight Labor Market"
        elif unemployment_rate < 0.06:
            return "🟡 Healthy Labor Market"
        elif unemployment_rate < 0.08:
            return "🟠 Rising Unemployment"
        else:
            return "🔴 Weak Labor Market"
    
    def _get_monetary_signal(self, fed_rate: float) -> str:
        """Get monetary policy signal based on Fed funds rate."""
        if fed_rate > 0.04:
            return "🔴 Restrictive Policy"
        elif fed_rate > 0.025:
            return "🟡 Neutral Policy"
        elif fed_rate > 0.01:
            return "🟢 Accommodative Policy"
        else:
            return "🔵 Highly Accommodative"
    
    def _get_housing_signal(self, housing_growth: float) -> str:
        """Get housing market signal."""
        if housing_growth > 0.10:
            return "🟢 Strong Housing Market"
        elif housing_growth > 0:
            return "🟡 Moderate Housing Activity"
        elif housing_growth > -0.10:
            return "🟠 Housing Slowdown"
        else:
            return "🔴 Housing Market Weakness"
    
    def _get_confidence_signal(self, confidence_level: float) -> str:
        """Get consumer confidence signal."""
        if confidence_level > 100:
            return "🟢 High Confidence"
        elif confidence_level > 80:
            return "🟡 Moderate Confidence"
        elif confidence_level > 60:
            return "🟠 Low Confidence"
        else:
            return "🔴 Very Low Confidence"
    
    def _calculate_trend(self, data: pd.Series, window: int = 12, inverse: bool = False) -> str:
        """Calculate trend direction over specified window."""
        if len(data) < window:
            return "📊 Insufficient Data"
        
        recent_data = data.iloc[-window:]
        slope = np.polyfit(range(len(recent_data)), recent_data.values, 1)[0]
        
        if inverse:
            slope = -slope
        
        if slope > 0.01:
            return "📈 Strong Uptrend"
        elif slope > 0:
            return "📊 Moderate Uptrend"
        elif slope > -0.01:
            return "➡️ Sideways"
        else:
            return "📉 Downtrend"
    
    def generate_forecasts(self, data: pd.DataFrame, steps: int = 12) -> Dict[str, pd.DataFrame]:
        """Generate forecasts for key economic indicators."""
        forecasts = {}
        
        key_indicators = ['GDP', 'CPI', 'Unemployment Rate', 'Federal Funds Rate']
        
        for indicator in key_indicators:
            if indicator in data.columns:
                series_data = data[indicator].dropna()
                if len(series_data) >= 24:  # Need sufficient data
                    try:
                        forecast = ensemble_forecast(
                            series_data, 
                            steps=steps,
                            models=['arima', 'prophet', 'rf']
                        )
                        forecasts[indicator] = forecast
                    except Exception as e:
                        print(f"Warning: Forecast failed for {indicator}: {e}")
        
        return forecasts
    
    def assess_business_cycle(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess current business cycle phase."""
        cycle_signals = {}
        
        # Calculate recession probability indicators
        recession_indicators = []
        
        # GDP indicator
        if 'GDP' in data.columns:
            gdp_data = data['GDP'].dropna()
            if len(gdp_data) >= 2:
                gdp_contraction = gdp_data.pct_change(2).iloc[-1] < 0
                recession_indicators.append(gdp_contraction)
        
        # Unemployment indicator
        if 'Unemployment Rate' in data.columns:
            unemployment_data = data['Unemployment Rate'].dropna()
            if len(unemployment_data) >= 6:
                unemployment_rising = unemployment_data.diff(6).iloc[-1] > 0.001
                recession_indicators.append(unemployment_rising)
        
        # Industrial Production indicator
        if 'Industrial Production' in data.columns:
            ip_data = data['Industrial Production'].dropna()
            if len(ip_data) >= 3:
                ip_decline = ip_data.pct_change(3).iloc[-1] < -0.01
                recession_indicators.append(ip_decline)
        
        # Calculate recession probability
        if recession_indicators:
            recession_probability = sum(recession_indicators) / len(recession_indicators)
        else:
            recession_probability = 0.0
        
        # Determine cycle phase
        if recession_probability > 0.66:
            cycle_phase = "🔴 Recession"
        elif recession_probability > 0.33:
            cycle_phase = "🟡 Slowdown"
        else:
            cycle_phase = "🟢 Expansion"
        
        cycle_signals = {
            'recession_probability': recession_probability,
            'cycle_phase': cycle_phase,
            'indicators': recession_indicators
        }
        
        return cycle_signals
    
    def generate_economic_report(self, indicators: List[str] = None, forecast_steps: int = 12) -> Dict[str, Any]:
        """Generate comprehensive economic analysis report."""
        
        # Fetch data
        data = self.fetch_latest_data(indicators)
        
        if data.empty:
            raise ValueError("No economic data available for analysis")
        
        # Generate analysis components
        signals = self.calculate_economic_signals(data)
        forecasts = self.generate_forecasts(data, forecast_steps)
        cycle_assessment = self.assess_business_cycle(data)
        
        # Compile report
        report = {
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_period': {
                'start': data.index.min().strftime('%Y-%m-%d'),
                'end': data.index.max().strftime('%Y-%m-%d'),
                'observations': len(data)
            },
            'economic_signals': signals,
            'business_cycle': cycle_assessment,
            'forecasts': forecasts,
            'key_insights': self._generate_key_insights(signals, cycle_assessment, forecasts)
        }
        
        return report
    
    def _generate_key_insights(self, signals: Dict, cycle: Dict, forecasts: Dict) -> List[str]:
        """Generate key insights from the analysis."""
        insights = []
        
        # Business cycle insight
        insights.append(f"Business Cycle: {cycle['cycle_phase']} (Recession Probability: {cycle['recession_probability']:.1%})")
        
        # Growth insights
        if 'GDP' in signals:
            gdp_signal = signals['GDP']
            insights.append(f"Growth: {gdp_signal['signal']} (YoY: {gdp_signal['growth_yoy']:.1%})")
        
        # Inflation insights
        if 'Inflation' in signals:
            inflation_signal = signals['Inflation']
            insights.append(f"Inflation: {inflation_signal['signal']} (YoY: {inflation_signal['yoy_change']:.1%})")
        
        # Labor market insights
        if 'Labor Market' in signals:
            labor_signal = signals['Labor Market']
            insights.append(f"Labor Market: {labor_signal['signal']} (Rate: {labor_signal['unemployment_rate']:.1%})")
        
        # Monetary policy insights
        if 'Monetary Policy' in signals:
            monetary_signal = signals['Monetary Policy']
            insights.append(f"Monetary Policy: {monetary_signal['signal']} (Rate: {monetary_signal['fed_funds_rate']:.1%})")
        
        # Forecast insights
        if forecasts:
            insights.append(f"Forecasts generated for {len(forecasts)} key indicators")
        
        return insights
    
    def save_report(self, report: Dict[str, Any], output_dir: str = "data/processed/reports") -> Tuple[str, str]:
        """Save economic report to files."""
        ensure_dir(output_dir)
        
        # Save report data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"{output_dir}/economic_report_{timestamp}.csv"
        
        # Convert report to DataFrame for saving
        report_data = []
        for section, content in report.items():
            if section in ['economic_signals', 'key_insights']:
                if isinstance(content, dict):
                    for key, value in content.items():
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, (int, float)):
                                    report_data.append({
                                        'section': section,
                                        'indicator': f"{key}_{sub_key}",
                                        'value': sub_value
                                    })
                        else:
                            report_data.append({
                                'section': section,
                                'indicator': key,
                                'value': str(value)
                            })
                elif isinstance(content, list):
                    for i, item in enumerate(content):
                        report_data.append({
                            'section': section,
                            'indicator': f"insight_{i+1}",
                            'value': item
                        })
        
        report_df = pd.DataFrame(report_data)
        save_df_csv(report_df, report_path)
        
        # Save forecasts if available
        forecast_path = None
        if report['forecasts']:
            forecast_path = f"{output_dir}/economic_forecasts_{timestamp}.csv"
            forecast_data = []
            
            for indicator, forecast_df in report['forecasts'].items():
                for date, row in forecast_df.iterrows():
                    forecast_data.append({
                        'indicator': indicator,
                        'date': date.strftime('%Y-%m-%d'),
                        'forecast': row['yhat'],
                        'lower_bound': row['lower'],
                        'upper_bound': row['upper']
                    })
            
            forecasts_df = pd.DataFrame(forecast_data)
            save_df_csv(forecasts_df, forecast_path)
        
        return report_path, forecast_path


def analyze_economy(indicators: List[str] = None, forecast_steps: int = 12, save_report: bool = True) -> Dict[str, Any]:
    """Convenience function to analyze the economy."""
    analyzer = EconomicAnalyzer()
    report = analyzer.generate_economic_report(indicators, forecast_steps)
    
    if save_report:
        report_path, forecast_path = analyzer.save_report(report)
        report['saved_files'] = {
            'report': report_path,
            'forecasts': forecast_path
        }
    
    return report
