import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from usa_econ.config import load_config
from usa_econ.data_sources.fred import get_series as fred_get_series
from usa_econ.models.arima import arima_forecast
from usa_econ.models.var import var_forecast
from usa_econ.models.metrics import mae, rmse, mape

# Page configuration
st.set_page_config(
    page_title="US Economic Forecaster Dashboard",
    page_icon="📈",
    layout="wide"
)

# Core economic indicators
CORE_INDICATORS = {
    "GDP": "GDP",
    "CPI": "CPIAUCSL", 
    "Unemployment Rate": "UNRATE",
    "Industrial Production": "INDPRO",
    "Housing Starts": "HOUST",
    "Retail Sales": "RSXFS",
    "10-Year Treasury": "DGS10",
    "Federal Funds Rate": "FEDFUNDS",
    "Consumer Confidence": "UMCSENT",
    "PMI": "NAPMI"
}

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_economic_data(series_ids, start_date="2015-01-01"):
    """Fetch multiple economic series from FRED"""
    cfg = load_config()
    data_dict = {}
    
    for name, series_id in series_ids.items():
        try:
            df = fred_get_series(series_id, cfg, start=start_date)
            data_dict[name] = df[series_id]
        except Exception as e:
            st.error(f"Error fetching {name}: {str(e)}")
    
    return pd.DataFrame(data_dict)

def create_forecast_chart(data, forecast_data=None, title=""):
    """Create interactive chart with historical data and forecasts"""
    fig = go.Figure()
    
    # Add historical data
    for col in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[col],
            mode='lines',
            name=f'{col} (Historical)',
            line=dict(width=2)
        ))
    
    # Add forecast if provided
    if forecast_data is not None:
        if 'yhat' in forecast_data.columns:
            fig.add_trace(go.Scatter(
                x=forecast_data.index,
                y=forecast_data['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(width=2, dash='dash')
            ))
            
            if 'lower' in forecast_data.columns and 'upper' in forecast_data.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_data.index,
                    y=forecast_data['upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_data.index,
                    y=forecast_data['lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(0,100,80,0.2)',
                    name='Confidence Interval'
                ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def calculate_economic_signals(data):
    """Calculate economic signals and insights"""
    signals = {}
    
    # GDP growth signal
    if 'GDP' in data.columns:
        gdp_growth = data['GDP'].pct_change(periods=4).iloc[-1]
        signals['GDP Growth'] = {
            'value': f"{gdp_growth:.2%}",
            'signal': '🟢 Strong' if gdp_growth > 0.03 else '🟡 Moderate' if gdp_growth > 0 else '🔴 Weak'
        }
    
    # Inflation signal
    if 'CPI' in data.columns:
        cpi_growth = data['CPI'].pct_change(periods=12).iloc[-1]
        signals['Inflation (YoY)'] = {
            'value': f"{cpi_growth:.2%}",
            'signal': '🔴 High' if cpi_growth > 0.04 else '🟡 Moderate' if cpi_growth > 0.02 else '🟢 Low'
        }
    
    # Unemployment signal
    if 'Unemployment Rate' in data.columns:
        unemployment = data['Unemployment Rate'].iloc[-1]
        signals['Unemployment'] = {
            'value': f"{unemployment:.1%}",
            'signal': '🟢 Low' if unemployment < 0.04 else '🟡 Moderate' if unemployment < 0.06 else '🔴 High'
        }
    
    # Interest rate signal
    if 'Federal Funds Rate' in data.columns:
        fed_rate = data['Federal Funds Rate'].iloc[-1]
        signals['Fed Funds Rate'] = {
            'value': f"{fed_rate:.2%}",
            'signal': '🔴 High' if fed_rate > 0.04 else '🟡 Moderate' if fed_rate > 0.02 else '🟢 Low'
        }
    
    return signals

def main():
    st.title("🇺🇸 US Economic Forecaster Dashboard")
    st.markdown("Expert forecasting toolkit for key US economic indicators")
    
    # Sidebar for controls
    st.sidebar.header("Dashboard Controls")
    
    # Date range selector
    end_date = datetime.now()
    start_date = st.sidebar.date_input(
        "Start Date",
        value=end_date - timedelta(days=5*365),
        max_value=end_date
    )
    
    # Indicator selection
    selected_indicators = st.sidebar.multiselect(
        "Select Economic Indicators",
        list(CORE_INDICATORS.keys()),
        default=['GDP', 'CPI', 'Unemployment Rate', 'Federal Funds Rate']
    )
    
    if not selected_indicators:
        st.warning("Please select at least one economic indicator")
        return
    
    # Fetch data
    with st.spinner("Fetching economic data..."):
        series_to_fetch = {k: CORE_INDICATORS[k] for k in selected_indicators}
        data = fetch_economic_data(series_to_fetch, start_date.strftime("%Y-%m-%d"))
    
    if data.empty:
        st.error("No data available. Please check your API keys.")
        return
    
    # Main dashboard layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Economic Indicators Trends")
        fig = create_forecast_chart(data, title="Selected Economic Indicators")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Economic Signals")
        signals = calculate_economic_signals(data)
        
        for indicator, info in signals.items():
            st.metric(
                label=indicator,
                value=info['value'],
                delta=info['signal']
            )
    
    # Forecasting section
    st.header("🔮 Economic Forecasting")
    
    forecast_col1, forecast_col2 = st.columns(2)
    
    with forecast_col1:
        st.subheader("ARIMA Forecast")
        forecast_target = st.selectbox("Select series to forecast", selected_indicators)
        forecast_steps = st.slider("Forecast horizon (months)", 1, 24, 12)
        
        if st.button("Generate ARIMA Forecast"):
            with st.spinner("Generating forecast..."):
                series_data = data[forecast_target].dropna()
                forecast = arima_forecast(series_data, steps=forecast_steps)
                
                forecast_fig = create_forecast_chart(
                    series_data.to_frame(), 
                    forecast, 
                    f"{forecast_target} - ARIMA Forecast"
                )
                st.plotly_chart(forecast_fig, use_container_width=True)
    
    with forecast_col2:
        st.subheader("Multivariate VAR Forecast")
        if len(selected_indicators) >= 2:
            var_series = st.multiselect(
                "Select series for VAR model",
                selected_indicators,
                default=selected_indicators[:3]
            )
            
            if len(var_series) >= 2:
                var_steps = st.slider("VAR forecast horizon", 1, 12, 6)
                
                if st.button("Generate VAR Forecast"):
                    with st.spinner("Generating VAR forecast..."):
                        var_data = data[var_series].dropna()
                        var_forecast = var_forecast(var_data, steps=var_steps)
                        
                        st.success("VAR Forecast Generated")
                        st.dataframe(var_forecast)
        else:
            st.info("Select at least 2 indicators for VAR forecasting")
    
    # Economic insights section
    st.header("📊 Economic Analysis & Insights")
    
    # Correlation analysis
    if len(selected_indicators) >= 2:
        st.subheader("Indicator Correlations")
        corr_data = data[selected_indicators].pct_change().corr()
        fig = px.imshow(
            corr_data,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent changes
    st.subheader("Recent Economic Changes")
    recent_data = data.iloc[-12:]  # Last 12 periods
    changes = recent_data.pct_change().iloc[-1]
    
    change_df = pd.DataFrame({
        'Indicator': changes.index,
        'Recent Change (%)': changes.values * 100
    })
    change_df = change_df.sort_values('Recent Change (%)', ascending=False)
    
    st.dataframe(change_df, use_container_width=True)

if __name__ == "__main__":
    main()
