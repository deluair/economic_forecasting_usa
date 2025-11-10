from __future__ import annotations

import pandas as pd
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')


def prophet_forecast(
    data: pd.Series | pd.DataFrame,
    steps: int = 12,
    freq: str = 'M',
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = False,
    daily_seasonality: bool = False,
    column: str | None = None,
    uncertainty_samples: int = 1000,
    alpha: float = 0.05
) -> pd.DataFrame:
    """Fit Prophet model and return forecast with confidence intervals.

    Args:
        data: Time series data (Series or DataFrame)
        steps: Number of periods to forecast
        freq: Frequency of the time series ('M' for monthly, 'D' for daily, etc.)
        yearly_seasonality: Include yearly seasonality
        weekly_seasonality: Include weekly seasonality
        daily_seasonality: Include daily seasonality
        column: Column name if data is DataFrame
        uncertainty_samples: Number of samples for uncertainty intervals
        alpha: Significance level for confidence intervals (default 0.05 for 95% CI)

    Returns:
        DataFrame with forecast containing columns: ds, yhat, yhat_lower, yhat_upper
    """
    
    # Prepare data for Prophet
    if isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("column must be provided when data is a DataFrame")
        y = data[column]
    else:
        y = data
    
    # Reset index to get dates
    if isinstance(y.index, pd.DatetimeIndex):
        df = pd.DataFrame({
            'ds': y.index,
            'y': y.values
        })
    else:
        # Assume first column is date
        df = y.reset_index()
        df.columns = ['ds', 'y']
    
    # Remove any missing values
    df = df.dropna()

    # Convert alpha to interval_width (e.g., alpha=0.05 -> interval_width=0.95)
    interval_width = 1 - alpha

    # Initialize and fit Prophet model
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        uncertainty_samples=uncertainty_samples,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        mcmc_samples=0,
        interval_width=interval_width,
        growth='linear'
    )
    
    # Fit the model
    model.fit(df)
    
    # Create future dataframe for forecasting
    future = model.make_future_dataframe(periods=steps, freq=freq)
    
    # Make predictions
    forecast = model.predict(future)
    
    # Return only the forecast portion
    forecast_result = forecast.iloc[-steps:].copy()
    
    # Rename columns to match our standard format
    result = pd.DataFrame({
        'yhat': forecast_result['yhat'],
        'lower': forecast_result['yhat_lower'],
        'upper': forecast_result['yhat_upper']
    })
    result.index = pd.to_datetime(forecast_result['ds'])
    
    return result


def prophet_with_regressors(
    data: pd.DataFrame,
    target_column: str,
    regressor_columns: list[str],
    steps: int = 12,
    freq: str = 'M'
) -> pd.DataFrame:
    """Fit Prophet model with additional regressors.
    
    Args:
        data: DataFrame with target and regressor columns
        target_column: Name of target variable column
        regressor_columns: List of regressor column names
        steps: Number of periods to forecast
        freq: Frequency of the time series
        
    Returns:
        DataFrame with forecast and components
    """
    
    # Prepare data
    df = data[[target_column] + regressor_columns].copy()
    df = df.reset_index()
    df.columns = ['ds'] + [target_column] + regressor_columns
    df = df.dropna()
    
    # Initialize model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    # Add regressors
    for reg in regressor_columns:
        model.add_regressor(reg)
    
    # Fit model
    model.fit(df)
    
    # Create future dataframe (need future values for regressors)
    future = model.make_future_dataframe(periods=steps, freq=freq)
    
    # For simplicity, use last known values for regressors in future periods
    # In practice, you'd need forecasts of regressors or assume they remain constant
    for reg in regressor_columns:
        last_value = df[reg].iloc[-1]
        future[reg] = last_value
    
    # Make predictions
    forecast = model.predict(future)
    
    # Return forecast portion
    forecast_result = forecast.iloc[-steps:].copy()
    
    result = pd.DataFrame({
        'yhat': forecast_result['yhat'],
        'lower': forecast_result['yhat_lower'],
        'upper': forecast_result['yhat_upper']
    })
    result.index = pd.to_datetime(forecast_result['ds'])
    
    return result
