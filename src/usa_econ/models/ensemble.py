from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from .arima import arima_forecast
from .prophet_model import prophet_forecast
from .lstm_model import lstm_forecast


def create_features(data: pd.Series, look_back: int = 12) -> pd.DataFrame:
    """Create features for machine learning models.
    
    Args:
        data: Time series data
        look_back: Number of lag periods to use as features
        
    Returns:
        DataFrame with engineered features
    """
    df = pd.DataFrame({'target': data})
    
    # Lag features
    for i in range(1, look_back + 1):
        df[f'lag_{i}'] = data.shift(i)
    
    # Rolling statistics
    for window in [3, 6, 12]:
        df[f'rolling_mean_{window}'] = data.rolling(window=window).mean()
        df[f'rolling_std_{window}'] = data.rolling(window=window).std()
        df[f'rolling_min_{window}'] = data.rolling(window=window).min()
        df[f'rolling_max_{window}'] = data.rolling(window=window).max()
    
    # Difference features
    df['diff_1'] = data.diff(1)
    df['diff_12'] = data.diff(12)
    df['pct_change_1'] = data.pct_change(1)
    df['pct_change_12'] = data.pct_change(12)
    
    # Trend features
    df['trend'] = np.arange(len(data))
    df['month'] = data.index.month if hasattr(data.index, 'month') else 0
    df['quarter'] = data.index.quarter if hasattr(data.index, 'quarter') else 0
    
    return df


def ensemble_forecast(
    data: pd.Series | pd.DataFrame,
    steps: int = 12,
    models: List[str] = ['arima', 'prophet', 'rf', 'gbm'],
    column: str | None = None,
    look_back: int = 12,
    test_size: int = 24
) -> pd.DataFrame:
    """Create ensemble forecast using multiple models.
    
    Args:
        data: Time series data
        steps: Number of periods to forecast
        models: List of models to include in ensemble
        column: Column name if data is DataFrame
        look_back: Number of lag periods for ML features
        test_size: Number of periods to use for validation
        
    Returns:
        DataFrame with ensemble forecast and individual model forecasts
    """
    
    # Prepare data
    if isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("column must be provided when data is a DataFrame")
        y = data[column]
    else:
        y = data
    
    # Remove missing values
    y = y.dropna()
    
    if len(y) < look_back + test_size + 10:
        raise ValueError(f"Not enough data. Need at least {look_back + test_size + 10} points")
    
    # Split data
    train_data = y.iloc[:-test_size]
    test_data = y.iloc[-test_size:]
    
    # Dictionary to store individual model forecasts
    model_forecasts = {}
    model_weights = {}
    
    # ARIMA model
    if 'arima' in models:
        try:
            # Simple ARIMA parameters - could be optimized
            arima_fc = arima_forecast(train_data, steps=steps, order=(1,1,1), seasonal_order=(0,1,1,12))
            model_forecasts['arima'] = arima_fc['yhat']
            
            # Calculate weight based on test performance
            arima_test_fc = arima_forecast(train_data, steps=test_size, order=(1,1,1), seasonal_order=(0,1,1,12))
            arima_mae = mean_absolute_error(test_data, arima_test_fc['yhat'])
            model_weights['arima'] = 1.0 / (1.0 + arima_mae)
        except Exception as e:
            print(f"ARIMA model failed: {e}")
    
    # Prophet model
    if 'prophet' in models:
        try:
            prophet_fc = prophet_forecast(train_data, steps=steps)
            model_forecasts['prophet'] = prophet_fc['yhat']
            
            # Calculate weight
            prophet_test_fc = prophet_forecast(train_data, steps=test_size)
            prophet_mae = mean_absolute_error(test_data, prophet_test_fc['yhat'])
            model_weights['prophet'] = 1.0 / (1.0 + prophet_mae)
        except Exception as e:
            print(f"Prophet model failed: {e}")
    
    # LSTM model
    if 'lstm' in models:
        try:
            lstm_fc = lstm_forecast(train_data, steps=steps, look_back=look_back, epochs=50, verbose=0)
            model_forecasts['lstm'] = lstm_fc['yhat']
            
            # Calculate weight
            lstm_test_fc = lstm_forecast(train_data, steps=test_size, look_back=look_back, epochs=50, verbose=0)
            lstm_mae = mean_absolute_error(test_data, lstm_test_fc['yhat'])
            model_weights['lstm'] = 1.0 / (1.0 + lstm_mae)
        except Exception as e:
            print(f"LSTM model failed: {e}")
    
    # Random Forest model
    if 'rf' in models:
        try:
            rf_forecast = ml_forecast(train_data, steps=steps, look_back=look_back, model_type='rf')
            model_forecasts['rf'] = rf_forecast['yhat']
            
            # Calculate weight
            rf_test_fc = ml_forecast(train_data, steps=test_size, look_back=look_back, model_type='rf')
            rf_mae = mean_absolute_error(test_data, rf_test_fc['yhat'])
            model_weights['rf'] = 1.0 / (1.0 + rf_mae)
        except Exception as e:
            print(f"Random Forest model failed: {e}")
    
    # Gradient Boosting model
    if 'gbm' in models:
        try:
            gbm_forecast = ml_forecast(train_data, steps=steps, look_back=look_back, model_type='gbm')
            model_forecasts['gbm'] = gbm_forecast['yhat']
            
            # Calculate weight
            gbm_test_fc = ml_forecast(train_data, steps=test_size, look_back=look_back, model_type='gbm')
            gbm_mae = mean_absolute_error(test_data, gbm_test_fc['yhat'])
            model_weights['gbm'] = 1.0 / (1.0 + gbm_mae)
        except Exception as e:
            print(f"GBM model failed: {e}")
    
    # Create ensemble forecast
    if not model_forecasts:
        raise ValueError("No models succeeded in generating forecasts")
    
    # Normalize weights
    total_weight = sum(model_weights.values())
    model_weights = {k: v/total_weight for k, v in model_weights.items()}
    
    # Combine forecasts
    ensemble_forecast = pd.Series(0.0, index=list(model_forecasts.values())[0].index)
    
    for model_name, forecast in model_forecasts.items():
        weight = model_weights.get(model_name, 1.0 / len(model_forecasts))
        ensemble_forecast += forecast * weight
    
    # Calculate confidence intervals (simplified approach)
    all_forecasts = pd.DataFrame(model_forecasts)
    forecast_std = all_forecasts.std(axis=1)
    
    # Create result DataFrame
    result = pd.DataFrame({
        'yhat': ensemble_forecast,
        'lower': ensemble_forecast - 1.96 * forecast_std,
        'upper': ensemble_forecast + 1.96 * forecast_std
    })
    
    # Add individual model forecasts as columns
    for model_name, forecast in model_forecasts.items():
        result[f'{model_name}_yhat'] = forecast
    
    return result


def ml_forecast(
    data: pd.Series,
    steps: int = 12,
    look_back: int = 12,
    model_type: str = 'rf'
) -> pd.DataFrame:
    """Machine learning forecast using Random Forest or Gradient Boosting.
    
    Args:
        data: Time series data
        steps: Number of periods to forecast
        look_back: Number of lag periods for features
        model_type: 'rf' for Random Forest, 'gbm' for Gradient Boosting
        
    Returns:
        DataFrame with forecast
    """
    
    # Create features
    feature_df = create_features(data, look_back)
    feature_df = feature_df.dropna()
    
    if len(feature_df) < look_back + 10:
        raise ValueError("Not enough data after feature creation")
    
    # Prepare training data
    X = feature_df.drop('target', axis=1)
    y = feature_df['target']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gbm':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("model_type must be 'rf' or 'gbm'")
    
    model.fit(X_scaled, y)
    
    # Generate forecasts iteratively
    forecasts = []
    last_features = X_scaled[-1:].copy()
    
    for _ in range(steps):
        # Predict next value
        pred = model.predict(last_features)[0]
        forecasts.append(pred)
        
        # Update features for next prediction (simplified approach)
        # In practice, you'd want to properly update all feature columns
        new_row = last_features[0].copy()
        new_row[0] = pred  # Update first lag feature
        last_features = new_row.reshape(1, -1)
    
    # Create result DataFrame
    last_date = data.index[-1]
    freq = data.index.freq or pd.infer_freq(data.index) or 'M'
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=steps,
        freq=freq
    )
    
    result = pd.DataFrame({
        'yhat': forecasts
    }, index=forecast_dates)
    
    return result


def weighted_ensemble(
    forecasts: Dict[str, pd.Series],
    weights: Dict[str, float] | None = None
) -> pd.Series:
    """Combine multiple forecasts using weights.
    
    Args:
        forecasts: Dictionary of model forecasts
        weights: Dictionary of model weights (if None, uses equal weights)
        
    Returns:
        Combined forecast
    """
    
    if not forecasts:
        raise ValueError("No forecasts provided")
    
    if weights is None:
        weights = {model: 1.0/len(forecasts) for model in forecasts.keys()}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    # Combine forecasts
    ensemble = pd.Series(0.0, index=list(forecasts.values())[0].index)
    
    for model_name, forecast in forecasts.items():
        weight = weights.get(model_name, 0)
        ensemble += forecast * weight
    
    return ensemble
