from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def prepare_lstm_data(data: np.ndarray, look_back: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Prepare data for LSTM training.
    
    Args:
        data: Time series data as numpy array
        look_back: Number of previous time steps to use as input features
        
    Returns:
        Tuple of (X, y) where X has shape (samples, look_back, 1) and y has shape (samples, 1)
    """
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)


def lstm_forecast(
    data: pd.Series | pd.DataFrame,
    steps: int = 12,
    look_back: int = 12,
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.2,
    column: str | None = None,
    lstm_units: int = 50,
    dropout_rate: float = 0.2,
    verbose: int = 0
) -> pd.DataFrame:
    """Fit LSTM model and return forecast with confidence intervals.
    
    Args:
        data: Time series data (Series or DataFrame)
        steps: Number of periods to forecast
        look_back: Number of previous time steps to use as input
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
        column: Column name if data is DataFrame
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate for regularization
        verbose: Verbosity mode (0=silent, 1=progress bar)
        
    Returns:
        DataFrame with forecast containing columns: yhat, lower, upper
    """
    
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is not available. Please install tensorflow to use LSTM forecasting.")
    
    # Prepare data
    if isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("column must be provided when data is a DataFrame")
        y = data[column].values
    else:
        y = data.values
    
    # Remove missing values
    y = y[~np.isnan(y)]
    
    if len(y) < look_back + 10:
        raise ValueError(f"Not enough data points. Need at least {look_back + 10}, got {len(y)}")
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Prepare training data
    X, y_train = prepare_lstm_data(y_scaled, look_back)
    
    # Reshape input for LSTM [samples, time steps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Build LSTM model
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train model
    model.fit(
        X, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose,
        shuffle=False
    )
    
    # Make forecasts
    forecasts = []
    current_batch = y_scaled[-look_back:].reshape((1, look_back, 1))
    
    for _ in range(steps):
        # Get prediction for next time step
        pred = model.predict(current_batch, verbose=0)[0, 0]
        forecasts.append(pred)
        
        # Update batch for next prediction
        current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
    
    # Inverse transform forecasts
    forecasts = np.array(forecasts).reshape(-1, 1)
    forecasts_unscaled = scaler.inverse_transform(forecasts).flatten()
    
    # Calculate confidence intervals using prediction uncertainty
    # For simplicity, use historical prediction errors as proxy
    train_predictions = model.predict(X, verbose=0).flatten()
    train_errors = np.abs(y_train - train_predictions)
    error_std = np.std(train_errors)
    
    # Create result DataFrame
    last_date = data.index[-1] if hasattr(data, 'index') else pd.Timestamp.now()
    
    if hasattr(data, 'index') and hasattr(data.index, 'freq'):
        freq = data.index.freq or pd.infer_freq(data.index)
    else:
        freq = 'M'  # Default to monthly
    
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=steps,
        freq=freq
    )
    
    result = pd.DataFrame({
        'yhat': forecasts_unscaled,
        'lower': forecasts_unscaled - 1.96 * error_std,
        'upper': forecasts_unscaled + 1.96 * error_std
    }, index=forecast_dates)
    
    return result


def multivariate_lstm_forecast(
    data: pd.DataFrame,
    target_column: str,
    steps: int = 12,
    look_back: int = 12,
    epochs: int = 100,
    batch_size: int = 32,
    lstm_units: int = 50,
    dropout_rate: float = 0.2,
    verbose: int = 0
) -> pd.DataFrame:
    """Multivariate LSTM forecasting using multiple time series.
    
    Args:
        data: DataFrame with multiple time series
        target_column: Name of target variable to forecast
        steps: Number of periods to forecast
        look_back: Number of previous time steps to use as input
        epochs: Number of training epochs
        batch_size: Batch size for training
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate for regularization
        verbose: Verbosity mode
        
    Returns:
        DataFrame with forecast for target variable
    """
    
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is not available. Please install tensorflow to use LSTM forecasting.")
    
    # Prepare data
    df = data.dropna()
    if len(df) < look_back + 10:
        raise ValueError(f"Not enough data points. Need at least {look_back + 10}, got {len(df)}")
    
    # Scale all features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    # Prepare sequences
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:(i + look_back)])
        y.append(scaled_data[i + look_back, df.columns.get_loc(target_column)])
    
    X, y = np.array(X), np.array(y)
    
    # Build model
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=(look_back, df.shape[1])),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train model
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=False)
    
    # Make forecasts
    forecasts = []
    current_batch = scaled_data[-look_back:].reshape((1, look_back, df.shape[1]))
    
    for _ in range(steps):
        pred = model.predict(current_batch, verbose=0)[0, 0]
        forecasts.append(pred)
        
        # Update batch (simplified - assumes other features remain constant)
        new_row = current_batch[0, -1, :].copy()
        new_row[df.columns.get_loc(target_column)] = pred
        current_batch = np.append(current_batch[:, 1:, :], [new_row.reshape(1, -1)], axis=1)
    
    # Inverse transform only the target variable
    forecasts = np.array(forecasts).reshape(-1, 1)
    
    # Create dummy array for inverse transform
    dummy = np.zeros((len(forecasts), df.shape[1]))
    dummy[:, df.columns.get_loc(target_column)] = forecasts.flatten()
    forecasts_unscaled = scaler.inverse_transform(dummy)[:, df.columns.get_loc(target_column)]
    
    # Create result DataFrame
    last_date = df.index[-1]
    freq = df.index.freq or pd.infer_freq(df.index) or 'M'
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=steps,
        freq=freq
    )
    
    result = pd.DataFrame({
        'yhat': forecasts_unscaled
    }, index=forecast_dates)
    
    return result
