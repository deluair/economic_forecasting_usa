import numpy as np
import pandas as pd


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAPE as a percentage (e.g., 5.0 means 5% error)

    Note:
        - Zero values in y_true are excluded from calculation to avoid division by zero
        - Returns NaN if all y_true values are zero
    """
    # Replace zeros with NaN to avoid division by zero
    y_true_nonzero = y_true.replace(0, np.nan)

    # Calculate percentage errors
    percentage_errors = np.abs((y_true_nonzero - y_pred) / y_true_nonzero)

    # Use nanmean to ignore NaN values (from zeros and any existing NaNs)
    # Returns NaN if all values are NaN
    return float(np.nanmean(percentage_errors) * 100)