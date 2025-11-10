from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .metrics import mae, rmse, mape
from .arima import arima_forecast
from .prophet_model import prophet_forecast
from .lstm_model import lstm_forecast
from .ensemble import ensemble_forecast


def backtest_model(
    data: pd.Series,
    model_func: callable,
    model_params: Dict[str, Any],
    test_size: int = 24,
    step_size: int = 1,
    column: str | None = None
) -> Dict[str, Any]:
    """Perform rolling window backtesting on a time series model.
    
    Args:
        data: Time series data
        model_func: Function that generates forecasts
        model_params: Parameters to pass to model function
        test_size: Size of test window
        step_size: Step size for rolling window
        column: Column name if data is DataFrame
        
    Returns:
        Dictionary with backtest results
    """
    
    if len(data) < test_size + 50:
        raise ValueError("Not enough data for backtesting")
    
    forecasts = []
    actuals = []
    dates = []
    
    # Rolling window backtesting
    for i in range(0, test_size, step_size):
        # Split data
        train_end = len(data) - test_size + i
        train_data = data.iloc[:train_end]
        test_point = data.iloc[train_end]
        
        try:
            # Generate forecast
            forecast = model_func(train_data, steps=1, **model_params)
            forecast_value = forecast['yhat'].iloc[0]
            
            forecasts.append(forecast_value)
            actuals.append(test_point)
            dates.append(data.index[train_end])
            
        except Exception as e:
            print(f"Backtest step {i} failed: {e}")
            continue
    
    if not forecasts:
        raise ValueError("No successful forecasts generated during backtesting")
    
    # Calculate metrics
    forecasts_series = pd.Series(forecasts, index=dates)
    actuals_series = pd.Series(actuals, index=dates)
    
    metrics = {
        'mae': mae(actuals_series, forecasts_series),
        'rmse': rmse(actuals_series, forecasts_series),
        'mape': mape(actuals_series, forecasts_series),
        'r2': r2_score(actuals_series, forecasts_series)
    }
    
    # Calculate directional accuracy
    actual_direction = np.diff(actuals) > 0
    forecast_direction = np.diff(forecasts) > 0
    directional_accuracy = np.mean(actual_direction == forecast_direction) if len(actual_direction) > 0 else 0
    
    metrics['directional_accuracy'] = directional_accuracy
    
    return {
        'forecasts': forecasts_series,
        'actuals': actuals_series,
        'metrics': metrics,
        'model_params': model_params
    }


def compare_models(
    data: pd.Series,
    models: Dict[str, Dict[str, Any]],
    test_size: int = 24,
    column: str | None = None
) -> pd.DataFrame:
    """Compare multiple models using backtesting.
    
    Args:
        data: Time series data
        models: Dictionary of model configurations
        test_size: Size of test window
        column: Column name if data is DataFrame
        
    Returns:
        DataFrame with model comparison results
    """
    
    results = {}
    
    model_functions = {
        'arima': arima_forecast,
        'prophet': prophet_forecast,
        'lstm': lstm_forecast,
        'ensemble': ensemble_forecast
    }
    
    for model_name, model_config in models.items():
        try:
            model_func = model_functions.get(model_config['type'])
            if model_func is None:
                print(f"Unknown model type: {model_config['type']}")
                continue
            
            result = backtest_model(
                data=data,
                model_func=model_func,
                model_params=model_config.get('params', {}),
                test_size=test_size,
                column=column
            )
            
            results[model_name] = result['metrics']
            results[model_name]['forecasts'] = result['forecasts']
            results[model_name]['actuals'] = result['actuals']
            
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
            continue
    
    if not results:
        raise ValueError("No models succeeded in comparison")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results).T
    
    # Select only metric columns for ranking
    metric_columns = ['mae', 'rmse', 'mape', 'r2', 'directional_accuracy']
    metrics_df = comparison_df[metric_columns]
    
    # Add rankings
    for metric in ['mae', 'rmse', 'mape']:
        comparison_df[f'{metric}_rank'] = metrics_df[metric].rank()
    
    for metric in ['r2', 'directional_accuracy']:
        comparison_df[f'{metric}_rank'] = metrics_df[metric].rank(ascending=False)
    
    # Add overall rank
    rank_columns = [col for col in comparison_df.columns if col.endswith('_rank')]
    comparison_df['overall_rank'] = comparison_df[rank_columns].mean(axis=1)
    
    return comparison_df.sort_values('overall_rank')


def model_selection_report(
    data: pd.Series,
    models: Dict[str, Dict[str, Any]],
    test_size: int = 24,
    save_path: str | None = None
) -> Dict[str, Any]:
    """Generate comprehensive model selection report.
    
    Args:
        data: Time series data
        models: Dictionary of model configurations
        test_size: Size of test window
        save_path: Path to save report plots
        
    Returns:
        Dictionary with report results
    """
    
    # Compare models
    comparison = compare_models(data, models, test_size)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Comparison Report', fontsize=16)
    
    # Plot 1: Metrics comparison
    metrics_to_plot = ['mae', 'rmse', 'mape', 'r2']
    comparison[metrics_to_plot].plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Error Metrics Comparison')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Forecast vs Actual for best model
    best_model = comparison.iloc[0].name
    if 'forecasts' in comparison.loc[best_model]:
        forecasts = comparison.loc[best_model, 'forecasts']
        actuals = comparison.loc[best_model, 'actuals']
        
        axes[0, 1].plot(actuals.index, actuals.values, label='Actual', marker='o')
        axes[0, 1].plot(forecasts.index, forecasts.values, label='Forecast', marker='x')
        axes[0, 1].set_title(f'Best Model Performance: {best_model}')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Rankings heatmap
    rank_columns = [col for col in comparison.columns if col.endswith('_rank')]
    if rank_columns:
        sns.heatmap(comparison[rank_columns], annot=True, ax=axes[1, 0], cmap='RdYlBu_r')
        axes[1, 0].set_title('Model Rankings')
    
    # Plot 4: Overall scores
    if 'overall_rank' in comparison.columns:
        comparison['overall_rank'].sort_values().plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Overall Model Ranking (lower is better)')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Generate text report
    report = {
        'best_model': best_model,
        'comparison_table': comparison,
        'model_count': len(models),
        'test_period': f"{test_size} periods",
        'data_points': len(data),
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return report


def cross_validation_forecast(
    data: pd.Series,
    model_func: callable,
    model_params: Dict[str, Any],
    n_splits: int = 5,
    test_size: int = 12
) -> Dict[str, Any]:
    """Perform time series cross-validation.
    
    Args:
        data: Time series data
        model_func: Function that generates forecasts
        model_params: Parameters to pass to model function
        n_splits: Number of CV splits
        test_size: Size of each test fold
        
    Returns:
        Dictionary with CV results
    """
    
    if len(data) < n_splits * test_size + 50:
        raise ValueError("Not enough data for cross-validation")
    
    fold_results = []
    all_forecasts = []
    all_actuals = []
    
    for fold in range(n_splits):
        # Calculate indices for this fold
        test_start = len(data) - (n_splits - fold) * test_size
        test_end = test_start + test_size
        
        train_data = data.iloc[:test_start]
        test_data = data.iloc[test_start:test_end]
        
        try:
            # Generate forecasts for this fold
            forecast = model_func(train_data, steps=test_size, **model_params)
            
            fold_metrics = {
                'fold': fold + 1,
                'mae': mae(test_data, forecast['yhat']),
                'rmse': rmse(test_data, forecast['yhat']),
                'mape': mape(test_data, forecast['yhat']),
                'r2': r2_score(test_data, forecast['yhat'])
            }
            
            fold_results.append(fold_metrics)
            all_forecasts.extend(forecast['yhat'].values)
            all_actuals.extend(test_data.values)
            
        except Exception as e:
            print(f"CV fold {fold + 1} failed: {e}")
            continue
    
    if not fold_results:
        raise ValueError("No successful CV folds")
    
    # Calculate average metrics across folds
    avg_metrics = {}
    for metric in ['mae', 'rmse', 'mape', 'r2']:
        values = [fold[metric] for fold in fold_results]
        avg_metrics[f'avg_{metric}'] = np.mean(values)
        avg_metrics[f'std_{metric}'] = np.std(values)
    
    return {
        'fold_results': fold_results,
        'avg_metrics': avg_metrics,
        'all_forecasts': pd.Series(all_forecasts),
        'all_actuals': pd.Series(all_actuals),
        'n_folds': len(fold_results)
    }


def forecast_uncertainty_analysis(
    forecasts: pd.DataFrame,
    actuals: pd.Series,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """Analyze forecast uncertainty and calibration.
    
    Args:
        forecasts: DataFrame with yhat, lower, upper columns
        actuals: Actual values
        confidence_level: Target confidence level
        
    Returns:
        Dictionary with uncertainty analysis results
    """
    
    # Check if actual values fall within confidence intervals
    within_ci = (actuals >= forecasts['lower']) & (actuals <= forecasts['upper'])
    coverage_rate = within_ci.mean()
    
    # Calculate prediction interval scores
    interval_width = forecasts['upper'] - forecasts['lower']
    sharpness = interval_width.mean()
    
    # Calculate calibration error
    target_coverage = confidence_level
    calibration_error = abs(coverage_rate - target_coverage)
    
    # Calculate reliability at different quantiles
    quantiles = [0.5, 0.8, 0.9, 0.95]
    reliability = {}
    
    for q in quantiles:
        alpha = 1 - q
        lower_q = forecasts['yhat'] - 1.96 * interval_width * alpha / 2
        upper_q = forecasts['yhat'] + 1.96 * interval_width * alpha / 2
        
        coverage_q = ((actuals >= lower_q) & (actuals <= upper_q)).mean()
        reliability[f'coverage_{q}'] = coverage_q
    
    return {
        'coverage_rate': coverage_rate,
        'target_coverage': target_coverage,
        'calibration_error': calibration_error,
        'sharpness': sharpness,
        'reliability': reliability,
        'well_calibrated': calibration_error < 0.05
    }
