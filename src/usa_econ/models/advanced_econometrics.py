from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX
try:
    from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
    VECM_AVAILABLE = True
except ImportError:
    VECM_AVAILABLE = False
try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    MARKOV_AVAILABLE = True
except ImportError:
    MARKOV_AVAILABLE = False
try:
    from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
    DYNAMIC_FACTOR_AVAILABLE = True
except ImportError:
    DYNAMIC_FACTOR_AVAILABLE = False
try:
    from statsmodels.tsa.statespace.unobserved_components import UnobservedComponents
    UNOBSERVED_COMPONENTS_AVAILABLE = True
except ImportError:
    UNOBSERVED_COMPONENTS_AVAILABLE = False
try:
    import pymc as pm
    import arviz as az
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False


def vecm_forecast(
    data: pd.DataFrame,
    steps: int = 12,
    coint_rank: int = 1,
    deterministic: str = "ci",
    k_ar_diff: int = 1
) -> pd.DataFrame:
    """Vector Error Correction Model forecast for cointegrated series.
    
    Args:
        data: DataFrame with cointegrated time series
        steps: Number of periods to forecast
        coint_rank: Cointegration rank (number of long-run relationships)
        deterministic: Deterministic terms ("ci", "li", "lo", "none")
        k_ar_diff: Number of lagged differences in the model
        
    Returns:
        DataFrame with forecasts for each series
    """
    
    if not VECM_AVAILABLE:
        raise ImportError("VECM requires statsmodels with vecm support. Upgrade statsmodels: pip install statsmodels --upgrade")
    
    # Check for cointegration
    if len(data.columns) < 2:
        raise ValueError("VECM requires at least 2 series")
    
    # Perform Johansen cointegration test
    try:
        johansen_test = coint_johansen(data, det_order=0, k_ar_diff=k_ar_diff)
        if coint_rank > johansen_test.rank:
            coint_rank = johansen_test.rank
            print(f"Warning: Cointegration rank reduced to {coint_rank}")
    except Exception as e:
        print(f"Johansen test failed: {e}")
    
    # Fit VECM model
    model = VECM(
        data,
        k_ar_diff=k_ar_diff,
        coint_rank=coint_rank,
        deterministic=deterministic
    )
    
    vecm_result = model.fit()
    
    # Generate forecasts
    forecast = vecm_result.predict(steps=steps)
    forecast_df = pd.DataFrame(forecast, columns=data.columns)
    
    # Create date index
    last_date = data.index[-1]
    freq = data.index.freq or pd.infer_freq(data.index) or 'M'
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=steps,
        freq=freq
    )
    forecast_df.index = forecast_dates
    
    # Calculate confidence intervals (simplified approach)
    std_errors = np.std(vecm_result.resid, axis=0)
    forecast_std = np.sqrt(np.arange(1, steps+1)[:, np.newaxis] * std_errors**2)
    
    for i, col in enumerate(data.columns):
        forecast_df[f'{col}_lower'] = forecast_df[col] - 1.96 * forecast_std[:, i]
        forecast_df[f'{col}_upper'] = forecast_df[col] + 1.96 * forecast_std[:, i]
    
    return forecast_df


def bayesian_var_forecast(
    data: pd.DataFrame,
    steps: int = 12,
    prior_mean: Optional[np.ndarray] = None,
    prior_cov: Optional[np.ndarray] = None,
    n_draws: int = 1000,
    tune: int = 500
) -> pd.DataFrame:
    """Bayesian Vector Autoregression with uncertainty quantification.
    
    Args:
        data: DataFrame with time series
        steps: Number of periods to forecast
        prior_mean: Prior mean for coefficients
        prior_cov: Prior covariance matrix
        n_draws: Number of MCMC draws
        tune: Number of tuning steps
        
    Returns:
        DataFrame with forecasts and credible intervals
    """
    
    if not BAYESIAN_AVAILABLE:
        # Fallback to standard VAR
        print("Bayesian VAR not available, using standard VAR fallback")
        from .var import var_forecast
        return var_forecast(data, steps=steps)
    
    n_vars = data.shape[1]
    n_obs = data.shape[0]
    
    # Standardize data
    data_mean = data.mean()
    data_std = data.std()
    data_standardized = (data - data_mean) / data_std
    
    # Determine optimal lag order (simplified)
    max_lags = min(4, n_obs // 10)
    best_lags = 1
    
    # Prepare data for Bayesian VAR
    y = data_standardized.values[best_lags:]
    X = np.column_stack([
        data_standardized.values[i:i-best_lags:-1].flatten() 
        for i in range(best_lags, n_obs)
    ])
    
    with pm.Model() as bvar_model:
        # Priors for coefficients
        if prior_mean is None:
            prior_mean = np.zeros(X.shape[1] * n_vars)
        if prior_cov is None:
            prior_cov = np.eye(X.shape[1] * n_vars) * 0.1
        
        # Coefficient matrix
        beta = pm.MvNormal(
            'beta', 
            mu=prior_mean, 
            cov=prior_cov,
            shape=(X.shape[1] * n_vars,)
        )
        
        # Residual covariance
        sigma = pm.Wishart('sigma', n=n_vars+1, V=np.eye(n_vars))
        
        # Expected values
        beta_matrix = beta.reshape((n_vars, -1))
        mu = pm.math.dot(X, beta_matrix.T)
        
        # Likelihood
        y_obs = pm.MvNormal('y_obs', mu=mu, cov=sigma, observed=y)
        
        # Sample
        trace = pm.sample(n_draws, tune=tune, cores=1, progressbar=False)
    
    # Generate forecasts
    forecast_draws = np.zeros((n_draws, steps, n_vars))
    
    for draw in range(min(n_draws, len(trace.posterior['beta']))):
        beta_draw = trace.posterior['beta'].isel(draw=0, chain=0).values
        beta_matrix = beta_draw.reshape((n_vars, -1))
        
        # Iterative forecasting
        forecast = np.zeros((steps, n_vars))
        last_obs = data_standardized.values[-best_lags:].flatten()
        
        for step in range(steps):
            x = last_obs[-best_lags*n_vars:].reshape(1, -1)
            forecast[step] = np.dot(x, beta_matrix.T).flatten()
            last_obs = np.concatenate([last_obs[n_vars:], forecast[step]])
        
        forecast_draws[draw] = forecast
    
    # Calculate summary statistics
    forecast_mean = np.mean(forecast_draws, axis=0)
    forecast_lower = np.percentile(forecast_draws, 2.5, axis=0)
    forecast_upper = np.percentile(forecast_draws, 97.5, axis=0)
    
    # Convert back to original scale
    forecast_mean = forecast_mean * data_std.values + data_mean.values
    forecast_lower = forecast_lower * data_std.values + data_mean.values
    forecast_upper = forecast_upper * data_std.values + data_mean.values
    
    # Create result DataFrame
    last_date = data.index[-1]
    freq = data.index.freq or pd.infer_freq(data.index) or 'M'
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=steps,
        freq=freq
    )
    
    result = pd.DataFrame(index=forecast_dates)
    
    for i, col in enumerate(data.columns):
        result[col] = forecast_mean[:, i]
        result[f'{col}_lower'] = forecast_lower[:, i]
        result[f'{col}_upper'] = forecast_upper[:, i]
    
    return result


def markov_switching_forecast(
    data: pd.Series,
    steps: int = 12,
    n_regimes: int = 2,
    switching_variance: bool = True,
    trend: str = 'c'
) -> pd.DataFrame:
    """Markov-Switching regression model for regime-dependent forecasting.
    
    Args:
        data: Time series data
        steps: Number of periods to forecast
        n_regimes: Number of regimes (typically 2 for expansion/recession)
        switching_variance: Whether to allow variance switching
        trend: Trend component ('c', 't', 'ct', 'nc')
        
    Returns:
        DataFrame with forecasts and regime probabilities
    """
    
    if not MARKOV_AVAILABLE:
        # Fallback to simpler model
        print("Markov-Switching not available, using ARIMA fallback")
        from .arima import arima_forecast
        return arima_forecast(data, steps=steps)
    
    # Prepare data with lagged values
    y = data.values
    n_obs = len(y)
    
    # Create lagged features
    max_lag = min(4, n_obs // 10)
    X = np.column_stack([y[max_lag-i:-i] for i in range(1, max_lag+1)])
    y_reg = y[max_lag:]
    
    # Fit Markov-Switching model
    model = MarkovRegression(
        y_reg,
        k_regimes=n_regimes,
        trend=trend,
        exog=X,
        switching_variance=switching_variance
    )
    
    results = model.fit(disp=False)
    
    # Generate forecasts
    forecast = np.zeros((steps, n_regimes))
    regime_probs = np.zeros((steps, n_regimes))
    
    # Get last observations for forecasting
    last_X = X[-1:].copy()
    current_regime_probs = results.smoothed_marginal_probabilities[-1:].values
    
    for step in range(steps):
        # Forecast for each regime
        for regime in range(n_regimes):
            forecast[step, regime] = results.predict(
                start=n_obs-step, 
                end=n_obs-step,
                regime=regime
            )[0]
        
        # Update regime probabilities
        if step < steps - 1:
            regime_probs[step] = current_regime_probs.flatten()
            current_regime_probs = results.predict_marginal_probabilities(
                n_obs-step, n_obs-step
            )
    
    # Weighted forecast by regime probabilities
    weighted_forecast = np.sum(forecast * regime_probs, axis=1)
    
    # Calculate confidence intervals
    forecast_std = np.sqrt(np.sum((forecast - weighted_forecast[:, np.newaxis])**2 * regime_probs, axis=1))
    lower_bound = weighted_forecast - 1.96 * forecast_std
    upper_bound = weighted_forecast + 1.96 * forecast_std
    
    # Create result DataFrame
    last_date = data.index[-1]
    freq = data.index.freq or pd.infer_freq(data.index) or 'M'
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=steps,
        freq=freq
    )
    
    result = pd.DataFrame({
        'yhat': weighted_forecast,
        'lower': lower_bound,
        'upper': upper_bound
    }, index=forecast_dates)
    
    # Add regime probabilities
    for regime in range(n_regimes):
        result[f'regime_{regime}_prob'] = regime_probs[:, regime]
    
    return result


def dynamic_factor_forecast(
    data: pd.DataFrame,
    steps: int = 12,
    n_factors: int = 3,
    factor_orders: int = 2,
    error_var: bool = False
) -> pd.DataFrame:
    """Dynamic Factor Model for forecasting multiple related series.
    
    Args:
        data: DataFrame with multiple time series
        steps: Number of periods to forecast
        n_factors: Number of latent factors
        factor_orders: Order of factor dynamics
        error_var: Whether to include error variance
        
    Returns:
        DataFrame with forecasts
    """
    
    if not DYNAMIC_FACTOR_AVAILABLE:
        # Fallback to VAR model
        print("Dynamic Factor model not available, using VAR fallback")
        from .var import var_forecast
        return var_forecast(data, steps=steps)
    
    # Fit Dynamic Factor Model
    model = DynamicFactor(
        data,
        k_factors=n_factors,
        factor_order=factor_orders,
        error_var=error_var
    )
    
    results = model.fit(disp=False)
    
    # Generate forecasts
    forecast = results.get_forecast(steps=steps)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    
    # Create result DataFrame
    result = forecast_mean.copy()
    
    # Add confidence intervals
    for i, col in enumerate(data.columns):
        result[f'{col}_lower'] = forecast_ci.iloc[:, 2*i]
        result[f'{col}_upper'] = forecast_ci.iloc[:, 2*i+1]
    
    # Rename columns to standard format
    result.columns = [f'yhat_{col}' if col in data.columns else col for col in result.columns]
    
    return result


def unobserved_components_forecast(
    data: pd.Series,
    steps: int = 12,
    level: bool = True,
    trend: bool = True,
    seasonal: Optional[int] = None,
    cycle: bool = False,
    irregular: bool = True,
    stochastic_level: bool = False,
    stochastic_trend: bool = False,
    stochastic_seasonal: bool = True,
    stochastic_cycle: bool = False
) -> pd.DataFrame:
    """Unobserved Components Model (structural time series) forecasting.
    
    Args:
        data: Time series data
        steps: Number of periods to forecast
        level: Include level component
        trend: Include trend component
        seasonal: Seasonal period (None for no seasonality)
        cycle: Include cycle component
        irregular: Include irregular component
        stochastic_level: Allow stochastic level
        stochastic_trend: Allow stochastic trend
        stochastic_seasonal: Allow stochastic seasonality
        stochastic_cycle: Allow stochastic cycle
        
    Returns:
        DataFrame with forecast and component decomposition
    """
    
    if not UNOBSERVED_COMPONENTS_AVAILABLE:
        # Fallback to ARIMA
        print("Unobserved Components model not available, using ARIMA fallback")
        from .arima import arima_forecast
        return arima_forecast(data, steps=steps)
    
    # Build model specification
    model = UnobservedComponents(
        data,
        level='llevel' if stochastic_level else 'deterministic constant' if level else None,
        trend='rtrend' if stochastic_trend else 'deterministic trend' if trend else None,
        seasonal=seasonal if seasonal else None,
        cycle=cycle,
        irregular=irregular,
        stochastic_seasonal=stochastic_seasonal,
        stochastic_cycle=stochastic_cycle
    )
    
    results = model.fit(disp=False)
    
    # Generate forecasts
    forecast = results.get_forecast(steps=steps)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    
    # Get component forecasts
    component_forecasts = {}
    if level:
        component_forecasts['level'] = results.level_forecast(steps=steps)
    if trend:
        component_forecasts['trend'] = results.trend_forecast(steps=steps)
    if seasonal:
        component_forecasts['seasonal'] = results.seasonal_forecast(steps=steps)
    if cycle:
        component_forecasts['cycle'] = results.cycle_forecast(steps=steps)
    
    # Create result DataFrame
    result = pd.DataFrame({
        'yhat': forecast_mean,
        'lower': forecast_ci.iloc[:, 0],
        'upper': forecast_ci.iloc[:, 1]
    }, index=forecast_mean.index)
    
    # Add component forecasts
    for component, values in component_forecasts.items():
        result[component] = values
    
    return result


def nowcast_economy(
    data: Dict[str, pd.Series],
    target_variable: str,
    steps: int = 1,
    mixed_frequency: bool = True
) -> pd.DataFrame:
    """Nowcasting using mixed-frequency data for real-time economic assessment.
    
    Args:
        data: Dictionary of economic indicators with different frequencies
        target_variable: Name of target variable to nowcast
        steps: Number of steps to nowcast (typically 1)
        mixed_frequency: Whether to handle mixed frequency data
        
    Returns:
        DataFrame with nowcast and real-time assessment
    """
    
    # Combine data into DataFrame
    combined_data = pd.DataFrame(data)
    
    if mixed_frequency:
        # Handle mixed frequency by forward-filling and interpolation
        combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')
        combined_data = combined_data.interpolate(method='time')
    
    # Remove missing values
    combined_data = combined_data.dropna()
    
    if target_variable not in combined_data.columns:
        raise ValueError(f"Target variable {target_variable} not found in data")
    
    # Use Dynamic Factor Model for nowcasting
    factor_model = DynamicFactor(
        combined_data,
        k_factors=min(3, len(combined_data.columns)//2),
        factor_order=1
    )
    
    factor_results = factor_model.fit(disp=False)
    
    # Generate nowcast
    nowcast = factor_results.get_forecast(steps=steps)
    nowcast_mean = nowcast.predicted_mean[target_variable]
    nowcast_ci = nowcast.conf_int()[[f'{target_variable}_lower', f'{target_variable}_upper']]
    
    # Create result DataFrame
    result = pd.DataFrame({
        'yhat': nowcast_mean,
        'lower': nowcast_ci.iloc[:, 0],
        'upper': nowcast_ci.iloc[:, 1]
    }, index=nowcast_mean.index)
    
    # Add real-time assessment
    latest_value = combined_data[target_variable].iloc[-1]
    nowcast_value = nowcast_mean.iloc[0]
    change = (nowcast_value - latest_value) / latest_value
    
    result['latest_value'] = latest_value
    result['nowcast_change'] = change
    result['assessment'] = 'Improving' if change > 0.01 else 'Stable' if abs(change) <= 0.01 else 'Declining'
    
    return result


def structural_break_analysis(
    data: pd.Series,
    max_breaks: int = 5,
    min_segment_length: int = 12
) -> Dict[str, Any]:
    """Detect structural breaks in economic time series.
    
    Args:
        data: Time series data
        max_breaks: Maximum number of breaks to detect
        min_segment_length: Minimum length of each segment
        
    Returns:
        Dictionary with break dates and analysis
    """
    
    try:
        from ruptures import Pelt, Binseg
        from ruptures.costs import CostL2
    except ImportError:
        raise ImportError("Install ruptures package for structural break detection: pip install ruptures")
    
    # Prepare data
    y = data.values
    n = len(y)
    
    # Detect breaks using PELT algorithm
    model = "l2"  # L2 norm cost
    algo = Pelt(model=model, min_size=min_segment_length).fit(y)
    breaks = algo.predict(pen=10)
    
    # Limit number of breaks
    if len(breaks) > max_breaks + 1:
        breaks = breaks[:max_breaks + 1]
    
    # Convert break indices to dates
    break_dates = [data.index[i-1] for i in breaks[:-1]]  # Exclude last point
    
    # Analyze each segment
    segments = []
    start_idx = 0
    
    for i, break_idx in enumerate(breaks):
        end_idx = break_idx
        segment_data = data.iloc[start_idx:end_idx]
        
        if len(segment_data) > 0:
            segment_info = {
                'start_date': data.index[start_idx],
                'end_date': data.index[end_idx-1],
                'length': len(segment_data),
                'mean': segment_data.mean(),
                'std': segment_data.std(),
                'trend': np.polyfit(range(len(segment_data)), segment_data.values, 1)[0],
                'growth_rate': (segment_data.iloc[-1] - segment_data.iloc[0]) / segment_data.iloc[0] if len(segment_data) > 1 else 0
            }
            segments.append(segment_info)
        
        start_idx = end_idx
    
    return {
        'break_dates': break_dates,
        'n_breaks': len(break_dates),
        'segments': segments,
        'break_detected': len(break_dates) > 0
    }
