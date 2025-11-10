from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.optimize import minimize
import networkx as nx


class EconomicRiskModeler:
    """Advanced economic risk modeling and scenario analysis system."""
    
    def __init__(self, confidence_level: float = 0.95):
        """Initialize the risk modeler."""
        self.confidence_level = confidence_level
        self.risk_factors = self._initialize_risk_factors()
        self.correlation_matrix = None
        self.stress_scenarios = self._initialize_stress_scenarios()
    
    def _initialize_risk_factors(self) -> Dict[str, Dict[str, Any]]:
        """Initialize economic risk factors."""
        return {
            'gdp_growth': {
                'volatility': 0.02,
                'mean': 0.025,
                'distribution': 'normal',
                'critical_threshold': -0.02
            },
            'inflation': {
                'volatility': 0.015,
                'mean': 0.025,
                'distribution': 'normal',
                'critical_threshold': 0.05
            },
            'unemployment': {
                'volatility': 0.02,
                'mean': 0.05,
                'distribution': 'normal',
                'critical_threshold': 0.08
            },
            'interest_rates': {
                'volatility': 0.025,
                'mean': 0.03,
                'distribution': 'normal',
                'critical_threshold': 0.06
            },
            'equity_returns': {
                'volatility': 0.15,
                'mean': 0.08,
                'distribution': 't',
                'df': 5,
                'critical_threshold': -0.20
            },
            'credit_spreads': {
                'volatility': 0.02,
                'mean': 0.02,
                'distribution': 'normal',
                'critical_threshold': 0.05
            }
        }
    
    def _initialize_stress_scenarios(self) -> Dict[str, Dict[str, float]]:
        """Initialize predefined stress scenarios."""
        return {
            'recession': {
                'gdp_growth': -0.03,
                'inflation': 0.01,
                'unemployment': 0.08,
                'interest_rates': 0.01,
                'equity_returns': -0.25,
                'credit_spreads': 0.04
            },
            'stagflation': {
                'gdp_growth': -0.01,
                'inflation': 0.06,
                'unemployment': 0.07,
                'interest_rates': 0.05,
                'equity_returns': -0.15,
                'credit_spreads': 0.03
            },
            'financial_crisis': {
                'gdp_growth': -0.05,
                'inflation': 0.02,
                'unemployment': 0.10,
                'interest_rates': 0.00,
                'equity_returns': -0.40,
                'credit_spreads': 0.08
            },
            'inflation_spike': {
                'gdp_growth': 0.01,
                'inflation': 0.08,
                'unemployment': 0.06,
                'interest_rates': 0.07,
                'equity_returns': -0.10,
                'credit_spreads': 0.02
            },
            'growth_boom': {
                'gdp_growth': 0.05,
                'inflation': 0.03,
                'unemployment': 0.04,
                'interest_rates': 0.04,
                'equity_returns': 0.20,
                'credit_spreads': 0.01
            }
        }
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: Optional[float] = None,
        method: str = 'historical',
        time_horizon: int = 1
    ) -> Dict[str, float]:
        """Calculate Value at Risk for economic indicators.
        
        Args:
            returns: Series of returns or changes
            confidence_level: Confidence level for VaR (default: instance level)
            method: VaR calculation method ('historical', 'parametric', 'monte_carlo')
            time_horizon: Time horizon in periods
            
        Returns:
            Dictionary with VaR metrics
        """
        
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 30:
            raise ValueError("Insufficient data for VaR calculation")
        
        if method == 'historical':
            var = np.percentile(returns_clean, (1 - confidence_level) * 100)
            cvar = returns_clean[returns_clean <= var].mean()
        
        elif method == 'parametric':
            # Fit normal distribution
            mu, sigma = stats.norm.fit(returns_clean)
            var = stats.norm.ppf(1 - confidence_level, mu, sigma)
            
            # Calculate CVaR (Expected Shortfall)
            cvar = mu - sigma * stats.norm.pdf(stats.norm.ppf(1 - confidence_level)) / (1 - confidence_level)
        
        elif method == 'monte_carlo':
            # Monte Carlo simulation
            n_simulations = 10000
            
            # Fit distribution
            if len(returns_clean) > 100:
                # Try t-distribution for better tail fit
                try:
                    params = stats.t.fit(returns_clean)
                    simulated = stats.t.rvs(*params, size=n_simulations)
                except:
                    mu, sigma = stats.norm.fit(returns_clean)
                    simulated = np.random.normal(mu, sigma, n_simulations)
            else:
                mu, sigma = stats.norm.fit(returns_clean)
                simulated = np.random.normal(mu, sigma, n_simulations)
            
            var = np.percentile(simulated, (1 - confidence_level) * 100)
            cvar = simulated[simulated <= var].mean()
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        # Scale for time horizon
        var_scaled = var * np.sqrt(time_horizon)
        cvar_scaled = cvar * np.sqrt(time_horizon)
        
        return {
            'var': var_scaled,
            'cvar': cvar_scaled,
            'confidence_level': confidence_level,
            'method': method,
            'time_horizon': time_horizon
        }
    
    def calculate_economic_var(
        self,
        data: pd.DataFrame,
        portfolio_weights: Optional[Dict[str, float]] = None,
        confidence_level: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculate portfolio-level Economic VaR.
        
        Args:
            data: DataFrame with economic indicators
            portfolio_weights: Weights for each indicator
            confidence_level: Confidence level for VaR
            
        Returns:
            Dictionary with Economic VaR results
        """
        
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        if portfolio_weights is None:
            # Equal weights
            portfolio_weights = {col: 1/len(data.columns) for col in data.columns}
        
        # Calculate portfolio returns
        portfolio_returns = pd.Series(0, index=data.index)
        
        for indicator, weight in portfolio_weights.items():
            if indicator in data.columns:
                returns = data[indicator].pct_change().dropna()
                portfolio_returns += returns * weight
        
        # Calculate individual VaRs
        individual_vars = {}
        for indicator in data.columns:
            returns = data[indicator].pct_change().dropna()
            individual_vars[indicator] = self.calculate_var(returns, confidence_level)
        
        # Calculate portfolio VaR
        portfolio_var = self.calculate_var(portfolio_returns, confidence_level)
        
        # Calculate diversification benefit
        weighted_individual_var = sum(
            individual_vars[ind]['var'] * weight**2 
            for ind, weight in portfolio_weights.items() 
            if ind in individual_vars
        )
        
        diversification_benefit = weighted_individual_var - portfolio_var['var']
        
        return {
            'portfolio_var': portfolio_var,
            'individual_vars': individual_vars,
            'portfolio_weights': portfolio_weights,
            'diversification_benefit': diversification_benefit,
            'diversification_ratio': portfolio_var['var'] / weighted_individual_var if weighted_individual_var != 0 else 1
        }
    
    def monte_carlo_simulation(
        self,
        initial_values: Dict[str, float],
        time_horizon: int = 252,  # Trading days
        n_simulations: int = 10000,
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Run Monte Carlo simulation for economic scenarios.
        
        Args:
            initial_values: Dictionary of initial values for each factor
            time_horizon: Number of time periods to simulate
            n_simulations: Number of Monte Carlo paths
            correlation_matrix: Correlation matrix for factors
            
        Returns:
            DataFrame with simulation results
        """
        
        factors = list(initial_values.keys())
        n_factors = len(factors)
        
        # Get factor parameters
        means = np.array([self.risk_factors[factor]['mean'] for factor in factors])
        volatilities = np.array([self.risk_factors[factor]['volatility'] for factor in factors])
        
        # Generate correlated random numbers
        if correlation_matrix is not None:
            # Cholesky decomposition for correlation
            cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix.values
            chol = np.linalg.cholesky(cov_matrix)
            
            # Generate uncorrelated random numbers
            random_shocks = np.random.normal(0, 1, (time_horizon, n_simulations, n_factors))
            
            # Apply correlation
            correlated_shocks = np.zeros_like(random_shocks)
            for t in range(time_horizon):
                for i in range(n_simulations):
                    correlated_shocks[t, i] = chol @ random_shocks[t, i]
        else:
            # Independent factors
            correlated_shocks = np.random.normal(0, 1, (time_horizon, n_simulations, n_factors))
            correlated_shocks = correlated_shocks * volatilities[np.newaxis, np.newaxis, :]
        
        # Simulate paths
        simulations = np.zeros((time_horizon + 1, n_simulations, n_factors))
        simulations[0, :, :] = np.array(list(initial_values.values()))[np.newaxis, :]
        
        for t in range(1, time_horizon + 1):
            simulations[t] = simulations[t-1] * np.exp(means * 1/252 + correlated_shocks[t-1])
        
        # Convert to DataFrame
        results = {}
        for i, factor in enumerate(factors):
            results[factor] = pd.DataFrame(
                simulations[:, :, i].T,
                columns=[f't_{j}' for j in range(time_horizon + 1)]
            )
        
        return pd.concat(results.values(), axis=1, keys=factors)
    
    def stress_test(
        self,
        portfolio_data: Dict[str, float],
        scenario: Union[str, Dict[str, float]] = 'recession'
    ) -> Dict[str, Any]:
        """Perform stress testing on economic portfolio.
        
        Args:
            portfolio_data: Current portfolio values/exposures
            scenario: Stress scenario name or custom scenario
            
        Returns:
            Dictionary with stress test results
        """
        
        if isinstance(scenario, str):
            if scenario not in self.stress_scenarios:
                raise ValueError(f"Unknown scenario: {scenario}")
            scenario_shocks = self.stress_scenarios[scenario]
        else:
            scenario_shocks = scenario
        
        # Calculate stressed values
        stressed_values = {}
        percentage_changes = {}
        
        for factor, current_value in portfolio_data.items():
            if factor in scenario_shocks:
                shock = scenario_shocks[factor]
                stressed_value = current_value * (1 + shock)
                stressed_values[factor] = stressed_value
                percentage_changes[factor] = shock
        
        # Calculate portfolio impact
        total_current = sum(portfolio_data.values())
        total_stressed = sum(stressed_values.values())
        portfolio_impact = (total_stressed - total_current) / total_current
        
        # Identify worst affected factors
        worst_factors = sorted(
            percentage_changes.items(),
            key=lambda x: x[1]
        )[:3]
        
        return {
            'scenario': scenario,
            'current_values': portfolio_data,
            'stressed_values': stressed_values,
            'percentage_changes': percentage_changes,
            'portfolio_impact': portfolio_impact,
            'worst_affected_factors': worst_factors,
            'stress_magnitude': abs(portfolio_impact)
        }
    
    def scenario_analysis(
        self,
        base_case: Dict[str, float],
        scenarios: Dict[str, Dict[str, float]] = None
    ) -> pd.DataFrame:
        """Perform comprehensive scenario analysis.
        
        Args:
            base_case: Base case economic parameters
            scenarios: Dictionary of scenarios to analyze
            
        Returns:
            DataFrame with scenario comparison
        """
        
        if scenarios is None:
            scenarios = self.stress_scenarios
        
        results = []
        
        for scenario_name, scenario_params in scenarios.items():
            scenario_result = {'scenario': scenario_name}
            
            for factor, base_value in base_case.items():
                if factor in scenario_params:
                    shocked_value = base_value * (1 + scenario_params[factor])
                    scenario_result[f'{factor}_base'] = base_value
                    scenario_result[f'{factor}_stressed'] = shocked_value
                    scenario_result[f'{factor}_change'] = scenario_params[factor]
            
            results.append(scenario_result)
        
        return pd.DataFrame(results)
    
    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive risk metrics.
        
        Args:
            returns: Return series
            benchmark: Benchmark return series (optional)
            
        Returns:
            Dictionary with risk metrics
        """
        
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 30:
            raise ValueError("Insufficient data for risk metrics calculation")
        
        metrics = {}
        
        # Basic metrics
        metrics['volatility'] = returns_clean.std() * np.sqrt(252)  # Annualized
        metrics['mean_return'] = returns_clean.mean() * 252
        metrics['sharpe_ratio'] = metrics['mean_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
        metrics['skewness'] = stats.skew(returns_clean)
        metrics['excess_kurtosis'] = stats.kurtosis(returns_clean) - 3
        
        # Downside metrics
        negative_returns = returns_clean[returns_clean < 0]
        if len(negative_returns) > 0:
            metrics['downside_deviation'] = negative_returns.std() * np.sqrt(252)
            metrics['max_drawdown'] = self._calculate_max_drawdown(returns_clean)
            metrics['calmar_ratio'] = metrics['mean_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        else:
            metrics['downside_deviation'] = 0
            metrics['max_drawdown'] = 0
            metrics['calmar_ratio'] = float('inf')
        
        # VaR metrics
        var_95 = self.calculate_var(returns_clean, 0.95)
        var_99 = self.calculate_var(returns_clean, 0.99)
        
        metrics['var_95'] = var_95['var']
        metrics['var_99'] = var_99['var']
        metrics['cvar_95'] = var_95['cvar']
        metrics['cvar_99'] = var_99['cvar']
        
        # Beta and Alpha (if benchmark provided)
        if benchmark is not None:
            benchmark_clean = benchmark.dropna()
            common_index = returns_clean.index.intersection(benchmark_clean.index)
            
            if len(common_index) > 30:
                returns_aligned = returns_clean.loc[common_index]
                benchmark_aligned = benchmark_clean.loc[common_index]
                
                # Calculate beta
                covariance = np.cov(returns_aligned, benchmark_aligned)[0, 1]
                benchmark_variance = np.var(benchmark_aligned)
                metrics['beta'] = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                # Calculate alpha
                risk_free_rate = 0.02  # Assume 2% risk-free rate
                metrics['alpha'] = (metrics['mean_return'] - risk_free_rate) - metrics['beta'] * (benchmark_aligned.mean() * 252 - risk_free_rate)
                metrics['correlation'] = np.corrcoef(returns_aligned, benchmark_aligned)[0, 1]
                metrics['tracking_error'] = (returns_aligned - benchmark_aligned).std() * np.sqrt(252)
                metrics['information_ratio'] = metrics['alpha'] / metrics['tracking_error'] if metrics['tracking_error'] > 0 else 0
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def network_risk_analysis(
        self,
        correlation_matrix: pd.DataFrame,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Analyze systemic risk using network theory.
        
        Args:
            correlation_matrix: Correlation matrix of economic factors
            threshold: Correlation threshold for network connections
            
        Returns:
            Dictionary with network risk metrics
        """
        
        # Create network from correlation matrix
        G = nx.from_pandas_adjacency(
            correlation_matrix.abs() > threshold,
            create_using=nx.Graph()
        )
        
        # Calculate network metrics
        metrics = {}
        
        if len(G.nodes) > 0:
            metrics['node_count'] = len(G.nodes)
            metrics['edge_count'] = len(G.edges)
            metrics['density'] = nx.density(G)
            
            # Centrality measures
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            eigenvector_centrality = nx.eigenvector_centrality(G)
            
            metrics['most_central_node'] = max(degree_centrality, key=degree_centrality.get)
            metrics['highest_betweenness'] = max(betweenness_centrality, key=betweenness_centrality.get)
            
            # Systemic risk measures
            metrics['average_clustering'] = nx.average_clustering(G)
            
            if nx.is_connected(G):
                metrics['average_path_length'] = nx.average_shortest_path_length(G)
                metrics['diameter'] = nx.diameter(G)
            else:
                metrics['average_path_length'] = float('inf')
                metrics['diameter'] = float('inf')
        
        return {
            'network_metrics': metrics,
            'centrality_measures': {
                'degree': degree_centrality if 'degree_centrality' in locals() else {},
                'betweenness': betweenness_centrality if 'betweenness_centrality' in locals() else {},
                'eigenvector': eigenvector_centrality if 'eigenvector_centrality' in locals() else {}
            },
            'systemic_risk_score': self._calculate_systemic_risk_score(metrics)
        }
    
    def _calculate_systemic_risk_score(self, network_metrics: Dict[str, Any]) -> float:
        """Calculate systemic risk score from network metrics."""
        
        if not network_metrics:
            return 0.0
        
        # Simple scoring based on network connectivity
        density_score = network_metrics.get('density', 0) * 40
        clustering_score = network_metrics.get('average_clustering', 0) * 30
        path_length_score = 20 if network_metrics.get('average_path_length', float('inf')) < 3 else 0
        
        return density_score + clustering_score + path_length_score
    
    def create_risk_report(
        self,
        economic_data: pd.DataFrame,
        portfolio_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Create comprehensive risk analysis report.
        
        Args:
            economic_data: DataFrame with economic indicators
            portfolio_weights: Portfolio weights for analysis
            
        Returns:
            Dictionary with risk report
        """
        
        report = {
            'analysis_date': datetime.now().isoformat(),
            'risk_summary': {},
            'var_analysis': {},
            'stress_test_results': {},
            'network_analysis': {},
            'recommendations': []
        }
        
        # Calculate returns
        returns_data = economic_data.pct_change().dropna()
        
        # Portfolio VaR analysis
        if portfolio_weights:
            var_results = self.calculate_economic_var(returns_data, portfolio_weights)
            report['var_analysis'] = var_results
        
        # Individual risk metrics
        individual_metrics = {}
        for factor in returns_data.columns:
            individual_metrics[factor] = self.calculate_risk_metrics(returns_data[factor])
        
        report['individual_metrics'] = individual_metrics
        
        # Stress testing
        current_values = economic_data.iloc[-1].to_dict()
        stress_results = {}
        
        for scenario_name in ['recession', 'stagflation', 'financial_crisis']:
            stress_results[scenario_name] = self.stress_test(current_values, scenario_name)
        
        report['stress_test_results'] = stress_results
        
        # Network analysis
        correlation_matrix = returns_data.corr()
        network_analysis = self.network_risk_analysis(correlation_matrix)
        report['network_analysis'] = network_analysis
        
        # Generate recommendations
        report['recommendations'] = self._generate_risk_recommendations(
            var_results if portfolio_weights else {},
            stress_results,
            network_analysis
        )
        
        return report
    
    def _generate_risk_recommendations(
        self,
        var_results: Dict[str, Any],
        stress_results: Dict[str, Any],
        network_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate risk management recommendations."""
        
        recommendations = []
        
        # VaR-based recommendations
        if var_results:
            portfolio_var = var_results.get('portfolio_var', {}).get('var', 0)
            if abs(portfolio_var) > 0.15:  # 15% VaR threshold
                recommendations.append("Consider reducing portfolio exposure due to high Value at Risk")
        
        # Stress test recommendations
        worst_stress = max(stress_results.items(), key=lambda x: abs(x[1].get('portfolio_impact', 0)))
        if abs(worst_stress[1].get('portfolio_impact', 0)) > 0.20:  # 20% stress loss threshold
            recommendations.append(f"High vulnerability to {worst_stress[0]} scenario - implement hedging strategies")
        
        # Network-based recommendations
        systemic_score = network_analysis.get('systemic_risk_score', 0)
        if systemic_score > 60:
            recommendations.append("High systemic risk detected - increase diversification across uncorrelated factors")
        
        if not recommendations:
            recommendations.append("Risk levels appear manageable with current portfolio composition")
        
        return recommendations
