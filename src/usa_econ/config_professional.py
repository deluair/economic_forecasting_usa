"""
Professional Configuration Management for Economic Intelligence Platform
========================================================================

Enterprise-grade configuration system with validation, security, and production settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')


class Environment(Enum):
    """Environment types for deployment."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Logging levels for the platform."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class APIConfiguration:
    """API configuration settings."""
    fred_api_key: Optional[str] = None
    bls_api_key: Optional[str] = None
    census_api_key: Optional[str] = None
    eia_api_key: Optional[str] = None
    alpha_vantage_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    news_api_key: Optional[str] = None
    twitter_api_key: Optional[str] = None
    twitter_api_secret: Optional[str] = None
    twitter_access_token: Optional[str] = None
    twitter_access_token_secret: Optional[str] = None
    
    def validate(self) -> List[str]:
        """Validate API configuration."""
        issues = []
        
        if not self.fred_api_key:
            issues.append("FRED API key is required for economic data")
        
        if not self.openai_api_key:
            issues.append("OpenAI API key is required for AI narratives")
        
        return issues


@dataclass
class ModelConfiguration:
    """Model configuration and parameters."""
    # Forecast settings
    default_forecast_horizon: int = 12
    default_confidence_level: float = 0.95
    ensemble_models: List[str] = field(default_factory=lambda: ['arima', 'prophet', 'rf'])
    
    # Advanced models
    vecm_max_lags: int = 4
    bayesian_var_draws: int = 2000
    markov_switching_regimes: int = 3
    dynamic_factor_factors: int = 3
    
    # Risk management
    var_confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    stress_scenarios: List[str] = field(default_factory=lambda: ['recession', 'stagflation', 'financial_crisis'])
    monte_carlo_simulations: int = 10000
    
    # Performance settings
    max_parallel_jobs: int = 4
    cache_enabled: bool = True
    cache_ttl_hours: int = 24
    
    def validate(self) -> List[str]:
        """Validate model configuration."""
        issues = []
        
        if self.default_forecast_horizon < 1 or self.default_forecast_horizon > 60:
            issues.append("Forecast horizon must be between 1 and 60 periods")
        
        if not 0.8 <= self.default_confidence_level <= 0.999:
            issues.append("Confidence level must be between 0.8 and 0.999")
        
        if self.monte_carlo_simulations < 1000:
            issues.append("Monte Carlo simulations should be at least 1000 for reliable results")
        
        return issues


@dataclass
class DataConfiguration:
    """Data management configuration."""
    # Data sources
    primary_sources: List[str] = field(default_factory=lambda: ['fred', 'bls', 'census'])
    real_time_sources: List[str] = field(default_factory=lambda: ['yahoo_finance', 'news_api'])
    
    # Quality settings
    min_observations: int = 50
    max_missing_percentage: float = 0.05
    outlier_detection: bool = True
    stationarity_tests: bool = True
    
    # Storage settings
    data_directory: Path = field(default_factory=lambda: Path("data"))
    backup_enabled: bool = True
    backup_retention_days: int = 90
    compression_enabled: bool = True
    
    def validate(self) -> List[str]:
        """Validate data configuration."""
        issues = []
        
        if self.min_observations < 30:
            issues.append("Minimum observations should be at least 30 for reliable modeling")
        
        if self.max_missing_percentage > 0.2:
            issues.append("Maximum missing percentage should not exceed 20%")
        
        return issues


@dataclass
class SecurityConfiguration:
    """Security and compliance settings."""
    # Access control
    authentication_enabled: bool = True
    api_rate_limit: int = 1000  # requests per hour
    session_timeout_minutes: int = 60
    
    # Data protection
    encryption_enabled: bool = True
    audit_logging: bool = True
    data_retention_days: int = 2555  # 7 years
    
    # Compliance
    gdpr_compliance: bool = True
    soc2_compliance: bool = True
    finra_compliance: bool = False
    
    def validate(self) -> List[str]:
        """Validate security configuration."""
        issues = []
        
        if self.api_rate_limit > 10000:
            issues.append("API rate limit should not exceed 10000 requests per hour")
        
        if self.data_retention_days < 365:
            issues.append("Data retention should be at least 1 year for compliance")
        
        return issues


@dataclass
class PerformanceConfiguration:
    """Performance and scaling settings."""
    # System resources
    max_memory_mb: int = 4096
    max_cpu_cores: int = 8
    thread_pool_size: int = 4
    
    # Caching
    redis_enabled: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    cache_ttl_seconds: int = 3600
    
    # Monitoring
    metrics_enabled: bool = True
    health_check_interval: int = 60
    performance_logging: bool = True
    
    def validate(self) -> List[str]:
        """Validate performance configuration."""
        issues = []
        
        if self.max_memory_mb < 1024:
            issues.append("Maximum memory should be at least 1024 MB")
        
        if self.thread_pool_size > self.max_cpu_cores * 2:
            issues.append("Thread pool size should not exceed 2x CPU cores")
        
        return issues


@dataclass
class ProfessionalConfig:
    """Professional platform configuration."""
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    log_level: LogLevel = LogLevel.INFO
    debug_mode: bool = False
    
    # Component configurations
    api: APIConfiguration = field(default_factory=APIConfiguration)
    models: ModelConfiguration = field(default_factory=ModelConfiguration)
    data: DataConfiguration = field(default_factory=DataConfiguration)
    security: SecurityConfiguration = field(default_factory=SecurityConfiguration)
    performance: PerformanceConfiguration = field(default_factory=PerformanceConfiguration)
    
    # Project settings
    project_root: Path = field(default_factory=lambda: Path.cwd())
    config_version: str = "1.0.0"
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Ensure data directory exists
        self.data.data_directory.mkdir(parents=True, exist_ok=True)
        
        # Set environment-specific defaults
        if self.environment == Environment.PRODUCTION:
            self.debug_mode = False
            self.log_level = LogLevel.INFO
            self.security.audit_logging = True
            self.performance.metrics_enabled = True
    
    @classmethod
    def from_env(cls) -> ProfessionalConfig:
        """Load configuration from environment variables."""
        # Load environment variables
        load_dotenv()
        
        # Create configuration
        config = cls()
        
        # Environment settings
        env_str = os.getenv('ENVIRONMENT', 'development').lower()
        config.environment = Environment(env_str) if env_str in [e.value for e in Environment] else Environment.DEVELOPMENT
        
        log_str = os.getenv('LOG_LEVEL', 'INFO').upper()
        config.log_level = LogLevel(log_str) if log_str in [l.value for l in LogLevel] else LogLevel.INFO
        config.debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        
        # API configuration
        config.api.fred_api_key = os.getenv('FRED_API_KEY')
        config.api.bls_api_key = os.getenv('BLS_API_KEY')
        config.api.census_api_key = os.getenv('CENSUS_API_KEY')
        config.api.eia_api_key = os.getenv('EIA_API_KEY')
        config.api.alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        config.api.openai_api_key = os.getenv('OPENAI_API_KEY')
        config.api.news_api_key = os.getenv('NEWS_API_KEY')
        config.api.twitter_api_key = os.getenv('TWITTER_API_KEY')
        config.api.twitter_api_secret = os.getenv('TWITTER_API_SECRET')
        config.api.twitter_access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        config.api.twitter_access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        
        # Model configuration
        config.models.default_forecast_horizon = int(os.getenv('DEFAULT_FORECAST_HORIZON', '12'))
        config.models.default_confidence_level = float(os.getenv('DEFAULT_CONFIDENCE_LEVEL', '0.95'))
        config.models.monte_carlo_simulations = int(os.getenv('MONTE_CARLO_SIMULATIONS', '10000'))
        config.models.cache_enabled = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
        
        # Data configuration
        config.data.min_observations = int(os.getenv('MIN_OBSERVATIONS', '50'))
        config.data.max_missing_percentage = float(os.getenv('MAX_MISSING_PERCENTAGE', '0.05'))
        config.data.backup_enabled = os.getenv('BACKUP_ENABLED', 'true').lower() == 'true'
        
        # Security configuration
        config.security.authentication_enabled = os.getenv('AUTHENTICATION_ENABLED', 'true').lower() == 'true'
        config.security.api_rate_limit = int(os.getenv('API_RATE_LIMIT', '1000'))
        config.security.encryption_enabled = os.getenv('ENCRYPTION_ENABLED', 'true').lower() == 'true'
        
        # Performance configuration
        config.performance.max_memory_mb = int(os.getenv('MAX_MEMORY_MB', '4096'))
        config.performance.max_cpu_cores = int(os.getenv('MAX_CPU_CORES', '8'))
        config.performance.metrics_enabled = os.getenv('METRICS_ENABLED', 'true').lower() == 'true'
        
        return config
    
    @classmethod
    def from_file(cls, config_path: Path) -> ProfessionalConfig:
        """Load configuration from file (JSON or YAML)."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        # Convert to configuration object
        config = cls()
        
        # Update configuration with file data
        if 'environment' in data:
            config.environment = Environment(data['environment'])
        if 'log_level' in data:
            config.log_level = LogLevel(data['log_level'])
        if 'debug_mode' in data:
            config.debug_mode = data['debug_mode']
        
        # Update nested configurations
        if 'api' in data:
            for key, value in data['api'].items():
                setattr(config.api, key, value)
        
        if 'models' in data:
            for key, value in data['models'].items():
                setattr(config.models, key, value)
        
        if 'data' in data:
            for key, value in data['data'].items():
                if key == 'data_directory':
                    setattr(config.data, key, Path(value))
                else:
                    setattr(config.data, key, value)
        
        if 'security' in data:
            for key, value in data['security'].items():
                setattr(config.security, key, value)
        
        if 'performance' in data:
            for key, value in data['performance'].items():
                setattr(config.performance, key, value)
        
        return config
    
    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to file."""
        # Convert to dictionary
        config_dict = {
            'environment': self.environment.value,
            'log_level': self.log_level.value,
            'debug_mode': self.debug_mode,
            'config_version': self.config_version,
            'api': {
                'fred_api_key': self.api.fred_api_key,
                'bls_api_key': self.api.bls_api_key,
                'census_api_key': self.api.census_api_key,
                'eia_api_key': self.api.eia_api_key,
                'alpha_vantage_api_key': self.api.alpha_vantage_api_key,
                'openai_api_key': self.api.openai_api_key,
                'news_api_key': self.api.news_api_key,
                'twitter_api_key': self.api.twitter_api_key,
                'twitter_api_secret': self.api.twitter_api_secret,
                'twitter_access_token': self.api.twitter_access_token,
                'twitter_access_token_secret': self.api.twitter_access_token_secret
            },
            'models': {
                'default_forecast_horizon': self.models.default_forecast_horizon,
                'default_confidence_level': self.models.default_confidence_level,
                'ensemble_models': self.models.ensemble_models,
                'vecm_max_lags': self.models.vecm_max_lags,
                'bayesian_var_draws': self.models.bayesian_var_draws,
                'markov_switching_regimes': self.models.markov_switching_regimes,
                'dynamic_factor_factors': self.models.dynamic_factor_factors,
                'var_confidence_levels': self.models.var_confidence_levels,
                'stress_scenarios': self.models.stress_scenarios,
                'monte_carlo_simulations': self.models.monte_carlo_simulations,
                'max_parallel_jobs': self.models.max_parallel_jobs,
                'cache_enabled': self.models.cache_enabled,
                'cache_ttl_hours': self.models.cache_ttl_hours
            },
            'data': {
                'primary_sources': self.data.primary_sources,
                'real_time_sources': self.data.real_time_sources,
                'min_observations': self.data.min_observations,
                'max_missing_percentage': self.data.max_missing_percentage,
                'outlier_detection': self.data.outlier_detection,
                'stationarity_tests': self.data.stationarity_tests,
                'data_directory': str(self.data.data_directory),
                'backup_enabled': self.data.backup_enabled,
                'backup_retention_days': self.data.backup_retention_days,
                'compression_enabled': self.data.compression_enabled
            },
            'security': {
                'authentication_enabled': self.security.authentication_enabled,
                'api_rate_limit': self.security.api_rate_limit,
                'session_timeout_minutes': self.security.session_timeout_minutes,
                'encryption_enabled': self.security.encryption_enabled,
                'audit_logging': self.security.audit_logging,
                'data_retention_days': self.security.data_retention_days,
                'gdpr_compliance': self.security.gdpr_compliance,
                'soc2_compliance': self.security.soc2_compliance,
                'finra_compliance': self.security.finra_compliance
            },
            'performance': {
                'max_memory_mb': self.performance.max_memory_mb,
                'max_cpu_cores': self.performance.max_cpu_cores,
                'thread_pool_size': self.performance.thread_pool_size,
                'redis_enabled': self.performance.redis_enabled,
                'redis_host': self.performance.redis_host,
                'redis_port': self.performance.redis_port,
                'cache_ttl_seconds': self.performance.cache_ttl_seconds,
                'metrics_enabled': self.performance.metrics_enabled,
                'health_check_interval': self.performance.health_check_interval,
                'performance_logging': self.performance.performance_logging
            }
        }
        
        # Save to file
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            else:
                json.dump(config_dict, f, indent=2)
    
    def validate(self) -> Dict[str, List[str]]:
        """Validate entire configuration."""
        issues = {
            'api': self.api.validate(),
            'models': self.models.validate(),
            'data': self.data.validate(),
            'security': self.security.validate(),
            'performance': self.performance.validate()
        }
        
        # Filter out empty issue lists
        return {k: v for k, v in issues.items() if v}
    
    def is_production_ready(self) -> bool:
        """Check if configuration is production-ready."""
        if self.environment != Environment.PRODUCTION:
            return False
        
        validation_issues = self.validate()
        if validation_issues:
            return False
        
        # Check critical APIs
        if not self.api.fred_api_key:
            return False
        
        # Check security settings
        if not self.security.authentication_enabled:
            return False
        
        if not self.security.encryption_enabled:
            return False
        
        # Check monitoring
        if not self.performance.metrics_enabled:
            return False
        
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary for display."""
        return {
            'environment': self.environment.value,
            'log_level': self.log_level.value,
            'debug_mode': self.debug_mode,
            'production_ready': self.is_production_ready(),
            'api_keys_configured': {
                'fred': bool(self.api.fred_api_key),
                'openai': bool(self.api.openai_api_key),
                'bls': bool(self.api.bls_api_key),
                'census': bool(self.api.census_api_key)
            },
            'model_settings': {
                'forecast_horizon': self.models.default_forecast_horizon,
                'confidence_level': self.models.default_confidence_level,
                'monte_carlo_simulations': self.models.monte_carlo_simulations,
                'cache_enabled': self.models.cache_enabled
            },
            'data_settings': {
                'min_observations': self.data.min_observations,
                'backup_enabled': self.data.backup_enabled,
                'data_directory': str(self.data.data_directory)
            },
            'security_settings': {
                'authentication': self.security.authentication_enabled,
                'encryption': self.security.encryption_enabled,
                'audit_logging': self.security.audit_logging
            }
        }


def load_professional_config() -> ProfessionalConfig:
    """Load professional configuration."""
    # Try to load from file first
    config_paths = [
        Path("config/professional.yaml"),
        Path("config/professional.json"),
        Path("professional_config.yaml"),
        Path("professional_config.json")
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            return ProfessionalConfig.from_file(config_path)
    
    # Fall back to environment variables
    return ProfessionalConfig.from_env()


def create_production_config_template() -> ProfessionalConfig:
    """Create production configuration template."""
    config = ProfessionalConfig()
    config.environment = Environment.PRODUCTION
    config.log_level = LogLevel.INFO
    config.debug_mode = False
    
    # Production security settings
    config.security.authentication_enabled = True
    config.security.encryption_enabled = True
    config.security.audit_logging = True
    config.security.api_rate_limit = 1000
    
    # Production performance settings
    config.performance.metrics_enabled = True
    config.performance.max_memory_mb = 8192
    config.performance.max_cpu_cores = 16
    
    # Production model settings
    config.models.monte_carlo_simulations = 10000
    config.models.cache_enabled = True
    config.models.max_parallel_jobs = 8
    
    return config
