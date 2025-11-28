"""
Centralized Configuration Management
Environment-aware settings with validation
"""

from pydantic import BaseSettings, validator, Field
from typing import Optional, List
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings with validation"""
    
    # Environment
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # API Keys (Free)
    FRED_API_KEY: str = Field(default="", env="FRED_API_KEY")
    NEWSAPI_KEY: str = Field(default="", env="NEWSAPI_KEY")
    ALPHA_VANTAGE_KEY: str = Field(default="demo", env="ALPHA_VANTAGE_KEY")
    
    # Future Premium APIs
    REFINITIV_APP_KEY: Optional[str] = Field(default=None, env="REFINITIV_APP_KEY")
    BLOOMBERG_API_KEY: Optional[str] = Field(default=None, env="BLOOMBERG_API_KEY")
    
    # Database URLs
    MONGODB_URI: str = Field(
        default="mongodb://localhost:27017/jpm_dashboard",
        env="MONGODB_URI"
    )
    POSTGRES_URI: str = Field(
        default="postgresql://jpm_user:password@localhost:5432/jpm_timeseries",
        env="POSTGRES_URI"
    )
    REDIS_URI: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URI"
    )
    
    # Email Alerts
    SMTP_EMAIL: Optional[str] = Field(default=None, env="SMTP_EMAIL")
    SMTP_PASSWORD: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    SMTP_SERVER: str = Field(default="smtp.gmail.com", env="SMTP_SERVER")
    SMTP_PORT: int = Field(default=587, env="SMTP_PORT")
    
    # Cache Settings
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    DATA_REFRESH_INTERVAL: int = Field(default=300, env="DATA_REFRESH_INTERVAL")  # 5 min
    
    # Rate Limiting (seconds)
    YAHOO_FINANCE_DELAY: float = Field(default=0.5, env="YAHOO_FINANCE_DELAY")
    FRED_DELAY: float = Field(default=0.5, env="FRED_DELAY")
    ALPHA_VANTAGE_DELAY: float = Field(default=12.0, env="ALPHA_VANTAGE_DELAY")
    NEWSAPI_DELAY: float = Field(default=1.0, env="NEWSAPI_DELAY")
    
    # Data Quality Thresholds
    MAX_MISSING_DATA_PCT: float = Field(default=10.0)  # Max 10% missing
    OUTLIER_STD_THRESHOLD: float = Field(default=5.0)  # 5 std dev
    MIN_DATA_POINTS: int = Field(default=30)  # Minimum data points for analysis
    
    # Risk Analytics
    VAR_CONFIDENCE_LEVEL: float = Field(default=0.95)
    RISK_FREE_RATE: float = Field(default=0.02)  # 2% annual
    
    # Backtest Settings
    BACKTEST_START_DATE: str = Field(default="2020-01-01")
    TRANSACTION_COST_BPS: float = Field(default=5.0)  # 5 basis points
    SLIPPAGE_BPS: float = Field(default=2.0)  # 2 basis points
    
    # Application Paths
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    CACHE_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data" / "cache")
    LOG_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    
    # Streamlit Config
    STREAMLIT_SERVER_PORT: int = Field(default=8501)
    STREAMLIT_SERVER_ADDRESS: str = Field(default="0.0.0.0")
    
    # FastAPI Config
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_WORKERS: int = Field(default=4)
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of {valid_levels}")
        return v.upper()
    
    @validator("FRED_API_KEY")
    def validate_fred_key(cls, v):
        if v and v in ["demo", "your_fred_api_key_here", ""]:
            import warnings
            warnings.warn("FRED API key not configured. Some features will be limited.")
        return v
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        valid_envs = ["development", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"Invalid environment. Must be one of {valid_envs}")
        return v
    
    def create_directories(self):
        """Create necessary directories"""
        for dir_path in [self.DATA_DIR, self.CACHE_DIR, self.LOG_DIR,
                         self.DATA_DIR / "raw", self.DATA_DIR / "processed",
                         self.DATA_DIR / "exports"]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Singleton instance
settings = Settings()
settings.create_directories()

# Export commonly used values
BASE_DIR = settings.BASE_DIR
DATA_DIR = settings.DATA_DIR
CACHE_DIR = settings.CACHE_DIR
LOG_DIR = settings.LOG_DIR

if __name__ == "__main__":
    # Test settings
    print("="*70)
    print("CONFIGURATION VALIDATION")
    print("="*70)
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Debug Mode: {settings.DEBUG}")
    print(f"Log Level: {settings.LOG_LEVEL}")
    print(f"\nAPI Keys Configured:")
    print(f"  FRED: {'✅' if settings.FRED_API_KEY and settings.FRED_API_KEY != 'demo' else '❌'}")
    print(f"  NewsAPI: {'✅' if settings.NEWSAPI_KEY and settings.NEWSAPI_KEY != 'demo' else '❌'}")
    print(f"  Alpha Vantage: {'✅' if settings.ALPHA_VANTAGE_KEY != 'demo' else '❌'}")
    print(f"\nDatabase URLs:")
    print(f"  MongoDB: {settings.MONGODB_URI}")
    print(f"  PostgreSQL: {settings.POSTGRES_URI}")
    print(f"  Redis: {settings.REDIS_URI}")
    print(f"\nDirectories:")
    print(f"  Base: {settings.BASE_DIR}")
    print(f"  Data: {settings.DATA_DIR}")
    print(f"  Cache: {settings.CACHE_DIR}")
    print(f"  Logs: {settings.LOG_DIR}")
    print("="*70)