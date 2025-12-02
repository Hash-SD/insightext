"""
Configuration management for MLOps Streamlit Text AI application.

This module provides centralized configuration using dataclass and environment variables.
All configuration values can be overridden using environment variables.
Supports Streamlit secrets for cloud deployment.
"""

import os
from dataclasses import dataclass, field
from typing import List
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Find .env file in project root
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path, override=True)
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

# Try to import streamlit for secrets support
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


def get_config_value(key: str, default: str = None) -> str:
    """
    Get configuration value dari environment variables atau Streamlit secrets.
    Priority: Environment variables > Streamlit secrets > Default
    
    Args:
        key: Configuration key
        default: Default value jika tidak ditemukan
        
    Returns:
        str: Configuration value
    """
    # Try environment variable first
    value = os.getenv(key)
    if value is not None:
        return value
    
    # Try Streamlit secrets if available
    if HAS_STREAMLIT and hasattr(st, 'secrets'):
        try:
            value = st.secrets.get(key)
            if value is not None:
                return str(value)
        except Exception:
            pass
    
    # Return default
    return default


@dataclass
class Settings:
    """
    Application settings with environment variable support.
    
    All settings can be overridden using environment variables.
    Default values are provided for local development.
    """
    
    # Database Configuration
    DATABASE_TYPE: str = field(
        default_factory=lambda: get_config_value('DATABASE_TYPE', 'sqlite')
    )
    DATABASE_URL: str = field(
        default_factory=lambda: get_config_value('DATABASE_URL', 'sqlite:///mlops_app.db')
    )
    
    # Supabase Configuration (jika DATABASE_TYPE=supabase)
    SUPABASE_URL: str = field(
        default_factory=lambda: get_config_value('SUPABASE_URL', '')
    )
    SUPABASE_KEY: str = field(
        default_factory=lambda: get_config_value('SUPABASE_KEY', '')
    )
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = field(
        default_factory=lambda: get_config_value('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    )
    MLFLOW_EXPERIMENT_NAME: str = field(
        default_factory=lambda: get_config_value('MLFLOW_EXPERIMENT_NAME', 'text-ai-system')
    )
    
    # Application Configuration
    APP_TITLE: str = field(
        default_factory=lambda: os.getenv('APP_TITLE', 'Sistem AI Berbasis Teks')
    )
    APP_ICON: str = field(
        default_factory=lambda: os.getenv('APP_ICON', '🤖')
    )
    MAX_INPUT_LENGTH: int = field(
        default_factory=lambda: int(os.getenv('MAX_INPUT_LENGTH', '5000'))
    )
    MIN_INPUT_LENGTH: int = field(
        default_factory=lambda: int(os.getenv('MIN_INPUT_LENGTH', '3'))
    )
    
    # Model Configuration - Naive Bayes models
    MODEL_VERSIONS: List[str] = field(
        default_factory=lambda: ['v1', 'v2']  # v1: Indonesian, v2: IMDB English
    )
    DEFAULT_MODEL_VERSION: str = field(
        default_factory=lambda: os.getenv('DEFAULT_MODEL_VERSION', 'v1')
    )
    
    # Logging Configuration
    LOG_FILE: str = field(
        default_factory=lambda: os.getenv('LOG_FILE', 'app.log')
    )
    LOG_LEVEL: str = field(
        default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO')
    )
    
    # Monitoring Configuration
    PREDICTION_HISTORY_LIMIT: int = field(
        default_factory=lambda: int(os.getenv('PREDICTION_HISTORY_LIMIT', '10'))
    )
    LATENCY_THRESHOLD_MS: float = field(
        default_factory=lambda: float(os.getenv('LATENCY_THRESHOLD_MS', '5000.0'))
    )
    
    # Database Retry Configuration
    DB_MAX_RETRIES: int = field(
        default_factory=lambda: int(os.getenv('DB_MAX_RETRIES', '3'))
    )
    DB_RETRY_DELAY: float = field(
        default_factory=lambda: float(os.getenv('DB_RETRY_DELAY', '1.0'))
    )
    
    # Privacy Configuration
    ENABLE_PII_DETECTION: bool = field(
        default_factory=lambda: os.getenv('ENABLE_PII_DETECTION', 'true').lower() == 'true'
    )
    
    # Admin Configuration
    ADMIN_PASSWORD: str = field(
        default_factory=lambda: get_config_value('ADMIN_PASSWORD', 'admin123secure')
    )
    
    def __post_init__(self):
        """Validate settings after initialization."""
        self._validate_settings()
    
    def _validate_settings(self):
        """Validate configuration values."""
        # Validate input length constraints
        if self.MIN_INPUT_LENGTH < 1:
            raise ValueError("MIN_INPUT_LENGTH must be at least 1")
        
        if self.MAX_INPUT_LENGTH < self.MIN_INPUT_LENGTH:
            raise ValueError("MAX_INPUT_LENGTH must be greater than MIN_INPUT_LENGTH")
        
        # Validate model version
        if self.DEFAULT_MODEL_VERSION not in self.MODEL_VERSIONS:
            raise ValueError(
                f"DEFAULT_MODEL_VERSION '{self.DEFAULT_MODEL_VERSION}' "
                f"must be one of {self.MODEL_VERSIONS}"
            )
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.LOG_LEVEL.upper() not in valid_log_levels:
            raise ValueError(
                f"LOG_LEVEL must be one of {valid_log_levels}, got '{self.LOG_LEVEL}'"
            )
        
        # Validate retry configuration
        if self.DB_MAX_RETRIES < 1:
            raise ValueError("DB_MAX_RETRIES must be at least 1")
        
        if self.DB_RETRY_DELAY < 0:
            raise ValueError("DB_RETRY_DELAY must be non-negative")
    
    def get_database_path(self) -> str:
        """
        Extract database file path from DATABASE_URL.
        
        Returns:
            str: Database file path for SQLite, or full URL for other databases
        """
        if self.DATABASE_URL.startswith('sqlite:///'):
            return self.DATABASE_URL.replace('sqlite:///', '')
        return self.DATABASE_URL
    
    def is_sqlite(self) -> bool:
        """
        Check if database is SQLite.
        
        Returns:
            bool: True if using SQLite, False otherwise
        """
        return self.DATABASE_URL.startswith('sqlite://')
    
    def is_postgresql(self) -> bool:
        """
        Check if database is PostgreSQL.
        
        Returns:
            bool: True if using PostgreSQL, False otherwise
        """
        return self.DATABASE_URL.startswith('postgresql://')
    
    def is_supabase(self) -> bool:
        """
        Check if database is Supabase (REST API).
        
        Returns:
            bool: True if using Supabase, False otherwise
        """
        return self.DATABASE_TYPE.lower() == 'supabase'


# Global settings instance
# This instance should be imported and used throughout the application
settings = Settings()


# Convenience function to reload settings (useful for testing)
def reload_settings() -> Settings:
    """
    Reload settings from environment variables.
    
    Returns:
        Settings: New settings instance with current environment variables
    """
    global settings
    settings = Settings()
    return settings
