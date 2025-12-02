"""
Configuration module for MLOps Streamlit Text AI application.

This module exports the global settings instance for use throughout the application.
"""

from config.settings import settings, Settings, reload_settings

__all__ = ['settings', 'Settings', 'reload_settings']
