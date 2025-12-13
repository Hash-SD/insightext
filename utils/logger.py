"""Logging utility for MLOps Streamlit Text AI application."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    log_format = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    
    # File handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    
    return logger


def log_error(logger: logging.Logger, error: Exception, context: Optional[dict] = None) -> None:
    """Log error with stack trace and context information."""
    error_message = f"Error: {type(error).__name__} - {str(error)}"
    
    if context:
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        error_message += f" | Context: {context_str}"
    
    logger.error(error_message, exc_info=True)
