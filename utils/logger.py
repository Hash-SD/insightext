"""
Logging utility untuk aplikasi MLOps Streamlit Text AI.
Menyediakan fungsi untuk setup logger dengan file dan console handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str,
    log_file: str,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger dengan file handler dan console handler.
    
    Args:
        name: Nama logger
        log_file: Path ke file log
        level: Level logging (default: logging.INFO)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
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


def log_error(
    logger: logging.Logger,
    error: Exception,
    context: Optional[dict] = None
) -> None:
    """
    Log error dengan stack trace dan context information.
    
    Args:
        logger: Logger instance
        error: Exception yang akan di-log
        context: Dictionary dengan context information tambahan
    """
    error_message = f"Error: {type(error).__name__} - {str(error)}"
    
    if context:
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        error_message += f" | Context: {context_str}"
    
    logger.error(error_message, exc_info=True)
