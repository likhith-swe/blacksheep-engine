"""
Black Sheep Ventures - Logging Configuration
Unified logging management supporting synchronized file rotation and console streams.
"""

import os
import sys
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler


def _ensure_utf8_stdout():
    """Ensure standard output and error streams use UTF-8 encoding."""
    if sys.platform == 'win32':
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# Target log directory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')


def setup_logger(name: str = 'blacksheep', level: int = logging.DEBUG) -> logging.Logger:
    """
    Configure and initialize application logger.
    
    Args:
        name: Logger name.
        level: Root logging level.
        
    Returns:
        Configured logger instance.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent propagation to avoid duplicate console entries
    logger.propagate = False
    
    if logger.handlers:
        return logger
    
    # Formatters
    detailed_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 1. Rotating daily file handler (DEBUG and above)
    log_filename = datetime.now().strftime('%Y-%m-%d') + '.log'
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, log_filename),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # 2. Console stream handler (INFO and above)
    _ensure_utf8_stdout()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = 'blacksheep') -> logging.Logger:
    """Retrieve logger instance by name (creates if not exists)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


# Default logger instance
logger = setup_logger()


# Convenience methods
def debug(msg: str, *args, **kwargs) -> None:
    logger.debug(msg, *args, **kwargs)

def info(msg: str, *args, **kwargs) -> None:
    logger.info(msg, *args, **kwargs)

def warning(msg: str, *args, **kwargs) -> None:
    logger.warning(msg, *args, **kwargs)

def error(msg: str, *args, **kwargs) -> None:
    logger.error(msg, *args, **kwargs)

def critical(msg: str, *args, **kwargs) -> None:
    logger.critical(msg, *args, **kwargs)
