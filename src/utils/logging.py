import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

from src.config.settings import settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: bool = True,
    rich_formatting: bool = True
) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (default from settings)
        log_file: Log file path (default from settings)
        console_output: Enable console output
        rich_formatting: Use rich formatting for console output
    """
    log_level = log_level or settings.log_level
    log_file = log_file or settings.log_file
    
    # Create log directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Add console handler
    if console_output:
        if rich_formatting:
            console_handler = RichHandler(
                console=Console(stderr=True),
                show_time=True,
                show_path=False,
                markup=True
            )
            console_handler.setFormatter(logging.Formatter("%(message)s"))
        else:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
        
        console_handler.setLevel(log_level)
        root_logger.addHandler(console_handler)
    
    # Add file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)
    
    # Configure third-party loggers
    logging.getLogger("watchdog").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Initialize logging on module import
setup_logging()