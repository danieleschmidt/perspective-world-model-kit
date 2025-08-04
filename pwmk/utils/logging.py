"""Logging utilities for PWMK."""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


class PWMKFormatter(logging.Formatter):
    """Custom formatter for PWMK logs with structured output."""
    
    def __init__(self, include_structured: bool = True):
        super().__init__()
        self.include_structured = include_structured
    
    def format(self, record: logging.LogRecord) -> str:
        # Basic formatting
        timestamp = datetime.fromtimestamp(record.created).isoformat()
        
        # Get extra fields
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in [
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                'filename', 'module', 'lineno', 'funcName', 'created', 
                'msecs', 'relativeCreated', 'thread', 'threadName', 
                'processName', 'process', 'getMessage', 'stack_info',
                'exc_info', 'exc_text', 'message'
            ]:
                extra_fields[key] = value
        
        # Create structured log entry
        log_entry = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if extra_fields:
            log_entry["extra"] = extra_fields
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        if self.include_structured:
            return json.dumps(log_entry, default=str)
        else:
            # Human-readable format
            message = f"[{timestamp}] {record.levelname} {record.name}: {record.getMessage()}"
            if extra_fields:
                message += f" {extra_fields}"
            return message


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = False,
    console: bool = True
) -> None:
    """
    Setup logging configuration for PWMK.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        structured: Whether to use structured JSON logging
        console: Whether to log to console
    """
    # Clear existing handlers
    logging.getLogger().handlers.clear()
    
    # Set level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger().setLevel(numeric_level)
    
    formatter = PWMKFormatter(include_structured=structured)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(numeric_level)
        logging.getLogger().addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ContextLogger:
    """Logger with context information for structured logging."""
    
    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        self.logger = logger
        self.context = context
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with context information."""
        extra = {**self.context, **kwargs}
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log_with_context(logging.CRITICAL, message, **kwargs)


def get_context_logger(name: str, **context) -> ContextLogger:
    """
    Get a context logger with additional fields.
    
    Args:
        name: Logger name
        **context: Context fields to include in all log messages
        
    Returns:
        ContextLogger instance
    """
    base_logger = get_logger(name)
    return ContextLogger(base_logger, context)


class LoggingMixin:
    """Mixin class to add logging capabilities to other classes."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        return self._logger
    
    def get_context_logger(self, **context) -> ContextLogger:
        """Get context logger with class information."""
        class_context = {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__
        }
        full_context = {**class_context, **context}
        return ContextLogger(self.logger, full_context)


# Configure default logging
setup_logging(level="INFO", structured=False, console=True)