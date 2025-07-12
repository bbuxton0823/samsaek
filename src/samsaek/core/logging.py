"""
Samsaek Logging Configuration
Structured logging setup for the Samsaek Multi-Agent AI System
"""

import sys
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import structlog
from structlog.stdlib import LoggerFactory
from structlog.processors import JSONRenderer
from structlog.dev import ConsoleRenderer

from .config import config, LogLevel


class SamsaekLogger:
    """
    Centralized logging configuration for Samsaek
    Provides structured logging with context preservation
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.setup_logging()
            self._initialized = True
    
    def setup_logging(self):
        """Configure structured logging for Samsaek"""
        
        # Configure standard library logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, config.log_level.value),
        )
        
        # Define processors based on environment
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            self._add_samsaek_context,
        ]
        
        # Add appropriate renderer based on configuration
        if config.log_format == "json" or config.is_production:
            processors.append(JSONRenderer())
        else:
            processors.append(ConsoleRenderer())
        
        # Configure structlog
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Configure file logging if specified
        if config.log_file:
            self._setup_file_logging()
    
    def _add_samsaek_context(self, logger, method_name, event_dict):
        """Add Samsaek-specific context to log entries"""
        event_dict["platform"] = "samsaek"
        event_dict["version"] = config.app_version
        event_dict["environment"] = config.environment.value
        
        # Add deployment context if available
        if config.k8s_namespace:
            event_dict["k8s_namespace"] = config.k8s_namespace
        
        return event_dict
    
    def _setup_file_logging(self):
        """Setup file logging handler"""
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setLevel(getattr(logging, config.log_level.value))
        
        # Use JSON format for file logging
        file_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s", '
            '"platform": "samsaek", "version": "' + config.app_version + '"}'
        )
        file_handler.setFormatter(file_formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(file_handler)
    
    def get_logger(self, name: str) -> structlog.stdlib.BoundLogger:
        """Get a structured logger instance"""
        return structlog.get_logger(name)
    
    def bind_context(self, logger: structlog.stdlib.BoundLogger, **kwargs) -> structlog.stdlib.BoundLogger:
        """Bind additional context to logger"""
        return logger.bind(**kwargs)


class SamsaekLoggerMixin:
    """
    Mixin class to add logging capabilities to any class
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__class__.__name__)
    
    def log_operation(self, operation: str, **context):
        """Log an operation with context"""
        self.logger.info(
            f"Starting {operation}",
            operation=operation,
            **context
        )
    
    def log_success(self, operation: str, duration: Optional[float] = None, **context):
        """Log successful operation"""
        log_data = {
            "operation": operation,
            "status": "success",
            **context
        }
        if duration:
            log_data["duration_seconds"] = duration
        
        self.logger.info(f"Completed {operation}", **log_data)
    
    def log_error(self, operation: str, error: Exception, **context):
        """Log operation error"""
        self.logger.error(
            f"Failed {operation}",
            operation=operation,
            status="error",
            error_type=type(error).__name__,
            error_message=str(error),
            **context
        )
    
    def log_warning(self, message: str, **context):
        """Log warning message"""
        self.logger.warning(message, **context)
    
    def log_debug(self, message: str, **context):
        """Log debug message"""
        self.logger.debug(message, **context)


# Global logger instance
_logger_instance = SamsaekLogger()


def setup_logging():
    """Initialize logging configuration"""
    global _logger_instance
    if not _logger_instance._initialized:
        _logger_instance.setup_logging()


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance"""
    return _logger_instance.get_logger(name)


def bind_context(logger: structlog.stdlib.BoundLogger, **kwargs) -> structlog.stdlib.BoundLogger:
    """Bind additional context to logger"""
    return _logger_instance.bind_context(logger, **kwargs)


# Performance monitoring decorator
def log_performance(operation_name: str = None):
    """
    Decorator to log performance metrics for operations
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            logger = get_logger("performance")
            
            start_time = time.time()
            logger.info(f"Starting {op_name}", operation=op_name)
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(
                    f"Completed {op_name}",
                    operation=op_name,
                    status="success",
                    duration_seconds=duration
                )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                logger.error(
                    f"Failed {op_name}",
                    operation=op_name,
                    status="error",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    duration_seconds=duration
                )
                
                raise
        
        return wrapper
    return decorator


# Context manager for operation logging
class LogOperation:
    """Context manager for logging operations with automatic success/failure tracking"""
    
    def __init__(self, operation_name: str, logger: structlog.stdlib.BoundLogger = None, **context):
        self.operation_name = operation_name
        self.logger = logger or get_logger("operation")
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation_name}", operation=self.operation_name, **self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation_name}",
                operation=self.operation_name,
                status="success",
                duration_seconds=duration,
                **self.context
            )
        else:
            self.logger.error(
                f"Failed {self.operation_name}",
                operation=self.operation_name,
                status="error",
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                duration_seconds=duration,
                **self.context
            )
        
        return False  # Don't suppress exceptions