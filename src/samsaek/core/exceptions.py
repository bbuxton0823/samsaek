"""
Samsaek Exception Classes
Custom exception hierarchy for the Samsaek Multi-Agent AI System
"""

from typing import Optional, Dict, Any
from datetime import datetime


class SamsaekException(Exception):
    """
    Base exception class for all Samsaek-related errors
    Provides structured error handling with context preservation
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.now().isoformat()
        
        # Add Samsaek-specific context
        self.context.update({
            "platform": "samsaek",
            "error_type": self.__class__.__name__,
            "timestamp": self.timestamp
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
            "timestamp": self.timestamp
        }
    
    def __str__(self):
        return f"[{self.error_code}] {self.message}"


class ConfigurationError(SamsaekException):
    """Raised when there's a configuration issue"""
    pass


class ValidationError(SamsaekException):
    """Raised when input validation fails"""
    pass


class AuthenticationError(SamsaekException):
    """Raised when authentication fails"""
    pass


class AuthorizationError(SamsaekException):
    """Raised when authorization fails"""
    pass


class ServiceUnavailableError(SamsaekException):
    """Raised when a required service is unavailable"""
    pass


class RateLimitExceededError(SamsaekException):
    """Raised when rate limits are exceeded"""
    pass


class TimeoutError(SamsaekException):
    """Raised when operations timeout"""
    pass


# AI Service Exceptions
class AIServiceError(SamsaekException):
    """Base exception for AI service-related errors"""
    pass


class GeminiServiceError(AIServiceError):
    """Raised when Google Gemini service encounters an error"""
    pass


class GeminiRateLimitError(GeminiServiceError):
    """Raised when Gemini API rate limits are exceeded"""
    pass


class GeminiAuthenticationError(GeminiServiceError):
    """Raised when Gemini API authentication fails"""
    pass


class GeminiModelError(GeminiServiceError):
    """Raised when Gemini model operations fail"""
    pass


# A2A Protocol Exceptions
class A2AProtocolError(SamsaekException):
    """Base exception for A2A protocol-related errors"""
    pass


class A2AMessageError(A2AProtocolError):
    """Raised when A2A message processing fails"""
    pass


class A2ARoutingError(A2AProtocolError):
    """Raised when A2A message routing fails"""
    pass


class A2AEncryptionError(A2AProtocolError):
    """Raised when A2A message encryption/decryption fails"""
    pass


class A2ASignatureError(A2AProtocolError):
    """Raised when A2A message signature verification fails"""
    pass


# Crew AI Exceptions
class CrewAIError(SamsaekException):
    """Base exception for Crew AI-related errors"""
    pass


class AgentError(CrewAIError):
    """Raised when agent operations fail"""
    pass


class TaskError(CrewAIError):
    """Raised when task execution fails"""
    pass


class WorkflowError(CrewAIError):
    """Raised when workflow orchestration fails"""
    pass


class AgentTimeoutError(CrewAIError):
    """Raised when agent operations timeout"""
    pass


class AgentCapacityError(CrewAIError):
    """Raised when agent capacity limits are exceeded"""
    pass


# Exa Search Exceptions
class ExaSearchError(SamsaekException):
    """Base exception for Exa search-related errors"""
    pass


class ExaAPIError(ExaSearchError):
    """Raised when Exa API operations fail"""
    pass


class ExaQueryError(ExaSearchError):
    """Raised when Exa query processing fails"""
    pass


class ExaRateLimitError(ExaSearchError):
    """Raised when Exa API rate limits are exceeded"""
    pass


# Weave Monitoring Exceptions
class WeaveError(SamsaekException):
    """Base exception for Weave monitoring-related errors"""
    pass


class WeaveInitializationError(WeaveError):
    """Raised when Weave initialization fails"""
    pass


class WeaveLoggingError(WeaveError):
    """Raised when Weave logging operations fail"""
    pass


class WeaveMetricsError(WeaveError):
    """Raised when Weave metrics collection fails"""
    pass


# Database Exceptions
class DatabaseError(SamsaekException):
    """Base exception for database-related errors"""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails"""
    pass


class DatabaseQueryError(DatabaseError):
    """Raised when database query execution fails"""
    pass


class DatabaseMigrationError(DatabaseError):
    """Raised when database migration fails"""
    pass


# Cache Exceptions
class CacheError(SamsaekException):
    """Base exception for cache-related errors"""
    pass


class CacheConnectionError(CacheError):
    """Raised when cache connection fails"""
    pass


class CacheOperationError(CacheError):
    """Raised when cache operations fail"""
    pass


# Message Queue Exceptions
class MessageQueueError(SamsaekException):
    """Base exception for message queue-related errors"""
    pass


class MessageQueueConnectionError(MessageQueueError):
    """Raised when message queue connection fails"""
    pass


class MessageQueuePublishError(MessageQueueError):
    """Raised when message publishing fails"""
    pass


class MessageQueueConsumerError(MessageQueueError):
    """Raised when message consumption fails"""
    pass


# API Exceptions
class APIError(SamsaekException):
    """Base exception for API-related errors"""
    pass


class APIValidationError(APIError):
    """Raised when API request validation fails"""
    pass


class APIAuthenticationError(APIError):
    """Raised when API authentication fails"""
    pass


class APIRateLimitError(APIError):
    """Raised when API rate limits are exceeded"""
    pass


class APINotFoundError(APIError):
    """Raised when API resource is not found"""
    pass


class APIServerError(APIError):
    """Raised when API server encounters an error"""
    pass


# Security Exceptions
class SecurityError(SamsaekException):
    """Base exception for security-related errors"""
    pass


class EncryptionError(SecurityError):
    """Raised when encryption operations fail"""
    pass


class DecryptionError(SecurityError):
    """Raised when decryption operations fail"""
    pass


class SignatureError(SecurityError):
    """Raised when signature operations fail"""
    pass


class CertificateError(SecurityError):
    """Raised when certificate operations fail"""
    pass


# Performance Exceptions
class PerformanceError(SamsaekException):
    """Base exception for performance-related errors"""
    pass


class MemoryError(PerformanceError):
    """Raised when memory limits are exceeded"""
    pass


class CPUError(PerformanceError):
    """Raised when CPU limits are exceeded"""
    pass


class NetworkError(PerformanceError):
    """Raised when network performance issues occur"""
    pass


# Utility functions for exception handling
def handle_exception(func):
    """
    Decorator to handle exceptions and convert them to SamsaekException
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SamsaekException:
            # Re-raise Samsaek exceptions as-is
            raise
        except Exception as e:
            # Convert other exceptions to SamsaekException
            raise SamsaekException(
                message=f"Unexpected error in {func.__name__}",
                error_code="UNEXPECTED_ERROR",
                context={"function": func.__name__},
                cause=e
            ) from e
    
    return wrapper


def create_error_response(exception: SamsaekException) -> Dict[str, Any]:
    """
    Create a standardized error response from a SamsaekException
    """
    return {
        "error": True,
        "error_code": exception.error_code,
        "message": exception.message,
        "context": exception.context,
        "timestamp": exception.timestamp
    }