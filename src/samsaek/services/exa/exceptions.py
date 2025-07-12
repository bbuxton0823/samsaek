"""
Exa Search Exception Classes
Custom exceptions for Exa search operations
"""

from ...core.exceptions import SamsaekException


class ExaSearchError(SamsaekException):
    """Base exception for Exa search errors"""
    pass


class ExaAPIError(ExaSearchError):
    """Exception for Exa API errors"""
    pass


class ExaQueryError(ExaSearchError):
    """Exception for Exa query processing errors"""
    pass


class ExaRateLimitError(ExaSearchError):
    """Exception for Exa rate limit errors"""
    pass


class ExaAuthenticationError(ExaSearchError):
    """Exception for Exa authentication errors"""
    pass


class ExaTimeoutError(ExaSearchError):
    """Exception for Exa timeout errors"""
    pass