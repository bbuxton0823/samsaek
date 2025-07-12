"""
A2A Protocol Exception Classes
Custom exceptions for A2A protocol operations
"""

from ...core.exceptions import SamsaekException


class A2AProtocolError(SamsaekException):
    """Base exception for A2A protocol errors"""
    pass


class A2AMessageError(A2AProtocolError):
    """Exception for A2A message processing errors"""
    pass


class A2ARoutingError(A2AProtocolError):
    """Exception for A2A message routing errors"""
    pass


class A2AEncryptionError(A2AProtocolError):
    """Exception for A2A encryption/decryption errors"""
    pass


class A2ASignatureError(A2AProtocolError):
    """Exception for A2A signature verification errors"""
    pass


class A2AAgentError(A2AProtocolError):
    """Exception for A2A agent operations"""
    pass


class A2AConnectionError(A2AProtocolError):
    """Exception for A2A connection issues"""
    pass


class A2ATimeoutError(A2AProtocolError):
    """Exception for A2A operation timeouts"""
    pass