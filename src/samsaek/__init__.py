"""
Samsaek: Multi-Agent AI System
Copyright (c) 2025 Manus AI

A sophisticated multi-agent AI platform that integrates Google Gemini, A2A Protocol,
Crew AI, Exa Search, and Weave monitoring for enterprise-grade AI automation.
"""

__version__ = "2.0.0"
__author__ = "Manus AI"
__email__ = "team@manusai.com"
__license__ = "MIT"

# Core imports
from .core.config import SamsaekConfig
from .core.logging import setup_logging
from .core.exceptions import SamsaekException

# Service imports
from .services.gemini import GeminiService
from .services.a2a import A2AProtocolGateway
from .services.crew import CrewOrchestrationEngine
from .services.exa import ExaSearchService
from .services.weave import WeaveMonitoringService

# API imports
from .api.main import create_app

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "SamsaekConfig",
    "setup_logging",
    "SamsaekException",
    "GeminiService",
    "A2AProtocolGateway", 
    "CrewOrchestrationEngine",
    "ExaSearchService",
    "WeaveMonitoringService",
    "create_app",
]