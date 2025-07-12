"""
Samsaek Configuration Management
Handles all configuration settings for the Samsaek Multi-Agent AI System
"""

import os
from typing import Optional, Dict, Any, List
from pydantic import BaseSettings, Field, validator
from enum import Enum


class Environment(str, Enum):
    """Environment types for Samsaek deployment"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SamsaekConfig(BaseSettings):
    """
    Comprehensive configuration for Samsaek Multi-Agent AI System
    Following the PRD specifications for enterprise-grade deployment
    """
    
    # Core Application Settings
    app_name: str = Field(default="Samsaek", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Deployment environment")
    debug: bool = Field(default=False, description="Debug mode")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host address")
    api_port: int = Field(default=8000, description="API port")
    api_prefix: str = Field(default="/api/v1", description="API prefix path")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    
    # Security Configuration
    secret_key: str = Field(description="Secret key for JWT and encryption")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(default=24, description="JWT expiration in hours")
    encryption_key: Optional[str] = Field(default=None, description="Encryption key for sensitive data")
    
    # Database Configuration
    database_url: str = Field(description="Database connection URL")
    database_pool_size: int = Field(default=10, description="Database connection pool size")
    database_max_overflow: int = Field(default=20, description="Database max overflow connections")
    
    # Time Series Database (for monitoring)
    timeseries_db_url: Optional[str] = Field(default=None, description="Time series database URL")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    redis_max_connections: int = Field(default=100, description="Redis max connections")
    
    # Message Queue Configuration (Kafka)
    kafka_bootstrap_servers: str = Field(default="localhost:9092", description="Kafka bootstrap servers")
    kafka_consumer_group: str = Field(default="samsaek-consumers", description="Kafka consumer group")
    kafka_auto_offset_reset: str = Field(default="latest", description="Kafka auto offset reset")
    
    # Google Gemini Configuration
    google_api_key: str = Field(description="Google API key for Gemini")
    google_project_id: Optional[str] = Field(default=None, description="Google Cloud project ID")
    gemini_model: str = Field(default="gemini-pro", description="Default Gemini model")
    gemini_max_tokens: int = Field(default=8192, description="Max tokens for Gemini")
    gemini_temperature: float = Field(default=0.7, description="Gemini temperature")
    gemini_request_timeout: int = Field(default=30, description="Gemini request timeout in seconds")
    gemini_rate_limit: int = Field(default=100, description="Gemini requests per minute")
    
    # A2A Protocol Configuration
    a2a_protocol_version: str = Field(default="1.0", description="A2A protocol version")
    a2a_agent_id: str = Field(description="Unique A2A agent identifier")
    a2a_encryption_enabled: bool = Field(default=True, description="Enable A2A message encryption")
    a2a_signature_enabled: bool = Field(default=True, description="Enable A2A message signatures")
    a2a_message_ttl: int = Field(default=3600, description="A2A message TTL in seconds")
    
    # Crew AI Configuration
    crew_max_agents: int = Field(default=50, description="Maximum number of concurrent agents")
    crew_task_timeout: int = Field(default=300, description="Crew task timeout in seconds")
    crew_retry_attempts: int = Field(default=3, description="Crew task retry attempts")
    crew_memory_enabled: bool = Field(default=True, description="Enable crew memory")
    
    # Exa Search Configuration
    exa_api_key: str = Field(description="Exa API key")
    exa_max_results: int = Field(default=10, description="Max Exa search results")
    exa_search_timeout: int = Field(default=15, description="Exa search timeout in seconds")
    exa_cache_ttl: int = Field(default=3600, description="Exa cache TTL in seconds")
    
    # Weave Monitoring Configuration
    weave_project_name: str = Field(default="samsaek", description="Weave project name")
    weave_api_key: Optional[str] = Field(default=None, description="Weave API key")
    weave_capture_all: bool = Field(default=True, description="Capture all Weave interactions")
    weave_sample_rate: float = Field(default=1.0, description="Weave sampling rate")
    weave_async_logging: bool = Field(default=True, description="Enable async logging")
    weave_batch_size: int = Field(default=100, description="Weave batch size")
    weave_flush_interval: int = Field(default=30, description="Weave flush interval in seconds")
    
    # Monitoring and Observability
    prometheus_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    prometheus_port: int = Field(default=8001, description="Prometheus metrics port")
    jaeger_enabled: bool = Field(default=False, description="Enable Jaeger tracing")
    jaeger_endpoint: Optional[str] = Field(default=None, description="Jaeger endpoint")
    
    # Logging Configuration
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    log_format: str = Field(default="json", description="Log format (json or text)")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    # Performance Configuration
    max_workers: int = Field(default=4, description="Maximum worker processes")
    worker_timeout: int = Field(default=30, description="Worker timeout in seconds")
    max_requests_per_minute: int = Field(default=1000, description="Rate limit per minute")
    
    # Caching Configuration
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_ttl: int = Field(default=3600, description="Default cache TTL in seconds")
    cache_max_size: int = Field(default=1000, description="Maximum cache size")
    
    # Storage Configuration
    storage_type: str = Field(default="local", description="Storage type (local, s3, gcs)")
    storage_bucket: Optional[str] = Field(default=None, description="Storage bucket name")
    storage_region: Optional[str] = Field(default=None, description="Storage region")
    
    # Kubernetes Configuration
    k8s_namespace: str = Field(default="samsaek", description="Kubernetes namespace")
    k8s_service_account: Optional[str] = Field(default=None, description="Kubernetes service account")
    
    # Health Check Configuration
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")
    health_check_timeout: int = Field(default=5, description="Health check timeout in seconds")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator("environment", pre=True)
    def validate_environment(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v
        
    @validator("cors_origins", pre=True)
    def validate_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
        
    @validator("debug", pre=True)
    def validate_debug(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return v
        
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == Environment.DEVELOPMENT
        
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == Environment.PRODUCTION
        
    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode"""
        return self.environment == Environment.TESTING
        
    @property
    def database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_max_overflow,
            "echo": self.debug,
        }
        
    @property
    def redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration"""
        return {
            "url": self.redis_url,
            "max_connections": self.redis_max_connections,
            "decode_responses": True,
        }
        
    @property
    def kafka_config(self) -> Dict[str, Any]:
        """Get Kafka configuration"""
        return {
            "bootstrap_servers": self.kafka_bootstrap_servers.split(","),
            "consumer_group": self.kafka_consumer_group,
            "auto_offset_reset": self.kafka_auto_offset_reset,
        }
        
    @property
    def gemini_config(self) -> Dict[str, Any]:
        """Get Gemini configuration"""
        return {
            "api_key": self.google_api_key,
            "project_id": self.google_project_id,
            "model": self.gemini_model,
            "max_tokens": self.gemini_max_tokens,
            "temperature": self.gemini_temperature,
            "timeout": self.gemini_request_timeout,
            "rate_limit": self.gemini_rate_limit,
        }
        
    @property
    def a2a_config(self) -> Dict[str, Any]:
        """Get A2A protocol configuration"""
        return {
            "version": self.a2a_protocol_version,
            "agent_id": self.a2a_agent_id,
            "encryption_enabled": self.a2a_encryption_enabled,
            "signature_enabled": self.a2a_signature_enabled,
            "message_ttl": self.a2a_message_ttl,
        }
        
    @property
    def crew_config(self) -> Dict[str, Any]:
        """Get Crew AI configuration"""
        return {
            "max_agents": self.crew_max_agents,
            "task_timeout": self.crew_task_timeout,
            "retry_attempts": self.crew_retry_attempts,
            "memory_enabled": self.crew_memory_enabled,
        }
        
    @property
    def exa_config(self) -> Dict[str, Any]:
        """Get Exa search configuration"""
        return {
            "api_key": self.exa_api_key,
            "max_results": self.exa_max_results,
            "timeout": self.exa_search_timeout,
            "cache_ttl": self.exa_cache_ttl,
        }
        
    @property
    def weave_config(self) -> Dict[str, Any]:
        """Get Weave monitoring configuration"""
        return {
            "project_name": self.weave_project_name,
            "api_key": self.weave_api_key,
            "capture_all": self.weave_capture_all,
            "sample_rate": self.weave_sample_rate,
            "async_logging": self.weave_async_logging,
            "batch_size": self.weave_batch_size,
            "flush_interval": self.weave_flush_interval,
        }


# Global configuration instance
config = SamsaekConfig()