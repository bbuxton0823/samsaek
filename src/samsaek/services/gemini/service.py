"""
Google Gemini Integration Service
Core service implementation for Google Gemini API integration
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, Any, Optional, List, Union, AsyncIterator
from datetime import datetime, timedelta
import httpx
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core import exceptions as google_exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ...core.config import config
from ...core.logging import get_logger, SamsaekLoggerMixin, LogOperation
from ...core.exceptions import (
    GeminiServiceError, 
    GeminiRateLimitError, 
    GeminiAuthenticationError,
    GeminiModelError,
    TimeoutError
)
from .models import (
    GeminiRequest, 
    GeminiResponse, 
    GeminiStreamResponse,
    GeminiConnectionConfig,
    GeminiModelType,
    GeminiUsageMetadata,
    GeminiCandidate,
    GeminiSafetyRating
)


class GeminiConnectionPool:
    """Connection pool for Gemini API requests"""
    
    def __init__(self, config: GeminiConnectionConfig):
        self.config = config
        self.logger = get_logger("gemini_pool")
        self._client = None
        self._lock = asyncio.Lock()
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling"""
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    limits = httpx.Limits(
                        max_connections=self.config.max_connections,
                        max_keepalive_connections=self.config.max_keepalive_connections,
                        keepalive_expiry=self.config.keepalive_expiry
                    )
                    
                    self._client = httpx.AsyncClient(
                        limits=limits,
                        timeout=httpx.Timeout(self.config.timeout)
                    )
        
        return self._client
    
    async def close(self):
        """Close connection pool"""
        if self._client:
            await self._client.aclose()
            self._client = None


class GeminiRateLimiter:
    """Rate limiter for Gemini API requests"""
    
    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.requests = []
        self.logger = get_logger("gemini_rate_limiter")
    
    async def acquire(self):
        """Acquire rate limit slot"""
        now = time.time()
        
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        
        if len(self.requests) >= self.requests_per_minute:
            # Calculate wait time
            oldest_request = min(self.requests)
            wait_time = 60 - (now - oldest_request)
            
            if wait_time > 0:
                self.logger.warning(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        self.requests.append(now)


class GeminiCache:
    """Cache for Gemini responses"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, tuple] = {}
        self.logger = get_logger("gemini_cache")
    
    def _generate_key(self, request: GeminiRequest) -> str:
        """Generate cache key from request"""
        # Create deterministic key from request content
        request_data = {
            "model": request.model,
            "contents": [
                {"type": content.content_type, "data": content.data}
                for content in request.contents
            ],
            "generation_config": request.generation_config.dict() if request.generation_config else None,
            "system_instruction": request.system_instruction
        }
        
        request_str = json.dumps(request_data, sort_keys=True)
        return hashlib.sha256(request_str.encode()).hexdigest()
    
    def get(self, request: GeminiRequest) -> Optional[GeminiResponse]:
        """Get cached response"""
        if not config.cache_enabled:
            return None
        
        key = self._generate_key(request)
        
        if key in self.cache:
            response_data, timestamp = self.cache[key]
            
            # Check if cache entry is still valid
            if time.time() - timestamp < self.ttl:
                self.logger.debug(f"Cache hit for key: {key[:8]}...")
                return GeminiResponse.parse_obj(response_data)
            else:
                # Remove expired entry
                del self.cache[key]
        
        return None
    
    def set(self, request: GeminiRequest, response: GeminiResponse):
        """Cache response"""
        if not config.cache_enabled:
            return
        
        key = self._generate_key(request)
        
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (response.dict(), time.time())
        self.logger.debug(f"Cached response for key: {key[:8]}...")


class GeminiService(SamsaekLoggerMixin):
    """
    Google Gemini Integration Service
    Handles all interactions with Google Gemini models for Samsaek
    """
    
    def __init__(self, connection_config: Optional[GeminiConnectionConfig] = None):
        super().__init__()
        
        # Initialize configuration
        self.config = connection_config or GeminiConnectionConfig(
            api_key=config.google_api_key,
            project_id=config.google_project_id,
            timeout=config.gemini_request_timeout,
            rate_limit_per_minute=config.gemini_rate_limit,
            samsaek_agent_id=config.a2a_agent_id,
            enable_monitoring=config.weave_capture_all,
            enable_caching=config.cache_enabled,
            cache_ttl=config.cache_ttl
        )
        
        # Initialize components
        self.connection_pool = GeminiConnectionPool(self.config)
        self.rate_limiter = GeminiRateLimiter(self.config.rate_limit_per_minute)
        self.cache = GeminiCache(max_size=config.cache_max_size, ttl=self.config.cache_ttl)
        
        # Initialize Gemini client
        self._initialize_client()
        
        # Performance tracking
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        
        self.logger.info("Gemini service initialized", 
                        agent_id=self.config.samsaek_agent_id,
                        monitoring_enabled=self.config.enable_monitoring)
    
    def _initialize_client(self):
        """Initialize Google Gemini client"""
        try:
            genai.configure(api_key=self.config.api_key)
            self.logger.info("Gemini client configured successfully")
        except Exception as e:
            raise GeminiAuthenticationError(
                "Failed to initialize Gemini client",
                context={"error": str(e)}
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((GeminiServiceError, google_exceptions.ServiceUnavailable))
    )
    async def generate_content(
        self, 
        request: GeminiRequest,
        stream: bool = False
    ) -> Union[GeminiResponse, AsyncIterator[GeminiStreamResponse]]:
        """
        Generate content using Gemini API
        
        Args:
            request: Gemini request object
            stream: Whether to stream the response
            
        Returns:
            GeminiResponse or AsyncIterator[GeminiStreamResponse]
        """
        with LogOperation(f"gemini_generate_{request.model}", self.logger,
                         model=request.model, stream=stream,
                         workflow_id=request.workflow_id, agent_id=request.agent_id):
            
            # Check cache first (only for non-streaming requests)
            if not stream and self.config.enable_caching:
                cached_response = self.cache.get(request)
                if cached_response:
                    self.logger.info("Returning cached response")
                    return cached_response
            
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            try:
                if stream:
                    return self._generate_content_stream(request)
                else:
                    return await self._generate_content_sync(request)
            
            except google_exceptions.ResourceExhausted as e:
                raise GeminiRateLimitError(
                    "Gemini API rate limit exceeded",
                    context={"request_model": request.model, "error": str(e)}
                )
            except google_exceptions.Unauthenticated as e:
                raise GeminiAuthenticationError(
                    "Gemini API authentication failed",
                    context={"error": str(e)}
                )
            except google_exceptions.DeadlineExceeded as e:
                raise TimeoutError(
                    "Gemini API request timeout",
                    context={"timeout": self.config.timeout, "error": str(e)}
                )
            except Exception as e:
                raise GeminiServiceError(
                    "Gemini API request failed",
                    context={"request_model": request.model, "error": str(e)}
                )
    
    async def _generate_content_sync(self, request: GeminiRequest) -> GeminiResponse:
        """Generate content synchronously"""
        start_time = time.time()
        
        try:
            # Initialize model
            model = genai.GenerativeModel(
                model_name=request.model,
                generation_config=self._build_generation_config(request),
                safety_settings=self._build_safety_settings(request),
                system_instruction=request.system_instruction
            )
            
            # Prepare content
            content_parts = []
            for content in request.contents:
                if content.content_type == "text":
                    content_parts.append(content.data)
                else:
                    # Handle multimodal content
                    content_parts.append({
                        "mime_type": content.mime_type,
                        "data": content.data
                    })
            
            # Generate response
            response = model.generate_content(content_parts)
            
            # Build Samsaek response
            samsaek_response = await self._build_samsaek_response(
                response, request, time.time() - start_time
            )
            
            # Cache response
            if self.config.enable_caching:
                self.cache.set(request, samsaek_response)
            
            # Update metrics
            self._update_metrics(samsaek_response)
            
            return samsaek_response
            
        except Exception as e:
            self.logger.error(f"Gemini content generation failed: {str(e)}")
            raise
    
    async def _generate_content_stream(self, request: GeminiRequest) -> AsyncIterator[GeminiStreamResponse]:
        """Generate content with streaming"""
        try:
            # Initialize model
            model = genai.GenerativeModel(
                model_name=request.model,
                generation_config=self._build_generation_config(request),
                safety_settings=self._build_safety_settings(request),
                system_instruction=request.system_instruction
            )
            
            # Prepare content
            content_parts = []
            for content in request.contents:
                if content.content_type == "text":
                    content_parts.append(content.data)
                else:
                    content_parts.append({
                        "mime_type": content.mime_type,
                        "data": content.data
                    })
            
            # Generate streaming response
            response_stream = model.generate_content(content_parts, stream=True)
            
            chunk_id = 0
            for chunk in response_stream:
                chunk_id += 1
                
                # Extract text from chunk
                chunk_text = ""
                if hasattr(chunk, 'text'):
                    chunk_text = chunk.text
                elif hasattr(chunk, 'parts'):
                    chunk_text = "".join([part.text for part in chunk.parts if hasattr(part, 'text')])
                
                yield GeminiStreamResponse(
                    chunk_id=str(chunk_id),
                    content=chunk_text,
                    is_final=chunk_id == len(list(response_stream)),
                    samsaek_context={
                        "workflow_id": request.workflow_id,
                        "agent_id": request.agent_id,
                        "model": request.model
                    }
                )
            
        except Exception as e:
            self.logger.error(f"Gemini streaming generation failed: {str(e)}")
            raise
    
    def _build_generation_config(self, request: GeminiRequest) -> Dict[str, Any]:
        """Build generation configuration"""
        if not request.generation_config:
            return {
                "temperature": config.gemini_temperature,
                "max_output_tokens": config.gemini_max_tokens
            }
        
        return request.generation_config.dict(exclude_none=True)
    
    def _build_safety_settings(self, request: GeminiRequest) -> List[Dict[str, Any]]:
        """Build safety settings"""
        if request.safety_settings:
            return [
                {
                    "category": setting.category,
                    "threshold": setting.threshold
                }
                for setting in request.safety_settings
            ]
        
        # Default safety settings
        return [
            {
                "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            }
        ]
    
    async def _build_samsaek_response(
        self, 
        gemini_response: Any, 
        request: GeminiRequest, 
        processing_time: float
    ) -> GeminiResponse:
        """Build Samsaek response from Gemini response"""
        
        # Extract candidates
        candidates = []
        if hasattr(gemini_response, 'candidates'):
            for i, candidate in enumerate(gemini_response.candidates):
                # Extract safety ratings
                safety_ratings = []
                if hasattr(candidate, 'safety_ratings'):
                    for rating in candidate.safety_ratings:
                        safety_ratings.append(GeminiSafetyRating(
                            category=rating.category.name,
                            probability=rating.probability.name,
                            blocked=rating.blocked
                        ))
                
                candidates.append(GeminiCandidate(
                    content={"parts": [{"text": candidate.text}] if hasattr(candidate, 'text') else []},
                    finish_reason=candidate.finish_reason.name if hasattr(candidate, 'finish_reason') else "STOP",
                    index=i,
                    safety_ratings=safety_ratings
                ))
        
        # Extract usage metadata
        usage_metadata = GeminiUsageMetadata(
            prompt_token_count=getattr(gemini_response.usage_metadata, 'prompt_token_count', 0),
            candidates_token_count=getattr(gemini_response.usage_metadata, 'candidates_token_count', 0),
            total_token_count=getattr(gemini_response.usage_metadata, 'total_token_count', 0)
        )
        
        # Build response
        response = GeminiResponse(
            candidates=candidates,
            usage_metadata=usage_metadata,
            model_version=request.model,
            samsaek_metadata={
                "workflow_id": request.workflow_id,
                "agent_id": request.agent_id,
                "task_id": request.task_id,
                "priority": request.priority,
                "platform": "samsaek"
            },
            processing_time=processing_time
        )
        
        return response
    
    def _update_metrics(self, response: GeminiResponse):
        """Update service metrics"""
        self.request_count += 1
        self.total_tokens += response.token_count
        self.total_cost += response.cost_estimate
        
        self.logger.info("Gemini request completed",
                        request_count=self.request_count,
                        tokens_used=response.token_count,
                        cost_estimate=response.cost_estimate,
                        total_cost=self.total_cost)
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a Gemini model"""
        try:
            model = genai.get_model(model_name)
            return {
                "name": model.name,
                "display_name": model.display_name,
                "description": model.description,
                "input_token_limit": model.input_token_limit,
                "output_token_limit": model.output_token_limit,
                "supported_generation_methods": model.supported_generation_methods,
                "temperature": model.temperature,
                "top_p": model.top_p,
                "top_k": model.top_k
            }
        except Exception as e:
            raise GeminiModelError(
                f"Failed to get model info for {model_name}",
                context={"model": model_name, "error": str(e)}
            )
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Gemini models"""
        try:
            models = []
            for model in genai.list_models():
                models.append({
                    "name": model.name,
                    "display_name": model.display_name,
                    "description": model.description,
                    "version": model.version,
                    "input_token_limit": model.input_token_limit,
                    "output_token_limit": model.output_token_limit
                })
            return models
        except Exception as e:
            raise GeminiModelError(
                "Failed to list models",
                context={"error": str(e)}
            )
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics"""
        return {
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "average_cost_per_request": self.total_cost / max(self.request_count, 1),
            "cache_size": len(self.cache.cache),
            "rate_limit_per_minute": self.rate_limiter.requests_per_minute,
            "current_requests_in_window": len(self.rate_limiter.requests)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Try to list models as a basic health check
            models = await self.list_models()
            return {
                "status": "healthy",
                "models_available": len(models),
                "rate_limiter_status": "active",
                "cache_status": "active" if self.config.enable_caching else "disabled",
                "connection_pool_status": "active"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.connection_pool.close()
        self.logger.info("Gemini service cleanup completed")