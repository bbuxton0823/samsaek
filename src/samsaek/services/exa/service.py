"""
Exa Search Integration Service
Neural search service for real-time information retrieval
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import httpx
import weave
from tenacity import retry, stop_after_attempt, wait_exponential

from ...core.config import config
from ...core.logging import get_logger, SamsaekLoggerMixin, LogOperation
from ...core.exceptions import (
    ExaSearchError,
    ExaAPIError,
    ExaQueryError,
    ExaRateLimitError,
    TimeoutError
)
from .models import (
    ExaSearchRequest,
    ExaSearchResponse,
    ExaSearchResult,
    ExaSearchConfig,
    ExaSearchStats,
    ExaSearchType
)


class ExaRateLimiter:
    """Rate limiter for Exa API requests"""
    
    def __init__(self, rpm: int = 60, rph: int = 1000):
        self.rpm = rpm
        self.rph = rph
        self.minute_requests = []
        self.hour_requests = []
        self.logger = get_logger("exa_rate_limiter")
    
    async def acquire(self):
        """Acquire rate limit slot"""
        now = time.time()
        
        # Remove old requests
        self.minute_requests = [req for req in self.minute_requests if now - req < 60]
        self.hour_requests = [req for req in self.hour_requests if now - req < 3600]
        
        # Check minute limit
        if len(self.minute_requests) >= self.rpm:
            wait_time = 60 - (now - min(self.minute_requests))
            if wait_time > 0:
                self.logger.warning(f"Rate limit reached (minute), waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # Check hour limit
        if len(self.hour_requests) >= self.rph:
            wait_time = 3600 - (now - min(self.hour_requests))
            if wait_time > 0:
                self.logger.warning(f"Rate limit reached (hour), waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # Add request
        self.minute_requests.append(now)
        self.hour_requests.append(now)


class ExaSearchCache:
    """Cache for Exa search responses"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, tuple] = {}
        self.logger = get_logger("exa_cache")
    
    def _generate_key(self, request: ExaSearchRequest) -> str:
        """Generate cache key from request"""
        key_data = {
            "query": request.query,
            "search_type": request.search_type,
            "num_results": request.num_results,
            "include_domains": request.include_domains,
            "exclude_domains": request.exclude_domains,
            "start_published_date": request.start_published_date.isoformat() if request.start_published_date else None,
            "end_published_date": request.end_published_date.isoformat() if request.end_published_date else None
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, request: ExaSearchRequest) -> Optional[ExaSearchResponse]:
        """Get cached response"""
        if not config.cache_enabled:
            return None
        
        key = self._generate_key(request)
        
        if key in self.cache:
            response_data, timestamp = self.cache[key]
            
            # Check if cache entry is still valid
            if time.time() - timestamp < self.ttl:
                self.logger.debug(f"Cache hit for query: {request.query[:50]}...")
                response = ExaSearchResponse.parse_obj(response_data)
                response.cached = True
                return response
            else:
                # Remove expired entry
                del self.cache[key]
        
        return None
    
    def set(self, request: ExaSearchRequest, response: ExaSearchResponse):
        """Cache response"""
        if not config.cache_enabled:
            return
        
        key = self._generate_key(request)
        
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (response.dict(), time.time())
        self.logger.debug(f"Cached response for query: {request.query[:50]}...")


class ExaSearchService(SamsaekLoggerMixin):
    """
    Exa Search Integration Service
    Provides neural search capabilities for Samsaek agents
    """
    
    def __init__(self, config: Optional[ExaSearchConfig] = None):
        super().__init__()
        
        # Initialize configuration
        self.config = config or ExaSearchConfig(
            api_key=config.exa_api_key,
            timeout=config.exa_search_timeout,
            cache_ttl=config.exa_cache_ttl,
            default_num_results=config.exa_max_results,
            enable_caching=config.cache_enabled,
            weave_integration=config.weave_capture_all
        )
        
        # Initialize components
        self.rate_limiter = ExaRateLimiter(
            rpm=self.config.requests_per_minute,
            rph=self.config.requests_per_hour
        )
        self.cache = ExaSearchCache(
            max_size=self.config.cache_max_size,
            ttl=self.config.cache_ttl
        )
        
        # HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Samsaek/2.0.0"
            }
        )
        
        # Statistics
        self.stats = ExaSearchStats()
        
        # Initialize Weave monitoring
        if self.config.weave_integration:
            weave.init("samsaek-exa")
        
        self.logger.info("Exa Search Service initialized")
    
    @weave.op()
    async def search(self, request: ExaSearchRequest) -> ExaSearchResponse:
        """
        Perform neural search using Exa API
        
        Args:
            request: Search request parameters
            
        Returns:
            ExaSearchResponse with results
        """
        with LogOperation(f"exa_search", self.logger,
                         query=request.query[:50] + "..." if len(request.query) > 50 else request.query,
                         search_type=request.search_type,
                         num_results=request.num_results,
                         workflow_id=request.workflow_id,
                         agent_id=request.agent_id):
            
            # Check cache first
            cached_response = self.cache.get(request)
            if cached_response:
                self.stats.cached_searches += 1
                self.logger.info("Returning cached search results")
                return cached_response
            
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            # Perform search
            start_time = time.time()
            try:
                response = await self._perform_search(request)
                search_time = time.time() - start_time
                
                # Cache response
                self.cache.set(request, response)
                
                # Update statistics
                self.stats.update_stats(
                    success=True,
                    search_time=search_time,
                    quality_score=response.quality_score or 0.0,
                    results_count=len(response.results)
                )
                
                self.logger.info(f"Search completed: {len(response.results)} results in {search_time:.2f}s")
                return response
                
            except Exception as e:
                search_time = time.time() - start_time
                self.stats.update_stats(
                    success=False,
                    search_time=search_time,
                    quality_score=0.0,
                    results_count=0
                )
                
                self.log_error("exa_search", e, query=request.query)
                raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _perform_search(self, request: ExaSearchRequest) -> ExaSearchResponse:
        """Perform the actual search request"""
        try:
            # For hackathon: Use mock data if no API key
            if not self.config.api_key or self.config.api_key == "your-exa-api-key-here":
                return self._generate_mock_response(request)
            
            # Prepare request parameters
            params = request.to_exa_params()
            
            # Make API request
            response = await self.client.post(
                f"{self.config.base_url}/search",
                json=params
            )
            
            # Handle response
            if response.status_code == 200:
                return self._parse_response(response.json(), request)
            elif response.status_code == 429:
                raise ExaRateLimitError("Exa API rate limit exceeded")
            elif response.status_code == 401:
                raise ExaAPIError("Exa API authentication failed")
            else:
                raise ExaAPIError(f"Exa API error: {response.status_code} - {response.text}")
                
        except httpx.TimeoutException:
            raise TimeoutError("Exa search request timed out")
        except httpx.RequestError as e:
            raise ExaAPIError(f"Exa API request failed: {str(e)}")
    
    def _generate_mock_response(self, request: ExaSearchRequest) -> ExaSearchResponse:
        """Generate mock response for hackathon demo"""
        mock_results = []
        
        # Generate mock results based on query
        query_lower = request.query.lower()
        
        if "ai" in query_lower or "artificial intelligence" in query_lower:
            mock_results = [
                ExaSearchResult(
                    id="mock_1",
                    url="https://arxiv.org/abs/2023.12345",
                    title="Recent Advances in Multi-Agent AI Systems",
                    score=0.95,
                    text="This paper presents recent advances in multi-agent AI systems, focusing on coordination and collaboration between intelligent agents.",
                    summary="A comprehensive review of multi-agent AI systems and their applications.",
                    author="AI Research Team",
                    published_date=datetime.now() - timedelta(days=30),
                    domain_authority=0.9,
                    content_quality=0.85
                ),
                ExaSearchResult(
                    id="mock_2",
                    url="https://blog.openai.com/multi-agent-systems",
                    title="Building Effective Multi-Agent Systems",
                    score=0.88,
                    text="Multi-agent systems represent a paradigm shift in AI development, enabling specialized agents to work together on complex tasks.",
                    summary="Best practices for building multi-agent AI systems.",
                    author="OpenAI Team",
                    published_date=datetime.now() - timedelta(days=15),
                    domain_authority=0.95,
                    content_quality=0.90
                )
            ]
        elif "samsaek" in query_lower:
            mock_results = [
                ExaSearchResult(
                    id="mock_3",
                    url="https://samsaek.ai/docs",
                    title="Samsaek Multi-Agent AI Platform Documentation",
                    score=0.99,
                    text="Samsaek is a revolutionary multi-agent AI platform that integrates Google Gemini, A2A protocol, and Crew AI for enterprise automation.",
                    summary="Official documentation for the Samsaek platform.",
                    author="Samsaek Team",
                    published_date=datetime.now() - timedelta(days=1),
                    domain_authority=0.85,
                    content_quality=0.95
                )
            ]
        else:
            # Generic mock results
            mock_results = [
                ExaSearchResult(
                    id="mock_generic_1",
                    url=f"https://example.com/search/{request.query.replace(' ', '-')}",
                    title=f"Information about {request.query}",
                    score=0.75,
                    text=f"This is relevant information about {request.query}. The content provides comprehensive details and insights.",
                    summary=f"A summary of information related to {request.query}.",
                    author="Research Team",
                    published_date=datetime.now() - timedelta(days=7),
                    domain_authority=0.70,
                    content_quality=0.75
                ),
                ExaSearchResult(
                    id="mock_generic_2",
                    url=f"https://knowledge.com/{request.query.replace(' ', '_')}",
                    title=f"Deep dive into {request.query}",
                    score=0.68,
                    text=f"An in-depth analysis of {request.query} covering various aspects and applications.",
                    summary=f"Comprehensive analysis of {request.query}.",
                    author="Knowledge Team",
                    published_date=datetime.now() - timedelta(days=14),
                    domain_authority=0.65,
                    content_quality=0.70
                )
            ]
        
        # Limit results to requested number
        mock_results = mock_results[:request.num_results]
        
        return ExaSearchResponse(
            results=mock_results,
            total_results=len(mock_results),
            query=request.query,
            search_type=request.search_type,
            search_time=0.5,  # Mock search time
            processing_time=0.6,
            samsaek_context=request.samsaek_context,
            workflow_id=request.workflow_id,
            agent_id=request.agent_id,
            cached=False
        )
    
    def _parse_response(self, data: Dict[str, Any], request: ExaSearchRequest) -> ExaSearchResponse:
        """Parse Exa API response"""
        results = []
        
        for result_data in data.get("results", []):
            result = ExaSearchResult(
                id=result_data.get("id", ""),
                url=result_data.get("url", ""),
                title=result_data.get("title", ""),
                score=result_data.get("score", 0.0),
                text=result_data.get("text"),
                html=result_data.get("html"),
                summary=result_data.get("summary"),
                highlights=result_data.get("highlights"),
                author=result_data.get("author"),
                published_date=self._parse_date(result_data.get("publishedDate")),
                crawled_date=self._parse_date(result_data.get("crawledDate")),
                links=result_data.get("links"),
                images=result_data.get("images")
            )
            results.append(result)
        
        return ExaSearchResponse(
            results=results,
            total_results=len(results),
            query=request.query,
            search_type=request.search_type,
            search_time=data.get("searchTime", 0.0),
            processing_time=data.get("processingTime"),
            samsaek_context=request.samsaek_context,
            workflow_id=request.workflow_id,
            agent_id=request.agent_id,
            cached=False
        )
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime"""
        if not date_str:
            return None
        
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except:
            return None
    
    async def search_similar(self, url: str, num_results: int = 10) -> ExaSearchResponse:
        """Find similar content to a given URL"""
        try:
            # For hackathon: Mock implementation
            if not self.config.api_key or self.config.api_key == "your-exa-api-key-here":
                return self._generate_mock_similar_response(url, num_results)
            
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            params = {
                "url": url,
                "num_results": num_results
            }
            
            response = await self.client.post(
                f"{self.config.base_url}/findSimilar",
                json=params
            )
            
            if response.status_code == 200:
                return self._parse_response(response.json(), ExaSearchRequest(query=f"similar:{url}"))
            else:
                raise ExaAPIError(f"Exa similar search failed: {response.status_code}")
                
        except Exception as e:
            raise ExaSearchError(f"Failed to search similar content: {str(e)}")
    
    def _generate_mock_similar_response(self, url: str, num_results: int) -> ExaSearchResponse:
        """Generate mock similar content response"""
        mock_results = [
            ExaSearchResult(
                id=f"similar_{i}",
                url=f"https://similar-{i}.com/content",
                title=f"Similar content to {url} #{i}",
                score=0.8 - (i * 0.1),
                text=f"This content is similar to the original at {url}.",
                summary=f"Similar content #{i}",
                published_date=datetime.now() - timedelta(days=i+1),
                domain_authority=0.7,
                content_quality=0.75
            )
            for i in range(min(num_results, 3))
        ]
        
        return ExaSearchResponse(
            results=mock_results,
            total_results=len(mock_results),
            query=f"similar:{url}",
            search_type=ExaSearchType.NEURAL,
            search_time=0.4,
            processing_time=0.5,
            cached=False
        )
    
    async def get_contents(self, urls: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get content from specific URLs"""
        try:
            # For hackathon: Mock implementation
            if not self.config.api_key or self.config.api_key == "your-exa-api-key-here":
                return self._generate_mock_contents(urls)
            
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            params = {
                "ids": urls,
                "contents": {
                    "text": True,
                    "highlights": True,
                    "summary": True
                }
            }
            
            response = await self.client.post(
                f"{self.config.base_url}/contents",
                json=params
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise ExaAPIError(f"Exa contents request failed: {response.status_code}")
                
        except Exception as e:
            raise ExaSearchError(f"Failed to get contents: {str(e)}")
    
    def _generate_mock_contents(self, urls: List[str]) -> Dict[str, Dict[str, Any]]:
        """Generate mock content for URLs"""
        contents = {}
        
        for url in urls:
            contents[url] = {
                "text": f"This is the content from {url}. It contains relevant information and insights.",
                "summary": f"Summary of content from {url}",
                "highlights": [f"Key point from {url}", f"Important insight from {url}"],
                "title": f"Content from {url}",
                "author": "Content Team",
                "published_date": datetime.now().isoformat()
            }
        
        return contents
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "search_stats": self.stats.dict(),
            "cache_stats": {
                "cache_size": len(self.cache.cache),
                "cache_hit_rate": self.stats.get_cache_hit_rate(),
                "cache_enabled": self.config.enable_caching
            },
            "rate_limit_stats": {
                "requests_per_minute": self.config.requests_per_minute,
                "requests_per_hour": self.config.requests_per_hour,
                "current_minute_requests": len(self.rate_limiter.minute_requests),
                "current_hour_requests": len(self.rate_limiter.hour_requests)
            },
            "service_info": {
                "api_configured": bool(self.config.api_key and self.config.api_key != "your-exa-api-key-here"),
                "mock_mode": not bool(self.config.api_key and self.config.api_key != "your-exa-api-key-here"),
                "base_url": self.config.base_url,
                "timeout": self.config.timeout
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Simple test search
            test_request = ExaSearchRequest(
                query="test health check",
                num_results=1
            )
            
            start_time = time.time()
            response = await self.search(test_request)
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "results_count": len(response.results),
                "mock_mode": not bool(self.config.api_key and self.config.api_key != "your-exa-api-key-here"),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def cleanup(self):
        """Cleanup service resources"""
        try:
            await self.client.aclose()
            self.logger.info("Exa Search Service cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")