"""
Exa Search Integration Models
Pydantic models for Exa neural search API interactions
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
from enum import Enum


class ExaSearchType(str, Enum):
    """Exa search types"""
    NEURAL = "neural"
    KEYWORD = "keyword"
    AUTO = "auto"


class ExaContentType(str, Enum):
    """Content types for Exa searches"""
    TEXT = "text"
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    ALL = "all"


class ExaCategory(str, Enum):
    """Content categories for filtering"""
    NEWS = "news"
    ACADEMIC = "academic"
    SOCIAL = "social"
    FORUM = "forum"
    COMPANY = "company"
    PERSONAL = "personal"
    GOVERNMENT = "government"
    EDUCATION = "education"
    REFERENCE = "reference"
    ENTERTAINMENT = "entertainment"
    SHOPPING = "shopping"
    TRAVEL = "travel"
    TECHNOLOGY = "technology"
    FINANCE = "finance"
    HEALTH = "health"
    SPORTS = "sports"
    ALL = "all"


class ExaTimeRange(BaseModel):
    """Time range for search results"""
    start_published_date: Optional[datetime] = Field(default=None, description="Start date filter")
    end_published_date: Optional[datetime] = Field(default=None, description="End date filter")
    start_crawled_date: Optional[datetime] = Field(default=None, description="Start crawl date filter")
    end_crawled_date: Optional[datetime] = Field(default=None, description="End crawl date filter")


class ExaSearchFilters(BaseModel):
    """Search filters for Exa queries"""
    include_domains: Optional[List[str]] = Field(default=None, description="Include only these domains")
    exclude_domains: Optional[List[str]] = Field(default=None, description="Exclude these domains")
    include_text: Optional[List[str]] = Field(default=None, description="Include results containing these terms")
    exclude_text: Optional[List[str]] = Field(default=None, description="Exclude results containing these terms")
    category: Optional[ExaCategory] = Field(default=None, description="Content category filter")
    content_type: Optional[ExaContentType] = Field(default=ExaContentType.ALL, description="Content type filter")
    time_range: Optional[ExaTimeRange] = Field(default=None, description="Time range filter")
    
    class Config:
        use_enum_values = True


class ExaSearchRequest(BaseModel):
    """Request model for Exa search operations"""
    query: str = Field(..., description="Search query", min_length=1)
    search_type: ExaSearchType = Field(default=ExaSearchType.NEURAL, description="Type of search")
    num_results: int = Field(default=10, description="Number of results to return", ge=1, le=100)
    
    # Content options
    include_domains: Optional[List[str]] = Field(default=None, description="Include only these domains")
    exclude_domains: Optional[List[str]] = Field(default=None, description="Exclude these domains")
    start_published_date: Optional[datetime] = Field(default=None, description="Start date filter")
    end_published_date: Optional[datetime] = Field(default=None, description="End date filter")
    start_crawled_date: Optional[datetime] = Field(default=None, description="Start crawl date filter")
    end_crawled_date: Optional[datetime] = Field(default=None, description="End crawl date filter")
    
    # Result options
    include_text: bool = Field(default=True, description="Include text content")
    text_length_limit: Optional[int] = Field(default=1000, description="Limit text length")
    include_html: bool = Field(default=False, description="Include HTML content")
    include_links: bool = Field(default=False, description="Include extracted links")
    include_images: bool = Field(default=False, description="Include image URLs")
    
    # Search options
    use_autoprompt: bool = Field(default=True, description="Use Exa's autoprompt feature")
    livecrawl: bool = Field(default=False, description="Enable live crawling")
    
    # Samsaek-specific fields
    samsaek_context: Optional[Dict[str, Any]] = Field(default=None, description="Samsaek context")
    workflow_id: Optional[str] = Field(default=None, description="Associated workflow ID")
    agent_id: Optional[str] = Field(default=None, description="Requesting agent ID")
    task_id: Optional[str] = Field(default=None, description="Associated task ID")
    priority: str = Field(default="normal", description="Search priority")
    
    class Config:
        use_enum_values = True
    
    @validator('num_results')
    def validate_num_results(cls, v):
        if not (1 <= v <= 100):
            raise ValueError('num_results must be between 1 and 100')
        return v
    
    @validator('text_length_limit')
    def validate_text_length(cls, v):
        if v is not None and v <= 0:
            raise ValueError('text_length_limit must be positive')
        return v
    
    def to_exa_params(self) -> Dict[str, Any]:
        """Convert to Exa API parameters"""
        params = {
            "query": self.query,
            "type": self.search_type,
            "num_results": self.num_results,
            "contents": {
                "text": self.include_text,
                "highlights": True,
                "summary": True
            },
            "use_autoprompt": self.use_autoprompt,
            "livecrawl": self.livecrawl
        }
        
        # Add optional parameters
        if self.include_domains:
            params["include_domains"] = self.include_domains
        
        if self.exclude_domains:
            params["exclude_domains"] = self.exclude_domains
        
        if self.start_published_date:
            params["start_published_date"] = self.start_published_date.isoformat()
        
        if self.end_published_date:
            params["end_published_date"] = self.end_published_date.isoformat()
        
        if self.start_crawled_date:
            params["start_crawled_date"] = self.start_crawled_date.isoformat()
        
        if self.end_crawled_date:
            params["end_crawled_date"] = self.end_crawled_date.isoformat()
        
        # Text options
        if self.text_length_limit:
            params["contents"]["text"] = {"max_characters": self.text_length_limit}
        
        if self.include_html:
            params["contents"]["html"] = True
        
        if self.include_links:
            params["contents"]["links"] = True
        
        if self.include_images:
            params["contents"]["images"] = True
        
        return params


class ExaSearchResult(BaseModel):
    """Individual search result from Exa"""
    id: str = Field(..., description="Result ID")
    url: str = Field(..., description="Result URL")
    title: str = Field(..., description="Result title")
    score: float = Field(..., description="Relevance score")
    
    # Content
    text: Optional[str] = Field(default=None, description="Extracted text content")
    html: Optional[str] = Field(default=None, description="HTML content")
    summary: Optional[str] = Field(default=None, description="AI-generated summary")
    highlights: Optional[List[str]] = Field(default=None, description="Key highlights")
    
    # Metadata
    author: Optional[str] = Field(default=None, description="Content author")
    published_date: Optional[datetime] = Field(default=None, description="Published date")
    crawled_date: Optional[datetime] = Field(default=None, description="Crawled date")
    
    # Additional content
    links: Optional[List[str]] = Field(default=None, description="Extracted links")
    images: Optional[List[str]] = Field(default=None, description="Image URLs")
    
    # Quality metrics
    domain_authority: Optional[float] = Field(default=None, description="Domain authority score")
    content_quality: Optional[float] = Field(default=None, description="Content quality score")
    freshness: Optional[float] = Field(default=None, description="Content freshness score")
    
    def get_preview(self, length: int = 200) -> str:
        """Get a preview of the content"""
        if self.text:
            return self.text[:length] + "..." if len(self.text) > length else self.text
        elif self.summary:
            return self.summary[:length] + "..." if len(self.summary) > length else self.summary
        else:
            return self.title


class ExaSearchResponse(BaseModel):
    """Response model from Exa search operations"""
    results: List[ExaSearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results found")
    query: str = Field(..., description="Original query")
    search_type: ExaSearchType = Field(..., description="Search type used")
    
    # Performance metrics
    search_time: float = Field(..., description="Search execution time in seconds")
    processing_time: Optional[float] = Field(default=None, description="Total processing time")
    
    # Quality metrics
    average_score: Optional[float] = Field(default=None, description="Average relevance score")
    quality_score: Optional[float] = Field(default=None, description="Overall quality score")
    
    # Metadata
    request_id: Optional[str] = Field(default=None, description="Request ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    # Samsaek-specific fields
    samsaek_context: Optional[Dict[str, Any]] = Field(default=None, description="Samsaek context")
    workflow_id: Optional[str] = Field(default=None, description="Associated workflow ID")
    agent_id: Optional[str] = Field(default=None, description="Requesting agent ID")
    cached: bool = Field(default=False, description="Whether result was cached")
    
    class Config:
        use_enum_values = True
    
    def __init__(self, **data):
        super().__init__(**data)
        # Calculate quality metrics
        if self.results:
            self.average_score = sum(r.score for r in self.results) / len(self.results)
            self.quality_score = self._calculate_quality_score()
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall quality score"""
        if not self.results:
            return 0.0
        
        # Weighted average of various quality factors
        total_score = 0.0
        weight_sum = 0.0
        
        for result in self.results:
            # Base score from relevance
            score = result.score * 0.4
            weight = 0.4
            
            # Domain authority contribution
            if result.domain_authority:
                score += result.domain_authority * 0.3
                weight += 0.3
            
            # Content quality contribution
            if result.content_quality:
                score += result.content_quality * 0.2
                weight += 0.2
            
            # Freshness contribution
            if result.freshness:
                score += result.freshness * 0.1
                weight += 0.1
            
            total_score += score
            weight_sum += weight
        
        return total_score / max(weight_sum, 1.0)
    
    def get_top_results(self, n: int = 5) -> List[ExaSearchResult]:
        """Get top N results by score"""
        return sorted(self.results, key=lambda x: x.score, reverse=True)[:n]
    
    def get_results_by_domain(self, domain: str) -> List[ExaSearchResult]:
        """Get results from specific domain"""
        return [r for r in self.results if domain in r.url]
    
    def get_recent_results(self, days: int = 30) -> List[ExaSearchResult]:
        """Get results published within last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [r for r in self.results if r.published_date and r.published_date >= cutoff_date]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return {
            "query": self.query,
            "search_type": self.search_type,
            "total_results": self.total_results,
            "results_count": len(self.results),
            "search_time": self.search_time,
            "processing_time": self.processing_time,
            "average_score": self.average_score,
            "quality_score": self.quality_score,
            "results": [
                {
                    "id": r.id,
                    "url": r.url,
                    "title": r.title,
                    "score": r.score,
                    "preview": r.get_preview(),
                    "published_date": r.published_date.isoformat() if r.published_date else None,
                    "domain_authority": r.domain_authority,
                    "content_quality": r.content_quality
                }
                for r in self.results
            ],
            "samsaek_context": self.samsaek_context,
            "cached": self.cached,
            "timestamp": self.timestamp.isoformat()
        }


class ExaSearchConfig(BaseModel):
    """Configuration for Exa search service"""
    api_key: str = Field(..., description="Exa API key")
    base_url: str = Field(default="https://api.exa.ai", description="Exa API base URL")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Retry delay in seconds")
    
    # Rate limiting
    requests_per_minute: int = Field(default=60, description="Rate limit per minute")
    requests_per_hour: int = Field(default=1000, description="Rate limit per hour")
    
    # Caching
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    cache_max_size: int = Field(default=1000, description="Maximum cache entries")
    
    # Default search options
    default_num_results: int = Field(default=10, description="Default number of results")
    default_search_type: ExaSearchType = Field(default=ExaSearchType.NEURAL, description="Default search type")
    default_text_length: int = Field(default=1000, description="Default text length limit")
    
    # Quality filtering
    min_score_threshold: float = Field(default=0.0, description="Minimum score threshold")
    enable_quality_filtering: bool = Field(default=True, description="Enable quality filtering")
    
    # Content processing
    enable_summarization: bool = Field(default=True, description="Enable AI summarization")
    enable_highlight_extraction: bool = Field(default=True, description="Enable highlight extraction")
    
    # Samsaek-specific settings
    samsaek_context: Optional[Dict[str, Any]] = Field(default=None, description="Samsaek context")
    weave_integration: bool = Field(default=True, description="Enable Weave monitoring")
    
    class Config:
        use_enum_values = True
    
    @validator('timeout')
    def validate_timeout(cls, v):
        if v <= 0:
            raise ValueError('timeout must be positive')
        return v
    
    @validator('min_score_threshold')
    def validate_score_threshold(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('min_score_threshold must be between 0.0 and 1.0')
        return v


class ExaSearchStats(BaseModel):
    """Statistics for Exa search operations"""
    total_searches: int = Field(default=0, description="Total searches performed")
    successful_searches: int = Field(default=0, description="Successful searches")
    failed_searches: int = Field(default=0, description="Failed searches")
    cached_searches: int = Field(default=0, description="Cached searches")
    
    # Performance metrics
    average_search_time: float = Field(default=0.0, description="Average search time")
    total_search_time: float = Field(default=0.0, description="Total search time")
    
    # Quality metrics
    average_quality_score: float = Field(default=0.0, description="Average quality score")
    total_results_returned: int = Field(default=0, description="Total results returned")
    
    # Usage metrics
    api_calls_made: int = Field(default=0, description="API calls made")
    tokens_used: int = Field(default=0, description="Tokens used")
    cost_estimate: float = Field(default=0.0, description="Estimated cost")
    
    # Timestamps
    start_time: datetime = Field(default_factory=datetime.now, description="Stats collection start")
    last_update: datetime = Field(default_factory=datetime.now, description="Last update")
    
    def update_stats(self, success: bool, search_time: float, quality_score: float, results_count: int):
        """Update search statistics"""
        self.total_searches += 1
        if success:
            self.successful_searches += 1
        else:
            self.failed_searches += 1
        
        self.total_search_time += search_time
        self.average_search_time = self.total_search_time / self.total_searches
        
        self.total_results_returned += results_count
        
        # Update quality score
        if quality_score > 0:
            current_avg = self.average_quality_score
            self.average_quality_score = ((current_avg * (self.total_searches - 1)) + quality_score) / self.total_searches
        
        self.last_update = datetime.now()
    
    def get_success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_searches == 0:
            return 0.0
        return (self.successful_searches / self.total_searches) * 100.0
    
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.total_searches == 0:
            return 0.0
        return (self.cached_searches / self.total_searches) * 100.0