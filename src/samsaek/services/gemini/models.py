"""
Google Gemini Integration Models
Pydantic models for Gemini API interactions
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class GeminiModelType(str, Enum):
    """Available Gemini model types"""
    GEMINI_PRO = "gemini-pro"
    GEMINI_PRO_VISION = "gemini-pro-vision"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"


class GeminiContentType(str, Enum):
    """Content types for Gemini requests"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class GeminiSafetyLevel(str, Enum):
    """Safety filtering levels"""
    BLOCK_NONE = "BLOCK_NONE"
    BLOCK_ONLY_HIGH = "BLOCK_ONLY_HIGH"
    BLOCK_MEDIUM_AND_ABOVE = "BLOCK_MEDIUM_AND_ABOVE"
    BLOCK_LOW_AND_ABOVE = "BLOCK_LOW_AND_ABOVE"


class GeminiContent(BaseModel):
    """Content part for Gemini requests"""
    content_type: GeminiContentType
    data: Union[str, bytes]
    mime_type: Optional[str] = None
    
    class Config:
        use_enum_values = True


class GeminiSafetySettings(BaseModel):
    """Safety settings for Gemini requests"""
    category: str
    threshold: GeminiSafetyLevel
    
    class Config:
        use_enum_values = True


class GeminiGenerationConfig(BaseModel):
    """Generation configuration for Gemini"""
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    max_output_tokens: Optional[int] = Field(default=None, ge=1)
    stop_sequences: Optional[List[str]] = None
    candidate_count: Optional[int] = Field(default=1, ge=1, le=8)
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError('temperature must be between 0.0 and 1.0')
        return v


class GeminiRequest(BaseModel):
    """Request model for Gemini API calls"""
    model: GeminiModelType = Field(default=GeminiModelType.GEMINI_PRO)
    contents: List[GeminiContent]
    generation_config: Optional[GeminiGenerationConfig] = None
    safety_settings: Optional[List[GeminiSafetySettings]] = None
    system_instruction: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    
    # Samsaek-specific fields
    samsaek_context: Optional[Dict[str, Any]] = None
    workflow_id: Optional[str] = None
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    priority: Optional[str] = Field(default="normal")
    
    class Config:
        use_enum_values = True
    
    @validator('contents')
    def validate_contents_not_empty(cls, v):
        if not v:
            raise ValueError('contents cannot be empty')
        return v
    
    def to_gemini_request(self) -> Dict[str, Any]:
        """Convert to Gemini API request format"""
        request_data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text" if content.content_type == GeminiContentType.TEXT else "inline_data": 
                            content.data if content.content_type == GeminiContentType.TEXT 
                            else {"mime_type": content.mime_type, "data": content.data}
                        }
                    ]
                }
                for content in self.contents
            ]
        }
        
        if self.generation_config:
            request_data["generationConfig"] = self.generation_config.dict(exclude_none=True)
        
        if self.safety_settings:
            request_data["safetySettings"] = [
                {"category": setting.category, "threshold": setting.threshold}
                for setting in self.safety_settings
            ]
        
        if self.system_instruction:
            request_data["systemInstruction"] = {"parts": [{"text": self.system_instruction}]}
        
        return request_data


class GeminiCitationMetadata(BaseModel):
    """Citation metadata from Gemini responses"""
    citation_sources: List[Dict[str, Any]] = []


class GeminiSafetyRating(BaseModel):
    """Safety rating from Gemini responses"""
    category: str
    probability: str
    blocked: bool = False


class GeminiUsageMetadata(BaseModel):
    """Usage metadata from Gemini responses"""
    prompt_token_count: int
    candidates_token_count: int
    total_token_count: int
    cached_content_token_count: Optional[int] = None


class GeminiCandidate(BaseModel):
    """Candidate response from Gemini"""
    content: Dict[str, Any]
    finish_reason: str
    index: int
    safety_ratings: List[GeminiSafetyRating] = []
    citation_metadata: Optional[GeminiCitationMetadata] = None
    
    @property
    def text(self) -> str:
        """Extract text from candidate content"""
        if "parts" in self.content:
            parts = self.content["parts"]
            if parts and "text" in parts[0]:
                return parts[0]["text"]
        return ""


class GeminiResponse(BaseModel):
    """Response model from Gemini API"""
    candidates: List[GeminiCandidate]
    usage_metadata: GeminiUsageMetadata
    model_version: Optional[str] = None
    
    # Samsaek-specific response fields
    samsaek_metadata: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    processing_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @property
    def text(self) -> str:
        """Get text from first candidate"""
        if self.candidates:
            return self.candidates[0].text
        return ""
    
    @property
    def all_texts(self) -> List[str]:
        """Get text from all candidates"""
        return [candidate.text for candidate in self.candidates]
    
    @property
    def token_count(self) -> int:
        """Get total token count"""
        return self.usage_metadata.total_token_count
    
    @property
    def cost_estimate(self) -> float:
        """Estimate cost based on token usage (placeholder calculation)"""
        # This would be replaced with actual pricing calculation
        input_tokens = self.usage_metadata.prompt_token_count
        output_tokens = self.usage_metadata.candidates_token_count
        
        # Placeholder pricing (per 1K tokens)
        input_cost_per_1k = 0.0005  # $0.0005 per 1K input tokens
        output_cost_per_1k = 0.0015  # $0.0015 per 1K output tokens
        
        return (input_tokens * input_cost_per_1k / 1000) + (output_tokens * output_cost_per_1k / 1000)


class GeminiStreamResponse(BaseModel):
    """Streaming response from Gemini API"""
    chunk_id: str
    content: str
    is_final: bool = False
    metadata: Optional[Dict[str, Any]] = None
    
    # Samsaek-specific streaming fields
    samsaek_context: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class GeminiError(BaseModel):
    """Error response from Gemini API"""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    
    # Samsaek-specific error fields
    samsaek_context: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class GeminiConnectionConfig(BaseModel):
    """Configuration for Gemini service connection"""
    api_key: str
    project_id: Optional[str] = None
    region: str = "us-central1"
    endpoint: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    rate_limit_per_minute: int = 100
    
    # Connection pooling
    max_connections: int = 10
    max_keepalive_connections: int = 5
    keepalive_expiry: int = 300
    
    # Samsaek-specific configuration
    samsaek_agent_id: Optional[str] = None
    samsaek_session_id: Optional[str] = None
    enable_monitoring: bool = True
    enable_caching: bool = True
    cache_ttl: int = 3600
    
    class Config:
        extra = "forbid"