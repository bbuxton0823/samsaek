"""
A2A Protocol Models
Pydantic models for Agent-to-Agent communication protocol
"""

from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
from enum import Enum
import uuid


class A2AMessageType(str, Enum):
    """A2A message types"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class A2AMessagePriority(str, Enum):
    """A2A message priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class A2AAgentCapability(str, Enum):
    """Agent capability types"""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    COORDINATION = "coordination"
    MONITORING = "monitoring"
    CUSTOM = "custom"


class A2AAgentStatus(str, Enum):
    """Agent status"""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    IDLE = "idle"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class A2ASecurityLevel(str, Enum):
    """Security levels for A2A communication"""
    NONE = "none"
    BASIC = "basic"
    ENCRYPTED = "encrypted"
    SIGNED = "signed"
    FULL = "full"  # Both encrypted and signed


class A2AAgent(BaseModel):
    """A2A Agent representation"""
    agent_id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent display name")
    version: str = Field(default="1.0.0", description="Agent version")
    capabilities: List[A2AAgentCapability] = Field(default=[], description="Agent capabilities")
    status: A2AAgentStatus = Field(default=A2AAgentStatus.ONLINE, description="Agent status")
    endpoint: Optional[str] = Field(default=None, description="Agent endpoint URL")
    last_seen: datetime = Field(default_factory=datetime.now, description="Last seen timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Agent metadata")
    
    # Samsaek-specific fields
    samsaek_context: Optional[Dict[str, Any]] = Field(default=None, description="Samsaek context")
    workflow_id: Optional[str] = Field(default=None, description="Associated workflow ID")
    parent_agent_id: Optional[str] = Field(default=None, description="Parent agent ID")
    child_agents: List[str] = Field(default=[], description="Child agent IDs")
    
    class Config:
        use_enum_values = True
    
    @validator('agent_id')
    def validate_agent_id(cls, v):
        if not v or len(v) < 3:
            raise ValueError('agent_id must be at least 3 characters')
        return v
    
    def is_available(self) -> bool:
        """Check if agent is available for communication"""
        return self.status in [A2AAgentStatus.ONLINE, A2AAgentStatus.IDLE]
    
    def can_handle_capability(self, capability: A2AAgentCapability) -> bool:
        """Check if agent can handle a specific capability"""
        return capability in self.capabilities
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "version": self.version,
            "capabilities": [cap.value for cap in self.capabilities],
            "status": self.status.value,
            "endpoint": self.endpoint,
            "last_seen": self.last_seen.isoformat(),
            "metadata": self.metadata,
            "samsaek_context": self.samsaek_context
        }


class A2AMessagePayload(BaseModel):
    """A2A message payload"""
    content: Union[str, Dict[str, Any]] = Field(..., description="Message content")
    content_type: str = Field(default="text/plain", description="Content type")
    schema_version: str = Field(default="1.0", description="Payload schema version")
    
    # Samsaek-specific payload fields
    samsaek_task_id: Optional[str] = Field(default=None, description="Task ID")
    samsaek_workflow_id: Optional[str] = Field(default=None, description="Workflow ID")
    samsaek_context: Optional[Dict[str, Any]] = Field(default=None, description="Samsaek context")


class A2AMessage(BaseModel):
    """A2A Protocol Message"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique message ID")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for threading")
    message_type: A2AMessageType = Field(..., description="Message type")
    priority: A2AMessagePriority = Field(default=A2AMessagePriority.NORMAL, description="Message priority")
    
    # Agent information
    from_agent: str = Field(..., description="Sender agent ID")
    to_agent: str = Field(..., description="Recipient agent ID")
    
    # Message content
    payload: A2AMessagePayload = Field(..., description="Message payload")
    
    # Routing and delivery
    ttl: int = Field(default=3600, description="Time to live in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_count: int = Field(default=0, description="Current retry count")
    
    # Security
    security_level: A2ASecurityLevel = Field(default=A2ASecurityLevel.BASIC, description="Security level")
    signature: Optional[str] = Field(default=None, description="Message signature")
    encryption_key_id: Optional[str] = Field(default=None, description="Encryption key ID")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Message creation time")
    expires_at: Optional[datetime] = Field(default=None, description="Message expiration time")
    delivered_at: Optional[datetime] = Field(default=None, description="Message delivery time")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Message metadata")
    
    # Samsaek-specific fields
    samsaek_platform: str = Field(default="samsaek", description="Platform identifier")
    samsaek_version: str = Field(default="2.0.0", description="Samsaek version")
    samsaek_session_id: Optional[str] = Field(default=None, description="Samsaek session ID")
    
    class Config:
        use_enum_values = True
    
    def __init__(self, **data):
        super().__init__(**data)
        # Set expiration time based on TTL
        if not self.expires_at:
            self.expires_at = self.created_at + timedelta(seconds=self.ttl)
    
    @validator('ttl')
    def validate_ttl(cls, v):
        if v <= 0:
            raise ValueError('TTL must be positive')
        return v
    
    @validator('max_retries')
    def validate_max_retries(cls, v):
        if v < 0:
            raise ValueError('max_retries cannot be negative')
        return v
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if not self.expires_at:
            return False
        return datetime.now() > self.expires_at
    
    def can_retry(self) -> bool:
        """Check if message can be retried"""
        return self.retry_count < self.max_retries
    
    def increment_retry(self):
        """Increment retry count"""
        self.retry_count += 1
    
    def mark_delivered(self):
        """Mark message as delivered"""
        self.delivered_at = datetime.now()
    
    def to_wire_format(self) -> Dict[str, Any]:
        """Convert message to wire format for transmission"""
        return {
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "message_type": self.message_type,
            "priority": self.priority,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "payload": {
                "content": self.payload.content,
                "content_type": self.payload.content_type,
                "schema_version": self.payload.schema_version,
                "samsaek_task_id": self.payload.samsaek_task_id,
                "samsaek_workflow_id": self.payload.samsaek_workflow_id,
                "samsaek_context": self.payload.samsaek_context
            },
            "ttl": self.ttl,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "security_level": self.security_level,
            "signature": self.signature,
            "encryption_key_id": self.encryption_key_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "metadata": self.metadata,
            "samsaek_platform": self.samsaek_platform,
            "samsaek_version": self.samsaek_version,
            "samsaek_session_id": self.samsaek_session_id
        }


class A2AResponse(BaseModel):
    """A2A Protocol Response"""
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique response ID")
    request_message_id: str = Field(..., description="Original request message ID")
    from_agent: str = Field(..., description="Responding agent ID")
    to_agent: str = Field(..., description="Original requesting agent ID")
    
    # Response status
    status: str = Field(..., description="Response status")
    success: bool = Field(..., description="Success flag")
    
    # Response content
    payload: Optional[A2AMessagePayload] = Field(default=None, description="Response payload")
    error_code: Optional[str] = Field(default=None, description="Error code if failed")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.now, description="Response creation time")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    
    # Samsaek-specific fields
    samsaek_platform: str = Field(default="samsaek", description="Platform identifier")
    samsaek_context: Optional[Dict[str, Any]] = Field(default=None, description="Samsaek context")
    
    class Config:
        use_enum_values = True
    
    def to_wire_format(self) -> Dict[str, Any]:
        """Convert response to wire format"""
        return {
            "response_id": self.response_id,
            "request_message_id": self.request_message_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "status": self.status,
            "success": self.success,
            "payload": self.payload.dict() if self.payload else None,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "processing_time": self.processing_time,
            "metadata": self.metadata,
            "samsaek_platform": self.samsaek_platform,
            "samsaek_context": self.samsaek_context
        }


class A2ARoutingRule(BaseModel):
    """Routing rule for A2A messages"""
    rule_id: str = Field(..., description="Unique rule ID")
    name: str = Field(..., description="Rule name")
    source_pattern: str = Field(..., description="Source agent pattern")
    destination_pattern: str = Field(..., description="Destination agent pattern")
    message_type_filter: Optional[List[A2AMessageType]] = Field(default=None, description="Message type filter")
    capability_filter: Optional[List[A2AAgentCapability]] = Field(default=None, description="Capability filter")
    priority_filter: Optional[List[A2AMessagePriority]] = Field(default=None, description="Priority filter")
    
    # Routing actions
    transform_payload: bool = Field(default=False, description="Transform payload")
    add_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata to add")
    
    # Rule status
    enabled: bool = Field(default=True, description="Rule enabled")
    created_at: datetime = Field(default_factory=datetime.now, description="Rule creation time")
    
    class Config:
        use_enum_values = True


class A2AProtocolConfig(BaseModel):
    """Configuration for A2A Protocol Gateway"""
    protocol_version: str = Field(default="1.0", description="A2A protocol version")
    agent_id: str = Field(..., description="This agent's ID")
    agent_name: str = Field(..., description="This agent's name")
    agent_capabilities: List[A2AAgentCapability] = Field(default=[], description="Agent capabilities")
    
    # Security settings
    encryption_enabled: bool = Field(default=True, description="Enable message encryption")
    signature_enabled: bool = Field(default=True, description="Enable message signatures")
    security_level: A2ASecurityLevel = Field(default=A2ASecurityLevel.FULL, description="Default security level")
    
    # Network settings
    listen_port: int = Field(default=8080, description="Port to listen on")
    max_connections: int = Field(default=100, description="Maximum concurrent connections")
    connection_timeout: int = Field(default=30, description="Connection timeout in seconds")
    
    # Message settings
    default_ttl: int = Field(default=3600, description="Default message TTL")
    max_message_size: int = Field(default=10485760, description="Maximum message size in bytes")  # 10MB
    retry_attempts: int = Field(default=3, description="Default retry attempts")
    retry_delay: float = Field(default=1.0, description="Retry delay in seconds")
    
    # Queue settings
    message_queue_size: int = Field(default=1000, description="Message queue size")
    batch_size: int = Field(default=10, description="Batch processing size")
    flush_interval: int = Field(default=5, description="Flush interval in seconds")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=8081, description="Metrics port")
    log_level: str = Field(default="INFO", description="Log level")
    
    # Samsaek-specific settings
    samsaek_platform: str = Field(default="samsaek", description="Platform identifier")
    samsaek_version: str = Field(default="2.0.0", description="Samsaek version")
    weave_integration: bool = Field(default=True, description="Enable Weave integration")
    
    class Config:
        use_enum_values = True
    
    @validator('listen_port')
    def validate_port(cls, v):
        if not (1 <= v <= 65535):
            raise ValueError('Port must be between 1 and 65535')
        return v
    
    @validator('max_message_size')
    def validate_message_size(cls, v):
        if v <= 0:
            raise ValueError('max_message_size must be positive')
        return v


class A2AMessageStats(BaseModel):
    """Statistics for A2A messages"""
    total_messages: int = Field(default=0, description="Total messages processed")
    successful_messages: int = Field(default=0, description="Successful messages")
    failed_messages: int = Field(default=0, description="Failed messages")
    retry_messages: int = Field(default=0, description="Retried messages")
    expired_messages: int = Field(default=0, description="Expired messages")
    
    # Message types
    request_messages: int = Field(default=0, description="Request messages")
    response_messages: int = Field(default=0, description="Response messages")
    notification_messages: int = Field(default=0, description="Notification messages")
    
    # Timing
    average_processing_time: float = Field(default=0.0, description="Average processing time")
    total_processing_time: float = Field(default=0.0, description="Total processing time")
    
    # Timestamps
    start_time: datetime = Field(default_factory=datetime.now, description="Stats collection start time")
    last_update: datetime = Field(default_factory=datetime.now, description="Last stats update")
    
    def update_stats(self, success: bool, processing_time: float, message_type: A2AMessageType):
        """Update statistics"""
        self.total_messages += 1
        if success:
            self.successful_messages += 1
        else:
            self.failed_messages += 1
        
        self.total_processing_time += processing_time
        self.average_processing_time = self.total_processing_time / self.total_messages
        
        if message_type == A2AMessageType.REQUEST:
            self.request_messages += 1
        elif message_type == A2AMessageType.RESPONSE:
            self.response_messages += 1
        elif message_type == A2AMessageType.NOTIFICATION:
            self.notification_messages += 1
        
        self.last_update = datetime.now()
    
    def get_success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_messages == 0:
            return 0.0
        return self.successful_messages / self.total_messages * 100.0