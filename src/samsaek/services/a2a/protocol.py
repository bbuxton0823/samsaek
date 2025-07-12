"""
A2A Protocol Implementation
JSON-RPC 2.0 over HTTP(S) implementation following official A2A specification
"""

import json
import uuid
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class A2AJSONRPCVersion(str, Enum):
    """JSON-RPC version"""
    V2_0 = "2.0"


class A2AMethodName(str, Enum):
    """A2A Protocol method names"""
    DISCOVER = "discover"
    QUERY_SKILL = "querySkill"
    EXECUTE_SKILL = "executeSkill"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    NOTIFY = "notify"
    PING = "ping"
    HEALTH_CHECK = "healthCheck"


class A2AAgentCard(BaseModel):
    """A2A Agent Card as per official specification"""
    id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent display name")
    description: str = Field(..., description="Agent description")
    version: str = Field(default="1.0.0", description="Agent version")
    
    # Connection information
    endpoint: str = Field(..., description="Agent endpoint URL")
    supported_protocols: List[str] = Field(default=["http", "https"], description="Supported protocols")
    
    # Capabilities
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    skills: List[Dict[str, Any]] = Field(default_factory=list, description="Agent skills")
    
    # Authentication and security
    auth_methods: List[str] = Field(default=["none"], description="Supported authentication methods")
    security_level: str = Field(default="basic", description="Security level")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    # Samsaek-specific extensions
    samsaek_platform: str = Field(default="samsaek", description="Platform identifier")
    samsaek_version: str = Field(default="2.0.0", description="Samsaek version")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "endpoint": self.endpoint,
            "supported_protocols": self.supported_protocols,
            "capabilities": self.capabilities,
            "skills": self.skills,
            "auth_methods": self.auth_methods,
            "security_level": self.security_level,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "samsaek_platform": self.samsaek_platform,
            "samsaek_version": self.samsaek_version
        }


class A2AJSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 Request as per A2A specification"""
    jsonrpc: A2AJSONRPCVersion = Field(default=A2AJSONRPCVersion.V2_0, description="JSON-RPC version")
    method: A2AMethodName = Field(..., description="Method name")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Method parameters")
    id: Union[str, int] = Field(default_factory=lambda: str(uuid.uuid4()), description="Request ID")
    
    # A2A-specific fields
    agent_id: str = Field(..., description="Requesting agent ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Request timestamp")
    
    class Config:
        use_enum_values = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
            "id": self.id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat()
        }
        
        if self.params is not None:
            result["params"] = self.params
        
        return result


class A2AJSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 Response as per A2A specification"""
    jsonrpc: A2AJSONRPCVersion = Field(default=A2AJSONRPCVersion.V2_0, description="JSON-RPC version")
    result: Optional[Any] = Field(default=None, description="Method result")
    error: Optional[Dict[str, Any]] = Field(default=None, description="Error information")
    id: Union[str, int] = Field(..., description="Request ID")
    
    # A2A-specific fields
    agent_id: str = Field(..., description="Responding agent ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    class Config:
        use_enum_values = True
    
    @validator('result', 'error')
    def validate_result_or_error(cls, v, values):
        # Exactly one of result or error must be present
        if 'result' in values and 'error' in values:
            if (values.get('result') is None) == (values.get('error') is None):
                raise ValueError('Exactly one of result or error must be present')
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat()
        }
        
        if self.result is not None:
            result["result"] = self.result
        
        if self.error is not None:
            result["error"] = self.error
        
        return result


class A2AJSONRPCNotification(BaseModel):
    """JSON-RPC 2.0 Notification as per A2A specification"""
    jsonrpc: A2AJSONRPCVersion = Field(default=A2AJSONRPCVersion.V2_0, description="JSON-RPC version")
    method: A2AMethodName = Field(..., description="Method name")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Method parameters")
    
    # A2A-specific fields (notifications don't have id)
    agent_id: str = Field(..., description="Notifying agent ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Notification timestamp")
    
    class Config:
        use_enum_values = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat()
        }
        
        if self.params is not None:
            result["params"] = self.params
        
        return result


class A2AError(BaseModel):
    """A2A JSON-RPC Error object"""
    code: int = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    data: Optional[Any] = Field(default=None, description="Additional error data")
    
    # Common A2A error codes
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # A2A-specific error codes
    AGENT_NOT_FOUND = -32001
    SKILL_NOT_AVAILABLE = -32002
    AUTHENTICATION_FAILED = -32003
    AUTHORIZATION_FAILED = -32004
    AGENT_UNAVAILABLE = -32005
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "code": self.code,
            "message": self.message
        }
        
        if self.data is not None:
            result["data"] = self.data
        
        return result


class A2ASkill(BaseModel):
    """A2A Skill definition"""
    name: str = Field(..., description="Skill name")
    description: str = Field(..., description="Skill description")
    version: str = Field(default="1.0.0", description="Skill version")
    
    # Skill metadata
    input_schema: Optional[Dict[str, Any]] = Field(default=None, description="Input schema")
    output_schema: Optional[Dict[str, Any]] = Field(default=None, description="Output schema")
    
    # Execution properties
    async_execution: bool = Field(default=False, description="Supports async execution")
    streaming: bool = Field(default=False, description="Supports streaming")
    
    # Requirements
    required_capabilities: List[str] = Field(default_factory=list, description="Required capabilities")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "async_execution": self.async_execution,
            "streaming": self.streaming,
            "required_capabilities": self.required_capabilities
        }


class A2ADiscoverRequest(BaseModel):
    """A2A Discover method request parameters"""
    capabilities: Optional[List[str]] = Field(default=None, description="Required capabilities")
    skills: Optional[List[str]] = Field(default=None, description="Required skills")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Additional filters")


class A2ADiscoverResponse(BaseModel):
    """A2A Discover method response"""
    agents: List[A2AAgentCard] = Field(..., description="Discovered agents")
    total_count: int = Field(..., description="Total number of agents")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "agents": [agent.to_dict() for agent in self.agents],
            "total_count": self.total_count
        }


class A2AQuerySkillRequest(BaseModel):
    """A2A QuerySkill method request parameters"""
    skill_name: str = Field(..., description="Skill name to query")
    version: Optional[str] = Field(default=None, description="Skill version")


class A2AQuerySkillResponse(BaseModel):
    """A2A QuerySkill method response"""
    available: bool = Field(..., description="Skill availability")
    skill: Optional[A2ASkill] = Field(default=None, description="Skill details")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {"available": self.available}
        if self.skill:
            result["skill"] = self.skill.to_dict()
        return result


class A2AExecuteSkillRequest(BaseModel):
    """A2A ExecuteSkill method request parameters"""
    skill_name: str = Field(..., description="Skill name to execute")
    input_data: Dict[str, Any] = Field(..., description="Input data for skill")
    execution_options: Optional[Dict[str, Any]] = Field(default=None, description="Execution options")
    
    # Execution settings
    async_execution: bool = Field(default=False, description="Execute asynchronously")
    streaming: bool = Field(default=False, description="Stream results")
    timeout: Optional[int] = Field(default=None, description="Execution timeout")


class A2AExecuteSkillResponse(BaseModel):
    """A2A ExecuteSkill method response"""
    execution_id: str = Field(..., description="Execution ID")
    status: str = Field(..., description="Execution status")
    output_data: Optional[Dict[str, Any]] = Field(default=None, description="Output data")
    error: Optional[str] = Field(default=None, description="Error message")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "execution_id": self.execution_id,
            "status": self.status
        }
        
        if self.output_data is not None:
            result["output_data"] = self.output_data
        
        if self.error is not None:
            result["error"] = self.error
        
        return result


# Standard A2A method implementations
def create_discover_request(
    agent_id: str,
    capabilities: Optional[List[str]] = None,
    skills: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None
) -> A2AJSONRPCRequest:
    """Create A2A discover request"""
    params = A2ADiscoverRequest(
        capabilities=capabilities,
        skills=skills,
        filters=filters
    )
    
    return A2AJSONRPCRequest(
        method=A2AMethodName.DISCOVER,
        params=params.dict(exclude_none=True),
        agent_id=agent_id
    )


def create_query_skill_request(
    agent_id: str,
    skill_name: str,
    version: Optional[str] = None
) -> A2AJSONRPCRequest:
    """Create A2A query skill request"""
    params = A2AQuerySkillRequest(
        skill_name=skill_name,
        version=version
    )
    
    return A2AJSONRPCRequest(
        method=A2AMethodName.QUERY_SKILL,
        params=params.dict(exclude_none=True),
        agent_id=agent_id
    )


def create_execute_skill_request(
    agent_id: str,
    skill_name: str,
    input_data: Dict[str, Any],
    execution_options: Optional[Dict[str, Any]] = None,
    async_execution: bool = False,
    streaming: bool = False,
    timeout: Optional[int] = None
) -> A2AJSONRPCRequest:
    """Create A2A execute skill request"""
    params = A2AExecuteSkillRequest(
        skill_name=skill_name,
        input_data=input_data,
        execution_options=execution_options,
        async_execution=async_execution,
        streaming=streaming,
        timeout=timeout
    )
    
    return A2AJSONRPCRequest(
        method=A2AMethodName.EXECUTE_SKILL,
        params=params.dict(exclude_none=True),
        agent_id=agent_id
    )


def create_success_response(
    request_id: Union[str, int],
    agent_id: str,
    result: Any
) -> A2AJSONRPCResponse:
    """Create successful A2A response"""
    return A2AJSONRPCResponse(
        id=request_id,
        agent_id=agent_id,
        result=result
    )


def create_error_response(
    request_id: Union[str, int],
    agent_id: str,
    error_code: int,
    error_message: str,
    error_data: Optional[Any] = None
) -> A2AJSONRPCResponse:
    """Create error A2A response"""
    error = A2AError(
        code=error_code,
        message=error_message,
        data=error_data
    )
    
    return A2AJSONRPCResponse(
        id=request_id,
        agent_id=agent_id,
        error=error.to_dict()
    )


def create_notification(
    agent_id: str,
    method: A2AMethodName,
    params: Optional[Dict[str, Any]] = None
) -> A2AJSONRPCNotification:
    """Create A2A notification"""
    return A2AJSONRPCNotification(
        method=method,
        params=params,
        agent_id=agent_id
    )