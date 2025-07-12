"""
Crew AI Orchestration Models
Pydantic models for multi-agent workflow management
"""

from typing import Optional, List, Dict, Any, Union, Callable
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
from enum import Enum
import uuid


class AgentRole(str, Enum):
    """Agent role types for specialized functions"""
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    REVIEWER = "reviewer"
    COORDINATOR = "coordinator"
    TRANSLATOR = "translator"
    SUMMARIZER = "summarizer"
    CLASSIFIER = "classifier"
    EXTRACTOR = "extractor"
    VALIDATOR = "validator"
    MONITOR = "monitor"
    CUSTOM = "custom"


class TaskType(str, Enum):
    """Task types for workflow organization"""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    REVIEW = "review"
    COORDINATION = "coordination"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    MONITORING = "monitoring"
    CUSTOM = "custom"


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRY = "retry"


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class AgentCapability(BaseModel):
    """Agent capability specification"""
    name: str = Field(..., description="Capability name")
    description: str = Field(..., description="Capability description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Capability parameters")
    enabled: bool = Field(default=True, description="Whether capability is enabled")


class SamsaekAgent(BaseModel):
    """Samsaek Agent model for Crew AI integration"""
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique agent ID")
    name: str = Field(..., description="Agent name")
    role: AgentRole = Field(..., description="Agent role")
    goal: str = Field(..., description="Agent goal")
    backstory: str = Field(..., description="Agent backstory")
    
    # Capabilities and tools
    capabilities: List[AgentCapability] = Field(default_factory=list, description="Agent capabilities")
    tools: List[str] = Field(default_factory=list, description="Available tools")
    
    # LLM configuration
    llm_model: str = Field(default="gemini-pro", description="LLM model to use")
    llm_temperature: float = Field(default=0.7, description="LLM temperature")
    llm_max_tokens: int = Field(default=8192, description="Maximum tokens")
    
    # Execution settings
    max_iter: int = Field(default=5, description="Maximum iterations")
    max_retry: int = Field(default=3, description="Maximum retry attempts")
    timeout: int = Field(default=300, description="Task timeout in seconds")
    
    # Memory and context
    memory_enabled: bool = Field(default=True, description="Enable agent memory")
    context_window: int = Field(default=4096, description="Context window size")
    
    # Samsaek-specific settings
    samsaek_context: Optional[Dict[str, Any]] = Field(default=None, description="Samsaek context")
    workflow_id: Optional[str] = Field(default=None, description="Associated workflow ID")
    parent_crew_id: Optional[str] = Field(default=None, description="Parent crew ID")
    
    # Status and metrics
    status: str = Field(default="idle", description="Agent status")
    tasks_completed: int = Field(default=0, description="Number of tasks completed")
    success_rate: float = Field(default=0.0, description="Task success rate")
    average_execution_time: float = Field(default=0.0, description="Average execution time")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    last_active: datetime = Field(default_factory=datetime.now, description="Last activity timestamp")
    
    class Config:
        use_enum_values = True
    
    @validator('llm_temperature')
    def validate_temperature(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('Temperature must be between 0.0 and 1.0')
        return v
    
    @validator('max_iter')
    def validate_max_iter(cls, v):
        if v <= 0:
            raise ValueError('max_iter must be positive')
        return v
    
    def add_capability(self, capability: AgentCapability):
        """Add a capability to the agent"""
        self.capabilities.append(capability)
    
    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a specific capability"""
        return any(cap.name == capability_name and cap.enabled for cap in self.capabilities)
    
    def get_capability(self, capability_name: str) -> Optional[AgentCapability]:
        """Get a specific capability"""
        for cap in self.capabilities:
            if cap.name == capability_name:
                return cap
        return None
    
    def update_metrics(self, success: bool, execution_time: float):
        """Update agent performance metrics"""
        self.tasks_completed += 1
        
        # Update success rate
        if success:
            self.success_rate = ((self.success_rate * (self.tasks_completed - 1)) + 1) / self.tasks_completed
        else:
            self.success_rate = (self.success_rate * (self.tasks_completed - 1)) / self.tasks_completed
        
        # Update average execution time
        self.average_execution_time = ((self.average_execution_time * (self.tasks_completed - 1)) + execution_time) / self.tasks_completed
        
        self.last_active = datetime.now()
    
    def to_crewai_agent(self) -> Dict[str, Any]:
        """Convert to CrewAI agent format"""
        return {
            "role": self.role,
            "goal": self.goal,
            "backstory": self.backstory,
            "tools": self.tools,
            "llm": {
                "model": self.llm_model,
                "temperature": self.llm_temperature,
                "max_tokens": self.llm_max_tokens
            },
            "max_iter": self.max_iter,
            "memory": self.memory_enabled,
            "verbose": True,
            "allow_delegation": True
        }


class TaskDependency(BaseModel):
    """Task dependency specification"""
    task_id: str = Field(..., description="Dependent task ID")
    dependency_type: str = Field(default="completion", description="Type of dependency")
    condition: Optional[str] = Field(default=None, description="Dependency condition")


class SamsaekTask(BaseModel):
    """Samsaek Task model for workflow execution"""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique task ID")
    name: str = Field(..., description="Task name")
    description: str = Field(..., description="Task description")
    task_type: TaskType = Field(..., description="Task type")
    
    # Task content
    prompt: str = Field(..., description="Task prompt/instructions")
    expected_output: str = Field(..., description="Expected output description")
    context: Dict[str, Any] = Field(default_factory=dict, description="Task context")
    
    # Assignment and execution
    assigned_agent_id: Optional[str] = Field(default=None, description="Assigned agent ID")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Task status")
    priority: int = Field(default=5, description="Task priority (1-10)")
    
    # Dependencies
    dependencies: List[TaskDependency] = Field(default_factory=list, description="Task dependencies")
    
    # Execution settings
    timeout: int = Field(default=300, description="Task timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_count: int = Field(default=0, description="Current retry count")
    
    # Results and metrics
    result: Optional[Any] = Field(default=None, description="Task result")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    
    # Samsaek-specific fields
    samsaek_context: Optional[Dict[str, Any]] = Field(default=None, description="Samsaek context")
    workflow_id: Optional[str] = Field(default=None, description="Associated workflow ID")
    crew_id: Optional[str] = Field(default=None, description="Associated crew ID")
    
    class Config:
        use_enum_values = True
    
    @validator('priority')
    def validate_priority(cls, v):
        if not (1 <= v <= 10):
            raise ValueError('Priority must be between 1 and 10')
        return v
    
    def add_dependency(self, task_id: str, dependency_type: str = "completion"):
        """Add a task dependency"""
        dependency = TaskDependency(task_id=task_id, dependency_type=dependency_type)
        self.dependencies.append(dependency)
    
    def can_execute(self, completed_tasks: List[str]) -> bool:
        """Check if task can be executed based on dependencies"""
        if not self.dependencies:
            return True
        
        for dep in self.dependencies:
            if dep.task_id not in completed_tasks:
                return False
        
        return True
    
    def start_execution(self):
        """Mark task as started"""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()
    
    def complete_execution(self, result: Any, success: bool = True):
        """Mark task as completed"""
        self.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.result = result
        
        if self.started_at:
            self.execution_time = (self.completed_at - self.started_at).total_seconds()
    
    def fail_execution(self, error_message: str):
        """Mark task as failed"""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error_message
        
        if self.started_at:
            self.execution_time = (self.completed_at - self.started_at).total_seconds()
    
    def can_retry(self) -> bool:
        """Check if task can be retried"""
        return self.retry_count < self.max_retries
    
    def retry(self):
        """Retry task execution"""
        if self.can_retry():
            self.retry_count += 1
            self.status = TaskStatus.RETRY
            self.started_at = None
            self.completed_at = None
            self.error_message = None
    
    def to_crewai_task(self) -> Dict[str, Any]:
        """Convert to CrewAI task format"""
        return {
            "description": self.description,
            "expected_output": self.expected_output,
            "context": self.context,
            "agent": None,  # Will be set by the crew
            "tools": [],
            "async_execution": False,
            "output_json": None,
            "output_pydantic": None,
            "output_file": None,
            "callback": None
        }


class WorkflowConfig(BaseModel):
    """Workflow configuration settings"""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique workflow ID")
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    version: str = Field(default="1.0.0", description="Workflow version")
    
    # Execution settings
    max_concurrent_tasks: int = Field(default=5, description="Maximum concurrent tasks")
    task_timeout: int = Field(default=300, description="Default task timeout")
    workflow_timeout: int = Field(default=3600, description="Workflow timeout")
    
    # Retry and error handling
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Retry delay in seconds")
    continue_on_error: bool = Field(default=False, description="Continue workflow on task failure")
    
    # Memory and context
    memory_enabled: bool = Field(default=True, description="Enable workflow memory")
    context_sharing: bool = Field(default=True, description="Enable context sharing between tasks")
    
    # Monitoring and logging
    enable_monitoring: bool = Field(default=True, description="Enable workflow monitoring")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Samsaek-specific settings
    samsaek_context: Optional[Dict[str, Any]] = Field(default=None, description="Samsaek context")
    weave_integration: bool = Field(default=True, description="Enable Weave monitoring")
    
    class Config:
        use_enum_values = True


class SamsaekCrew(BaseModel):
    """Samsaek Crew model for multi-agent coordination"""
    crew_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique crew ID")
    name: str = Field(..., description="Crew name")
    description: str = Field(..., description="Crew description")
    
    # Crew composition
    agents: List[SamsaekAgent] = Field(default_factory=list, description="Crew agents")
    tasks: List[SamsaekTask] = Field(default_factory=list, description="Crew tasks")
    
    # Execution settings
    process: str = Field(default="sequential", description="Execution process (sequential/hierarchical)")
    manager_agent_id: Optional[str] = Field(default=None, description="Manager agent ID for hierarchical")
    
    # Configuration
    config: WorkflowConfig = Field(default_factory=WorkflowConfig, description="Workflow configuration")
    
    # Status and metrics
    status: WorkflowStatus = Field(default=WorkflowStatus.CREATED, description="Crew status")
    current_task_id: Optional[str] = Field(default=None, description="Currently executing task ID")
    
    # Results
    results: List[Any] = Field(default_factory=list, description="Task results")
    final_result: Optional[Any] = Field(default=None, description="Final crew result")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    
    # Samsaek-specific fields
    samsaek_context: Optional[Dict[str, Any]] = Field(default=None, description="Samsaek context")
    workflow_id: Optional[str] = Field(default=None, description="Associated workflow ID")
    
    class Config:
        use_enum_values = True
    
    def add_agent(self, agent: SamsaekAgent):
        """Add an agent to the crew"""
        agent.parent_crew_id = self.crew_id
        self.agents.append(agent)
    
    def add_task(self, task: SamsaekTask):
        """Add a task to the crew"""
        task.crew_id = self.crew_id
        task.workflow_id = self.workflow_id
        self.tasks.append(task)
    
    def get_agent_by_id(self, agent_id: str) -> Optional[SamsaekAgent]:
        """Get agent by ID"""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None
    
    def get_task_by_id(self, task_id: str) -> Optional[SamsaekTask]:
        """Get task by ID"""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def get_available_agents(self) -> List[SamsaekAgent]:
        """Get agents that are available for task assignment"""
        return [agent for agent in self.agents if agent.status == "idle"]
    
    def get_ready_tasks(self) -> List[SamsaekTask]:
        """Get tasks that are ready for execution"""
        completed_task_ids = [task.task_id for task in self.tasks if task.status == TaskStatus.COMPLETED]
        
        ready_tasks = []
        for task in self.tasks:
            if task.status == TaskStatus.PENDING and task.can_execute(completed_task_ids):
                ready_tasks.append(task)
        
        # Sort by priority (higher priority first)
        ready_tasks.sort(key=lambda t: t.priority, reverse=True)
        return ready_tasks
    
    def get_completion_percentage(self) -> float:
        """Get workflow completion percentage"""
        if not self.tasks:
            return 0.0
        
        completed_tasks = len([task for task in self.tasks if task.status == TaskStatus.COMPLETED])
        return (completed_tasks / len(self.tasks)) * 100.0
    
    def is_completed(self) -> bool:
        """Check if all tasks are completed"""
        return all(task.status == TaskStatus.COMPLETED for task in self.tasks)
    
    def has_failed_tasks(self) -> bool:
        """Check if any tasks have failed"""
        return any(task.status == TaskStatus.FAILED for task in self.tasks)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        total_tasks = len(self.tasks)
        completed_tasks = len([task for task in self.tasks if task.status == TaskStatus.COMPLETED])
        failed_tasks = len([task for task in self.tasks if task.status == TaskStatus.FAILED])
        
        total_execution_time = 0.0
        for task in self.tasks:
            if task.execution_time:
                total_execution_time += task.execution_time
        
        return {
            "crew_id": self.crew_id,
            "name": self.name,
            "status": self.status,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "completion_percentage": self.get_completion_percentage(),
            "total_execution_time": total_execution_time,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "samsaek_context": self.samsaek_context
        }
    
    def to_crewai_crew(self) -> Dict[str, Any]:
        """Convert to CrewAI crew format"""
        return {
            "agents": [agent.to_crewai_agent() for agent in self.agents],
            "tasks": [task.to_crewai_task() for task in self.tasks],
            "process": self.process,
            "manager_llm": None,  # Will be set if hierarchical
            "function_calling_llm": None,
            "memory": self.config.memory_enabled,
            "cache": True,
            "max_rpm": 10,
            "language": "en",
            "full_output": True,
            "step_callback": None,
            "task_callback": None,
            "share_crew": False,
            "output_log_file": None,
            "logger": None
        }