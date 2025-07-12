"""
Crew AI Orchestration Engine
Core engine for managing multi-agent workflows in Samsaek
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import weave
from crewai import Agent, Task, Crew
from crewai.process import Process
from langchain_google_genai import ChatGoogleGenerativeAI

from ...core.config import config
from ...core.logging import get_logger, SamsaekLoggerMixin, LogOperation
from ...core.exceptions import (
    CrewAIError,
    AgentError,
    TaskError,
    WorkflowError,
    AgentTimeoutError,
    AgentCapacityError
)
from ..gemini.service import GeminiService
from ..a2a.gateway import A2AProtocolGateway
from .models import (
    SamsaekAgent,
    SamsaekTask,
    SamsaekCrew,
    WorkflowConfig,
    TaskStatus,
    WorkflowStatus,
    AgentRole,
    TaskType
)


class AgentManager:
    """Manages agent lifecycle and resource allocation"""
    
    def __init__(self, max_agents: int = 50):
        self.max_agents = max_agents
        self.active_agents: Dict[str, SamsaekAgent] = {}
        self.agent_pool: Dict[str, Agent] = {}  # CrewAI agents
        self.agent_load: Dict[str, int] = {}  # Current task load per agent
        self.logger = get_logger("agent_manager")
    
    def create_agent(self, agent_spec: SamsaekAgent, gemini_service: GeminiService) -> Agent:
        """Create a CrewAI agent from Samsaek agent specification"""
        try:
            # Configure LLM for the agent following CrewAI standards
            llm = ChatGoogleGenerativeAI(
                model=agent_spec.llm_model,
                temperature=agent_spec.llm_temperature,
                max_tokens=agent_spec.llm_max_tokens,
                google_api_key=config.google_api_key
            )
            
            # Create CrewAI agent following official specification
            crew_agent = Agent(
                role=agent_spec.role,
                goal=agent_spec.goal,
                backstory=agent_spec.backstory,
                llm=llm,
                tools=agent_spec.tools,
                max_iter=agent_spec.max_iter,
                max_execution_time=agent_spec.timeout,
                memory=agent_spec.memory_enabled,
                verbose=True,
                allow_delegation=True,
                step_callback=None,  # Optional callback for step monitoring
                system_template=None,  # Optional system template
                prompt_template=None,  # Optional prompt template
                response_template=None,  # Optional response template
                allow_code_execution=False,  # Security setting
                max_retry_limit=agent_spec.max_retry
            )
            
            # Store in pools
            self.active_agents[agent_spec.agent_id] = agent_spec
            self.agent_pool[agent_spec.agent_id] = crew_agent
            self.agent_load[agent_spec.agent_id] = 0
            
            self.logger.info(f"Created agent: {agent_spec.agent_id} ({agent_spec.role})")
            return crew_agent
            
        except Exception as e:
            raise AgentError(
                f"Failed to create agent {agent_spec.agent_id}",
                context={"agent_id": agent_spec.agent_id, "error": str(e)}
            )
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get CrewAI agent by ID"""
        return self.agent_pool.get(agent_id)
    
    def get_available_agent(self, required_role: Optional[AgentRole] = None) -> Optional[str]:
        """Get available agent for task assignment"""
        available_agents = []
        
        for agent_id, agent_spec in self.active_agents.items():
            if agent_spec.status == "idle":
                if required_role is None or agent_spec.role == required_role:
                    current_load = self.agent_load.get(agent_id, 0)
                    available_agents.append((agent_id, current_load))
        
        if not available_agents:
            return None
        
        # Return agent with lowest load
        available_agents.sort(key=lambda x: x[1])
        return available_agents[0][0]
    
    def assign_task(self, agent_id: str, task_id: str):
        """Assign task to agent"""
        if agent_id in self.active_agents:
            self.active_agents[agent_id].status = "busy"
            self.agent_load[agent_id] = self.agent_load.get(agent_id, 0) + 1
            self.logger.debug(f"Assigned task {task_id} to agent {agent_id}")
    
    def release_task(self, agent_id: str, task_id: str):
        """Release task from agent"""
        if agent_id in self.active_agents:
            self.agent_load[agent_id] = max(0, self.agent_load.get(agent_id, 1) - 1)
            if self.agent_load[agent_id] == 0:
                self.active_agents[agent_id].status = "idle"
            self.logger.debug(f"Released task {task_id} from agent {agent_id}")
    
    def get_agent_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        total_agents = len(self.active_agents)
        active_agents = len([a for a in self.active_agents.values() if a.status == "busy"])
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "idle_agents": total_agents - active_agents,
            "average_load": sum(self.agent_load.values()) / max(total_agents, 1),
            "agent_details": [
                {
                    "agent_id": agent.agent_id,
                    "role": agent.role,
                    "status": agent.status,
                    "tasks_completed": agent.tasks_completed,
                    "success_rate": agent.success_rate,
                    "current_load": self.agent_load.get(agent.agent_id, 0)
                }
                for agent in self.active_agents.values()
            ]
        }


class TaskScheduler:
    """Manages task scheduling and execution"""
    
    def __init__(self, max_concurrent_tasks: int = 5):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.running_tasks: Dict[str, SamsaekTask] = {}
        self.task_queue: List[SamsaekTask] = []
        self.completed_tasks: Dict[str, SamsaekTask] = {}
        self.logger = get_logger("task_scheduler")
    
    def schedule_task(self, task: SamsaekTask) -> bool:
        """Schedule a task for execution"""
        if len(self.running_tasks) >= self.max_concurrent_tasks:
            self.task_queue.append(task)
            self.logger.info(f"Task {task.task_id} queued (max concurrent reached)")
            return False
        
        self.running_tasks[task.task_id] = task
        self.logger.info(f"Task {task.task_id} scheduled for execution")
        return True
    
    def complete_task(self, task_id: str, result: Any, success: bool = True):
        """Mark task as completed"""
        if task_id in self.running_tasks:
            task = self.running_tasks.pop(task_id)
            task.complete_execution(result, success)
            self.completed_tasks[task_id] = task
            
            # Schedule next task from queue
            if self.task_queue:
                next_task = self.task_queue.pop(0)
                self.schedule_task(next_task)
    
    def fail_task(self, task_id: str, error_message: str):
        """Mark task as failed"""
        if task_id in self.running_tasks:
            task = self.running_tasks.pop(task_id)
            task.fail_execution(error_message)
            
            # Retry if possible
            if task.can_retry():
                task.retry()
                self.task_queue.append(task)
                self.logger.info(f"Task {task_id} will be retried (attempt {task.retry_count})")
            else:
                self.completed_tasks[task_id] = task
                self.logger.error(f"Task {task_id} failed permanently: {error_message}")
    
    def get_task_metrics(self) -> Dict[str, Any]:
        """Get task scheduling metrics"""
        return {
            "running_tasks": len(self.running_tasks),
            "queued_tasks": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "max_concurrent": self.max_concurrent_tasks,
            "success_rate": self._calculate_success_rate()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate task success rate"""
        if not self.completed_tasks:
            return 0.0
        
        successful_tasks = len([t for t in self.completed_tasks.values() if t.status == TaskStatus.COMPLETED])
        return (successful_tasks / len(self.completed_tasks)) * 100.0


class CrewOrchestrationEngine(SamsaekLoggerMixin):
    """
    Crew AI Orchestration Engine
    Manages multi-agent workflows and task coordination for Samsaek
    """
    
    def __init__(self, gemini_service: GeminiService, a2a_gateway: Optional[A2AProtocolGateway] = None):
        super().__init__()
        
        self.gemini_service = gemini_service
        self.a2a_gateway = a2a_gateway
        
        # Initialize components
        self.agent_manager = AgentManager(max_agents=config.crew_max_agents)
        self.task_scheduler = TaskScheduler(max_concurrent_tasks=5)
        
        # Active workflows
        self.active_crews: Dict[str, SamsaekCrew] = {}
        self.crew_pool: Dict[str, Crew] = {}  # CrewAI crews
        
        # Performance tracking
        self.workflow_count = 0
        self.total_execution_time = 0.0
        
        # Initialize Weave monitoring
        if config.weave_capture_all:
            weave.init("samsaek-crew")
        
        self.logger.info("Crew Orchestration Engine initialized")
    
    @weave.op()
    async def create_crew(self, crew_spec: SamsaekCrew) -> str:
        """Create a new crew for workflow execution"""
        try:
            with LogOperation(f"create_crew_{crew_spec.crew_id}", self.logger,
                             crew_id=crew_spec.crew_id, name=crew_spec.name):
                
                # Create agents
                crew_agents = []
                for agent_spec in crew_spec.agents:
                    crew_agent = self.agent_manager.create_agent(agent_spec, self.gemini_service)
                    crew_agents.append(crew_agent)
                
                # Create tasks
                crew_tasks = []
                for task_spec in crew_spec.tasks:
                    crew_task = self._create_crewai_task(task_spec)
                    crew_tasks.append(crew_task)
                
                # Assign tasks to agents
                self._assign_tasks_to_agents(crew_spec, crew_agents, crew_tasks)
                
                # Create CrewAI crew following official specification
                crew = Crew(
                    agents=crew_agents,
                    tasks=crew_tasks,
                    process=Process.sequential if crew_spec.process == "sequential" else Process.hierarchical,
                    memory=crew_spec.config.memory_enabled,
                    cache=True,
                    max_rpm=10,  # Rate limiting
                    language="en",
                    full_output=True,
                    verbose=True,
                    step_callback=None,  # Optional step callback
                    task_callback=None,  # Optional task callback
                    share_crew=False,  # Don't share crew context
                    config=None,  # Optional crew configuration
                    max_execution_time=crew_spec.config.workflow_timeout,
                    embedder=None,  # Optional embedder for memory
                    planning=False,  # Disable planning mode
                    planning_llm=None  # Optional planning LLM
                )
                
                # Store crew
                self.active_crews[crew_spec.crew_id] = crew_spec
                self.crew_pool[crew_spec.crew_id] = crew
                
                self.logger.info(f"Created crew: {crew_spec.crew_id}")
                return crew_spec.crew_id
                
        except Exception as e:
            raise CrewAIError(
                f"Failed to create crew {crew_spec.crew_id}",
                context={"crew_id": crew_spec.crew_id, "error": str(e)}
            )
    
    @weave.op()
    async def execute_crew(self, crew_id: str) -> Dict[str, Any]:
        """Execute a crew workflow"""
        if crew_id not in self.active_crews:
            raise WorkflowError(f"Crew {crew_id} not found")
        
        crew_spec = self.active_crews[crew_id]
        crew = self.crew_pool[crew_id]
        
        try:
            with LogOperation(f"execute_crew_{crew_id}", self.logger,
                             crew_id=crew_id, name=crew_spec.name):
                
                # Update crew status
                crew_spec.status = WorkflowStatus.RUNNING
                crew_spec.started_at = datetime.now()
                
                # Execute crew
                start_time = time.time()
                result = await asyncio.get_event_loop().run_in_executor(
                    None, crew.kickoff
                )
                execution_time = time.time() - start_time
                
                # Update crew status
                crew_spec.status = WorkflowStatus.COMPLETED
                crew_spec.completed_at = datetime.now()
                crew_spec.final_result = result
                
                # Update metrics
                self.workflow_count += 1
                self.total_execution_time += execution_time
                
                # Update agent metrics
                for agent_spec in crew_spec.agents:
                    agent_spec.update_metrics(True, execution_time / len(crew_spec.agents))
                
                execution_summary = crew_spec.get_execution_summary()
                execution_summary.update({
                    "execution_time": execution_time,
                    "result": result,
                    "samsaek_platform": "samsaek",
                    "engine_version": "2.0.0"
                })
                
                self.logger.info(f"Crew {crew_id} completed successfully", **execution_summary)
                return execution_summary
                
        except Exception as e:
            # Update crew status
            crew_spec.status = WorkflowStatus.FAILED
            crew_spec.completed_at = datetime.now()
            
            # Update agent metrics
            for agent_spec in crew_spec.agents:
                agent_spec.update_metrics(False, 0.0)
            
            self.log_error("execute_crew", e, crew_id=crew_id)
            raise WorkflowError(
                f"Failed to execute crew {crew_id}",
                context={"crew_id": crew_id, "error": str(e)}
            )
    
    def _create_crewai_task(self, task_spec: SamsaekTask) -> Task:
        """Create CrewAI task from Samsaek task specification following official spec"""
        return Task(
            description=task_spec.description,
            expected_output=task_spec.expected_output,
            tools=[],  # Tools will be inherited from agent
            agent=None,  # Will be assigned later
            context=task_spec.context,
            output_json=None,  # Optional JSON output schema
            output_pydantic=None,  # Optional Pydantic output model
            output_file=None,  # Optional output file path
            callback=None,  # Optional callback function
            human_input=False,  # No human input required
            async_execution=False,  # Synchronous execution
            config=None,  # Optional task configuration
            converter_cls=None  # Optional output converter
        )
    
    def _assign_tasks_to_agents(self, crew_spec: SamsaekCrew, crew_agents: List[Agent], crew_tasks: List[Task]):
        """Assign tasks to appropriate agents"""
        for i, task_spec in enumerate(crew_spec.tasks):
            if task_spec.assigned_agent_id:
                # Find specific agent
                agent_spec = crew_spec.get_agent_by_id(task_spec.assigned_agent_id)
                if agent_spec:
                    agent_index = next(
                        (j for j, a in enumerate(crew_spec.agents) if a.agent_id == task_spec.assigned_agent_id),
                        0
                    )
                    crew_tasks[i].agent = crew_agents[agent_index]
            else:
                # Auto-assign based on task type and agent capabilities
                best_agent_index = self._find_best_agent_for_task(task_spec, crew_spec.agents)
                crew_tasks[i].agent = crew_agents[best_agent_index]
    
    def _find_best_agent_for_task(self, task_spec: SamsaekTask, agents: List[SamsaekAgent]) -> int:
        """Find the best agent for a task based on capabilities"""
        # Simple heuristic: match task type to agent role
        task_role_mapping = {
            TaskType.RESEARCH: AgentRole.RESEARCHER,
            TaskType.ANALYSIS: AgentRole.ANALYST,
            TaskType.GENERATION: AgentRole.WRITER,
            TaskType.REVIEW: AgentRole.REVIEWER,
            TaskType.COORDINATION: AgentRole.COORDINATOR,
            TaskType.SUMMARIZATION: AgentRole.SUMMARIZER,
            TaskType.CLASSIFICATION: AgentRole.CLASSIFIER,
            TaskType.EXTRACTION: AgentRole.EXTRACTOR,
            TaskType.VALIDATION: AgentRole.VALIDATOR,
            TaskType.MONITORING: AgentRole.MONITOR
        }
        
        preferred_role = task_role_mapping.get(task_spec.task_type)
        
        # Find agent with matching role
        for i, agent in enumerate(agents):
            if agent.role == preferred_role:
                return i
        
        # Fallback to first available agent
        return 0
    
    async def get_crew_status(self, crew_id: str) -> Dict[str, Any]:
        """Get crew execution status"""
        if crew_id not in self.active_crews:
            raise WorkflowError(f"Crew {crew_id} not found")
        
        crew_spec = self.active_crews[crew_id]
        return crew_spec.get_execution_summary()
    
    async def cancel_crew(self, crew_id: str):
        """Cancel crew execution"""
        if crew_id not in self.active_crews:
            raise WorkflowError(f"Crew {crew_id} not found")
        
        crew_spec = self.active_crews[crew_id]
        crew_spec.status = WorkflowStatus.CANCELLED
        crew_spec.completed_at = datetime.now()
        
        # Release agents
        for agent in crew_spec.agents:
            agent.status = "idle"
        
        self.logger.info(f"Crew {crew_id} cancelled")
    
    async def list_crews(self) -> List[Dict[str, Any]]:
        """List all active crews"""
        return [crew.get_execution_summary() for crew in self.active_crews.values()]
    
    async def get_engine_metrics(self) -> Dict[str, Any]:
        """Get orchestration engine metrics"""
        agent_metrics = self.agent_manager.get_agent_metrics()
        task_metrics = self.task_scheduler.get_task_metrics()
        
        return {
            "workflow_count": self.workflow_count,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": self.total_execution_time / max(self.workflow_count, 1),
            "active_crews": len(self.active_crews),
            "agent_metrics": agent_metrics,
            "task_metrics": task_metrics,
            "platform": "samsaek",
            "engine_version": "2.0.0"
        }
    
    async def create_simple_workflow(
        self,
        workflow_name: str,
        tasks: List[Dict[str, Any]],
        agents: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Create a simple workflow with auto-generated agents"""
        try:
            # Generate agents if not provided
            if not agents:
                agents = self._generate_agents_for_tasks(tasks)
            
            # Create agent specs
            agent_specs = []
            for agent_config in agents:
                agent_spec = SamsaekAgent(
                    name=agent_config["name"],
                    role=agent_config["role"],
                    goal=agent_config["goal"],
                    backstory=agent_config["backstory"],
                    capabilities=agent_config.get("capabilities", []),
                    tools=agent_config.get("tools", [])
                )
                agent_specs.append(agent_spec)
            
            # Create task specs
            task_specs = []
            for task_config in tasks:
                task_spec = SamsaekTask(
                    name=task_config["name"],
                    description=task_config["description"],
                    task_type=task_config.get("task_type", TaskType.CUSTOM),
                    prompt=task_config.get("prompt", task_config["description"]),
                    expected_output=task_config.get("expected_output", "Complete the task as described"),
                    context=task_config.get("context", {})
                )
                task_specs.append(task_spec)
            
            # Create crew
            crew_spec = SamsaekCrew(
                name=workflow_name,
                description=f"Auto-generated workflow: {workflow_name}",
                agents=agent_specs,
                tasks=task_specs,
                config=WorkflowConfig(
                    name=workflow_name,
                    description=f"Simple workflow: {workflow_name}",
                    max_concurrent_tasks=min(len(tasks), 3),
                    samsaek_context={"auto_generated": True}
                ),
                samsaek_context={"auto_generated": True, "task_count": len(tasks)}
            )
            
            crew_id = await self.create_crew(crew_spec)
            self.logger.info(f"Created simple workflow: {workflow_name} (ID: {crew_id})")
            return crew_id
            
        except Exception as e:
            raise WorkflowError(
                f"Failed to create simple workflow {workflow_name}",
                context={"workflow_name": workflow_name, "error": str(e)}
            )
    
    def _generate_agents_for_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate appropriate agents for given tasks"""
        agents = []
        
        # Analyze task types to determine needed agents
        task_types = [task.get("task_type", TaskType.CUSTOM) for task in tasks]
        
        # Create agents based on task requirements
        if any(t in [TaskType.RESEARCH, "research"] for t in task_types):
            agents.append({
                "name": "Research Specialist",
                "role": AgentRole.RESEARCHER,
                "goal": "Gather comprehensive information from various sources",
                "backstory": "Expert researcher with deep knowledge of information gathering and analysis techniques",
                "capabilities": ["web_search", "data_analysis", "fact_checking"],
                "tools": ["search_tool", "web_scraper"]
            })
        
        if any(t in [TaskType.ANALYSIS, "analysis"] for t in task_types):
            agents.append({
                "name": "Analysis Expert",
                "role": AgentRole.ANALYST,
                "goal": "Analyze data and provide insights",
                "backstory": "Data analyst with expertise in pattern recognition and statistical analysis",
                "capabilities": ["data_analysis", "pattern_recognition", "statistical_analysis"],
                "tools": ["analysis_tool", "chart_generator"]
            })
        
        if any(t in [TaskType.GENERATION, "generation"] for t in task_types):
            agents.append({
                "name": "Content Creator",
                "role": AgentRole.WRITER,
                "goal": "Create high-quality content based on requirements",
                "backstory": "Professional writer with expertise in various content formats and styles",
                "capabilities": ["content_creation", "writing", "editing"],
                "tools": ["writing_tool", "grammar_checker"]
            })
        
        # Default agent if no specific agents created
        if not agents:
            agents.append({
                "name": "General Assistant",
                "role": AgentRole.CUSTOM,
                "goal": "Complete various tasks as assigned",
                "backstory": "Versatile assistant capable of handling diverse tasks",
                "capabilities": ["general_assistance", "task_completion"],
                "tools": ["general_tool"]
            })
        
        return agents
    
    async def cleanup(self):
        """Cleanup orchestration engine resources"""
        try:
            # Cancel all active crews
            for crew_id in list(self.active_crews.keys()):
                await self.cancel_crew(crew_id)
            
            # Clear pools
            self.active_crews.clear()
            self.crew_pool.clear()
            
            self.logger.info("Crew Orchestration Engine cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            metrics = await self.get_engine_metrics()
            return {
                "status": "healthy",
                "active_crews": len(self.active_crews),
                "total_workflows": self.workflow_count,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }