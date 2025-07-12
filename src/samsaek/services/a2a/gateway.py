"""
A2A Protocol Gateway
Core implementation of Google's Agent-to-Agent communication protocol
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Callable, Set
from datetime import datetime, timedelta
import aiohttp
from aiohttp import web, WSMsgType
import aioredis
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import weave

from ...core.config import config
from ...core.logging import get_logger, SamsaekLoggerMixin, LogOperation
from ...core.exceptions import (
    A2AProtocolError,
    A2AMessageError,
    A2ARoutingError,
    A2AEncryptionError,
    TimeoutError
)
from .models import (
    A2AMessage,
    A2AResponse,
    A2AAgent,
    A2AProtocolConfig,
    A2AMessageType,
    A2ASecurityLevel,
    A2AMessageStats,
    A2ARoutingRule,
    A2AMessagePayload
)
from .security import A2ACryptoManager


class A2AMessageRouter:
    """Message routing engine for A2A protocol"""
    
    def __init__(self):
        self.routing_rules: List[A2ARoutingRule] = []
        self.logger = get_logger("a2a_router")
    
    def add_routing_rule(self, rule: A2ARoutingRule):
        """Add a routing rule"""
        self.routing_rules.append(rule)
        self.logger.info(f"Added routing rule: {rule.name}")
    
    def remove_routing_rule(self, rule_id: str):
        """Remove a routing rule"""
        self.routing_rules = [rule for rule in self.routing_rules if rule.rule_id != rule_id]
        self.logger.info(f"Removed routing rule: {rule_id}")
    
    def route_message(self, message: A2AMessage) -> A2AMessage:
        """Apply routing rules to message"""
        for rule in self.routing_rules:
            if not rule.enabled:
                continue
                
            if self._matches_rule(message, rule):
                self.logger.debug(f"Applying routing rule: {rule.name}")
                message = self._apply_rule(message, rule)
        
        return message
    
    def _matches_rule(self, message: A2AMessage, rule: A2ARoutingRule) -> bool:
        """Check if message matches routing rule"""
        # Check source pattern
        if not self._matches_pattern(message.from_agent, rule.source_pattern):
            return False
        
        # Check destination pattern
        if not self._matches_pattern(message.to_agent, rule.destination_pattern):
            return False
        
        # Check message type filter
        if rule.message_type_filter and message.message_type not in rule.message_type_filter:
            return False
        
        # Check priority filter
        if rule.priority_filter and message.priority not in rule.priority_filter:
            return False
        
        return True
    
    def _matches_pattern(self, value: str, pattern: str) -> bool:
        """Check if value matches pattern (supports wildcards)"""
        if pattern == "*":
            return True
        
        if pattern.endswith("*"):
            return value.startswith(pattern[:-1])
        
        if pattern.startswith("*"):
            return value.endswith(pattern[1:])
        
        return value == pattern
    
    def _apply_rule(self, message: A2AMessage, rule: A2ARoutingRule) -> A2AMessage:
        """Apply routing rule to message"""
        # Add metadata
        if rule.add_metadata:
            message.metadata.update(rule.add_metadata)
        
        # Transform payload if needed
        if rule.transform_payload:
            # This would contain custom transformation logic
            self.logger.debug(f"Transforming payload for rule: {rule.name}")
        
        return message


class A2AMessageQueue:
    """Message queue for A2A protocol"""
    
    def __init__(self, redis_url: str, kafka_config: Dict[str, Any]):
        self.redis_url = redis_url
        self.kafka_config = kafka_config
        self.redis_client = None
        self.kafka_producer = None
        self.kafka_consumer = None
        self.logger = get_logger("a2a_queue")
        self.message_handlers: Dict[str, Callable] = {}
    
    async def initialize(self):
        """Initialize message queue connections"""
        try:
            # Initialize Redis
            self.redis_client = await aioredis.from_url(self.redis_url)
            
            # Initialize Kafka
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.kafka_config["bootstrap_servers"],
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            
            self.kafka_consumer = KafkaConsumer(
                'samsaek-a2a-messages',
                bootstrap_servers=self.kafka_config["bootstrap_servers"],
                group_id=self.kafka_config["consumer_group"],
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )
            
            self.logger.info("Message queue initialized successfully")
            
        except Exception as e:
            raise A2AProtocolError(
                "Failed to initialize message queue",
                context={"error": str(e)}
            )
    
    async def publish_message(self, message: A2AMessage) -> bool:
        """Publish message to queue"""
        try:
            message_data = message.to_wire_format()
            
            # Publish to Kafka for persistent storage
            self.kafka_producer.send('samsaek-a2a-messages', message_data)
            
            # Store in Redis for quick access
            await self.redis_client.setex(
                f"a2a:message:{message.message_id}",
                message.ttl,
                json.dumps(message_data)
            )
            
            # Add to recipient's queue
            await self.redis_client.lpush(
                f"a2a:queue:{message.to_agent}",
                message.message_id
            )
            
            self.logger.info(f"Message published: {message.message_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish message: {str(e)}")
            return False
    
    async def consume_messages(self, agent_id: str) -> List[A2AMessage]:
        """Consume messages for agent"""
        try:
            messages = []
            
            # Get message IDs from agent's queue
            message_ids = await self.redis_client.lrange(f"a2a:queue:{agent_id}", 0, -1)
            
            for message_id in message_ids:
                message_data = await self.redis_client.get(f"a2a:message:{message_id}")
                if message_data:
                    message_dict = json.loads(message_data)
                    message = A2AMessage.parse_obj(message_dict)
                    
                    # Check if message has expired
                    if not message.is_expired():
                        messages.append(message)
                    else:
                        # Remove expired message
                        await self.redis_client.delete(f"a2a:message:{message_id}")
                        await self.redis_client.lrem(f"a2a:queue:{agent_id}", 0, message_id)
            
            return messages
            
        except Exception as e:
            raise A2AProtocolError(
                f"Failed to consume messages for agent {agent_id}",
                context={"agent_id": agent_id, "error": str(e)}
            )
    
    async def acknowledge_message(self, message_id: str, agent_id: str):
        """Acknowledge message processing"""
        try:
            await self.redis_client.delete(f"a2a:message:{message_id}")
            await self.redis_client.lrem(f"a2a:queue:{agent_id}", 0, message_id)
            self.logger.debug(f"Message acknowledged: {message_id}")
        except Exception as e:
            self.logger.error(f"Failed to acknowledge message {message_id}: {str(e)}")
    
    async def cleanup(self):
        """Cleanup queue connections"""
        if self.redis_client:
            await self.redis_client.close()
        if self.kafka_producer:
            self.kafka_producer.close()
        if self.kafka_consumer:
            self.kafka_consumer.close()


class A2AProtocolGateway(SamsaekLoggerMixin):
    """
    A2A Protocol Gateway
    Implements Google's Agent-to-Agent communication protocol for Samsaek
    """
    
    def __init__(self, config: A2AProtocolConfig):
        super().__init__()
        self.config = config
        self.agent_registry: Dict[str, A2AAgent] = {}
        self.message_stats = A2AMessageStats()
        self.is_running = False
        
        # Initialize components
        self.crypto_manager = A2ACryptoManager(config.agent_id)
        self.router = A2AMessageRouter()
        self.message_queue = A2AMessageQueue(
            redis_url=config.redis_url if hasattr(config, 'redis_url') else "redis://localhost:6379",
            kafka_config={
                "bootstrap_servers": config.kafka_servers if hasattr(config, 'kafka_servers') else ["localhost:9092"],
                "consumer_group": "samsaek-a2a-gateway"
            }
        )
        
        # WebSocket connections
        self.websocket_connections: Set[web.WebSocketResponse] = set()
        
        # Message handlers
        self.message_handlers: Dict[A2AMessageType, Callable] = {
            A2AMessageType.REQUEST: self._handle_request,
            A2AMessageType.RESPONSE: self._handle_response,
            A2AMessageType.NOTIFICATION: self._handle_notification,
            A2AMessageType.HEARTBEAT: self._handle_heartbeat,
            A2AMessageType.ERROR: self._handle_error
        }
        
        # Initialize Weave monitoring if enabled
        if config.weave_integration:
            weave.init("samsaek-a2a")
        
        self.logger.info("A2A Protocol Gateway initialized",
                        agent_id=config.agent_id,
                        protocol_version=config.protocol_version)
    
    async def start(self):
        """Start the A2A gateway"""
        try:
            with LogOperation("a2a_gateway_start", self.logger):
                # Initialize message queue
                await self.message_queue.initialize()
                
                # Register this agent
                await self.register_agent(A2AAgent(
                    agent_id=self.config.agent_id,
                    name=self.config.agent_name,
                    capabilities=self.config.agent_capabilities,
                    endpoint=f"ws://localhost:{self.config.listen_port}/ws"
                ))
                
                # Start HTTP server
                app = web.Application()
                app.router.add_get('/ws', self._websocket_handler)
                app.router.add_post('/message', self._http_message_handler)
                app.router.add_get('/agents', self._get_agents_handler)
                app.router.add_get('/health', self._health_check_handler)
                app.router.add_get('/metrics', self._metrics_handler)
                
                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, '0.0.0.0', self.config.listen_port)
                await site.start()
                
                # Start message processing loop
                asyncio.create_task(self._message_processing_loop())
                
                self.is_running = True
                self.logger.info(f"A2A Gateway started on port {self.config.listen_port}")
                
        except Exception as e:
            raise A2AProtocolError(
                "Failed to start A2A gateway",
                context={"error": str(e)}
            )
    
    async def stop(self):
        """Stop the A2A gateway"""
        try:
            with LogOperation("a2a_gateway_stop", self.logger):
                self.is_running = False
                
                # Close WebSocket connections
                for ws in self.websocket_connections.copy():
                    await ws.close()
                
                # Cleanup message queue
                await self.message_queue.cleanup()
                
                self.logger.info("A2A Gateway stopped")
                
        except Exception as e:
            self.logger.error(f"Error stopping A2A gateway: {str(e)}")
    
    @weave.op()
    async def send_message(self, message: A2AMessage) -> A2AResponse:
        """Send A2A message"""
        start_time = time.time()
        
        try:
            with LogOperation("a2a_send_message", self.logger,
                             message_id=message.message_id,
                             from_agent=message.from_agent,
                             to_agent=message.to_agent,
                             message_type=message.message_type):
                
                # Validate message
                await self._validate_message(message)
                
                # Apply routing rules
                message = self.router.route_message(message)
                
                # Sign message if required
                if message.security_level in [A2ASecurityLevel.SIGNED, A2ASecurityLevel.FULL]:
                    message.signature = self.crypto_manager.sign_message(message)
                
                # Encrypt message if required
                encrypted_data = None
                if message.security_level in [A2ASecurityLevel.ENCRYPTED, A2ASecurityLevel.FULL]:
                    encrypted_data = self.crypto_manager.encrypt_message(message)
                
                # Publish to message queue
                success = await self.message_queue.publish_message(message)
                
                if not success:
                    raise A2AMessageError("Failed to publish message to queue")
                
                # Update statistics
                processing_time = time.time() - start_time
                self.message_stats.update_stats(True, processing_time, message.message_type)
                
                # Create response
                response = A2AResponse(
                    request_message_id=message.message_id,
                    from_agent=self.config.agent_id,
                    to_agent=message.from_agent,
                    status="queued",
                    success=True,
                    processing_time=processing_time,
                    samsaek_context={
                        "platform": "samsaek",
                        "gateway_version": self.config.samsaek_version,
                        "encrypted": encrypted_data is not None
                    }
                )
                
                return response
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.message_stats.update_stats(False, processing_time, message.message_type)
            
            self.log_error("send_message", e,
                          message_id=message.message_id,
                          from_agent=message.from_agent,
                          to_agent=message.to_agent)
            
            raise A2AMessageError(
                "Failed to send message",
                context={
                    "message_id": message.message_id,
                    "from_agent": message.from_agent,
                    "to_agent": message.to_agent,
                    "error": str(e)
                }
            )
    
    async def register_agent(self, agent: A2AAgent):
        """Register an agent with the gateway"""
        try:
            self.agent_registry[agent.agent_id] = agent
            self.logger.info(f"Agent registered: {agent.agent_id}")
            
            # Store in Redis for persistence
            if self.message_queue.redis_client:
                await self.message_queue.redis_client.setex(
                    f"a2a:agent:{agent.agent_id}",
                    3600,  # 1 hour TTL
                    json.dumps(agent.to_dict())
                )
            
        except Exception as e:
            raise A2AProtocolError(
                f"Failed to register agent {agent.agent_id}",
                context={"agent_id": agent.agent_id, "error": str(e)}
            )
    
    async def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        try:
            if agent_id in self.agent_registry:
                del self.agent_registry[agent_id]
                self.logger.info(f"Agent unregistered: {agent_id}")
                
                # Remove from Redis
                if self.message_queue.redis_client:
                    await self.message_queue.redis_client.delete(f"a2a:agent:{agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to unregister agent {agent_id}: {str(e)}")
    
    async def get_agent(self, agent_id: str) -> Optional[A2AAgent]:
        """Get agent information"""
        if agent_id in self.agent_registry:
            return self.agent_registry[agent_id]
        
        # Try to load from Redis
        if self.message_queue.redis_client:
            agent_data = await self.message_queue.redis_client.get(f"a2a:agent:{agent_id}")
            if agent_data:
                agent_dict = json.loads(agent_data)
                return A2AAgent.parse_obj(agent_dict)
        
        return None
    
    async def list_agents(self) -> List[A2AAgent]:
        """List all registered agents"""
        return list(self.agent_registry.values())
    
    async def _validate_message(self, message: A2AMessage):
        """Validate A2A message"""
        # Check if message has expired
        if message.is_expired():
            raise A2AMessageError(f"Message {message.message_id} has expired")
        
        # Check if recipient agent exists
        recipient = await self.get_agent(message.to_agent)
        if not recipient:
            raise A2ARoutingError(f"Recipient agent {message.to_agent} not found")
        
        # Check if recipient is available
        if not recipient.is_available():
            raise A2ARoutingError(f"Recipient agent {message.to_agent} is not available")
    
    async def _message_processing_loop(self):
        """Main message processing loop"""
        while self.is_running:
            try:
                # Process messages for this agent
                messages = await self.message_queue.consume_messages(self.config.agent_id)
                
                for message in messages:
                    try:
                        await self._process_message(message)
                        await self.message_queue.acknowledge_message(message.message_id, self.config.agent_id)
                    except Exception as e:
                        self.logger.error(f"Error processing message {message.message_id}: {str(e)}")
                
                # Sleep briefly to avoid busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in message processing loop: {str(e)}")
                await asyncio.sleep(1)
    
    async def _process_message(self, message: A2AMessage):
        """Process incoming message"""
        try:
            # Verify signature if present
            if message.signature:
                is_valid = self.crypto_manager.verify_signature(message, message.signature)
                if not is_valid:
                    raise A2AMessageError(f"Invalid signature for message {message.message_id}")
            
            # Handle message based on type
            handler = self.message_handlers.get(message.message_type)
            if handler:
                await handler(message)
            else:
                self.logger.warning(f"No handler for message type: {message.message_type}")
            
        except Exception as e:
            self.logger.error(f"Error processing message {message.message_id}: {str(e)}")
            raise
    
    async def _handle_request(self, message: A2AMessage):
        """Handle request message"""
        self.logger.info(f"Handling request: {message.message_id}")
        # Custom request handling logic would go here
    
    async def _handle_response(self, message: A2AMessage):
        """Handle response message"""
        self.logger.info(f"Handling response: {message.message_id}")
        # Custom response handling logic would go here
    
    async def _handle_notification(self, message: A2AMessage):
        """Handle notification message"""
        self.logger.info(f"Handling notification: {message.message_id}")
        # Custom notification handling logic would go here
    
    async def _handle_heartbeat(self, message: A2AMessage):
        """Handle heartbeat message"""
        self.logger.debug(f"Handling heartbeat: {message.message_id}")
        # Update agent's last seen timestamp
        if message.from_agent in self.agent_registry:
            self.agent_registry[message.from_agent].last_seen = datetime.now()
    
    async def _handle_error(self, message: A2AMessage):
        """Handle error message"""
        self.logger.error(f"Handling error message: {message.message_id}")
        # Custom error handling logic would go here
    
    async def _websocket_handler(self, request):
        """WebSocket handler for real-time communication"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websocket_connections.add(ws)
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        message = A2AMessage.parse_obj(data)
                        response = await self.send_message(message)
                        await ws.send_text(json.dumps(response.to_wire_format()))
                    except Exception as e:
                        error_response = {
                            "error": True,
                            "message": str(e),
                            "timestamp": datetime.now().isoformat()
                        }
                        await ws.send_text(json.dumps(error_response))
                elif msg.type == WSMsgType.ERROR:
                    self.logger.error(f"WebSocket error: {ws.exception()}")
                    break
        finally:
            self.websocket_connections.discard(ws)
        
        return ws
    
    async def _http_message_handler(self, request):
        """HTTP handler for message sending"""
        try:
            data = await request.json()
            message = A2AMessage.parse_obj(data)
            response = await self.send_message(message)
            return web.json_response(response.to_wire_format())
        except Exception as e:
            return web.json_response({
                "error": True,
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }, status=400)
    
    async def _get_agents_handler(self, request):
        """HTTP handler for listing agents"""
        try:
            agents = await self.list_agents()
            return web.json_response([agent.to_dict() for agent in agents])
        except Exception as e:
            return web.json_response({
                "error": True,
                "message": str(e)
            }, status=500)
    
    async def _health_check_handler(self, request):
        """HTTP handler for health check"""
        return web.json_response({
            "status": "healthy",
            "agent_id": self.config.agent_id,
            "registered_agents": len(self.agent_registry),
            "websocket_connections": len(self.websocket_connections),
            "message_stats": self.message_stats.dict(),
            "timestamp": datetime.now().isoformat()
        })
    
    async def _metrics_handler(self, request):
        """HTTP handler for metrics"""
        return web.json_response({
            "message_stats": self.message_stats.dict(),
            "agent_count": len(self.agent_registry),
            "connection_count": len(self.websocket_connections),
            "uptime": (datetime.now() - self.message_stats.start_time).total_seconds(),
            "timestamp": datetime.now().isoformat()
        })