import asyncio
import uuid
import json
import logging
from typing import Dict, List, Any, Optional, Type, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime
import weakref
from contextlib import asynccontextmanager
import traceback

from core.models import AgentStatus, MessageType, ResourceType, AgentMessage, AgentResource, AgentCapability
from core.registry import AgentRegistry


class BaseAgent(ABC):
    def __init__(self, namespace: str, name: str, framework: 'AgentFramework'):
        self.id = str(uuid.uuid4())
        self.namespace = namespace
        self.name = name
        self.status = AgentStatus.INITIALIZING
        self.framework: 'AgentFramework' = weakref.proxy(framework)
        self.capabilities: List[AgentCapability] = []
        self.message_queue = asyncio.Queue()
        self.stop_event = asyncio.Event()
        self.task: Optional[asyncio.Task] = None
        self.last_heartbeat = datetime.now()
        self.logger = logging.getLogger(f"{self.namespace}.{self.name}")

    @abstractmethod
    async def initialize(self) -> bool:
        pass

    async def start(self):
        if self.status not in [AgentStatus.ACTIVE, AgentStatus.BUSY]:
            await self.framework.registry.register_agent(self)
            self.status = AgentStatus.ACTIVE
            self.task = asyncio.create_task(self._run())

    async def stop(self):
        if self.status != AgentStatus.TERMINATED:
            self.stop_event.set()
            await self.framework.registry.unregister_agent(self.id)
            self.status = AgentStatus.TERMINATED
            if self.task:
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    pass

    async def _run(self):
        try:
            while not self.stop_event.is_set():
                self.last_heartbeat = datetime.now()
                self.framework.registry.update_agent_status(self.id, AgentStatus.ACTIVE)
                try:
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                    await self.process_message(message)
                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    self.logger.error(f"Error in _run loop: {e}", exc_info=True)
                    self.status = AgentStatus.ERROR
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
        finally:
            self.status = AgentStatus.TERMINATED

    async def process_message(self, message: AgentMessage):
        self.status = AgentStatus.BUSY
        try:
            if message.message_type == MessageType.COMMAND or message.message_type == MessageType.REQUEST:
                action = message.payload.get("action")
                params = message.payload.get("params", {})
                response_payload = {"status": "failed", "error": "Action not found or not handled."}
                
                found_capability = False
                for capability in self.capabilities:
                    if capability.name == action or capability.namespace == action:
                        response_payload = await capability.handler(params)
                        found_capability = True
                        break
                
                if not found_capability:
                    response_payload = {"status": "error", "message": f"Unknown action: {action}"}

                if message.message_type == MessageType.REQUEST:
                    response_message = AgentMessage(
                        sender_id=self.id,
                        receiver_id=message.sender_id,
                        message_type=MessageType.RESPONSE,
                        payload=response_payload,
                        correlation_id=message.id
                    )
                    await self.framework.message_bus.send_message(response_message)

            elif message.message_type == MessageType.EVENT:
                await self.handle_event(message.payload)
            elif message.message_type == MessageType.RESPONSE:
                await self.handle_response(message.payload, message.correlation_id)
            elif message.message_type == MessageType.HEARTBEAT:
                pass
            elif message.message_type == MessageType.ERROR:
                self.logger.error(f"Received error message from {message.sender_id}: {message.payload.get('error')}")

        except Exception as e:
            self.logger.error(f"Error processing message: {e}", exc_info=True)
            if message.message_type == MessageType.REQUEST:
                error_response = AgentMessage(
                    sender_id=self.id,
                    receiver_id=message.sender_id,
                    message_type=MessageType.ERROR,
                    payload={"error": str(e), "original_message": message.payload},
                    correlation_id=message.id
                )
                await self.framework.message_bus.send_message(error_response)
        finally:
            self.status = AgentStatus.ACTIVE

    async def send_message(self, receiver_id: str, action: str, params: Dict[str, Any], message_type: MessageType = MessageType.REQUEST) -> str:
        message = AgentMessage(
            sender_id=self.id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload={"action": action, "params": params}
        )
        await self.framework.message_bus.send_message(message)
        return message.id

    async def create_agent(self, namespace: str, name: str, agent_class: Type['BaseAgent'], initial_params: Optional[Dict[str, Any]] = None) -> Optional['BaseAgent']:
        try:
            new_agent = await self.framework.agent_factory.create_agent_instance(namespace, name, agent_class, self.framework, initial_params)
            return new_agent
        except Exception as e:
            self.logger.error(f"Failed to create new agent {namespace}.{name}: {e}")
        return None

    async def publish_event(self, event_type: str, data: Dict[str, Any]):
        event_message = AgentMessage(
            sender_id=self.id,
            message_type=MessageType.EVENT,
            receiver_id="all",
            payload={"event_type": event_type, "data": data}
        )
        await self.framework.message_bus.send_message(event_message)

    async def handle_event(self, event_payload: Dict[str, Any]):
        pass

    async def handle_response(self, response_payload: Dict[str, Any], correlation_id: Optional[str]):
        pass

    def __str__(self):
        return f"{self.namespace}.{self.name} (ID: {self.id[:8]}...)"


class MessageBus:
    def __init__(self, framework: 'AgentFramework'):
        self.framework = weakref.proxy(framework)
        self.message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self.logger = logging.getLogger("MessageBus")
        self._listener_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self):
        if not self._listener_task:
            self._stop_event.clear()
            self._listener_task = asyncio.create_task(self._listen_for_messages())

    async def stop(self):
        if self._listener_task:
            self._stop_event.set()
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None

    async def send_message(self, message: AgentMessage) -> bool:
        try:
            await self.message_queue.put(message)
            return True
        except Exception as e:
            self.logger.error(f"Failed to queue message {message.id}: {e}")
            return False

    async def _listen_for_messages(self):
        while not self._stop_event.is_set():
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                if message.receiver_id == "all":
                    for agent in self.framework.registry.list_all_agents():
                        if agent.id != message.sender_id:
                            await agent.message_queue.put(message)
                else:
                    receiver_agent = self.framework.registry.get_agent(message.receiver_id)
                    if receiver_agent:
                        await receiver_agent.message_queue.put(message)
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                self.logger.error(f"Error processing message in MessageBus: {e}", exc_info=True)
            await asyncio.sleep(0.05)


class ResourceManager:
    def __init__(self):
        self.resources: Dict[str, AgentResource] = {}
        self.logger = logging.getLogger("ResourceManager")

    async def create_resource(self, resource: AgentResource) -> bool:
        if resource.id in self.resources:
            return False
        self.resources[resource.id] = resource
        return True

    async def get_resource(self, resource_id: str) -> Optional[AgentResource]:
        return self.resources.get(resource_id)

    async def update_resource(self, resource_id: str, new_data: Dict[str, Any]) -> bool:
        resource = self.resources.get(resource_id)
        if resource:
            if isinstance(resource.data, dict) and isinstance(new_data, dict):
                resource.data.update(new_data)
            else:
                resource.data = new_data
            resource.last_modified = datetime.now()
            return True
        return False

    async def delete_resource(self, resource_id: str) -> bool:
        if resource_id in self.resources:
            del self.resources[resource_id]
            return True
        return False

    def find_resources_by_type(self, resource_type: ResourceType) -> List[AgentResource]:
        return [r for r in self.resources.values() if r.type == resource_type]

    def find_resources_by_owner(self, owner_agent_id: str) -> List[AgentResource]:
        return [r for r in self.resources.values() if r.owner_agent_id == owner_agent_id]

    def list_all_resources(self) -> List[AgentResource]:
        return list(self.resources.values())


class AgentFactory:
    def __init__(self, framework: 'AgentFramework'):
        self.framework = weakref.proxy(framework)
        self.logger = logging.getLogger("AgentFactory")

    async def create_agent_instance(self, namespace: str, name: str, agent_class: Type[BaseAgent], framework: 'AgentFramework', initial_params: Optional[Dict[str, Any]] = None) -> Optional[BaseAgent]:
        try:
            agent = agent_class(name=name, framework=framework)
            agent.namespace = namespace
            
            if initial_params:
                for key, value in initial_params.items():
                    if hasattr(agent, key):
                        setattr(agent, key, value)

            if await agent.initialize():
                await agent.start()
                return agent
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error creating agent {namespace}.{name}: {e}", exc_info=True)
            return None

    @classmethod
    async def create_agent_ecosystem(cls, framework: 'AgentFramework') -> Dict[str, BaseAgent]:
        agents: Dict[str, BaseAgent] = {}
        
        try:
            from agents.specialized_agents import StrategistAgent, WorkflowDesignerAgent, CodeGeneratorAgent, TestGeneratorAgent, BuildAgent
        except ImportError:
            logging.error("Could not import specialized agents.")
            return agents

        factory = AgentFactory(framework)

        strategist_agent = await factory.create_agent_instance(
            "agent.planning.strategist", "strategist", StrategistAgent, framework
        )
        if strategist_agent:
            agents['strategist'] = strategist_agent

        workflow_designer_agent = await factory.create_agent_instance(
            "agent.planning.workflow_designer", "workflow_designer", WorkflowDesignerAgent, framework
        )
        if workflow_designer_agent:
            agents['workflow_designer'] = workflow_designer_agent

        code_generator_agent = await factory.create_agent_instance(
            "agent.build.code.generator", "code_generator", CodeGeneratorAgent, framework
        )
        if code_generator_agent:
            agents['code_generator'] = code_generator_agent
            
        build_agent_instance = await factory.create_agent_instance(
            "agent.build.builder", "builder", BuildAgent, framework
        )
        if build_agent_instance:
            agents['build_agent'] = build_agent_instance

        test_generator_agent = await factory.create_agent_instance(
            "agent.test.generator", "test_generator", TestGeneratorAgent, framework
        )
        if test_generator_agent:
            agents['test_generator'] = test_generator_agent

        return {k: v for k, v in agents.items() if v is not None}


class AgentFramework:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger("AgentFramework")
        self.config = config or {}
        self.registry = AgentRegistry()
        self.message_bus = MessageBus(self)
        self.resource_manager = ResourceManager()
        self.agent_factory = AgentFactory(self)
        self.is_running = False
        self.health_check_task: Optional[asyncio.Task] = None

    async def start(self):
        if self.is_running:
            return

        await self.message_bus.start()
        self.health_check_task = asyncio.create_task(self._run_health_checks())
        self.is_running = True

    async def stop(self):
        if not self.is_running:
            return

        agents_to_stop = list(self.registry.list_all_agents())
        for agent in agents_to_stop:
            try:
                await agent.stop()
            except Exception as e:
                self.logger.error(f"Error stopping agent {agent.name}: {e}")

        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        await self.message_bus.stop()
        self.is_running = False

    async def _run_health_checks(self):
        try:
            while self.is_running:
                for agent in self.registry.list_all_agents():
                    if (datetime.now() - agent.last_heartbeat).total_seconds() > 60:
                        if agent.status != AgentStatus.ERROR and agent.status != AgentStatus.TERMINATED:
                            self.registry.update_agent_status(agent.id, AgentStatus.ERROR)
                await asyncio.sleep(30)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in framework health check: {e}", exc_info=True)


async def example_usage():
    framework = AgentFramework()
    try:
        await framework.start()
        agents = await AgentFactory.create_agent_ecosystem(framework)
        
        strategist = agents.get('strategist')
        workflow_designer = agents.get('workflow_designer')
        
        if strategist and workflow_designer:
            message_id = await strategist.send_message(
                workflow_designer.id,
                "action.create.workflow",
                {
                    "tasks": ["analyze_requirements", "design_architecture", "implement_features"],
                    "priority": "high"
                }
            )
            
            await asyncio.sleep(2)
            
            from agents.specialized_agents import BuildAgent
            new_agent = await strategist.create_agent(
                "agent.test",
                "unit_tester",
                BuildAgent
            )
            
            all_agents = framework.registry.list_all_agents()
            for agent in all_agents:
                print(f"  - {agent.namespace}.{agent.name} ({agent.id}) - {agent.status.value}")
            
    finally:
        await framework.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())