# framework/agent_factory.py

import logging
from typing import Dict, Optional, Type
import asyncio # For _agent_initialization_wrapper

from agentapi.interfaces.agent_interfaces import BaseAgent
from agentapi.models.agent_models import AgentInfo, AgentStatus # Needed for agent_info registration

class AgentFactory:
    def __init__(self, framework):
        self.framework = framework # Keep a reference to the main framework
        self.logger = logging.getLogger("AgentFactory")

    async def _agent_initialization_wrapper(self, agent: BaseAgent) -> bool:
        """Handles agent initialization and initial registration with framework."""
        try:
            if await agent.initialize():
                agent.status = AgentStatus.ACTIVE
                self.logger.info(f"Agent {agent.id} initialized and started.")
                # Register the agent info in the framework's registry
                agent_info = AgentInfo(
                    id=agent.id,
                    name=agent.name,
                    namespace=agent.namespace,
                    status=agent.status,
                    last_heartbeat=agent.last_heartbeat
                )
                await self.framework.registry.register_agent(agent_info)
                # Register agent's message queue with the message bus
                await self.framework.message_bus.register_agent_queue(agent.id, agent.message_queue)
                return True
            else:
                agent.status = AgentStatus.ERROR
                self.logger.error(f"Agent {agent.id} failed to initialize.")
                return False
        except Exception as e:
            agent.status = AgentStatus.ERROR
            self.logger.error(f"Error during agent {agent.id} initialization: {e}", exc_info=True)
            return False

    async def create_agent(self, namespace: str, name: str, agent_class: Type[BaseAgent], creator_agent_id: Optional[str] = None) -> Optional[BaseAgent]:
        """Creates an agent instance and starts it."""
        try:
            # Here, you might add security checks using self.framework.security_manager
            # E.g., if not await self.framework.security_manager.authorize_action(creator_agent_id, Permission.CREATE_AGENTS):
            #    raise PermissionDeniedError("Cannot create agent.")
            
            self.logger.info(f"Attempting to create agent {namespace}.{name} of type {agent_class.__name__}.")
            
            agent = agent_class(namespace, name, self.framework)
            
            # Start agent background tasks
            agent._message_listener_task = asyncio.create_task(agent._message_listener())
            agent._heartbeat_sender_task = asyncio.create_task(agent._heartbeat_sender())
            
            # Perform initialization and registration in a wrapper to update agent status correctly
            if await self._agent_initialization_wrapper(agent):
                self.framework._active_agents[agent.id] = agent # Add to framework's active agents
                self.logger.info(f"Agent {agent.id} ({agent.name}) created by {creator_agent_id if creator_agent_id else 'system'}.")
                return agent
            else:
                self.logger.error(f"Failed to create and initialize agent {namespace}.{name}.")
                return None
        except Exception as e:
            self.logger.error(f"Failed to create agent {namespace}.{name}: {e}", exc_info=True)
            return None
    
    async def create_agent_ecosystem(self) -> Dict[str, BaseAgent]:
        """Creates a predefined set of agents for the ecosystem."""
        agents: Dict[str, BaseAgent] = {}
        self.logger.info("Creating initial agent ecosystem...")
        
        # Load agents from config
        if self.framework.config.agents:
            for agent_config in self.framework.config.agents:
                if agent_config.enabled:
                    try:
                        # Dynamically import agent class based on namespace and name conventions
                        # Example: 'agent.core.StrategistAgent'
                        module_path = f"agents.{agent_config.namespace.replace('.', '_')}"
                        class_name = f"{agent_config.name.replace('_', '').capitalize()}Agent" # Simple heuristic
                        
                        # Fallback for common agents if specific import fails
                        if agent_config.name == "strategist":
                            from agents.specialized_agents import StrategistAgent as AgentClass
                        elif agent_config.name == "workflow_designer":
                            from agents.specialized_agents import WorkflowDesignerAgent as AgentClass
                        elif agent_config.name == "code_generator":
                            from agents.specialized_agents import CodeGeneratorAgent as AgentClass
                        elif agent_config.name == "build_agent":
                            from agents.specialized_agents import BuildAgent as AgentClass
                        else:
                            # Generic dynamic import (requires specific file/class naming)
                            module = __import__(module_path, fromlist=[class_name])
                            AgentClass = getattr(module, class_name)
                            
                        agent = await self.create_agent(
                            namespace=agent_config.namespace,
                            name=agent_config.name,
                            agent_class=AgentClass,
                            creator_agent_id="framework_init"
                        )
                        if agent:
                            agents[agent_config.name] = agent
                    except ImportError as e:
                        self.logger.error(f"Could not import agent class for {agent_config.namespace}.{agent_config.name}: {e}")
                    except AttributeError as e:
                        self.logger.error(f"Agent class {class_name} not found in module {module_path} for {agent_config.name}: {e}")
                    except Exception as e:
                        self.logger.error(f"Error creating agent {agent_config.name}: {e}", exc_info=True)
        else:
            self.logger.info("No agents defined in framework configuration to auto-create.")

        self.logger.info(f"Created initial agent ecosystem with {len(agents)} agents.")
        return agents