# framework/registry.py

import logging
from typing import Dict, List, Optional
from datetime import datetime

from agentapi.models.agent_models import AgentInfo, AgentStatus, AgentCapability, AgentResource

class AgentRegistry:
    def __init__(self):
        self._agents: Dict[str, AgentInfo] = {}
        self._capabilities: Dict[str, List[AgentCapability]] = {}
        self._resources: Dict[str, AgentResource] = {}
        self.logger = logging.getLogger("AgentRegistry")

    async def register_agent(self, agent_info: AgentInfo):
        self._agents[agent_info.id] = agent_info
        self.logger.info(f"Agent {agent_info.name} ({agent_info.id}) registered.")

    async def unregister_agent(self, agent_id: str):
        if agent_id in self._agents:
            del self._agents[agent_id]
            self.logger.info(f"Agent {agent_id} unregistered.")
        if agent_id in self._capabilities:
            del self._capabilities[agent_id]

    async def update_agent_status(self, agent_id: str, status: AgentStatus, last_heartbeat: datetime):
        if agent_id in self._agents:
            self._agents[agent_id].status = status
            self._agents[agent_id].last_heartbeat = last_heartbeat

    async def get_agent_info(self, agent_id: str) -> Optional[AgentInfo]:
        return self._agents.get(agent_id)

    async def list_all_agents(self) -> List[AgentInfo]:
        return list(self._agents.values())

    async def register_agent_capability(self, agent_id: str, capability: AgentCapability):
        if agent_id not in self._capabilities:
            self._capabilities[agent_id] = []
        self._capabilities[agent_id].append(capability)
        self.logger.info(f"Capability '{capability.name}' registered for agent {agent_id}.")

    async def get_agent_capabilities(self, agent_id: str) -> List[AgentCapability]:
        return self._capabilities.get(agent_id, [])
    
    async def find_agents_by_capability(self, capability_name: str) -> List[AgentInfo]:
        found_agents = []
        for agent_id, capabilities in self._capabilities.items():
            if any(cap.name == capability_name for cap in capabilities):
                agent_info = await self.get_agent_info(agent_id)
                if agent_info and agent_info.status == AgentStatus.ACTIVE:
                    found_agents.append(agent_info)
        return found_agents

    async def add_resource(self, resource: AgentResource):
        self._resources[resource.id] = resource
        self.logger.info(f"Resource {resource.name} ({resource.id}) added to registry.")

    async def get_resource(self, resource_id: str) -> Optional[AgentResource]:
        return self._resources.get(resource_id)
    
    async def get_resource_by_name(self, name: str, namespace: str) -> Optional[AgentResource]:
        for res in self._resources.values():
            if res.name == name and res.namespace == namespace:
                return res
        return None

    async def update_resource(self, resource: AgentResource):
        if resource.id in self._resources:
            self._resources[resource.id] = resource
            self.logger.info(f"Resource {resource.name} ({resource.id}) updated in registry.")

    async def delete_resource(self, resource_id: str):
        if resource_id in self._resources:
            del self._resources[resource_id]
            self.logger.info(f"Resource {resource_id} deleted from registry.")