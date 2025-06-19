# framework/resource_manager.py

import logging
from typing import Optional

from agentapi.models.agent_models import AgentResource
from framework.registry import AgentRegistry

class ResourceManager:
    def __init__(self, registry: AgentRegistry):
        self._registry = registry
        self.logger = logging.getLogger("ResourceManager")

    async def add_resource(self, resource: AgentResource):
        await self._registry.add_resource(resource)
        self.logger.info(f"Resource '{resource.name}' ({resource.id}) managed.")

    async def get_resource(self, resource_id: str) -> Optional[AgentResource]:
        return await self._registry.get_resource(resource_id)

    async def get_resource_by_name(self, name: str, namespace: str) -> Optional[AgentResource]:
        return await self._registry.get_resource_by_name(name, namespace)

    async def update_resource(self, resource: AgentResource):
        await self._registry.update_resource(resource)
        self.logger.info(f"Resource '{resource.name}' ({resource.id}) updated.")

    async def delete_resource(self, resource_id: str):
        await self._registry.delete_resource(resource_id)
        self.logger.info(f"Resource '{resource_id}' deleted.")