# framework/message_bus.py

import asyncio
import logging
from typing import Dict

from agentapi.models.agent_models import AgentMessage
from framework.registry import AgentRegistry # Import AgentRegistry

class MessageBus:
    def __init__(self, registry: AgentRegistry):
        self._registry = registry
        self._message_queues: Dict[str, asyncio.Queue] = {}
        self.logger = logging.getLogger("MessageBus")

    async def register_agent_queue(self, agent_id: str, queue: asyncio.Queue):
        self._message_queues[agent_id] = queue
        self.logger.info(f"Message queue registered for agent {agent_id}.")

    async def unregister_agent_queue(self, agent_id: str):
        if agent_id in self._message_queues:
            del self._message_queues[agent_id]
            self.logger.info(f"Message queue unregistered for agent {agent_id}.")

    async def send_message(self, message: AgentMessage):
        if message.receiver_id == "broadcast":
            for agent_id in self._message_queues.keys():
                try:
                    await self._message_queues[agent_id].put(message)
                    self.logger.debug(f"Broadcast message {message.id} sent to {agent_id}.")
                except Exception as e:
                    self.logger.error(f"Failed to send broadcast message to {agent_id}: {e}")
        elif message.receiver_id in self._message_queues:
            try:
                await self._message_queues[message.receiver_id].put(message)
                self.logger.debug(f"Message {message.id} sent to {message.receiver_id}.")
            except Exception as e:
                self.logger.error(f"Failed to send message {message.id} to {message.receiver_id}: {e}")
        else:
            self.logger.warning(f"Receiver {message.receiver_id} not found for message {message.id}.")