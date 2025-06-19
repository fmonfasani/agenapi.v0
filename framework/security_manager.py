# framework/security_manager.py

import logging
from typing import Dict, List, Optional, Set
from datetime import datetime

from agentapi.models.security_models import User, AgentAuthenticationEntry, SecurityToken, SecurityLevel, Permission, UserRole, AgentRole, AuthenticationMethod

class SecurityManager:
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.agent_auth_entries: Dict[str, AgentAuthenticationEntry] = {}
        self.roles: Dict[str, UserRole | AgentRole] = {} # Use Union for clarity
        self.logger = logging.getLogger("SecurityManager")
        self._initialize_default_roles()
        # Potentially: self._active_sessions: Dict[str, SecurityToken] = {} # if managing active sessions

    def _initialize_default_roles(self):
        self.roles['admin'] = UserRole(name='admin', permissions=set(Permission))
        self.roles['developer'] = UserRole(name='developer', permissions={
            Permission.READ_AGENTS, Permission.WRITE_AGENTS,
            Permission.READ_RESOURCES, Permission.WRITE_RESOURCES,
            Permission.SEND_MESSAGES, Permission.EXECUTE_ACTIONS,
            Permission.MONITOR_SYSTEM
        })
        self.roles['basic_agent'] = AgentRole(name='basic_agent', permissions={
            Permission.SEND_MESSAGES, Permission.READ_RESOURCES, Permission.WRITE_RESOURCES
        })
        self.logger.info("Default roles initialized.")

    async def authenticate_user(self, username, password) -> Optional[User]:
        user = next((u for u in self.users.values() if u.username == username), None)
        if user and user.hashed_password == password:
            user.last_login = datetime.now()
            # If managing active sessions, you'd create and store a token here
            self.logger.info(f"User '{username}' authenticated successfully.")
            return user
        self.logger.warning(f"Authentication failed for user '{username}'.")
        return None

    async def authorize_action(self, subject_id: str, required_permission: Permission) -> bool:
        if subject_id in self.users:
            user = self.users[subject_id]
            for role_name in user.roles:
                role = self.roles.get(role_name)
                if role and role.has_permission(required_permission):
                    self.logger.debug(f"User '{subject_id}' authorized for {required_permission.value}.")
                    return True
        elif subject_id in self.agent_auth_entries:
            agent_entry = self.agent_auth_entries[subject_id]
            if agent_entry.security_level == SecurityLevel.RESTRICTED or required_permission in agent_entry.permissions:
                self.logger.debug(f"Agent '{subject_id}' authorized for {required_permission.value}.")
                return True
        
        self.logger.warning(f"Subject '{subject_id}' NOT authorized for {required_permission.value}.")
        return False

    async def create_user(self, user: User):
        if user.id in self.users or user.username in [u.username for u in self.users.values()]:
            raise ValueError("User with this ID or username already exists.")
        self.users[user.id] = user
        self.logger.info(f"User '{user.username}' ({user.id}) created.")

    async def get_user(self, user_id: str) -> Optional[User]:
        return self.users.get(user_id)

    async def add_agent_authentication_entry(self, entry: AgentAuthenticationEntry):
        self.agent_auth_entries[entry.agent_id] = entry
        self.logger.info(f"Auth entry added for agent {entry.agent_id}.")

    async def shutdown(self):
        """Performs cleanup for SecurityManager (e.g., clearing active sessions/caches)."""
        self.logger.info("Shutting down SecurityManager...")
        # Example: clear any in-memory token caches or active sessions
        # if hasattr(self, '_active_sessions'):
        #     self._active_sessions.clear()
        #     self.logger.debug("Cleared active sessions.")
        self.logger.info("SecurityManager shut down.")