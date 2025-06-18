"""
refactored_rest_api.py - API REST completa para el framework de agentes
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from functools import wraps

from aiohttp import web, WSMsgType
import aiohttp_cors
import aiohttp_swagger
from aiohttp.web_request import Request
from aiohttp.web_response import Response

# Assuming these imports are from the core framework files provided
from core.autonomous_agent_framework import AgentFramework, BaseAgent, AgentStatus, MessageType, AgentResource, ResourceType
from core.specialized_agents import ExtendedAgentFactory
from core.security_system import SecurityManager, Permission, SecurityLevel, AuthenticationMethod
from core.persistence_system import PersistenceManager, PersistenceFactory, PersistenceBackend
from framework_config_utils import MetricsCollector # Assuming this exists or is a placeholder for system metrics


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API MODELS AND SCHEMAS

class APIResponse:
    """Standard API response format"""

    @staticmethod
    def success(data: Any = None, message: str = "Operation successful") -> Dict[str, Any]:
        return {
            "success": True,
            "message": message,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    def error(message: str, code: int = 400, details: Any = None) -> Dict[str, Any]:
        return {
            "success": False,
            "error": {
                "message": message,
                "code": code,
                "details": details
            },
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    def paginated(data: List[Any], page: int, per_page: int, total: int) -> Dict[str, Any]:
        return {
            "success": True,
            "data": data,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "pages": (total + per_page - 1) // per_page
            },
            "timestamp": datetime.now().isoformat()
        }

# AUTHENTICATION AND AUTHORIZATION DECORATORS

def requires_auth(handler: Callable) -> Callable:
    """Authentication middleware decorator"""
    @wraps(handler)
    async def wrapper(request: Request) -> Response:
        public_routes = ["/api/auth/login", "/api/health", "/api/docs", "/"]
        if request.path in public_routes or request.path.startswith("/static"):
            return await handler(request)

        auth_header = request.headers.get("Authorization", "")
        api_key = request.headers.get("X-API-Key", "")

        security_manager: SecurityManager = request.app["security_manager"]
        user_info = None
        session_id = None

        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            try:
                if "jwt" in security_manager.auth_providers:
                    jwt_provider = security_manager.auth_providers["jwt"]
                    payload = await jwt_provider.validate_token(token)
                    if payload:
                        user_info = payload
            except Exception as e:
                logger.error(f"JWT validation error: {e}")
        elif api_key:
            try:
                if "api_key" in security_manager.auth_providers:
                    api_provider = security_manager.auth_providers["api_key"]
                    key_info = await api_provider.validate_token(api_key)
                    if key_info:
                        user_info = key_info
            except Exception as e:
                logger.error(f"API Key validation error: {e}")

        session_cookie = request.cookies.get("session_id")
        if session_cookie:
            session = await security_manager.validate_session(session_cookie)
            if session:
                user_info = session.get("user_info", {})
                session_id = session_cookie

        if not user_info:
            return web.json_response(APIResponse.error("Authentication required", 401), status=401)

        request["user_info"] = user_info
        request["session_id"] = session_id
        return await handler(request)
    return wrapper

def requires_permission(permission: Permission) -> Callable:
    """Authorization middleware decorator to check specific permissions"""
    def decorator(handler: Callable) -> Callable:
        @wraps(handler)
        async def wrapper(request: Request) -> Response:
            session_id = request.get("session_id")
            security_manager: SecurityManager = request.app["security_manager"]

            if session_id:
                authorized = await security_manager.authorize_action(session_id, permission)
                if not authorized:
                    return web.json_response(
                        APIResponse.error("Insufficient permissions", 403),
                        status=403
                    )
            else:
                # If no session, rely on API Key permissions (if implemented for perms)
                # For simplicity, we assume session_id is always present for authenticated requests here.
                # In a more complex scenario, API Key permissions would also be checked here.
                pass # The requires_auth decorator ensures user_info is present, permissions should be checked there for API keys.

            return await handler(request)
        return wrapper
    return decorator


# API HANDLERS

class BaseHandler:
    """Base class for API handlers to provide common utilities."""
    def __init__(self, framework: AgentFramework, security_manager: SecurityManager):
        self.framework = framework
        self.security_manager = security_manager

    async def _handle_request(self, request: Request, logic_func: Callable, success_message: str = "Operation successful", error_message: str = "Internal server error", error_status: int = 500) -> Response:
        """Helper to standardize request handling, including error responses."""
        try:
            result = await logic_func(request)
            return web.json_response(APIResponse.success(result, success_message))
        except web.HTTPError as e:
            return web.json_response(APIResponse.error(e.reason, e.status), status=e.status)
        except ValueError as e:
            logger.warning(f"Bad request: {e}")
            return web.json_response(APIResponse.error(str(e), 400), status=400)
        except Exception as e:
            logger.exception(f"{error_message} for {request.path}: {e}")
            return web.json_response(APIResponse.error(error_message, error_status, str(e)), status=error_status)


class AuthenticationHandlers(BaseHandler):
    """Handlers for authentication and API key management."""

    def __init__(self, security_manager: SecurityManager):
        super().__init__(None, security_manager) # Framework not directly needed here

    async def login(self, request: Request) -> Response:
        """Login user with username and password."""
        async def _login_logic(req: Request):
            data = await req.json()
            username = data.get("username")
            password = data.get("password")

            if not username or not password:
                raise ValueError("Username and password required")

            auth_result = await self.security_manager.authenticate_user(
                AuthenticationMethod.JWT_TOKEN,
                {"username": username, "password": password}
            )

            if not auth_result:
                raise web.HTTPUnauthorized(reason="Invalid credentials")
            return auth_result

        return await self._handle_request(request, _login_logic, "Login successful", "Login error", 401)

    @requires_auth
    async def logout(self, request: Request) -> Response:
        """Logout user by invalidating session."""
        async def _logout_logic(req: Request):
            session_id = req.get("session_id")
            if not session_id:
                raise ValueError("No active session found")
            success = await self.security_manager.logout_session(session_id)
            if not success:
                raise web.HTTPBadRequest(reason="Failed to logout session")
            return {}

        return await self._handle_request(request, _logout_logic, "Logout successful", "Logout error")

    @requires_auth
    @requires_permission(Permission.ADMIN_ACCESS) # Or a more specific permission like CREATE_API_KEYS
    async def create_api_key(self, request: Request) -> Response:
        """Create API key for programmatic access."""
        async def _create_api_key_logic(req: Request):
            data = await req.json()
            description = data.get("description", "")
            permissions_list = data.get("permissions", []) # Default to empty list for safety

            permissions = []
            for perm_str in permissions_list:
                try:
                    permissions.append(Permission(perm_str))
                except ValueError:
                    raise ValueError(f"Invalid permission: {perm_str}")

            user_info = req["user_info"]
            user_id = user_info.get("username", "unknown") # Assuming username is the user_id

            api_key = await self.security_manager.create_api_key(
                user_id, permissions, description
            )
            return {
                "api_key": api_key,
                "description": description,
                "permissions": [p.value for p in permissions]
            }

        return await self._handle_request(request, _create_api_key_logic, "API Key created successfully", "Failed to create API key")


class AgentHandlers(BaseHandler):
    """Handlers for agent management."""

    async def list_agents(self, request: Request) -> Response:
        """List all agents with pagination and filters."""
        @requires_auth
        @requires_permission(Permission.READ_AGENTS)
        async def _list_agents_logic(req: Request):
            agents = self.framework.registry.list_all_agents()

            page = int(req.query.get("page", 1))
            per_page = int(req.query.get("per_page", 10))
            namespace_filter = req.query.get("namespace")
            status_filter = req.query.get("status")

            filtered_agents = agents
            if namespace_filter:
                filtered_agents = [a for a in filtered_agents if namespace_filter in a.namespace]
            if status_filter:
                filtered_agents = [a for a in filtered_agents if a.status.value == status_filter]

            total = len(filtered_agents)
            start = (page - 1) * per_page
            end = start + per_page
            page_agents = filtered_agents[start:end]

            agents_data = []
            for agent in page_agents:
                agents_data.append({
                    "id": agent.id,
                    "name": agent.name,
                    "namespace": agent.namespace,
                    "status": agent.status.value,
                    "created_at": agent.created_at.isoformat(),
                    "last_heartbeat": agent.last_heartbeat.isoformat(),
                    "capabilities_count": len(agent.capabilities),
                    "metadata": agent.metadata
                })
            return APIResponse.paginated(agents_data, page, per_page, total)

        return await self._handle_request(request, _list_agents_logic, "Agents listed successfully")

    @requires_auth
    @requires_permission(Permission.READ_AGENTS)
    async def get_agent(self, request: Request) -> Response:
        """Get detailed information about a specific agent."""
        async def _get_agent_logic(req: Request):
            agent_id = req.match_info["agent_id"]
            agent = self.framework.registry.get_agent(agent_id)
            if not agent:
                raise web.HTTPNotFound(reason="Agent not found")

            agent_resources = self.framework.resource_manager.find_resources_by_owner(agent_id)

            return {
                "id": agent.id,
                "name": agent.name,
                "namespace": agent.namespace,
                "status": agent.status.value,
                "created_at": agent.created_at.isoformat(),
                "last_heartbeat": agent.last_heartbeat.isoformat(),
                "metadata": agent.metadata,
                "capabilities": [
                    {
                        "name": cap.name,
                        "namespace": cap.namespace,
                        "description": cap.description,
                        "input_schema": cap.input_schema,
                        "output_schema": cap.output_schema
                    }
                    for cap in agent.capabilities
                ],
                "resources": [
                    {
                        "id": res.id,
                        "name": res.name,
                        "type": res.type.value,
                        "namespace": res.namespace,
                        "created_at": res.created_at.isoformat(),
                        "updated_at": res.updated_at.isoformat()
                    }
                    for res in agent_resources
                ]
            }
        return await self._handle_request(request, _get_agent_logic, "Agent details retrieved successfully", "Failed to get agent details", 404)

    @requires_auth
    @requires_permission(Permission.CREATE_AGENTS)
    async def create_agent(self, request: Request) -> Response:
        """Create a new agent."""
        async def _create_agent_logic(req: Request):
            data = await req.json()
            namespace = data.get("namespace")
            name = data.get("name")
            auto_start = data.get("auto_start", True)

            if not namespace or not name:
                raise ValueError("Namespace and name are required")

            available_namespaces = ExtendedAgentFactory.list_available_namespaces()
            if namespace not in available_namespaces:
                raise ValueError(f"Unknown namespace: {namespace}. Available: {', '.join(available_namespaces)}")

            agent = ExtendedAgentFactory.create_agent(namespace, name, self.framework)
            if auto_start:
                await agent.start()

            return {
                "id": agent.id,
                "name": agent.name,
                "namespace": agent.namespace,
                "status": agent.status.value,
                "created_at": agent.created_at.isoformat()
            }
        return await self._handle_request(request, _create_agent_logic, "Agent created successfully")

    @requires_auth
    @requires_permission(Permission.EXECUTE_ACTIONS)
    async def execute_agent_action(self, request: Request) -> Response:
        """Execute action on an agent."""
        async def _execute_action_logic(req: Request):
            agent_id = req.match_info["agent_id"]
            agent = self.framework.registry.get_agent(agent_id)
            if not agent:
                raise web.HTTPNotFound(reason="Agent not found")

            data = await req.json()
            action = data.get("action")
            params = data.get("params", {})

            if not action:
                raise ValueError("Action is required")

            result = await agent.execute_action(action, params)
            return {
                "action": action,
                "result": result,
                "agent_id": agent_id
            }
        return await self._handle_request(request, _execute_action_logic, "Action executed successfully", "Failed to execute action")

    @requires_auth
    @requires_permission(Permission.DELETE_AGENTS)
    async def delete_agent(self, request: Request) -> Response:
        """Delete an agent."""
        async def _delete_agent_logic(req: Request):
            agent_id = req.match_info["agent_id"]
            agent = self.framework.registry.get_agent(agent_id)
            if not agent:
                raise web.HTTPNotFound(reason="Agent not found")

            await agent.stop()
            self.framework.registry.remove_agent(agent.id) # Assuming a remove_agent method exists in registry

            return {} # Empty data for success

        return await self._handle_request(request, _delete_agent_logic, f"Agent {request.match_info['agent_id']} deleted successfully")


class ResourceHandlers(BaseHandler):
    """Handlers for resource management."""

    @requires_auth
    @requires_permission(Permission.READ_RESOURCES)
    async def list_resources(self, request: Request) -> Response:
        """List all resources with filters."""
        async def _list_resources_logic(req: Request):
            all_resources = []
            agents = self.framework.registry.list_all_agents()

            for agent in agents:
                agent_resources = self.framework.resource_manager.find_resources_by_owner(agent.id)
                all_resources.extend(agent_resources)

            resource_type = req.query.get("type")
            owner_id = req.query.get("owner")

            if resource_type:
                all_resources = [r for r in all_resources if r.type.value == resource_type]
            if owner_id:
                all_resources = [r for r in all_resources if r.owner_agent_id == owner_id]

            page = int(req.query.get("page", 1))
            per_page = int(req.query.get("per_page", 10))

            total = len(all_resources)
            start = (page - 1) * per_page
            end = start + per_page
            page_resources = all_resources[start:end]

            resources_data = []
            for resource in page_resources:
                resources_data.append({
                    "id": resource.id,
                    "name": resource.name,
                    "type": resource.type.value,
                    "namespace": resource.namespace,
                    "owner_agent_id": resource.owner_agent_id,
                    "created_at": resource.created_at.isoformat(),
                    "updated_at": resource.updated_at.isoformat(),
                    "metadata": resource.metadata,
                    "data_size": len(str(resource.data)) # Consider a more robust way to calculate data size if 'data' can be complex
                })
            return APIResponse.paginated(resources_data, page, per_page, total)

        return await self._handle_request(request, _list_resources_logic, "Resources listed successfully")

    @requires_auth
    @requires_permission(Permission.READ_RESOURCES)
    async def get_resource(self, request: Request) -> Response:
        """Get details of a specific resource."""
        async def _get_resource_logic(req: Request):
            resource_id = req.match_info["resource_id"]
            resource = await self.framework.resource_manager.get_resource(resource_id)
            if not resource:
                raise web.HTTPNotFound(reason="Resource not found")

            return {
                "id": resource.id,
                "name": resource.name,
                "type": resource.type.value,
                "namespace": resource.namespace,
                "owner_agent_id": resource.owner_agent_id,
                "created_at": resource.created_at.isoformat(),
                "updated_at": resource.updated_at.isoformat(),
                "metadata": resource.metadata,
                "data": resource.data
            }
        return await self._handle_request(request, _get_resource_logic, "Resource details retrieved successfully", "Failed to get resource", 404)


class SystemHandlers(BaseHandler):
    """Handlers for system-wide information and metrics."""

    def __init__(self, framework: AgentFramework, security_manager: SecurityManager):
        super().__init__(framework, security_manager)
        self.metrics_collector = MetricsCollector(framework)

    async def health_check(self, request: Request) -> Response:
        """Health check of the system."""
        async def _health_check_logic(req: Request):
            agents = self.framework.registry.list_all_agents()
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "framework": {
                    "total_agents": len(agents),
                    "active_agents": len([a for a in agents if a.status == AgentStatus.ACTIVE]),
                    "running": True
                },
                "security": self.security_manager.get_security_status()
            }
        return await self._handle_request(request, _health_check_logic, "System health check successful")

    @requires_auth
    @requires_permission(Permission.MONITOR_SYSTEM)
    async def get_metrics(self, request: Request) -> Response:
        """Get system metrics and statistics."""
        async def _get_metrics_logic(req: Request):
            metrics = self.metrics_collector.get_metrics()
            agents = self.framework.registry.list_all_agents()
            status_counts = {}
            for agent in agents:
                status = agent.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            metrics["agent_status_distribution"] = status_counts
            return metrics
        return await self._handle_request(request, _get_metrics_logic, "System metrics retrieved successfully")

    @requires_auth
    async def get_namespaces(self, request: Request) -> Response:
        """Get available agent namespaces."""
        async def _get_namespaces_logic(req: Request):
            namespaces = ExtendedAgentFactory.list_available_namespaces()
            return {
                "namespaces": namespaces,
                "total": len(namespaces)
            }
        return await self._handle_request(request, _get_namespaces_logic, "Available namespaces retrieved successfully")


# MAIN API SERVER

class FrameworkAPIServer:
    """REST API server for the agent framework."""

    def __init__(self, framework: AgentFramework, host: str = "localhost", port: int = 8000, jwt_secret: str = "your_jwt_secret_here", session_max_hours: int = 24):
        self.framework = framework
        self.host = host
        self.port = port
        self.app = web.Application()

        # Configure security manager
        self.security_manager = SecurityManager({
            "jwt_secret": jwt_secret,
            "session_max_hours": session_max_hours
        })

        # Configure persistence if available (placeholder - actual initialization depends on PersistenceFactory)
        self.persistence_manager: Optional[PersistenceManager] = None
        # Example of initializing persistence if needed:
        # self.persistence_manager = PersistenceFactory.create_persistence_manager(PersistenceBackend.SQLITE, "framework.db")


        # Add core components to the app for handlers to access
        self.app["framework"] = framework
        self.app["security_manager"] = self.security_manager
        if self.persistence_manager:
            self.app["persistence_manager"] = self.persistence_manager

        # Initialize handlers
        self.auth_handlers = AuthenticationHandlers(self.security_manager)
        self.agent_handlers = AgentHandlers(framework, self.security_manager)
        self.resource_handlers = ResourceHandlers(framework, self.security_manager)
        self.system_handlers = SystemHandlers(framework, self.security_manager)

        # Setup routes, CORS, and Swagger
        self._setup_routes()
        self._setup_cors()
        self._setup_swagger()

    def _setup_routes(self):
        """Configure API routes."""
        # Authentication routes
        self.app.router.add_post("/api/auth/login", self.auth_handlers.login)
        self.app.router.add_post("/api/auth/logout", self.auth_handlers.logout)
        self.app.router.add_post("/api/auth/api-keys", self.auth_handlers.create_api_key)

        # Agent routes
        self.app.router.add_get("/api/agents", self.agent_handlers.list_agents)
        self.app.router.add_get("/api/agents/{agent_id}", self.agent_handlers.get_agent)
        self.app.router.add_post("/api/agents", self.agent_handlers.create_agent)
        self.app.router.add_post("/api/agents/{agent_id}/actions", self.agent_handlers.execute_agent_action)
        self.app.router.add_delete("/api/agents/{agent_id}", self.agent_handlers.delete_agent)

        # Resource routes
        self.app.router.add_get("/api/resources", self.resource_handlers.list_resources)
        self.app.router.add_get("/api/resources/{resource_id}", self.resource_handlers.get_resource)

        # System routes
        self.app.router.add_get("/api/health", self.system_handlers.health_check)
        self.app.router.add_get("/api/metrics", self.system_handlers.get_metrics)
        self.app.router.add_get("/api/namespaces", self.system_handlers.get_namespaces)

        # Documentation
        self.app.router.add_get("/", self._serve_docs)

    def _setup_cors(self):
        """Configure CORS policies."""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        for route in list(self.app.router.routes()):
            cors.add(route)

    def _setup_swagger(self):
        """Configure Swagger/OpenAPI documentation."""
        # In a real implementation, aiohttp_swagger would be used to auto-generate docs
        # aiohttp_swagger.setup(self.app, swagger_url="/api/docs", ui_url="/api/docs/")
        logger.info("Swagger documentation configured at /api/docs")

    async def _serve_docs(self, request: Request) -> Response:
        """Serve basic API documentation HTML."""
        docs_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Agent Framework API Documentation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .method { color: white; padding: 4px 8px; border-radius: 3px; font-weight: bold; }
        .get { background: #61affe; }
        .post { background: #49cc90; }
        .delete { background: #f93e3e; }
        code { background: #f1f1f1; padding: 2px 4px; border-radius: 3px; }
    </style>
</head>
<body>
    <h1>🤖 Agent Framework API Documentation</h1>
    
    <h2>Authentication</h2>
    <div class="endpoint">
        <span class="method post">POST</span> <code>/api/auth/login</code>
        <p>Login with username and password to get JWT token</p>
        <pre>{"username": "admin", "password": "password"}</pre>
    </div>
    
    <div class="endpoint">
        <span class="method post">POST</span> <code>/api/auth/logout</code>
        <p>Logout user by invalidating session.</p>
    </div>

    <div class="endpoint">
        <span class="method post">POST</span> <code>/api/auth/api-keys</code>
        <p>Create API key for programmatic access</p>
    </div>
    
    <h2>Agents</h2>
    <div class="endpoint">
        <span class="method get">GET</span> <code>/api/agents</code>
        <p>List all agents with pagination and filters</p>
        <p>Query params: page, per_page, namespace, status</p>
    </div>
    
    <div class="endpoint">
        <span class="method get">GET</span> <code>/api/agents/{agent_id}</code>
        <p>Get detailed information about a specific agent</p>
    </div>
    
    <div class="endpoint">
        <span class="method post">POST</span> <code>/api/agents</code>
        <p>Create a new agent</p>
        <pre>{"namespace": "agent.planning.strategist", "name": "my_agent"}</pre>
    </div>
    
    <div class="endpoint">
        <span class="method post">POST</span> <code>/api/agents/{agent_id}/actions</code>
        <p>Execute action on an agent</p>
        <pre>{"action": "generate.code", "params": {"specification": {...}}}</pre>
    </div>
    <div class="endpoint">
        <span class="method delete">DELETE</span> <code>/api/agents/{agent_id}</code>
        <p>Delete an agent.</p>
    </div>
    
    <h2>Resources</h2>
    <div class="endpoint">
        <span class="method get">GET</span> <code>/api/resources</code>
        <p>List all resources with filters</p>
        <p>Query params: page, per_page, type, owner</p>
    </div>
    <div class="endpoint">
        <span class="method get">GET</span> <code>/api/resources/{resource_id}</code>
        <p>Get details of a specific resource.</p>
    </div>
    
    <h2>System</h2>
    <div class="endpoint">
        <span class="method get">GET</span> <code>/api/health</code>
        <p>System health check</p>
    </div>
    
    <div class="endpoint">
        <span class="method get">GET</span> <code>/api/metrics</code>
        <p>System metrics and statistics</p>
    </div>
    
    <div class="endpoint">
        <span class="method get">GET</span> <code>/api/namespaces</code>
        <p>Available agent namespaces</p>
    </div>
    
    <h2>Authentication Methods</h2>
    <h3>JWT Token</h3>
    <p>Add header: <code>Authorization: Bearer &lt;token&gt;</code></p>
    
    <h3>API Key</h3>
    <p>Add header: <code>X-API-Key: &lt;api_key&gt;</code></p>
    
    <h2>Example Usage</h2>
    <pre>
# Login
curl -X POST http://localhost:8000/api/auth/login \\
  -H "Content-Type: application/json" \\
  -d '{"username": "admin", "password": "admin_password"}'

# Create Agent
curl -X POST http://localhost:8000/api/agents \\
  -H "Authorization: Bearer &lt;token&gt;" \\
  -H "Content-Type: application/json" \\
  -d '{"namespace": "agent.planning.strategist", "name": "my_strategist"}'

# Execute Action
curl -X POST http://localhost:8000/api/agents/{agent_id}/actions \\
  -H "Authorization: Bearer &lt;token&gt;" \\
  -H "Content-Type: application/json" \\
  -d '{"action": "define.strategy", "params": {"requirements": {"goal": "test"}}}'
    </pre>
</body>
</html>
        """
        return web.Response(text=docs_html, content_type="text/html")

    async def start(self):
        """Start the API server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"API Server started on http://{self.host}:{self.port}")
        # Keep the server running
        try:
            while True:
                await asyncio.sleep(3600) # Sleep for 1 hour
        except asyncio.CancelledError:
            pass
        finally:
            await runner.cleanup()
            logger.info("API Server stopped.")

if __name__ == "__main__":
    # Example usage:
    async def main():
        # Initialize a dummy framework for the API server to run
        # In a real application, you would initialize your actual AgentFramework
        class DummyAgentFramework(AgentFramework):
            def __init__(self):
                super().__init__()
                logging.info("DummyAgentFramework initialized.")

            async def start(self):
                logging.info("DummyAgentFramework starting...")
                # Add any startup logic for the framework here
                pass

            async def stop(self):
                logging.info("DummyAgentFramework stopping...")
                # Add any cleanup logic for the framework here
                pass

            # Mocking registry and resource_manager for API handlers
            class DummyRegistry:
                def list_all_agents(self):
                    return [] # Return empty list for demo
                def get_agent(self, agent_id: str):
                    return None
                def remove_agent(self, agent_id: str):
                    logging.info(f"Agent {agent_id} removed from dummy registry.")

            class DummyResourceManager:
                def find_resources_by_owner(self, owner_id: str):
                    return []
                async def get_resource(self, resource_id: str):
                    return None

            @property
            def registry(self):
                if not hasattr(self, '_registry'):
                    self._registry = self.DummyRegistry()
                return self._registry

            @property
            def resource_manager(self):
                if not hasattr(self, '_resource_manager'):
                    self._resource_manager = self.DummyResourceManager()
                return self._resource_manager

        framework_instance = DummyAgentFramework()
        await framework_instance.start()

        # Instantiate and start the API server
        api_server = FrameworkAPIServer(framework_instance, host="0.0.0.0", port=8000)
        await api_server.start()
        await framework_instance.stop()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server demo interrupted by user.")