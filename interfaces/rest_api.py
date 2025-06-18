"""
rest_api.py - API REST completa para el framework de agentes
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from aiohttp import web, WSMsgType
import aiohttp_cors
from aiohttp.web_request import Request
from aiohttp.web_response import Response
import aiohttp_swagger

from core.autonomous_agent_framework import AgentFramework, BaseAgent, AgentStatus, MessageType, AgentResource, ResourceType
from core.specialized_agents import ExtendedAgentFactory
from core.security_system import SecurityManager, Permission, SecurityLevel, AuthenticationMethod
from core.persistence_system import PersistenceManager, PersistenceFactory, PersistenceBackend
from systems.framework_config_utils import MetricsCollector

# ================================
# API MODELS AND SCHEMAS
# ================================

class APIResponse:
    """Respuesta estándar de la API"""
    
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

# ================================
# AUTHENTICATION MIDDLEWARE
# ================================

async def auth_middleware(request: Request, handler):
    """Middleware de autenticación"""
    
    # Rutas públicas que no requieren autenticación
    public_routes = ["/api/auth/login", "/api/health", "/api/docs", "/"]
    
    if request.path in public_routes or request.path.startswith("/static"):
        return await handler(request)
        
    # Obtener token de autorización
    auth_header = request.headers.get("Authorization", "")
    api_key = request.headers.get("X-API-Key", "")
    
    security_manager: SecurityManager = request.app["security_manager"]
    
    user_info = None
    session_id = None
    
    # Intentar autenticación con JWT
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        try:
            if "jwt" in security_manager.auth_providers:
                jwt_provider = security_manager.auth_providers["jwt"]
                payload = await jwt_provider.validate_token(token)
                if payload:
                    user_info = payload
        except Exception as e:
            logging.error(f"JWT validation error: {e}")
            
    # Intentar autenticación con API Key
    elif api_key:
        try:
            if "api_key" in security_manager.auth_providers:
                api_provider = security_manager.auth_providers["api_key"]
                key_info = await api_provider.validate_token(api_key)
                if key_info:
                    user_info = key_info
        except Exception as e:
            logging.error(f"API Key validation error: {e}")
            
    # Verificar si hay sesión válida
    session_cookie = request.cookies.get("session_id")
    if session_cookie:
        session = await security_manager.validate_session(session_cookie)
        if session:
            user_info = session.get("user_info", {})
            session_id = session_cookie
            
    if not user_info:
        return web.json_response(
            APIResponse.error("Authentication required", 401),
            status=401
        )
        
    # Añadir información de usuario al request
    request["user_info"] = user_info
    request["session_id"] = session_id
    
    return await handler(request)

# ================================
# API HANDLERS
# ================================

class AuthenticationHandlers:
    """Handlers para autenticación"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        
    async def login(self, request: Request) -> Response:
        """Login de usuario"""
        try:
            data = await request.json()
            username = data.get("username")
            password = data.get("password")
            
            if not username or not password:
                return web.json_response(
                    APIResponse.error("Username and password required"),
                    status=400
                )
                
            # Autenticar
            auth_result = await self.security_manager.authenticate_user(
                AuthenticationMethod.JWT_TOKEN,
                {"username": username, "password": password}
            )
            
            if auth_result:
                return web.json_response(
                    APIResponse.success(auth_result, "Login successful")
                )
            else:
                return web.json_response(
                    APIResponse.error("Invalid credentials", 401),
                    status=401
                )
                
        except Exception as e:
            logging.error(f"Login error: {e}")
            return web.json_response(
                APIResponse.error("Internal server error", 500),
                status=500
            )
            
    async def logout(self, request: Request) -> Response:
        """Logout de usuario"""
        session_id = request.get("session_id")
        
        if session_id:
            success = await self.security_manager.logout_session(session_id)
            if success:
                return web.json_response(
                    APIResponse.success(message="Logout successful")
                )
                
        return web.json_response(
            APIResponse.error("No active session found"),
            status=400
        )
        
    async def create_api_key(self, request: Request) -> Response:
        """Crear API key"""
        try:
            data = await request.json()
            description = data.get("description", "")
            permissions_list = data.get("permissions", ["read_agents"])
            
            # Convertir strings a enums
            permissions = []
            for perm_str in permissions_list:
                try:
                    permissions.append(Permission(perm_str))
                except ValueError:
                    return web.json_response(
                        APIResponse.error(f"Invalid permission: {perm_str}"),
                        status=400
                    )
                    
            user_info = request["user_info"]
            user_id = user_info.get("username", "unknown")
            
            api_key = self.security_manager.create_api_key(
                user_id, permissions, description
            )
            
            return web.json_response(
                APIResponse.success({
                    "api_key": api_key,
                    "description": description,
                    "permissions": permissions_list
                })
            )
            
        except Exception as e:
            logging.error(f"Create API key error: {e}")
            return web.json_response(
                APIResponse.error("Failed to create API key", 500),
                status=500
            )

class AgentHandlers:
    """Handlers para gestión de agentes"""
    
    def __init__(self, framework: AgentFramework, security_manager: SecurityManager):
        self.framework = framework
        self.security_manager = security_manager
        
    async def list_agents(self, request: Request) -> Response:
        """Listar todos los agentes"""
        # Verificar permisos
        session_id = request.get("session_id")
        if session_id:
            authorized = await self.security_manager.authorize_action(
                session_id, Permission.READ_AGENTS
            )
            if not authorized:
                return web.json_response(
                    APIResponse.error("Insufficient permissions", 403),
                    status=403
                )
                
        try:
            agents = self.framework.registry.list_all_agents()
            
            # Parámetros de paginación
            page = int(request.query.get("page", 1))
            per_page = int(request.query.get("per_page", 10))
            
            # Filtros
            namespace_filter = request.query.get("namespace")
            status_filter = request.query.get("status")
            
            # Aplicar filtros
            filtered_agents = agents
            if namespace_filter:
                filtered_agents = [a for a in filtered_agents if namespace_filter in a.namespace]
            if status_filter:
                filtered_agents = [a for a in filtered_agents if a.status.value == status_filter]
                
            # Paginación
            total = len(filtered_agents)
            start = (page - 1) * per_page
            end = start + per_page
            page_agents = filtered_agents[start:end]
            
            # Serializar agentes
            agents_data = []
            for agent in page_agents:
                agent_data = {
                    "id": agent.id,
                    "name": agent.name,
                    "namespace": agent.namespace,
                    "status": agent.status.value,
                    "created_at": agent.created_at.isoformat(),
                    "last_heartbeat": agent.last_heartbeat.isoformat(),
                    "capabilities_count": len(agent.capabilities),
                    "metadata": agent.metadata
                }
                agents_data.append(agent_data)
                
            return web.json_response(
                APIResponse.paginated(agents_data, page, per_page, total)
            )
            
        except Exception as e:
            logging.error(f"List agents error: {e}")
            return web.json_response(
                APIResponse.error("Failed to list agents", 500),
                status=500
            )
            
    async def get_agent(self, request: Request) -> Response:
        """Obtener detalles de un agente específico"""
        agent_id = request.match_info["agent_id"]
        
        try:
            agent = self.framework.registry.get_agent(agent_id)
            if not agent:
                return web.json_response(
                    APIResponse.error("Agent not found", 404),
                    status=404
                )
                
            # Obtener recursos del agente
            agent_resources = self.framework.resource_manager.find_resources_by_owner(agent_id)
            
            agent_data = {
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
            
            return web.json_response(
                APIResponse.success(agent_data)
            )
            
        except Exception as e:
            logging.error(f"Get agent error: {e}")
            return web.json_response(
                APIResponse.error("Failed to get agent details", 500),
                status=500
            )
            
    async def create_agent(self, request: Request) -> Response:
        """Crear nuevo agente"""
        # Verificar permisos
        session_id = request.get("session_id")
        if session_id:
            authorized = await self.security_manager.authorize_action(
                session_id, Permission.CREATE_AGENTS
            )
            if not authorized:
                return web.json_response(
                    APIResponse.error("Insufficient permissions", 403),
                    status=403
                )
                
        try:
            data = await request.json()
            namespace = data.get("namespace")
            name = data.get("name")
            auto_start = data.get("auto_start", True)
            
            if not namespace or not name:
                return web.json_response(
                    APIResponse.error("Namespace and name are required"),
                    status=400
                )
                
            # Verificar que el namespace es válido
            available_namespaces = ExtendedAgentFactory.list_available_namespaces()
            if namespace not in available_namespaces:
                return web.json_response(
                    APIResponse.error(f"Unknown namespace: {namespace}"),
                    status=400
                )
                
            # Crear agente
            agent = ExtendedAgentFactory.create_agent(namespace, name, self.framework)
            
            if auto_start:
                await agent.start()
                
            agent_data = {
                "id": agent.id,
                "name": agent.name,
                "namespace": agent.namespace,
                "status": agent.status.value,
                "created_at": agent.created_at.isoformat()
            }
            
            return web.json_response(
                APIResponse.success(agent_data, "Agent created successfully")
            )
            
        except Exception as e:
            logging.error(f"Create agent error: {e}")
            return web.json_response(
                APIResponse.error("Failed to create agent", 500),
                status=500
            )
            
    async def execute_agent_action(self, request: Request) -> Response:
        """Ejecutar acción en un agente"""
        agent_id = request.match_info["agent_id"]
        
        # Verificar permisos
        session_id = request.get("session_id")
        if session_id:
            authorized = await self.security_manager.authorize_action(
                session_id, Permission.EXECUTE_ACTIONS
            )
            if not authorized:
                return web.json_response(
                    APIResponse.error("Insufficient permissions", 403),
                    status=403
                )
                
        try:
            agent = self.framework.registry.get_agent(agent_id)
            if not agent:
                return web.json_response(
                    APIResponse.error("Agent not found", 404),
                    status=404
                )
                
            data = await request.json()
            action = data.get("action")
            params = data.get("params", {})
            
            if not action:
                return web.json_response(
                    APIResponse.error("Action is required"),
                    status=400
                )
                
            # Ejecutar acción
            result = await agent.execute_action(action, params)
            
            return web.json_response(
                APIResponse.success({
                    "action": action,
                    "result": result,
                    "agent_id": agent_id
                })
            )
            
        except Exception as e:
            logging.error(f"Execute action error: {e}")
            return web.json_response(
                APIResponse.error(f"Failed to execute action: {str(e)}", 500),
                status=500
            )
            
    async def delete_agent(self, request: Request) -> Response:
        """Eliminar agente"""
        agent_id = request.match_info["agent_id"]
        
        # Verificar permisos
        session_id = request.get("session_id")
        if session_id:
            authorized = await self.security_manager.authorize_action(
                session_id, Permission.DELETE_AGENTS
            )
            if not authorized:
                return web.json_response(
                    APIResponse.error("Insufficient permissions", 403),
                    status=403
                )
                
        try:
            agent = self.framework.registry.get_agent(agent_id)
            if not agent:
                return web.json_response(
                    APIResponse.error("Agent not found", 404),
                    status=404
                )
                
            # Detener y eliminar agente
            await agent.stop()
            
            return web.json_response(
                APIResponse.success(message=f"Agent {agent_id} deleted successfully")
            )
            
        except Exception as e:
            logging.error(f"Delete agent error: {e}")
            return web.json_response(
                APIResponse.error("Failed to delete agent", 500),
                status=500
            )

class ResourceHandlers:
    """Handlers para gestión de recursos"""
    
    def __init__(self, framework: AgentFramework, security_manager: SecurityManager):
        self.framework = framework
        self.security_manager = security_manager
        
    async def list_resources(self, request: Request) -> Response:
        """Listar recursos"""
        try:
            # Obtener todos los recursos
            all_resources = []
            agents = self.framework.registry.list_all_agents()
            
            for agent in agents:
                agent_resources = self.framework.resource_manager.find_resources_by_owner(agent.id)
                all_resources.extend(agent_resources)
                
            # Filtros
            resource_type = request.query.get("type")
            owner_id = request.query.get("owner")
            
            if resource_type:
                all_resources = [r for r in all_resources if r.type.value == resource_type]
            if owner_id:
                all_resources = [r for r in all_resources if r.owner_agent_id == owner_id]
                
            # Paginación
            page = int(request.query.get("page", 1))
            per_page = int(request.query.get("per_page", 10))
            
            total = len(all_resources)
            start = (page - 1) * per_page
            end = start + per_page
            page_resources = all_resources[start:end]
            
            # Serializar recursos
            resources_data = []
            for resource in page_resources:
                resource_data = {
                    "id": resource.id,
                    "name": resource.name,
                    "type": resource.type.value,
                    "namespace": resource.namespace,
                    "owner_agent_id": resource.owner_agent_id,
                    "created_at": resource.created_at.isoformat(),
                    "updated_at": resource.updated_at.isoformat(),
                    "metadata": resource.metadata,
                    "data_size": len(str(resource.data))
                }
                resources_data.append(resource_data)
                
            return web.json_response(
                APIResponse.paginated(resources_data, page, per_page, total)
            )
            
        except Exception as e:
            logging.error(f"List resources error: {e}")
            return web.json_response(
                APIResponse.error("Failed to list resources", 500),
                status=500
            )
            
    async def get_resource(self, request: Request) -> Response:
        """Obtener detalles de un recurso"""
        resource_id = request.match_info["resource_id"]
        
        try:
            resource = await self.framework.resource_manager.get_resource(resource_id)
            if not resource:
                return web.json_response(
                    APIResponse.error("Resource not found", 404),
                    status=404
                )
                
            resource_data = {
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
            
            return web.json_response(
                APIResponse.success(resource_data)
            )
            
        except Exception as e:
            logging.error(f"Get resource error: {e}")
            return web.json_response(
                APIResponse.error("Failed to get resource", 500),
                status=500
            )

class SystemHandlers:
    """Handlers para información del sistema"""
    
    def __init__(self, framework: AgentFramework, security_manager: SecurityManager):
        self.framework = framework
        self.security_manager = security_manager
        self.metrics_collector = MetricsCollector(framework)
        
    async def health_check(self, request: Request) -> Response:
        """Health check del sistema"""
        agents = self.framework.registry.list_all_agents()
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "framework": {
                "total_agents": len(agents),
                "active_agents": len([a for a in agents if a.status == AgentStatus.ACTIVE]),
                "running": True
            },
            "security": self.security_manager.get_security_status()
        }
        
        return web.json_response(APIResponse.success(health_data))
        
    async def get_metrics(self, request: Request) -> Response:
        """Obtener métricas del sistema"""
        try:
            metrics = self.metrics_collector.get_metrics()
            
            # Añadir métricas adicionales
            agents = self.framework.registry.list_all_agents()
            status_counts = {}
            for agent in agents:
                status = agent.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
                
            metrics["agent_status_distribution"] = status_counts
            
            return web.json_response(APIResponse.success(metrics))
            
        except Exception as e:
            logging.error(f"Get metrics error: {e}")
            return web.json_response(
                APIResponse.error("Failed to get metrics", 500),
                status=500
            )
            
    async def get_namespaces(self, request: Request) -> Response:
        """Obtener namespaces disponibles"""
        try:
            namespaces = ExtendedAgentFactory.list_available_namespaces()
            
            return web.json_response(
                APIResponse.success({
                    "namespaces": namespaces,
                    "total": len(namespaces)
                })
            )
            
        except Exception as e:
            logging.error(f"Get namespaces error: {e}")
            return web.json_response(
                APIResponse.error("Failed to get namespaces", 500),
                status=500
            )

# ================================
# MAIN API SERVER
# ================================

class FrameworkAPIServer:
    """Servidor de API REST para el framework"""
    
    def __init__(self, framework: AgentFramework, host: str = "localhost", port: int = 8000):
        self.framework = framework
        self.host = host
        self.port = port
        self.app = web.Application(middlewares=[auth_middleware])
        
        # Configurar seguridad
        self.security_manager = SecurityManager({
            "jwt_secret": "your_jwt_secret_here",
            "session_max_hours": 24
        })
        
        # Configurar persistencia si está disponible
        self.persistence_manager = None
        
        # Añadir componentes al app
        self.app["framework"] = framework
        self.app["security_manager"] = self.security_manager
        
        # Configurar handlers
        self.auth_handlers = AuthenticationHandlers(self.security_manager)
        self.agent_handlers = AgentHandlers(framework, self.security_manager)
        self.resource_handlers = ResourceHandlers(framework, self.security_manager)
        self.system_handlers = SystemHandlers(framework, self.security_manager)
        
        # Configurar rutas
        self._setup_routes()
        
        # Configurar CORS
        self._setup_cors()
        
        # Configurar documentación Swagger
        self._setup_swagger()
        
    def _setup_routes(self):
        """Configurar rutas de la API"""
        
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
        """Configurar CORS"""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Aplicar CORS a todas las rutas
        for route in list(self.app.router.routes()):
            cors.add(route)
            
    def _setup_swagger(self):
        """Configurar documentación Swagger"""
        # En una implementación real, usarías aiohttp_swagger
        pass
        
    async def _serve_docs(self, request: Request) -> Response:
        """Servir documentación de la API"""
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
    
    <h2>Resources</h2>
    <div class="endpoint">
        <span class="method get">GET</span> <code>/api/resources</code>
        <p>List all resources with filters</p>
        <p>Query params: page, per_page, type, owner</p>
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
        """Iniciar servidor de API"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        logging.info(f"API server started at http://{self.host}:{self.port}")
        print(f"🌐 API server available at: http://{self.host}:{self.port}")
        print(f"📚 API documentation: http://{self.host}:{self.port}")
        
        return runner

# ================================
# EXAMPLE USAGE
# ================================

async def api_demo():
    """Demo de la API REST"""
    
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Starting API Demo")
    print("="*50)
    
    # Crear framework
    framework = AgentFramework()
    await framework.start()
    
    # Crear algunos agentes para demo
    strategist = ExtendedAgentFactory.create_agent("agent.planning.strategist", "strategist", framework)
    generator = ExtendedAgentFactory.create_agent("agent.build.code.generator", "generator", framework)
    
    await strategist.start()
    await generator.start()
    
    # Crear servidor API
    api_server = FrameworkAPIServer(framework, host="localhost", port=8000)
    runner = await api_server.start()
    
    print(f"\n✅ API server running with {len(framework.registry.list_all_agents())} agents")
    print("\nAvailable endpoints:")
    print("• POST /api/auth/login - Authentication")
    print("• GET  /api/agents - List agents")
    print("• POST /api/agents - Create agent")
    print("• GET  /api/health - Health check")
    print("• GET  /api/metrics - System metrics")
    print("\nTest the API:")
    print("curl http://localhost:8000/api/health")
    print("\nPress Ctrl+C to stop...")
    
    try:
        # Mantener servidor corriendo
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping API server...")
        
    finally:
        await framework.stop()
        await runner.cleanup()
        print("👋 API demo stopped")

if __name__ == "__main__":
    asyncio.run(api_demo())