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
from core.monitoring_system import MetricsCollector # <-- CAMBIO AQUI

# ================================\
# API MODELS AND SCHEMAS
# ================================\

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

# ================================\
# AUTHENTICATION AND AUTHORIZATION
# ================================\

async def jwt_middleware(app, handler):
    """Middleware para autenticación JWT"""
    async def middleware_handler(request: Request):
        security_manager: SecurityManager = app['security_manager']
        
        # Rutas públicas (no requieren autenticación)
        public_paths = ['/api/auth/login', '/api/health', '/swagger', '/swagger/swagger.json']
        if request.path in public_paths or request.path.startswith('/static/'):
            return await handler(request)

        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return web.json_response(APIResponse.error("Authentication required: Bearer token missing"), status=401)
        
        token = auth_header.split(' ')[1]
        
        try:
            payload = security_manager.validate_jwt_token(token)
            request['user'] = payload # Almacenar payload del usuario en la request
            return await handler(request)
        except Exception as e:
            logging.error(f"JWT validation failed: {e}")
            return web.json_response(APIResponse.error("Invalid or expired token", details=str(e)), status=401)
    
    return middleware_handler

def authorize(required_permission: Permission):
    """Decorator para autorización de permisos"""
    def decorator(func):
        async def wrapper(request: Request):
            user_payload = request.get('user')
            if not user_payload:
                return web.json_response(APIResponse.error("Authorization failed: No user context (middleware error)"), status=403)
            
            security_manager: SecurityManager = request.app['security_manager']
            
            if not security_manager.authorize_user_permission(user_payload['user_id'], required_permission):
                return web.json_response(APIResponse.error(f"Permission denied: Requires {required_permission.value}"), status=403)
            
            return await func(request)
        return wrapper
    return decorator


# ================================\
# API SERVER
# ================================\

class FrameworkAPIServer:
    """Servidor API REST para el Framework de Agentes"""
    
    def __init__(self, framework: AgentFramework, security_manager: SecurityManager, persistence_manager: PersistenceManager, host: str = "0.0.0.0", port: int = 8000):
        self.framework = framework
        self.security_manager = security_manager
        self.persistence_manager = persistence_manager
        self.host = host
        self.port = port
        self.app = web.Application(middlewares=[jwt_middleware])
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Initialize MetricsCollector
        self.metrics_collector = MetricsCollector(framework)
        
        self._setup_routes()
        self._setup_cors()
        self._setup_swagger()

        # Store important components in app for middleware/handlers
        self.app['framework'] = framework
        self.app['security_manager'] = security_manager
        self.app['persistence_manager'] = persistence_manager
        self.app['metrics_collector'] = self.metrics_collector # Make it accessible

    def _setup_routes(self):
        """Configura las rutas de la API."""
        self.app.router.add_get("/api/health", self.health_check)
        self.app.router.add_post("/api/auth/login", self.login)
        
        self.app.router.add_get("/api/agents", self.list_agents)
        self.app.router.add_post("/api/agents", self.create_agent)
        self.app.router.add_get("/api/agents/{agent_id}", self.get_agent_details)
        self.app.router.add_put("/api/agents/{agent_id}", self.update_agent)
        self.app.router.add_delete("/api/agents/{agent_id}", self.delete_agent)
        self.app.router.add_post("/api/agents/{agent_id}/start", self.start_agent)
        self.app.router.add_post("/api/agents/{agent_id}/stop", self.stop_agent)

        self.app.router.add_post("/api/messages", self.send_message)
        self.app.router.add_get("/api/messages", self.list_messages)
        self.app.router.add_get("/api/messages/{message_id}", self.get_message_details)

        self.app.router.add_get("/api/resources", self.list_resources)
        self.app.router.add_post("/api/resources", self.create_resource)
        self.app.router.add_get("/api/resources/{resource_id}", self.get_resource_details)
        self.app.router.add_put("/api/resources/{resource_id}", self.update_resource)
        self.app.router.add_delete("/api/resources/{resource_id}", self.delete_resource)

        self.app.router.add_get("/api/metrics", self.get_metrics)
        
        # WebSocket para streaming de logs o eventos
        self.app.router.add_get("/ws/events", self.websocket_handler)

    def _setup_cors(self):
        """Configura CORS para permitir solicitudes desde el dashboard web."""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
            )
        })
        for route in list(self.app.router.routes()):
            cors.add(route)

    def _setup_swagger(self):
        """Configura Swagger/OpenAPI UI."""
        aiohttp_swagger.setup_swagger(
            self.app,
            swagger_url="/swagger",
            ui_version=3,
            title="Autonomous Agent Framework API",
            description="API para interactuar con el Framework de Agentes Autónomos.",
            swagger_info={
                'info': {
                    'title': 'Autonomous Agent Framework API',
                    'version': '1.0.0',
                    'description': 'Comprehensive API for managing autonomous agents.',
                }
            }
        )
        self.logger.info(f"Swagger UI available at http://{self.host}:{self.port}/swagger")

    async def start(self):
        """Inicia el servidor API."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        self.logger.info(f"API Server started on http://{self.host}:{self.port}")
        return runner # Return runner to allow graceful shutdown

    async def stop(self):
        """Detiene el servidor API."""
        self.logger.info("Stopping API Server...")
        if self.app.cleanup_ctx:
            # Call cleanup functions if any are registered
            await self.app.cleanup()
        await self.app.shutdown()
        await self.app.cleanup()
        self.logger.info("API Server stopped.")


    # ================================\
    # API HANDLERS
    # ================================\

    async def health_check(self, request: Request):
        """
        ---
        summary: Health Check
        description: Checks the health and status of the framework and its components.
        tags:
          - System
        produces:
          - application/json
        responses:
          200:
            description: System is healthy
            schema:
              type: object
              properties:
                success: {type: boolean}
                message: {type: string}
                data:
                  type: object
                  properties:
                    status: {type: string, description: Overall system status}
                    framework: {type: object, description: Framework core status}
                    api: {type: object, description: API server status}
                    persistence: {type: object, description: Persistence system status}
                    security: {type: object, description: Security system status}
                    monitoring: {type: object, description: Monitoring system status}
        """
        framework: AgentFramework = request.app['framework']
        security_manager: SecurityManager = request.app['security_manager']
        persistence_manager: PersistenceManager = request.app['persistence_manager']
        metrics_collector: MetricsCollector = request.app['metrics_collector']
        
        framework_status = await framework.get_status()
        security_status = security_manager.get_security_status()
        persistence_status = await persistence_manager.get_status()
        monitoring_status = metrics_collector.get_monitoring_status() # Use metrics_collector for monitoring status

        overall_status = "healthy"
        if framework_status["status"] == "degraded" or \
           security_status["status"] == "degraded" or \
           persistence_status["status"] == "degraded" or \
           monitoring_status["status"] == "degraded":
            overall_status = "degraded"
        if framework_status["status"] == "error" or \
           security_status["status"] == "error" or \
           persistence_status["status"] == "error" or \
           monitoring_status["status"] == "error":
            overall_status = "error"

        response_data = {
            "status": overall_status,
            "message": "System operational",
            "framework": framework_status,
            "api": {"status": "healthy", "port": self.port},
            "persistence": persistence_status,
            "security": security_status,
            "monitoring": monitoring_status
        }
        return web.json_response(APIResponse.success(response_data))

    async def login(self, request: Request):
        """
        ---
        summary: User Login
        description: Authenticates a user and returns an authentication token.
        tags:
          - Authentication
        parameters:
          - in: body
            name: body
            schema:
              type: object
              required:
                - method
              properties:
                method: {type: string, enum: [api_key, jwt_token], description: Authentication method}
                username: {type: string, description: Username (for JWT_TOKEN)}
                password: {type: string, description: Password (for JWT_TOKEN)}
                api_key: {type: string, description: API Key (for API_KEY)}
        responses:
          200:
            description: Login successful
            schema:
              type: object
              properties:
                success: {type: boolean}
                data:
                  type: object
                  properties:
                    token: {type: string, description: JWT or API Key token}
          400: {description: Invalid input}
          401: {description: Authentication failed}
        """
        security_manager: SecurityManager = request.app['security_manager']
        data = await request.json()
        
        auth_method_str = data.get("method")
        username = data.get("username")
        password = data.get("password")
        api_key = data.get("api_key")

        if not auth_method_str:
            return web.json_response(APIResponse.error("Authentication method is required"), status=400)

        try:
            auth_method = AuthenticationMethod(auth_method_str.upper())
        except ValueError:
            return web.json_response(APIResponse.error("Invalid authentication method"), status=400)

        token = None
        if auth_method == AuthenticationMethod.JWT_TOKEN:
            if not username or not password:
                return web.json_response(APIResponse.error("Username and password are required for JWT login"), status=400)
            
            user_id = await security_manager.authenticate_user_jwt(username, password)
            if user_id:
                token = security_manager.generate_jwt_token(user_id)
            else:
                return web.json_response(APIResponse.error("Invalid username or password"), status=401)
        
        elif auth_method == AuthenticationMethod.API_KEY:
            if not api_key:
                return web.json_response(APIResponse.error("API Key is required for API_KEY login"), status=400)
            
            user_id = await security_manager.authenticate_api_key(api_key)
            if user_id:
                token = api_key # For API Key method, the key itself acts as the token
            else:
                return web.json_response(APIResponse.error("Invalid API Key"), status=401)
        
        else:
            return web.json_response(APIResponse.error(f"Authentication method {auth_method_str} not supported"), status=400)

        if token:
            return web.json_response(APIResponse.success({"token": token}, "Login successful"))
        else:
            return web.json_response(APIResponse.error("Authentication failed"), status=401)

    @authorize(Permission.READ_AGENTS)
    async def list_agents(self, request: Request):
        """
        ---
        summary: List Agents
        description: Retrieves a list of all registered agents in the framework.
        tags:
          - Agents
        produces:
          - application/json
        responses:
          200:
            description: List of agents
            schema:
              type: object
              properties:
                success: {type: boolean}
                data:
                  type: array
                  items:
                    type: object
                    properties:
                      id: {type: string}
                      name: {type: string}
                      namespace: {type: string}
                      status: {type: string}
                      capabilities: {type: array, items: {type: object}}
                      last_heartbeat: {type: string, format: date-time}
          403: {description: Permission denied}
        """
        framework: AgentFramework = request.app['framework']
        agents = framework.registry.list_all_agents()
        agent_data = [
            {
                "id": agent.id,
                "name": agent.name,
                "namespace": agent.namespace,
                "status": agent.status.value,
                "capabilities": [{"name": c.name, "namespace": c.namespace, "description": c.description} for c in agent.capabilities],
                "last_heartbeat": agent.last_heartbeat.isoformat() if agent.last_heartbeat else None
            } for agent in agents
        ]
        return web.json_response(APIResponse.success(agent_data))

    @authorize(Permission.CREATE_AGENTS)
    async def create_agent(self, request: Request):
        """
        ---
        summary: Create Agent
        description: Creates a new agent in the framework.
        tags:
          - Agents
        parameters:
          - in: body
            name: body
            schema:
              type: object
              required:
                - namespace
                - name
              properties:
                namespace: {type: string, description: Namespace of the agent (e.g., agent.planning.strategist)}
                name: {type: string, description: Unique name of the agent}
                description: {type: string, description: Optional description for the agent}
                capabilities:
                  type: array
                  items:
                    type: object
                    properties:
                      name: {type: string}
                      namespace: {type: string}
                      description: {type: string}
        responses:
          200:
            description: Agent created successfully
          400: {description: Invalid input}
          403: {description: Permission denied}
          409: {description: Agent with this name/ID already exists}
        """
        framework: AgentFramework = request.app['framework']
        data = await request.json()
        
        namespace = data.get("namespace")
        name = data.get("name")
        description = data.get("description", "")
        capabilities_data = data.get("capabilities", [])

        if not namespace or not name:
            return web.json_response(APIResponse.error("Namespace and name are required"), status=400)
        
        try:
            # Use ExtendedAgentFactory to create agents dynamically from string namespace
            agent = ExtendedAgentFactory.create_agent(namespace, name, framework, description=description)
            if not agent:
                return web.json_response(APIResponse.error(f"Invalid agent namespace or type: {namespace}"), status=400)
            
            # Add capabilities from payload if provided
            for cap_data in capabilities_data:
                agent.add_capability(cap_data["name"], cap_data["namespace"], cap_data.get("description", ""))

            await framework.registry.register_agent(agent)
            await agent.initialize() # Initialize the agent after registration
            
            return web.json_response(APIResponse.success({"id": agent.id, "name": agent.name, "status": agent.status.value}, "Agent created"))
        except ValueError as e:
            return web.json_response(APIResponse.error(str(e)), status=409)
        except Exception as e:
            self.logger.exception("Error creating agent")
            return web.json_response(APIResponse.error(f"Internal server error: {e}"), status=500)

    @authorize(Permission.READ_AGENTS)
    async def get_agent_details(self, request: Request):
        """
        ---
        summary: Get Agent Details
        description: Retrieves detailed information about a specific agent by ID.
        tags:
          - Agents
        parameters:
          - in: path
            name: agent_id
            type: string
            required: true
            description: ID of the agent to retrieve.
        responses:
          200:
            description: Agent details
          404: {description: Agent not found}
          403: {description: Permission denied}
        """
        framework: AgentFramework = request.app['framework']
        agent_id = request.match_info['agent_id']
        
        agent = framework.registry.get_agent(agent_id)
        if not agent:
            return web.json_response(APIResponse.error(f"Agent with ID {agent_id} not found"), status=404)
        
        agent_data = {
            "id": agent.id,
            "name": agent.name,
            "namespace": agent.namespace,
            "status": agent.status.value,
            "description": agent.description,
            "capabilities": [{"name": c.name, "namespace": c.namespace, "description": c.description} for c in agent.capabilities],
            "last_heartbeat": agent.last_heartbeat.isoformat() if agent.last_heartbeat else None,
            "created_at": agent.created_at.isoformat(),
            "metrics": agent.get_metrics() # Example: agent-specific metrics
        }
        return web.json_response(APIResponse.success(agent_data))

    @authorize(Permission.WRITE_AGENTS)
    async def update_agent(self, request: Request):
        """
        ---
        summary: Update Agent
        description: Updates the properties of an existing agent.
        tags:
          - Agents
        parameters:
          - in: path
            name: agent_id
            type: string
            required: true
            description: ID of the agent to update.
          - in: body
            name: body
            schema:
              type: object
              properties:
                name: {type: string}
                description: {type: string}
                status: {type: string, enum: [initializing, active, busy, idle, error, terminated, suspended]}
        responses:
          200:
            description: Agent updated successfully
          400: {description: Invalid input}
          403: {description: Permission denied}
          404: {description: Agent not found}
        """
        framework: AgentFramework = request.app['framework']
        agent_id = request.match_info['agent_id']
        data = await request.json()

        agent = framework.registry.get_agent(agent_id)
        if not agent:
            return web.json_response(APIResponse.error(f"Agent with ID {agent_id} not found"), status=404)
        
        if 'name' in data:
            agent.name = data['name']
        if 'description' in data:
            agent.description = data['description']
        if 'status' in data:
            try:
                agent.status = AgentStatus(data['status'].lower())
            except ValueError:
                return web.json_response(APIResponse.error("Invalid agent status"), status=400)
        
        # In a real system, you might save the agent state to persistence here
        # await request.app['persistence_manager'].save_agent_state(agent) # Assuming this method exists

        return web.json_response(APIResponse.success({"id": agent.id, "name": agent.name, "status": agent.status.value}, "Agent updated"))

    @authorize(Permission.DELETE_AGENTS)
    async def delete_agent(self, request: Request):
        """
        ---
        summary: Delete Agent
        description: Deletes an agent from the framework by ID.
        tags:
          - Agents
        parameters:
          - in: path
            name: agent_id
            type: string
            required: true
            description: ID of the agent to delete.
        responses:
          200:
            description: Agent deleted successfully
          403: {description: Permission denied}
          404: {description: Agent not found}
        """
        framework: AgentFramework = request.app['framework']
        agent_id = request.match_info['agent_id']
        
        agent = framework.registry.get_agent(agent_id)
        if not agent:
            return web.json_response(APIResponse.error(f"Agent with ID {agent_id} not found"), status=404)
        
        await framework.registry.deregister_agent(agent_id)
        return web.json_response(APIResponse.success(None, f"Agent {agent_id} deleted"))

    @authorize(Permission.EXECUTE_ACTIONS)
    async def start_agent(self, request: Request):
        """
        ---
        summary: Start Agent
        description: Starts a registered agent.
        tags:
          - Agents
        parameters:
          - in: path
            name: agent_id
            type: string
            required: true
            description: ID of the agent to start.
        responses:
          200:
            description: Agent started successfully
          404: {description: Agent not found}
          400: {description: Agent already active or error starting}
          403: {description: Permission denied}
        """
        framework: AgentFramework = request.app['framework']
        agent_id = request.match_info['agent_id']
        
        agent = framework.registry.get_agent(agent_id)
        if not agent:
            return web.json_response(APIResponse.error(f"Agent with ID {agent_id} not found"), status=404)
        
        if agent.status == AgentStatus.ACTIVE or agent.status == AgentStatus.BUSY:
            return web.json_response(APIResponse.error(f"Agent {agent_id} is already active."), status=400)
            
        try:
            await agent.start()
            return web.json_response(APIResponse.success({"id": agent.id, "status": agent.status.value}, f"Agent {agent_id} started"))
        except Exception as e:
            self.logger.exception(f"Error starting agent {agent_id}")
            return web.json_response(APIResponse.error(f"Error starting agent: {e}"), status=500)

    @authorize(Permission.EXECUTE_ACTIONS)
    async def stop_agent(self, request: Request):
        """
        ---
        summary: Stop Agent
        description: Stops a running agent.
        tags:
          - Agents
        parameters:
          - in: path
            name: agent_id
            type: string
            required: true
            description: ID of the agent to stop.
        responses:
          200:
            description: Agent stopped successfully
          404: {description: Agent not found}
          400: {description: Agent already inactive or error stopping}
          403: {description: Permission denied}
        """
        framework: AgentFramework = request.app['framework']
        agent_id = request.match_info['agent_id']
        
        agent = framework.registry.get_agent(agent_id)
        if not agent:
            return web.json_response(APIResponse.error(f"Agent with ID {agent_id} not found"), status=404)
        
        if agent.status == AgentStatus.TERMINATED:
            return web.json_response(APIResponse.error(f"Agent {agent_id} is already terminated."), status=400)
            
        try:
            await agent.stop()
            return web.json_response(APIResponse.success({"id": agent.id, "status": agent.status.value}, f"Agent {agent_id} stopped"))
        except Exception as e:
            self.logger.exception(f"Error stopping agent {agent_id}")
            return web.json_response(APIResponse.error(f"Error stopping agent: {e}"), status=500)

    @authorize(Permission.EXECUTE_ACTIONS)
    async def send_message(self, request: Request):
        """
        ---
        summary: Send Message
        description: Sends a message (command or request) to a specific agent.
        tags:
          - Messaging
        parameters:
          - in: body
            name: body
            schema:
              type: object
              required:
                - receiver_id
                - message_type
                - payload
              properties:
                receiver_id: {type: string, description: ID of the receiving agent}
                message_type: {type: string, enum: [command, request, response, event], description: Type of message}
                payload: {type: object, description: Message payload (e.g., {'action': 'do_something', 'params': {}})}
        responses:
          200:
            description: Message sent successfully
          400: {description: Invalid input}
          404: {description: Receiver agent not found}
          403: {description: Permission denied}
        """
        framework: AgentFramework = request.app['framework']
        sender_id = request['user']['user_id'] # Use authenticated user/agent ID as sender
        data = await request.json()

        receiver_id = data.get("receiver_id")
        message_type_str = data.get("message_type")
        payload = data.get("payload")

        if not receiver_id or not message_type_str or not payload:
            return web.json_response(APIResponse.error("Receiver ID, message type, and payload are required"), status=400)
        
        receiver_agent = framework.registry.get_agent(receiver_id)
        if not receiver_agent:
            return web.json_response(APIResponse.error(f"Receiver agent with ID {receiver_id} not found"), status=404)

        try:
            message_type = MessageType(message_type_str.upper())
            message_id = await framework.message_bus.send_message(sender_id, receiver_id, message_type, payload)
            return web.json_response(APIResponse.success({"message_id": message_id}, "Message sent"))
        except ValueError:
            return web.json_response(APIResponse.error("Invalid message type"), status=400)
        except Exception as e:
            self.logger.exception("Error sending message")
            return web.json_response(APIResponse.error(f"Internal server error: {e}"), status=500)

    @authorize(Permission.READ_MESSAGES)
    async def list_messages(self, request: Request):
        """
        ---
        summary: List Messages
        description: Retrieves a list of messages, optionally filtered by sender, receiver, or type.
        tags:
          - Messaging
        parameters:
          - in: query
            name: sender_id
            type: string
            description: Filter messages by sender agent ID.
          - in: query
            name: receiver_id
            type: string
            description: Filter messages by receiver agent ID.
          - in: query
            name: message_type
            type: string
            enum: [command, request, response, event, heartbeat, error]
            description: Filter messages by type.
          - in: query
            name: limit
            type: integer
            description: Maximum number of messages to return (default: 100).
        responses:
          200:
            description: List of messages
          403: {description: Permission denied}
        """
        persistence_manager: PersistenceManager = request.app['persistence_manager']
        
        sender_id = request.query.get("sender_id")
        receiver_id = request.query.get("receiver_id")
        message_type_str = request.query.get("message_type")
        limit = int(request.query.get("limit", 100))

        message_type = None
        if message_type_str:
            try:
                message_type = MessageType(message_type_str.upper())
            except ValueError:
                return web.json_response(APIResponse.error("Invalid message type filter"), status=400)

        messages = await persistence_manager.backend.load_messages(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            limit=limit
        )
        
        message_data = [asdict(msg) for msg in messages] # Assuming AgentMessage is a dataclass
        # Convert datetime objects to ISO format strings
        for msg in message_data:
            if isinstance(msg.get('timestamp'), datetime):
                msg['timestamp'] = msg['timestamp'].isoformat()
            if isinstance(msg.get('message_type'), Enum):
                msg['message_type'] = msg['message_type'].value
        
        return web.json_response(APIResponse.success(message_data))

    @authorize(Permission.READ_MESSAGES)
    async def get_message_details(self, request: Request):
        """
        ---
        summary: Get Message Details
        description: Retrieves details of a specific message by ID.
        tags:
          - Messaging
        parameters:
          - in: path
            name: message_id
            type: string
            required: true
            description: ID of the message to retrieve.
        responses:
          200:
            description: Message details
          404: {description: Message not found}
          403: {description: Permission denied}
        """
        persistence_manager: PersistenceManager = request.app['persistence_manager']
        message_id = request.match_info['message_id']
        
        # This currently relies on backend being able to query by message_id directly
        # If not, you might need to iterate through messages or implement specific index.
        # For simplicity, assuming a method exists:
        message = await persistence_manager.backend.load_message_by_id(message_id) 
        
        if not message:
            return web.json_response(APIResponse.error(f"Message with ID {message_id} not found"), status=404)
        
        message_data = asdict(message)
        if isinstance(message_data.get('timestamp'), datetime):
            message_data['timestamp'] = message_data['timestamp'].isoformat()
        if isinstance(message_data.get('message_type'), Enum):
            message_data['message_type'] = message_data['message_type'].value

        return web.json_response(APIResponse.success(message_data))


    @authorize(Permission.READ_RESOURCES)
    async def list_resources(self, request: Request):
        """
        ---
        summary: List Resources
        description: Retrieves a list of all managed resources.
        tags:
          - Resources
        produces:
          - application/json
        responses:
          200:
            description: List of resources
        """
        framework: AgentFramework = request.app['framework']
        resources = framework.resource_manager.list_all_resources()
        resource_data = [asdict(r) for r in resources]
        # Convert Enum and datetime objects to strings
        for res in resource_data:
            if 'type' in res and isinstance(res['type'], Enum):
                res['type'] = res['type'].value
            if 'created_at' in res and isinstance(res['created_at'], datetime):
                res['created_at'] = res['created_at'].isoformat()
            if 'last_updated' in res and isinstance(res['last_updated'], datetime):
                res['last_updated'] = res['last_updated'].isoformat()
        return web.json_response(APIResponse.success(resource_data))

    @authorize(Permission.WRITE_RESOURCES)
    async def create_resource(self, request: Request):
        """
        ---
        summary: Create Resource
        description: Creates a new managed resource.
        tags:
          - Resources
        parameters:
          - in: body
            name: body
            schema:
              type: object
              required:
                - type
                - name
                - namespace
                - data
                - owner_agent_id
              properties:
                type: {type: string, enum: [code, infra, workflow, ui, data, test, security, release]}
                name: {type: string}
                namespace: {type: string}
                data: {type: object, description: JSON data of the resource}
                owner_agent_id: {type: string}
                description: {type: string}
        responses:
          200: {description: Resource created successfully}
          400: {description: Invalid input}
          403: {description: Permission denied}
        """
        framework: AgentFramework = request.app['framework']
        data = await request.json()

        resource_type_str = data.get("type")
        name = data.get("name")
        namespace = data.get("namespace")
        resource_data_payload = data.get("data")
        owner_agent_id = data.get("owner_agent_id")
        description = data.get("description", "")

        if not all([resource_type_str, name, namespace, resource_data_payload, owner_agent_id]):
            return web.json_response(APIResponse.error("Type, name, namespace, data, and owner_agent_id are required"), status=400)
        
        try:
            resource_type = ResourceType(resource_type_str.lower())
        except ValueError:
            return web.json_response(APIResponse.error("Invalid resource type"), status=400)

        new_resource = AgentResource(
            type=resource_type,
            name=name,
            namespace=namespace,
            data=resource_data_payload,
            owner_agent_id=owner_agent_id,
            description=description
        )
        
        await framework.resource_manager.create_resource(new_resource)
        return web.json_response(APIResponse.success({"id": new_resource.id, "name": new_resource.name}, "Resource created"))

    @authorize(Permission.READ_RESOURCES)
    async def get_resource_details(self, request: Request):
        """
        ---
        summary: Get Resource Details
        description: Retrieves details of a specific resource by ID.
        tags:
          - Resources
        parameters:
          - in: path
            name: resource_id
            type: string
            required: true
            description: ID of the resource to retrieve.
        responses:
          200: {description: Resource details}
          404: {description: Resource not found}
          403: {description: Permission denied}
        """
        framework: AgentFramework = request.app['framework']
        resource_id = request.match_info['resource_id']
        
        resource = framework.resource_manager.get_resource(resource_id)
        if not resource:
            return web.json_response(APIResponse.error(f"Resource with ID {resource_id} not found"), status=404)
        
        resource_data = asdict(resource)
        if 'type' in resource_data and isinstance(resource_data['type'], Enum):
            resource_data['type'] = resource_data['type'].value
        if 'created_at' in resource_data and isinstance(resource_data['created_at'], datetime):
            resource_data['created_at'] = resource_data['created_at'].isoformat()
        if 'last_updated' in resource_data and isinstance(resource_data['last_updated'], datetime):
            resource_data['last_updated'] = resource_data['last_updated'].isoformat()
            
        return web.json_response(APIResponse.success(resource_data))

    @authorize(Permission.WRITE_RESOURCES)
    async def update_resource(self, request: Request):
        """
        ---
        summary: Update Resource
        description: Updates an existing managed resource by ID.
        tags:
          - Resources
        parameters:
          - in: path
            name: resource_id
            type: string
            required: true
            description: ID of the resource to update.
          - in: body
            name: body
            schema:
              type: object
              properties:
                name: {type: string}
                namespace: {type: string}
                data: {type: object, description: Updated JSON data of the resource}
                description: {type: string}
        responses:
          200: {description: Resource updated successfully}
          400: {description: Invalid input}
          403: {description: Permission denied}
          404: {description: Resource not found}
        """
        framework: AgentFramework = request.app['framework']
        resource_id = request.match_info['resource_id']
        data = await request.json()

        resource = framework.resource_manager.get_resource(resource_id)
        if not resource:
            return web.json_response(APIResponse.error(f"Resource with ID {resource_id} not found"), status=404)
        
        if 'name' in data:
            resource.name = data['name']
        if 'namespace' in data:
            resource.namespace = data['namespace']
        if 'data' in data:
            resource.data = data['data']
        if 'description' in data:
            resource.description = data['description']
        
        resource.last_updated = datetime.now()
        
        await framework.resource_manager.update_resource(resource) # Assuming update_resource method exists
        return web.json_response(APIResponse.success({"id": resource.id, "name": resource.name}, "Resource updated"))

    @authorize(Permission.WRITE_RESOURCES) # Or DELETE_RESOURCES if specific
    async def delete_resource(self, request: Request):
        """
        ---
        summary: Delete Resource
        description: Deletes a managed resource by ID.
        tags:
          - Resources
        parameters:
          - in: path
            name: resource_id
            type: string
            required: true
            description: ID of the resource to delete.
        responses:
          200: {description: Resource deleted successfully}
          403: {description: Permission denied}
          404: {description: Resource not found}
        """
        framework: AgentFramework = request.app['framework']
        resource_id = request.match_info['resource_id']
        
        resource = framework.resource_manager.get_resource(resource_id)
        if not resource:
            return web.json_response(APIResponse.error(f"Resource with ID {resource_id} not found"), status=404)
        
        framework.resource_manager.delete_resource(resource_id)
        return web.json_response(APIResponse.success(None, f"Resource {resource_id} deleted"))

    @authorize(Permission.MONITOR_SYSTEM)
    async def get_metrics(self, request: Request):
        """
        ---
        summary: Get System Metrics
        description: Retrieves real-time system metrics.
        tags:
          - Monitoring
        produces:
          - application/json
        responses:
          200:
            description: System metrics data
        """
        metrics_collector: MetricsCollector = request.app['metrics_collector']
        metrics = metrics_collector.export_metrics()
        
        # Convert Metric objects to dicts and Enum/datetime to strings
        serializable_metrics = {}
        for key, metric in metrics.items():
            metric_data = asdict(metric)
            if 'type' in metric_data and isinstance(metric_data['type'], Enum):
                metric_data['type'] = metric_data['type'].value
            if 'timestamp' in metric_data and isinstance(metric_data['timestamp'], datetime):
                metric_data['timestamp'] = metric_data['timestamp'].isoformat()
            serializable_metrics[key] = metric_data
            
        return web.json_response(APIResponse.success(serializable_metrics))

    async def websocket_handler(self, request: Request):
        """
        ---
        summary: WebSocket Event Stream
        description: Establishes a WebSocket connection for real-time system events and logs.
        tags:
          - System
          - Real-time
        responses:
          101: {description: WebSocket upgrade successful}
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.logger.info("WebSocket connection established.")
        self.app['websockets'].append(ws) # Store reference to active websockets
        
        # Initial data push (e.g., current metrics, agent list)
        await ws.send_json(APIResponse.success({"type": "initial_data", "metrics": self.metrics_collector.export_metrics()}))
        await ws.send_json(APIResponse.success({"type": "initial_data", "agents": [asdict(a) for a in self.framework.registry.list_all_agents()]}))

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    # Handle incoming messages from WebSocket clients if needed (e.g., commands)
                    self.logger.info(f"Received WS message: {msg.json()}")
                    # Echo back for demo
                    # await ws.send_json(APIResponse.success({"echo": msg.json()}))
                elif msg.type == WSMsgType.ERROR:
                    self.logger.error(f"WebSocket connection closed with exception: {ws.exception()}")
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
        finally:
            self.logger.info("WebSocket connection closed.")
            self.app['websockets'].remove(ws) # Remove reference

# Example usage (for direct execution during development)
async def main():
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Starting API Demo")
    print("="*50)
    
    # Crear framework
    framework = AgentFramework()
    await framework.start()
    
    # Iniciar Seguridad
    security_manager = SecurityManager(framework)
    await security_manager.initialize()

    # Iniciar Persistencia
    persistence_manager = PersistenceFactory.create_persistence_manager(
        framework, PersistenceBackend.SQLITE, "api_demo.db"
    )
    await persistence_manager.initialize()

    # Crear algunos agentes para demo
    strategist = ExtendedAgentFactory.create_agent("agent.planning.strategist", "strategist", framework)
    generator = ExtendedAgentFactory.create_agent("agent.build.code.generator", "generator", framework)
    
    await strategist.start()
    await generator.start()
    
    # Crear servidor API
    api_server = FrameworkAPIServer(framework, security_manager, persistence_manager, host="localhost", port=8000)
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
        await persistence_manager.close()
        await runner.cleanup() # Clean up aiohttp runner

if __name__ == "__main__":
    asyncio.run(main())