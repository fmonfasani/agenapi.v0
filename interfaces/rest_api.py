# interfaces/rest_api.py

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from aiohttp import web, WSMsgType
import aiohttp_cors
from aiohttp.web_request import Request
from aiohttp.web_response import Response
import aiohttp_swagger # For API documentation

# Import from new modular structure
from autonomous_agent_framework import AgentFramework
from framework.security_manager import SecurityManager # Assuming new path
from framework.persistence_manager import PersistenceManager # Assuming new path
from framework.monitoring_manager import MonitoringManager # Assuming new path (conceptual MonitoringOrchestrator)
from agentapi.interfaces.agent_interfaces import BaseAgent, AgentStatus, MessageType # Keep for type hints
from agentapi.models.agent_models import AgentResource, ResourceType # Keep for type hints
from agentapi.models.general_models import APIResponse # For standard responses

# ================================
# API MODELS AND SCHEMAS (using APIResponse from general_models)
# ================================


class FrameworkAPIServer:
    """Servidor web para la API REST del framework."""
    
    def __init__(self, framework: AgentFramework, host: str = "0.0.0.0", port: int = 8000):
        self.framework = framework
        self.security_manager = framework.security_manager
        self.persistence_manager = framework.persistence_manager
        self.monitoring_manager = framework.monitoring_manager
        self.host = host
        self.port = port
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.logger = logging.getLogger("FrameworkAPIServer")
        self._setup_routes()
        self._setup_cors()
        # self._setup_swagger() # Disabled for brevity in this example

    def _setup_routes(self):
        self.app.router.add_get('/api/health', self.health_check)
        self.app.router.add_post('/api/auth/login', self.user_login)
        self.app.router.add_get('/api/agents', self.list_agents)
        self.app.router.add_post('/api/agents', self.create_agent)
        self.app.router.add_get('/api/agents/{agent_id}', self.get_agent_info)
        self.app.router.add_get('/api/resources', self.list_resources)
        self.app.router.add_post('/api/resources', self.create_resource)
        self.app.router.add_get('/api/metrics', self.get_metrics)
        self.app.router.add_get('/api/alerts', self.get_alerts)
        self.app.router.add_post('/api/messages', self.send_message) # For external systems to send messages to agents

        self.logger.info("API routes set up.")

    def _setup_cors(self):
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            )
        })
        for route in list(self.app.router.routes()):
            cors.add(route)
        self.logger.info("CORS configured.")

    # def _setup_swagger(self):
    #     aiohttp_swagger.setup_swagger(
    #         self.app,
    #         swagger_url="/api/docs",
    #         ui_url="/api/docs",
    #         swagger_from_file="swagger.yaml" # You would create a swagger.yaml definition
    #     )
    #     self.logger.info("Swagger documentation configured at /api/docs.")

    async def start(self) -> web.AppRunner:
        """Starts the aiohttp web server."""
        self.logger.info(f"Starting API server on {self.host}:{self.port}...")
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        self.logger.info("API server started.")
        return self.runner

    async def stop(self):
        """Stops the aiohttp web server."""
        self.logger.info("Stopping API server...")
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        self.logger.info("API server stopped.")

    # ================================
    # API ENDPOINT HANDLERS
    # ================================

    async def health_check(self, request: Request):
        """
        ---
        summary: Health check endpoint
        tags:
          - System
        produces:
          - application/json
        responses:
          '200':
            description: System is healthy
            schema:
              type: object
              properties:
                status: {type: string}
        """
        overall_status = "healthy" # Simplified for demo
        # In a real scenario, you'd query monitoring_manager.get_health_status()
        return web.json_response(APIResponse.success_response({"status": overall_status}))

    async def user_login(self, request: Request):
        """
        ---
        summary: Authenticate a user
        tags:
          - Authentication
        consumes:
          - application/json
        produces:
          - application/json
        parameters:
          - in: body
            name: credentials
            schema:
              type: object
              required:
                - username
                - password
              properties:
                username: {type: string}
                password: {type: string}
        responses:
          '200':
            description: Authentication successful, returns a token
          '401':
            description: Invalid credentials
        """
        try:
            data = await request.json()
            username = data.get('username')
            password = data.get('password')

            user = await self.security_manager.authenticate_user(username, password)
            if user:
                # In a real system, generate a JWT token here
                token = "dummy_jwt_token_for_" + user.id
                return web.json_response(APIResponse.success_response({"token": token, "user_id": user.id}))
            else:
                return web.json_response(APIResponse.error_response("Invalid credentials", code=401), status=401)
        except Exception as e:
            self.logger.error(f"Login error: {e}", exc_info=True)
            return web.json_response(APIResponse.error_response(f"Internal server error: {e}"), status=500)

    # Agent Management Endpoints
    async def list_agents(self, request: Request):
        """
        ---
        summary: List all registered agents
        tags:
          - Agents
        produces:
          - application/json
        responses:
          '200':
            description: List of agents
        """
        # Basic authorization check
        # if not await self.security_manager.authorize_action(request.headers.get('Authorization'), Permission.READ_AGENTS):
        #     return web.json_response(APIResponse.error_response("Unauthorized", code=403), status=403)
            
        agents_info = await self.framework.registry.list_all_agents()
        return web.json_response(APIResponse.success_response([a.to_dict() for a in agents_info]))

    async def get_agent_info(self, request: Request):
        """
        ---
        summary: Get details of a specific agent
        tags:
          - Agents
        parameters:
          - in: path
            name: agent_id
            type: string
            required: true
            description: ID of the agent to retrieve
        produces:
          - application/json
        responses:
          '200':
            description: Agent details
          '404':
            description: Agent not found
        """
        agent_id = request.match_info.get('agent_id')
        agent_info = await self.framework.registry.get_agent_info(agent_id)
        if agent_info:
            return web.json_response(APIResponse.success_response(agent_info.to_dict()))
        return web.json_response(APIResponse.error_response(f"Agent {agent_id} not found", code=404), status=404)

    async def create_agent(self, request: Request):
        """
        ---
        summary: Create a new agent
        tags:
          - Agents
        consumes:
          - application/json
        parameters:
          - in: body
            name: agent_details
            schema:
              type: object
              required:
                - namespace
                - name
                - agent_class_path # e.g., "agents.specialized_agents.StrategistAgent"
              properties:
                namespace: {type: string}
                name: {type: string}
                agent_class_path: {type: string}
                creator_agent_id: {type: string} # Optional, if created by another agent via API
        produces:
          - application/json
        responses:
          '201':
            description: Agent created successfully
          '400':
            description: Invalid input
        """
        try:
            data = await request.json()
            namespace = data.get('namespace')
            name = data.get('name')
            agent_class_path = data.get('agent_class_path')
            creator_agent_id = data.get('creator_agent_id') # From payload, or context if auth token used

            # Dynamic import of agent class
            module_name, class_name = agent_class_path.rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            agent_class = getattr(module, class_name)

            new_agent = await self.framework.agent_factory.create_agent(
                namespace=namespace,
                name=name,
                agent_class=agent_class,
                creator_agent_id=creator_agent_id # Could be user ID from auth token or an agent ID
            )
            if new_agent:
                return web.json_response(APIResponse.success_response(new_agent.to_dict(), "Agent created successfully"), status=201)
            else:
                return web.json_response(APIResponse.error_response("Failed to create agent", code=400), status=400)
        except ImportError as e:
            self.logger.error(f"Agent class import error: {e}", exc_info=True)
            return web.json_response(APIResponse.error_response(f"Invalid agent_class_path: {e}", code=400), status=400)
        except AttributeError as e:
            self.logger.error(f"Agent class not found: {e}", exc_info=True)
            return web.json_response(APIResponse.error_response(f"Agent class not found: {e}", code=400), status=400)
        except Exception as e:
            self.logger.error(f"Error creating agent: {e}", exc_info=True)
            return web.json_response(APIResponse.error_response(f"Internal server error: {e}", code=500), status=500)

    # Resource Management Endpoints
    async def list_resources(self, request: Request):
        """
        ---
        summary: List all registered resources
        tags:
          - Resources
        produces:
          - application/json
        responses:
          '200':
            description: List of resources
        """
        # A more complex framework would have a list_all_resources method in resource_manager
        # For now, we'll assume it accesses the registry.
        resources_info = [r.to_dict() for r in await self.framework.registry.list_all_agents()] # Placeholder for actual resource listing
        return web.json_response(APIResponse.success_response(resources_info))

    async def create_resource(self, request: Request):
        """
        ---
        summary: Create a new resource
        tags:
          - Resources
        consumes:
          - application/json
        parameters:
          - in: body
            name: resource_details
            schema:
              type: object
              required:
                - name
                - type
                - namespace
                - data
              properties:
                name: {type: string}
                type: {type: string, enum: [code, infra, workflow, ui, data, test, security, release, document, config, log]}
                namespace: {type: string}
                data: {type: object}
                owner_agent_id: {type: string} # Optional
        produces:
          - application/json
        responses:
          '201':
            description: Resource created successfully
          '400':
            description: Invalid input
        """
        try:
            data = await request.json()
            resource_type_str = data.get('type')
            
            try:
                resource_type = ResourceType[resource_type_str.upper()]
            except KeyError:
                return web.json_response(APIResponse.error_response(f"Invalid resource type: {resource_type_str}. Must be one of {list(ResourceType.__members__.keys())}", code=400), status=400)
            
            resource_name = data.get('name')
            namespace = data.get('namespace')
            resource_content = data.get('data')
            owner_agent_id = data.get('owner_agent_id')

            resource = AgentResource(
                name=resource_name,
                type=resource_type,
                namespace=namespace,
                data=resource_content,
                owner_agent_id=owner_agent_id
            )
            await self.framework.resource_manager.add_resource(resource)
            return web.json_response(APIResponse.success_response(resource.to_dict(), "Resource created successfully"), status=201)
        except Exception as e:
            self.logger.error(f"Error creating resource: {e}", exc_info=True)
            return web.json_response(APIResponse.error_response(f"Internal server error: {e}", code=500), status=500)

    # Monitoring Endpoints
    async def get_metrics(self, request: Request):
        """
        ---
        summary: Get system metrics
        tags:
          - Monitoring
        produces:
          - application/json
        responses:
          '200':
            description: List of collected metrics
        """
        # In a real system, you might filter by type, agent_id, time range
        metrics_data = [m.to_dict() for m in self.monitoring_manager.metrics] # Accessing internal list for demo
        return web.json_response(APIResponse.success_response(metrics_data))

    async def get_alerts(self, request: Request):
        """
        ---
        summary: Get active and resolved alerts
        tags:
          - Monitoring
        produces:
          - application/json
        responses:
          '200':
            description: List of alerts
        """
        alerts_data = [a.to_dict() for a in self.monitoring_manager.alerts] # Accessing internal list for demo
        return web.json_response(APIResponse.success_response(alerts_data))

    # Messaging Endpoint
    async def send_message(self, request: Request):
        """
        ---
        summary: Send a message from an external source to an agent
        tags:
          - Messaging
        consumes:
          - application/json
        parameters:
          - in: body
            name: message_details
            schema:
              type: object
              required:
                - receiver_id
                - message_type
                - content
              properties:
                receiver_id: {type: string}
                message_type: {type: string, enum: [command, request, response, event, heartbeat, error]}
                content: {type: object}
                sender_id: {type: string, default: "api_gateway"} # Identifier for the external sender
                trace_id: {type: string} # Optional for tracing conversations
        produces:
          - application/json
        responses:
          '200':
            description: Message sent successfully
          '400':
            description: Invalid input or receiver not found
        """
        try:
            data = await request.json()
            receiver_id = data.get('receiver_id')
            message_type_str = data.get('message_type')
            content = data.get('content')
            sender_id = data.get('sender_id', 'api_gateway')
            trace_id = data.get('trace_id')

            try:
                message_type = MessageType[message_type_str.upper()]
            except KeyError:
                return web.json_response(APIResponse.error_response(f"Invalid message type: {message_type_str}. Must be one of {list(MessageType.__members__.keys())}", code=400), status=400)

            # A simple way to simulate sending from an "agent" (the API)
            # In a real system, the API might expose a specific "gateway" agent ID
            # Or the framework itself sends it on behalf of the API.
            
            # Since AgentFramework doesn't have a direct "send_message" itself,
            # we can directly use the message_bus if the sender_id is 'api_gateway' or similar.
            # For a more secure approach, create a dedicated 'GatewayAgent' that sits between API and internal bus.

            message = AgentMessage(
                sender_id=sender_id,
                receiver_id=receiver_id,
                message_type=message_type,
                content=content,
                trace_id=trace_id
            )
            await self.framework.message_bus.send_message(message)
            return web.json_response(APIResponse.success_response({"message_id": message.id}, "Message sent successfully"))
        except Exception as e:
            self.logger.error(f"Error sending message: {e}", exc_info=True)
            return web.json_response(APIResponse.error_response(f"Internal server error: {e}", code=500), status=500)


async def api_demo():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("APIDemo")

    # Create a dummy config file for the framework
    from agentapi.config import create_sample_config_file
    sample_config_path = "api_framework_config.yaml"
    create_sample_config_file(sample_config_path)

    framework = AgentFramework(config_path=sample_config_path)
    api_server = FrameworkAPIServer(framework, host="localhost", port=8000)
    
    runner = None
    try:
        async with framework.lifespan(): # This will start the framework and its agents
            runner = await api_server.start()
            
            # Give framework time to start agents and register them
            await asyncio.sleep(2) 

            # Access initial agents for demo from the framework's internal _active_agents
            strategist = next((a for a in framework._active_agents.values() if a.name == "master_agent" or a.name == "strategist"), None)
            
            if strategist:
                logger.info(f"Strategist agent found: {strategist.id}")
                # You can now send messages to agents via the API or directly if needed for demo setup
                # For this demo, agents are already started by framework.lifespan()
            else:
                logger.warning("Strategist agent not found. API demo may not function fully.")

            logger.info(f"\n✅ API server running with {len(await framework.registry.list_all_agents())} agents")
            logger.info("\nAvailable endpoints:")
            logger.info("• POST /api/auth/login - Authentication")
            logger.info("• GET  /api/agents - List agents")
            logger.info("• POST /api/agents - Create agent")
            logger.info("• GET  /api/health - Health check")
            logger.info("• GET  /api/metrics - System metrics")
            logger.info("• POST /api/messages - Send message to agent")
            logger.info("\nTest the API (e.g., in another terminal):")
            logger.info("curl http://localhost:8000/api/health")
            logger.info("curl -X POST -H \"Content-Type: application/json\" -d '{\"receiver_id\": \"<agent_id_from_list_agents>\", \"message_type\": \"command\", \"content\": {\"action\": \"status\"}}' http://localhost:8000/api/messages")
            logger.info("\nPress Ctrl+C to stop...")
            
            # Keep server running until interrupted
            while True:
                await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("\n\n🛑 Stopping API server demo...")
    except Exception as e:
        logger.critical(f"An unhandled error occurred during API demo: {e}", exc_info=True)
    finally:
        if runner:
            await api_server.stop()
        # Clean up the sample config file
        if Path(sample_config_path).exists():
            os.remove(sample_config_path)
            logger.info(f"Cleaned up {sample_config_path}")
        logger.info("👋 API demo stopped.")

if __name__ == "__main__":
    asyncio.run(api_demo())