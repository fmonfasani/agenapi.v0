# web_dashboard.py - Dashboard web para monitoreo del framework de agentes

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path

import aiohttp
from aiohttp import web, WSMsgType
import aiohttp_cors

from core.autonomous_agent_framework import AgentFramework, AgentStatus
from framework_config_utils import MetricsCollector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for Dashboard Server
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8080
DASHBOARD_STATIC_DIR = "dashboard_static"
MONITORING_INTERVAL_SECONDS = 10
WEBSOCKET_RECONNECT_SECONDS = 5
MAX_LOG_ENTRIES = 10

class DashboardServer:
    """
    Web server for the agent framework monitoring dashboard.

    This server provides REST API endpoints for agent status, messages, resources,
    and metrics, as well as a WebSocket for real-time updates.
    It serves a static HTML dashboard interface.
    """

    def __init__(self, framework: AgentFramework, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        """
        Initializes the DashboardServer.

        Args:
            framework: The AgentFramework instance to monitor.
            host: The host address for the web server.
            port: The port for the web server.
        """
        self.framework = framework
        self.host = host
        self.port = port
        self.app = web.Application()
        self.websockets: List[web.WebSocketResponse] = []
        self.metrics_collector = MetricsCollector(framework)
        
        self._setup_routes()
        self._setup_cors()
        logging.info("DashboardServer initialized.")

    def _setup_routes(self):
        """Configures the server's API and static file routes."""
        logging.info("Setting up server routes.")
        # API Routes
        self.app.router.add_get("/api/status", self.get_status)
        self.app.router.add_get("/api/agents", self.get_agents)
        self.app.router.add_get("/api/agents/{agent_id}", self.get_agent_detail)
        self.app.router.add_post("/api/agents/{agent_id}/action", self.execute_agent_action)
        self.app.router.add_get("/api/messages", self.get_messages)
        self.app.router.add_get("/api/resources", self.get_resources)
        self.app.router.add_get("/api/metrics", self.get_metrics)
        
        # WebSocket for real-time updates
        self.app.router.add_get("/ws", self.websocket_handler)
        
        # Static files (HTML, CSS, JS)
        static_path = Path(DASHBOARD_STATIC_DIR)
        if not static_path.is_dir():
            logging.warning(f"Static directory '{DASHBOARD_STATIC_DIR}' not found. Dashboard HTML will not be served.")
            # Fallback for index if static dir is missing, though better to raise error or ensure dir exists
            self.app.router.add_get("/", self.index_fallback)
        else:
            self.app.router.add_get("/", self.index)
            self.app.router.add_static("/static/", path=static_path, name="static")
            logging.info(f"Serving static files from: {static_path.resolve()}")

    def _setup_cors(self):
        """Configures CORS for the application."""
        logging.info("Setting up CORS policies.")
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

    async def get_status(self, request: web.Request) -> web.Response:
        """Handles GET /api/status - Returns general framework status."""
        agents = self.framework.registry.list_all_agents()
        
        status_data = {
            "framework_status": "active" if agents else "idle",
            "total_agents": len(agents),
            "active_agents": len([a for a in agents if a.status == AgentStatus.ACTIVE]),
            "busy_agents": len([a for a in agents if a.status == AgentStatus.BUSY]),
            "error_agents": len([a for a in agents if a.status == AgentStatus.ERROR]),
            "uptime": str(datetime.now() - datetime.now().replace(hour=0, minute=0, second=0)),
            "timestamp": datetime.now().isoformat()
        }
        logging.debug(f"Status requested: {status_data}")
        return web.json_response(status_data)
        
    async def get_agents(self, request: web.Request) -> web.Response:
        """Handles GET /api/agents - Returns a list of all agents."""
        agents = self.framework.registry.list_all_agents()
        
        agents_data = []
        for agent in agents:
            agent_info = {
                "id": agent.id,
                "name": agent.name,
                "namespace": agent.namespace,
                "status": agent.status.value,
                "created_at": agent.created_at.isoformat(),
                "last_heartbeat": agent.last_heartbeat.isoformat(),
                "capabilities_count": len(agent.capabilities),
                "metadata": agent.metadata
            }
            agents_data.append(agent_info)
        
        logging.debug(f"Agents list requested: {len(agents_data)} agents.")
        return web.json_response({"agents": agents_data})
        
    async def get_agent_detail(self, request: web.Request) -> web.Response:
        """Handles GET /api/agents/{agent_id} - Returns details for a specific agent."""
        agent_id = request.match_info["agent_id"]
        agent = self.framework.registry.get_agent(agent_id)
        
        if not agent:
            logging.warning(f"Agent detail requested for non-existent agent: {agent_id}")
            return web.json_response({"error": "Agent not found"}, status=404)
            
        agent_resources = self.framework.resource_manager.find_resources_by_owner(agent_id)
        
        # In a real implementation, you would fetch recent messages from a persistence layer.
        recent_messages = [] 
        
        agent_detail = {
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
                    "created_at": res.created_at.isoformat()
                }
                for res in agent_resources
            ],
            "recent_messages": recent_messages
        }
        logging.debug(f"Agent detail requested for {agent_id}.")
        return web.json_response(agent_detail)
        
    async def execute_agent_action(self, request: web.Request) -> web.Response:
        """Handles POST /api/agents/{agent_id}/action - Executes an action on an agent."""
        agent_id = request.match_info["agent_id"]
        agent = self.framework.registry.get_agent(agent_id)
        
        if not agent:
            logging.warning(f"Action requested for non-existent agent: {agent_id}")
            return web.json_response({"error": "Agent not found"}, status=404)
            
        try:
            request_data = await request.json()
            action = request_data.get("action", "")
            params = request_data.get("params", {})
            
            logging.info(f"Executing action '{action}' on agent '{agent_id}' with params: {params}")
            result = await agent.execute_action(action, params)
            
            await self._broadcast_update({
                "type": "agent_action",
                "agent_id": agent_id,
                "action": action,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            
            return web.json_response({
                "success": True,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in action request for agent {agent_id}")
            return web.json_response({"error": "Invalid JSON payload", "success": False}, status=400)
        except Exception as e:
            logging.error(f"Error executing action on agent {agent_id}: {e}", exc_info=True)
            return web.json_response({
                "error": str(e),
                "success": False
            }, status=500)
            
    async def get_messages(self, request: web.Request) -> web.Response:
        """Handles GET /api/messages - Returns system messages."""
        # In a real implementation, this would retrieve messages from a persistent store.
        messages = [
            {
                "id": "msg_001",
                "sender": "strategist",
                "receiver": "generator",
                "action": "generate.code",
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            },
            {
                "id": "msg_002",
                "sender": "generator",
                "receiver": "test_generator",
                "action": "generate.tests",
                "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat(),
                "status": "completed"
            }
        ]
        logging.debug(f"Messages requested: {len(messages)} messages.")
        return web.json_response({"messages": messages})
        
    async def get_resources(self, request: web.Request) -> web.Response:
        """Handles GET /api/resources - Returns system resources."""
        all_resources = []
        agents = self.framework.registry.list_all_agents()
        
        for agent in agents:
            agent_resources = self.framework.resource_manager.find_resources_by_owner(agent.id)
            for resource in agent_resources:
                resource_info = {
                    "id": resource.id,
                    "name": resource.name,
                    "type": resource.type.value,
                    "namespace": resource.namespace,
                    "owner_agent_id": resource.owner_agent_id,
                    "created_at": resource.created_at.isoformat(),
                    "updated_at": resource.updated_at.isoformat(),
                    "data_size": len(str(resource.data))
                }
                all_resources.append(resource_info)
                
        logging.debug(f"Resources requested: {len(all_resources)} resources.")
        return web.json_response({"resources": all_resources})
        
    async def get_metrics(self, request: web.Request) -> web.Response:
        """Handles GET /api/metrics - Returns system metrics."""
        metrics = self.metrics_collector.get_metrics()
        
        agents = self.framework.registry.list_all_agents()
        status_counts = {}
        for agent in agents:
            status = agent.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
        metrics["agent_status_distribution"] = status_counts
        # This uptime calculation is reset daily, consider a more robust uptime tracking.
        metrics["framework_uptime"] = str(datetime.now() - datetime.now().replace(hour=0, minute=0, second=0))
        
        logging.debug("Metrics requested.")
        return web.json_response(metrics)
        
    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """Handles WebSocket connections for real-time updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websockets.append(ws)
        logging.info(f"WebSocket connected from {request.remote}. Total connections: {len(self.websockets)}")
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_websocket_message(ws, data)
                    except json.JSONDecodeError:
                        logging.warning(f"Received invalid JSON over WebSocket: {msg.data}")
                        await ws.send_str(json.dumps({"error": "Invalid JSON format"}))
                elif msg.type == WSMsgType.ERROR:
                    logging.error(f"WebSocket connection error: {ws.exception()}")
                    break
                elif msg.type == WSMsgType.CLOSE:
                    logging.info(f"WebSocket connection closed by client: {request.remote}")
                    break
        except Exception as e:
            logging.error(f"WebSocket handler unexpected error: {e}", exc_info=True)
        finally:
            # Clean up disconnected WebSocket
            if ws in self.websockets:
                self.websockets.remove(ws)
                logging.info(f"WebSocket disconnected from {request.remote}. Remaining connections: {len(self.websockets)}")
            await ws.close() # Ensure WebSocket is properly closed
            
        return ws
        
    async def _handle_websocket_message(self, ws: web.WebSocketResponse, data: Dict[str, Any]):
        """Processes messages received from a WebSocket client."""
        message_type = data.get("type", "")
        
        if message_type == "subscribe":
            await ws.send_str(json.dumps({
                "type": "subscription_confirmed",
                "timestamp": datetime.now().isoformat()
            }))
            logging.debug("WebSocket client subscribed to updates.")
            
        elif message_type == "get_status":
            status = await self._get_current_status()
            await ws.send_str(json.dumps({
                "type": "status_update",
                "data": status,
                "timestamp": datetime.now().isoformat()
            }))
            logging.debug("Sent current status via WebSocket.")
        else:
            logging.warning(f"Unknown WebSocket message type received: {message_type}")
            await ws.send_str(json.dumps({"error": f"Unknown message type: {message_type}"}))
            
    async def _broadcast_update(self, update_data: Dict[str, Any]):
        """Broadcasts an update message to all connected WebSockets."""
        if not self.websockets:
            logging.debug("No WebSockets connected to broadcast update.")
            return
            
        message = json.dumps(update_data)
        disconnected_websockets = []
        
        for ws in self.websockets:
            try:
                await ws.send_str(message)
            except Exception as e:
                logging.error(f"Failed to send WebSocket message to {ws.close_code}: {e}")
                disconnected_websockets.append(ws)
                
        # Remove disconnected WebSockets after iteration
        for ws in disconnected_websockets:
            if ws in self.websockets:
                self.websockets.remove(ws)
                logging.info(f"Removed disconnected WebSocket. Remaining: {len(self.websockets)}")
                
    async def _get_current_status(self) -> Dict[str, Any]:
        """Fetches the current status summary of the framework."""
        agents = self.framework.registry.list_all_agents()
        
        return {
            "total_agents": len(agents),
            "active_agents": len([a for a in agents if a.status == AgentStatus.ACTIVE]),
            "system_status": "active" if agents else "idle",
            "timestamp": datetime.now().isoformat()
        }
        
    async def index(self, request: web.Request) -> web.Response:
        """Serves the main dashboard HTML page from static files."""
        index_file_path = Path(DASHBOARD_STATIC_DIR) / "index.html"
        if not index_file_path.is_file():
            logging.error(f"Dashboard index.html not found at: {index_file_path}")
            return web.Response(text="Dashboard HTML not found.", status=500)
            
        logging.info(f"Serving dashboard HTML from: {index_file_path}")
        return web.FileResponse(index_file_path)

    async def index_fallback(self, request: web.Request) -> web.Response:
        """Fallback for serving index if static directory is not configured."""
        logging.warning("Serving minimal fallback HTML as static directory is not properly set up.")
        return web.Response(text="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Framework Dashboard (Fallback)</title>
</head>
<body>
    <h1>Agent Framework Dashboard</h1>
    <p>The full dashboard HTML/CSS/JS is expected to be in the 'dashboard_static' directory.</p>
    <p>Please ensure the 'dashboard_static' directory exists and contains 'index.html'.</p>
    <p>Current time: <span id="current-time"></span></p>
    <script>
        document.getElementById('current-time').textContent = new Date().toLocaleString();
    </script>
</body>
</html>
        """, content_type="text/html")
            
    async def start_monitoring_loop(self):
        """Starts a background task for continuous monitoring and broadcasting updates."""
        logging.info("Starting background monitoring loop.")
        asyncio.create_task(self._monitoring_loop())
        
    async def _monitoring_loop(self):
        """Continuous loop for collecting and broadcasting framework status."""
        while True:
            try:
                status = await self._get_current_status()
                await self._broadcast_update({
                    "type": "status_update",
                    "data": status,
                    "timestamp": datetime.now().isoformat()
                })
                await asyncio.sleep(MONITORING_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                logging.info("Monitoring loop cancelled.")
                break
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}", exc_info=True)
                await asyncio.sleep(WEBSOCKET_RECONNECT_SECONDS) # Wait before retrying on error
                
    async def start(self) -> web.AppRunner:
        """Starts the dashboard server and its monitoring loop."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        await self.start_monitoring_loop()
        
        logging.info(f"Dashboard server started at http://{self.host}:{self.port}")
        print(f"🌐 Dashboard available at: http://{self.host}:{self.port}")
        
        return runner

    async def stop(self, runner: web.AppRunner):
        """Stops the dashboard server."""
        logging.info("Stopping DashboardServer.")
        if runner:
            await runner.cleanup()
        # Optionally, explicitly close all websockets
        for ws in list(self.websockets):
            await ws.close()
        self.websockets.clear()
        logging.info("DashboardServer stopped.")

# DASHBOARD INTEGRATION

class DashboardIntegration:
    """Provides an integration layer between the AgentFramework and the DashboardServer."""
    
    def __init__(self, framework: AgentFramework):
        """
        Initializes the DashboardIntegration.

        Args:
            framework: The AgentFramework instance to integrate with.
        """
        self.framework = framework
        self.dashboard_server: DashboardServer | None = None
        self.runner: web.AppRunner | None = None
        
    async def start_dashboard(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> web.AppRunner:
        """Starts the integrated dashboard."""
        logging.info(f"Starting dashboard integration on {host}:{port}")
        self.dashboard_server = DashboardServer(self.framework, host, port)
        self.runner = await self.dashboard_server.start()
        return self.runner
        
    async def stop_dashboard(self):
        """Stops the integrated dashboard."""
        logging.info("Stopping dashboard integration.")
        if self.dashboard_server and self.runner:
            await self.dashboard_server.stop(self.runner)
            self.dashboard_server = None
            self.runner = None
        else:
            logging.info("Dashboard server not running or already stopped.")


# EXAMPLE USAGE

async def dashboard_demo():
    """Demonstrates the web dashboard functionality."""
    logging.info("Starting dashboard demo.")
    
    # Imports for demo purposes
    from core.autonomous_agent_framework import AgentFramework, AgentResource, ResourceType
    from core.specialized_agents import ExtendedAgentFactory # Assuming this path is correct
    
    framework = AgentFramework()
    await framework.start()
    
    # Create some agents to display in the dashboard
    strategist = ExtendedAgentFactory.create_agent("agent.planning.strategist", "strategist", framework)
    generator = ExtendedAgentFactory.create_agent("agent.build.code.generator", "generator", framework)
    tester = ExtendedAgentFactory.create_agent("agent.test.generator", "tester", framework)
    
    await strategist.start()
    await generator.start()
    await tester.start()
    
    dashboard_integration = DashboardIntegration(framework)
    runner = await dashboard_integration.start_dashboard(host=DEFAULT_HOST, port=DEFAULT_PORT)
    
    print("\n" + "="*60)
    print("🌐 DASHBOARD DEMO STARTED")
    print("="*60)
    print(f"Dashboard URL: http://{DEFAULT_HOST}:{DEFAULT_PORT}")
    print(f"Active agents: {len(framework.registry.list_all_agents())}")
    print("\nFeatures available:")
    print("• Real-time agent monitoring")
    print("• System metrics and statistics")
    print("• WebSocket live updates")
    print("• Agent status visualization")
    print("• Activity logging")
    print("\nPress Ctrl+C to stop...")
    
    try:
        while True:
            await asyncio.sleep(10)
            
            # Simulate message exchange between agents
            await strategist.send_message(
                generator.id,
                "action.generate.component",
                {"specification": {"name": "DemoComponent"}}
            )
            
            await asyncio.sleep(5)
            
            await generator.send_message(
                tester.id,
                "action.generate.tests",
                {"code": "demo code"}
            )
            
            # Create resources occasionally
            demo_resource = AgentResource(
                type=ResourceType.CODE,
                name=f"demo_resource_{datetime.now().strftime('%H%M%S')}",
                namespace="resource.demo",
                data={"content": "Demo resource data"},
                owner_agent_id=strategist.id
            )
            await framework.resource_manager.create_resource(demo_resource)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping dashboard demo...")
        logging.info("KeyboardInterrupt received. Stopping demo.")
    finally:
        await framework.stop()
        await dashboard_integration.stop_dashboard() # Use integration to stop
        print("👋 Dashboard demo stopped")

if __name__ == "__main__":
    # Ensure the static directory exists for the dashboard HTML
    Path(DASHBOARD_STATIC_DIR).mkdir(exist_ok=True)
    
    # Create a dummy index.html for demonstration if it doesn't exist
    index_html_path = Path(DASHBOARD_STATIC_DIR) / "index.html"
    if not index_html_path.exists():
        with open(index_html_path, "w") as f:
            f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Framework Dashboard</title>
    <style>
        /* Minimal CSS for demonstration. Full CSS would be in a separate file. */
        body { font-family: sans-serif; margin: 20px; background-color: #f4f7f6; color: #333; }
        .dashboard-container { max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1, h2 { color: #2c3e50; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .stat-card { background: #e9f7ef; padding: 15px; border-radius: 5px; text-align: center; }
        .stat-card h3 { color: #28a745; margin-bottom: 5px; }
        .stat-value { font-size: 2em; font-weight: bold; color: #2c3e50; }
        .card { background: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 15px; }
        .agent-item { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #eee; }
        .agent-item:last-child { border-bottom: none; }
        .agent-status { padding: 4px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold; }
        .status-active { background-color: #d4edda; color: #155724; }
        .status-busy { background-color: #fff3cd; color: #856404; }
        .status-idle { background-color: #d1ecf1; color: #0c5460; }
        .status-error { background-color: #f8d7da; color: #721c24; }
        .log-entry { background: #f8f8f8; border-left: 4px solid #6c757d; padding: 8px; margin-bottom: 5px; border-radius: 3px; }
        .log-timestamp { font-size: 0.8em; color: #666; text-align: right; }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <h1>Agent Framework Dashboard</h1>
        <p>System Status: <span id="system-status">Loading...</span> | Last Update: <span id="last-update">--</span></p>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Agents</h3>
                <div class="stat-value" id="total-agents">0</div>
            </div>
            <div class="stat-card">
                <h3>Active Agents</h3>
                <div class="stat-value" id="active-agents">0</div>
            </div>
            <div class="stat-card">
                <h3>Messages Sent (Mock)</h3>
                <div class="stat-value" id="messages-sent">0</div>
            </div>
            <div class="stat-card">
                <h3>Resources Created (Mock)</h3>
                <div class="stat-value" id="resources-created">0</div>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div class="card">
                <h2>Active Agents</h2>
                <div id="agent-list">
                    Loading agents...
                </div>
            </div>
            <div class="card">
                <h2>Activity Log</h2>
                <div id="activity-log">
                    Loading activity...
                </div>
            </div>
        </div>
    </div>

    <script>
        let websocket = null;
        const WS_RECONNECT_INTERVAL = 5000; // milliseconds
        const DATA_REFRESH_INTERVAL = 30000; // milliseconds
        const MAX_LOG_ENTRIES = 10;

        document.addEventListener('DOMContentLoaded', function() {
            initWebSocket();
            loadInitialData();
            setInterval(loadInitialData, DATA_REFRESH_INTERVAL);
        });

        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            websocket = new WebSocket(wsUrl);
            
            websocket.onopen = function() {
                console.log('WebSocket connected');
                addLogEntry('WebSocket connected', 'info');
                websocket.send(JSON.stringify({type: 'subscribe'}));
            };
            
            websocket.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            websocket.onclose = function() {
                console.log('WebSocket disconnected. Reconnecting...');
                addLogEntry('WebSocket disconnected, attempting reconnect...', 'warning');
                setTimeout(initWebSocket, WS_RECONNECT_INTERVAL);
            };
            
            websocket.onerror = function(error) {
                console.error('WebSocket error:', error);
                addLogEntry('WebSocket error', 'error');
                websocket.close(); // Force close to trigger onclose and reconnect
            };
        }
        
        function handleWebSocketMessage(data) {
            switch(data.type) {
                case 'agent_action':
                    addLogEntry(`Agent ${data.agent_id} executed ${data.action}`, 'info');
                    loadInitialData(); // Refresh agents and stats after an action
                    break;
                case 'status_update':
                    updateStats(data.data);
                    document.getElementById('last-update').textContent = new Date(data.timestamp).toLocaleTimeString();
                    break;
                case 'subscription_confirmed':
                    addLogEntry('Subscribed to real-time updates', 'success');
                    break;
                default:
                    console.log('Received unknown WebSocket message type:', data.type, data);
            }
        }
        
        async function loadInitialData() {
            try {
                const agentsResponse = await fetch('/api/agents');
                const agentsData = await agentsResponse.json();
                updateAgentsList(agentsData.agents);
                
                const metricsResponse = await fetch('/api/metrics');
                const metricsData = await metricsResponse.json();
                updateStats(metricsData);
                
                // You can add more initial data fetching here (e.g., messages, resources)
                // For this refactor, we focus on agents and metrics as they are most dynamic

            } catch (error) {
                console.error('Failed to load initial dashboard data:', error);
                addLogEntry('Failed to load dashboard data. Check server logs.', 'error');
            }
        }
        
        function updateStats(data) {
            document.getElementById('total-agents').textContent = data.total_agents || 0;
            document.getElementById('active-agents').textContent = data.active_agents || 0;
            // Mock values as actual counts might need more complex aggregation from messages/resources
            document.getElementById('messages-sent').textContent = data.messages_sent || 0; 
            document.getElementById('resources-created').textContent = data.resources_created || 0;
            document.getElementById('system-status').textContent = data.system_status || 'Unknown';
            const statusIndicator = document.querySelector('.status-indicator');
            if (statusIndicator) {
                statusIndicator.className = `status-indicator status-${data.system_status}`;
            }
        }
        
        function updateAgentsList(agents) {
            const agentList = document.getElementById('agent-list');
            agentList.innerHTML = ''; // Clear previous list
            
            if (!agents || agents.length === 0) {
                agentList.innerHTML = '<p>No agents currently active.</p>';
                return;
            }
            
            agents.forEach(agent => {
                const agentItem = document.createElement('div');
                agentItem.className = 'agent-item';
                agentItem.innerHTML = `
                    <div class="agent-info">
                        <h4>${agent.name} (${agent.id.substring(0, 8)}...)</h4>
                        <div class="agent-namespace">${agent.namespace}</div>
                    </div>
                    <div class="agent-status status-${agent.status.toLowerCase()}">${agent.status}</div>
                `;
                agentList.appendChild(agentItem);
            });
        }
        
        function addLogEntry(message, type = 'info') {
            const logContainer = document.getElementById('activity-log');
            const logEntry = document.createElement('div');
            logEntry.className = `log-entry log-${type}`; // Add type class for potential styling
            
            const timestamp = new Date().toLocaleTimeString();
            logEntry.innerHTML = `
                <div>${message}</div>
                <div class="log-timestamp">${timestamp}</div>
            `;
            
            // Add new entry to the top
            if (logContainer.firstChild) {
                logContainer.insertBefore(logEntry, logContainer.firstChild);
            } else {
                logContainer.appendChild(logEntry);
            }
            
            // Keep only the last N entries
            while (logContainer.children.length > MAX_LOG_ENTRIES) {
                logContainer.removeChild(logContainer.lastChild);
            }
        }
    </script>
</body>
</html>
            """)
        logging.info(f"Created dummy {index_html_path} for demo.")

    asyncio.run(dashboard_demo())