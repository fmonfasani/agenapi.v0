# interfaces/web_dashboard.py

import asyncio
import json
import logging
import os # Added for file cleanup
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import aiohttp
from aiohttp import web, WSMsgType
import aiohttp_cors
import weakref

# Importaciones actualizadas para AgentStatus, MessageType, AgentResource, ResourceType
from autonomous_agent_framework import AgentFramework # Corrected import path
from agentapi.interfaces.agent_interfaces import BaseAgent # For type hints
from agentapi.models.agent_models import AgentStatus, MessageType, AgentResource, ResourceType # Corrected import paths
from framework.monitoring_manager import MonitoringManager # Using the new MonitoringManager

# ================================
# DASHBOARD SERVER
# ================================

class DashboardServer:
    """Servidor web para el dashboard de monitoreo"""
    
    def __init__(self, framework: AgentFramework, host: str = "0.0.0.0", port: int = 8080):
        self.framework = framework
        self.host = host
        self.port = port
        self.app = web.Application()
        self.websockets: List[web.WebSocketResponse] = []
        self.monitoring_manager = framework.monitoring_manager # Access through framework
        self.dashboard_data = {
            "agents": {},
            "messages": [], # Realistically, this would be a limited buffer or pulled from persistence
            "resources": {},
            "metrics": {},
            "system_status": "initializing"
        }
        self.logger = logging.getLogger("DashboardServer")
        self._setup_routes()
        self._setup_cors()
        self._data_update_task: Optional[asyncio.Task] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None


    def _setup_routes(self):
        self.app.router.add_get('/', self.index_handler)
        self.app.router.add_get('/ws', self.websocket_handler)
        self.app.router.add_get('/api/dashboard_data', self.get_dashboard_data_http) # For initial load
        self.logger.info("Dashboard routes set up.")

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

    async def start(self):
        """Starts the aiohttp web server and background data update task."""
        self.logger.info(f"Starting Dashboard server on {self.host}:{self.port}...")
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        
        # Start background task to update dashboard data and push to websockets
        self._data_update_task = asyncio.create_task(self._update_dashboard_data_loop())

        self.logger.info("Dashboard server started.")

    async def stop(self):
        """Stops the aiohttp web server and cleans up resources."""
        self.logger.info("Stopping Dashboard server...")
        
        if self._data_update_task:
            self._data_update_task.cancel()
            try: await self._data_update_task
            except asyncio.CancelledError: pass
            self.logger.info("Dashboard data update task cancelled.")

        # Close all active websockets
        for ws in list(self.websockets):
            if not ws.closed:
                await ws.close(code=1000, message='Server shutting down')
        self.websockets.clear()
        self.logger.info(f"Closed {len(self.websockets)} active websockets.")

        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        
        self.logger.info("Dashboard server stopped.")

    async def _update_dashboard_data_loop(self):
        """Periodically updates dashboard data and sends it to connected websockets."""
        while True:
            try:
                await self._update_dashboard_data()
                await self._send_data_to_websockets()
                await asyncio.sleep(self.framework.config.monitoring.metrics_collection_interval) # Use config interval
            except asyncio.CancelledError:
                self.logger.info("Dashboard data update loop cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in dashboard data update loop: {e}", exc_info=True)
                await asyncio.sleep(5) # Avoid rapid error loop

    async def _update_dashboard_data(self):
        """Collects current data from the framework's managers."""
        # Agents data
        agents_info = await self.framework.registry.list_all_agents()
        self.dashboard_data["agents"] = {a.id: a.to_dict() for a in agents_info}
        self.dashboard_data["system_status"] = "active" if self.framework._running else "inactive"

        # Metrics (get latest from monitoring manager)
        # Assuming monitoring_manager keeps a recent history of metrics
        metrics_from_monitor = self.monitoring_manager.metrics # This is a simple list for demo
        latest_metrics_dict = {}
        for metric in metrics_from_monitor:
            latest_metrics_dict[metric.name] = metric.to_dict()
        self.dashboard_data["metrics"] = latest_metrics_dict

        # Alerts
        alerts_from_monitor = self.monitoring_manager.alerts
        self.dashboard_data["alerts"] = [a.to_dict() for a in alerts_from_monitor if a.status == AlertStatus.ACTIVE]

        # Resources (if ResourceManager had a list_all method)
        # self.dashboard_data["resources"] = {r.id: r.to_dict() for r in await self.framework.resource_manager.list_all_resources()}
        self.dashboard_data["resources"] = {} # Placeholder

        # Message history (if message bus stored some)
        # self.dashboard_data["messages"] = [m.to_dict() for m in self.framework.message_bus.get_recent_messages()]
        self.dashboard_data["messages"] = [] # Placeholder

    async def _send_data_to_websockets(self):
        """Sends the current dashboard data to all connected WebSocket clients."""
        if not self.websockets:
            return

        json_data = json.dumps(self.dashboard_data, default=str) # Handle datetime objects for JSON serialization
        
        closed_websockets = []
        for ws in self.websockets:
            if ws.closed:
                closed_websockets.append(ws)
                continue
            try:
                await ws.send_str(json_data)
            except Exception as e:
                self.logger.error(f"Failed to send data to websocket: {e}", exc_info=True)
                closed_websockets.append(ws)
        
        # Clean up closed websockets
        for ws in closed_websockets:
            self.websockets.remove(ws)


    # HTTP Endpoint for initial data fetch (e.g., if JS client connects later)
    async def get_dashboard_data_http(self, request: web.Request):
        await self._update_dashboard_data() # Ensure data is fresh
        return web.json_response(self.dashboard_data, default=str)

    # WebSocket Handler
    async def websocket_handler(self, request: web.Request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.websockets.append(ws)
        self.logger.info(f"WebSocket client connected from {request.remote}. Total: {len(self.websockets)}")

        # Send initial data immediately
        await self._update_dashboard_data()
        await ws.send_str(json.dumps(self.dashboard_data, default=str))

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    # Handle incoming messages from dashboard (e.g., commands, filters)
                    self.logger.info(f"Received WS message: {msg.data}")
                    # Example: if msg.data == 'ping': await ws.send_str('pong')
                elif msg.type == WSMsgType.ERROR:
                    self.logger.error(f"WebSocket connection closed with exception: {ws.exception()}")
        except asyncio.CancelledError:
            self.logger.info("WebSocket handler cancelled.")
        finally:
            self.websockets.remove(ws)
            self.logger.info(f"WebSocket client disconnected. Total: {len(self.websockets)}")


async def dashboard_demo():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("DashboardDemo")

    # Create a dummy config file for the framework
    from agentapi.config import create_sample_config_file
    sample_config_path = "dashboard_framework_config.yaml"
    create_sample_config_file(sample_config_path)

    framework = AgentFramework(config_path=sample_config_path)
    dashboard_server = DashboardServer(framework, host="localhost", port=8080)

    try:
        async with framework.lifespan(): # This starts the framework and its agents
            await dashboard_server.start()

            logger.info(f"\n✅ Dashboard server running at http://localhost:8080")
            logger.info("\nOpen your browser to http://localhost:8080 to see the dashboard.")
            logger.info("Press Ctrl+C to stop...")

            # Simple HTML page for demonstration
            index_html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Autonomous Agent Framework Dashboard</title>
                <style>
                    body { font-family: sans-serif; margin: 20px; background-color: #f0f2f5; color: #333; }
                    .container { max-width: 1200px; margin: auto; background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    h1 { color: #0056b3; }
                    h2 { color: #007bff; border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 20px; }
                    pre { background-color: #e9ecef; padding: 10px; border-radius: 4px; overflow-x: auto; }
                    .status-active { color: green; font-weight: bold; }
                    .status-error { color: red; font-weight: bold; }
                    .status-terminated { color: gray; }
                    .metric-item { margin-bottom: 5px; }
                    .alert-info { color: blue; }
                    .alert-warning { color: orange; }
                    .alert-critical { color: red; font-weight: bold; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Autonomous Agent Framework Dashboard</h1>

                    <h2>System Status: <span id="system-status">Loading...</span></h2>

                    <h2>Agents (<span id="agent-count">0</span>)</h2>
                    <div id="agents-list">Loading agents...</div>

                    <h2>Metrics</h2>
                    <div id="metrics-list">Loading metrics...</div>

                    <h2>Alerts</h2>
                    <div id="alerts-list">No alerts.</div>
                </div>

                <script>
                    const statusSpan = document.getElementById('system-status');
                    const agentCountSpan = document.getElementById('agent-count');
                    const agentsListDiv = document.getElementById('agents-list');
                    const metricsListDiv = document.getElementById('metrics-list');
                    const alertsListDiv = document.getElementById('alerts-list');

                    let ws;

                    function connectWebSocket() {
                        ws = new WebSocket('ws://localhost:8080/ws');

                        ws.onopen = function() {
                            console.log('WebSocket connection opened.');
                        };

                        ws.onmessage = function(event) {
                            const data = JSON.parse(event.data);
                            console.log('Received data:', data);
                            updateDashboard(data);
                        };

                        ws.onclose = function() {
                            console.log('WebSocket connection closed. Reconnecting in 5 seconds...');
                            setTimeout(connectWebSocket, 5000);
                        };

                        ws.onerror = function(err) {
                            console.error('WebSocket error:', err);
                            ws.close();
                        };
                    }

                    function updateDashboard(data) {
                        // System Status
                        statusSpan.textContent = data.system_status.toUpperCase();
                        statusSpan.className = 'status-' + data.system_status;

                        // Agents
                        const agents = Object.values(data.agents);
                        agentCountSpan.textContent = agents.length;
                        agentsListDiv.innerHTML = agents.length === 0 ? 'No agents registered.' : '';
                        agents.forEach(agent => {
                            const agentDiv = document.createElement('div');
                            agentDiv.innerHTML = `
                                <strong>${agent.namespace}.${agent.name}</strong> (ID: ${agent.id.substring(0, 8)}...) - 
                                <span class="status-${agent.status.toLowerCase()}">${agent.status}</span>
                                <br>Last Heartbeat: ${new Date(agent.last_heartbeat).toLocaleTimeString()}
                            `;
                            agentsListDiv.appendChild(agentDiv);
                        });

                        // Metrics
                        metricsListDiv.innerHTML = '';
                        const metrics = Object.values(data.metrics).sort((a,b) => a.name.localeCompare(b.name));
                        if (metrics.length === 0) {
                            metricsListDiv.innerHTML = 'No metrics collected yet.';
                        } else {
                            metrics.forEach(metric => {
                                const metricDiv = document.createElement('div');
                                metricDiv.className = 'metric-item';
                                metricDiv.textContent = `${metric.name}: ${metric.value.toFixed(2)} ${metric.unit} (Tags: ${JSON.stringify(metric.tags)})`;
                                metricsListDiv.appendChild(metricDiv);
                            });
                        }

                        // Alerts
                        alertsListDiv.innerHTML = '';
                        if (data.alerts.length === 0) {
                            alertsListDiv.innerHTML = 'No active alerts.';
                        } else {
                            data.alerts.forEach(alert => {
                                const alertDiv = document.createElement('div');
                                alertDiv.className = `alert-${alert.severity.toLowerCase()}`;
                                alertDiv.innerHTML = `
                                    <strong>${alert.severity.toUpperCase()} ALERT:</strong> ${alert.message} 
                                    (Rule: ${alert.rule_name}, Time: ${new Date(alert.timestamp).toLocaleTimeString()})
                                `;
                                alertsListDiv.appendChild(alertDiv);
                            });
                        }
                    }

                    // Initial connection
                    connectWebSocket();
                </script>
            </body>
            </html>
            """
            # Serve the HTML content (in a real app, this would be from a static file)
            dashboard_server.app.router.add_get('/', lambda r: web.Response(text=index_html_content, content_type='text/html'))


            # Keep server running until interrupted
            while True:
                await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("\n\n🛑 Stopping dashboard demo...")
        
    except Exception as e:
        logger.critical(f"An unhandled error occurred during dashboard demo: {e}", exc_info=True)
        
    finally:
        # Stop dashboard server and framework
        if dashboard_server.runner: # Check if runner was successfully started
            await dashboard_server.stop()
        if framework._running: # Ensure framework is stopped if it's still running
             await framework.stop()

        # Clean up the sample config file
        if Path(sample_config_path).exists():
            os.remove(sample_config_path)
            logger.info(f"Cleaned up {sample_config_path}")
        logger.info("👋 Dashboard demo stopped")

if __name__ == "__main__":
    asyncio.run(dashboard_demo())