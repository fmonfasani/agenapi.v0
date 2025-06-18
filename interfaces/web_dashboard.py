"""
web_dashboard.py - Dashboard web para monitoreo del framework de agentes
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import aiohttp
from aiohttp import web, WSMsgType
import aiohttp_cors
import weakref

from core.autonomous_agent_framework import AgentFramework, BaseAgent, AgentStatus, MessageType
from systems.framework_config_utils import MetricsCollector

# ================================
# DASHBOARD SERVER
# ================================

class DashboardServer:
    """Servidor web para el dashboard de monitoreo"""
    
    def __init__(self, framework: AgentFramework, host: str = "localhost", port: int = 8080):
        self.framework = framework
        self.host = host
        self.port = port
        self.app = web.Application()
        self.websockets: List[web.WebSocketResponse] = []
        self.metrics_collector = MetricsCollector(framework)
        self.dashboard_data = {
            "agents": {},
            "messages": [],
            "resources": {},
            "metrics": {},
            "system_status": "active"
        }
        
        # Configurar rutas
        self._setup_routes()
        
        # Configurar CORS
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
            
    def _setup_routes(self):
        """Configurar rutas del servidor"""
        
        # Rutas de API
        self.app.router.add_get("/api/status", self.get_status)
        self.app.router.add_get("/api/agents", self.get_agents)
        self.app.router.add_get("/api/agents/{agent_id}", self.get_agent_detail)
        self.app.router.add_post("/api/agents/{agent_id}/action", self.execute_agent_action)
        self.app.router.add_get("/api/messages", self.get_messages)
        self.app.router.add_get("/api/resources", self.get_resources)
        self.app.router.add_get("/api/metrics", self.get_metrics)
        
        # WebSocket para actualizaciones en tiempo real
        self.app.router.add_get("/ws", self.websocket_handler)
        
        # Archivos estáticos (HTML, CSS, JS)
        self.app.router.add_get("/", self.index)
        self.app.router.add_static("/static/", path="dashboard_static/", name="static")
        
    async def get_status(self, request):
        """Obtener estado general del framework"""
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
        
        return web.json_response(status_data)
        
    async def get_agents(self, request):
        """Obtener lista de todos los agentes"""
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
            
        return web.json_response({"agents": agents_data})
        
    async def get_agent_detail(self, request):
        """Obtener detalles de un agente específico"""
        agent_id = request.match_info["agent_id"]
        agent = self.framework.registry.get_agent(agent_id)
        
        if not agent:
            return web.json_response({"error": "Agent not found"}, status=404)
            
        # Obtener recursos del agente
        agent_resources = self.framework.resource_manager.find_resources_by_owner(agent_id)
        
        # Obtener mensajes recientes (simulado)
        recent_messages = []  # En una implementación real, obtendrías del sistema de persistencia
        
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
        
        return web.json_response(agent_detail)
        
    async def execute_agent_action(self, request):
        """Ejecutar acción en un agente"""
        agent_id = request.match_info["agent_id"]
        agent = self.framework.registry.get_agent(agent_id)
        
        if not agent:
            return web.json_response({"error": "Agent not found"}, status=404)
            
        try:
            request_data = await request.json()
            action = request_data.get("action", "")
            params = request_data.get("params", {})
            
            # Ejecutar acción
            result = await agent.execute_action(action, params)
            
            # Broadcast actualización vía WebSocket
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
            
        except Exception as e:
            return web.json_response({
                "error": str(e),
                "success": False
            }, status=500)
            
    async def get_messages(self, request):
        """Obtener mensajes del sistema"""
        # En una implementación real, obtendrías de persistencia
        # Por ahora, simulamos algunos mensajes
        
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
        
        return web.json_response({"messages": messages})
        
    async def get_resources(self, request):
        """Obtener recursos del sistema"""
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
                
        return web.json_response({"resources": all_resources})
        
    async def get_metrics(self, request):
        """Obtener métricas del sistema"""
        metrics = self.metrics_collector.get_metrics()
        
        # Añadir métricas adicionales
        agents = self.framework.registry.list_all_agents()
        status_counts = {}
        for agent in agents:
            status = agent.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
        metrics["agent_status_distribution"] = status_counts
        metrics["framework_uptime"] = str(datetime.now() - datetime.now().replace(hour=0, minute=0, second=0))
        
        return web.json_response(metrics)
        
    async def websocket_handler(self, request):
        """Manejar conexiones WebSocket"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websockets.append(ws)
        logging.info(f"WebSocket connected: {len(self.websockets)} total connections")
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    # Manejar mensajes del cliente
                    try:
                        data = json.loads(msg.data)
                        await self._handle_websocket_message(ws, data)
                    except json.JSONDecodeError:
                        await ws.send_str(json.dumps({"error": "Invalid JSON"}))
                elif msg.type == WSMsgType.ERROR:
                    logging.error(f"WebSocket error: {ws.exception()}")
                    break
        except Exception as e:
            logging.error(f"WebSocket handler error: {e}")
        finally:
            if ws in self.websockets:
                self.websockets.remove(ws)
                
        return ws
        
    async def _handle_websocket_message(self, ws: web.WebSocketResponse, data: Dict[str, Any]):
        """Manejar mensajes del WebSocket"""
        message_type = data.get("type", "")
        
        if message_type == "subscribe":
            # Cliente se suscribe a actualizaciones
            await ws.send_str(json.dumps({
                "type": "subscription_confirmed",
                "timestamp": datetime.now().isoformat()
            }))
            
        elif message_type == "get_status":
            # Enviar estado actual
            status = await self._get_current_status()
            await ws.send_str(json.dumps({
                "type": "status_update",
                "data": status,
                "timestamp": datetime.now().isoformat()
            }))
            
    async def _broadcast_update(self, update_data: Dict[str, Any]):
        """Enviar actualización a todos los WebSockets conectados"""
        if not self.websockets:
            return
            
        message = json.dumps(update_data)
        disconnected = []
        
        for ws in self.websockets:
            try:
                await ws.send_str(message)
            except Exception as e:
                logging.error(f"Failed to send WebSocket message: {e}")
                disconnected.append(ws)
                
        # Remover conexiones desconectadas
        for ws in disconnected:
            if ws in self.websockets:
                self.websockets.remove(ws)
                
    async def _get_current_status(self) -> Dict[str, Any]:
        """Obtener estado actual del framework"""
        agents = self.framework.registry.list_all_agents()
        
        return {
            "total_agents": len(agents),
            "active_agents": len([a for a in agents if a.status == AgentStatus.ACTIVE]),
            "system_status": "active" if agents else "idle",
            "timestamp": datetime.now().isoformat()
        }
        
    async def index(self, request):
        """Servir página principal del dashboard"""
        html_content = self._generate_dashboard_html()
        return web.Response(text=html_content, content_type="text/html")
        
    def _generate_dashboard_html(self) -> str:
        """Generar HTML del dashboard"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Framework Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .dashboard {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .header h1 {
            color: #4a5568;
            margin-bottom: 10px;
            font-size: 2.5rem;
            font-weight: 700;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-active { background: #48bb78; }
        .status-idle { background: #ed8936; }
        .status-error { background: #f56565; }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-card h3 {
            color: #4a5568;
            margin-bottom: 10px;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stat-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2d3748;
        }
        
        .content-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .card h2 {
            color: #4a5568;
            margin-bottom: 15px;
            font-size: 1.3rem;
        }
        
        .agent-list {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .agent-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px;
            border-bottom: 1px solid #e2e8f0;
            transition: background 0.2s ease;
        }
        
        .agent-item:hover {
            background: rgba(99, 102, 241, 0.1);
        }
        
        .agent-info h4 {
            color: #2d3748;
            margin-bottom: 4px;
        }
        
        .agent-namespace {
            color: #718096;
            font-size: 0.85rem;
        }
        
        .agent-status {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .status-active { background: #c6f6d5; color: #22543d; }
        .status-busy { background: #fed7d7; color: #742a2a; }
        .status-idle { background: #feebc8; color: #7b341e; }
        .status-error { background: #fed7d7; color: #742a2a; }
        
        .log-entry {
            padding: 8px;
            border-left: 3px solid #6366f1;
            margin-bottom: 8px;
            background: #f8fafc;
            border-radius: 0 8px 8px 0;
        }
        
        .log-timestamp {
            color: #718096;
            font-size: 0.8rem;
        }
        
        .refresh-btn {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .refresh-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
        }
        
        @media (max-width: 768px) {
            .content-grid {
                grid-template-columns: 1fr;
            }
            
            .stats-grid {
                grid-template-columns: 1fr 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>🤖 Agent Framework Dashboard</h1>
            <p>
                <span class="status-indicator status-active"></span>
                System Status: <span id="system-status">Active</span> | 
                Last Update: <span id="last-update">--</span>
            </p>
        </div>
        
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
                <h3>Messages Sent</h3>
                <div class="stat-value" id="messages-sent">0</div>
            </div>
            <div class="stat-card">
                <h3>Resources Created</h3>
                <div class="stat-value" id="resources-created">0</div>
            </div>
        </div>
        
        <div class="content-grid">
            <div class="card">
                <h2>🤖 Active Agents</h2>
                <button class="refresh-btn" onclick="refreshAgents()">Refresh</button>
                <div class="agent-list" id="agent-list">
                    <!-- Agents will be loaded here -->
                </div>
            </div>
            
            <div class="card">
                <h2>📋 Activity Log</h2>
                <div id="activity-log">
                    <!-- Activity will be loaded here -->
                </div>
            </div>
        </div>
    </div>

    <script>
        let websocket = null;
        let isConnected = false;
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initWebSocket();
            loadInitialData();
            
            // Auto refresh every 30 seconds
            setInterval(loadInitialData, 30000);
        });
        
        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            websocket = new WebSocket(wsUrl);
            
            websocket.onopen = function() {
                isConnected = true;
                console.log('WebSocket connected');
                addLogEntry('WebSocket connected', 'info');
                
                // Subscribe to updates
                websocket.send(JSON.stringify({type: 'subscribe'}));
            };
            
            websocket.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            websocket.onclose = function() {
                isConnected = false;
                console.log('WebSocket disconnected');
                addLogEntry('WebSocket disconnected', 'warning');
                
                // Reconnect after 5 seconds
                setTimeout(initWebSocket, 5000);
            };
            
            websocket.onerror = function(error) {
                console.error('WebSocket error:', error);
                addLogEntry('WebSocket error', 'error');
            };
        }
        
        function handleWebSocketMessage(data) {
            switch(data.type) {
                case 'agent_action':
                    addLogEntry(`Agent ${data.agent_id} executed ${data.action}`, 'info');
                    refreshAgents();
                    break;
                case 'status_update':
                    updateStats(data.data);
                    break;
                case 'subscription_confirmed':
                    addLogEntry('Subscribed to real-time updates', 'success');
                    break;
            }
        }
        
        async function loadInitialData() {
            try {
                // Load agents
                const agentsResponse = await fetch('/api/agents');
                const agentsData = await agentsResponse.json();
                updateAgentsList(agentsData.agents);
                
                // Load metrics
                const metricsResponse = await fetch('/api/metrics');
                const metricsData = await metricsResponse.json();
                updateStats(metricsData);
                
                // Update timestamp
                document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                
            } catch (error) {
                console.error('Failed to load data:', error);
                addLogEntry('Failed to load dashboard data', 'error');
            }
        }
        
        function updateStats(data) {
            document.getElementById('total-agents').textContent = data.active_agents || 0;
            document.getElementById('active-agents').textContent = data.active_agents || 0;
            document.getElementById('messages-sent').textContent = data.messages_sent || 0;
            document.getElementById('resources-created').textContent = data.resources_created || 0;
        }
        
        function updateAgentsList(agents) {
            const agentList = document.getElementById('agent-list');
            agentList.innerHTML = '';
            
            if (agents.length === 0) {
                agentList.innerHTML = '<p>No agents currently active</p>';
                return;
            }
            
            agents.forEach(agent => {
                const agentItem = document.createElement('div');
                agentItem.className = 'agent-item';
                agentItem.innerHTML = `
                    <div class="agent-info">
                        <h4>${agent.name}</h4>
                        <div class="agent-namespace">${agent.namespace}</div>
                    </div>
                    <div class="agent-status status-${agent.status}">${agent.status}</div>
                `;
                agentList.appendChild(agentItem);
            });
        }
        
        function addLogEntry(message, type = 'info') {
            const logContainer = document.getElementById('activity-log');
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry';
            
            const timestamp = new Date().toLocaleTimeString();
            logEntry.innerHTML = `
                <div>${message}</div>
                <div class="log-timestamp">${timestamp}</div>
            `;
            
            logContainer.insertBefore(logEntry, logContainer.firstChild);
            
            // Keep only last 10 entries
            while (logContainer.children.length > 10) {
                logContainer.removeChild(logContainer.lastChild);
            }
        }
        
        async function refreshAgents() {
            try {
                const response = await fetch('/api/agents');
                const data = await response.json();
                updateAgentsList(data.agents);
                addLogEntry('Agents list refreshed', 'info');
            } catch (error) {
                console.error('Failed to refresh agents:', error);
                addLogEntry('Failed to refresh agents', 'error');
            }
        }
    </script>
</body>
</html>
        """
        
    async def start_monitoring_loop(self):
        """Iniciar loop de monitoreo en background"""
        asyncio.create_task(self._monitoring_loop())
        
    async def _monitoring_loop(self):
        """Loop continuo de monitoreo"""
        while True:
            try:
                # Recopilar datos actuales
                status = await self._get_current_status()
                
                # Broadcast a WebSockets conectados
                await self._broadcast_update({
                    "type": "status_update",
                    "data": status,
                    "timestamp": datetime.now().isoformat()
                })
                
                await asyncio.sleep(10)  # Actualizar cada 10 segundos
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)
                
    async def start(self):
        """Iniciar el servidor del dashboard"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        # Iniciar monitoreo
        await self.start_monitoring_loop()
        
        logging.info(f"Dashboard server started at http://{self.host}:{self.port}")
        print(f"🌐 Dashboard available at: http://{self.host}:{self.port}")
        
        return runner

# ================================
# DASHBOARD INTEGRATION
# ================================

class DashboardIntegration:
    """Integración del dashboard con el framework"""
    
    def __init__(self, framework: AgentFramework):
        self.framework = framework
        self.dashboard_server = None
        
    async def start_dashboard(self, host: str = "localhost", port: int = 8080):
        """Iniciar dashboard integrado"""
        self.dashboard_server = DashboardServer(self.framework, host, port)
        runner = await self.dashboard_server.start()
        return runner
        
    async def stop_dashboard(self):
        """Detener dashboard"""
        if self.dashboard_server:
            # El runner se cierra automáticamente cuando el framework se detiene
            pass

# ================================
# EXAMPLE USAGE
# ================================

async def dashboard_demo():
    """Demo del dashboard web"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Crear framework
    from core.autonomous_agent_framework import AgentFramework
    from core.specialized_agents import ExtendedAgentFactory
    
    framework = AgentFramework()
    await framework.start()
    
    # Crear algunos agentes para mostrar en el dashboard
    strategist = ExtendedAgentFactory.create_agent("agent.planning.strategist", "strategist", framework)
    generator = ExtendedAgentFactory.create_agent("agent.build.code.generator", "generator", framework)
    tester = ExtendedAgentFactory.create_agent("agent.test.generator", "tester", framework)
    
    await strategist.start()
    await generator.start()
    await tester.start()
    
    # Crear integración del dashboard
    dashboard_integration = DashboardIntegration(framework)
    runner = await dashboard_integration.start_dashboard(host="localhost", port=8080)
    
    print("\n" + "="*60)
    print("🌐 DASHBOARD DEMO STARTED")
    print("="*60)
    print(f"Dashboard URL: http://localhost:8080")
    print(f"Active agents: {len(framework.registry.list_all_agents())}")
    print("\nFeatures available:")
    print("• Real-time agent monitoring")
    print("• System metrics and statistics")
    print("• WebSocket live updates")
    print("• Agent status visualization")
    print("• Activity logging")
    print("\nPress Ctrl+C to stop...")
    
    try:
        # Simular actividad para mostrar en el dashboard
        while True:
            await asyncio.sleep(10)
            
            # Simular intercambio de mensajes entre agentes
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
            
            # Crear recursos ocasionalmente
            from core.autonomous_agent_framework import AgentResource, ResourceType
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
        
    finally:
        await framework.stop()
        await runner.cleanup()
        print("👋 Dashboard demo stopped")

if __name__ == "__main__":
    asyncio.run(dashboard_demo())