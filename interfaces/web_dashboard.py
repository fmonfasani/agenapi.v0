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

# Importaciones actualizadas para AgentStatus, MessageType, AgentResource, ResourceType
from core.autonomous_agent_framework import AgentFramework, BaseAgent
from core.models import AgentStatus, MessageType, AgentResource, ResourceType # <-- CAMBIO AQUI
from systems.monitoring_system import MetricsCollector

# ================================\
# DASHBOARD SERVER
# ================================\

class DashboardServer:
    """Servidor web para el dashboard de monitoreo"""
    
    def __init__(self, framework: AgentFramework, host: str = "localhost", port: int = 8080):
        self.framework = framework
        self.host = host
        self.port = port
        self.app = web.Application()
        self.websockets: List[web.WebSocketResponse] = []
        self.metrics_collector = MetricsCollector(framework) # Instanciado desde el sistema de monitoreo
        self.dashboard_data = {
            "agents": {},
            "messages": [],
            "resources": {},
            "metrics": {},
            "system_status": "active"
        }
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)

        # Configurar rutas
        self._setup_routes()
        
        # Configurar CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
            )
        })
        for route in list(self.app.router.routes()):
            if route.resource:
                cors.add(route)

        # Tarea de actualización en segundo plano
        self._update_task = None

    def _setup_routes(self):
        """Configura las rutas del servidor web."""
        self.app.router.add_get('/', self.handle_root)
        self.app.router.add_get('/ws', self.websocket_handler)
        self.app.router.add_get('/api/dashboard/data', self.get_dashboard_data)
        
        # Servir archivos estáticos (HTML, CSS, JS para el dashboard)
        # Asumiendo que los archivos estáticos están en un directorio 'static'
        current_dir = Path(__file__).parent
        static_dir = current_dir / "static"
        if not static_dir.exists():
            self.logger.warning(f"Static directory not found at {static_dir}. Dashboard might not load correctly.")
            # Create a minimal index.html for testing if static folder is missing
            static_dir.mkdir(exist_ok=True)
            with open(static_dir / "index.html", "w") as f:
                f.write("<!DOCTYPE html><html><head><title>Agent Framework Dashboard</title></head><body><h1>Dashboard Not Configured</h1><p>Static files (index.html, etc.) are missing from the 'static' directory.</p><p>Ensure 'static' directory exists in the same location as web_dashboard.py and contains your dashboard files.</p></body></html>")

        self.app.router.add_static('/static/', path=static_dir, name='static')


    async def handle_root(self, request: web.Request):
        """Sirve el archivo HTML principal del dashboard."""
        current_dir = Path(__file__).parent
        index_file = current_dir / "static" / "index.html"
        if index_file.exists():
            return web.FileResponse(index_file)
        return web.Response(text="Dashboard HTML not found. Ensure 'static/index.html' exists.", status=404)

    async def get_dashboard_data(self, request: web.Request):
        """Endpoint REST para obtener todos los datos del dashboard."""
        await self._update_dashboard_data() # Ensure data is fresh
        return web.json_response(self.dashboard_data)

    async def websocket_handler(self, request: web.Request):
        """Maneja las conexiones WebSocket para actualizaciones en tiempo real."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.logger.info("WebSocket connection established for dashboard.")
        
        # Usamos weakref para evitar referencias circulares y permitir que los WS se cierren
        self.websockets.append(weakref.ref(ws))

        try:
            # Enviar datos iniciales
            await ws.send_json(self.dashboard_data)
            
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    if msg.data == 'close':
                        await ws.close()
                    else:
                        self.logger.debug(f"Received WS message: {msg.data}")
                elif msg.type == WSMsgType.ERROR:
                    self.logger.error(f"WebSocket connection closed with exception: {ws.exception()}")
        except Exception as e:
            self.logger.error(f"WebSocket error in dashboard: {e}")
        finally:
            self.logger.info("WebSocket connection closed for dashboard.")
            # Eliminar la referencia débil del cliente cerrado
            self.websockets = [ref for ref in self.websockets if ref() is not None and ref() != ws]

    async def _update_dashboard_data(self):
        """Actualiza los datos del dashboard recopilando información del framework."""
        # Agentes
        agents_info = []
        for agent in self.framework.registry.list_all_agents():
            agents_info.append({
                "id": agent.id,
                "name": agent.name,
                "namespace": agent.namespace,
                "status": agent.status.value,
                "last_heartbeat": agent.last_heartbeat.isoformat() if agent.last_heartbeat else None,
                "is_alive": agent.is_alive
            })
        self.dashboard_data["agents"] = {a["id"]: a for a in agents_info}

        # Mensajes recientes (ejemplo: los últimos 50 mensajes del bus)
        # Esto es un placeholder; en un sistema real, se recuperaría del PersistenceManager
        recent_messages = []
        # Simulate fetching some recent messages. In a real system, MessageBus or PersistenceManager would provide this.
        # For this demo, we can't directly read from the message bus's internal queue.
        # A more complex integration with PersistenceManager would be needed here.
        # For now, just keep an empty list or fetch from a dummy source.
        # Example of how you *might* get recent messages if MessageBus exposed them:
        # for msg in self.framework.message_bus.get_recent_messages(limit=50):
        #     recent_messages.append(asdict(msg))
        self.dashboard_data["messages"] = recent_messages

        # Recursos
        resources_info = []
        for resource in self.framework.resource_manager.list_all_resources():
            resources_info.append({
                "id": resource.id,
                "name": resource.name,
                "namespace": resource.namespace,
                "type": resource.type.value,
                "owner_agent_id": resource.owner_agent_id,
                "created_at": resource.created_at.isoformat(),
                "updated_at": resource.updated_at.isoformat() if resource.updated_at else None
            })
        self.dashboard_data["resources"] = {r["id"]: r for r in resources_info}

        # Métricas del sistema
        latest_metrics = self.metrics_collector.get_latest_metrics()
        self.dashboard_data["metrics"] = {name: m.value for name, m in latest_metrics.items()}

        # Estado general
        # (Esto debería ser idealmente del HealthChecker del MonitoringOrchestrator)
        self.dashboard_data["system_status"] = "active" if self.framework.is_running else "inactive"
        if not self.metrics_collector.get_latest_metrics():
            self.dashboard_data["system_status"] = "degraded" # Example

        self.logger.debug("Dashboard data updated.")

    async def _broadcast_dashboard_data(self):
        """Envía los datos actualizados a todos los clientes WebSocket conectados."""
        # Limpiar referencias muertas
        self.websockets = [ref for ref in self.websockets if ref() is not None]
        
        if not self.websockets:
            self.logger.debug("No active WebSocket clients to broadcast to.")
            return

        await self._update_dashboard_data()
        
        for ws_ref in self.websockets:
            ws = ws_ref()
            if ws:
                try:
                    await ws.send_json(self.dashboard_data)
                except Exception as e:
                    self.logger.warning(f"Failed to send data to WebSocket client: {e}. Closing connection.")
                    # Consider removing on send error, but aiohttp usually handles it on next message
                    pass
            else:
                self.logger.debug("Dead WebSocket reference found and removed.")

    async def _update_loop(self):
        """Bucle que actualiza y emite datos del dashboard periódicamente."""
        while True:
            await self._broadcast_dashboard_data()
            await asyncio.sleep(5)  # Actualizar cada 5 segundos

    async def start(self):
        """Inicia el servidor del dashboard y el bucle de actualización."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        self.logger.info(f"Dashboard Server started on http://{self.host}:{self.port}")
        self.logger.info(f"WebSocket endpoint for real-time updates: ws://{self.host}:{self.port}/ws")
        
        # Iniciar la tarea de actualización en segundo plano
        self._update_task = asyncio.create_task(self._update_loop())
        return self.runner # Return runner for graceful shutdown in main app

    async def stop(self):
        """Detiene el servidor del dashboard y el bucle de actualización."""
        self.logger.info("Stopping Dashboard Server...")
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task # Wait for the task to finish cancelling
            except asyncio.CancelledError:
                self.logger.debug("Dashboard update loop cancelled.")

        # Cerrar todas las conexiones WebSocket activas
        for ws_ref in list(self.websockets): # Iterate on a copy
            ws = ws_ref()
            if ws:
                await ws.close()
        self.websockets.clear()

        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        self.logger.info("Dashboard Server stopped.")

# ================================\
# DEMO / USAGE EXAMPLE
# (This will eventually be moved to end_to_end_example.py)
# ================================\

async def dashboard_demo():
    """Ejemplo de uso del Dashboard Server."""
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Starting Dashboard Demo")
    print("="*50)
    
    # Crear framework
    framework = AgentFramework()
    await framework.start()
    
    # Crear algunos agentes para demo
    from core.specialized_agents import ExtendedAgentFactory # Import here for demo
    strategist = ExtendedAgentFactory.create_agent("agent.planning.strategist", "strategist", framework)
    generator = ExtendedAgentFactory.create_agent("agent.build.code.generator", "generator", framework)
    tester = ExtendedAgentFactory.create_agent("agent.test.unit_tester", "tester", framework)
    
    await strategist.start()
    await generator.start()
    await tester.start()

    # Iniciar el sistema de monitoreo para que el dashboard tenga métricas reales
    monitoring = MonitoringOrchestrator(framework)
    await monitoring.start_monitoring()
    
    # Crear y lanzar el servidor del dashboard
    dashboard_server = DashboardServer(framework, host="localhost", port=8080)
    runner = await dashboard_server.start()
    
    print(f"\n✅ Dashboard server running with {len(framework.registry.list_all_agents())} agents")
    print(f"🌐 Visit http://localhost:8080 to view the dashboard.")
    print("• Real-time agent status")
    print("• System metrics")
    print("• Resource monitoring")
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
            # Ya no se necesita esta importación interna si ya están en el encabezado.
            # from core.autonomous_agent_framework import AgentResource, ResourceType
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
        await monitoring.stop_monitoring() # Stop monitoring before framework
        await framework.stop()
        await dashboard_server.stop()
        print("👋 Dashboard demo stopped")

if __name__ == "__main__":
    asyncio.run(dashboard_demo())