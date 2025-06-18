"""
framework_cli.py - Herramienta de línea de comandos para el Framework de Agentes
"""

import asyncio
import click
import json
import yaml
import sys
import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import aiohttp
import getpass
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.tree import Tree

# Imports del framework
from core.autonomous_agent_framework import AgentFramework
from core.specialized_agents import ExtendedAgentFactory
from core.security_system import SecurityManager, Permission, AuthenticationMethod
from core.persistence_system import PersistenceFactory, PersistenceBackend
from systems.deployment_system import DeploymentOrchestrator, DeploymentEnvironment, DeploymentStrategy
from core.backup_recovery_system import DisasterRecoveryOrchestrator
from systems.monitoring_system import MonitoringOrchestrator, MetricsCollector # <-- CAMBIO AQUI

console = Console()

# ================================\
# CLI CONFIGURATION
# ================================\

class CLIConfig:
    """Configuración de la CLI"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".agent-framework"
        self.config_file = self.config_dir / "config.yaml"
        self.config_dir.mkdir(exist_ok=True)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración desde el archivo o usa defaults."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        return {
            "api_base_url": "http://localhost:8000",
            "dashboard_base_url": "http://localhost:8080",
            "auth_token": None
        }

    def _save_config(self):
        """Guarda la configuración actual en el archivo."""
        with open(self.config_file, 'w') as f:
            yaml.safe_dump(self.config, f)

    def get(self, key: str, default: Any = None) -> Any:
        """Obtiene un valor de la configuración."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """Establece un valor en la configuración y lo guarda."""
        self.config[key] = value
        self._save_config()

cli_config = CLIConfig()

# ================================\
# HELPER FUNCTIONS
# ================================\

def print_success(message: str):
    console.print(f"[bold green]✅ {message}[/bold green]")

def print_error(message: str):
    console.print(f"[bold red]❌ {message}[/bold red]", err=True)

def print_info(message: str):
    console.print(f"[bold blue]ℹ️ {message}[/bold blue]")

def print_warning(message: str):
    console.print(f"[bold yellow]⚠️ {message}[/bold yellow]")

async def api_call(method: str, path: str, data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Realiza una llamada a la API del framework."""
    base_url = cli_config.get("api_base_url")
    url = f"{base_url}{path}"
    
    default_headers = {"Content-Type": "application/json"}
    auth_token = cli_config.get("auth_token")
    if auth_token:
        default_headers["Authorization"] = f"Bearer {auth_token}"
    
    if headers:
        default_headers.update(headers)

    try:
        async with aiohttp.ClientSession() as session:
            if method.upper() == "GET":
                async with session.get(url, headers=default_headers) as response:
                    return await response.json()
            elif method.upper() == "POST":
                async with session.post(url, json=data, headers=default_headers) as response:
                    return await response.json()
            elif method.upper() == "PUT":
                async with session.put(url, json=data, headers=default_headers) as response:
                    return await response.json()
            elif method.upper() == "DELETE":
                async with session.delete(url, headers=default_headers) as response:
                    return await response.json()
            else:
                return APIResponse.error("Unsupported HTTP method", code=405)
    except aiohttp.ClientConnectionError:
        print_error(f"Failed to connect to API at {base_url}. Is the server running?")
        return {"success": False, "error": {"message": "API server unreachable", "code": 503}}
    except Exception as e:
        print_error(f"An error occurred during API call: {e}")
        return {"success": False, "error": {"message": str(e), "code": 500}}

# ================================\
# CLI COMMANDS
# ================================\

@click.group()
@click.pass_context
def cli(ctx):
    """Herramienta de línea de comandos para gestionar el Autonomous Agent Framework."""
    ctx.ensure_object(dict)
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s') # Default to WARNING for CLI

@cli.command()
@click.option('--url', default="http://localhost:8000", help='Base URL for the API server.')
@click.option('--dashboard-url', default="http://localhost:8080", help='Base URL for the Web Dashboard.')
def configure(url, dashboard_url):
    """Configura la URL base del servidor API y del dashboard."""
    cli_config.set("api_base_url", url)
    cli_config.set("dashboard_base_url", dashboard_url)
    print_success(f"API base URL set to: {url}")
    print_success(f"Dashboard base URL set to: {dashboard_url}")

@cli.command()
@click.option('--username', prompt='API Username', help='Username for API authentication.')
@click.option('--password', prompt='API Password', hide_input=True, confirmation_prompt=False, help='Password for API authentication.')
@click.option('--api-key', help='Optional API Key for authentication (overrides username/password if provided).')
def login(username, password, api_key):
    """Autentica con el servidor API y guarda el token."""
    async def _login():
        payload = {}
        if api_key:
            payload = {"method": AuthenticationMethod.API_KEY.value, "api_key": api_key}
            print_info("Attempting login with API Key...")
        else:
            payload = {"method": AuthenticationMethod.JWT_TOKEN.value, "username": username, "password": password}
            print_info("Attempting login with Username/Password...")

        response = await api_call("POST", "/api/auth/login", data=payload)
        
        if response.get("success"):
            token = response["data"]["token"]
            cli_config.set("auth_token", token)
            print_success("Logged in successfully. Token stored.")
        else:
            print_error(f"Login failed: {response.get('error', {}).get('message', 'Unknown error')}")
            cli_config.set("auth_token", None) # Clear any old token

    asyncio.run(_login())

@cli.command()
def logout():
    """Elimina el token de autenticación almacenado."""
    cli_config.set("auth_token", None)
    print_success("Logged out. Authentication token cleared.")

@cli.command()
def status():
    """Muestra el estado actual del framework y los servicios."""
    async def _status():
        print_info("Fetching system status...")
        response = await api_call("GET", "/api/health")
        
        if response.get("success"):
            data = response["data"]
            
            table = Table(title="Framework Status", show_lines=True)
            table.add_column("Component", style="cyan", no_wrap=True)
            table.add_column("Status", style="magenta")
            table.add_column("Details", style="green")
            
            table.add_row("Overall System", "[bold]" + data['status'].upper() + "[/bold]", data.get('message', ''))
            table.add_row("Agent Framework Core", data['framework']['status'], f"Agents: {data['framework']['total_agents']} (Active: {data['framework']['active_agents']})")
            table.add_row("REST API", data['api']['status'], f"Port: {data['api']['port']}")
            table.add_row("Persistence System", data['persistence']['status'], f"Backend: {data['persistence']['backend']} (Messages: {data['persistence']['total_messages']}, Resources: {data['persistence']['total_resources']})")
            table.add_row("Security System", data['security']['status'], f"Auth Enabled: {data['security']['authentication_enabled']} (Auth Method: {data['security']['auth_method']})")
            table.add_row("Monitoring System", data['monitoring']['status'], f"Metrics Collected: {data['monitoring']['metrics_collected']}")
            
            console.print(table)
            
            # Additional details from metrics if available
            metrics_response = await api_call("GET", "/api/metrics")
            if metrics_response.get("success"):
                metrics_data = metrics_response["data"]
                if metrics_data:
                    console.print("\n[bold]📈 Key Metrics:[/bold]")
                    metrics_table = Table(show_header=False, show_lines=False, box=None)
                    metrics_table.add_column(style="cyan")
                    metrics_table.add_column(style="green")
                    
                    # Display a few key metrics
                    for i, (name, metric) in enumerate(metrics_data.items()):
                        if i >= 5: # Limit to 5 for brevity
                            break
                        metrics_table.add_row(f"  {metric['name']}:", f"{metric['value']:.2f} {metric['unit']}")
                    console.print(metrics_table)
            
        else:
            print_error(f"Failed to retrieve status: {response.get('error', {}).get('message', 'Unknown error')}")
            
    asyncio.run(_status())

@cli.command()
@click.option('--full', is_flag=True, help='Show full details for each agent.')
def agents(full):
    """Lista todos los agentes registrados."""
    async def _agents():
        print_info("Fetching agent list...")
        response = await api_call("GET", "/api/agents")
        
        if response.get("success"):
            agents_data = response["data"]
            if not agents_data:
                print_info("No agents found.")
                return
            
            table = Table(title="Registered Agents", show_lines=True)
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Name", style="magenta")
            table.add_column("Namespace", style="yellow")
            table.add_column("Status", style="green")
            if full:
                table.add_column("Capabilities", style="blue")
                table.add_column("Last Heartbeat", style="dim")

            for agent in agents_data:
                status_color = "green" if agent['status'] == 'ACTIVE' else "red" if agent['status'] == 'ERROR' else "yellow"
                
                row_data = [
                    agent['id'][:8] + "...",
                    agent['name'],
                    agent['namespace'],
                    f"[{status_color}]{agent['status']}[/{status_color}]"
                ]
                if full:
                    capabilities = ", ".join([cap['name'] for cap in agent.get('capabilities', [])]) or "N/A"
                    last_heartbeat = datetime.fromisoformat(agent['last_heartbeat']).strftime('%Y-%m-%d %H:%M:%S') if agent['last_heartbeat'] else "N/A"
                    row_data.extend([capabilities, last_heartbeat])
                table.add_row(*row_data)
            
            console.print(table)
        else:
            print_error(f"Failed to list agents: {response.get('error', {}).get('message', 'Unknown error')}")

    asyncio.run(_agents())

@cli.command()
@click.argument('namespace')
@click.argument('name')
@click.option('--description', help='Description for the new agent.')
@click.option('--capabilities', multiple=True, help='List of capabilities (e.g., "action.generate.code").')
def create_agent(namespace, name, description, capabilities):
    """Crea un nuevo agente en el framework."""
    async def _create_agent():
        print_info(f"Creating agent '{name}' in namespace '{namespace}'...")
        agent_data = {
            "namespace": namespace,
            "name": name,
            "description": description,
            "capabilities": [{"name": cap, "namespace": cap, "description": f"Capability {cap}"} for cap in capabilities]
        }
        
        response = await api_call("POST", "/api/agents", data=agent_data)
        
        if response.get("success"):
            agent_id = response["data"]["id"]
            print_success(f"Agent '{name}' created with ID: {agent_id}")
        else:
            print_error(f"Failed to create agent: {response.get('error', {}).get('message', 'Unknown error')}")

    asyncio.run(_create_agent())

@cli.command()
@click.argument('agent_id_or_name')
def delete_agent(agent_id_or_name):
    """Elimina un agente por ID o nombre."""
    async def _delete_agent():
        print_warning(f"Attempting to delete agent: {agent_id_or_name}...")
        if not Confirm.ask(f"Are you sure you want to delete agent '{agent_id_or_name}'?", default=False):
            print_info("Operation cancelled.")
            return

        # First, try to find by name if not an ID
        agent_id = agent_id_or_name
        if len(agent_id_or_name) < 30: # Assume short string is a name, UUIDs are longer
            response = await api_call("GET", "/api/agents")
            if response.get("success"):
                found_agent = next((a for a in response["data"] if a["name"] == agent_id_or_name), None)
                if found_agent:
                    agent_id = found_agent["id"]
                else:
                    print_error(f"Agent '{agent_id_or_name}' not found by name or ID.")
                    return
            else:
                print_error(f"Could not retrieve agent list to resolve name: {response.get('error', {}).get('message', 'Unknown error')}")
                return

        response = await api_call("DELETE", f"/api/agents/{agent_id}")
        
        if response.get("success"):
            print_success(f"Agent '{agent_id_or_name}' (ID: {agent_id[:8]}...) deleted successfully.")
        else:
            print_error(f"Failed to delete agent '{agent_id_or_name}': {response.get('error', {}).get('message', 'Unknown error')}")

    asyncio.run(_delete_agent())

@cli.command()
@click.argument('agent_id')
@click.argument('action')
@click.argument('params', type=str, required=False)
def send_action(agent_id, action, params):
    """Envía una acción a un agente específico con parámetros JSON."""
    async def _send_action():
        print_info(f"Sending action '{action}' to agent '{agent_id}'...")
        try:
            parsed_params = json.loads(params) if params else {}
        except json.JSONDecodeError:
            print_error("Invalid JSON for parameters. Please use valid JSON string (e.g., '{\"key\":\"value\"}').")
            return
            
        message_data = {
            "receiver_id": agent_id,
            "message_type": "COMMAND",
            "payload": {
                "action": action,
                "parameters": parsed_params
            }
        }
        
        response = await api_call("POST", "/api/messages", data=message_data)
        
        if response.get("success"):
            message_id = response["data"]["message_id"]
            print_success(f"Action '{action}' sent to agent '{agent_id}'. Message ID: {message_id}")
        else:
            print_error(f"Failed to send action: {response.get('error', {}).get('message', 'Unknown error')}")

    asyncio.run(_send_action())

@cli.command()
def metrics():
    """Muestra las métricas del sistema."""
    async def _metrics():
        print_info("Fetching system metrics...")
        response = await api_call("GET", "/api/metrics")
        
        if response.get("success"):
            metrics_data = response["data"]
            if not metrics_data:
                print_info("No metrics available.")
                return
            
            table = Table(title="System Metrics", show_lines=True)
            table.add_column("Metric Name", style="cyan")
            table.add_column("Value", style="magenta")
            table.add_column("Unit", style="yellow")
            table.add_column("Timestamp", style="green")
            table.add_column("Tags", style="blue")
            
            for name, metric in metrics_data.items():
                tags = json.dumps(metric.get('tags', {})) if metric.get('tags') else "N/A"
                table.add_row(
                    metric['name'],
                    f"{metric['value']:.2f}",
                    metric.get('unit', ''),
                    datetime.fromisoformat(metric['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
                    tags
                )
            console.print(table)
        else:
            print_error(f"Failed to retrieve metrics: {response.get('error', {}).get('message', 'Unknown error')}")

    asyncio.run(_metrics())

@cli.command()
def logs():
    """Muestra los logs recientes del sistema (requiere endpoint de logs)."""
    print_warning("This command requires a '/api/logs' endpoint which is not yet implemented in the REST API.")
    print_info("Please check the system logs directly from the running services or framework.")

@cli.command()
def dashboard():
    """Abre el dashboard web en el navegador por defecto."""
    dashboard_url = cli_config.get("dashboard_base_url")
    if dashboard_url:
        print_info(f"Opening dashboard at: {dashboard_url}")
        click.launch(dashboard_url)
    else:
        print_error("Dashboard URL not configured. Please run 'agent-cli configure --dashboard-url <URL>'.")


@cli.command()
def start_dev_env():
    """Inicia un entorno de desarrollo local (API + Dashboard + Framework)."""
    async def _start_dev_env():
        print_info("🚀 Starting local development environment...")
        
        framework = AgentFramework()
        await framework.start()
        
        # Agentes de demostración
        strategist = ExtendedAgentFactory.create_agent("agent.planning.strategist", "strategist", framework)
        generator = ExtendedAgentFactory.create_agent("agent.build.code.generator", "generator", framework)
        await strategist.start()
        await generator.start()

        # Iniciar Seguridad
        security_manager = SecurityManager(framework)
        await security_manager.initialize()

        # Iniciar Persistencia
        persistence_manager = PersistenceFactory.create_persistence_manager(
            framework, PersistenceBackend.SQLITE, "framework_dev.db"
        )
        await persistence_manager.initialize()
        framework.persistence_manager = persistence_manager # Link for demo

        # Iniciar Monitoreo
        monitoring_orchestrator = MonitoringOrchestrator(framework)
        await monitoring_orchestrator.start_monitoring()

        # Iniciar API
        from interfaces.rest_api import FrameworkAPIServer # Import dynamically to avoid circular issues
        api_server = FrameworkAPIServer(framework, security_manager, persistence_manager, host="0.0.0.0", port=8000)
        api_runner = await api_server.start()

        # Iniciar Dashboard
        from interfaces.web_dashboard import DashboardServer # Import dynamically
        dashboard_server = DashboardServer(framework, host="0.0.0.0", port=8080)
        dashboard_runner = await dashboard_server.start()

        print_success("Development environment started:")
        print_info(f"  API: http://localhost:8000")
        print_info(f"  Dashboard: http://localhost:8080")
        print_info(f"  Agents running: {len(framework.registry.list_all_agents())}")
        print_info("Press Ctrl+C to stop...")

        try:
            while True:
                await asyncio.sleep(1) # Keep event loop running
        except KeyboardInterrupt:
            print_warning("\n🛑 Shutting down development environment...")
        finally:
            # Shutdown in reverse order
            if dashboard_server:
                await dashboard_server.stop()
            if api_server:
                await api_server.stop()
            if monitoring_orchestrator:
                await monitoring_orchestrator.stop_monitoring()
            if persistence_manager:
                await persistence_manager.close()
            if framework:
                await framework.stop()
            print_success("Development environment stopped.")

    asyncio.run(_start_dev_env())


@cli.command()
@click.option('--environment', type=click.Choice([e.value for e in DeploymentEnvironment]), default='development', help='Deployment environment.')
@click.option('--strategy', type=click.Choice([s.value for s in DeploymentStrategy]), default='standalone', help='Deployment strategy.')
@click.option('--output-dir', default='./deployment_output', help='Output directory for deployment files.')
def deploy(environment, strategy, output_dir):
    """Genera archivos de deployment para un entorno y estrategia dados."""
    async def _deploy():
        print_info(f"Preparing deployment for '{environment}' environment with '{strategy}' strategy...")
        orchestrator = DeploymentOrchestrator()
        
        # Placeholder for real framework/security/persistence configs
        # In a real scenario, these would come from a more robust config system or actual running services
        framework_config_data = {"name": "MyDeployedFramework", "version": "1.0.0"}
        security_config_data = {"authentication_enabled": True, "jwt_secret": "my_secret_key"}
        persistence_config_data = {"backend": "sqlite", "connection_string": "deployed_framework.db"}
        api_config_data = {"host": "0.0.0.0", "port": 8000}
        monitoring_config_data = {"enable_monitoring": True}
        agents_config_data = [
            {"namespace": "agent.planning", "name": "planner", "auto_start": True},
            {"namespace": "agent.build", "name": "builder", "auto_start": True}
        ]
        scaling_config_data = {"min_agents": 1, "max_agents": 5}


        config = orchestrator.create_deployment_config(
            DeploymentEnvironment(environment),
            DeploymentStrategy(strategy),
            framework_config=framework_config_data,
            security_config=security_config_data,
            persistence_config=persistence_config_data,
            api_config=api_config_data,
            monitoring_config=monitoring_config_data,
            agents_config=agents_config_data,
            scaling_config=scaling_config_data
        )
        
        success = await orchestrator.deploy(config, Path(output_dir))
        
        if success:
            print_success(f"Deployment files generated successfully in '{output_dir}' for {environment}/{strategy}.")
            if strategy == 'docker':
                print_info(f"  Run: cd {output_dir} && docker-compose up -d")
            elif strategy == 'kubernetes':
                print_info(f"  Run: cd {output_dir} && kubectl apply -f .")
            elif strategy == 'standalone':
                print_info(f"  Review {output_dir}/run.sh for execution instructions.")
        else:
            print_error("Deployment generation failed.")

    asyncio.run(_deploy())

@cli.command()
@click.option('--type', type=click.Choice(['full', 'incremental']), default='full', help='Type of backup to perform.')
@click.option('--storage', type=click.Choice([s.value for s in PersistenceBackend]), default='local', help='Storage backend for the backup.')
def backup(type, storage):
    """Realiza un backup del estado del framework."""
    async def _backup():
        print_info(f"Initiating {type} backup to {storage} storage...")
        
        # Need a running framework instance to perform backup
        # For CLI, we can either connect to a running one or start a minimal one
        # For simplicity in CLI, we'll simulate or assume connection to persistence
        
        # This part assumes PersistenceManager is already configured/running or can be initialized minimally
        # In a real CLI, it would connect to the running framework's PersistenceManager.
        # For this demo, let's just make sure the DisasterRecoveryOrchestrator can function.
        
        # To make this command functional without a full running framework,
        # we'd need the PersistenceManager to be directly instantiable and configurable via CLI options,
        # or have the CLI connect to a running API that exposes backup functionality.
        # For now, let's connect to the persistence manager if it was started by `start_dev_env`
        # or create a dummy one for demonstration purposes.

        # Simplified approach for demo: Assume persistence config
        temp_framework = AgentFramework() # Minimal framework for DR Orchestrator
        persistence_manager = PersistenceFactory.create_persistence_manager(
            temp_framework, PersistenceBackend.SQLITE, "framework_cli_temp.db"
        )
        await persistence_manager.initialize() # Initialize a dummy persistence

        dr_orchestrator = DisasterRecoveryOrchestrator(temp_framework, persistence_manager)

        try:
            if type == 'full':
                result = await dr_orchestrator.backup_engine.perform_full_backup()
            elif type == 'incremental':
                result = await dr_orchestrator.backup_engine.perform_incremental_backup()
            else:
                print_error("Invalid backup type.")
                return

            if result and result.status == "completed":
                print_success(f"Backup completed: {result.backup_id} (Size: {result.size_bytes} bytes, Path: {result.file_path})")
            else:
                print_error(f"Backup failed: {result.status if result else 'Unknown error'}")
        finally:
            await persistence_manager.close()
            await temp_framework.stop() # Clean up dummy framework

    asyncio.run(_backup())


@cli.command()
@click.argument('backup_id', required=False)
@click.option('--point-in-time', help='Timestamp (YYYY-MM-DD HH:MM:SS) to restore to.')
@click.option('--source-path', help='Optional path to a specific backup file or directory.')
def restore(backup_id, point_in_time, source_path):
    """Restaura el estado del framework desde un backup."""
    async def _restore():
        print_info("Initiating restore operation...")
        
        # Similar to backup, this needs a PersistenceManager context
        temp_framework = AgentFramework()
        persistence_manager = PersistenceFactory.create_persistence_manager(
            temp_framework, PersistenceBackend.SQLITE, "framework_cli_temp.db"
        )
        await persistence_manager.initialize()

        dr_orchestrator = DisasterRecoveryOrchestrator(temp_framework, persistence_manager)

        try:
            if backup_id:
                result = await dr_orchestrator.backup_engine.restore_backup(backup_id)
            elif point_in_time:
                try:
                    dt = datetime.strptime(point_in_time, '%Y-%m-%d %H:%M:%S')
                    result = await dr_orchestrator.backup_engine.restore_to_point_in_time(dt)
                except ValueError:
                    print_error("Invalid point-in-time format. Use YYYY-MM-DD HH:MM:SS.")
                    return
            elif source_path:
                print_warning("Restoring from a specific path bypasses backup history. Use with caution.")
                result = await dr_orchestrator.backup_engine.restore_from_path(source_path)
            else:
                print_error("Please provide either a --backup-id, --point-in-time, or --source-path for restore.")
                return

            if result and result["success"]:
                print_success(f"Restore completed successfully: {result.get('message', '')}")
            else:
                print_error(f"Restore failed: {result.get('error', 'Unknown error')}")
        finally:
            await persistence_manager.close()
            await temp_framework.stop()

    asyncio.run(_restore())

@cli.command()
@click.option('--limit', type=int, default=10, help='Number of recent backups to list.')
def list_backups(limit):
    """Lista los backups recientes."""
    async def _list_backups():
        print_info("Fetching backup history...")
        
        temp_framework = AgentFramework()
        persistence_manager = PersistenceFactory.create_persistence_manager(
            temp_framework, PersistenceBackend.SQLITE, "framework_cli_temp.db"
        )
        await persistence_manager.initialize()

        dr_orchestrator = DisasterRecoveryOrchestrator(temp_framework, persistence_manager)

        try:
            history = dr_orchestrator.backup_engine.get_backup_history()
            if not history:
                print_info("No backups found.")
                return

            table = Table(title="Backup History", show_lines=True)
            table.add_column("ID", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Status", style="yellow")
            table.add_column("Created At", style="green")
            table.add_column("Size (Bytes)", style="blue")
            table.add_column("Path", style="dim")

            for backup in history[:limit]:
                table.add_row(
                    backup.backup_id[:8] + "...",
                    backup.backup_type.value,
                    backup.status.value,
                    backup.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                    str(backup.size_bytes),
                    backup.file_path
                )
            console.print(table)
        finally:
            await persistence_manager.close()
            await temp_framework.stop()

    asyncio.run(_list_backups())

# ================================\
# INTERACTIVE MODE (Future Expansion)
# ================================\

# This section is for a more advanced interactive shell.
# Currently, `start_dev_env` provides a basic long-running process.

# async def list_agents_interactive():
#     """List agents in interactive mode"""
#     async with FrameworkAPIClient() as client:
#         response = await client.get('/api/agents')
#         if response.get('success'):
#             agents = response['data']
#             if not agents:
#                 console.print("[yellow]No agents registered.[/yellow]")
#                 return
            
#             tree = Tree("🤖 Agents")
#             for agent in agents:
#                 status_icon = "🟢" if agent['status'] == 'active' else "🔴"
#                 tree.add(f"{status_icon} {agent['name']} ({agent['namespace']})")
            
#             console.print(tree)
#         else:
#             print_error("Failed to list agents")

# async def system_status_interactive():
#     """Show system status in interactive mode"""
#     async with FrameworkAPIClient() as client:
#         try:
#             response = await client.get('/api/health')
            
#             if response.get('success'):
#                 data = response['data']
                
#                 status_icon = "🟢" if data['status'] == 'healthy' else "🔴"
#                 console.print(f"{status_icon} System Status: {data['status']}")
#                 console.print(f"📊 Total Agents: {data['framework']['total_agents']}")
#                 console.print(f"✅ Active Agents: {data['framework']['active_agents']}")
#             else:
#                 print_error("Failed to get system status")
                
#         except Exception as e:
#             print_error(f"System unreachable: {e}")

# ================================\
# MAIN ENTRY POINT
# ================================\

if __name__ == '__main__':
    try:
        cli()
    except KeyboardInterrupt:
        print_info("\nOperation cancelled")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)