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
from functools import wraps

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.tree import Tree

# Imports del framework (assuming they are in a 'core' and 'systems' directory relative to where this script runs)
try:
    from core.autonomous_agent_framework import AgentFramework
    from core.specialized_agents import ExtendedAgentFactory
    from core.security_system import SecurityManager, Permission, AuthenticationMethod
    from core.persistence_system import PersistenceFactory, PersistenceBackend
    from systems.deployment_system import DeploymentOrchestrator, DeploymentEnvironment, DeploymentStrategy
    from core.backup_recovery_system import DisasterRecoveryOrchestrator
    from core.monitoring_system import MonitoringOrchestrator
except ImportError as e:
    print(f"Error importing framework components: {e}")
    print("Please ensure your PYTHONPATH includes the 'core' and 'systems' directories or run from the project root.")
    sys.exit(1)


console = Console()
logger = logging.getLogger(__name__)


# CLI CONFIGURATION

class CLIConfig:
    """Configuración de la CLI"""

    def __init__(self):
        self.config_dir = Path.home() / ".agent-framework-cli" # Changed to avoid potential conflicts
        self.config_file = self.config_dir / "config.yaml"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Cargar configuración desde el archivo YAML."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                print_error(f"Error loading configuration file: {e}")
                return {}
        return {}

    def save_config(self):
        """Guardar configuración en el archivo YAML."""
        try:
            with open(self.config_file, "w") as f:
                yaml.dump(self.config, f, indent=2)
        except IOError as e:
            print_error(f"Error saving configuration file: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Obtener valor de configuración."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """Establecer valor de configuración y guardarlo."""
        self.config[key] = value
        self.save_config()

cli_config = CLIConfig()


# API CLIENT

class FrameworkAPIClient:
    """Cliente para la API del framework."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, token: Optional[str] = None):
        self.base_url = base_url or cli_config.get("api_url", "http://localhost:8000")
        self.api_key = api_key or cli_config.get("api_key")
        self.token = token or cli_config.get("token")
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Inicializa la sesión aiohttp."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cierra la sesión aiohttp."""
        if self.session:
            await self.session.close()
            self.session = None # Ensure session is marked as closed

    def _get_headers(self) -> Dict[str, str]:
        """Obtener headers de autenticación."""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        elif self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    async def _request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Método genérico para realizar solicitudes HTTP."""
        if not self.session:
            raise RuntimeError("API client session is not initialized. Use 'async with' or call __aenter__.")

        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        logger.debug(f"Making {method} request to {url} with headers {headers} and data {data}")

        try:
            async with self.session.request(method, url, headers=headers, json=data) as response:
                response.raise_for_status() # Raises an aiohttp.ClientResponseError for 4xx/5xx responses
                return await response.json()
        except aiohttp.ClientResponseError as e:
            error_text = await e.response.text()
            raise click.ClickException(f"API Error ({e.status} {e.message}): {error_text}")
        except aiohttp.ClientConnectionError as e:
            raise click.ClickException(f"Connection Error: Could not connect to the API server at {self.base_url}. Is it running?")
        except Exception as e:
            raise click.ClickException(f"An unexpected error occurred: {e}")

    async def get(self, endpoint: str) -> Dict[str, Any]:
        """Realiza una solicitud GET."""
        return await self._request("GET", endpoint)

    async def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Realiza una solicitud POST."""
        return await self._request("POST", endpoint, data)

    async def delete(self, endpoint: str) -> Dict[str, Any]:
        """Realiza una solicitud DELETE."""
        return await self._request("DELETE", endpoint)


# UTILITY FUNCTIONS

def print_success(message: str):
    """Imprimir mensaje de éxito."""
    console.print(f"✅ {message}", style="bold green")

def print_error(message: str):
    """Imprimir mensaje de error."""
    console.print(f"❌ {message}", style="bold red")

def print_warning(message: str):
    """Imprimir mensaje de advertencia."""
    console.print(f"⚠️ {message}", style="bold yellow")

def print_info(message: str):
    """Imprimir mensaje informativo."""
    console.print(f"ℹ️ {message}", style="bold blue")

def format_datetime(dt_str: Optional[str]) -> str:
    """Formatear datetime string a un formato legible."""
    if not dt_str:
        return "N/A"
    try:
        # Handle various ISO 8601 formats, including those with 'Z' for UTC
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except ValueError:
        return dt_str # Return as is if parsing fails

def async_command(f):
    """Decorador para ejecutar comandos Click de forma asíncrona."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

# Pass the API client to commands
pass_api_client = click.make_pass_decorator(FrameworkAPIClient)


# MAIN CLI GROUP

@click.group()
@click.option('--debug', is_flag=True, help='Enable debug logging.')
@click.option('--config-file', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help='Path to a custom configuration file.')
@click.option('--api-url', help='Override the API base URL.')
@click.option('--api-key', help='Override the API key for authentication.')
@click.option('--token', help='Override the JWT token for authentication.')
@click.pass_context
def cli(ctx: click.Context, debug: bool, config_file: Optional[Path], api_url: Optional[str],
        api_key: Optional[str], token: Optional[str]):
    """
    🤖 Agent Framework CLI

    Manage autonomous agents, deployments, and system operations.
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format='%(message)s') # Default for rich output

    if config_file:
        cli_config.config_file = config_file
        cli_config.config = cli_config._load_config()

    # Initialize FrameworkAPIClient and pass it to the context
    ctx.obj = FrameworkAPIClient(base_url=api_url, api_key=api_key, token=token)


# CONFIG COMMANDS

@cli.group()
def config():
    """Manage CLI configuration."""
    pass

@config.command()
@click.option('--api-url', prompt="API Base URL", default=lambda: cli_config.get('api_url', 'http://localhost:8000'),
              help='The base URL for the Agent Framework API.')
@click.option('--api-key', prompt="API Key (leave empty if using token)", default='', hide_input=True, required=False,
              help='API key for authenticating with the framework.')
def setup(api_url: str, api_key: str):
    """Set up or update CLI configuration."""
    cli_config.set('api_url', api_url)
    if api_key:
        cli_config.set('api_key', api_key)
        cli_config.set('token', None) # Clear token if API key is set
    else:
        # Prompt for clearing API key if user didn't provide one
        if cli_config.get('api_key') and Confirm.ask("Clear existing API Key?"):
            cli_config.set('api_key', None)

    print_success("Configuration saved.")

@config.command()
def show():
    """Show current CLI configuration."""
    table = Table(title="CLI Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")

    for key, value in cli_config.config.items():
        # Hide sensitive keys
        if 'key' in key.lower() or 'token' in key.lower():
            value = '*' * 8 if value else "None"
        table.add_row(key, str(value))

    console.print(table)


# AUTH COMMANDS

@cli.group()
def auth():
    """Manage authentication credentials."""
    pass

@auth.command()
@click.option('--username', prompt=True, help='Your username for the framework.')
@click.option('--password', prompt=True, hide_input=True, help='Your password for the framework.')
@click.option('--save', is_flag=True, help='Save the received token to configuration.')
@async_command
@pass_api_client
async def login(api_client: FrameworkAPIClient, username: str, password: str, save: bool):
    """Log in to the framework and obtain a JWT token."""
    response = await api_client.post('/api/auth/login', {
        'username': username,
        'password': password
    })

    if response.get('success'):
        token = response['data']['access_token']
        user_info = response['data'].get('user_info', {'username': username})

        if save:
            cli_config.set('token', token)
            cli_config.set('username', user_info['username'])
            cli_config.set('api_key', None) # Clear API key if token is set
            print_success(f"Logged in as [bold]{user_info['username']}[/] and token saved.")
        else:
            print_success(f"Logged in as [bold]{user_info['username']}[/]")
            print_info(f"Temporary Token: [dim]{token}[/]")
            print_info("Use '--save' option to persist token.")
    else:
        print_error(f"Login failed: {response.get('message', 'Unknown error.')}")

@auth.command()
def logout():
    """Log out and clear saved authentication credentials."""
    cli_config.set('token', None)
    cli_config.set('username', None)
    cli_config.set('api_key', None) # Clear API key too on logout
    print_success("Logged out successfully. All credentials cleared.")

@auth.command()
@click.option('--description', prompt=True, default='CLI Generated API Key',
              help='A description for the new API key.')
@click.option('--permissions', multiple=True,
              type=click.Choice([p.value for p in Permission]),
              default=[Permission.READ_AGENTS.value, Permission.EXECUTE_ACTIONS.value],
              help='Permissions to grant to this API key (can be specified multiple times).')
@async_command
@pass_api_client
async def create_api_key(api_client: FrameworkAPIClient, description: str, permissions: List[str]):
    """Create a new API key with specified permissions."""
    response = await api_client.post('/api/auth/api-keys', {
        'description': description,
        'permissions': permissions
    })

    if response.get('success'):
        api_key = response['data']['api_key']
        print_success("API key created successfully.")
        print_info(f"New API Key: [bold magenta]{api_key}[/]")
        print_warning("Please save this key securely. It will not be shown again.")

        if Confirm.ask("Save this API key to CLI configuration?"):
            cli_config.set('api_key', api_key)
            cli_config.set('token', None) # Clear token if API key is set
            print_success("API key saved to configuration.")
    else:
        print_error(f"Failed to create API key: {response.get('message', 'Unknown error.')}")


# AGENT COMMANDS

@cli.group()
def agents():
    """Manage autonomous agents within the framework."""
    pass

@agents.command(name="list") # Renamed to avoid conflict with Python's built-in list
@click.option('--namespace', help='Filter agents by their namespace.')
@click.option('--status', help='Filter agents by their status (e.g., active, idle, error).')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table',
              help='Output format for the agent list.')
@async_command
@pass_api_client
async def list_agents(api_client: FrameworkAPIClient, namespace: Optional[str], status: Optional[str], output_format: str):
    """List all registered agents."""
    params = {}
    if namespace:
        params['namespace'] = namespace
    if status:
        params['status'] = status

    query_string = '&'.join(f"{k}={v}" for k, v in params.items())
    endpoint = f"/api/agents?{query_string}" if query_string else "/api/agents"

    response = await api_client.get(endpoint)

    if response.get('success'):
        agents = response['data']

        if output_format == 'json':
            console.print_json(json.dumps(agents, indent=2))
        else:
            if not agents:
                print_info("No agents found matching the criteria.")
                return

            table = Table(title=f"Agents ({len(agents)} found)")
            table.add_column("ID", style="cyan", justify="left")
            table.add_column("Name", style="green", justify="left")
            table.add_column("Namespace", style="blue", justify="left")
            table.add_column("Status", style="magenta", justify="left")
            table.add_column("Created At", style="yellow", justify="left")

            for agent in agents:
                status_style = "bold green" if agent.get('status') == 'ACTIVE' else "bold yellow" if agent.get('status') == 'IDLE' else "bold red"
                table.add_row(
                    agent.get('id', 'N/A')[:8] + "...",
                    agent.get('name', 'N/A'),
                    agent.get('namespace', 'N/A'),
                    f"[{status_style}]{agent.get('status', 'N/A')}[/]",
                    format_datetime(agent.get('created_at'))
                )
            console.print(table)
    else:
        print_error(f"Failed to list agents: {response.get('message', 'Unknown error.')}")

@agents.command()
@click.argument('agent_id')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table',
              help='Output format for agent details.')
@async_command
@pass_api_client
async def show(api_client: FrameworkAPIClient, agent_id: str, output_format: str):
    """Show detailed information about a specific agent."""
    response = await api_client.get(f'/api/agents/{agent_id}')

    if response.get('success'):
        agent = response['data']

        if output_format == 'json':
            console.print_json(json.dumps(agent, indent=2))
        else:
            info = f"""
[bold cyan]ID:[/] {agent.get('id', 'N/A')}
[bold cyan]Name:[/] {agent.get('name', 'N/A')}
[bold cyan]Namespace:[/] {agent.get('namespace', 'N/A')}
[bold cyan]Status:[/] [bold {("green" if agent.get('status') == 'ACTIVE' else "yellow" if agent.get('status') == 'IDLE' else "red")}]{agent.get('status', 'N/A')}[/]
[bold cyan]Created At:[/] {format_datetime(agent.get('created_at'))}
[bold cyan]Last Heartbeat:[/] {format_datetime(agent.get('last_heartbeat'))}
            """
            console.print(Panel(info.strip(), title=f"Agent: {agent.get('name', 'N/A')}", border_style="blue"))

            # Capabilities
            if agent.get('capabilities'):
                cap_table = Table(title="Capabilities", show_header=True, header_style="bold magenta")
                cap_table.add_column("Name", style="green")
                cap_table.add_column("Namespace", style="blue")
                cap_table.add_column("Description", style="yellow")
                cap_table.add_column("Input Schema", style="dim")
                cap_table.add_column("Output Schema", style="dim")

                for cap in agent['capabilities']:
                    cap_table.add_row(
                        cap.get('name', 'N/A'),
                        cap.get('namespace', 'N/A'),
                        (cap.get('description', '')[:70] + "...") if len(cap.get('description', '')) > 70 else cap.get('description', 'N/A'),
                        json.dumps(cap.get('input_schema', {}), indent=2),
                        json.dumps(cap.get('output_schema', {}), indent=2)
                    )
                console.print(cap_table)
            else:
                print_info(f"No capabilities found for agent {agent_id}.")

            # Resources
            if agent.get('resources'):
                res_table = Table(title="Owned Resources", show_header=True, header_style="bold magenta")
                res_table.add_column("ID", style="cyan")
                res_table.add_column("Name", style="green")
                res_table.add_column("Type", style="blue")
                res_table.add_column("Created At", style="yellow")
                res_table.add_column("Size (bytes)", style="magenta")

                for res in agent['resources']:
                    res_table.add_row(
                        res.get('id', 'N/A')[:8] + "...",
                        res.get('name', 'N/A'),
                        res.get('type', 'N/A'),
                        format_datetime(res.get('created_at')),
                        str(res.get('size_bytes', 'N/A'))
                    )
                console.print(res_table)
            else:
                print_info(f"No resources found for agent {agent_id}.")
    else:
        print_error(f"Agent '{agent_id}' not found: {response.get('message', 'Unknown error.')}")

@agents.command()
@click.option('--namespace', prompt=True,
              type=click.Choice([
                  'agent.planning.strategist', 'agent.planning.workflow',
                  'agent.build.code.generator', 'agent.build.ux.generator',
                  'agent.test.generator', 'agent.security.sentinel',
                  'agent.monitor.progress'
              ]), help='The namespace of the agent to create.')
@click.option('--name', prompt=True, help='A unique name for the new agent.')
@click.option('--auto-start/--no-auto-start', is_flag=True, default=True,
              help='Whether the agent should start automatically upon creation.')
@async_command
@pass_api_client
async def create(api_client: FrameworkAPIClient, namespace: str, name: str, auto_start: bool):
    """Create a new agent instance."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task(f"Creating agent [bold green]{name}[/]...", total=None)
        response = await api_client.post('/api/agents', {
            'namespace': namespace,
            'name': name,
            'auto_start': auto_start
        })
        progress.remove_task(task)

    if response.get('success'):
        agent = response['data']
        print_success(f"Agent [bold]{agent.get('name', 'N/A')}[/] created successfully!")
        console.print(f"  [cyan]ID:[/][dim] {agent.get('id', 'N/A')}[/]")
        console.print(f"  [cyan]Namespace:[/][dim] {agent.get('namespace', 'N/A')}[/]")
        console.print(f"  [cyan]Status:[/][bold {('green' if agent.get('status') == 'ACTIVE' else 'yellow')}]{agent.get('status', 'N/A')}[/]")
    else:
        print_error(f"Failed to create agent: {response.get('message', 'Unknown error.')}")

@agents.command()
@click.argument('agent_id')
@click.option('--action', prompt=True, help='The name of the action to execute.')
@click.option('--params', help='JSON string of parameters for the action.', default='{}')
@async_command
@pass_api_client
async def execute(api_client: FrameworkAPIClient, agent_id: str, action: str, params: str):
    """Execute a specific action on an agent."""
    params_dict = {}
    try:
        params_dict = json.loads(params)
    except json.JSONDecodeError:
        print_error("Invalid JSON format for --params.")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task(f"Executing action [bold yellow]{action}[/] on agent [bold cyan]{agent_id[:8]}...[/]...", total=None)
        response = await api_client.post(f'/api/agents/{agent_id}/actions', {
            'action': action,
            'params': params_dict
        })
        progress.remove_task(task)

    if response.get('success'):
        result = response['data'].get('result')
        print_success(f"Action '[bold]{action}[/]' executed successfully!")
        if result:
            console.print(Panel(
                Syntax(json.dumps(result, indent=2), "json", theme="monokai", line_numbers=True),
                title="Action Result",
                border_style="green"
            ))
        else:
            print_info("Action returned no specific result data.")
    else:
        print_error(f"Failed to execute action: {response.get('message', 'Unknown error.')}")

@agents.command()
@click.argument('agent_id')
@click.option('--force', is_flag=True, help='Force deletion without confirmation.')
@async_command
@pass_api_client
async def delete(api_client: FrameworkAPIClient, agent_id: str, force: bool):
    """Delete an agent from the framework."""
    if not force:
        if not Confirm.ask(f"Are you sure you want to delete agent [bold red]{agent_id}[/]? This action is irreversible."):
            print_info("Operation cancelled.")
            return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task(f"Deleting agent [bold cyan]{agent_id[:8]}...[/]...", total=None)
        response = await api_client.delete(f'/api/agents/{agent_id}')
        progress.remove_task(task)

    if response.get('success'):
        print_success(f"Agent [bold]{agent_id}[/] deleted successfully.")
    else:
        print_error(f"Failed to delete agent: {response.get('message', 'Unknown error.')}")


# SYSTEM COMMANDS

@cli.group()
def system():
    """Manage and monitor the overall framework system."""
    pass

@system.command()
@async_command
@pass_api_client
async def status(api_client: FrameworkAPIClient):
    """Display the current health and operational status of the framework."""
    health_response = await api_client.get('/api/health')
    metrics_response = await api_client.get('/api/metrics')

    health_data = health_response.get('data', {})
    metrics_data = metrics_response.get('data', {})

    # System health panel
    health_status = health_data.get('status', 'UNKNOWN')
    health_style = "bold green" if health_status == 'healthy' else "bold red"
    health_info = f"""
[bold cyan]Overall Status:[/] [{health_style}]{health_status.upper()}[/]
[bold cyan]Framework Running:[/] {'[green]✅ Yes[/]' if health_data.get('framework', {}).get('running', False) else '[red]❌ No[/]'}
[bold cyan]Total Agents:[/] {health_data.get('framework', {}).get('total_agents', 'N/A')}
[bold cyan]Active Agents:[/] {health_data.get('framework', {}).get('active_agents', 'N/A')}
    """
    console.print(Panel(health_info.strip(), title="System Health Overview", border_style="blue"))

    # Metrics table
    metrics_table = Table(title="Key System Metrics", show_header=True, header_style="bold blue")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    metrics_table.add_column("Unit", style="dim")

    # Example: Flattening some common metrics, assuming nested structure from API
    relevant_metrics = {
        'cpu_usage_percent': {'path': 'system_metrics.cpu_usage_percent', 'unit': '%'},
        'memory_usage_percent': {'path': 'system_metrics.memory_usage_percent', 'unit': '%'},
        'disk_usage_percent': {'path': 'system_metrics.disk_usage_percent', 'unit': '%'},
        'agents_total': {'path': 'framework_metrics.total_agents', 'unit': 'agents'},
        'agents_active': {'path': 'framework_metrics.active_agents', 'unit': 'agents'},
        'messages_processed_per_sec': {'path': 'message_bus_metrics.messages_processed_per_sec', 'unit': 'msg/s'},
        'active_alerts': {'path': 'monitoring_metrics.active_alerts', 'unit': 'alerts'},
        'backups_last_successful': {'path': 'backup_metrics.last_successful_backup_time', 'unit': ''},
    }

    def get_nested_value(d, path):
        parts = path.split('.')
        for part in parts:
            if isinstance(d, dict) and part in d:
                d = d[part]
            else:
                return 'N/A'
        return d

    for metric_name, details in relevant_metrics.items():
        value = get_nested_value(metrics_data, details['path'])
        if isinstance(value, (int, float)):
            value = f"{value:.2f}" if isinstance(value, float) else str(value)
        elif isinstance(value, str) and 'timestamp' in metric_name:
             value = format_datetime(value)

        metrics_table.add_row(
            metric_name.replace('_', ' ').title(),
            str(value),
            details['unit']
        )
    console.print(metrics_table)

@system.command()
def logs():
    """Show system logs (placeholder - would stream from actual logging system)."""
    print_info("Log streaming is not implemented in this demo.")
    print_info("In a real implementation, this would connect to the central logging system (e.g., ELK, Splunk).")


# DEPLOYMENT COMMANDS

@cli.group()
def deploy():
    """Manage deployment configurations and artifacts."""
    pass

@deploy.command()
@click.option('--environment', type=click.Choice(['development', 'staging', 'production']),
              prompt=True, help='The target environment for the deployment.')
@click.option('--strategy', type=click.Choice(['docker', 'kubernetes']), prompt=True,
              help='The deployment strategy to use (e.g., Docker Compose, Kubernetes manifests).')
@click.option('--output-dir', type=click.Path(file_okay=False, writable=True, path_type=Path),
              default='./deployment_artifacts', help='Directory to save generated deployment files.')
@click.option('--domain', help='Optional: Domain name for production deployments (e.g., example.com).')
@click.option('--db-url', help='Optional: Database connection URL for production (e.g., postgresql://user:pass@host:port/db).')
@async_command
async def generate(environment: str, strategy: str, output_dir: Path, domain: Optional[str], db_url: Optional[str]):
    """Generate deployment files based on environment and strategy."""
    orchestrator = DeploymentOrchestrator() # This would typically be initialized with framework context

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Generating deployment files...", total=None)

        kwargs = {}
        if domain:
            kwargs['domain'] = domain
        if db_url:
            kwargs['db_url'] = db_url

        try:
            config = orchestrator.create_deployment_config(
                DeploymentEnvironment(environment),
                DeploymentStrategy(strategy),
                **kwargs
            )
            # In a real scenario, `deploy` would write files to output_dir
            # For this demo, we'll simulate it.
            success = await orchestrator.deploy(config, output_dir) # Simulate deployment logic

            if success:
                print_success(f"Deployment files for [bold]{environment}[/] ([bold]{strategy}[/]) generated in [dim]{output_dir}[/]")
                console.print(Panel(
                    f"To deploy, navigate to [bold cyan]{output_dir}[/] and execute the generated script:\n"
                    f"  [green]$ cd {output_dir}\n"
                    f"  $ ./deploy.sh[/]",
                    title="Next Steps",
                    border_style="yellow"
                ))
            else:
                print_error("Failed to generate deployment files. Check parameters and orchestrator logic.")
        except Exception as e:
            print_error(f"Error during deployment file generation: {e}")
        finally:
            progress.remove_task(task)


# INTERACTIVE MODE

@cli.command()
def interactive():
    """Start interactive mode for simplified agent and system management."""
    console.print(Panel.fit(
        "[bold blue]🤖 Agent Framework Interactive Mode[/]\n\n"
        "Type 'help' for available commands or 'exit' to quit.",
        title="Welcome",
        border_style="green"
    ))

    # A simple dictionary to map interactive commands to their async functions
    interactive_commands = {
        "agents list": list_agents_interactive,
        "system status": system_status_interactive,
        "exit": lambda: sys.exit(0),
        "quit": lambda: sys.exit(0),
        "q": lambda: sys.exit(0)
    }

    # Initialize a temporary API client for interactive mode
    # This ensures each interactive session has its own client
    api_client_for_interactive = FrameworkAPIClient()

    while True:
        try:
            command_line = Prompt.ask("[bold cyan]framework>[/]", default="help").strip()
            if not command_line:
                continue

            if command_line.lower() == 'help':
                print_interactive_help()
                continue

            # Basic command parsing for interactive mode
            parts = command_line.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            found_command = False
            for interactive_cmd_prefix, func in interactive_commands.items():
                if command_line.startswith(interactive_cmd_prefix):
                    found_command = True
                    try:
                        # For simplicity, interactive commands directly call their async handlers
                        # In a more complex interactive mode, you'd parse args more robustly
                        asyncio.run(func(api_client_for_interactive))
                    except click.ClickException as e:
                        print_error(e.message)
                    except Exception as e:
                        print_error(f"Error executing command: {e}")
                    break
            
            if not found_command:
                print_warning(f"Unknown command: '{command_line}'. Type 'help' for available commands.")

        except EOFError: # Ctrl-D
            print_info("\nGoodbye!")
            break
        except KeyboardInterrupt: # Ctrl-C
            print_info("\nGoodbye!")
            break
        except Exception as e:
            print_error(f"An unexpected error occurred in interactive mode: {e}")

def print_interactive_help():
    """Print help message for interactive mode."""
    help_text = """
[bold cyan]Available Commands (Interactive Mode):[/]

[green]agents list[/]       - List all agents with simplified view.
[green]system status[/]     - Show overall system health and key metrics.
[green]help[/]              - Show this help message.
[green]exit[/] / [green]quit[/] / [green]q[/] - Exit interactive mode.

[yellow]Note:[/] Interactive mode offers a subset of CLI functionalities.
For advanced operations and options, please use the full 'framework_cli.py <command> <subcommand> --options' syntax.
    """
    console.print(Panel(help_text.strip(), title="Interactive Help", border_style="blue"))

async def list_agents_interactive(api_client: FrameworkAPIClient):
    """Interactive handler for listing agents."""
    response = await api_client.get('/api/agents')

    if response.get('success'):
        agents = response['data']
        if not agents:
            print_info("No agents found.")
            return

        tree = Tree("🤖 [bold green]Agents Overview[/]", guide_style="dim")
        for agent in agents:
            status = agent.get('status', 'UNKNOWN')
            status_icon = "🟢" if status == 'ACTIVE' else "🟡" if status == 'IDLE' else "🔴"
            agent_node = tree.add(f"{status_icon} [bold]{agent.get('name', 'N/A')}[/] ([dim]{agent.get('namespace', 'N/A')}[/])")
            agent_node.add(f"[cyan]ID:[/][dim] {agent.get('id', 'N/A')[:8]}...[/]")
            agent_node.add(f"[cyan]Status:[/][bold {('green' if status == 'ACTIVE' else 'yellow' if status == 'IDLE' else 'red')}]{status}[/]")
        console.print(tree)
    else:
        print_error(f"Failed to list agents: {response.get('message', 'Unknown error.')}")

async def system_status_interactive(api_client: FrameworkAPIClient):
    """Interactive handler for showing system status."""
    health_response = await api_client.get('/api/health')
    
    if health_response.get('success'):
        data = health_response['data']
        
        status = data.get('status', 'UNKNOWN')
        status_icon = "🟢" if status == 'healthy' else "🔴"
        console.print(Panel(
            f"{status_icon} [bold blue]System Health Status:[/][bold {('green' if status == 'healthy' else 'red')}]{status.upper()}[/]\n"
            f"[bold cyan]Framework Running:[/]{' [green]✅ Yes[/]' if data.get('framework', {}).get('running', False) else ' [red]❌ No[/]'}\n"
            f"[bold cyan]Total Agents:[/]{data.get('framework', {}).get('total_agents', 'N/A')}\n"
            f"[bold cyan]Active Agents:[/]{data.get('framework', {}).get('active_agents', 'N/A')}",
            title="System Status",
            border_style="yellow"
        ))
    else:
        print_error(f"Failed to get system status: {health_response.get('message', 'Unknown error.')}")


# MAIN ENTRY POINT

if __name__ == '__main__':
    try:
        # Before running the CLI, check if the API is configured
        if not cli_config.get('api_url'):
            print_warning("API URL not configured. Please run 'framework config setup' first.")
            # Optionally, you could prompt for setup here, but for a CLI, explicit command is better.

        cli(obj=FrameworkAPIClient()) # Pass an initial client instance to the context
    except click.ClickException as e:
        print_error(e.message)
        sys.exit(e.exit_code)
    except Exception as e:
        print_error(f"An unexpected fatal error occurred: {e}")
        logger.exception("Fatal error in CLI main execution:") # Log full traceback
        sys.exit(1)