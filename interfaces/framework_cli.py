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
from core.monitoring_system import MonitoringOrchestrator

console = Console()

# ================================
# CLI CONFIGURATION
# ================================

class CLIConfig:
    """Configuración de la CLI"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".agent-framework"
        self.config_file = self.config_dir / "config.yaml"
        self.config_dir.mkdir(exist_ok=True)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Cargar configuración"""
        if self.config_file.exists():
            with open(self.config_file) as f:
                return yaml.safe_load(f) or {}
        return {}
        
    def save_config(self):
        """Guardar configuración"""
        with open(self.config_file, "w") as f:
            yaml.dump(self.config, f)
            
    def get(self, key: str, default: Any = None) -> Any:
        """Obtener valor de configuración"""
        return self.config.get(key, default)
        
    def set(self, key: str, value: Any):
        """Establecer valor de configuración"""
        self.config[key] = value
        self.save_config()

cli_config = CLIConfig()

# ================================
# API CLIENT
# ================================

class FrameworkAPIClient:
    """Cliente para la API del framework"""
    
    def __init__(self, base_url: str = None, api_key: str = None, token: str = None):
        self.base_url = base_url or cli_config.get("api_url", "http://localhost:8000")
        self.api_key = api_key or cli_config.get("api_key")
        self.token = token or cli_config.get("token")
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    def _get_headers(self) -> Dict[str, str]:
        """Obtener headers de autenticación"""
        headers = {"Content-Type": "application/json"}
        
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        elif self.api_key:
            headers["X-API-Key"] = self.api_key
            
        return headers
        
    async def get(self, endpoint: str) -> Dict[str, Any]:
        """GET request"""
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        
        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise click.ClickException(f"API Error ({response.status}): {error_text}")
                
    async def post(self, endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """POST request"""
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        
        async with self.session.post(url, headers=headers, json=data) as response:
            if response.status in [200, 201]:
                return await response.json()
            else:
                error_text = await response.text()
                raise click.ClickException(f"API Error ({response.status}): {error_text}")
                
    async def delete(self, endpoint: str) -> Dict[str, Any]:
        """DELETE request"""
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        
        async with self.session.delete(url, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise click.ClickException(f"API Error ({response.status}): {error_text}")

# ================================
# UTILITY FUNCTIONS
# ================================

def print_success(message: str):
    """Imprimir mensaje de éxito"""
    console.print(f"✅ {message}", style="bold green")
    
def print_error(message: str):
    """Imprimir mensaje de error"""
    console.print(f"❌ {message}", style="bold red")
    
def print_warning(message: str):
    """Imprimir mensaje de advertencia"""
    console.print(f"⚠️ {message}", style="bold yellow")
    
def print_info(message: str):
    """Imprimir mensaje informativo"""
    console.print(f"ℹ️ {message}", style="bold blue")

async def run_async_command(coro):
    """Ejecutar comando asíncrono"""
    try:
        return await coro
    except Exception as e:
        print_error(f"Command failed: {str(e)}")
        sys.exit(1)

def format_datetime(dt_str: str) -> str:
    """Formatear datetime string"""
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return dt_str

# ================================
# MAIN CLI GROUP
# ================================

@click.group()
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--config-file', type=click.Path(), help='Configuration file path')
def cli(debug, config_file):
    """
    🤖 Agent Framework CLI
    
    Manage autonomous agents, deployments, and system operations.
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    
    if config_file:
        cli_config.config_file = Path(config_file)
        cli_config.config = cli_config._load_config()

# ================================
# CONFIG COMMANDS
# ================================

@cli.group()
def config():
    """Configuration management"""
    pass

@config.command()
@click.option('--api-url', prompt=True, default=lambda: cli_config.get('api_url', 'http://localhost:8000'))
@click.option('--api-key', help='API key for authentication')
def setup(api_url, api_key):
    """Setup CLI configuration"""
    cli_config.set('api_url', api_url)
    
    if api_key:
        cli_config.set('api_key', api_key)
    
    print_success("Configuration saved")

@config.command()
def show():
    """Show current configuration"""
    table = Table(title="CLI Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")
    
    for key, value in cli_config.config.items():
        # Ocultar claves sensibles
        if 'key' in key.lower() or 'token' in key.lower():
            value = '*' * 8
        table.add_row(key, str(value))
    
    console.print(table)

# ================================
# AUTH COMMANDS
# ================================

@cli.group()
def auth():
    """Authentication management"""
    pass

@auth.command()
@click.option('--username', prompt=True)
@click.option('--password', prompt=True, hide_input=True)
@click.option('--save', is_flag=True, help='Save token to config')
def login(username, password, save):
    """Login to the framework"""
    
    async def do_login():
        async with FrameworkAPIClient() as client:
            try:
                response = await client.post('/api/auth/login', {
                    'username': username,
                    'password': password
                })
                
                if response.get('success'):
                    token = response['data']['access_token']
                    user_info = response['data']['user_info']
                    
                    if save:
                        cli_config.set('token', token)
                        cli_config.set('username', user_info['username'])
                    
                    print_success(f"Logged in as {user_info['username']}")
                    if not save:
                        print_info(f"Token: {token}")
                else:
                    print_error("Login failed")
                    
            except Exception as e:
                print_error(f"Login error: {e}")
    
    asyncio.run(do_login())

@auth.command()
def logout():
    """Logout and clear saved credentials"""
    cli_config.set('token', None)
    cli_config.set('username', None)
    print_success("Logged out successfully")

@auth.command()
@click.option('--description', prompt=True, default='CLI API Key')
@click.option('--permissions', multiple=True, 
              type=click.Choice(['read_agents', 'write_agents', 'create_agents', 'delete_agents', 
                               'execute_actions', 'admin_access']),
              default=['read_agents', 'execute_actions'])
def create_api_key(description, permissions):
    """Create a new API key"""
    
    async def do_create():
        async with FrameworkAPIClient() as client:
            response = await client.post('/api/auth/api-keys', {
                'description': description,
                'permissions': list(permissions)
            })
            
            if response.get('success'):
                api_key = response['data']['api_key']
                print_success("API key created")
                print_info(f"API Key: {api_key}")
                
                if Confirm.ask("Save API key to config?"):
                    cli_config.set('api_key', api_key)
                    print_success("API key saved to config")
            else:
                print_error("Failed to create API key")
    
    asyncio.run(do_create())

# ================================
# AGENT COMMANDS
# ================================

@cli.group()
def agents():
    """Agent management"""
    pass

@agents.command()
@click.option('--namespace', help='Filter by namespace')
@click.option('--status', help='Filter by status')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table')
def list(namespace, status, output_format):
    """List all agents"""
    
    async def do_list():
        async with FrameworkAPIClient() as client:
            params = {}
            if namespace:
                params['namespace'] = namespace
            if status:
                params['status'] = status
                
            # Construir query string
            query_string = '&'.join(f"{k}={v}" for k, v in params.items())
            endpoint = f"/api/agents?{query_string}" if query_string else "/api/agents"
            
            response = await client.get(endpoint)
            
            if response.get('success'):
                agents = response['data']
                
                if output_format == 'json':
                    console.print_json(json.dumps(agents, indent=2))
                else:
                    table = Table(title=f"Agents ({len(agents)} found)")
                    table.add_column("ID", style="cyan")
                    table.add_column("Name", style="green")
                    table.add_column("Namespace", style="blue")
                    table.add_column("Status", style="magenta")
                    table.add_column("Created", style="yellow")
                    
                    for agent in agents:
                        status_style = "green" if agent['status'] == 'active' else "red"
                        table.add_row(
                            agent['id'][:8] + "...",
                            agent['name'],
                            agent['namespace'],
                            f"[{status_style}]{agent['status']}[/]",
                            format_datetime(agent['created_at'])
                        )
                    
                    console.print(table)
            else:
                print_error("Failed to list agents")
    
    asyncio.run(do_list())

@agents.command()
@click.argument('agent_id')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table')
def show(agent_id, output_format):
    """Show detailed information about an agent"""
    
    async def do_show():
        async with FrameworkAPIClient() as client:
            response = await client.get(f'/api/agents/{agent_id}')
            
            if response.get('success'):
                agent = response['data']
                
                if output_format == 'json':
                    console.print_json(json.dumps(agent, indent=2))
                else:
                    # Agent info panel
                    info = f"""
[bold cyan]ID:[/] {agent['id']}
[bold cyan]Name:[/] {agent['name']}
[bold cyan]Namespace:[/] {agent['namespace']}
[bold cyan]Status:[/] {agent['status']}
[bold cyan]Created:[/] {format_datetime(agent['created_at'])}
[bold cyan]Last Heartbeat:[/] {format_datetime(agent['last_heartbeat'])}
                    """
                    console.print(Panel(info.strip(), title="Agent Information"))
                    
                    # Capabilities
                    if agent.get('capabilities'):
                        cap_table = Table(title="Capabilities")
                        cap_table.add_column("Name", style="green")
                        cap_table.add_column("Namespace", style="blue")
                        cap_table.add_column("Description", style="yellow")
                        
                        for cap in agent['capabilities']:
                            cap_table.add_row(
                                cap['name'],
                                cap['namespace'],
                                cap['description'][:50] + "..." if len(cap['description']) > 50 else cap['description']
                            )
                        
                        console.print(cap_table)
                    
                    # Resources
                    if agent.get('resources'):
                        res_table = Table(title="Resources")
                        res_table.add_column("ID", style="cyan")
                        res_table.add_column("Name", style="green")
                        res_table.add_column("Type", style="blue")
                        res_table.add_column("Created", style="yellow")
                        
                        for res in agent['resources']:
                            res_table.add_row(
                                res['id'][:8] + "...",
                                res['name'],
                                res['type'],
                                format_datetime(res['created_at'])
                            )
                        
                        console.print(res_table)
            else:
                print_error(f"Agent {agent_id} not found")
    
    asyncio.run(do_show())

@agents.command()
@click.option('--namespace', prompt=True, 
              type=click.Choice([
                  'agent.planning.strategist', 'agent.planning.workflow',
                  'agent.build.code.generator', 'agent.build.ux.generator',
                  'agent.test.generator', 'agent.security.sentinel',
                  'agent.monitor.progress'
              ]))
@click.option('--name', prompt=True)
@click.option('--auto-start', is_flag=True, default=True, help='Auto-start the agent')
def create(namespace, name, auto_start):
    """Create a new agent"""
    
    async def do_create():
        async with FrameworkAPIClient() as client:
            response = await client.post('/api/agents', {
                'namespace': namespace,
                'name': name,
                'auto_start': auto_start
            })
            
            if response.get('success'):
                agent = response['data']
                print_success(f"Agent created: {agent['id']}")
                print_info(f"Name: {agent['name']}")
                print_info(f"Namespace: {agent['namespace']}")
                print_info(f"Status: {agent['status']}")
            else:
                print_error("Failed to create agent")
    
    asyncio.run(do_create())

@agents.command()
@click.argument('agent_id')
@click.option('--action', prompt=True)
@click.option('--params', help='JSON parameters for the action')
def execute(agent_id, action, params):
    """Execute an action on an agent"""
    
    async def do_execute():
        params_dict = {}
        if params:
            try:
                params_dict = json.loads(params)
            except json.JSONDecodeError:
                print_error("Invalid JSON in params")
                return
        
        async with FrameworkAPIClient() as client:
            response = await client.post(f'/api/agents/{agent_id}/actions', {
                'action': action,
                'params': params_dict
            })
            
            if response.get('success'):
                result = response['data']['result']
                print_success(f"Action '{action}' executed successfully")
                
                if result:
                    console.print(Panel(
                        Syntax(json.dumps(result, indent=2), "json"),
                        title="Result"
                    ))
            else:
                print_error("Failed to execute action")
    
    asyncio.run(do_execute())

@agents.command()
@click.argument('agent_id')
@click.option('--force', is_flag=True, help='Force deletion without confirmation')
def delete(agent_id, force):
    """Delete an agent"""
    
    if not force:
        if not Confirm.ask(f"Are you sure you want to delete agent {agent_id}?"):
            print_info("Operation cancelled")
            return
    
    async def do_delete():
        async with FrameworkAPIClient() as client:
            response = await client.delete(f'/api/agents/{agent_id}')
            
            if response.get('success'):
                print_success(f"Agent {agent_id} deleted")
            else:
                print_error("Failed to delete agent")
    
    asyncio.run(do_delete())

# ================================
# SYSTEM COMMANDS
# ================================

@cli.group()
def system():
    """System management"""
    pass

@system.command()
def status():
    """Show system status"""
    
    async def do_status():
        async with FrameworkAPIClient() as client:
            try:
                health = await client.get('/api/health')
                metrics = await client.get('/api/metrics')
                
                if health.get('success') and metrics.get('success'):
                    health_data = health['data']
                    metrics_data = metrics['data']
                    
                    # System health panel
                    health_info = f"""
[bold green]Status:[/] {health_data['status']}
[bold cyan]Total Agents:[/] {health_data['framework']['total_agents']}
[bold cyan]Active Agents:[/] {health_data['framework']['active_agents']}
[bold cyan]Framework Running:[/] {'✅' if health_data['framework']['running'] else '❌'}
                    """
                    console.print(Panel(health_info.strip(), title="System Health"))
                    
                    # Metrics table
                    metrics_table = Table(title="System Metrics")
                    metrics_table.add_column("Metric", style="cyan")
                    metrics_table.add_column("Value", style="green")
                    
                    for key, value in metrics_data.items():
                        if isinstance(value, dict):
                            continue  # Skip complex objects for simple view
                        metrics_table.add_row(key.replace('_', ' ').title(), str(value))
                    
                    console.print(metrics_table)
                    
                else:
                    print_error("Failed to get system status")
                    
            except Exception as e:
                print_error(f"System unreachable: {e}")
    
    asyncio.run(do_status())

@system.command()
def logs():
    """Show system logs (placeholder - would stream from actual logging system)"""
    print_info("Log streaming not implemented in this demo")
    print_info("In a real implementation, this would connect to the logging system")

# ================================
# DEPLOYMENT COMMANDS
# ================================

@cli.group()
def deploy():
    """Deployment management"""
    pass

@deploy.command()
@click.option('--environment', type=click.Choice(['development', 'staging', 'production']), 
              prompt=True)
@click.option('--strategy', type=click.Choice(['docker', 'kubernetes']), prompt=True)
@click.option('--output-dir', default='./deployment')
@click.option('--domain', help='Domain for production deployments')
@click.option('--db-url', help='Database URL for production')
def generate(environment, strategy, output_dir, domain, db_url):
    """Generate deployment files"""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating deployment files...", total=None)
        
        async def do_generate():
            orchestrator = DeploymentOrchestrator()
            
            kwargs = {}
            if domain:
                kwargs['domain'] = domain
            if db_url:
                kwargs['db_url'] = db_url
            
            config = orchestrator.create_deployment_config(
                DeploymentEnvironment(environment),
                DeploymentStrategy(strategy),
                **kwargs
            )
            
            success = await orchestrator.deploy(config, output_dir)
            
            progress.update(task, completed=True)
            
            if success:
                print_success(f"Deployment files generated in {output_dir}")
                print_info(f"Environment: {environment}")
                print_info(f"Strategy: {strategy}")
                print_info(f"Run: cd {output_dir} && ./deploy.sh")
            else:
                print_error("Failed to generate deployment files")
        
        asyncio.run(do_generate())

# ================================
# INTERACTIVE MODE
# ================================

@cli.command()
def interactive():
    """Start interactive mode"""
    console.print(Panel.fit(
        "[bold blue]🤖 Agent Framework Interactive Mode[/]\n\n"
        "Type 'help' for available commands or 'exit' to quit.",
        title="Welcome"
    ))
    
    while True:
        try:
            command = Prompt.ask("[bold cyan]framework[/]", default="help")
            
            if command.lower() in ['exit', 'quit', 'q']:
                print_info("Goodbye!")
                break
            elif command.lower() == 'help':
                print_interactive_help()
            elif command.startswith('agents list'):
                asyncio.run(run_async_command(list_agents_interactive()))
            elif command.startswith('system status'):
                asyncio.run(run_async_command(system_status_interactive()))
            else:
                print_warning(f"Unknown command: {command}")
                print_info("Type 'help' for available commands")
                
        except KeyboardInterrupt:
            print_info("\nGoodbye!")
            break
        except Exception as e:
            print_error(f"Error: {e}")

def print_interactive_help():
    """Print help for interactive mode"""
    help_text = """
[bold cyan]Available Commands:[/]

[green]agents list[/]          - List all agents
[green]system status[/]        - Show system status
[green]help[/]                 - Show this help
[green]exit[/]                 - Exit interactive mode

[yellow]Note:[/] This is a simplified interactive mode.
Use the full CLI commands for advanced features.
    """
    console.print(Panel(help_text.strip(), title="Interactive Help"))

async def list_agents_interactive():
    """List agents in interactive mode"""
    async with FrameworkAPIClient() as client:
        response = await client.get('/api/agents')
        
        if response.get('success'):
            agents = response['data']
            
            if not agents:
                print_info("No agents found")
                return
            
            tree = Tree("🤖 Agents")
            for agent in agents:
                status_icon = "🟢" if agent['status'] == 'active' else "🔴"
                tree.add(f"{status_icon} {agent['name']} ({agent['namespace']})")
            
            console.print(tree)
        else:
            print_error("Failed to list agents")

async def system_status_interactive():
    """Show system status in interactive mode"""
    async with FrameworkAPIClient() as client:
        try:
            response = await client.get('/api/health')
            
            if response.get('success'):
                data = response['data']
                
                status_icon = "🟢" if data['status'] == 'healthy' else "🔴"
                console.print(f"{status_icon} System Status: {data['status']}")
                console.print(f"📊 Total Agents: {data['framework']['total_agents']}")
                console.print(f"✅ Active Agents: {data['framework']['active_agents']}")
            else:
                print_error("Failed to get system status")
                
        except Exception as e:
            print_error(f"System unreachable: {e}")

# ================================
# MAIN ENTRY POINT
# ================================

if __name__ == '__main__':
    try:
        cli()
    except KeyboardInterrupt:
        print_info("\nOperation cancelled")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)