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

from core.autonomous_agent_framework import AgentFramework
from core.security_system import SecurityManager, Permission, AuthenticationMethod
from core.persistence_system import PersistenceFactory, PersistenceBackend
from systems.deployment_system import DeploymentOrchestrator, DeploymentEnvironment, DeploymentStrategy
from core.backup_recovery_system import DisasterRecoveryOrchestrator, BackupType, BackupStatus
from core.monitoring_system import MonitoringOrchestrator, AlertSeverity, AlertStatus
from framework_config_utils import FrameworkBuilder, FrameworkConfig, AgentConfig # Importar las clases de config

console = Console()

class CLIConfig:
    def __init__(self):
        self.config_dir = Path.home() / ".agent-framework"
        self.config_file = self.config_dir / "config.yaml"
        self.config_dir.mkdir(exist_ok=True)
        self.config = self._load_config()

    def _load_config(self):
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        return {"api_url": "http://localhost:8000"}

    def _save_config(self):
        with open(self.config_file, 'w') as f:
            yaml.safe_dump(self.config, f)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self._save_config()

cli_config = CLIConfig()

def print_success(message: str):
    console.print(f"[bold green]✅ {message}[/bold green]")

def print_error(message: str):
    console.print(f"[bold red]❌ {message}[/bold red]", file=sys.stderr)

def print_info(message: str):
    console.print(f"[bold blue]ℹ️ {message}[/bold blue]")

class FrameworkAPIClient:
    def __init__(self, base_url: str = None, token: str = None):
        self.base_url = base_url or cli_config.get("api_url")
        self.token = token or cli_config.get("token")
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _request(self, method: str, path: str, **kwargs):
        headers = kwargs.pop('headers', {})
        if self.token:
            headers['Authorization'] = f"Bearer {self.token}"
        kwargs['headers'] = headers

        url = f"{self.base_url}{path}"
        try:
            async with self.session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            print_error(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = await e.response.json()
                    print_error(f"API Error Details: {error_details.get('error', {}).get('message', 'No message')}")
                except:
                    pass # Couldn't parse JSON error
            return {"success": False, "error": str(e)}
        except Exception as e:
            print_error(f"An unexpected error occurred: {e}")
            return {"success": False, "error": str(e)}

    async def get(self, path: str):
        return await self._request("GET", path)

    async def post(self, path: str, data: Dict[str, Any]):
        return await self._request("POST", path, json=data)

    async def put(self, path: str, data: Dict[str, Any]):
        return await self._request("PUT", path, json=data)

    async def delete(self, path: str):
        return await self._request("DELETE", path)

@click.group()
def cli():
    pass

@cli.group()
def config():
    pass

@config.command(name="set")
@click.argument("key")
@click.argument("value")
def set_config(key, value):
    cli_config.set(key, value)
    print_success(f"Config '{key}' set to '{value}'")

@config.command(name="get")
@click.argument("key")
def get_config(key):
    value = cli_config.get(key)
    if value is not None:
        print_info(f"Config '{key}': {value}")
    else:
        print_error(f"Config '{key}' not found.")

@config.command(name="show")
def show_config():
    panel_content = ""
    for key, value in cli_config.config.items():
        panel_content += f"[bold blue]{key}[/bold blue]: {value}\n"
    console.print(Panel(panel_content, title="CLI Configuration", border_style="cyan"))

@cli.command()
@click.option("--username", prompt="Username", help="Username for authentication")
@click.option("--password", prompt="Password", hide_input=True, confirmation_prompt=True, help="Password for authentication")
@click.option("--save-token", is_flag=True, help="Save token to config file")
@click.pass_context
def login(ctx, username, password, save_token):
    async def _login():
        async with FrameworkAPIClient() as client:
            response = await client.post('/api/auth/login', {'username': username, 'password': password})
            if response.get('success'):
                token = response['data']['token']
                print_success("Login successful!")
                print_info(f"Your token (valid for 1 hour): {token[:40]}...")
                if save_token:
                    cli_config.set("token", token)
                    print_success("Token saved to config.")
            else:
                print_error(f"Login failed: {response.get('error', {}).get('message', 'Unknown error')}")
    asyncio.run(_login())

@cli.command()
@click.option("--name", prompt="Agent Name", help="Name of the agent")
@click.option("--namespace", prompt="Agent Namespace", help="Namespace of the agent (e.g., agent.planning)")
@click.option("--agent-class", prompt="Agent Class Name", help="Python class name for the agent (e.g., StrategistAgent)")
def create_agent(name, namespace, agent_class):
    async def _create_agent():
        async with FrameworkAPIClient() as client:
            response = await client.post('/api/agents', {
                'name': name,
                'namespace': namespace,
                'agent_class_name': agent_class
            })
            if response.get('success'):
                agent_id = response['data']['agent_id']
                print_success(f"Agent '{name}' ({namespace}) created with ID: {agent_id}")
            else:
                print_error(f"Failed to create agent: {response.get('error', {}).get('message', 'Unknown error')}")
    asyncio.run(_create_agent())

@cli.command()
def list_agents():
    async def _list_agents():
        async with FrameworkAPIClient() as client:
            response = await client.get('/api/agents')
            if response.get('success'):
                agents = response['data']['agents']
                table = Table(title="Registered Agents")
                table.add_column("ID", style="cyan", no_wrap=True)
                table.add_column("Name", style="magenta")
                table.add_column("Namespace", style="green")
                table.add_column("Status", style="yellow")
                table.add_column("Last Heartbeat", style="blue")

                for agent in agents:
                    status_color = "green" if agent['status'] == 'active' else "red" if agent['status'] == 'error' else "yellow"
                    table.add_row(
                        agent['id'][:8] + "...",
                        agent['name'],
                        agent['namespace'],
                        f"[{status_color}]{agent['status']}[/{status_color}]",
                        datetime.fromisoformat(agent['last_heartbeat']).strftime('%Y-%m-%d %H:%M:%S')
                    )
                console.print(table)
            else:
                print_error(f"Failed to list agents: {response.get('error', {}).get('message', 'Unknown error')}")
    asyncio.run(_list_agents())

@cli.command()
@click.argument("agent_id")
@click.option("--action", prompt="Action Name", help="Action to execute")
@click.option("--params", help="JSON parameters for the action", default="{}")
def execute_agent_action(agent_id, action, params):
    async def _execute_action():
        try:
            params_dict = json.loads(params)
        except json.JSONDecodeError:
            print_error("Invalid JSON for parameters.")
            return

        async with FrameworkAPIClient() as client:
            response = await client.post(f'/api/agents/{agent_id}/execute', {
                'action': action,
                'params': params_dict
            })
            if response.get('success'):
                print_success(f"Action '{action}' executed on agent {agent_id}. Result: {response['data']}")
            else:
                print_error(f"Failed to execute action: {response.get('error', {}).get('message', 'Unknown error')}")
    asyncio.run(_execute_action())

@cli.command()
@click.argument("agent_id")
def get_agent_details(agent_id):
    async def _get_agent_details():
        async with FrameworkAPIClient() as client:
            response = await client.get(f'/api/agents/{agent_id}')
            if response.get('success'):
                agent = response['data']['agent']
                panel_content = f"""
[bold]ID:[/bold] {agent['id']}
[bold]Name:[/bold] {agent['name']}
[bold]Namespace:[/bold] {agent['namespace']}
[bold]Status:[/bold] {agent['status']}
[bold]Last Heartbeat:[/bold] {datetime.fromisoformat(agent['last_heartbeat']).strftime('%Y-%m-%d %H:%M:%S')}

[bold]Capabilities:[/bold]
"""
                if agent['capabilities']:
                    for cap in agent['capabilities']:
                        panel_content += f"  - [cyan]{cap['name']}[/cyan] ([italic]{cap['namespace']}[/italic]): {cap['description']}\n"
                else:
                    panel_content += "  No capabilities listed.\n"

                console.print(Panel(panel_content, title=f"Agent Details: {agent['name']}", border_style="green"))
            else:
                print_error(f"Failed to get agent details: {response.get('error', {}).get('message', 'Unknown error')}")
    asyncio.run(_get_agent_details())

@cli.command()
@click.argument("agent_id")
def delete_agent(agent_id):
    async def _delete_agent():
        if not Confirm.ask(f"Are you sure you want to delete agent {agent_id}?", default=False):
            print_info("Operation cancelled.")
            return

        async with FrameworkAPIClient() as client:
            response = await client.delete(f'/api/agents/{agent_id}')
            if response.get('success'):
                print_success(f"Agent {agent_id} deleted successfully.")
            else:
                print_error(f"Failed to delete agent: {response.get('error', {}).get('message', 'Unknown error')}")
    asyncio.run(_delete_agent())

@cli.command()
def health():
    async def _health():
        async with FrameworkAPIClient() as client:
            response = await client.get('/api/health')
            if response.get('success'):
                data = response['data']
                status_icon = "🟢" if data['status'] == 'healthy' else "🔴"
                panel_content = f"""
{status_icon} [bold]Overall Status:[/bold] {data['status']}
[bold]Timestamp:[/bold] {datetime.fromisoformat(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}

[bold]Framework:[/bold]
  [dim]Total Agents:[/dim] {data['framework']['total_agents']}
  [dim]Active Agents:[/dim] {data['framework']['active_agents']}
  [dim]Messages in Bus:[/dim] {data['framework']['messages_in_bus']}

[bold]Components Status:[/bold]
"""
                for comp, stat in data['components'].items():
                    comp_icon = "🟢" if stat['status'] == 'healthy' else "🔴"
                    panel_content += f"  {comp_icon} [dim]{comp}:[/dim] {stat['status']}\n"
                    if stat.get('details'):
                        for k, v in stat['details'].items():
                            panel_content += f"    - {k}: {v}\n"
                
                console.print(Panel(panel_content, title="System Health", border_style="cyan"))
            else:
                print_error(f"Failed to get health status: {response.get('error', {}).get('message', 'Unknown error')}")
    asyncio.run(_health())

@cli.command()
@click.option("--limit", default=10, help="Number of metrics to display.")
def metrics(limit):
    async def _metrics():
        async with FrameworkAPIClient() as client:
            response = await client.get('/api/metrics')
            if response.get('success'):
                metrics_data = response['data']['metrics']
                table = Table(title=f"System Metrics (Latest {limit})")
                table.add_column("Name", style="cyan")
                table.add_column("Value", style="magenta")
                table.add_column("Unit", style="green")
                table.add_column("Timestamp", style="blue")
                table.add_column("Tags", style="dim")

                for metric in list(metrics_data.values())[:limit]:
                    table.add_row(
                        metric['name'],
                        f"{metric['value']:.2f}",
                        metric['unit'],
                        datetime.fromisoformat(metric['timestamp']).strftime('%H:%M:%S'),
                        json.dumps(metric['tags']) if metric['tags'] else ""
                    )
                console.print(table)
            else:
                print_error(f"Failed to get metrics: {response.get('error', {}).get('message', 'Unknown error')}")
    asyncio.run(_metrics())

@cli.command()
@click.option("--limit", default=10, help="Number of alerts to display.")
@click.option("--status", type=click.Choice([s.value for s in AlertStatus]), default=None, help="Filter by alert status.")
@click.option("--severity", type=click.Choice([s.value for s in AlertSeverity]), default=None, help="Filter by alert severity.")
def alerts(limit, status, severity):
    async def _alerts():
        async with FrameworkAPIClient() as client:
            params = {}
            if status: params["status"] = status
            if severity: params["severity"] = severity
            params["limit"] = limit

            response = await client.get('/api/alerts')
            
            if response.get('success'):
                alerts_data = response['data']['alerts']
                table = Table(title=f"System Alerts (Latest {limit})")
                table.add_column("ID", style="cyan", no_wrap=True)
                table.add_column("Rule", style="magenta")
                table.add_column("Severity", style="red")
                table.add_column("Status", style="yellow")
                table.add_column("Timestamp", style="blue")
                table.add_column("Message", style="green")

                for alert in alerts_data:
                    severity_color = "red" if alert['severity'] in ['critical', 'fatal'] else "yellow" if alert['severity'] == 'warning' else "blue"
                    status_color = "green" if alert['status'] == 'resolved' else "red" if alert['status'] == 'active' else "yellow"
                    table.add_row(
                        alert['id'][:8] + "...",
                        alert['rule_name'],
                        f"[{severity_color}]{alert['severity']}[/{severity_color}]",
                        f"[{status_color}]{alert['status']}[/{status_color}]",
                        datetime.fromisoformat(alert['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
                        alert['message']
                    )
                console.print(table)
            else:
                print_error(f"Failed to get alerts: {response.get('error', {}).get('message', 'Unknown error')}")
    asyncio.run(_alerts())

@cli.command()
@click.option("--backup-type", type=click.Choice([bt.value for bt in BackupType]), prompt="Backup Type", help="Type of backup to create (full, incremental)")
@click.option("--parent-id", help="Parent backup ID for incremental backups")
def create_backup(backup_type, parent_id):
    async def _create_backup():
        async with FrameworkAPIClient() as client:
            data = {"backup_type": backup_type}
            if backup_type == BackupType.INCREMENTAL.value:
                if not parent_id:
                    print_error("Parent backup ID is required for incremental backups.")
                    return
                data["parent_id"] = parent_id
            
            response = await client.post('/api/backup/create', data)
            if response.get('success'):
                backup_meta = response['data']['backup']
                print_success(f"Backup '{backup_meta['backup_id']}' ({backup_meta['backup_type']}) initiated. Status: {backup_meta['status']}")
            else:
                print_error(f"Failed to create backup: {response.get('error', {}).get('message', 'Unknown error')}")
    asyncio.run(_create_backup())

@cli.command()
@click.argument("backup_id")
def restore_backup(backup_id):
    async def _restore_backup():
        if not Confirm.ask(f"WARNING: This will restore the system to backup '{backup_id}'. Current state will be lost. Continue?", default=False):
            print_info("Operation cancelled.")
            return

        async with FrameworkAPIClient() as client:
            response = await client.post(f'/api/backup/{backup_id}/restore', {})
            if response.get('success'):
                print_success(f"System restoration from backup '{backup_id}' initiated. Please check system health.")
            else:
                print_error(f"Failed to restore backup: {response.get('error', {}).get('message', 'Unknown error')}")
    asyncio.run(_restore_backup())

@cli.command()
@click.option("--limit", default=10, help="Number of backup records to display.")
def list_backups(limit):
    async def _list_backups():
        async with FrameworkAPIClient() as client:
            response = await client.get('/api/backup/history')
            if response.get('success'):
                backups = response['data']['history']
                table = Table(title=f"Backup History (Latest {limit})")
                table.add_column("ID", style="cyan", no_wrap=True)
                table.add_column("Type", style="magenta")
                table.add_column("Status", style="yellow")
                table.add_column("Created At", style="blue")
                table.add_column("Size (Bytes)", style="green")
                table.add_column("Parent ID", style="dim")

                for backup in backups[:limit]:
                    status_color = "green" if backup['status'] == BackupStatus.COMPLETED.value else "red" if backup['status'] == BackupStatus.FAILED.value else "yellow"
                    table.add_row(
                        backup['backup_id'][:8] + "...",
                        backup['backup_type'],
                        f"[{status_color}]{backup['status']}[/{status_color}]",
                        datetime.fromisoformat(backup['created_at']).strftime('%Y-%m-%d %H:%M:%S'),
                        str(backup['size_bytes']),
                        backup['parent_backup_id'][:8] + "..." if backup['parent_backup_id'] else ""
                    )
                console.print(table)
            else:
                print_error(f"Failed to list backups: {response.get('error', {}).get('message', 'Unknown error')}")
    asyncio.run(_list_backups())

@cli.command()
@click.option("--env", type=click.Choice([e.value for e in DeploymentEnvironment]), prompt="Deployment Environment", help="Environment to deploy to")
@click.option("--strategy", type=click.Choice([s.value for s in DeploymentStrategy]), prompt="Deployment Strategy", help="Strategy to use for deployment")
@click.option("--output-dir", prompt="Output Directory", help="Directory to save deployment files")
@click.option("--config-file", type=click.Path(exists=True), help="Optional YAML config file for detailed settings")
def deploy(env, strategy, output_dir, config_file):
    async def _deploy():
        deploy_config = {}
        if config_file:
            with open(config_file, 'r') as f:
                deploy_config = yaml.safe_load(f)

        framework_builder = FrameworkBuilder()
        full_framework_config = framework_builder.create_sample_config() # Get default structure
        
        # Merge CLI options and config file into a DeployConfig structure
        final_config_data = {
            "environment": env,
            "strategy": strategy,
            "framework_config": deploy_config.get("framework_config", full_framework_config.framework_config),
            "security_config": deploy_config.get("security_config", {}),
            "persistence_config": deploy_config.get("persistence_config", {}),
            "api_config": deploy_config.get("api_config", {}),
            "agents_config": deploy_config.get("agents_config", []),
            "monitoring_config": deploy_config.get("monitoring_config", {}),
            "scaling_config": deploy_config.get("scaling_config", {}),
        }
        
        # Ensure enums are correctly set
        final_config_data["environment"] = DeploymentEnvironment(final_config_data["environment"])
        final_config_data["strategy"] = DeploymentStrategy(final_config_data["strategy"])

        deployment_config = DeploymentConfig(**final_config_data)

        # We need a framework instance to pass to the DeploymentOrchestrator, even if it's not fully running
        temp_framework = AgentFramework() 
        orchestrator = DeploymentOrchestrator(temp_framework)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task(f"[cyan]Deploying to {env} with {strategy}...", total=100)
            
            # Simulate progress
            for i in range(1, 101):
                progress.update(task, advance=1)
                await asyncio.sleep(0.01) # Small delay for visualization

            success = await orchestrator.deploy(deployment_config, output_dir)
            
            if success:
                print_success(f"Deployment files generated successfully in {output_dir}")
                print_info("Next steps: Review generated files and run the deploy script.")
            else:
                print_error("Deployment failed. Check logs for details.")
    asyncio.run(_deploy())

@cli.command()
def list_deployments():
    async def _list_deployments():
        # Instantiate without a full framework, as this is just to list records
        temp_framework = AgentFramework() 
        orchestrator = DeploymentOrchestrator(temp_framework) 

        deployments = orchestrator.list_deployments()
        if not deployments:
            print_info("No deployments found.")
            return
        
        table = Table(title="Framework Deployments")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Environment", style="magenta")
        table.add_column("Strategy", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Deployed At", style="blue")
        table.add_column("Output Dir", style="dim")

        for dep in deployments:
            status_color = "green" if dep.status == 'COMPLETED' else "red" if dep.status == 'FAILED' else "yellow"
            table.add_row(
                dep.deployment_id[:8] + "...",
                dep.config.environment.value,
                dep.config.strategy.value,
                f"[{status_color}]{dep.status}[/{status_color}]",
                dep.deployed_at.strftime('%Y-%m-%d %H:%M:%S'),
                str(dep.output_dir)
            )
        console.print(table)
    asyncio.run(_list_deployments())

async def system_status_interactive():
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

@cli.command()
def interactive():
    print_info("Starting interactive monitoring mode... Press Ctrl+C to exit.")
    try:
        while True:
            asyncio.run(system_status_interactive())
            asyncio.run(list_agents_interactive())
            asyncio.run(alerts_interactive())
            asyncio.sleep(5) 
    except KeyboardInterrupt:
        print_info("\nInteractive mode stopped.")

async def list_agents_interactive():
    async with FrameworkAPIClient() as client:
        response = await client.get('/api/agents')
        if response.get('success'):
            agents = response['data']['agents']
            
            tree = Tree("🤖 Agents")
            for agent in agents:
                status_icon = "🟢" if agent['status'] == 'active' else "🔴"
                tree.add(f"{status_icon} {agent['name']} ({agent['namespace']}) - {agent['status']}")
            
            console.print(tree)
        else:
            print_error("Failed to list agents")

async def alerts_interactive():
    async with FrameworkAPIClient() as client:
        response = await client.get('/api/alerts')
        if response.get('success'):
            alerts_data = response['data']['alerts']
            if alerts_data:
                alert_panel_content = "[bold red]🚨 Active Alerts:[/bold red]\n"
                for alert in alerts_data:
                    if alert['status'] == AlertStatus.ACTIVE.value:
                        timestamp = datetime.fromisoformat(alert['timestamp']).strftime('%H:%M:%S')
                        alert_panel_content += f"  - [{alert['severity'].lower()}]{alert['severity']}:[/] {alert['message']} ([dim]{timestamp}[/dim])\n"
                if alert_panel_content != "[bold red]🚨 Active Alerts:[/bold red]\n":
                    console.print(Panel(alert_panel_content, border_style="red"))
                else:
                    print_info("No active alerts.")
            else:
                print_info("No alerts found.")
        else:
            print_error("Failed to retrieve alerts.")

if __name__ == '__main__':
    try:
        cli()
    except KeyboardInterrupt:
        print_info("\nOperation cancelled")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)