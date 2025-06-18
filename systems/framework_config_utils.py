import json
import yaml
import os
import asyncio
import logging
import argparse
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# Configure logging for the entire module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================================
# CONFIGURATION CLASSES
# ================================

@dataclass
class AgentConfig:
    """
    Configuration for a specific agent.

    Attributes:
        namespace (str): The namespace of the agent.
        name (str): The unique name of the agent.
        enabled (bool): Whether the agent is enabled. Defaults to True.
        auto_start (bool): Whether the agent should start automatically. Defaults to True.
        max_concurrent_tasks (int): Maximum number of concurrent tasks the agent can handle. Defaults to 10.
        heartbeat_interval (int): Interval in seconds for agent heartbeats. Defaults to 30.
        restart_on_failure (bool): Whether the agent should restart on failure. Defaults to True.
        custom_settings (Dict[str, Any]): Custom settings for the agent. Defaults to an empty dictionary.
    """
    namespace: str
    name: str
    enabled: bool = True
    auto_start: bool = True
    max_concurrent_tasks: int = 10
    heartbeat_interval: int = 30
    restart_on_failure: bool = True
    custom_settings: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initializes custom_settings to an empty dict if None."""
        if self.custom_settings is None:
            self.custom_settings = {}

@dataclass
class FrameworkConfig:
    """
    Main configuration for the autonomous agent framework.

    Attributes:
        name (str): Name of the framework. Defaults to "Autonomous Agent Framework".
        version (str): Version of the framework. Defaults to "1.0.0".
        message_queue_size (int): Size of the message queue. Defaults to 1000.
        message_timeout (int): Timeout for messages in seconds. Defaults to 30.
        enable_message_persistence (bool): Whether to enable message persistence. Defaults to False.
        max_resources (int): Maximum number of resources. Defaults to 10000.
        resource_cleanup_interval (int): Interval for resource cleanup in seconds. Defaults to 3600.
        enable_monitoring (bool): Whether to enable framework monitoring. Defaults to True.
        health_check_interval (int): Interval for health checks in seconds. Defaults to 60.
        metrics_collection (bool): Whether to enable metrics collection. Defaults to True.
        log_level (str): Logging level (e.g., "INFO", "DEBUG"). Defaults to "INFO".
        enable_agent_authentication (bool): Whether to enable agent authentication. Defaults to False.
        max_agent_lifetime (int): Maximum lifetime for agents in seconds. Defaults to 86400 (24 hours).
        agents (List[AgentConfig]): List of agent configurations. Defaults to an empty list.
    """
    name: str = "Autonomous Agent Framework"
    version: str = "1.0.0"
    
    message_queue_size: int = 1000
    message_timeout: int = 30
    enable_message_persistence: bool = False
    
    max_resources: int = 10000
    resource_cleanup_interval: int = 3600
    
    enable_monitoring: bool = True
    health_check_interval: int = 60
    metrics_collection: bool = True
    log_level: str = "INFO"
    
    enable_agent_authentication: bool = False
    max_agent_lifetime: int = 86400  # 24 hours
    
    agents: Optional[List[AgentConfig]] = None
    
    def __post_init__(self):
        """Initializes agents list to empty if None."""
        if self.agents is None:
            self.agents = []

class ConfigFormat(Enum):
    """Enumeration for supported configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml" # TOML requires a separate library like `toml` or `tomli`

# ================================
# CONFIGURATION MANAGER
# ================================

class ConfigManager:
    """
    Manages the loading, saving, and validation of framework configurations.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initializes the ConfigManager.

        Args:
            config_path (Optional[str]): The default path to the configuration file.
                                          Defaults to "agent_framework_config.yaml".
        """
        self.config_path = Path(config_path) if config_path else Path("agent_framework_config.yaml")
        self.config: FrameworkConfig = FrameworkConfig()
        
    def load_config(self, path: Optional[str] = None) -> FrameworkConfig:
        """
        Loads configuration from a specified file.

        Args:
            path (Optional[str]): The path to the configuration file. If None,
                                  uses the default `self.config_path`.

        Returns:
            FrameworkConfig: The loaded configuration. If the file is not found
                             or an error occurs, returns the default configuration.
        """
        config_file = Path(path) if path else self.config_path
        
        if not config_file.exists():
            logger.warning(f"Config file {config_file} not found, using defaults.")
            return self.config
            
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix == '.json':
                    data = json.load(f)
                elif config_file.suffix in ('.yaml', '.yml'):
                    data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_file.suffix}")
                    
            # Convert agent configs
            if 'agents' in data and data['agents'] is not None:
                agents = []
                for agent_data in data['agents']:
                    agents.append(AgentConfig(**agent_data))
                data['agents'] = agents
                
            self.config = FrameworkConfig(**data)
            logger.info(f"Configuration loaded from {config_file}.")
            
        except Exception as e:
            logger.error(f"Error loading config from {config_file}: {e}", exc_info=True)
            
        return self.config
        
    def save_config(self, config: FrameworkConfig, path: Optional[str] = None, 
                    format: ConfigFormat = ConfigFormat.YAML):
        """
        Saves the given configuration to a file.

        Args:
            config (FrameworkConfig): The configuration object to save.
            path (Optional[str]): The path to save the configuration file. If None,
                                  uses the default `self.config_path`.
            format (ConfigFormat): The format to save the configuration (JSON, YAML, or TOML).
                                   Defaults to YAML.
        """
        config_file = Path(path) if path else self.config_path
        
        # Convert to dict
        config_dict = asdict(config)
        
        try:
            with open(config_file, 'w') as f:
                if format == ConfigFormat.JSON:
                    json.dump(config_dict, f, indent=2)
                elif format == ConfigFormat.YAML:
                    yaml.dump(config_dict, f, default_flow_style=False)
                elif format == ConfigFormat.TOML:
                    try:
                        import toml
                        toml.dump(config_dict, f)
                        logger.info(f"Configuration saved to {config_file} in TOML format.")
                        return
                    except ImportError:
                        logger.warning("TOML library not found. Saving as YAML instead.")
                        yaml.dump(config_dict, f, default_flow_style=False)
                else:
                    logger.warning(f"Unsupported config format '{format}'. Saving as YAML instead.")
                    yaml.dump(config_dict, f, default_flow_style=False)
                    
            logger.info(f"Configuration saved to {config_file}.")
            
        except Exception as e:
            logger.error(f"Error saving config to {config_file}: {e}", exc_info=True)
            
    def create_default_config(self) -> FrameworkConfig:
        """
        Creates and returns a default framework configuration.

        Returns:
            FrameworkConfig: A `FrameworkConfig` object with predefined default agents.
        """
        agents = [
            AgentConfig("agent.planning.strategist", "strategist"),
            AgentConfig("agent.planning.workflow", "workflow_designer"),
            AgentConfig("agent.build.code.generator", "code_generator"),
            AgentConfig("agent.build.ux.generator", "ux_generator"),
            AgentConfig("agent.test.generator", "test_generator"),
            AgentConfig("agent.security.sentinel", "security_sentinel"),
            AgentConfig("agent.monitor.progress", "progress_monitor"),
        ]
        
        config = FrameworkConfig(
            name="Default Agent Framework",
            agents=agents
        )
        
        logger.info("Default configuration created.")
        return config
        
    def validate_config(self, config: FrameworkConfig) -> List[str]:
        """
        Validates the given framework configuration.

        Args:
            config (FrameworkConfig): The configuration object to validate.

        Returns:
            List[str]: A list of error messages if validation fails, otherwise an empty list.
        """
        errors = []
        
        # Validate unique agent names
        agent_names = [agent.name for agent in config.agents]
        if len(agent_names) != len(set(agent_names)):
            errors.append("Duplicate agent names found.")
            
        # Validar namespaces (requires `ExtendedAgentFactory` to be available)
        try:
            from core.specialized_agents import ExtendedAgentFactory
            available_namespaces = ExtendedAgentFactory.list_available_namespaces()
            
            for agent in config.agents:
                if agent.namespace not in available_namespaces:
                    errors.append(f"Unknown namespace for agent '{agent.name}': {agent.namespace}.")
        except ImportError:
            logger.warning("Could not import ExtendedAgentFactory for namespace validation. Skipping.")
            errors.append("Namespace validation skipped: 'core.specialized_agents.ExtendedAgentFactory' not found.")
        
        # Validate numeric configurations
        if config.message_queue_size <= 0:
            errors.append("message_queue_size must be positive.")
            
        if config.health_check_interval <= 0:
            errors.append("health_check_interval must be positive.")
            
        if config.max_resources <= 0:
            errors.append("max_resources must be positive.")

        if config.resource_cleanup_interval <= 0:
            errors.append("resource_cleanup_interval must be positive.")
            
        if config.message_timeout <= 0:
            errors.append("message_timeout must be positive.")

        if config.max_agent_lifetime <= 0:
            errors.append("max_agent_lifetime must be positive.")
            
        return errors

---

## Framework Builder

class FrameworkBuilder:
    """
    Constructs and initializes the agent framework based on loaded configuration.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initializes the FrameworkBuilder.

        Args:
            config_path (Optional[str]): Path to the configuration file.
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        
    async def build_framework(self) -> tuple[Any, Dict[str, Any]]:
        """
        Builds the agent framework and initializes agents based on the configuration.

        Returns:
            tuple[Any, Dict[str, Any]]: A tuple containing the initialized framework instance
                                        and a dictionary of created agents.
        """
        try:
            from core.autonomous_agent_framework import AgentFramework
            from core.specialized_agents import ExtendedAgentFactory
        except ImportError as e:
            logger.critical(f"Failed to import core framework components: {e}. "
                            "Ensure 'core.autonomous_agent_framework' and 'core.specialized_agents' are available.")
            raise RuntimeError("Missing core framework dependencies.")

        # Configure logging based on framework settings
        logging.getLogger().setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))
        logger.info(f"Setting global log level to: {self.config.log_level.upper()}")
        
        # Create framework
        framework = AgentFramework()
        
        # Start framework (e.g., initialize message bus, resource manager)
        await framework.start()
        logger.info("Agent Framework started.")
        
        # Create configured agents
        agents = {}
        for agent_config in self.config.agents:
            if agent_config.enabled:
                try:
                    agent = ExtendedAgentFactory.create_agent(
                        agent_config.namespace,
                        agent_config.name,
                        framework # Pass framework instance to agent factory
                    )
                    
                    if agent_config.auto_start:
                        await agent.start()
                        logger.info(f"Agent '{agent_config.name}' started automatically.")
                    else:
                        logger.info(f"Agent '{agent_config.name}' created but not auto-started.")
                        
                    agents[agent_config.name] = agent
                    logger.info(f"Successfully created agent: {agent_config.name}.")
                    
                except Exception as e:
                    logger.error(f"Failed to create or start agent '{agent_config.name}': {e}", exc_info=True)
                    
        return framework, agents
        
    def create_sample_config(self, path: str = "sample_config.yaml"):
        """
        Creates a sample configuration file at the specified path.

        Args:
            path (str): The path where the sample configuration file will be saved.
                        Defaults to "sample_config.yaml".
        """
        config = self.config_manager.create_default_config()
        self.config_manager.save_config(config, path)
        logger.info(f"Sample configuration created at: {path}.")

---

## Utilities

class AgentOrchestrator:
    """
    Orchestrates complex workflows involving multiple agents.
    """
    
    def __init__(self, framework: Any, agents: Dict[str, Any]):
        """
        Initializes the AgentOrchestrator.

        Args:
            framework (Any): The initialized agent framework instance.
            agents (Dict[str, Any]): A dictionary of instantiated agents by their name.
        """
        self.framework = framework
        self.agents = agents
        
    async def execute_workflow(self, workflow_definition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a complete workflow defined by a dictionary.

        Args:
            workflow_definition (Dict[str, Any]): A dictionary defining the workflow steps,
                                                  agents, actions, parameters, and dependencies.

        Returns:
            Dict[str, Any]: A dictionary containing the results of each step,
                            keyed by step ID.
        """
        results = {}
        steps = workflow_definition.get("steps", [])
        
        for step in steps:
            step_id = step["id"]
            agent_name = step["agent"]
            action = step["action"]
            params = step.get("params", {})
            
            # Resolve dependencies by injecting previous step results into current step params
            if "dependencies" in step:
                for dep_id in step["dependencies"]:
                    if dep_id in results:
                        # Ensure we don't overwrite existing params if a dependency also has a 'params' key
                        # A more sophisticated merging strategy might be needed for complex cases
                        if isinstance(results[dep_id], dict):
                            params.update(results[dep_id])
                        else:
                            # If dependency result is not a dict, add it under a specific key or warn
                            params[f'{dep_id}_result'] = results[dep_id]
                    else:
                        logger.warning(f"Dependency '{dep_id}' for step '{step_id}' not found. Skipping dependency injection.")
                        
            # Execute step
            try:
                if agent_name in self.agents:
                    agent = self.agents[agent_name]
                    logger.info(f"Executing step '{step_id}' with agent '{agent_name}' and action '{action}'.")
                    result = await agent.execute_action(action, params)
                    results[step_id] = result
                    logger.info(f"Step '{step_id}' completed successfully.")
                else:
                    raise ValueError(f"Agent '{agent_name}' required for step '{step_id}' not found.")
                    
            except Exception as e:
                logger.error(f"Error in step '{step_id}' (Agent: {agent_name}, Action: {action}): {e}", exc_info=True)
                results[step_id] = {"error": str(e), "status": "failed"}
                
        return results
        
    async def create_development_pipeline(self, project_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates and executes a predefined development pipeline workflow.

        Args:
            project_spec (Dict[str, Any]): A dictionary containing project specifications
                                          (e.g., requirements, tasks, code specifications).

        Returns:
            Dict[str, Any]: The results of the executed development pipeline.
        """
        logger.info("Creating and executing development pipeline.")
        pipeline = {
            "steps": [
                {
                    "id": "strategy",
                    "agent": "strategist",
                    "action": "define.strategy",
                    "params": {"requirements": project_spec.get("requirements", "")}
                },
                {
                    "id": "workflow",
                    "agent": "workflow_designer", 
                    "action": "design.workflow",
                    "params": {"tasks": project_spec.get("tasks", [])},
                    "dependencies": ["strategy"]
                },
                {
                    "id": "code_generation",
                    "agent": "code_generator",
                    "action": "generate.component",
                    "params": {"specification": project_spec.get("code_spec", {})},
                    "dependencies": ["workflow"]
                },
                {
                    "id": "test_generation",
                    "agent": "test_generator",
                    "action": "generate.tests",
                    "params": {"test_framework": "pytest"},
                    "dependencies": ["code_generation"]
                },
                {
                    "id": "security_scan",
                    "agent": "security_sentinel",
                    "action": "scan.vulnerabilities",
                    "params": {"scan_type": "comprehensive"},
                    "dependencies": ["code_generation"]
                }
            ]
        }
        
        return await self.execute_workflow(pipeline)

class MetricsCollector:
    """
    Collects and provides metrics about the framework's operation.
    """
    
    def __init__(self, framework: Any):
        """
        Initializes the MetricsCollector.

        Args:
            framework (Any): The initialized agent framework instance.
        """
        self.framework = framework
        self.metrics = {
            "agents_created": 0,
            "agents_terminated": 0,
            "messages_sent": 0,
            "resources_created": 0,
            "errors": 0
        }
        
    def record_agent_created(self):
        """Increments the count of agents created."""
        self.metrics["agents_created"] += 1
        logger.debug("Metric: agent_created incremented.")
        
    def record_agent_terminated(self):
        """Increments the count of agents terminated."""
        self.metrics["agents_terminated"] += 1
        logger.debug("Metric: agent_terminated incremented.")
        
    def record_message_sent(self):
        """Increments the count of messages sent."""
        self.metrics["messages_sent"] += 1
        logger.debug("Metric: message_sent incremented.")
        
    def record_resource_created(self):
        """Increments the count of resources created."""
        self.metrics["resources_created"] += 1
        logger.debug("Metric: resources_created incremented.")
        
    def record_error(self):
        """Increments the count of errors."""
        self.metrics["errors"] += 1
        logger.debug("Metric: errors incremented.")
        
    def get_metrics(self) -> Dict[str, Any]:
        """
        Retrieves the current state of collected metrics.

        Returns:
            Dict[str, Any]: A dictionary of current metrics, including active agents
                            and a timestamp.
        """
        # Assuming framework has a registry and list_all_agents method
        try:
            active_agents = len(self.framework.registry.list_all_agents())
        except AttributeError:
            logger.warning("Framework registry not available or list_all_agents method missing. Active agents count may be inaccurate.")
            active_agents = -1 # Indicate unknown
        
        return {
            **self.metrics,
            "active_agents": active_agents,
            "timestamp": __import__("time").time() # Using __import__ for lazy import
        }
        
    def export_metrics(self, format: str = "json") -> str:
        """
        Exports the current metrics in a specified format.

        Args:
            format (str): The desired output format ("json", "yaml", or other). Defaults to "json".

        Returns:
            str: The metrics as a formatted string.
        """
        metrics = self.get_metrics()
        
        if format == "json":
            return json.dumps(metrics, indent=2)
        elif format == "yaml":
            return yaml.dump(metrics, default_flow_style=False)
        else:
            logger.warning(f"Unsupported export format '{format}'. Returning string representation.")
            return str(metrics)

---

## CLI Utilities

class FrameworkCLI:
    """
    Command-Line Interface for interacting with the autonomous agent framework.
    """
    
    def __init__(self):
        """Initializes the FrameworkCLI."""
        self.framework = None
        self.agents = {}
        self.parser = self._setup_arg_parser()
        
    def _setup_arg_parser(self) -> argparse.ArgumentParser:
        """
        Sets up the argument parser for the CLI.

        Returns:
            argparse.ArgumentParser: The configured argument parser.
        """
        parser = argparse.ArgumentParser(
            description="Manage the Autonomous Agent Framework.",
            prog="framework_cli"
        )
        
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Start command
        start_parser = subparsers.add_parser("start", help="Start the agent framework.")
        start_parser.add_argument(
            "--config", 
            type=str, 
            default="agent_framework_config.yaml", 
            help="Path to the framework configuration file."
        )

        # Stop command
        stop_parser = subparsers.add_parser("stop", help="Stop the agent framework.")

        # List agents command
        list_parser = subparsers.add_parser("list", help="List active agents.")

        # Send command to agent
        send_parser = subparsers.add_parser("send", help="Send a command to a specific agent.")
        send_parser.add_argument(
            "agent_name", 
            type=str, 
            help="Name of the agent to send the command to."
        )
        send_parser.add_argument(
            "action", 
            type=str, 
            help="Action to execute on the agent."
        )
        send_parser.add_argument(
            "--params", 
            type=str, 
            default="{}", 
            help="JSON string of parameters for the action (e.g., '{\"key\": \"value\"}')."
        )

        # Show resources command
        resources_parser = subparsers.add_parser("resources", help="Show resources managed by the framework.")

        # Create sample config command
        create_config_parser = subparsers.add_parser("create-sample-config", help="Create a sample configuration file.")
        create_config_parser.add_argument(
            "--path", 
            type=str, 
            default="sample_config.yaml", 
            help="Path to save the sample configuration file."
        )
        
        return parser

    async def start_framework(self, config_path: Optional[str] = None):
        """
        Starts the framework from the CLI, loading configuration and initializing agents.

        Args:
            config_path (Optional[str]): Path to the configuration file.
        """
        logger.info("Starting framework from CLI...")
        builder = FrameworkBuilder(config_path)
        try:
            self.framework, self.agents = await builder.build_framework()
            logger.info(f"Framework started successfully with {len(self.agents)} agents.")
            print(f"Framework started successfully with {len(self.agents)} agents.") # CLI feedback
        except Exception as e:
            logger.critical(f"Failed to start framework: {e}", exc_info=True)
            print(f"Error: Failed to start framework. Check logs for details.") # CLI feedback
            
    async def stop_framework(self):
        """Stops the running framework and all active agents."""
        logger.info("Stopping framework...")
        if self.framework:
            await self.framework.stop()
            logger.info("Framework stopped.")
            print("Framework stopped.") # CLI feedback
        else:
            logger.warning("No framework instance to stop.")
            print("No framework instance to stop.") # CLI feedback
        
    def list_agents(self):
        """Lists all active agents in the framework."""
        if not self.agents:
            logger.info("No agents currently active.")
            print("No agents active.") # CLI feedback
            return
            
        logger.info("Listing active agents.")
        print("Active agents:") # CLI feedback
        for name, agent in self.agents.items():
            # Assuming agent has a .status attribute, adjust if needed
            agent_status = getattr(agent, 'status', 'UNKNOWN')
            print(f"  - {name} ({agent.namespace}) - Status: {agent_status}")
            
    async def send_command(self, agent_name: str, action: str, params: str = "{}"):
        """
        Sends a command (action) to a specific agent.

        Args:
            agent_name (str): The name of the target agent.
            action (str): The action to execute on the agent.
            params (str): JSON string of parameters for the action. Defaults to "{}".
        """
        if agent_name not in self.agents:
            logger.error(f"Attempted to send command to non-existent agent: {agent_name}.")
            print(f"Error: Agent '{agent_name}' not found.") # CLI feedback
            return
            
        try:
            params_dict = json.loads(params)
            agent = self.agents[agent_name]
            logger.info(f"Sending command to agent '{agent_name}': action='{action}', params={params_dict}.")
            result = await agent.execute_action(action, params_dict)
            print(f"Result from {agent_name}: {json.dumps(result, indent=2)}") # CLI feedback
            logger.info(f"Command to '{agent_name}' executed successfully.")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON parameters for command: {params}", exc_info=True)
            print(f"Error: Invalid JSON format for parameters. Please use valid JSON string.") # CLI feedback
        except Exception as e:
            logger.error(f"Error executing command on agent '{agent_name}': {e}", exc_info=True)
            print(f"Error executing command: {e}") # CLI feedback
            
    def show_resources(self):
        """Displays resources managed by the framework's resource manager."""
        if not self.framework or not hasattr(self.framework, 'resource_manager'):
            logger.warning("Framework not started or resource manager not available.")
            print("Framework not started or resource manager not available.") # CLI feedback
            return
            
        all_resources = []
        for agent in self.agents.values():
            try:
                # Assuming resource_manager.find_resources_by_owner exists and works with agent.id
                agent_resources = self.framework.resource_manager.find_resources_by_owner(agent.id)
                all_resources.extend(agent_resources)
            except AttributeError:
                logger.warning(f"Agent '{agent.name}' does not have an 'id' attribute or resource manager cannot query by owner ID.")
            except Exception as e:
                logger.error(f"Error fetching resources for agent '{agent.name}': {e}", exc_info=True)
                
        if not all_resources:
            logger.info("No resources found in the framework.")
            print("No resources found.") # CLI feedback
            return
            
        logger.info("Displaying framework resources.")
        print("Managed Resources:") # CLI feedback
        for resource in all_resources:
            # Assuming resource has .type (Enum), .name, .owner_agent_id attributes
            resource_type = getattr(resource, 'type', 'UNKNOWN_TYPE')
            resource_name = getattr(resource, 'name', 'UNKNOWN_NAME')
            owner_id = getattr(resource, 'owner_agent_id', 'UNKNOWN_OWNER')
            print(f"  - Type: {resource_type.value if hasattr(resource_type, 'value') else resource_type}, Name: {resource_name} (Owner: {owner_id})")

    def run(self):
        """
        Parses command-line arguments and executes the corresponding framework command.
        This is the main entry point for the CLI.
        """
        args = self.parser.parse_args()

        if not hasattr(args, 'command') or args.command is None:
            self.parser.print_help()
            return

        # Execute the chosen command
        try:
            if args.command == "start":
                asyncio.run(self.start_framework(args.config))
            elif args.command == "stop":
                asyncio.run(self.stop_framework())
            elif args.command == "list":
                self.list_agents()
            elif args.command == "send":
                asyncio.run(self.send_command(args.agent_name, args.action, args.params))
            elif args.command == "resources":
                self.show_resources()
            elif args.command == "create-sample-config":
                builder = FrameworkBuilder() # Use a builder instance for this action
                builder.create_sample_config(args.path)
            else:
                self.parser.print_help()
        except Exception as e:
            logger.critical(f"An unhandled error occurred during CLI command '{args.command}': {e}", exc_info=True)
            print(f"An unexpected error occurred: {e}")

---

## Example Usage

async def example_with_config():
    """
    Demonstrates the usage of the framework configuration, builder,
    orchestrator, and metrics collector through direct function calls.
    This is separate from the CLI example.
    """
    logger.info("Starting example_with_config demonstration.")
    
    # Create a sample configuration file
    builder = FrameworkBuilder()
    sample_config_path = "example_config.yaml"
    builder.create_sample_config(sample_config_path)
    logger.info(f"Sample configuration file generated at: {sample_config_path}")
    
    # Build the framework using the generated configuration
    logger.info("Building framework with configuration...")
    framework, agents = await builder.build_framework()
    
    # Create an orchestrator
    orchestrator = AgentOrchestrator(framework, agents)
    
    # Define a project specification for the development pipeline
    project_spec = {
        "requirements": "Develop a secure user management system with CRUD operations.",
        "tasks": [
            {"name": "Design API", "type": "design"},
            {"name": "Implement backend", "type": "development"},
            {"name": "Create tests", "type": "testing"},
            {"name": "Deploy", "type": "deployment"}
        ],
        "code_spec": {
            "name": "UserManagerService",
            "language": "Python",
            "framework": "FastAPI",
            "methods": [
                {"name": "create_user", "parameters": [{"name": "user_data", "type": "dict"}]},
                {"name": "get_user", "parameters": [{"name": "user_id", "type": "str"}]},
                {"name": "update_user", "parameters": [{"name": "user_id", "type": "str"}, {"name": "update_data", "type": "dict"}]},
                {"name": "delete_user", "parameters": [{"name": "user_id", "type": "str"}]}
            ]
        }
    }
    
    # Execute the development pipeline
    logger.info("Executing development pipeline...")
    pipeline_results = await orchestrator.create_development_pipeline(project_spec)
    
    print("\n--- Pipeline Execution Summary ---") # CLI feedback
    for step_id, result in pipeline_results.items():
        status_icon = '✓' if 'error' not in result else '✗'
        error_msg = f" (Error: {result['error']})" if 'error' in result else ''
        print(f"  {status_icon} Step '{step_id}': {'Completed' if 'error' not in result else 'Failed'}{error_msg}")
    print("----------------------------------\n")

    # Show framework metrics
    metrics = MetricsCollector(framework)
    logger.info("Collecting and exporting framework metrics.")
    print("Framework Metrics:\n", metrics.export_metrics(format="yaml")) # Export in YAML for a different view
    
    # Clean up: stop the framework
    logger.info("Stopping the framework and agents.")
    await framework.stop()
    logger.info("Example demonstration finished.")

if __name__ == "__main__":
    # --- DUMMY CLASSES FOR INDEPENDENT EXECUTION ---
    # These classes are mocks to allow this script to run even if the
    # 'core' framework components (autonomous_agent_framework, specialized_agents)
    # are not actually present in the Python environment.
    # In a real project, these imports would directly refer to your actual
    # framework implementation.
    try:
        from core.autonomous_agent_framework import AgentFramework
        from core.specialized_agents import ExtendedAgentFactory
    except ImportError:
        logger.warning("Could not import actual core framework components. Using dummy classes for demonstration.")
        
        class DummyAgent:
            def __init__(self, namespace, name, framework):
                self.namespace = namespace
                self.name = name
                self.framework = framework
                self.id = f"{namespace}:{name}"
                self.status = "IDLE"
            async def start(self):
                self.status = "RUNNING"
                logger.info(f"Dummy agent {self.name} started.")
            async def stop(self):
                self.status = "STOPPED"
                logger.info(f"Dummy agent {self.name} stopped.")
            async def execute_action(self, action, params):
                logger.info(f"Dummy agent {self.name} executing action {action} with {params}")
                # Simulate some work and return
                await asyncio.sleep(0.1)
                return {"status": "success", "action_executed": action, "params_received": params}

        class DummyAgentFramework:
            def __init__(self):
                self.registry = self # Mock registry to return self for list_all_agents
                self._agents = {} # Internal list to keep track of dummy agents
                self.resource_manager = self # Mock resource manager for this example
            async def start(self):
                logger.info("Dummy Agent Framework started.")
            async def stop(self):
                logger.info("Dummy Agent Framework stopped.")
                for agent in self._agents.values():
                    await agent.stop()
            def list_all_agents(self):
                return list(self._agents.values())
            def find_resources_by_owner(self, owner_id):
                # Simple mock for resources, adjust as needed for testing specific scenarios
                class DummyResource:
                    def __init__(self, r_type, name, owner):
                        # Ensure type is an Enum to match the expected behavior in show_resources
                        self.type = Enum('ResourceType', {'DATA': 'data', 'MODEL': 'model', 'CODE': 'code'})[r_type.upper()]
                        self.name = name
                        self.owner_agent_id = owner
                
                # Example: If a code generator agent exists, return a dummy code resource
                if owner_id == "agent.build.code.generator:code_generator":
                    return [DummyResource('code', 'generated_code_123', owner_id)]
                return []

        class DummyExtendedAgentFactory:
            _namespaces = {
                "agent.planning.strategist", "agent.planning.workflow",
                "agent.build.code.generator", "agent.build.ux.generator",
                "agent.test.generator", "agent.security.sentinel",
                "agent.monitor.progress"
            }
            @staticmethod
            def create_agent(namespace, name, framework):
                agent = DummyAgent(namespace, name, framework)
                framework._agents[name] = agent # Add to framework's internal agent list
                return agent
            @staticmethod
            def list_available_namespaces():
                return list(DummyExtendedAgentFactory._namespaces)
        
        # Override actual imports with dummy ones for local execution without full framework
        import sys
        sys.modules['core.autonomous_agent_framework'] = type('module', (object,), {'AgentFramework': DummyAgentFramework})
        sys.modules['core.specialized_agents'] = type('module', (object,), {'ExtendedAgentFactory': DummyExtendedAgentFactory})

    # Deciding whether to run the direct example or the CLI
    # You can comment out one or the other to test
    # asyncio.run(example_with_config()) # Direct function call example

    # To run the CLI:
    # 1. Save the code as a Python file (e.g., framework_cli.py)
    # 2. Open your terminal
    # 3. Try these commands:
    #    python framework_cli.py create-sample-config
    #    python framework_cli.py start --config sample_config.yaml
    #    python framework_cli.py list
    #    python framework_cli.py send strategist define.strategy --params '{"requirements": "Create a new reporting module"}'
    #    python framework_cli.py resources
    #    python framework_cli.py stop

    cli = FrameworkCLI()
    cli.run()