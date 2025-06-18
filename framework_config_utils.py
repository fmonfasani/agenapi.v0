import json
import yaml
import os
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from pathlib import Path
import asyncio

from core.autonomous_agent_framework import AgentFramework, BaseAgent
from specialized_agents import StrategistAgent, WorkflowDesignerAgent, CodeGeneratorAgent, TestGeneratorAgent, BuildAgent # Importar agentes específicos para la factoría
from core.persistence_system import PersistenceManager, PersistenceConfig, PersistenceBackend
from core.security_system import SecurityManager
from core.monitoring_system import MonitoringOrchestrator, MonitoringConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FrameworkConfigUtils")

@dataclass
class AgentConfig:
    namespace: str
    name: str
    enabled: bool = True
    auto_start: bool = True
    max_concurrent_tasks: int = 10
    heartbeat_interval: int = 30
    restart_on_failure: bool = True
    custom_settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FrameworkConfig:
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

@dataclass
class CoreServicesConfig:
    framework: FrameworkConfig = field(default_factory=FrameworkConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    security: Dict[str, Any] = field(default_factory=lambda: {
        "jwt_secret": "your_super_secret_key",
        "enable_authentication": True,
        "enable_authorization": True,
        "auth_token_expiry_minutes": 60
    })
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    agents: List[AgentConfig] = field(default_factory=list)

class FrameworkBuilder:
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self._framework_instance: Optional[AgentFramework] = None
        self._persistence_manager: Optional[PersistenceManager] = None
        self._security_manager: Optional[SecurityManager] = None
        self._monitoring_orchestrator: Optional[MonitoringOrchestrator] = None
        self.config: Optional[CoreServicesConfig] = None
        self.agent_class_map: Dict[str, Type[BaseAgent]] = {
            "agent.planning.strategist": StrategistAgent,
            "agent.planning.workflow_designer": WorkflowDesignerAgent,
            "agent.build.code.generator": CodeGeneratorAgent,
            "agent.test.generator": TestGeneratorAgent,
            "agent.build.builder": BuildAgent,
        }

    def create_sample_config(self, output_path: Union[str, Path] = "framework_config_sample.yaml"):
        sample_config = CoreServicesConfig(
            framework=FrameworkConfig(
                name="MyAgentSystem",
                log_level="INFO",
                message_timeout=45
            ),
            persistence=PersistenceConfig(
                backend=PersistenceBackend.SQLITE,
                connection_string="data/my_agent_system.db",
                auto_save_interval=120
            ),
            security={
                "jwt_secret": "my_secure_secret_for_jwt_tokens",
                "enable_authentication": True,
                "enable_authorization": True
            },
            monitoring=MonitoringConfig(
                enable_monitoring=True,
                health_check_interval=30,
                alerting={"enabled": True, "email_alerts": False}
            ),
            agents=[
                AgentConfig(namespace="agent.planning.strategist", name="MainStrategist", auto_start=True),
                AgentConfig(namespace="agent.build.code.generator", name="PythonCodeGen", auto_start=True),
                AgentConfig(namespace="agent.test.generator", name="UnitTestGen", auto_start=False),
                AgentConfig(namespace="agent.internal.logger", name="ActivityLogger", enabled=False)
            ]
        )
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

        # Custom JSON encoder for Enum
        class EnumEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Enum):
                    return obj.value
                return json.JSONEncoder.default(self, obj)

        # Convert to dict and then dump as YAML
        config_dict = json.loads(json.dumps(asdict(sample_config), cls=EnumEncoder))
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, indent=4)
        logger.info(f"Sample configuration created at {output_path}")
        return sample_config

    async def load_config(self) -> Optional[CoreServicesConfig]:
        if not self.config_path or not self.config_path.exists():
            logger.error(f"Configuration file not found at {self.config_path}")
            return None
        
        with open(self.config_path, 'r') as f:
            raw_config = yaml.safe_load(f)

        # Manually parse with Enum conversion
        framework_cfg = FrameworkConfig(**raw_config.get("framework", {}))
        
        persistence_raw = raw_config.get("persistence", {})
        persistence_cfg = PersistenceConfig(
            backend=PersistenceBackend(persistence_raw.get("backend", "sqlite")),
            connection_string=persistence_raw.get("connection_string", "framework.db"),
            auto_save_interval=persistence_raw.get("auto_save_interval", 60),
            max_message_history=persistence_raw.get("max_message_history", 1000),
            enable_compression=persistence_raw.get("enable_compression", False),
            backup_enabled=persistence_raw.get("backup_enabled", True),
            backup_interval=persistence_raw.get("backup_interval", 3600)
        )

        security_cfg = raw_config.get("security", {})
        
        monitoring_raw = raw_config.get("monitoring", {})
        monitoring_cfg = MonitoringConfig(
            enable_monitoring=monitoring_raw.get("enable_monitoring", True),
            health_check_interval=monitoring_raw.get("health_check_interval", 60),
            metrics_collection=monitoring_raw.get("metrics_collection", True),
            log_level=monitoring_raw.get("log_level", "INFO"),
            alerting=monitoring_raw.get("alerting", {}),
            resource_monitoring=monitoring_raw.get("resource_monitoring", True)
        )

        agents_cfg = [AgentConfig(**a) for a in raw_config.get("agents", [])]

        self.config = CoreServicesConfig(
            framework=framework_cfg,
            persistence=persistence_cfg,
            security=security_cfg,
            monitoring=monitoring_cfg,
            agents=agents_cfg
        )
        logger.info(f"Configuration loaded from {self.config_path}")
        return self.config

    async def build_framework(self) -> Tuple[AgentFramework, Dict[str, BaseAgent]]:
        if not self.config:
            await self.load_config()
            if not self.config:
                raise ValueError("Framework configuration not loaded.")

        logger.info("Building AgentFramework components...")

        # 1. Initialize Framework
        self._framework_instance = AgentFramework(self.config.framework.__dict__)
        await self._framework_instance.start()
        logger.info("AgentFramework core started.")

        # 2. Initialize Persistence Manager
        self._persistence_manager = PersistenceManager(self._framework_instance, self.config.persistence.__dict__)
        await self._persistence_manager.initialize()
        logger.info("PersistenceManager initialized.")

        # 3. Initialize Security Manager
        self._security_manager = SecurityManager(self._framework_instance, self.config.security)
        await self._security_manager.initialize()
        logger.info("SecurityManager initialized.")

        # 4. Initialize Monitoring Orchestrator
        self._monitoring_orchestrator = MonitoringOrchestrator(self._framework_instance, self.config.monitoring.__dict__)
        await self._monitoring_orchestrator.start_monitoring()
        logger.info("MonitoringOrchestrator initialized.")

        # 5. Deploy Agents
        deployed_agents: Dict[str, BaseAgent] = {}
        for agent_cfg in self.config.agents:
            if agent_cfg.enabled:
                agent_class = self.agent_class_map.get(agent_cfg.namespace)
                if agent_class:
                    agent_instance = await self._framework_instance.agent_factory.create_agent_instance(
                        agent_cfg.namespace, agent_cfg.name, agent_class, self._framework_instance, agent_cfg.custom_settings
                    )
                    if agent_instance:
                        deployed_agents[agent_instance.id] = agent_instance
                        if agent_cfg.auto_start:
                            await agent_instance.start()
                            logger.info(f"Agent {agent_cfg.name} auto-started.")
                        else:
                            logger.info(f"Agent {agent_cfg.name} created but not auto-started.")
                    else:
                        logger.error(f"Failed to create agent instance for {agent_cfg.name}")
                else:
                    logger.warning(f"No agent class mapping found for namespace: {agent_cfg.namespace}. Skipping agent {agent_cfg.name}.")
        logger.info(f"Deployed {len(deployed_agents)} agents.")

        logger.info("Framework building process completed.")
        return self._framework_instance, deployed_agents

    async def shutdown_framework(self):
        logger.info("Shutting down framework components...")
        if self._monitoring_orchestrator:
            await self._monitoring_orchestrator.stop_monitoring()
        if self._persistence_manager:
            await self._persistence_manager.close()
        if self._framework_instance:
            await self._framework_instance.stop()
        logger.info("Framework shutdown complete.")

class AgentOrchestrator:
    def __init__(self, framework: AgentFramework, agents: Dict[str, BaseAgent]):
        self.framework = framework
        self.agents = agents
        self.logger = logging.getLogger("AgentOrchestrator")

    async def create_development_pipeline(self, project_spec: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"Orchestrating development pipeline for: {project_spec.get('goal', 'unknown project')}")
        results = {}

        strategist = next((a for a in self.agents.values() if a.namespace == "agent.planning.strategist"), None)
        workflow_designer = next((a for a in self.agents.values() if a.namespace == "agent.planning.workflow_designer"), None)
        code_generator = next((a for a in self.agents.values() if a.namespace == "agent.build.code.generator"), None)
        test_generator = next((a for a in self.agents.values() if a.namespace == "agent.test.generator"), None)
        
        if not (strategist and code_generator and test_generator):
            self.logger.error("Missing one or more required agents for development pipeline.")
            results["error"] = "Missing required agents"
            return results

        # Step 1: Strategy Definition
        self.logger.info("Step 1: Strategist defines overall plan.")
        strategy_response = await strategist.execute_action(
            "define.strategy", 
            {"requirements": project_spec.get("goal"), "constraints": {}}
        )
        results["strategy_definition"] = strategy_response

        # Step 2: Code Generation
        self.logger.info("Step 2: Code Generator creates core components.")
        code_spec = project_spec.get("code_spec", {})
        code_response = await code_generator.execute_action(
            "generate.component", 
            {"specification": code_spec, "language": "python"}
        )
        results["code_generation"] = code_response
        
        # Assume code generation creates a resource. Find it.
        generated_code_resource = None
        if code_response and "resource_id" in code_response:
             generated_code_resource = await self.framework.resource_manager.get_resource(code_response["resource_id"])
        
        if not generated_code_resource and "code" in code_response: # Fallback for demo if no resource_id
            # Create a mock resource if the agent didn't return a resource ID
            generated_code_resource = AgentResource(
                type=self.framework.resource_manager.ResourceType.CODE,
                name="generated_code_fallback",
                namespace="orchestrator.temp",
                data={"content": code_response["code"]},
                owner_agent_id="orchestrator"
            )
            await self.framework.resource_manager.create_resource(generated_code_resource)

        # Step 3: Test Generation
        if generated_code_resource:
            self.logger.info("Step 3: Test Generator creates tests for the generated code.")
            test_response = await test_generator.execute_action(
                "generate.tests", 
                {"code": generated_code_resource.data.get("content"), "test_framework": "pytest", "code_resource_id": generated_code_resource.id}
            )
            results["test_generation"] = test_response
        else:
            self.logger.warning("No generated code resource found, skipping test generation.")
            results["test_generation"] = {"error": "No code to test"}

        self.logger.info("Development pipeline completed.")
        return results

class MetricsCollector:
    def __init__(self, framework: AgentFramework):
        self.framework = framework
        self.metrics: Dict[str, Any] = {} 
        self.logger = logging.getLogger("MetricsCollector")

    def collect_system_metrics(self):
        import psutil
        self._add_metric("system.cpu.usage", psutil.cpu_percent(interval=None), "gauge", unit="%")
        self._add_metric("system.memory.usage", psutil.virtual_memory().percent, "gauge", unit="%")
        self._add_metric("system.disk.usage", psutil.disk_usage('/').percent, "gauge", unit="%")
        self._add_metric("system.network.bytes_sent", psutil.net_io_counters().bytes_sent, "counter", unit="bytes")
        self._add_metric("system.network.bytes_recv", psutil.net_io_counters().bytes_recv, "counter", unit="bytes")
        
    def collect_framework_metrics(self):
        total_agents = len(self.framework.registry.list_all_agents())
        active_agents = sum(1 for agent in self.framework.registry.list_all_agents() if agent.status == self.framework.AgentStatus.ACTIVE)
        messages_in_bus = self.framework.message_bus.message_queue.qsize()

        self._add_metric("framework.agents.total", total_agents, "gauge")
        self._add_metric("framework.agents.active", active_agents, "gauge")
        self._add_metric("framework.message_bus.queue_size", messages_in_bus, "gauge")
        self._add_metric("framework.resources.total", len(self.framework.resource_manager.list_all_resources()), "gauge")

    def _add_metric(self, name: str, value: Union[float, int], metric_type: str, tags: Optional[Dict[str, str]] = None, unit: str = ""):
        self.metrics[name] = {
            "name": name,
            "type": metric_type,
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "tags": tags or {},
            "unit": unit
        }

    def get_metric(self, name: str) -> Optional[Dict[str, Any]]:
        return self.metrics.get(name)

    def export_metrics(self) -> Dict[str, Any]:
        return self.metrics

async def config_utils_example():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🚀 Starting Framework Configuration and Utils Demo")
    print("="*50)

    config_path = "config/demo_config.yaml"
    builder = FrameworkBuilder(config_path=config_path)

    print("\n1. Creating sample config file...")
    builder.create_sample_config(config_path)
    print(f"   ✅ Sample config saved to {config_path}")

    print("\n2. Building framework from config...")
    framework, agents = await builder.build_framework()
    print(f"   ✅ Framework and {len(agents)} agents built and started from config.")
    print(f"      Framework name: {framework.config.get('name')}")
    print(f"      Persistence backend: {framework.persistence_manager.config.get('backend')}")
    print(f"      Security enabled: {framework.security_manager.config.get('enable_authentication')}")

    orchestrator = AgentOrchestrator(framework, agents)

    print("\n3. Running a development pipeline via orchestrator...")
    project_spec = {
        "goal": "Create user management system",
        "tasks": [
            {"name": "Design API", "type": "design"},
            {"name": "Implement backend", "type": "development"},
            {"name": "Create tests", "type": "testing"}
        ],
        "code_spec": {
            "name": "UserManager",
            "methods": [
                {"name": "create_user", "parameters": [{"name": "user_data", "type": "dict"}]},
                {"name": "update_user", "parameters": [{"name": "user_id", "type": "str"}]},
                {"name": "delete_user", "parameters": [{"name": "user_id", "type": "str"}]}
            ]
        }
    }
    
    results = await orchestrator.create_development_pipeline(project_spec)
    
    print("\n   Pipeline completed:")
    for step_id, result in results.items():
        print(f"     - {step_id}: {'✓' if 'error' not in result else '✗'} - {result.get('message', str(result))[:50]}...")
        
    print("\n4. Collecting and displaying metrics...")
    metrics = MetricsCollector(framework)
    metrics.collect_system_metrics()
    metrics.collect_framework_metrics()
    
    exported_metrics = metrics.export_metrics()
    print("   📊 Metrics collected:")
    for name, metric_data in list(exported_metrics.items())[:5]: # Show top 5
        print(f"     - {name}: {metric_data['value']:.2f} {metric_data['unit']} (Type: {metric_data['type']})")

    finally:
        print("\n5. Shutting down framework components...")
        await builder.shutdown_framework()
        print("   ✅ Demo completed and framework shut down.")

if __name__ == "__main__":
    asyncio.run(config_utils_example())