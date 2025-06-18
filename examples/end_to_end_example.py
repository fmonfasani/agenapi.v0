import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path


from core.autonomous_agent_framework import AgentFramework
from core.security_system import SecurityManager, Permission, AuthenticationMethod
from core.persistence_system import PersistenceFactory, PersistenceBackend
from interfaces.rest_api import FrameworkAPIServer
from interfaces.web_dashboard import DashboardServer
from core.monitoring_system import MonitoringOrchestrator
from core.backup_recovery_system import DisasterRecoveryOrchestrator
from systems.deployment_system import DeploymentOrchestrator, DeploymentEnvironment, DeploymentStrategy
from systems.plugin_system import PluginManager, ExternalAPIPlugin
from framework_config_utils import FrameworkBuilder
from core.models import AgentResource, ResourceType # Importar de core.models
from specialized_agents import StrategistAgent, CodeGeneratorAgent, TestGeneratorAgent # Importar solo lo necesario

class E2EFrameworkDemo:
    def __init__(self):
        self.framework = None
        self.security_manager = None
        self.persistence_manager = None
        self.api_server = None
        self.dashboard_server = None
        self.monitoring = None
        self.backup_system = None
        self.plugin_manager = None
        
        self.demo_agents = {}
        self.demo_resources = []
        self.demo_users = {}

    async def setup(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        print("🚀 Setting up Agent Framework E2E Demo...")
        print("="*50)

        print("\n1. Initializing Framework Core...")
        self.framework = AgentFramework()
        await self.framework.start()
        print("   ✅ Framework started.")

        print("\n2. Setting up Persistence Layer (SQLite)...")
        persistence_config = {
            "backend": PersistenceBackend.SQLITE.value,
            "connection_string": "e2e_framework.db",
            "auto_save_interval": 10
        }
        self.persistence_manager = PersistenceManager(self.framework, persistence_config)
        await self.persistence_manager.initialize()
        print("   ✅ Persistence initialized.")

        print("\n3. Setting up Security System...")
        security_config = {
            "jwt_secret": "supersecretkey_e2e_demo_123",
            "enable_authentication": True,
            "enable_authorization": True
        }
        self.security_manager = SecurityManager(self.framework, security_config)
        await self.security_manager.initialize()
        print("   ✅ Security system initialized.")

        print("\n4. Setting up REST API Server (Port 8000)...")
        self.api_server = FrameworkAPIServer(self.framework, self.security_manager, self.persistence_manager, host="0.0.0.0", port=8000)
        await self.api_server.start()
        print("   ✅ API Server started.")

        print("\n5. Setting up Web Dashboard (Port 8080)...")
        self.dashboard_server = DashboardServer(self.framework, host="0.0.0.0", port=8080)
        await self.dashboard_server.start()
        print("   ✅ Web Dashboard started.")
        
        print("\n6. Setting up Monitoring System...")
        monitoring_config = {
            "enable_monitoring": True,
            "health_check_interval": 5,
            "metrics_collection": True,
            "alerting": {"enabled": True, "email_alerts": False, "webhook_alerts": []}
        }
        self.monitoring = MonitoringOrchestrator(self.framework, monitoring_config)
        await self.monitoring.start_monitoring()
        print("   ✅ Monitoring system initialized.")

        print("\n7. Setting up Backup & Recovery System...")
        backup_config = {
            "backup_engine": {
                "backup_dir": "./e2e_backups",
                "retention_days": 1
            },
            "monitor_interval_seconds": 30
        }
        self.backup_system = DisasterRecoveryOrchestrator(self.framework, self.persistence_manager, backup_config)
        await self.backup_system.start_monitoring_backups()
        print("   ✅ Backup & Recovery system initialized.")

        print("\n8. Setting up Plugin Manager...")
        self.plugin_manager = PluginManager(self.framework)
        # Load example plugins (e.g., mock OpenAI, GitHub)
        # await self.plugin_manager.load_plugins_from_directory("./plugins")
        # print("   ✅ Plugin Manager initialized (no external plugins loaded for basic demo).")

        print("\n9. Deploying Demo Agents...")
        # Usar la factoría del framework para crear agentes
        strategist = await self.framework.agent_factory.create_agent_instance("agent.planning.strategist", "strategist-e2e", StrategistAgent, self.framework)
        code_gen = await self.framework.agent_factory.create_agent_instance("agent.build.code.generator", "code_generator-e2e", CodeGeneratorAgent, self.framework)
        test_gen = await self.framework.agent_factory.create_agent_instance("agent.test.generator", "test_generator-e2e", TestGeneratorAgent, self.framework)

        if strategist: self.demo_agents['strategist'] = strategist
        if code_gen: self.demo_agents['code_generator'] = code_gen
        if test_gen: self.demo_agents['test_generator'] = test_gen

        for agent in self.demo_agents.values():
            if agent:
                print(f"   ➕ Started agent: {agent.name}")
        print("   ✅ Demo Agents deployed and started.")

        print("\n10. Simulating user registration and login...")
        # Register a mock user via security manager
        user_id = "test_user_e2e"
        password = "test_password"
        registration_success = await self.security_manager.register_user(user_id, password)
        if registration_success:
            self.demo_users[user_id] = password
            print(f"   ✅ User '{user_id}' registered.")
            
            # Login and get a token
            login_result = await self.security_manager.authenticate_user(user_id, password)
            if login_result["authenticated"]:
                print(f"   ✅ User '{user_id}' logged in. Token: {login_result['token'][:20]}...")
                self.demo_users[user_id] = login_result['token'] # Store token for API calls
            else:
                print(f"   ❌ User '{user_id}' login failed.")
        else:
            print(f"   ❌ User '{user_id}' registration failed (might already exist).")
            # Try to login if registration failed (e.g., already registered from previous run)
            login_result = await self.security_manager.authenticate_user(user_id, password)
            if login_result["authenticated"]:
                print(f"   ✅ User '{user_id}' logged in (pre-existing). Token: {login_result['token'][:20]}...")
                self.demo_users[user_id] = login_result['token']
            else:
                print(f"   ❌ User '{user_id}' pre-existing login failed.")


        print("\n11. Simulating initial resource creation...")
        initial_resource = AgentResource(
            type=ResourceType.CODE,
            name="initial_project_spec",
            namespace="project.init",
            data={"requirements": "Build a scalable microservice system.", "priority": "high"},
            owner_agent_id=self.demo_agents['strategist'].id if 'strategist' in self.demo_agents else "system"
        )
        await self.framework.resource_manager.create_resource(initial_resource)
        self.demo_resources.append(initial_resource)
        print(f"   ✅ Initial resource '{initial_resource.name}' created.")

        print("\nSetup complete. All components are running.")
        print("="*50)

    async def run_core_scenario(self):
        print("\n🔄 Running Core Scenario: Agent Collaboration")
        
        strategist = self.demo_agents.get('strategist')
        code_gen = self.demo_agents.get('code_generator')
        test_gen = self.demo_agents.get('test_generator')

        if not (strategist and code_gen and test_gen):
            print("   ⚠️ Not all demo agents are available. Skipping core scenario.")
            return

        print("\n   1. Strategist defines a plan for Code Generator...")
        plan_result = await strategist.send_message(
            code_gen.id,
            "action.generate.component",
            {
                "specification": {
                    "name": "UserAuthenticationService",
                    "methods": [
                        {"name": "login", "parameters": [{"name": "username", "type": "str"}, {"name": "password", "type": "str"}]},
                        {"name": "register", "parameters": [{"name": "user_data", "type": "dict"}]}
                    ]
                },
                "language": "python"
            }
        )
        print(f"      Strategist sent command to Code Generator. Message ID: {plan_result}")
        await asyncio.sleep(2) # Give time for message processing

        print("\n   2. Code Generator generates code and stores it as a resource...")
        # Simular que code_gen procesa el mensaje y crea un recurso
        generated_code_resource = AgentResource(
            type=ResourceType.CODE,
            name="user_auth_service.py",
            namespace="service.auth",
            data={"content": "# Python code for UserAuthenticationService\nclass UserAuth:\n    pass\n"},
            owner_agent_id=code_gen.id
        )
        await self.framework.resource_manager.create_resource(generated_code_resource)
        self.demo_resources.append(generated_code_resource)
        print(f"      Code Generator created resource: {generated_code_resource.name}")
        await asyncio.sleep(1)

        print("\n   3. Test Generator creates tests for the generated code...")
        test_spec = {
            "code_resource_id": generated_code_resource.id,
            "test_framework": "pytest",
            "test_type": "unit"
        }
        test_result_msg_id = await test_gen.send_message(
            test_gen.id, # Sending to itself for demo simplification
            "action.generate.tests",
            test_spec
        )
        print(f"      Test Generator requested test generation. Message ID: {test_result_msg_id}")
        await asyncio.sleep(2)

        print("\n   4. Simulating test results as a resource...")
        test_results_resource = AgentResource(
            type=ResourceType.TEST,
            name="user_auth_tests.json",
            namespace="test.auth",
            data={"passed": 5, "failed": 0, "coverage": "85%"},
            owner_agent_id=test_gen.id
        )
        await self.framework.resource_manager.create_resource(test_results_resource)
        self.demo_resources.append(test_results_resource)
        print(f"      Test Generator created test results resource: {test_results_resource.name}")
        
        print("\nCore Scenario: Agent Collaboration Completed.")

    async def run_security_demo(self):
        print("\n🔒 Running Security Demo...")
        
        if not self.security_manager or "test_user_e2e" not in self.demo_users:
            print("   ⚠️ Security system or user not available. Skipping security demo.")
            return

        user_token = self.demo_users["test_user_e2e"]
        
        print("   1. User attempts to access agent list with token...")
        # In a real API call, this token would be in the Authorization header
        # For demo, we just validate it directly
        is_valid, claims = await self.security_manager.validate_token(user_token)
        if is_valid:
            print(f"      ✅ Token valid. Claims: {claims.get('permissions')}")
            # Simulate API call authorization check
            if Permission.READ_AGENTS.value in claims.get('permissions', []):
                print("      ✅ User has 'read_agents' permission.")
            else:
                print("      ❌ User lacks 'read_agents' permission.")
        else:
            print("      ❌ Token invalid.")

        print("   2. Simulating an audit event...")
        await self.security_manager.audit_logger.log_event(
            action="API_ACCESS",
            user_id="test_user_e2e",
            result="SUCCESS",
            details={"endpoint": "/api/agents", "method": "GET"}
        )
        print("      ✅ Audit event logged.")

        print("\nSecurity Demo Completed. Check logs for audit events.")

    async def run_monitoring_demo(self):
        print("\n📊 Running Monitoring Demo...")
        
        if not self.monitoring:
            print("   ⚠️ Monitoring system not available. Skipping monitoring demo.")
            return
            
        print("   1. Collecting system metrics (CPU, Memory)...")
        # Metrics are collected automatically by the monitoring system's background task
        await asyncio.sleep(2) # Give time for a few collection cycles
        latest_metrics = self.monitoring.metrics_collector.get_latest_metrics()
        cpu_metric = latest_metrics.get("system.cpu.usage")
        mem_metric = latest_metrics.get("system.memory.usage")
        
        if cpu_metric and mem_metric:
            print(f"      CPU Usage: {cpu_metric.value:.2f}% at {cpu_metric.timestamp.strftime('%H:%M:%S')}")
            print(f"      Memory Usage: {mem_metric.value:.2f}% at {mem_metric.timestamp.strftime('%H:%M:%S')}")
            print("      ✅ Metrics collected.")
        else:
            print("      ❌ Failed to collect system metrics.")

        print("   2. Simulating an alert (e.g., agent error)...")
        # Manually trigger an agent error status for demo
        strategist = self.demo_agents.get('strategist')
        if strategist:
            await self.framework.registry.update_agent_status(strategist.id, AgentStatus.ERROR)
            print(f"      Agent '{strategist.name}' status set to ERROR to trigger alert.")
            await asyncio.sleep(self.monitoring.health_checker.health_check_interval + 1) # Wait for health check to pick up
            
            alerts = self.monitoring.alert_manager.get_active_alerts()
            if any("AgentUnresponsive" in alert.rule_name for alert in alerts):
                print("      ✅ Alert triggered for unresponsive agent.")
            else:
                print("      ❌ Alert for unresponsive agent not triggered as expected.")
            
            # Reset agent status
            await self.framework.registry.update_agent_status(strategist.id, AgentStatus.ACTIVE)
        
        print("\nMonitoring Demo Completed.")

    async def run_backup_recovery_demo(self):
        print("\n💾 Running Backup & Recovery Demo...")
        if not self.backup_system:
            print("   ⚠️ Backup & Recovery system not available. Skipping demo.")
            return
            
        print("   1. Triggering a full backup...")
        full_backup_meta = await self.backup_system.trigger_full_backup()
        if full_backup_meta and full_backup_meta.status == BackupStatus.COMPLETED:
            print(f"      ✅ Full backup '{full_backup_meta.backup_id}' created successfully.")
        else:
            print("      ❌ Full backup failed or not completed.")
            return

        print("   2. Simulating a system crash and recovery...")
        print("      Stopping framework to simulate crash...")
        await self.framework.stop()
        # In a real scenario, persistent data might be cleared here.
        # For demo, the restore operation will effectively "reset" the state.
        
        print("      Re-initializing framework for recovery...")
        self.framework = AgentFramework()
        await self.framework.start()
        # Persistence manager needs to be re-initialized with the new framework instance for restore
        persistence_config = {
            "backend": PersistenceBackend.SQLITE.value,
            "connection_string": "e2e_framework.db",
            "auto_save_interval": 10
        }
        self.persistence_manager = PersistenceManager(self.framework, persistence_config)
        await self.persistence_manager.initialize()
        
        # Re-init DR orchestrator with new framework and persistence
        backup_config = {
            "backup_engine": { "backup_dir": "./e2e_backups", "retention_days": 1},
            "monitor_interval_seconds": 30
        }
        self.backup_system = DisasterRecoveryOrchestrator(self.framework, self.persistence_manager, backup_config)

        recovery_result = await self.backup_system.disaster_recovery_plan("system_crash")
        
        if recovery_result["success"]:
            print("      ✅ System recovered successfully from backup!")
            print(f"         Agents after recovery: {len(self.framework.registry.list_all_agents())}")
            # Verify a demo agent is back
            recovered_strategist = self.framework.registry.get_agent(self.demo_agents['strategist'].id) if 'strategist' in self.demo_agents else None
            if recovered_strategist and recovered_strategist.status == AgentStatus.ACTIVE:
                print(f"         ✅ Strategist agent ({recovered_strategist.name}) recovered and active.")
            else:
                print("         ❌ Strategist agent NOT recovered as expected.")
        else:
            print(f"      ❌ System recovery failed: {recovery_result['error']}")
            
        print("\nBackup & Recovery Demo Completed.")

    async def run_deployment_demo(self):
        print("\n📦 Running Deployment Demo (Generating configs)...")
        
        orchestrator = DeploymentOrchestrator(self.framework) # Use the main framework instance
        
        print("   1. Generating Staging Deployment (Docker Compose)...")
        staging_config = orchestrator.create_deployment_config(
            DeploymentEnvironment.STAGING,
            DeploymentStrategy.DOCKER_COMPOSE,
            api_config={"port": 8002, "host": "0.0.0.0"},
            agents_config=[
                {"namespace": "agent.planning.strategist", "name": "staging_strategist", "auto_start": True},
                {"namespace": "agent.build.code.generator", "name": "staging_code_gen", "auto_start": False}
            ]
        )
        success = await orchestrator.deploy(staging_config, "./e2e_deployments/staging")
        if success:
            print("      ✅ Staging deployment files generated in ./e2e_deployments/staging")
        else:
            print("      ❌ Staging deployment generation failed.")

        print("   2. Generating Production Deployment (Kubernetes manifests)...")
        prod_config = orchestrator.create_deployment_config(
            DeploymentEnvironment.PRODUCTION,
            DeploymentStrategy.KUBERNETES,
            jwt_secret="prod_e2e_super_secure_secret",
            db_url="postgresql://prod_user:prod_pass@prod-db-cluster/framework_prod",
            api_config={"port": 443, "host": "0.0.0.0", "enable_https": True}
        )
        success = await orchestrator.deploy(prod_config, "./e2e_deployments/production")
        if success:
            print("      ✅ Production deployment files generated in ./e2e_deployments/production")
        else:
            print("      ❌ Production deployment generation failed.")
            
        print("\nDeployment Demo Completed. Review generated files in 'e2e_deployments/' directory.")

    async def run_complete_demo(self):
        await self.setup()
        await self.run_core_scenario()
        await self.run_security_demo()
        await self.run_monitoring_demo()
        await self.run_backup_recovery_demo()
        await self.run_deployment_demo()

    async def cleanup(self):
        print("\n🧹 Running cleanup...")
        try:
            if self.dashboard_server:
                await self.dashboard_server.stop()
            if self.api_server:
                await self.api_server.stop()
            if self.backup_system:
                await self.backup_system.stop_monitoring_backups()
            if self.monitoring:
                await self.monitoring.stop_monitoring()
                
            if self.framework:
                await self.framework.stop()
                
            if self.persistence_manager:
                await self.persistence_manager.close()
                
            print("✅ Cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

async def main():
    demo = E2EFrameworkDemo()
    
    try:
        await demo.run_complete_demo()
        
        print(f"\n⏳ Services will remain active for 30 seconds...")
        print(f"🌐 Visit http://localhost:8080 to see the dashboard")
        print(f"📡 Try API endpoints at http://localhost:8000/api/")
        print(f"📚 API documentation at http://localhost:8000/")
        
        await asyncio.sleep(30)
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await demo.cleanup()
        print(f"\n👋 Thank you for trying the Agent Framework!")

if __name__ == "__main__":
    asyncio.run(main())