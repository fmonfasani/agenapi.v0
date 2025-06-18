"""
complete_example.py - Ejemplo completo del Framework de Agentes Autónomos
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any

# Imports del framework
from autonomous_agent_framework import AgentFramework, BaseAgent
from specialized_agents import ExtendedAgentFactory
from framework_config_utils import FrameworkBuilder, AgentOrchestrator, MetricsCollector

# ================================
# AGENTE CUSTOM EXAMPLE
# ================================

class ProjectManagerAgent(BaseAgent):
    """Agente gerente de proyecto - ejemplo de agente custom"""
    
    def __init__(self, name: str, framework):
        super().__init__("agent.management.project", name, framework)
        self.active_projects = {}
        
    async def initialize(self) -> bool:
        return True
        
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "create.project":
            return await self._create_project(params)
        elif action == "manage.team":
            return await self._manage_team(params)
        elif action == "deploy.solution":
            return await self._deploy_solution(params)
        return {"error": f"Unknown action: {action}"}
        
    async def _create_project(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Crear y gestionar un proyecto completo usando otros agentes"""
        project_name = params.get("name", "New Project")
        requirements = params.get("requirements", {})
        
        print(f"\n🚀 PROJECT MANAGER: Creating project '{project_name}'")
        
        # 1. Solicitar estrategia al agente estratega
        strategist = self.framework.registry.find_agents_by_namespace("agent.planning.strategist")
        if strategist:
            strategy_result = await strategist[0].execute_action("define.strategy", {
                "requirements": requirements,
                "constraints": {"timeline": "8 weeks", "budget": "medium"}
            })
            print(f"✅ Strategy defined: {strategy_result['strategy']['vision']}")
            
        # 2. Diseñar workflow
        workflow_agents = self.framework.registry.find_agents_by_namespace("agent.planning.workflow")
        if workflow_agents:
            workflow_result = await workflow_agents[0].execute_action("design.workflow", {
                "tasks": requirements.get("tasks", []),
                "name": f"{project_name} Workflow"
            })
            print(f"✅ Workflow designed with {len(workflow_result['workflow']['steps'])} steps")
            
        # 3. Crear equipo de agentes especializados
        team_agents = await self._create_specialized_team(requirements)
        print(f"✅ Team assembled: {len(team_agents)} specialized agents")
        
        # 4. Coordinar desarrollo
        development_result = await self._coordinate_development(team_agents, requirements)
        print(f"✅ Development coordinated: {len(development_result)} components created")
        
        # 5. Monitorear progreso
        await self._setup_monitoring(project_name)
        
        project_data = {
            "name": project_name,
            "status": "active",
            "team_size": len(team_agents),
            "components": development_result,
            "created_at": datetime.now().isoformat()
        }
        
        self.active_projects[project_name] = project_data
        
        return {
            "project_id": project_name,
            "status": "created",
            "team_agents": [agent.id for agent in team_agents],
            "summary": project_data
        }
        
    async def _create_specialized_team(self, requirements: Dict[str, Any]) -> list:
        """Crear equipo especializado de agentes"""
        team = []
        
        # Crear agentes según necesidades
        if requirements.get("needs_backend", True):
            backend_agent = await self.create_agent(
                "agent.build.code.generator",
                "backend_specialist",
                ExtendedAgentFactory.AGENT_CLASSES["agent.build.code.generator"]
            )
            if backend_agent:
                team.append(backend_agent)
                
        if requirements.get("needs_frontend", True):
            frontend_agent = await self.create_agent(
                "agent.build.ux.generator", 
                "frontend_specialist",
                ExtendedAgentFactory.AGENT_CLASSES["agent.build.ux.generator"]
            )
            if frontend_agent:
                team.append(frontend_agent)
                
        if requirements.get("needs_testing", True):
            test_agent = await self.create_agent(
                "agent.test.generator",
                "test_specialist", 
                ExtendedAgentFactory.AGENT_CLASSES["agent.test.generator"]
            )
            if test_agent:
                team.append(test_agent)
                
        # Siempre incluir seguridad
        security_agent = await self.create_agent(
            "agent.security.sentinel",
            "security_specialist",
            ExtendedAgentFactory.AGENT_CLASSES["agent.security.sentinel"]
        )
        if security_agent:
            team.append(security_agent)
            
        return team
        
    async def _coordinate_development(self, team_agents: list, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinar desarrollo entre agentes del equipo"""
        results = {}
        
        for agent in team_agents:
            if "code.generator" in agent.namespace:
                # Generar componentes backend
                code_result = await agent.execute_action("generate.component", {
                    "specification": {
                        "name": requirements.get("main_component", "MainService"),
                        "methods": [
                            {"name": "process_data", "parameters": [{"name": "data", "type": "dict"}]},
                            {"name": "validate_input", "parameters": [{"name": "input", "type": "str"}]}
                        ]
                    },
                    "language": "python"
                })
                results["backend_code"] = code_result
                
            elif "ux.generator" in agent.namespace:
                # Generar componentes frontend
                ui_result = await agent.execute_action("generate.ui", {
                    "design_spec": {
                        "name": "MainDashboard",
                        "elements": [
                            {"type": "h1", "content": "Dashboard"},
                            {"type": "button", "content": "Submit", "props": "onClick={handleSubmit}"}
                        ]
                    },
                    "framework": "react"
                })
                results["frontend_ui"] = ui_result
                
            elif "test.generator" in agent.namespace:
                # Generar tests
                if "backend_code" in results:
                    test_result = await agent.execute_action("generate.tests", {
                        "code": results["backend_code"]["code"],
                        "test_framework": "pytest"
                    })
                    results["tests"] = test_result
                    
            elif "security.sentinel" in agent.namespace:
                # Escanear seguridad
                security_result = await agent.execute_action("scan.vulnerabilities", {
                    "target": "generated_application",
                    "scan_type": "comprehensive"
                })
                results["security_scan"] = security_result
                
        return results
        
    async def _setup_monitoring(self, project_name: str):
        """Configurar monitoreo del proyecto"""
        monitor_agents = self.framework.registry.find_agents_by_namespace("agent.monitor.progress")
        if monitor_agents:
            await monitor_agents[0].execute_action("track.progress", {
                "task_id": project_name,
                "status": "in_development",
                "progress": 25
            })
            
    async def _manage_team(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Gestionar equipo de agentes"""
        team_action = params.get("action", "status")
        
        if team_action == "status":
            agents = self.framework.registry.list_all_agents()
            team_status = []
            for agent in agents:
                if agent.id != self.id:  # Excluir self
                    team_status.append({
                        "id": agent.id,
                        "name": agent.name,
                        "namespace": agent.namespace,
                        "status": agent.status.value,
                        "last_heartbeat": agent.last_heartbeat.isoformat()
                    })
                    
            return {"team_status": team_status}
            
        elif team_action == "optimize":
            # Redistribuir tareas basado en carga
            return {"message": "Team optimization completed"}
            
        return {"error": "Unknown team action"}
        
    async def _deploy_solution(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Desplegar solución completa"""
        project_name = params.get("project_name", "")
        environment = params.get("environment", "staging")
        
        if project_name not in self.active_projects:
            return {"error": f"Project {project_name} not found"}
            
        # Simulación de despliegue
        deployment_steps = [
            "Building application",
            "Running tests", 
            "Security validation",
            "Infrastructure setup",
            "Application deployment",
            "Health checks"
        ]
        
        print(f"\n🚀 DEPLOYING {project_name} to {environment}:")
        for i, step in enumerate(deployment_steps):
            print(f"  {i+1}. {step}...")
            await asyncio.sleep(0.5)  # Simular tiempo de procesamiento
            
        # Actualizar estado del proyecto
        self.active_projects[project_name]["status"] = "deployed"
        self.active_projects[project_name]["environment"] = environment
        self.active_projects[project_name]["deployed_at"] = datetime.now().isoformat()
        
        return {
            "project_name": project_name,
            "deployment_status": "success",
            "environment": environment,
            "url": f"https://{project_name.lower()}.{environment}.company.com"
        }

# ================================
# DEMO COMPLETO
# ================================

async def complete_demo():
    """Demo completo del framework"""
    
    print("="*80)
    print("🤖 AUTONOMOUS AGENT FRAMEWORK - COMPLETE DEMO")
    print("="*80)
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 1. Crear y configurar framework
    print("\n📋 Step 1: Creating framework...")
    framework = AgentFramework()
    await framework.start()
    
    # 2. Crear ecosistema básico de agentes
    print("\n📋 Step 2: Creating agent ecosystem...")
    agents = await ExtendedAgentFactory.create_full_ecosystem(framework)
    
    # 3. Crear agente gerente de proyecto custom
    print("\n📋 Step 3: Creating custom Project Manager agent...")
    pm_agent = ProjectManagerAgent("project_manager", framework)
    await pm_agent.start()
    agents["project_manager"] = pm_agent
    
    # 4. Configurar métricas
    metrics = MetricsCollector(framework)
    orchestrator = AgentOrchestrator(framework, agents)
    
    print(f"\n✅ Framework initialized with {len(agents)} agents:")
    for name, agent in agents.items():
        print(f"   • {agent.namespace} - {name} ({agent.status.value})")
        
    # 5. DEMO: Crear proyecto completo
    print("\n" + "="*60)
    print("🎯 DEMO: Creating complete project using agent collaboration")
    print("="*60)
    
    project_spec = {
        "name": "E-Commerce Platform",
        "requirements": {
            "goal": "Build scalable e-commerce platform",
            "tasks": [
                {"name": "User Management", "type": "backend"},
                {"name": "Product Catalog", "type": "backend"},
                {"name": "Shopping Cart", "type": "frontend"},
                {"name": "Payment Processing", "type": "integration"},
                {"name": "Order Management", "type": "backend"}
            ],
            "main_component": "ECommerceAPI",
            "needs_backend": True,
            "needs_frontend": True,
            "needs_testing": True
        }
    }
    
    # Crear proyecto usando Project Manager
    project_result = await pm_agent.execute_action("create.project", project_spec)
    
    print(f"\n🎉 Project created successfully!")
    print(f"   Project ID: {project_result['project_id']}")
    print(f"   Team size: {len(project_result['team_agents'])} agents")
    
    # 6. Monitorear estado del equipo
    print("\n📋 Step 6: Monitoring team status...")
    team_status = await pm_agent.execute_action("manage.team", {"action": "status"})
    
    print("\n👥 Team Status:")
    for agent_info in team_status["team_status"][:5]:  # Mostrar solo 5 para brevedad
        print(f"   • {agent_info['name']} ({agent_info['namespace']}) - {agent_info['status']}")
    
    # 7. Simular comunicación entre agentes
    print("\n📋 Step 7: Agent-to-agent communication demo...")
    
    # El strategist solicita un código específico al generator
    strategist = agents["strategist"]
    generator = agents["generator"]
    
    # Enviar mensaje directo
    message_id = await strategist.send_message(
        generator.id,
        "action.generate.component",
        {
            "specification": {
                "name": "PaymentProcessor",
                "methods": [
                    {"name": "process_payment", "parameters": [{"name": "amount", "type": "float"}]},
                    {"name": "validate_card", "parameters": [{"name": "card_number", "type": "str"}]}
                ]
            },
            "language": "python"
        }
    )
    
    print(f"   📨 Message sent from strategist to generator: {message_id}")
    await asyncio.sleep(1)  # Dar tiempo para procesamiento
    
    # 8. Desplegar proyecto
    print("\n📋 Step 8: Deploying project...")
    deployment_result = await pm_agent.execute_action("deploy.solution", {
        "project_name": "E-Commerce Platform",
        "environment": "production"
    })
    
    print(f"\n🚀 Deployment completed!")
    print(f"   Status: {deployment_result['deployment_status']}")
    print(f"   URL: {deployment_result['url']}")
    
    # 9. Mostrar recursos creados
    print("\n📋 Step 9: Checking created resources...")
    all_resources = []
    for agent in agents.values():
        agent_resources = framework.resource_manager.find_resources_by_owner(agent.id)
        all_resources.extend(agent_resources)
        
    print(f"\n📦 Resources created: {len(all_resources)}")
    resource_summary = {}
    for resource in all_resources:
        resource_type = resource.type.value
        resource_summary[resource_type] = resource_summary.get(resource_type, 0) + 1
        
    for res_type, count in resource_summary.items():
        print(f"   • {res_type}: {count}")
    
    # 10. Métricas finales
    print("\n📋 Step 10: Final metrics...")
    final_metrics = metrics.get_metrics()
    print(f"\n📊 Framework Metrics:")
    print(f"   • Active agents: {final_metrics['active_agents']}")
    print(f"   • Messages sent: {final_metrics['messages_sent']}")
    print(f"   • Resources created: {len(all_resources)}")
    print(f"   • Errors: {final_metrics['errors']}")
    
    # 11. Demo de auto-gestión: Crear agente dinámicamente
    print("\n📋 Step 11: Dynamic agent creation demo...")
    
    # El PM crea un agente especializado para analytics
    new_agent = await pm_agent.create_agent(
        "agent.analytics.reporter",
        "analytics_agent",
        ProjectManagerAgent  # Reutilizamos la clase como ejemplo
    )
    
    if new_agent:
        print(f"   ✅ Created dynamic agent: {new_agent.id}")
        
        # El nuevo agente puede reportar métricas
        await new_agent.send_message(
            pm_agent.id,
            "action.report.metrics",
            {"metrics": final_metrics}
        )
        print(f"   📊 Analytics agent reported metrics to PM")
    
    # 12. Cleanup demo
    print("\n📋 Step 12: Graceful shutdown...")
    
    # Notificar a todos los agentes sobre el shutdown
    await pm_agent.broadcast_message("system.shutdown", {"reason": "demo_complete"})
    
    print("\n" + "="*60)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Summary:")
    print(f"• Framework managed {len(agents)} agents")
    print(f"• Created complete e-commerce project")
    print(f"• Generated {len(all_resources)} resources")
    print(f"• Demonstrated agent autonomy and collaboration")
    print(f"• Showed dynamic agent creation and management")
    print("="*60)
    
    # Dar tiempo para que los mensajes se procesen
    await asyncio.sleep(2)
    
    # Shutdown framework
    await framework.stop()
    print("\n👋 Framework shutdown complete")

# ================================
# EXAMPLES DE USO ESPECÍFICO
# ================================

async def quick_start_example():
    """Ejemplo de inicio rápido"""
    print("🚀 Quick Start Example")
    
    # Framework mínimo
    framework = AgentFramework()
    await framework.start()
    
    # Crear solo algunos agentes
    strategist = ExtendedAgentFactory.create_agent("agent.planning.strategist", "strategist", framework)
    generator = ExtendedAgentFactory.create_agent("agent.build.code.generator", "generator", framework)
    
    await strategist.start()
    await generator.start()
    
    # Solicitar código
    result = await generator.execute_action("generate.component", {
        "specification": {"name": "SimpleAPI", "methods": [{"name": "get_data"}]},
        "language": "python"
    })
    
    print("Generated code preview:")
    print(result["code"][:200] + "...")
    
    await framework.stop()

async def enterprise_example():
    """Ejemplo para uso empresarial"""
    print("🏢 Enterprise Example")
    
    # Usar configuración
    builder = FrameworkBuilder()
    framework, agents = await builder.build_framework()
    
    # Crear orquestador
    orchestrator = AgentOrchestrator(framework, agents)
    
    # Pipeline empresarial
    project = {
        "goal": "Microservices architecture",
        "tasks": [
            {"name": "API Gateway", "type": "infrastructure"},
            {"name": "User Service", "type": "microservice"},
            {"name": "Payment Service", "type": "microservice"},
            {"name": "Notification Service", "type": "microservice"}
        ],
        "code_spec": {
            "name": "MicroserviceBase",
            "methods": [
                {"name": "health_check", "parameters": []},
                {"name": "process_request", "parameters": [{"name": "request", "type": "dict"}]}
            ]
        }
    }
    
    results = await orchestrator.create_development_pipeline(project)
    print(f"Enterprise pipeline completed: {len(results)} steps")
    
    await framework.stop()

# ================================
# MAIN EXECUTION
# ================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            asyncio.run(quick_start_example())
        elif sys.argv[1] == "enterprise": 
            asyncio.run(enterprise_example())
        else:
            print("Usage: python complete_example.py [quick|enterprise]")
    else:
        asyncio.run(complete_demo())
        