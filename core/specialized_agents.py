"""
specialized_agents.py - Implementaciones específicas de agentes por namespace
"""

# Importaciones actualizadas
from core.autonomous_agent_framework import BaseAgent, AgentFramework # <-- MANTENIDO Framework, BaseAgent ahora de aquí
from core.models import AgentCapability, AgentResource, ResourceType # <-- CAMBIO AQUI

from typing import Dict, Any, List, Optional, Type
import asyncio
import json
import logging
import uuid

# ================================\
# PLANNING AGENTS
# ================================\

class StrategistAgent(BaseAgent):
    """agent.planning.strategist - Agente estratega"""
    
    def __init__(self, name: str, framework: AgentFramework):
        super().__init__("agent.planning.strategist", name, framework)
        
    async def initialize(self) -> bool:
        self.capabilities = [
            AgentCapability(
                name="define.strategy",
                namespace="agent.planning.strategist.define",
                description="Define strategic plans and roadmaps",
                input_schema={"type": "object", "properties": {"requirements": {"type": "object"}, "constraints": {"type": "object"}}},
                output_schema={"type": "object", "properties": {"strategy": {"type": "object"}, "roadmap": {"type": "array"}}},
                handler=self._define_strategy
            ),
            AgentCapability(
                name="propose.agents",
                namespace="agent.planning.strategist.propose_agents",
                description="Propose required agents for a given task",
                input_schema={"type": "object", "properties": {"task_description": {"type": "string"}}},
                output_schema={"type": "object", "properties": {"recommended_agents": {"type": "array"}}},
                handler=self._propose_agents
            )
        ]
        self.logger.info(f"StrategistAgent {self.id} initialized with {len(self.capabilities)} capabilities.")
        return True
        
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Método principal para ejecutar acciones solicitadas por otros agentes."""
        for capability in self.capabilities:
            if capability.name == action:
                return await capability.handler(params)
        return {"status": "error", "message": f"Action '{action}' not supported by StrategistAgent."}
        
    async def _define_strategy(self, params: Dict[str, Any]) -> Dict[str, Any]:
        requirements = params.get("requirements", {})
        constraints = params.get("constraints", {})
        
        self.logger.info(f"Defining strategy for requirements: {requirements}")
        
        # Lógica compleja de definición de estrategia (simulada)
        strategy = {
            "overall_goal": requirements.get("goal", "Develop a new product"),
            "key_phases": ["Discovery", "Design", "Development", "Testing", "Deployment"],
            "constraints_applied": constraints
        }
        roadmap = [
            {"phase": "Discovery", "tasks": ["Market research", "User stories"]},
            {"phase": "Design", "tasks": ["Architecture", "UI/UX"]},
            # ...
        ]
        
        # Publicar un evento para que otros agentes sepan que una estrategia ha sido definida
        await self.publish_event("strategy.defined", {"strategy_id": str(uuid.uuid4()), "details": strategy})

        return {"status": "success", "strategy": strategy, "roadmap": roadmap}

    async def _propose_agents(self, params: Dict[str, Any]) -> Dict[str, Any]:
        task_description = params.get("task_description", "")
        self.logger.info(f"Proposing agents for task: {task_description}")
        
        # Lógica simple para proponer agentes basada en palabras clave
        recommended_agents = []
        if "design" in task_description.lower() or "architecture" in task_description.lower():
            recommended_agents.append({"name": "workflow_designer", "namespace": "agent.planning.workflow_designer"})
        if "code" in task_description.lower() or "implement" in task_description.lower():
            recommended_agents.append({"name": "code_generator", "namespace": "agent.build.code.generator"})
        if "test" in task_description.lower() or "qa" in task_description.lower():
            recommended_agents.append({"name": "test_generator", "namespace": "agent.test.generator"})
        if "deploy" in task_description.lower():
            recommended_agents.append({"name": "build_agent", "namespace": "agent.build.builder"})
            
        return {"status": "success", "recommended_agents": recommended_agents}


class WorkflowDesignerAgent(BaseAgent):
    """agent.planning.workflow_designer - Agente diseñador de flujos de trabajo"""
    
    def __init__(self, name: str, framework: AgentFramework):
        super().__init__("agent.planning.workflow_designer", name, framework)
        
    async def initialize(self) -> bool:
        self.capabilities = [
            AgentCapability(
                name="create.workflow",
                namespace="agent.planning.workflow_designer.create",
                description="Design and define multi-agent workflows",
                input_schema={"type": "object", "properties": {"tasks": {"type": "array"}, "priority": {"type": "string"}}},
                output_schema={"type": "object", "properties": {"workflow": {"type": "object"}}},
                handler=self._create_workflow
            )
        ]
        self.logger.info(f"WorkflowDesignerAgent {self.id} initialized with {len(self.capabilities)} capabilities.")
        return True

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        for capability in self.capabilities:
            if capability.name == action:
                return await capability.handler(params)
        return {"status": "error", "message": f"Action '{action}' not supported by WorkflowDesignerAgent."}

    async def _create_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        tasks = params.get("tasks", [])
        priority = params.get("priority", "medium")
        
        self.logger.info(f"Designing workflow for tasks: {tasks} with priority: {priority}")
        
        # Simular la lógica de diseño de flujo de trabajo
        workflow_steps = []
        for i, task in enumerate(tasks):
            workflow_steps.append({
                "step_id": f"step_{i+1}",
                "task": task,
                "status": "pending",
                "assigned_agent": "auto" # En un sistema real, se asignaría un agente específico
            })
            
        workflow = {
            "id": str(uuid.uuid4()),
            "name": f"Workflow for {tasks[0] if tasks else 'unspecified task'}",
            "steps": workflow_steps,
            "created_at": datetime.now().isoformat(),
            "status": "draft"
        }
        
        # Almacenar el workflow como un recurso
        workflow_resource = AgentResource(
            type=ResourceType.WORKFLOW,
            name=f"workflow-{workflow['id']}",
            namespace="workflow.generated",
            data=workflow,
            owner_agent_id=self.id
        )
        await self.framework.resource_manager.create_resource(workflow_resource)

        await self.publish_event("workflow.created", {"workflow_id": workflow["id"], "details": workflow})

        return {"status": "success", "workflow": workflow, "resource_id": workflow_resource.id}

# ================================\
# DEVELOPMENT/BUILD AGENTS
# ================================\

class CodeGeneratorAgent(BaseAgent):
    """agent.build.code.generator - Agente generador de código"""

    def __init__(self, name: str, framework: AgentFramework):
        super().__init__("agent.build.code.generator", name, framework)

    async def initialize(self) -> bool:
        self.capabilities = [
            AgentCapability(
                name="generate.component",
                namespace="agent.build.code.generator.generate_component",
                description="Generates code for a specified component based on a specification.",
                input_schema={"type": "object", "properties": {"specification": {"type": "object"}, "language": {"type": "string"}}},
                output_schema={"type": "object", "properties": {"code": {"type": "string"}, "file_name": {"type": "string"}}},
                handler=self._generate_component_code
            ),
            AgentCapability(
                name="refactor.code",
                namespace="agent.build.code.generator.refactor",
                description="Refactors existing code based on best practices.",
                input_schema={"type": "object", "properties": {"code": {"type": "string"}, "refactoring_goal": {"type": "string"}}},
                output_schema={"type": "object", "properties": {"refactored_code": {"type": "string"}}},
                handler=self._refactor_code
            )
        ]
        self.logger.info(f"CodeGeneratorAgent {self.id} initialized with {len(self.capabilities)} capabilities.")
        return True
    
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        for capability in self.capabilities:
            if capability.name == action:
                return await capability.handler(params)
        return {"status": "error", "message": f"Action '{action}' not supported by CodeGeneratorAgent."}

    async def _generate_component_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        specification = params.get("specification", {})
        language = params.get("language", "python")
        
        self.logger.info(f"Generating {language} code for component: {specification.get('name')}")
        
        # Simular generación de código (aquí se integraría con un LLM o generador de plantillas)
        component_name = specification.get("name", "untitled_component")
        methods = specification.get("methods", [])
        
        code_lines = [f"class {component_name}:"]
        for method in methods:
            method_name = method.get("name", "unnamed_method")
            method_params = ", ".join([f"{p['name']}: {p['type']}" for p in method.get("parameters", [])])
            code_lines.append(f"    def {method_name}(self, {method_params}):")
            code_lines.append(f"        # TODO: Implement {method_name} logic")
            code_lines.append(f"        self.framework.logger.info(f\"Executing {method_name} in {self.name}\")")
            code_lines.append(f"        return {{'status': 'success', 'message': '{method_name} executed'}}")
            code_lines.append("")
        
        generated_code = "\n".join(code_lines)
        file_name = f"{component_name.lower()}.{language.replace('python', 'py')}"

        # Almacenar el código generado como un recurso
        code_resource = AgentResource(
            type=ResourceType.CODE,
            name=file_name,
            namespace=f"code.generated.{component_name.lower()}",
            data=generated_code,
            owner_agent_id=self.id
        )
        await self.framework.resource_manager.create_resource(code_resource)

        await self.publish_event("code.generated", {"component": component_name, "file": file_name, "resource_id": code_resource.id})

        return {"status": "success", "code": generated_code, "file_name": file_name, "resource_id": code_resource.id}

    async def _refactor_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        code = params.get("code")
        refactoring_goal = params.get("refactoring_goal", "improve readability")

        self.logger.info(f"Refactoring code with goal: {refactoring_goal}")

        # Simular refactorización de código
        refactored_code = code + "\n# Code refactored to " + refactoring_goal + "\n"
        
        # Almacenar el código refactorizado como un nuevo recurso o actualizar el existente
        refactored_resource = AgentResource(
            type=ResourceType.CODE,
            name="refactored_code_" + str(uuid.uuid4())[:8] + ".py",
            namespace="code.refactored",
            data=refactored_code,
            owner_agent_id=self.id
        )
        await self.framework.resource_manager.create_resource(refactored_resource)

        await self.publish_event("code.refactored", {"original_code_checksum": "xyz", "refactoring_goal": refactoring_goal, "resource_id": refactored_resource.id})
        
        return {"status": "success", "refactored_code": refactored_code, "resource_id": refactored_resource.id}


class BuildAgent(BaseAgent):
    """agent.build.builder - Agente de construcción y compilación"""

    def __init__(self, name: str, framework: AgentFramework):
        super().__init__("agent.build.builder", name, framework)

    async def initialize(self) -> bool:
        self.capabilities = [
            AgentCapability(
                name="build.project",
                namespace="agent.build.builder.build_project",
                description="Builds a project from source code.",
                input_schema={"type": "object", "properties": {"source_code_resource_id": {"type": "string"}, "build_config": {"type": "object"}}},
                output_schema={"type": "object", "properties": {"build_status": {"type": "string"}, "artifacts_resource_id": {"type": "string"}}},
                handler=self._build_project
            ),
            AgentCapability(
                name="deploy.artifact",
                namespace="agent.build.builder.deploy_artifact",
                description="Deploys a generated artifact to a target environment.",
                input_schema={"type": "object", "properties": {"artifact_resource_id": {"type": "string"}, "target_environment": {"type": "string"}}},
                output_schema={"type": "object", "properties": {"deployment_status": {"type": "string"}}},
                handler=self._deploy_artifact
            )
        ]
        self.logger.info(f"BuildAgent {self.id} initialized with {len(self.capabilities)} capabilities.")
        return True

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        for capability in self.capabilities:
            if capability.name == action:
                return await capability.handler(params)
        return {"status": "error", "message": f"Action '{action}' not supported by BuildAgent."}
    
    async def _build_project(self, params: Dict[str, Any]) -> Dict[str, Any]:
        source_code_resource_id = params.get("source_code_resource_id")
        build_config = params.get("build_config", {})

        self.logger.info(f"Building project from resource ID: {source_code_resource_id}")

        source_code_resource = await self.framework.resource_manager.get_resource(source_code_resource_id)
        if not source_code_resource:
            self.logger.error(f"Source code resource {source_code_resource_id} not found.")
            return {"status": "failed", "error": "Source code resource not found."}
        
        # Simular proceso de compilación/construcción
        self.logger.info(f"Simulating build for {source_code_resource.name}...")
        await asyncio.sleep(2) # Simular trabajo
        
        build_output = f"Compiled artifact for {source_code_resource.name} with config {build_config}"
        
        # Crear un recurso para el artefacto de construcción
        artifact_resource = AgentResource(
            type=ResourceType.RELEASE, # O un tipo más específico como 'BUILD_ARTIFACT'
            name=f"{source_code_resource.name}_build_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            namespace="build.artifacts",
            data=build_output,
            owner_agent_id=self.id
        )
        await self.framework.resource_manager.create_resource(artifact_resource)

        await self.publish_event("build.completed", {"source_resource_id": source_code_resource_id, "artifact_resource_id": artifact_resource.id})

        return {"status": "success", "build_status": "completed", "artifacts_resource_id": artifact_resource.id}

    async def _deploy_artifact(self, params: Dict[str, Any]) -> Dict[str, Any]:
        artifact_resource_id = params.get("artifact_resource_id")
        target_environment = params.get("target_environment", "development")

        self.logger.info(f"Deploying artifact {artifact_resource_id} to {target_environment} environment.")

        artifact_resource = await self.framework.resource_manager.get_resource(artifact_resource_id)
        if not artifact_resource:
            self.logger.error(f"Artifact resource {artifact_resource_id} not found.")
            return {"status": "failed", "error": "Artifact resource not found."}
        
        # Simular despliegue
        self.logger.info(f"Simulating deployment of {artifact_resource.name} to {target_environment}...")
        await asyncio.sleep(3) # Simular trabajo

        deployment_status = "deployed"
        if target_environment == "production":
            deployment_status = "deployed_to_production" # Más detallado
        
        await self.publish_event("deployment.completed", {"artifact_resource_id": artifact_resource_id, "environment": target_environment, "status": deployment_status})

        return {"status": "success", "deployment_status": deployment_status}


# ================================\
# TESTING AGENTS
# ================================\

class TestGeneratorAgent(BaseAgent):
    """agent.test.generator - Agente generador de pruebas"""

    def __init__(self, name: str, framework: AgentFramework):
        super().__init__("agent.test.generator", name, framework)

    async def initialize(self) -> bool:
        self.capabilities = [
            AgentCapability(
                name="generate.tests",
                namespace="agent.test.generator.generate_tests",
                description="Generates test cases for given code or specification.",
                input_schema={"type": "object", "properties": {"code": {"type": "string"}, "test_framework": {"type": "string"}}},
                output_schema={"type": "object", "properties": {"test_code": {"type": "string"}, "test_report": {"type": "object"}}},
                handler=self._generate_tests
            ),
            AgentCapability(
                name="execute.tests",
                namespace="agent.test.generator.execute_tests",
                description="Executes generated tests and provides a report.",
                input_schema={"type": "object", "properties": {"test_code_resource_id": {"type": "string"}, "target_code_resource_id": {"type": "string"}}},
                output_schema={"type": "object", "properties": {"execution_status": {"type": "string"}, "full_report": {"type": "object"}}},
                handler=self._execute_tests
            )
        ]
        self.logger.info(f"TestGeneratorAgent {self.id} initialized with {len(self.capabilities)} capabilities.")
        return True

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        for capability in self.capabilities:
            if capability.name == action:
                return await capability.handler(params)
        return {"status": "error", "message": f"Action '{action}' not supported by TestGeneratorAgent."}

    async def _generate_tests(self, params: Dict[str, Any]) -> Dict[str, Any]:
        code = params.get("code", "")
        test_framework = params.get("test_framework", "pytest")
        
        self.logger.info(f"Generating tests for code using {test_framework}.")

        # Simular generación de código de prueba
        test_code_content = f"""
import {test_framework}

def test_generated_function():
    # Test for: {code[:50]}...
    assert True # Placeholder for actual test logic
"""
        test_report_summary = {"tests_generated": 1, "framework": test_framework, "code_checksum": "abc"}

        # Almacenar el código de prueba como un recurso
        test_resource = AgentResource(
            type=ResourceType.TEST,
            name=f"test_case_{uuid.uuid4()}.py",
            namespace="test.generated",
            data=test_code_content,
            owner_agent_id=self.id
        )
        await self.framework.resource_manager.create_resource(test_resource)

        await self.publish_event("tests.generated", {"test_resource_id": test_resource.id, "summary": test_report_summary})

        return {"status": "success", "test_code": test_code_content, "test_report": test_report_summary, "resource_id": test_resource.id}

    async def _execute_tests(self, params: Dict[str, Any]) -> Dict[str, Any]:
        test_code_resource_id = params.get("test_code_resource_id")
        target_code_resource_id = params.get("target_code_resource_id")

        self.logger.info(f"Executing tests from {test_code_resource_id} against {target_code_resource_id}")

        test_code_res = await self.framework.resource_manager.get_resource(test_code_resource_id)
        target_code_res = await self.framework.resource_manager.get_resource(target_code_resource_id)

        if not test_code_res or not target_code_res:
            self.logger.error("Test or target code resource not found.")
            return {"status": "failed", "error": "Missing test or target code resource."}

        # Simular ejecución de pruebas
        self.logger.info("Simulating test execution...")
        await asyncio.sleep(2) # Simular trabajo

        execution_status = "passed"
        full_report = {
            "total_tests": 1,
            "passed": 1,
            "failed": 0,
            "errors": 0,
            "duration_seconds": 1.5,
            "details": f"Tests for {target_code_res.name} executed successfully."
        }

        await self.publish_event("tests.executed", {"test_resource_id": test_code_resource_id, "target_resource_id": target_code_resource_id, "status": execution_status, "report": full_report})

        return {"status": "success", "execution_status": execution_status, "full_report": full_report}

# ================================\
# Agent Factory Extension (if needed)
# ================================\

class ExtendedAgentFactory:
    """
    Una factoría extendida para crear agentes especializados.
    Podría vivir en un módulo separado como `agents/factory.py` o similar.
    """
    @staticmethod
    def create_agent(namespace: str, name: str, agent_class: Type[BaseAgent], framework: AgentFramework, initial_params: Optional[Dict[str, Any]] = None) -> BaseAgent:
        """Crea y devuelve una instancia de un agente especializado."""
        agent = agent_class(name=name, framework=framework)
        agent.namespace = namespace # Asegurar que el namespace sea el correcto

        if initial_params:
            for key, value in initial_params.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)
                else:
                    logging.warning(f"Initial parameter '{key}' not found on agent {name}.")

        # Note: initialize() and start() are usually called by the framework or a higher-level orchestrator
        # after creation, not directly by the factory unless it's explicitly designed to do so.
        # For the demo, we might call them immediately after creation for convenience.
        return agent

# ================================\
# DEMO
# ================================\

async def advanced_example():
    """Ejemplo de interacción entre agentes especializados."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("🚀 Starting Advanced Agent Collaboration Demo")
    print("="*50)

    framework = AgentFramework()
    await framework.start()

    # Crear instancias de agentes especializados usando la factoría
    strategist = ExtendedAgentFactory.create_agent("agent.planning.strategist", "StrategistAlpha", framework)
    workflow_designer = ExtendedAgentFactory.create_agent("agent.planning.workflow_designer", "WorkflowDesignerGamma", framework)
    code_generator = ExtendedAgentFactory.create_agent("agent.build.code.generator", "CodeGeneratorOmega", framework)
    test_generator = ExtendedAgentFactory.create_agent("agent.test.generator", "TestGeneratorBeta", framework)
    build_agent = ExtendedAgentFactory.create_agent("agent.build.builder", "BuildAgentSigma", framework)

    # Inicializar y arrancar todos los agentes
    agents_to_start = [strategist, workflow_designer, code_generator, test_generator, build_agent]
    for agent in agents_to_start:
        await agent.initialize()
        await agent.start()
        print(f"Agent {agent.name} ({agent.id}) started.")

    try:
        # Escenario de colaboración:
        # 1. Estratega define una estrategia
        # 2. Diseñador de Workflow crea un workflow basado en la estrategia
        # 3. Generador de Código genera código basado en una especificación
        # 4. Generador de Tests genera tests para el código
        # 5. Build Agent compila el código y luego lo despliega (simulado)

        print("\n--- Starting Collaboration Scenario ---")

        # 1. Estratega define una estrategia
        strategy_result = await strategist.execute_action(
            "define.strategy",
            {"requirements": {"goal": "Develop a microservice for user management"}, "constraints": {"budget": "low"}}
        )
        print("Strategy defined:", strategy_result["strategy"]["overall_goal"])

        # 2. Diseñador de Workflow crea un workflow
        workflow_result = await workflow_designer.execute_action(
            "create.workflow",
            {"tasks": ["design_api", "implement_service", "write_tests", "deploy_service"], "priority": "high"}
        )
        print("Workflow designed with", len(workflow_result["workflow"]["steps"]), "steps")

        # 3. Generar código
        code_result = await code_generator.execute_action("generate.component", {
            "specification": {
                "name": "UserService",
                "methods": [
                    {"name": "create_user", "parameters": [{"name": "user_data", "type": "dict"}]},
                    {"name": "get_user", "parameters": [{"name": "user_id", "type": "str"}]}
                ]
            },
            "language": "python"
        })
        print(f"Generated code for UserService. Resource ID: {code_result['resource_id']}")

        # 4. Generar tests
        test_result = await test_generator.execute_action("generate.tests", {
            "code": code_result["code"], # Usar el código generado
            "test_framework": "pytest"
        })
        print(f"Generated tests for UserService. Test Resource ID: {test_result['resource_id']}")

        # 5. Ejecutar tests (opcional, una capacidad del TestGeneratorAgent)
        execution_report = await test_generator.execute_action("execute.tests", {
            "test_code_resource_id": test_result["resource_id"],
            "target_code_resource_id": code_result["resource_id"]
        })
        print(f"Test execution status: {execution_report['execution_status']}")

        # 6. Build Agent compila y despliega (usando los recursos creados)
        build_result = await build_agent.execute_action("build.project", {
            "source_code_resource_id": code_result["resource_id"],
            "build_config": {"target": "docker_image"}
        })
        print(f"Build status: {build_result['build_status']}. Artifact Resource ID: {build_result['artifacts_resource_id']}")

        deploy_result = await build_agent.execute_action("deploy.artifact", {
            "artifact_resource_id": build_result["artifacts_resource_id"],
            "target_environment": "staging"
        })
        print(f"Deployment status: {deploy_result['deployment_status']}")


        # Mostrar recursos creados
        all_resources = []
        for agent in agents_to_start:
            agent_resources = framework.resource_manager.find_resources_by_owner(agent.id)
            all_resources.extend(agent_resources)
            
        print(f"\nTotal resources created: {len(all_resources)}")
        for resource in all_resources:
            print(f"  - {resource.type.value}: {resource.name} (owner: {resource.owner_agent_id})")
            
    finally:
        await framework.stop()
        print("\nAdvanced demo finished.")

if __name__ == "__main__":
    asyncio.run(advanced_example())