"""
specialized_agents.py - Implementaciones específicas de agentes por namespace
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional

# Importaciones actualizadas de los modelos y clases base desde 'core'
from core.autonomous_agent_framework import BaseAgent, AgentFramework # BaseAgent y AgentFramework permanecen aquí
from core.models import AgentCapability, AgentResource, ResourceType, AgentStatus # Modelos movidos a core.models
from core.registry import AgentRegistry # Para listar agentes, aunque ya se accede via framework.registry

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
                name="define_strategy",
                namespace="agent.planning.strategist.define",
                description="Define strategic plans and roadmaps",
                input_schema={"requirements": "object", "constraints": "object"},
                output_schema={"strategy": "object", "roadmap": "array"},
                handler=self._define_strategy
            ),
            AgentCapability(
                name="request.workflow.design",
                namespace="agent.planning.strategist.request.workflow",
                description="Requests a workflow design from a Workflow Designer Agent.",
                input_schema={"project_goal": {"type": "string"}, "scope": {"type": "string"}},
                output_schema={"status": {"type": "string"}, "workflow_id": {"type": "string"}},
                handler=self._request_workflow_design
            )
        ]
        self.logger.info(f"{self.name} initialized with {len(self.capabilities)} capabilities.")
        return True
        
    # Método genérico para ejecutar acciones si no se usa directamente el MessageBus
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        for capability in self.capabilities:
            if capability.name == action or capability.namespace == action:
                return await capability.handler(params)
        return {"error": f"Unknown action: {action}"}
        
    async def _define_strategy(self, params: Dict[str, Any]) -> Dict[str, Any]:
        requirements = params.get("requirements", {})
        constraints = params.get("constraints", {})
        
        # Lógica de definición de estrategia
        strategy = {
            "overall_goal": requirements.get("goal", "Develop a new product"),
            "phases": [
                {"name": "Discovery", "duration": "2 weeks"},
                {"name": "Design", "duration": "3 weeks"},
                {"name": "Development", "duration": "8 weeks"},
                {"name": "Testing", "duration": "4 weeks"},
                {"name": "Deployment", "duration": "1 week"}
            ],
            "key_constraints": constraints
        }
        
        self.logger.info(f"Strategy defined: {strategy['overall_goal']}")
        
        # Opcional: Crear un recurso de tipo 'workflow' para la estrategia
        strategy_resource = AgentResource(
            type=ResourceType.WORKFLOW,
            name=f"strategy_{self.name}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            namespace="strategy.document",
            data=strategy,
            owner_agent_id=self.id
        )
        await self.framework.resource_manager.create_resource(strategy_resource)
        
        return {"status": "success", "strategy": strategy, "resource_id": strategy_resource.id}

    async def _request_workflow_design(self, params: Dict[str, Any]) -> Dict[str, Any]:
        project_goal = params.get("project_goal", "No specific project goal.")
        scope = params.get("scope", "Entire system.")

        self.logger.info(f"Requesting workflow design for: {project_goal} (Scope: {scope})")

        # Enviar mensaje al WorkflowDesignerAgent
        # Necesitamos encontrar el ID del WorkflowDesignerAgent
        workflow_designer_agent_id = None
        for agent_info in self.framework.registry.get_agent_info_list():
            if agent_info.namespace == "agent.planning.workflow_designer":
                workflow_designer_agent_id = agent_info.id
                break
        
        if not workflow_designer_agent_id:
            self.logger.error("WorkflowDesignerAgent not found in registry.")
            return {"status": "failed", "error": "WorkflowDesignerAgent not available."}

        response_future = asyncio.Future()
        
        # Registrar un callback para la respuesta
        correlation_id = str(uuid.uuid4()) # Generar un ID de correlación único para esta solicitud
        self.framework.message_bus.register_response_handler(correlation_id, response_future) # Asumimos que MessageBus tiene este método

        message_id = await self.send_message(
            receiver_id=workflow_designer_agent_id,
            action="action.create.workflow",
            params={
                "goal": project_goal,
                "scope": scope,
                "requester_id": self.id # Para que el diseñador sepa a quién responder
            },
            message_type=MessageType.REQUEST,
            correlation_id=correlation_id # Incluir el ID de correlación
        )

        try:
            response_message = await asyncio.wait_for(response_future, timeout=30.0) # Esperar la respuesta
            self.logger.info(f"Received response for workflow design: {response_message.payload}")
            return {"status": "success", "workflow": response_message.payload.get("workflow"), "message_id": message_id}
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout waiting for workflow design response from {workflow_designer_agent_id}")
            return {"status": "failed", "error": "Timeout waiting for workflow design."}
        finally:
            self.framework.message_bus.unregister_response_handler(correlation_id) # Limpiar el handler


class WorkflowDesignerAgent(BaseAgent):
    """agent.planning.workflow_designer - Agente diseñador de flujos de trabajo"""

    def __init__(self, name: str, framework: AgentFramework):
        super().__init__("agent.planning.workflow_designer", name, framework)

    async def initialize(self) -> bool:
        self.capabilities = [
            AgentCapability(
                name="create.workflow",
                namespace="agent.planning.workflow_designer.create",
                description="Designs a comprehensive development workflow.",
                input_schema={
                    "goal": {"type": "string"}, 
                    "scope": {"type": "string"}, 
                    "requester_id": {"type": "string", "optional": True}
                },
                output_schema={"workflow": {"type": "object"}, "status": {"type": "string"}},
                handler=self._create_workflow
            )
        ]
        self.logger.info(f"{self.name} initialized with {len(self.capabilities)} capabilities.")
        return True

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        for capability in self.capabilities:
            if capability.name == action or capability.namespace == action:
                return await capability.handler(params)
        return {"error": f"Unknown action: {action}"}

    async def _create_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        goal = params.get("goal", "Generic project development")
        scope = params.get("scope", "Full stack")
        requester_id = params.get("requester_id")

        self.logger.info(f"Designing workflow for goal: {goal} (Scope: {scope})")

        # Lógica de diseño de flujo de trabajo (ejemplo simplificado)
        workflow = {
            "name": f"Workflow for {goal}",
            "description": f"Designed workflow for '{goal}' with scope '{scope}'.",
            "steps": [
                {"name": "Requirements Analysis", "agent": "strategist", "status": "pending"},
                {"name": "Architecture Design", "agent": "architect", "status": "pending"},
                {"name": "Frontend Development", "agent": "code_generator", "language": "react", "status": "pending"},
                {"name": "Backend Development", "agent": "code_generator", "language": "python", "status": "pending"},
                {"name": "Database Setup", "agent": "infra_agent", "status": "pending"},
                {"name": "Unit Testing", "agent": "test_generator", "status": "pending"},
                {"name": "Integration Testing", "agent": "test_runner", "status": "pending"},
                {"name": "Deployment Preparation", "agent": "build_agent", "status": "pending"}
            ],
            "estimated_duration_days": 30
        }

        self.logger.info(f"Workflow '{workflow['name']}' designed.")

        # Opcional: Crear un recurso de tipo 'workflow'
        workflow_resource = AgentResource(
            type=ResourceType.WORKFLOW,
            name=f"workflow_{self.name}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            namespace="workflow.design",
            data=workflow,
            owner_agent_id=self.id
        )
        await self.framework.resource_manager.create_resource(workflow_resource)
        self.logger.info(f"Workflow saved as resource: {workflow_resource.id}")

        response_payload = {"status": "success", "workflow": workflow, "resource_id": workflow_resource.id}

        if requester_id:
            # Enviar una respuesta al agente que hizo la solicitud original
            response_message = AgentMessage(
                sender_id=self.id,
                receiver_id=requester_id,
                message_type=MessageType.RESPONSE,
                payload=response_payload,
                correlation_id=params.get("correlation_id") # Usar el ID de correlación original si existe
            )
            await self.framework.message_bus.send_message(response_message)
            self.logger.info(f"Sent workflow design response to {requester_id}")

        return response_payload


# ================================\
# DEVELOPMENT AGENTS
# ================================\

class CodeGeneratorAgent(BaseAgent):
    """agent.build.code.generator - Agente generador de código"""

    def __init__(self, name: str, framework: AgentFramework):
        super().__init__("agent.build.code.generator", name, framework)

    async def initialize(self) -> bool:
        self.capabilities = [
            AgentCapability(
                name="generate.component",
                namespace="agent.build.code.generator.generate",
                description="Generates code for a specified component.",
                input_schema={
                    "specification": {"type": "object", "description": "Component specification"},
                    "language": {"type": "string", "enum": ["python", "javascript", "java"]}
                },
                output_schema={"code": {"type": "string"}, "status": {"type": "string"}},
                handler=self._generate_component_code
            )
        ]
        self.logger.info(f"{self.name} initialized with {len(self.capabilities)} capabilities.")
        return True

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        for capability in self.capabilities:
            if capability.name == action or capability.namespace == action:
                return await capability.handler(params)
        return {"error": f"Unknown action: {action}"}

    async def _generate_component_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        spec = params.get("specification", {})
        language = params.get("language", "python")

        self.logger.info(f"Generating {language} code for component: {spec.get('name', 'unnamed')}")

        # Lógica de generación de código (simulada)
        generated_code = f"# Generated {language} code for {spec.get('name', 'component')}\n\n"
        if language == "python":
            generated_code += f"class {spec.get('name', 'MyComponent')}:\n"
            for method in spec.get("methods", []):
                params_str = ", ".join([f"{p['name']}: {p['type']}" for p in method.get("parameters", [])])
                generated_code += f"    def {method['name']}(self, {params_str}):\n"
                generated_code += f"        # Implementation for {method['name']}\n        pass\n"
        elif language == "javascript":
            generated_code += f"class {spec.get('name', 'MyComponent')} {{\n"
            for method in spec.get("methods", []):
                params_str = ", ".join([p['name'] for p in method.get("parameters", [])])
                generated_code += f"    {method['name']}({params_str}) {{\n"
                generated_code += f"        // Implementation for {method['name']}\n    }}\n"
            generated_code += "}\n"

        # Crear un recurso de tipo 'code'
        code_resource = AgentResource(
            type=ResourceType.CODE,
            name=f"{spec.get('name', 'generated_code')}.{language}",
            namespace="code.generated",
            data=generated_code,
            owner_agent_id=self.id
        )
        await self.framework.resource_manager.create_resource(code_resource)
        self.logger.info(f"Generated code saved as resource: {code_resource.id}")

        return {"status": "success", "code": generated_code, "resource_id": code_resource.id}


class TestGeneratorAgent(BaseAgent):
    """agent.test.generator - Agente generador de tests"""

    def __init__(self, name: str, framework: AgentFramework):
        super().__init__("agent.test.generator", name, framework)

    async def initialize(self) -> bool:
        self.capabilities = [
            AgentCapability(
                name="generate.tests",
                namespace="agent.test.generator.generate",
                description="Generates test cases for given code.",
                input_schema={
                    "code": {"type": "string", "description": "Code to be tested"},
                    "test_framework": {"type": "string", "enum": ["pytest", "jest", "junit"]}
                },
                output_schema={"tests": {"type": "string"}, "status": {"type": "string"}},
                handler=self._generate_tests_for_code
            )
        ]
        self.logger.info(f"{self.name} initialized with {len(self.capabilities)} capabilities.")
        return True

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        for capability in self.capabilities:
            if capability.name == action or capability.namespace == action:
                return await capability.handler(params)
        return {"error": f"Unknown action: {action}"}

    async def _generate_tests_for_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        code_to_test = params.get("code", "")
        test_framework = params.get("test_framework", "pytest")

        self.logger.info(f"Generating {test_framework} tests for provided code.")

        # Lógica de generación de tests (simulada)
        generated_tests = f"# Generated {test_framework} tests\n\n"
        if test_framework == "pytest":
            generated_tests += "import pytest\n\ndef test_example_function():\n    assert True # Placeholder test\n"
        elif test_framework == "jest":
            generated_tests += "test('example test', () => {\n  expect(true).toBe(true);\n});\n"
        
        # Crear un recurso de tipo 'test'
        test_resource = AgentResource(
            type=ResourceType.TEST,
            name=f"tests_for_code_{datetime.now().strftime('%Y%m%d%H%M%S')}.{test_framework}",
            namespace="test.generated",
            data=generated_tests,
            owner_agent_id=self.id
        )
        await self.framework.resource_manager.create_resource(test_resource)
        self.logger.info(f"Generated tests saved as resource: {test_resource.id}")

        return {"status": "success", "tests": generated_tests, "resource_id": test_resource.id}


class BuildAgent(BaseAgent):
    """agent.build.builder - Agente de construcción (compilación/empaquetado)"""

    def __init__(self, name: str, framework: AgentFramework):
        super().__init__("agent.build.builder", name, framework)

    async def initialize(self) -> bool:
        self.capabilities = [
            AgentCapability(
                name="build.project",
                namespace="agent.build.builder.build",
                description="Builds a project from source code.",
                input_schema={
                    "source_code_resource_id": {"type": "string", "description": "ID of the code resource"},
                    "build_command": {"type": "string", "description": "Command to execute build (e.g., 'npm run build', 'mvn clean install')", "optional": True},
                    "output_format": {"type": "string", "enum": ["docker_image", "zip", "executable"], "optional": True}
                },
                output_schema={"build_artifact_id": {"type": "string"}, "status": {"type": "string"}, "logs": {"type": "string"}},
                handler=self._build_project
            )
        ]
        self.logger.info(f"{self.name} initialized with {len(self.capabilities)} capabilities.")
        return True

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        for capability in self.capabilities:
            if capability.name == action or capability.namespace == action:
                return await capability.handler(params)
        return {"error": f"Unknown action: {action}"}

    async def _build_project(self, params: Dict[str, Any]) -> Dict[str, Any]:
        source_code_resource_id = params.get("source_code_resource_id")
        build_command = params.get("build_command", "build")
        output_format = params.get("output_format", "zip")

        if not source_code_resource_id:
            return {"status": "failed", "error": "source_code_resource_id is required."}

        source_resource = await self.framework.resource_manager.get_resource(source_code_resource_id)
        if not source_resource or source_resource.type != ResourceType.CODE:
            return {"status": "failed", "error": f"Source code resource {source_code_resource_id} not found or is not of type CODE."}

        self.logger.info(f"Building project from resource {source_code_resource_id} with command '{build_command}'.")

        # Simular proceso de construcción
        build_logs = []
        build_logs.append(f"Starting build for {source_resource.name}...")
        build_logs.append(f"Executing command: {build_command}...")
        await asyncio.sleep(2) # Simular trabajo
        build_logs.append("Build process completed successfully.")

        # Crear un artefacto de construcción como recurso
        artifact_data = {
            "format": output_format,
            "content": f"Simulated {output_format} artifact for {source_resource.name}",
            "source_resource_id": source_code_resource_id
        }

        build_artifact_resource = AgentResource(
            type=ResourceType.RELEASE, # O un tipo más específico como ARTIFACT
            name=f"build_artifact_{source_resource.name.replace('.', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.{output_format}",
            namespace="build.artifacts",
            data=artifact_data,
            owner_agent_id=self.id
        )
        await self.framework.resource_manager.create_resource(build_artifact_resource)
        self.logger.info(f"Build artifact saved as resource: {build_artifact_resource.id}")

        return {
            "status": "success",
            "build_artifact_id": build_artifact_resource.id,
            "logs": "\n".join(build_logs)
        }

# ================================\
# HELPER FOR DEMO: EXTENDED AGENT FACTORY
# (This will eventually be refactored into core/agent_factory.py)
# ================================\

class ExtendedAgentFactory:
    """
    Factoría extendida para crear agentes especializados.
    Esta clase se usará en los ejemplos y tests, y eventualmente se integrará en el
    framework principal como core.agent_factory.
    """
    @staticmethod
    async def create_agent(namespace: str, name: str, agent_class: Type[BaseAgent], framework: AgentFramework) -> BaseAgent:
        """Crea una instancia de un agente y la registra en el framework."""
        agent = agent_class(name=name, framework=framework)
        agent.namespace = namespace # Asegurar el namespace correcto
        await agent.initialize()
        await agent.start()
        return agent

# Este ejemplo de uso ahora se moverá a end_to_end_example.py o a un archivo de tests
# y no será parte de este módulo una vez que se complete la refactorización.
"""
async def advanced_example():
    \"\"\"Ejemplo avanzado de colaboración entre agentes.\"\"\"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("AdvancedExample")

    framework = AgentFramework()
    
    try:
        await framework.start()

        # Crear agentes
        strategist = await ExtendedAgentFactory.create_agent("agent.planning.strategist", "Strategist", framework)
        workflow_designer = await ExtendedAgentFactory.create_agent("agent.planning.workflow_designer", "WorkflowDesigner", framework)
        code_generator = await ExtendedAgentFactory.create_agent("agent.build.code.generator", "CodeGenerator", framework)
        test_generator = await ExtendedAgentFactory.create_agent("agent.test.generator", "TestGenerator", framework)
        build_agent_instance = await ExtendedAgentFactory.create_agent("agent.build.builder", "BuildAgent", framework)
        
        agents = {
            'strategist': strategist,
            'workflow_designer': workflow_designer,
            'code_generator': code_generator,
            'test_generator': test_generator,
            'build_agent': build_agent_instance
        }

        # 1. El estratega define una estrategia (crea un recurso)
        strategy_result = await strategist.execute_action("define_strategy", {
            "requirements": {"goal": "Develop a user authentication service", "tech_stack": "Python, React"},
            "constraints": {"budget": "medium", "time": "3 months"}
        })
        print(f"Strategy defined. Resource ID: {strategy_result['resource_id']}")
        
        # 2. El estratega solicita un workflow al diseñador de workflows
        workflow_result = await strategist.execute_action("request.workflow.design", {
            "project_goal": "User Authentication Service",
            "scope": "Frontend and Backend"
        })
        
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
        
        print("Generated code for UserService")
        
        # 4. Generar tests
        test_result = await test_generator.execute_action("generate.tests", {
            "code": code_result["code"],
            "test_framework": "pytest"
        })
        
        print("Generated tests for UserService")

        # 5. Build del código
        build_result = await build_agent_instance.execute_action("build.project", {
            "source_code_resource_id": code_result["resource_id"],
            "build_command": "python setup.py build", # Comando de ejemplo
            "output_format": "zip"
        })
        print(f"Build completed. Artifact ID: {build_result['build_artifact_id']}")
        
        # Mostrar recursos creados
        all_resources = []
        for agent in agents.values():
            agent_resources = framework.resource_manager.find_resources_by_owner(agent.id)
            all_resources.extend(agent_resources)
            
        print(f"\nTotal resources created: {len(all_resources)}")
        for resource in all_resources:
            print(f"  - {resource.type.value}: {resource.name} (owner: {resource.owner_agent_id})")
            
    finally:
        await framework.stop()

if __name__ == "__main__":
    asyncio.run(advanced_example())
"""