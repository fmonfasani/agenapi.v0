"""
specialized_agents.py - Implementaciones específicas de agentes por namespace
"""

from core.autonomous_agent_framework import BaseAgent, AgentCapability, AgentResource, ResourceType
from typing import Dict, Any, List
import asyncio
import json
import logging

# ================================
# PLANNING AGENTS
# ================================

class StrategistAgent(BaseAgent):
    """agent.planning.strategist - Agente estratega"""
    
    def __init__(self, name: str, framework):
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
            )
        ]
        return True
        
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "define.strategy":
            return await self._define_strategy(params)
        return {"error": f"Unknown action: {action}"}
        
    async def _define_strategy(self, params: Dict[str, Any]) -> Dict[str, Any]:
        requirements = params.get("requirements", {})
        constraints = params.get("constraints", {})
        
        # Crear estrategia basada en requisitos
        strategy = {
            "vision": f"Deliver {requirements.get('goal', 'solution')} efficiently",
            "objectives": [
                "Analyze current state",
                "Design optimal solution",
                "Implement incrementally",
                "Monitor and optimize"
            ],
            "phases": [
                {
                    "name": "Discovery",
                    "duration": "2 weeks",
                    "deliverables": ["requirements_doc", "architecture_design"]
                },
                {
                    "name": "Development",
                    "duration": "6 weeks", 
                    "deliverables": ["core_features", "tests", "documentation"]
                },
                {
                    "name": "Deployment",
                    "duration": "1 week",
                    "deliverables": ["production_release", "monitoring_setup"]
                }
            ],
            "risks": constraints.get("risks", []),
            "success_metrics": ["quality", "performance", "user_satisfaction"]
        }
        
        # Crear recurso de estrategia
        strategy_resource = AgentResource(
            type=ResourceType.WORKFLOW,
            name="project_strategy",
            namespace="resource.workflow.strategy",
            data=strategy,
            owner_agent_id=self.id
        )
        
        await self.framework.resource_manager.create_resource(strategy_resource)
        
        return {
            "strategy": strategy,
            "resource_id": strategy_resource.id
        }

class WorkflowAgent(BaseAgent):
    """agent.planning.workflow - Agente de diseño de workflows"""
    
    def __init__(self, name: str, framework):
        super().__init__("agent.planning.workflow", name, framework)
        
    async def initialize(self) -> bool:
        self.capabilities = [
            AgentCapability(
                name="design_workflow",
                namespace="agent.planning.workflow.design",
                description="Design and optimize workflows",
                input_schema={"tasks": "array", "constraints": "object"},
                output_schema={"workflow": "object"},
                handler=self._design_workflow
            )
        ]
        return True
        
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "design.workflow":
            return await self._design_workflow(params)
        return {"error": f"Unknown action: {action}"}
        
    async def _design_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        tasks = params.get("tasks", [])
        constraints = params.get("constraints", {})
        
        # Analizar dependencias entre tareas
        workflow_steps = []
        for i, task in enumerate(tasks):
            step = {
                "id": f"step_{i+1}",
                "name": task.get("name", f"Task {i+1}"),
                "type": task.get("type", "manual"),
                "dependencies": task.get("dependencies", []),
                "estimated_duration": task.get("duration", "1h"),
                "assigned_agent": task.get("agent", None),
                "parallel_execution": task.get("parallel", False)
            }
            workflow_steps.append(step)
        
        workflow = {
            "id": f"workflow_{len(tasks)}_tasks",
            "name": params.get("name", "Generated Workflow"),
            "steps": workflow_steps,
            "parallel_branches": self._identify_parallel_branches(workflow_steps),
            "critical_path": self._calculate_critical_path(workflow_steps),
            "estimated_total_time": self._estimate_total_time(workflow_steps)
        }
        
        return {"workflow": workflow}
        
    def _identify_parallel_branches(self, steps: List[Dict]) -> List[List[str]]:
        """Identificar ramas que pueden ejecutarse en paralelo"""
        parallel_branches = []
        independent_steps = [step for step in steps if not step["dependencies"]]
        if len(independent_steps) > 1:
            parallel_branches.append([step["id"] for step in independent_steps])
        return parallel_branches
        
    def _calculate_critical_path(self, steps: List[Dict]) -> List[str]:
        """Calcular ruta crítica del workflow"""
        # Implementación simplificada
        return [step["id"] for step in steps]
        
    def _estimate_total_time(self, steps: List[Dict]) -> str:
        """Estimar tiempo total del workflow"""
        # Implementación simplificada
        return f"{len(steps) * 2}h"

# ================================
# BUILD AGENTS
# ================================

class CodeGeneratorAgent(BaseAgent):
    """agent.build.code.generator - Generador de código"""
    
    def __init__(self, name: str, framework):
        super().__init__("agent.build.code.generator", name, framework)
        
    async def initialize(self) -> bool:
        self.capabilities = [
            AgentCapability(
                name="generate_component",
                namespace="agent.build.code.generator.component",
                description="Generate code components",
                input_schema={"specification": "object", "language": "string"},
                output_schema={"code": "string", "files": "array"},
                handler=self._generate_component
            )
        ]
        return True
        
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "generate.component":
            return await self._generate_component(params)
        return {"error": f"Unknown action: {action}"}
        
    async def _generate_component(self, params: Dict[str, Any]) -> Dict[str, Any]:
        spec = params.get("specification", {})
        language = params.get("language", "python")
        
        if language == "python":
            code = self._generate_python_component(spec)
        elif language == "javascript":
            code = self._generate_javascript_component(spec)
        else:
            return {"error": f"Unsupported language: {language}"}
        
        # Crear recurso de código
        code_resource = AgentResource(
            type=ResourceType.CODE,
            name=spec.get("name", "generated_component"),
            namespace="resource.code.component",
            data={"code": code, "language": language, "specification": spec},
            owner_agent_id=self.id
        )
        
        await self.framework.resource_manager.create_resource(code_resource)
        
        return {
            "code": code,
            "resource_id": code_resource.id,
            "language": language
        }
        
    def _generate_python_component(self, spec: Dict[str, Any]) -> str:
        """Generar componente Python"""
        name = spec.get("name", "Component")
        methods = spec.get("methods", [])
        
        code = f'''"""
{name} - Generated component
"""

class {name}:
    """Generated component based on specification"""
    
    def __init__(self):
        self.initialized = True
        
'''
        
        for method in methods:
            method_name = method.get("name", "method")
            params = method.get("parameters", [])
            return_type = method.get("return_type", "None")
            
            param_str = ", ".join([f"{p['name']}: {p.get('type', 'Any')}" for p in params])
            
            code += f'''    def {method_name}(self{", " + param_str if param_str else ""}) -> {return_type}:
        """Generated method: {method.get('description', 'No description')}"""
        # TODO: Implement {method_name}
        pass
        
'''
        
        return code
        
    def _generate_javascript_component(self, spec: Dict[str, Any]) -> str:
        """Generar componente JavaScript"""
        name = spec.get("name", "Component")
        methods = spec.get("methods", [])
        
        code = f'''/**
 * {name} - Generated component
 */

class {name} {{
    constructor() {{
        this.initialized = true;
    }}
    
'''
        
        for method in methods:
            method_name = method.get("name", "method")
            params = method.get("parameters", [])
            
            param_str = ", ".join([p["name"] for p in params])
            
            code += f'''    {method_name}({param_str}) {{
        // TODO: Implement {method_name}
        // {method.get('description', 'No description')}
    }}
    
'''
        
        code += "}\n\nmodule.exports = " + name + ";"
        
        return code

class UXGeneratorAgent(BaseAgent):
    """agent.build.ux.generator - Generador de UX/UI"""
    
    def __init__(self, name: str, framework):
        super().__init__("agent.build.ux.generator", name, framework)
        
    async def initialize(self) -> bool:
        self.capabilities = [
            AgentCapability(
                name="generate_ui",
                namespace="agent.build.ux.generator.ui",
                description="Generate UI components and layouts",
                input_schema={"design_spec": "object", "framework": "string"},
                output_schema={"ui_code": "string", "styles": "string"},
                handler=self._generate_ui
            )
        ]
        return True
        
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "generate.ui":
            return await self._generate_ui(params)
        return {"error": f"Unknown action: {action}"}
        
    async def _generate_ui(self, params: Dict[str, Any]) -> Dict[str, Any]:
        design_spec = params.get("design_spec", {})
        framework = params.get("framework", "react")
        
        if framework == "react":
            ui_code = self._generate_react_component(design_spec)
            styles = self._generate_css_styles(design_spec)
        else:
            return {"error": f"Unsupported framework: {framework}"}
        
        # Crear recurso de UI
        ui_resource = AgentResource(
            type=ResourceType.UI,
            name=design_spec.get("name", "generated_ui"),
            namespace="resource.ui.component",
            data={"ui_code": ui_code, "styles": styles, "framework": framework},
            owner_agent_id=self.id
        )
        
        await self.framework.resource_manager.create_resource(ui_resource)
        
        return {
            "ui_code": ui_code,
            "styles": styles,
            "resource_id": ui_resource.id
        }
        
    def _generate_react_component(self, spec: Dict[str, Any]) -> str:
        """Generar componente React"""
        name = spec.get("name", "Component")
        props = spec.get("props", [])
        elements = spec.get("elements", [])
        
        props_interface = "interface Props {\n"
        for prop in props:
            props_interface += f"  {prop['name']}: {prop.get('type', 'string')};\n"
        props_interface += "}"
        
        jsx_elements = ""
        for element in elements:
            element_type = element.get("type", "div")
            content = element.get("content", "")
            props_str = element.get("props", "")
            
            jsx_elements += f"      <{element_type} {props_str}>{content}</{element_type}>\n"
        
        code = f'''import React from 'react';
import './{name}.css';

{props_interface}

const {name}: React.FC<Props> = (props) => {{
  return (
    <div className="{name.lower()}">
{jsx_elements}    </div>
  );
}};

export default {name};
'''
        
        return code
        
    def _generate_css_styles(self, spec: Dict[str, Any]) -> str:
        """Generar estilos CSS"""
        name = spec.get("name", "component")
        colors = spec.get("colors", {"primary": "#007bff", "secondary": "#6c757d"})
        
        css = f'''.{name.lower()} {{
  display: flex;
  flex-direction: column;
  padding: 1rem;
  background-color: {colors.get("background", "#ffffff")};
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}}

.{name.lower()} h1, .{name.lower()} h2, .{name.lower()} h3 {{
  color: {colors.get("primary", "#007bff")};
  margin-bottom: 0.5rem;
}}

.{name.lower()} button {{
  background-color: {colors.get("primary", "#007bff")};
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  cursor: pointer;
}}

.{name.lower()} button:hover {{
  opacity: 0.9;
}}
'''
        
        return css

# ================================
# TEST AGENTS
# ================================

class TestGeneratorAgent(BaseAgent):
    """agent.test.generator - Generador de tests"""
    
    def __init__(self, name: str, framework):
        super().__init__("agent.test.generator", name, framework)
        
    async def initialize(self) -> bool:
        self.capabilities = [
            AgentCapability(
                name="generate_tests",
                namespace="agent.test.generator.unit",
                description="Generate unit tests for code",
                input_schema={"code": "string", "test_framework": "string"},
                output_schema={"test_code": "string"},
                handler=self._generate_tests
            )
        ]
        return True
        
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "generate.tests":
            return await self._generate_tests(params)
        return {"error": f"Unknown action: {action}"}
        
    async def _generate_tests(self, params: Dict[str, Any]) -> Dict[str, Any]:
        code = params.get("code", "")
        test_framework = params.get("test_framework", "pytest")
        
        if test_framework == "pytest":
            test_code = self._generate_pytest_tests(code, params)
        elif test_framework == "jest":
            test_code = self._generate_jest_tests(code, params)
        else:
            return {"error": f"Unsupported test framework: {test_framework}"}
        
        # Crear recurso de test
        test_resource = AgentResource(
            type=ResourceType.TEST,
            name="generated_tests",
            namespace="resource.test.unit",
            data={"test_code": test_code, "framework": test_framework},
            owner_agent_id=self.id
        )
        
        await self.framework.resource_manager.create_resource(test_resource)
        
        return {
            "test_code": test_code,
            "resource_id": test_resource.id
        }
        
    def _generate_pytest_tests(self, code: str, params: Dict[str, Any]) -> str:
        """Generar tests con pytest"""
        # Análisis simple del código para extraer clases y métodos
        class_name = self._extract_class_name(code)
        methods = self._extract_methods(code)
        
        test_code = f'''"""
Generated tests for {class_name}
"""

import pytest
from unittest.mock import Mock, patch
from {class_name.lower()} import {class_name}


class Test{class_name}:
    
    def setup_method(self):
        self.instance = {class_name}()
        
    def test_initialization(self):
        assert self.instance is not None
        assert hasattr(self.instance, 'initialized')
        
'''
        
        for method in methods:
            test_code += f'''    def test_{method}(self):
        # TODO: Implement test for {method}
        result = self.instance.{method}()
        assert result is not None
        
'''
        
        return test_code
        
    def _generate_jest_tests(self, code: str, params: Dict[str, Any]) -> str:
        """Generar tests con Jest"""
        class_name = self._extract_class_name(code)
        methods = self._extract_methods(code)
        
        test_code = f'''/**
 * Generated tests for {class_name}
 */

const {class_name} = require('./{class_name.lower()}');

describe('{class_name}', () => {{
    let instance;
    
    beforeEach(() => {{
        instance = new {class_name}();
    }});
    
    test('should initialize correctly', () => {{
        expect(instance).toBeDefined();
        expect(instance.initialized).toBe(true);
    }});
    
'''
        
        for method in methods:
            test_code += f'''    test('should execute {method}', () => {{
        // TODO: Implement test for {method}
        const result = instance.{method}();
        expect(result).toBeDefined();
    }});
    
'''
        
        test_code += "});"
        
        return test_code
        
    def _extract_class_name(self, code: str) -> str:
        """Extraer nombre de clase del código"""
        lines = code.split('\n')
        for line in lines:
            if line.strip().startswith('class '):
                return line.split()[1].split('(')[0].split(':')[0]
        return "Component"
        
    def _extract_methods(self, code: str) -> List[str]:
        """Extraer nombres de métodos del código"""
        methods = []
        lines = code.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('def ') and not stripped.startswith('def __'):
                method_name = stripped.split('(')[0].replace('def ', '')
                methods.append(method_name)
        return methods

# ================================
# SECURITY AGENTS
# ================================

class SecuritySentinelAgent(BaseAgent):
    """agent.security.sentinel - Agente centinela de seguridad"""
    
    def __init__(self, name: str, framework):
        super().__init__("agent.security.sentinel", name, framework)
        
    async def initialize(self) -> bool:
        self.capabilities = [
            AgentCapability(
                name="scan_vulnerabilities",
                namespace="agent.security.sentinel.scan",
                description="Scan for security vulnerabilities",
                input_schema={"target": "string", "scan_type": "string"},
                output_schema={"threats": "array", "recommendations": "array"},
                handler=self._scan_vulnerabilities
            )
        ]
        return True
        
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "scan.vulnerabilities":
            return await self._scan_vulnerabilities(params)
        return {"error": f"Unknown action: {action}"}
        
    async def _scan_vulnerabilities(self, params: Dict[str, Any]) -> Dict[str, Any]:
        target = params.get("target", "")
        scan_type = params.get("scan_type", "basic")
        
        # Simulación de escaneo de seguridad
        threats = []
        recommendations = []
        
        if "password" in target.lower():
            threats.append({
                "severity": "high",
                "type": "weak_authentication",
                "description": "Weak password requirements detected"
            })
            recommendations.append("Implement strong password policy")
            
        if "http://" in target:
            threats.append({
                "severity": "medium",
                "type": "insecure_protocol",
                "description": "Insecure HTTP protocol detected"
            })
            recommendations.append("Use HTTPS instead of HTTP")
            
        # Crear recurso de audit de seguridad
        security_resource = AgentResource(
            type=ResourceType.SECURITY,
            name="security_scan_report",
            namespace="resource.security.audit",
            data={
                "target": target,
                "scan_type": scan_type,
                "threats": threats,
                "recommendations": recommendations,
                "scan_date": str(asyncio.get_event_loop().time())
            },
            owner_agent_id=self.id
        )
        
        await self.framework.resource_manager.create_resource(security_resource)
        
        return {
            "threats": threats,
            "recommendations": recommendations,
            "resource_id": security_resource.id
        }

# ================================
# MONITOR AGENTS
# ================================

class ProgressMonitorAgent(BaseAgent):
    """agent.monitor.progress - Monitor de progreso"""
    
    def __init__(self, name: str, framework):
        super().__init__("agent.monitor.progress", name, framework)
        self.tracked_tasks = {}
        
    async def initialize(self) -> bool:
        self.capabilities = [
            AgentCapability(
                name="track_progress",
                namespace="agent.monitor.progress.track",
                description="Track progress of tasks and projects",
                input_schema={"task_id": "string", "status": "string"},
                output_schema={"progress": "object"},
                handler=self._track_progress
            )
        ]
        return True
        
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "track.progress":
            return await self._track_progress(params)
        elif action == "get.progress":
            return await self._get_progress(params)
        return {"error": f"Unknown action: {action}"}
        
    async def _track_progress(self, params: Dict[str, Any]) -> Dict[str, Any]:
        task_id = params.get("task_id", "")
        status = params.get("status", "unknown")
        progress_percentage = params.get("progress", 0)
        
        if task_id not in self.tracked_tasks:
            self.tracked_tasks[task_id] = {
                "created_at": asyncio.get_event_loop().time(),
                "history": []
            }
            
        self.tracked_tasks[task_id]["current_status"] = status
        self.tracked_tasks[task_id]["progress"] = progress_percentage
        self.tracked_tasks[task_id]["last_updated"] = asyncio.get_event_loop().time()
        self.tracked_tasks[task_id]["history"].append({
            "status": status,
            "progress": progress_percentage,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        return {
            "task_id": task_id,
            "status": status,
            "progress": progress_percentage
        }
        
    async def _get_progress(self, params: Dict[str, Any]) -> Dict[str, Any]:
        task_id = params.get("task_id", "")
        
        if task_id in self.tracked_tasks:
            return {"progress": self.tracked_tasks[task_id]}
        else:
            return {"error": f"Task {task_id} not found"}

# ================================
# AGENT FACTORY EXTENDED
# ================================

class ExtendedAgentFactory:
    """Factory extendido para crear todos los tipos de agentes"""
    
    AGENT_CLASSES = {
        # Planning agents
        "agent.planning.strategist": StrategistAgent,
        "agent.planning.workflow": WorkflowAgent,
        
        # Build agents
        "agent.build.code.generator": CodeGeneratorAgent,
        "agent.build.ux.generator": UXGeneratorAgent,
        
        # Test agents
        "agent.test.generator": TestGeneratorAgent,
        
        # Security agents
        "agent.security.sentinel": SecuritySentinelAgent,
        
        # Monitor agents
        "agent.monitor.progress": ProgressMonitorAgent,
    }
    
    @classmethod
    def create_agent(cls, namespace: str, name: str, framework) -> BaseAgent:
        """Crear agente por namespace"""
        if namespace in cls.AGENT_CLASSES:
            agent_class = cls.AGENT_CLASSES[namespace]
            return agent_class(name, framework)
        else:
            raise ValueError(f"Unknown agent namespace: {namespace}")
            
    @classmethod
    async def create_full_ecosystem(cls, framework) -> Dict[str, BaseAgent]:
        """Crear ecosistema completo de agentes"""
        agents = {}
        
        # Crear uno de cada tipo
        for namespace, agent_class in cls.AGENT_CLASSES.items():
            agent_name = namespace.split('.')[-1]
            agent = agent_class(agent_name, framework)
            await agent.start()
            agents[agent_name] = agent
            
        return agents
        
    @classmethod
    def list_available_namespaces(cls) -> List[str]:
        """Listar namespaces disponibles"""
        return list(cls.AGENT_CLASSES.keys())

# ================================
# EXAMPLE USAGE WITH SPECIALIZED AGENTS
# ================================

async def advanced_example():
    """Ejemplo avanzado con agentes especializados"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Crear framework
    from core.autonomous_agent_framework import AgentFramework
    framework = AgentFramework()
    await framework.start()
    
    try:
        # Crear ecosistema completo
        agents = await ExtendedAgentFactory.create_full_ecosystem(framework)
        
        print(f"Created {len(agents)} specialized agents:")
        for name, agent in agents.items():
            print(f"  - {agent.namespace} ({name})")
            
        # Ejemplo de flujo completo
        strategist = agents["strategist"]
        workflow_agent = agents["workflow"]
        code_generator = agents["generator"]
        test_generator = agents["generator"] # test generator
        
        # 1. Crear estrategia
        strategy_result = await strategist.execute_action("define.strategy", {
            "requirements": {"goal": "build web application"},
            "constraints": {"timeline": "8 weeks", "team_size": 5}
        })
        
        print("Strategy created:", strategy_result["strategy"]["vision"])
        
        # 2. Diseñar workflow
        workflow_result = await workflow_agent.execute_action("design.workflow", {
            "tasks": [
                {"name": "Setup project", "type": "setup"},
                {"name": "Design API", "type": "design"},
                {"name": "Implement backend", "type": "development"},
                {"name": "Create frontend", "type": "development"},
                {"name": "Write tests", "type": "testing"}
            ]
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