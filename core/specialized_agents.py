"""
specialized_agents.py - Implementaciones específicas de agentes por namespace

Este módulo contiene la definición de varios agentes especializados que extienden
la funcionalidad del framework de agentes autónomos. Incluye agentes para planificación,
construcción (generación de código y UI), pruebas, seguridad y monitoreo.
También define una factoría extendida para la creación de estos agentes.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional

from core.autonomous_agent_framework import BaseAgent, AgentCapability, AgentResource, ResourceType

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ================================
# AGENTES DE PLANIFICACIÓN
# ================================

class StrategistAgent(BaseAgent):
    """
    agent.planning.strategist - Agente estratega

    Este agente es responsable de definir planes estratégicos y hojas de ruta
    basadas en requisitos y restricciones.
    """

    def __init__(self, name: str, framework):
        super().__init__("agent.planning.strategist", name, framework)

    async def initialize(self) -> bool:
        """Inicializa las capacidades del agente estratega."""
        self.capabilities = [
            AgentCapability(
                name="define_strategy",
                namespace="agent.planning.strategist.define",
                description="Define planes estratégicos y hojas de ruta.",
                input_schema={"requirements": "object", "constraints": "object"},
                output_schema={"strategy": "object", "resource_id": "string"},
                handler=self._define_strategy_handler
            )
        ]
        logging.info(f"{self.namespace}.{self.name} inicializado con éxito.")
        return True

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta una acción específica del agente estratega."""
        if action == "define.strategy":
            return await self._define_strategy_handler(params)
        logging.warning(f"Acción desconocida para StrategistAgent: {action}")
        return {"error": f"Unknown action: {action}"}

    async def _define_strategy_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maneja la definición de una estrategia.

        Args:
            params (Dict[str, Any]): Parámetros que incluyen 'requirements' y 'constraints'.

        Returns:
            Dict[str, Any]: La estrategia definida y el ID del recurso creado.
        """
        requirements = params.get("requirements", {})
        constraints = params.get("constraints", {})

        strategy = self._create_strategy_document(requirements, constraints)

        # Crear y persistir el recurso de estrategia
        strategy_resource = AgentResource(
            type=ResourceType.WORKFLOW,
            name="project_strategy",
            namespace="resource.workflow.strategy",
            data=strategy,
            owner_agent_id=self.id
        )

        await self.framework.resource_manager.create_resource(strategy_resource)
        logging.info(f"Estrategia '{strategy.get('vision')}' creada y persistida como recurso: {strategy_resource.id}")

        return {
            "strategy": strategy,
            "resource_id": strategy_resource.id
        }

    def _create_strategy_document(self, requirements: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea la estructura de datos de la estrategia.
        """
        return {
            "vision": f"Entregar {requirements.get('goal', 'solución')} de manera eficiente.",
            "objectives": [
                "Analizar estado actual",
                "Diseñar solución óptima",
                "Implementar incrementalmente",
                "Monitorear y optimizar"
            ],
            "phases": [
                {
                    "name": "Descubrimiento",
                    "duration": "2 semanas",
                    "deliverables": ["documento_requisitos", "diseño_arquitectura"]
                },
                {
                    "name": "Desarrollo",
                    "duration": "6 semanas",
                    "deliverables": ["características_principales", "pruebas", "documentación"]
                },
                {
                    "name": "Despliegue",
                    "duration": "1 semana",
                    "deliverables": ["lanzamiento_producción", "configuración_monitoreo"]
                }
            ],
            "risks": constraints.get("risks", []),
            "success_metrics": ["calidad", "rendimiento", "satisfacción_usuario"]
        }


class WorkflowAgent(BaseAgent):
    """
    agent.planning.workflow - Agente de diseño de workflows

    Este agente se encarga de diseñar y optimizar workflows basados en tareas y restricciones.
    """

    def __init__(self, name: str, framework):
        super().__init__("agent.planning.workflow", name, framework)

    async def initialize(self) -> bool:
        """Inicializa las capacidades del agente de workflow."""
        self.capabilities = [
            AgentCapability(
                name="design_workflow",
                namespace="agent.planning.workflow.design",
                description="Diseña y optimiza workflows.",
                input_schema={"tasks": "array", "constraints": "object", "name": "string"},
                output_schema={"workflow": "object"},
                handler=self._design_workflow_handler
            )
        ]
        logging.info(f"{self.namespace}.{self.name} inicializado con éxito.")
        return True

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta una acción específica del agente de workflow."""
        if action == "design.workflow":
            return await self._design_workflow_handler(params)
        logging.warning(f"Acción desconocida para WorkflowAgent: {action}")
        return {"error": f"Unknown action: {action}"}

    async def _design_workflow_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maneja el diseño de un workflow.

        Args:
            params (Dict[str, Any]): Parámetros que incluyen 'tasks', 'constraints' y 'name'.

        Returns:
            Dict[str, Any]: El workflow diseñado.
        """
        tasks = params.get("tasks", [])
        workflow_name = params.get("name", "Workflow Generado")

        workflow_steps = self._create_workflow_steps(tasks)

        workflow = {
            "id": f"workflow_{len(tasks)}_tasks",
            "name": workflow_name,
            "steps": workflow_steps,
            "parallel_branches": self._identify_parallel_branches(workflow_steps),
            "critical_path": self._calculate_critical_path(workflow_steps),
            "estimated_total_time": self._estimate_total_time(workflow_steps)
        }
        logging.info(f"Workflow '{workflow_name}' diseñado con {len(workflow_steps)} pasos.")
        return {"workflow": workflow}

    def _create_workflow_steps(self, tasks: List[Dict]) -> List[Dict]:
        """Crea la lista de pasos del workflow a partir de las tareas."""
        workflow_steps = []
        for i, task in enumerate(tasks):
            step = {
                "id": f"step_{i+1}",
                "name": task.get("name", f"Tarea {i+1}"),
                "type": task.get("type", "manual"),
                "dependencies": task.get("dependencies", []),
                "estimated_duration": task.get("duration", "1h"),
                "assigned_agent": task.get("agent", None),
                "parallel_execution": task.get("parallel", False)
            }
            workflow_steps.append(step)
        return workflow_steps

    def _identify_parallel_branches(self, steps: List[Dict]) -> List[List[str]]:
        """Identifica ramas que pueden ejecutarse en paralelo."""
        parallel_branches = []
        independent_steps = [step for step in steps if not step["dependencies"]]
        if len(independent_steps) > 1:
            parallel_branches.append([step["id"] for step in independent_steps])
        return parallel_branches

    def _calculate_critical_path(self, steps: List[Dict]) -> List[str]:
        """
        Calcula la ruta crítica del workflow (implementación simplificada).
        Una implementación más robusta requeriría un algoritmo de CPM real.
        """
        return [step["id"] for step in steps]

    def _estimate_total_time(self, steps: List[Dict]) -> str:
        """
        Estima el tiempo total del workflow (implementación simplificada).
        Asume una duración promedio por paso para esta demostración.
        """
        return f"{len(steps) * 2}h"


# ================================
# AGENTES DE CONSTRUCCIÓN
# ================================

class CodeGeneratorAgent(BaseAgent):
    """
    agent.build.code.generator - Generador de código

    Este agente genera componentes de código en diferentes lenguajes.
    """

    def __init__(self, name: str, framework):
        super().__init__("agent.build.code.generator", name, framework)

    async def initialize(self) -> bool:
        """Inicializa las capacidades del agente generador de código."""
        self.capabilities = [
            AgentCapability(
                name="generate_component",
                namespace="agent.build.code.generator.component",
                description="Genera componentes de código.",
                input_schema={"specification": "object", "language": "string"},
                output_schema={"code": "string", "resource_id": "string", "language": "string"},
                handler=self._generate_component_handler
            )
        ]
        logging.info(f"{self.namespace}.{self.name} inicializado con éxito.")
        return True

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta una acción específica del agente generador de código."""
        if action == "generate.component":
            return await self._generate_component_handler(params)
        logging.warning(f"Acción desconocida para CodeGeneratorAgent: {action}")
        return {"error": f"Unknown action: {action}"}

    async def _generate_component_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maneja la generación de un componente de código.

        Args:
            params (Dict[str, Any]): Parámetros que incluyen 'specification' y 'language'.

        Returns:
            Dict[str, Any]: El código generado y el ID del recurso.
        """
        spec = params.get("specification", {})
        language = params.get("language", "python").lower()

        code_generators = {
            "python": self._generate_python_component,
            "javascript": self._generate_javascript_component
        }

        generator_func = code_generators.get(language)
        if not generator_func:
            logging.error(f"Lenguaje no soportado para generación de código: {language}")
            return {"error": f"Unsupported language: {language}"}

        code = generator_func(spec)

        # Crear y persistir el recurso de código
        code_resource = AgentResource(
            type=ResourceType.CODE,
            name=spec.get("name", "generated_component"),
            namespace="resource.code.component",
            data={"code": code, "language": language, "specification": spec},
            owner_agent_id=self.id
        )

        await self.framework.resource_manager.create_resource(code_resource)
        logging.info(f"Componente de código '{spec.get('name', 'generated_component')}' generado en {language} y persistido.")

        return {
            "code": code,
            "resource_id": code_resource.id,
            "language": language
        }

    def _generate_python_component(self, spec: Dict[str, Any]) -> str:
        """Genera un componente Python."""
        name = spec.get("name", "Componente")
        methods = spec.get("methods", [])

        class_code_lines = [
            f'"""',
            f'{name} - Componente generado',
            f'"""',
            f'',
            f'class {name}:',
            f'    """Componente generado basado en la especificación"""',
            f'    ',
            f'    def __init__(self):',
            f'        self.initialized = True',
            f'        ',
        ]

        for method in methods:
            method_name = method.get("name", "metodo")
            params = method.get("parameters", [])
            return_type = method.get("return_type", "None")

            param_str = ", ".join([f"{p['name']}: {p.get('type', 'Any')}" for p in params])
            method_signature = f"self{', ' + param_str if param_str else ''}"

            class_code_lines.extend([
                f'    def {method_name}({method_signature}) -> {return_type}:',
                f'        """Método generado: {method.get("description", "Sin descripción")}"""',
                f'        # TODO: Implementar {method_name}',
                f'        pass',
                f'        ',
            ])
        return "\n".join(class_code_lines)

    def _generate_javascript_component(self, spec: Dict[str, Any]) -> str:
        """Genera un componente JavaScript."""
        name = spec.get("name", "Componente")
        methods = spec.get("methods", [])

        class_code_lines = [
            f'/**',
            f' * {name} - Componente generado',
            f' */',
            f'',
            f'class {name} {{',
            f'    constructor() {{',
            f'        this.initialized = true;',
            f'    }}',
            f'    ',
        ]

        for method in methods:
            method_name = method.get("name", "method")
            params = method.get("parameters", [])
            param_str = ", ".join([p["name"] for p in params])

            class_code_lines.extend([
                f'    {method_name}({param_str}) {{',
                f'        // TODO: Implementar {method_name}',
                f'        // {method.get("description", "No description")}',
                f'    }}',
                f'    ',
            ])

        class_code_lines.append("}")
        class_code_lines.append(f"module.exports = {name};")
        return "\n".join(class_code_lines)


class UXGeneratorAgent(BaseAgent):
    """
    agent.build.ux.generator - Generador de UX/UI

    Este agente genera componentes de interfaz de usuario y estilos.
    """

    def __init__(self, name: str, framework):
        super().__init__("agent.build.ux.generator", name, framework)

    async def initialize(self) -> bool:
        """Inicializa las capacidades del agente generador de UX."""
        self.capabilities = [
            AgentCapability(
                name="generate_ui",
                namespace="agent.build.ux.generator.ui",
                description="Genera componentes y layouts de UI.",
                input_schema={"design_spec": "object", "framework": "string"},
                output_schema={"ui_code": "string", "styles": "string", "resource_id": "string"},
                handler=self._generate_ui_handler
            )
        ]
        logging.info(f"{self.namespace}.{self.name} inicializado con éxito.")
        return True

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta una acción específica del agente generador de UX."""
        if action == "generate.ui":
            return await self._generate_ui_handler(params)
        logging.warning(f"Acción desconocida para UXGeneratorAgent: {action}")
        return {"error": f"Unknown action: {action}"}

    async def _generate_ui_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maneja la generación de componentes de UI.

        Args:
            params (Dict[str, Any]): Parámetros que incluyen 'design_spec' y 'framework'.

        Returns:
            Dict[str, Any]: El código UI, estilos y el ID del recurso.
        """
        design_spec = params.get("design_spec", {})
        framework = params.get("framework", "react").lower()

        if framework == "react":
            ui_code = self._generate_react_component(design_spec)
            styles = self._generate_css_styles(design_spec)
        else:
            logging.error(f"Framework no soportado para generación de UI: {framework}")
            return {"error": f"Unsupported framework: {framework}"}

        # Crear y persistir el recurso de UI
        ui_resource = AgentResource(
            type=ResourceType.UI,
            name=design_spec.get("name", "generated_ui_component"),
            namespace="resource.ui.component",
            data={"ui_code": ui_code, "styles": styles, "framework": framework},
            owner_agent_id=self.id
        )

        await self.framework.resource_manager.create_resource(ui_resource)
        logging.info(f"Componente UI '{design_spec.get('name', 'generated_ui_component')}' generado en {framework} y persistido.")

        return {
            "ui_code": ui_code,
            "styles": styles,
            "resource_id": ui_resource.id
        }

    def _generate_react_component(self, spec: Dict[str, Any]) -> str:
        """Genera un componente React."""
        name = spec.get("name", "Componente")
        props = spec.get("props", [])
        elements = spec.get("elements", [])

        props_interface_lines = ["interface Props {"]
        for prop in props:
            props_interface_lines.append(f"  {prop['name']}: {prop.get('type', 'string')};")
        props_interface_lines.append("}")
        props_interface = "\n".join(props_interface_lines)

        jsx_elements_lines = []
        for element in elements:
            element_type = element.get("type", "div")
            content = element.get("content", "")
            props_str = element.get("props", "")
            jsx_elements_lines.append(f"          <{element_type} {props_str}>{content}</{element_type}>")
        jsx_elements = "\n".join(jsx_elements_lines)

        return f'''import React from 'react';
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

    def _generate_css_styles(self, spec: Dict[str, Any]) -> str:
        """Genera estilos CSS."""
        name = spec.get("name", "componente")
        colors = spec.get("colors", {"primary": "#007bff", "secondary": "#6c757d", "background": "#ffffff"})

        return f'''.{name.lower()} {{
  display: flex;
  flex-direction: column;
  padding: 1rem;
  background-color: {colors.get("background")};
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}}

.{name.lower()} h1, .{name.lower()} h2, .{name.lower()} h3 {{
  color: {colors.get("primary")};
  margin-bottom: 0.5rem;
}}

.{name.lower()} button {{
  background-color: {colors.get("primary")};
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


# ================================
# AGENTES DE PRUEBAS
# ================================

class TestGeneratorAgent(BaseAgent):
    """
    agent.test.generator - Generador de tests

    Este agente genera tests unitarios para código en diferentes frameworks.
    """

    def __init__(self, name: str, framework):
        super().__init__("agent.test.generator", name, framework)

    async def initialize(self) -> bool:
        """Inicializa las capacidades del agente generador de tests."""
        self.capabilities = [
            AgentCapability(
                name="generate_tests",
                namespace="agent.test.generator.unit",
                description="Genera tests unitarios para código.",
                input_schema={"code": "string", "test_framework": "string"},
                output_schema={"test_code": "string", "resource_id": "string"},
                handler=self._generate_tests_handler
            )
        ]
        logging.info(f"{self.namespace}.{self.name} inicializado con éxito.")
        return True

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta una acción específica del agente generador de tests."""
        if action == "generate.tests":
            return await self._generate_tests_handler(params)
        logging.warning(f"Acción desconocida para TestGeneratorAgent: {action}")
        return {"error": f"Unknown action: {action}"}

    async def _generate_tests_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maneja la generación de tests.

        Args:
            params (Dict[str, Any]): Parámetros que incluyen 'code' y 'test_framework'.

        Returns:
            Dict[str, Any]: El código de los tests y el ID del recurso.
        """
        code = params.get("code", "")
        test_framework = params.get("test_framework", "pytest").lower()

        test_generators = {
            "pytest": self._generate_pytest_tests,
            "jest": self._generate_jest_tests
        }

        generator_func = test_generators.get(test_framework)
        if not generator_func:
            logging.error(f"Framework de test no soportado: {test_framework}")
            return {"error": f"Unsupported test framework: {test_framework}"}

        test_code = generator_func(code)

        # Crear y persistir el recurso de test
        test_resource = AgentResource(
            type=ResourceType.TEST,
            name="generated_tests",
            namespace="resource.test.unit",
            data={"test_code": test_code, "framework": test_framework},
            owner_agent_id=self.id
        )

        await self.framework.resource_manager.create_resource(test_resource)
        logging.info(f"Tests generados en {test_framework} y persistidos como recurso: {test_resource.id}")

        return {
            "test_code": test_code,
            "resource_id": test_resource.id
        }

    def _extract_class_name(self, code: str) -> str:
        """Extrae el nombre de la clase principal del código."""
        for line in code.split('\n'):
            stripped_line = line.strip()
            if stripped_line.startswith('class '):
                return stripped_line.split()[1].split('(')[0].split(':')[0]
        return "Component"

    def _extract_methods(self, code: str) -> List[str]:
        """Extrae los nombres de los métodos públicos del código."""
        methods = []
        for line in code.split('\n'):
            stripped_line = line.strip()
            if stripped_line.startswith('def ') and not stripped_line.startswith('def __'):
                method_name = stripped_line.split('(')[0].replace('def ', '')
                methods.append(method_name)
        return methods

    def _generate_pytest_tests(self, code: str) -> str:
        """Genera tests con pytest."""
        class_name = self._extract_class_name(code)
        methods = self._extract_methods(code)

        test_code_lines = [
            f'"""',
            f'Tests generados para {class_name}',
            f'"""',
            f'',
            f'import pytest',
            f'from unittest.mock import Mock, patch',
            f'from {class_name.lower()} import {class_name}',
            f'',
            f'',
            f'class Test{class_name}:',
            f'    ',
            f'    def setup_method(self):',
            f'        self.instance = {class_name}()',
            f'        ',
            f'    def test_initialization(self):',
            f'        assert self.instance is not None',
            f'        assert hasattr(self.instance, "initialized")',
            f'        ',
        ]

        for method in methods:
            test_code_lines.extend([
                f'    def test_{method}(self):',
                f'        # TODO: Implementar test para {method}',
                f'        result = self.instance.{method}()',
                f'        assert result is not None',
                f'        ',
            ])
        return "\n".join(test_code_lines)

    def _generate_jest_tests(self, code: str) -> str:
        """Genera tests con Jest."""
        class_name = self._extract_class_name(code)
        methods = self._extract_methods(code)

        test_code_lines = [
            f'/**',
            f' * Tests generados para {class_name}',
            f' */',
            f'',
            f"const {class_name} = require('./{class_name.lower()}');",
            f'',
            f"describe('{class_name}', () => {{",
            f'    let instance;',
            f'    ',
            f'    beforeEach(() => {{',
            f'        instance = new {class_name}();',
            f'    }});',
            f'    ',
            f"    test('should initialize correctly', () => {{",
            f'        expect(instance).toBeDefined();',
            f'        expect(instance.initialized).toBe(true);',
            f'    }});',
            f'    ',
        ]

        for method in methods:
            test_code_lines.extend([
                f"    test('should execute {method}', () => {{",
                f'        // TODO: Implementar test para {method}',
                f'        const result = instance.{method}();',
                f'        expect(result).toBeDefined();',
                f'    }});',
                f'    ',
            ])

        test_code_lines.append("});")
        return "\n".join(test_code_lines)


# ================================
# AGENTES DE SEGURIDAD
# ================================

class SecuritySentinelAgent(BaseAgent):
    """
    agent.security.sentinel - Agente centinela de seguridad

    Este agente se encarga de escanear vulnerabilidades de seguridad y
    proporcionar recomendaciones.
    """

    def __init__(self, name: str, framework):
        super().__init__("agent.security.sentinel", name, framework)

    async def initialize(self) -> bool:
        """Inicializa las capacidades del agente centinela de seguridad."""
        self.capabilities = [
            AgentCapability(
                name="scan_vulnerabilities",
                namespace="agent.security.sentinel.scan",
                description="Escanea vulnerabilidades de seguridad.",
                input_schema={"target": "string", "scan_type": "string"},
                output_schema={"threats": "array", "recommendations": "array", "resource_id": "string"},
                handler=self._scan_vulnerabilities_handler
            )
        ]
        logging.info(f"{self.namespace}.{self.name} inicializado con éxito.")
        return True

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta una acción específica del agente centinela de seguridad."""
        if action == "scan.vulnerabilities":
            return await self._scan_vulnerabilities_handler(params)
        logging.warning(f"Acción desconocida para SecuritySentinelAgent: {action}")
        return {"error": f"Unknown action: {action}"}

    async def _scan_vulnerabilities_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maneja el escaneo de vulnerabilidades.

        Args:
            params (Dict[str, Any]): Parámetros que incluyen 'target' y 'scan_type'.

        Returns:
            Dict[str, Any]: Las amenazas detectadas, recomendaciones y el ID del recurso.
        """
        target = params.get("target", "")
        scan_type = params.get("scan_type", "basic")

        threats, recommendations = self._simulate_security_scan(target, scan_type)

        # Crear y persistir el recurso de auditoría de seguridad
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
        logging.info(f"Reporte de seguridad para '{target}' generado y persistido como recurso: {security_resource.id}")

        return {
            "threats": threats,
            "recommendations": recommendations,
            "resource_id": security_resource.id
        }

    def _simulate_security_scan(self, target: str, scan_type: str) -> Tuple[List[Dict[str, str]], List[str]]:
        """
        Simula un escaneo de seguridad y devuelve amenazas y recomendaciones.
        """
        threats = []
        recommendations = []

        if "password" in target.lower():
            threats.append({
                "severity": "high",
                "type": "weak_authentication",
                "description": "Se detectaron requisitos de contraseña débiles."
            })
            recommendations.append("Implementar política de contraseñas robusta.")

        if "http://" in target:
            threats.append({
                "severity": "medium",
                "type": "insecure_protocol",
                "description": "Se detectó el uso de protocolo HTTP inseguro."
            })
            recommendations.append("Usar HTTPS en lugar de HTTP.")

        logging.info(f"Escaneo de seguridad simulado para '{target}' (tipo: {scan_type}). Amenazas: {len(threats)}")
        return threats, recommendations


# ================================
# AGENTES DE MONITOREO
# ================================

class ProgressMonitorAgent(BaseAgent):
    """
    agent.monitor.progress - Monitor de progreso

    Este agente rastrea el progreso de tareas y proyectos.
    """

    def __init__(self, name: str, framework):
        super().__init__("agent.monitor.progress", name, framework)
        self.tracked_tasks: Dict[str, Dict[str, Any]] = {}

    async def initialize(self) -> bool:
        """Inicializa las capacidades del agente monitor de progreso."""
        self.capabilities = [
            AgentCapability(
                name="track_progress",
                namespace="agent.monitor.progress.track",
                description="Rastrea el progreso de tareas y proyectos.",
                input_schema={"task_id": "string", "status": "string", "progress": "number"},
                output_schema={"task_id": "string", "status": "string", "progress": "number"},
                handler=self._track_progress_handler
            ),
            AgentCapability(
                name="get_progress",
                namespace="agent.monitor.progress.get",
                description="Obtiene el progreso de una tarea específica.",
                input_schema={"task_id": "string"},
                output_schema={"progress": "object"},
                handler=self._get_progress_handler
            )
        ]
        logging.info(f"{self.namespace}.{self.name} inicializado con éxito.")
        return True

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta una acción específica del agente monitor de progreso."""
        if action == "track.progress":
            return await self._track_progress_handler(params)
        elif action == "get.progress":
            return await self._get_progress_handler(params)
        logging.warning(f"Acción desconocida para ProgressMonitorAgent: {action}")
        return {"error": f"Unknown action: {action}"}

    async def _track_progress_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maneja el seguimiento del progreso de una tarea.

        Args:
            params (Dict[str, Any]): Parámetros que incluyen 'task_id', 'status' y 'progress'.

        Returns:
            Dict[str, Any]: El estado actualizado del progreso de la tarea.
        """
        task_id = params.get("task_id", "")
        status = params.get("status", "unknown")
        progress_percentage = params.get("progress", 0)

        current_time = asyncio.get_event_loop().time()

        if task_id not in self.tracked_tasks:
            self.tracked_tasks[task_id] = {
                "created_at": current_time,
                "history": []
            }

        self.tracked_tasks[task_id]["current_status"] = status
        self.tracked_tasks[task_id]["progress"] = progress_percentage
        self.tracked_tasks[task_id]["last_updated"] = current_time
        self.tracked_tasks[task_id]["history"].append({
            "status": status,
            "progress": progress_percentage,
            "timestamp": current_time
        })
        logging.info(f"Progreso de tarea '{task_id}' actualizado: Estado='{status}', Progreso={progress_percentage}%")

        return {
            "task_id": task_id,
            "status": status,
            "progress": progress_percentage
        }

    async def _get_progress_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maneja la obtención del progreso de una tarea.

        Args:
            params (Dict[str, Any]): Parámetros que incluyen 'task_id'.

        Returns:
            Dict[str, Any]: El progreso de la tarea o un mensaje de error si no se encuentra.
        """
        task_id = params.get("task_id", "")
        if task_id in self.tracked_tasks:
            logging.info(f"Obtenido progreso para tarea '{task_id}'.")
            return {"progress": self.tracked_tasks[task_id]}
        else:
            logging.warning(f"Solicitud de progreso para tarea no encontrada: {task_id}")
            return {"error": f"Task {task_id} not found"}


# ================================
# FACTORÍA DE AGENTES EXTENDIDA
# ================================

class ExtendedAgentFactory:
    """
    Factory extendido para crear todos los tipos de agentes especializados.
    """

    AGENT_CLASSES = {
        "agent.planning.strategist": StrategistAgent,
        "agent.planning.workflow": WorkflowAgent,
        "agent.build.code.generator": CodeGeneratorAgent,
        "agent.build.ux.generator": UXGeneratorAgent,
        "agent.test.generator": TestGeneratorAgent,
        "agent.security.sentinel": SecuritySentinelAgent,
        "agent.monitor.progress": ProgressMonitorAgent,
    }

    @classmethod
    def create_agent(cls, namespace: str, name: str, framework) -> BaseAgent:
        """
        Crea una instancia de agente por su namespace.

        Args:
            namespace (str): El namespace completo del agente (ej. "agent.planning.strategist").
            name (str): El nombre de la instancia del agente.
            framework: La instancia del AgentFramework.

        Returns:
            BaseAgent: Una nueva instancia del agente.

        Raises:
            ValueError: Si el namespace del agente es desconocido.
        """
        agent_class = cls.AGENT_CLASSES.get(namespace)
        if agent_class:
            logging.info(f"Creando agente: {namespace} con nombre '{name}'")
            return agent_class(name, framework)
        else:
            raise ValueError(f"Namespace de agente desconocido: {namespace}")

    @classmethod
    async def create_full_ecosystem(cls, framework) -> Dict[str, BaseAgent]:
        """
        Crea un ecosistema completo con una instancia de cada tipo de agente especializado.

        Args:
            framework: La instancia del AgentFramework.

        Returns:
            Dict[str, BaseAgent]: Un diccionario de agentes creados, donde la clave es el nombre corto del agente.
        """
        agents = {}
        logging.info("Creando ecosistema completo de agentes especializados...")
        for namespace, agent_class in cls.AGENT_CLASSES.items():
            agent_name = namespace.split('.')[-1]  # Ej. 'strategist' de 'agent.planning.strategist'
            agent = agent_class(agent_name, framework)
            await agent.start()
            agents[agent_name] = agent
            logging.info(f"Agente {namespace} ('{agent_name}') añadido al ecosistema.")
        logging.info(f"Ecosistema creado con {len(agents)} agentes.")
        return agents

    @classmethod
    def list_available_namespaces(cls) -> List[str]:
        """
        Lista todos los namespaces de agentes disponibles en la factoría.

        Returns:
            List[str]: Una lista de strings que representan los namespaces.
        """
        return list(cls.AGENT_CLASSES.keys())


# ================================
# EJEMPLO DE USO AVANZADO
# ================================

async def advanced_example():
    """
    Demostración de un flujo avanzado utilizando los agentes especializados.
    """
    logging.info("Iniciando demostración avanzada del framework de agentes...")

    from core.autonomous_agent_framework import AgentFramework
    framework = AgentFramework()
    await framework.start()

    try:
        # 1. Crear ecosistema completo de agentes
        agents = await ExtendedAgentFactory.create_full_ecosystem(framework)

        logging.info(f"Agentes especializados creados:")
        for name, agent in agents.items():
            logging.info(f"  - {agent.namespace} (ID: {agent.id})")

        # Asignar agentes a variables para un uso más fácil
        strategist_agent: StrategistAgent = agents["strategist"]
        workflow_agent: WorkflowAgent = agents["workflow"]
        code_generator_agent: CodeGeneratorAgent = agents["generator"]
        test_generator_agent: TestGeneratorAgent = agents["generator"] # Los dos pueden ser el mismo "generator" agent

        # 2. Flujo de ejemplo: de estrategia a código y tests
        logging.info("\n--- Iniciando flujo de trabajo integrado ---")

        # Paso A: Crear estrategia con el StrategistAgent
        logging.info("Paso A: Creando estrategia...")
        strategy_result = await strategist_agent.execute_action("define.strategy", {
            "requirements": {"goal": "construir aplicación web escalable"},
            "constraints": {"timeline": "10 semanas", "team_size": 7, "risks": ["integracion_api", "rendimiento_db"]}
        })
        logging.info(f"Estrategia creada: {strategy_result['strategy']['vision']}")
        logging.info(f"Recurso de estrategia ID: {strategy_result['resource_id']}")

        # Paso B: Diseñar workflow con el WorkflowAgent
        logging.info("\nPaso B: Diseñando workflow...")
        workflow_result = await workflow_agent.execute_action("design.workflow", {
            "name": "Desarrollo de Aplicación Web",
            "tasks": [
                {"name": "Configuración del Proyecto", "type": "setup"},
                {"name": "Diseño de API REST", "type": "design"},
                {"name": "Implementación de Backend", "type": "development", "dependencies": ["step_2"]},
                {"name": "Desarrollo de Frontend", "type": "development", "dependencies": ["step_2"], "parallel": True},
                {"name": "Escritura de Pruebas Unitarias", "type": "testing", "dependencies": ["step_3", "step_4"]},
                {"name": "Despliegue Inicial", "type": "deployment", "dependencies": ["step_5"]}
            ]
        })
        logging.info(f"Workflow diseñado con {len(workflow_result['workflow']['steps'])} pasos.")
        logging.info(f"Ruta crítica del workflow: {workflow_result['workflow']['critical_path']}")

        # Paso C: Generar código con el CodeGeneratorAgent
        logging.info("\nPaso C: Generando código para UserService...")
        code_specification = {
            "name": "UserService",
            "methods": [
                {"name": "create_user", "parameters": [{"name": "user_data", "type": "dict"}], "description": "Crea un nuevo usuario en el sistema."},
                {"name": "get_user", "parameters": [{"name": "user_id", "type": "str"}], "return_type": "Optional[dict]", "description": "Obtiene los detalles de un usuario por su ID."},
                {"name": "update_user", "parameters": [{"name": "user_id", "type": "str"}, {"name": "update_data", "type": "dict"}], "description": "Actualiza la información de un usuario existente."},
            ]
        }
        code_result = await code_generator_agent.execute_action("generate.component", {
            "specification": code_specification,
            "language": "python"
        })
        logging.info(f"Código generado para {code_specification['name']}.")
        logging.info(f"Recurso de código ID: {code_result['resource_id']}")
        # print("\n--- Código Generado ---\n", code_result['code'])

        # Paso D: Generar tests para el código generado con el TestGeneratorAgent
        logging.info("\nPaso D: Generando tests para el UserService...")
        test_result = await test_generator_agent.execute_action("generate.tests", {
            "code": code_result["code"],
            "test_framework": "pytest"
        })
        logging.info(f"Tests generados para {code_specification['name']} usando pytest.")
        logging.info(f"Recurso de tests ID: {test_result['resource_id']}")
        # print("\n--- Tests Generados ---\n", test_result['test_code'])

        # Paso E: Simular un escaneo de seguridad con SecuritySentinelAgent
        logging.info("\nPaso E: Ejecutando escaneo de seguridad simulado...")
        security_agent: SecuritySentinelAgent = agents["sentinel"]
        scan_result = await security_agent.execute_action("scan.vulnerabilities", {
            "target": "aplicacion.miempresa.com",
            "scan_type": "web_application"
        })
        logging.info(f"Escaneo de seguridad completado. Amenazas encontradas: {len(scan_result['threats'])}")
        for threat in scan_result['threats']:
            logging.warning(f"  - Amenaza: {threat['description']} (Severidad: {threat['severity']})")
        logging.info(f"Recomendaciones de seguridad: {', '.join(scan_result['recommendations'])}")
        logging.info(f"Recurso de seguridad ID: {scan_result['resource_id']}")

        # Paso F: Monitorear el progreso de una tarea simulada
        logging.info("\nPaso F: Monitoreando el progreso de una tarea simulada...")
        monitor_agent: ProgressMonitorAgent = agents["progress"]
        await monitor_agent.execute_action("track.progress", {"task_id": "implement_feature_X", "status": "in_progress", "progress": 50})
        await asyncio.sleep(1) # Simular tiempo
        await monitor_agent.execute_action("track.progress", {"task_id": "implement_feature_X", "status": "completed", "progress": 100})
        
        task_progress = await monitor_agent.execute_action("get.progress", {"task_id": "implement_feature_X"})
        if task_progress and "progress" in task_progress:
            logging.info(f"Progreso final de 'implement_feature_X': {task_progress['progress']['current_status']} ({task_progress['progress']['progress']}%)")
        else:
            logging.warning(f"No se pudo obtener el progreso para 'implement_feature_X'.")

        # Mostrar todos los recursos creados durante la demostración
        logging.info("\n--- Recursos Creados Durante la Demostración ---")
        all_resources = []
        for agent in agents.values():
            agent_resources = framework.resource_manager.find_resources_by_owner(agent.id)
            all_resources.extend(agent_resources)

        logging.info(f"Total de recursos creados: {len(all_resources)}")
        for resource in all_resources:
            logging.info(f"  - Tipo: {resource.type.value}, Nombre: {resource.name}, ID: {resource.id}, Propietario: {resource.owner_agent_id}")

    finally:
        # Asegurarse de detener el framework al finalizar
        logging.info("\nDeteniendo el framework de agentes...")
        await framework.stop()
        logging.info("Demostración avanzada finalizada.")


if __name__ == "__main__":
    asyncio.run(advanced_example())