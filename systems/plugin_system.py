"""
plugin_system.py - Sistema de plugins y extensiones para el framework
"""

import importlib
import inspect
import os
import json
from typing import Dict, Any, List, Type, Callable, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import logging
import asyncio

from core.autonomous_agent_framework import BaseAgent, AgentCapability

# ================================
# PLUGIN SYSTEM CORE
# ================================

@dataclass
class PluginMetadata:
    """Metadatos de un plugin"""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = None
    agent_types: List[str] = None
    capabilities: List[str] = None
    config_schema: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.agent_types is None:
            self.agent_types = []
        if self.capabilities is None:
            self.capabilities = []
        if self.config_schema is None:
            self.config_schema = {}

class PluginInterface(ABC):
    """Interfaz base para plugins"""
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Obtener metadatos del plugin"""
        pass
        
    @abstractmethod
    async def initialize(self, framework, config: Dict[str, Any] = None) -> bool:
        """Inicializar el plugin"""
        pass
        
    @abstractmethod
    async def shutdown(self) -> bool:
        """Limpiar recursos del plugin"""
        pass
        
    @abstractmethod
    def get_agent_classes(self) -> Dict[str, Type[BaseAgent]]:
        """Obtener clases de agentes que proporciona el plugin"""
        pass
        
    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        """Obtener capacidades adicionales"""
        pass

class PluginManager:
    """Gestor de plugins del framework"""
    
    def __init__(self, plugins_dir: str = "plugins"):
        self.plugins_dir = Path(plugins_dir)
        self.loaded_plugins: Dict[str, PluginInterface] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        self.framework = None
        
    def set_framework(self, framework):
        """Establecer referencia al framework"""
        self.framework = framework
        
    async def load_plugins(self, config_file: str = "plugins_config.json"):
        """Cargar todos los plugins disponibles"""
        
        # Cargar configuración de plugins
        await self._load_plugin_configs(config_file)
        
        # Buscar archivos de plugins
        if not self.plugins_dir.exists():
            logging.warning(f"Plugins directory {self.plugins_dir} not found")
            return
            
        for plugin_file in self.plugins_dir.glob("*.py"):
            if plugin_file.name.startswith("__"):
                continue
                
            await self._load_plugin(plugin_file)
            
    async def _load_plugin_configs(self, config_file: str):
        """Cargar configuraciones de plugins"""
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    self.plugin_configs = json.load(f)
            except Exception as e:
                logging.error(f"Error loading plugin configs: {e}")
                
    async def _load_plugin(self, plugin_file: Path):
        """Cargar un plugin específico"""
        try:
            # Importar módulo del plugin
            spec = importlib.util.spec_from_file_location(
                plugin_file.stem, plugin_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Buscar clases que implementen PluginInterface
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, PluginInterface) and 
                    obj != PluginInterface):
                    
                    plugin_instance = obj()
                    metadata = plugin_instance.get_metadata()
                    
                    # Verificar dependencias
                    if not await self._check_dependencies(metadata.dependencies):
                        logging.warning(f"Plugin {metadata.name} dependencies not met")
                        continue
                        
                    # Obtener configuración específica del plugin
                    plugin_config = self.plugin_configs.get(metadata.name, {})
                    
                    # Inicializar plugin
                    if await plugin_instance.initialize(self.framework, plugin_config):
                        self.loaded_plugins[metadata.name] = plugin_instance
                        logging.info(f"Plugin loaded: {metadata.name} v{metadata.version}")
                        
                        # Registrar agentes del plugin
                        await self._register_plugin_agents(plugin_instance)
                    else:
                        logging.error(f"Failed to initialize plugin: {metadata.name}")
                        
        except Exception as e:
            logging.error(f"Error loading plugin {plugin_file}: {e}")
            
    async def _check_dependencies(self, dependencies: List[str]) -> bool:
        """Verificar dependencias del plugin"""
        for dep in dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                return False
        return True
        
    async def _register_plugin_agents(self, plugin: PluginInterface):
        """Registrar agentes proporcionados por el plugin"""
        if not self.framework:
            return
            
        agent_classes = plugin.get_agent_classes()
        for namespace, agent_class in agent_classes.items():
            # Registrar en el factory extendido
            from core.specialized_agents import ExtendedAgentFactory
            ExtendedAgentFactory.AGENT_CLASSES[namespace] = agent_class
            logging.info(f"Registered agent class: {namespace}")
            
    async def unload_plugin(self, plugin_name: str):
        """Descargar un plugin"""
        if plugin_name in self.loaded_plugins:
            plugin = self.loaded_plugins[plugin_name]
            await plugin.shutdown()
            del self.loaded_plugins[plugin_name]
            logging.info(f"Plugin unloaded: {plugin_name}")
            
    def get_loaded_plugins(self) -> Dict[str, PluginMetadata]:
        """Obtener lista de plugins cargados"""
        return {
            name: plugin.get_metadata() 
            for name, plugin in self.loaded_plugins.items()
        }
        
    async def shutdown_all_plugins(self):
        """Cerrar todos los plugins"""
        for plugin_name in list(self.loaded_plugins.keys()):
            await self.unload_plugin(plugin_name)

# ================================
# EXTERNAL API INTEGRATIONS PLUGIN
# ================================

class ExternalAPIPlugin(PluginInterface):
    """Plugin para integraciones con APIs externas"""
    
    def __init__(self):
        self.api_clients = {}
        self.framework = None
        
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="external_apis",
            version="1.0.0",
            description="Integration with external APIs (GitHub, OpenAI, etc.)",
            author="Framework Team",
            dependencies=["requests", "aiohttp"],
            agent_types=["agent.integration.github", "agent.integration.openai", "agent.integration.slack"],
            capabilities=["github_operations", "ai_generation", "slack_messaging"]
        )
        
    async def initialize(self, framework, config: Dict[str, Any] = None) -> bool:
        self.framework = framework
        
        # Configurar clientes API
        if config:
            await self._setup_api_clients(config)
            
        return True
        
    async def shutdown(self) -> bool:
        # Cerrar conexiones API
        for client in self.api_clients.values():
            if hasattr(client, 'close'):
                await client.close()
        return True
        
    def get_agent_classes(self) -> Dict[str, Type[BaseAgent]]:
        return {
            "agent.integration.github": GitHubAgent,
            "agent.integration.openai": OpenAIAgent,
            "agent.integration.slack": SlackAgent
        }
        
    def get_capabilities(self) -> List[AgentCapability]:
        return []  # Las capacidades se definen en cada agente
        
    async def _setup_api_clients(self, config: Dict[str, Any]):
        """Configurar clientes de APIs externas"""
        import aiohttp
        
        # Cliente HTTP genérico
        self.api_clients['http'] = aiohttp.ClientSession()
        
        # Configuraciones específicas
        if 'github_token' in config:
            self.api_clients['github_token'] = config['github_token']
            
        if 'openai_api_key' in config:
            self.api_clients['openai_key'] = config['openai_api_key']
            
        if 'slack_token' in config:
            self.api_clients['slack_token'] = config['slack_token']

# ================================
# GITHUB INTEGRATION AGENT
# ================================

class GitHubAgent(BaseAgent):
    """Agente para integración con GitHub"""
    
    def __init__(self, name: str, framework, github_token: str = None):
        super().__init__("agent.integration.github", name, framework)
        self.github_token = github_token
        self.api_base = "https://api.github.com"
        
    async def initialize(self) -> bool:
        self.capabilities = [
            AgentCapability(
                name="create_repository",
                namespace="agent.integration.github.repo.create",
                description="Create GitHub repository",
                input_schema={"name": "string", "description": "string", "private": "boolean"},
                output_schema={"repo_url": "string", "clone_url": "string"},
                handler=self._create_repository
            ),
            AgentCapability(
                name="create_issue",
                namespace="agent.integration.github.issue.create",
                description="Create GitHub issue",
                input_schema={"repo": "string", "title": "string", "body": "string"},
                output_schema={"issue_url": "string", "issue_number": "integer"},
                handler=self._create_issue
            ),
            AgentCapability(
                name="push_code",
                namespace="agent.integration.github.code.push",
                description="Push code to repository",
                input_schema={"repo": "string", "files": "object", "commit_message": "string"},
                output_schema={"commit_sha": "string", "commit_url": "string"},
                handler=self._push_code
            )
        ]
        return True
        
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "create.repository":
            return await self._create_repository(params)
        elif action == "create.issue":
            return await self._create_issue(params)
        elif action == "push.code":
            return await self._push_code(params)
        elif action == "analyze.repository":
            return await self._analyze_repository(params)
        return {"error": f"Unknown action: {action}"}
        
    async def _create_repository(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Crear repositorio en GitHub"""
        import aiohttp
        
        if not self.github_token:
            return {"error": "GitHub token not configured"}
            
        repo_data = {
            "name": params.get("name", "new-repo"),
            "description": params.get("description", "Created by Agent Framework"),
            "private": params.get("private", False),
            "auto_init": True
        }
        
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base}/user/repos",
                    headers=headers,
                    json=repo_data
                ) as response:
                    if response.status == 201:
                        result = await response.json()
                        return {
                            "repo_url": result["html_url"],
                            "clone_url": result["clone_url"],
                            "ssh_url": result["ssh_url"]
                        }
                    else:
                        error_data = await response.json()
                        return {"error": f"GitHub API error: {error_data.get('message', 'Unknown error')}"}
                        
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}
            
    async def _create_issue(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Crear issue en GitHub"""
        import aiohttp
        
        repo = params.get("repo", "")  # formato: "owner/repo"
        
        issue_data = {
            "title": params.get("title", "New Issue"),
            "body": params.get("body", "Created by Agent Framework"),
            "labels": params.get("labels", ["agent-generated"])
        }
        
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base}/repos/{repo}/issues",
                    headers=headers,
                    json=issue_data
                ) as response:
                    if response.status == 201:
                        result = await response.json()
                        return {
                            "issue_url": result["html_url"],
                            "issue_number": result["number"]
                        }
                    else:
                        return {"error": f"Failed to create issue: {response.status}"}
                        
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}
            
    async def _push_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Push código a repositorio (simulado)"""
        # En una implementación real, usarías git o la API de GitHub
        repo = params.get("repo", "")
        files = params.get("files", {})
        commit_message = params.get("commit_message", "Automated commit by Agent Framework")
        
        # Simulación de push
        return {
            "commit_sha": "abc123def456",
            "commit_url": f"https://github.com/{repo}/commit/abc123def456",
            "files_pushed": len(files),
            "message": commit_message
        }
        
    async def _analyze_repository(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analizar repositorio existente"""
        import aiohttp
        
        repo = params.get("repo", "")
        
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Obtener información del repo
                async with session.get(f"{self.api_base}/repos/{repo}", headers=headers) as response:
                    if response.status == 200:
                        repo_data = await response.json()
                        
                        # Obtener commits recientes
                        async with session.get(f"{self.api_base}/repos/{repo}/commits", headers=headers) as commits_response:
                            commits_data = await commits_response.json() if commits_response.status == 200 else []
                            
                        # Obtener issues
                        async with session.get(f"{self.api_base}/repos/{repo}/issues", headers=headers) as issues_response:
                            issues_data = await issues_response.json() if issues_response.status == 200 else []
                            
                        return {
                            "name": repo_data["name"],
                            "description": repo_data["description"],
                            "language": repo_data["language"],
                            "stars": repo_data["stargazers_count"],
                            "forks": repo_data["forks_count"],
                            "open_issues": repo_data["open_issues_count"],
                            "recent_commits": len(commits_data),
                            "last_updated": repo_data["updated_at"]
                        }
                    else:
                        return {"error": f"Repository not found or access denied"}
                        
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

# ================================
# OPENAI INTEGRATION AGENT
# ================================

class OpenAIAgent(BaseAgent):
    """Agente para integración con OpenAI"""
    
    def __init__(self, name: str, framework, openai_key: str = None):
        super().__init__("agent.integration.openai", name, framework)
        self.openai_key = openai_key
        
    async def initialize(self) -> bool:
        self.capabilities = [
            AgentCapability(
                name="generate_code",
                namespace="agent.integration.openai.code.generate",
                description="Generate code using OpenAI",
                input_schema={"prompt": "string", "language": "string"},
                output_schema={"code": "string", "explanation": "string"},
                handler=self._generate_code
            ),
            AgentCapability(
                name="review_code",
                namespace="agent.integration.openai.code.review",
                description="Review code using OpenAI",
                input_schema={"code": "string"},
                output_schema={"review": "string", "suggestions": "array"},
                handler=self._review_code
            )
        ]
        return True
        
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "generate.code":
            return await self._generate_code(params)
        elif action == "review.code":
            return await self._review_code(params)
        elif action == "explain.concept":
            return await self._explain_concept(params)
        return {"error": f"Unknown action: {action}"}
        
    async def _generate_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generar código usando OpenAI"""
        # Simulación (en implementación real usarías la API de OpenAI)
        prompt = params.get("prompt", "")
        language = params.get("language", "python")
        
        # Simulación de respuesta de OpenAI
        if language == "python":
            code = f'''def generated_function():
    """
    Generated based on prompt: {prompt}
    """
    # TODO: Implement functionality
    return "Generated by OpenAI Agent"
'''
        else:
            code = f'// Generated code for: {prompt}\nfunction generatedFunction() {\n    return "Generated by OpenAI Agent";\n}'
            
        return {
            "code": code,
            "explanation": f"Generated {language} code based on the prompt: '{prompt}'",
            "language": language
        }
        
    async def _review_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Revisar código usando OpenAI"""
        code = params.get("code", "")
        
        # Simulación de review
        suggestions = [
            "Consider adding error handling",
            "Add type hints for better code clarity",
            "Consider breaking this into smaller functions",
            "Add unit tests for this functionality"
        ]
        
        review = f"Code review completed. Found {len(suggestions)} suggestions for improvement."
        
        return {
            "review": review,
            "suggestions": suggestions,
            "rating": "B+",
            "complexity": "Medium"
        }
        
    async def _explain_concept(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Explicar concepto usando OpenAI"""
        concept = params.get("concept", "")
        level = params.get("level", "intermediate")
        
        return {
            "concept": concept,
            "explanation": f"Explanation of {concept} at {level} level (generated by OpenAI Agent)",
            "examples": [f"Example 1 for {concept}", f"Example 2 for {concept}"],
            "further_reading": [f"Resource 1 about {concept}", f"Resource 2 about {concept}"]
        }

# ================================
# SLACK INTEGRATION AGENT
# ================================

class SlackAgent(BaseAgent):
    """Agente para integración con Slack"""
    
    def __init__(self, name: str, framework, slack_token: str = None):
        super().__init__("agent.integration.slack", name, framework)
        self.slack_token = slack_token
        
    async def initialize(self) -> bool:
        self.capabilities = [
            AgentCapability(
                name="send_message",
                namespace="agent.integration.slack.message.send",
                description="Send message to Slack channel",
                input_schema={"channel": "string", "message": "string"},
                output_schema={"message_id": "string", "timestamp": "string"},
                handler=self._send_message
            ),
            AgentCapability(
                name="create_channel",
                namespace="agent.integration.slack.channel.create",
                description="Create Slack channel",
                input_schema={"name": "string", "purpose": "string"},
                output_schema={"channel_id": "string", "channel_name": "string"},
                handler=self._create_channel
            )
        ]
        return True
        
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "send.message":
            return await self._send_message(params)
        elif action == "create.channel":
            return await self._create_channel(params)
        elif action == "notify.team":
            return await self._notify_team(params)
        return {"error": f"Unknown action: {action}"}
        
    async def _send_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Enviar mensaje a Slack"""
        channel = params.get("channel", "#general")
        message = params.get("message", "")
        
        # Simulación de envío de mensaje
        return {
            "message_id": "msg_123456",
            "timestamp": "1640995200.123456",
            "channel": channel,
            "status": "sent"
        }
        
    async def _create_channel(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Crear canal en Slack"""
        name = params.get("name", "")
        purpose = params.get("purpose", "")
        
        # Simulación de creación de canal
        return {
            "channel_id": "C123456789",
            "channel_name": name,
            "purpose": purpose,
            "created": True
        }
        
    async def _notify_team(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Notificar al equipo sobre eventos importantes"""
        event_type = params.get("event_type", "general")
        message = params.get("message", "")
        urgency = params.get("urgency", "normal")
        
        # Personalizar mensaje según urgencia
        if urgency == "high":
            formatted_message = f"🚨 URGENT: {message}"
        elif urgency == "medium":
            formatted_message = f"⚠️ ATTENTION: {message}"
        else:
            formatted_message = f"ℹ️ INFO: {message}"
            
        # Simular envío
        return {
            "notification_sent": True,
            "message": formatted_message,
            "recipients": ["@channel"],
            "urgency": urgency
        }

# ================================
# PLUGIN FACTORY
# ================================

class PluginFactory:
    """Factory para crear instancias de plugins"""
    
    AVAILABLE_PLUGINS = {
        "external_apis": ExternalAPIPlugin,
    }
    
    @classmethod
    def create_plugin(cls, plugin_name: str) -> Optional[PluginInterface]:
        """Crear instancia de plugin"""
        if plugin_name in cls.AVAILABLE_PLUGINS:
            return cls.AVAILABLE_PLUGINS[plugin_name]()
        return None
        
    @classmethod
    def list_available_plugins(cls) -> List[str]:
        """Listar plugins disponibles"""
        return list(cls.AVAILABLE_PLUGINS.keys())

# ================================
# EXAMPLE USAGE
# ================================

async def plugin_system_demo():
    """Demo del sistema de plugins"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Crear framework
    from autonomous_agent_framework import AgentFramework
    framework = AgentFramework()
    await framework.start()
    
    # Crear y configurar plugin manager
    plugin_manager = PluginManager()
    plugin_manager.set_framework(framework)
    
    # Crear plugin de APIs externas
    api_plugin = ExternalAPIPlugin()
    plugin_config = {
        "github_token": "your_github_token",
        "openai_api_key": "your_openai_key",
        "slack_token": "your_slack_token"
    }
    
    # Inicializar plugin
    await api_plugin.initialize(framework, plugin_config)
    
    # Crear agentes de integración
    github_agent = GitHubAgent("github_bot", framework, plugin_config.get("github_token"))
    openai_agent = OpenAIAgent("ai_assistant", framework, plugin_config.get("openai_api_key"))
    slack_agent = SlackAgent("slack_notifier", framework, plugin_config.get("slack_token"))
    
    await github_agent.start()
    await openai_agent.start() 
    await slack_agent.start()
    
    print("🔌 Plugin system demo:")
    print(f"✅ GitHub Agent: {github_agent.id}")
    print(f"✅ OpenAI Agent: {openai_agent.id}")
    print(f"✅ Slack Agent: {slack_agent.id}")
    
    # Demo de funcionalidades
    
    # 1. Crear repositorio en GitHub
    repo_result = await github_agent.execute_action("create.repository", {
        "name": "agent-generated-project",
        "description": "Project created by autonomous agents",
        "private": False
    })
    print(f"\n📦 Repository creation: {repo_result}")
    
    # 2. Generar código con OpenAI
    code_result = await openai_agent.execute_action("generate.code", {
        "prompt": "Create a REST API endpoint for user management",
        "language": "python"
    })
    print(f"\n🤖 Generated code preview: {code_result['code'][:100]}...")
    
    # 3. Notificar en Slack
    slack_result = await slack_agent.execute_action("notify.team", {
        "event_type": "deployment",
        "message": "New repository created and code generated successfully",
        "urgency": "medium"
    })
    print(f"\n💬 Slack notification: {slack_result['message']}")
    
    # 4. Demo de colaboración entre agentes
    print(f"\n🤝 Agent collaboration demo:")
    
    # OpenAI genera código, GitHub lo pushea, Slack notifica
    generated_code = await openai_agent.execute_action("generate.code", {
        "prompt": "User authentication service",
        "language": "python"
    })
    
    push_result = await github_agent.execute_action("push.code", {
        "repo": "user/agent-generated-project",
        "files": {"auth_service.py": generated_code["code"]},
        "commit_message": "Add user authentication service (generated by AI)"
    })
    
    await slack_agent.execute_action("send.message", {
        "channel": "#development",
        "message": f"New code pushed! Commit: {push_result['commit_sha']}"
    })
    
    print("✅ Collaboration complete: AI → GitHub → Slack")
    
    await framework.stop()

if __name__ == "__main__":
    asyncio.run(plugin_system_demo())