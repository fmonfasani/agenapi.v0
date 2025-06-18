"""
plugin_system.py - Sistema de plugins y extensiones para el framework.
Este módulo proporciona la infraestructura para cargar, gestionar e interactuar con plugins,
permitiendo la extensión dinámica de las capacidades del framework y la integración de
funcionalidades como APIs externas y nuevos tipos de agentes.
"""

import importlib
import inspect
import os
import json
from typing import Dict, Any, List, Type, Callable, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import asyncio

# --- Mocks para ejecución independiente ---
# Estas clases se usan para permitir que `plugin_system.py` se ejecute de forma aislada
# si las clases reales de 'core' no están disponibles. En un entorno de framework completo,
# se importarían directamente de `core.autonomous_agent_framework` y `core.specialized_agents`.
try:
    from core.autonomous_agent_framework import BaseAgent, AgentCapability
    from core.specialized_agents import ExtendedAgentFactory
    _CORE_FRAMEWORK_LOADED = True
except ImportError:
    logging.warning("Core framework components (BaseAgent, AgentCapability, ExtendedAgentFactory) not found. Using mocks for standalone demonstration.")
    _CORE_FRAMEWORK_LOADED = False

    @dataclass
    class AgentCapability:
        name: str
        namespace: str
        description: str
        input_schema: Dict[str, Any]
        output_schema: Dict[str, Any]
        handler: Callable[[Dict[str, Any]], Any]
        is_async: bool = True # Assume async handlers for mocks

    class BaseAgent(ABC):
        """Mock BaseAgent para demostración independiente."""
        def __init__(self, namespace: str, name: str, framework: Any):
            self.namespace = namespace
            self.name = name
            self.framework = framework
            self.id = f"{namespace}:{name}"
            self.capabilities: List[AgentCapability] = []
            self.status: str = "INITIALIZED"
            logging.getLogger(self.namespace).info(f"Mock Agent {self.name} initialized.")

        async def start(self) -> bool:
            self.status = "RUNNING"
            logging.getLogger(self.namespace).info(f"Mock Agent {self.name} started.")
            return True

        async def stop(self) -> bool:
            self.status = "STOPPED"
            logging.getLogger(self.namespace).info(f"Mock Agent {self.name} stopped.")
            return True
        
        @abstractmethod
        async def initialize(self) -> bool:
            """Abstract method to initialize the agent's specific components."""
            pass

        async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
            """
            Executes a registered action. This mock implementation will look up
            the action in the agent's capabilities.
            """
            for cap in self.capabilities:
                if cap.name == action or cap.namespace == action:
                    logging.getLogger(self.namespace).info(f"Mock Agent {self.name} executing capability '{cap.name}'.")
                    if cap.is_async:
                        return await cap.handler(params)
                    else:
                        return cap.handler(params)
            logging.getLogger(self.namespace).warning(f"Mock Agent {self.name} received unknown action: {action}")
            return {"error": f"Unknown action: {action}"}

    class MockAgentFramework:
        """Mock Framework para simular el entorno."""
        def __init__(self):
            self.registry = self # Simplistic mock for agent listing
            self._agents = {} # Tracks agents instantiated via the mock factory
            self.resource_manager = type('ResourceManager', (object,), {
                'find_resources_by_owner': lambda self, owner_id: [],
                'list_all_resources': lambda self: []
            })() # Minimal mock resource manager
            logging.getLogger("Framework").info("Mock Framework initialized.")

        async def start(self):
            logging.getLogger("Framework").info("Mock Framework started.")

        async def stop(self):
            logging.getLogger("Framework").info("Mock Framework stopped.")
        
        def list_all_agents(self) -> List[BaseAgent]:
            return list(self._agents.values())

    class MockExtendedAgentFactory:
        """Mock Factory para agentes en plugins."""
        AGENT_CLASSES: Dict[str, Type[BaseAgent]] = {}

        @staticmethod
        def create_agent(namespace: str, name: str, framework: Any, **kwargs: Any) -> BaseAgent:
            agent_class = MockExtendedAgentFactory.AGENT_CLASSES.get(namespace)
            if not agent_class:
                raise ValueError(f"Agent class for namespace '{namespace}' not registered.")
            
            # Pass kwargs to agent constructor
            agent_instance = agent_class(name, framework, **kwargs) 
            framework._agents[name] = agent_instance # Register with mock framework
            return agent_instance

        @staticmethod
        def list_available_namespaces() -> List[str]:
            return list(MockExtendedAgentFactory.AGENT_CLASSES.keys())

    # Sobreescribir las importaciones reales con las mocks si no se cargaron
    import sys
    sys.modules['core.autonomous_agent_framework'] = type('module', (object,), {'BaseAgent': BaseAgent})
    sys.modules['core.specialized_agents'] = type('module', (object,), {'ExtendedAgentFactory': MockExtendedAgentFactory})
    
# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Interfaces y Clases Base del Sistema de Plugins ---

@dataclass
class PluginMetadata:
    """
    Metadatos para un plugin, proporcionando información esencial sobre su identidad y capacidades.
    """
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = None
    agent_types: List[str] = None
    capabilities: List[str] = None
    config_schema: Dict[str, Any] = None
    
    def __post_init__(self):
        """Inicializa listas vacías si no se proporcionan dependencias, tipos de agente o capacidades."""
        self.dependencies = self.dependencies if self.dependencies is not None else []
        self.agent_types = self.agent_types if self.agent_types is not None else []
        self.capabilities = self.capabilities if self.capabilities is not None else []
        self.config_schema = self.config_schema if self.config_schema is not None else {}

class PluginInterface(ABC):
    """
    Interfaz abstracta que todo plugin debe implementar para ser compatible con el framework.
    Define los métodos esenciales para la gestión del ciclo de vida del plugin.
    """
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """
        Retorna los metadatos del plugin.
        """
        pass
        
    @abstractmethod
    async def initialize(self, framework: Any, config: Dict[str, Any] = None) -> bool:
        """
        Inicializa el plugin, configurando recursos y estableciendo conexiones.
        
        Args:
            framework (Any): Una referencia a la instancia del framework principal.
            config (Dict[str, Any], optional): Configuración específica para este plugin.
        
        Returns:
            bool: True si la inicialización fue exitosa, False en caso contrario.
        """
        pass
        
    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Realiza la limpieza de recursos cuando el plugin es descargado o el framework se apaga.
        
        Returns:
            bool: True si la limpieza fue exitosa, False en caso contrario.
        """
        pass
        
    @abstractmethod
    def get_agent_classes(self) -> Dict[str, Type[BaseAgent]]:
        """
        Retorna un diccionario de clases de agentes que este plugin proporciona,
        mapeadas por su namespace completo (e.g., 'agent.integration.github').
        
        Returns:
            Dict[str, Type[BaseAgent]]: Diccionario de namespaces a clases de agentes.
        """
        pass
        
    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        """
        Retorna una lista de capacidades globales que este plugin añade al framework,
        adicionales a las que puedan tener sus agentes.
        
        Returns:
            List[AgentCapability]: Lista de objetos AgentCapability.
        """
        pass

class PluginManager:
    """
    Gestiona el descubrimiento, carga, inicialización y descarga de plugins
    dentro del framework.
    """
    
    def __init__(self, plugins_dir: str = "plugins"):
        self.plugins_dir = Path(plugins_dir)
        self.loaded_plugins: Dict[str, PluginInterface] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        self.framework: Optional[Any] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def set_framework(self, framework: Any):
        """
        Establece la referencia a la instancia del framework principal.
        Debe llamarse antes de cargar plugins.
        
        Args:
            framework (Any): La instancia del framework principal.
        """
        self.framework = framework
        self.logger.info("Framework reference set for PluginManager.")
        
    async def load_plugins(self, config_file: str = "plugins_config.json"):
        """
        Descubre y carga todos los plugins encontrados en el directorio configurado.
        
        Args:
            config_file (str): La ruta al archivo de configuración de plugins (JSON).
        """
        if self.framework is None:
            self.logger.error("Framework reference not set. Cannot load plugins.")
            raise RuntimeError("Framework reference must be set using set_framework() before loading plugins.")

        await self._load_plugin_configs(config_file)
        
        if not self.plugins_dir.exists():
            self.logger.warning(f"Plugins directory '{self.plugins_dir}' not found. No plugins to load.")
            return
            
        self.logger.info(f"Discovering plugins in '{self.plugins_dir}'...")
        for plugin_file in self.plugins_dir.iterdir():
            if plugin_file.suffix == ".py" and not plugin_file.name.startswith("__"):
                await self._load_plugin(plugin_file)
            
    async def _load_plugin_configs(self, config_file: str):
        """
        Carga la configuración específica para cada plugin desde un archivo JSON.
        
        Args:
            config_file (str): La ruta al archivo de configuración.
        """
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    self.plugin_configs = json.load(f)
                self.logger.info(f"Plugin configurations loaded from '{config_path}'.")
            except json.JSONDecodeError as e:
                self.logger.error(f"Error decoding plugin config JSON from '{config_path}': {e}")
            except Exception as e:
                self.logger.error(f"An unexpected error occurred loading plugin configs from '{config_path}': {e}", exc_info=True)
        else:
            self.logger.info(f"Plugin configuration file '{config_path}' not found. Proceeding without specific plugin configs.")
            
    async def _load_plugin(self, plugin_file: Path):
        """
        Carga e inicializa un plugin individual desde un archivo.
        
        Args:
            plugin_file (Path): La ruta al archivo del plugin.
        """
        module_name = plugin_file.stem
        try:
            spec = importlib.util.spec_from_file_location(module_name, plugin_file)
            if spec is None or spec.loader is None:
                self.logger.error(f"Could not create module spec for plugin file: {plugin_file}")
                return

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            plugin_found = False
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, PluginInterface) and obj is not PluginInterface:
                    plugin_instance = obj()
                    metadata = plugin_instance.get_metadata()
                    
                    if metadata.name in self.loaded_plugins:
                        self.logger.warning(f"Plugin '{metadata.name}' already loaded. Skipping '{plugin_file.name}'.")
                        continue

                    if not await self._check_dependencies(metadata.dependencies):
                        self.logger.warning(f"Plugin '{metadata.name}' (from {plugin_file.name}) has unmet dependencies: {metadata.dependencies}. Skipping.")
                        continue
                        
                    plugin_config = self.plugin_configs.get(metadata.name, {})
                    
                    self.logger.info(f"Initializing plugin '{metadata.name}' v{metadata.version} from '{plugin_file.name}'...")
                    if await plugin_instance.initialize(self.framework, plugin_config):
                        self.loaded_plugins[metadata.name] = plugin_instance
                        self.logger.info(f"Plugin '{metadata.name}' loaded successfully.")
                        await self._register_plugin_elements(plugin_instance)
                        plugin_found = True
                    else:
                        self.logger.error(f"Failed to initialize plugin '{metadata.name}' from '{plugin_file.name}'.")
            
            if not plugin_found:
                self.logger.warning(f"No valid PluginInterface implementation found in '{plugin_file.name}'.")
                    
        except Exception as e:
            self.logger.error(f"Error processing plugin file '{plugin_file}': {e}", exc_info=True)
            
    async def _check_dependencies(self, dependencies: List[str]) -> bool:
        """
        Verifica si todas las dependencias de Python listadas por el plugin están disponibles.
        
        Args:
            dependencies (List[str]): Nombres de los módulos a verificar.
        
        Returns:
            bool: True si todas las dependencias están instaladas, False en caso contrario.
        """
        for dep in dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                self.logger.warning(f"Missing dependency for plugin: '{dep}'.")
                return False
        return True
            
    async def _register_plugin_elements(self, plugin: PluginInterface):
        """
        Registra los agentes y capacidades proporcionados por el plugin en el framework.
        
        Args:
            plugin (PluginInterface): La instancia del plugin a registrar.
        """
        if not _CORE_FRAMEWORK_LOADED:
            self.logger.warning("Skipping agent and capability registration: Core framework components are not loaded (using mocks).")
            return

        # Registrar clases de agentes
        agent_classes = plugin.get_agent_classes()
        for namespace, agent_class in agent_classes.items():
            ExtendedAgentFactory.register_agent_class(namespace, agent_class)
            self.logger.info(f"Agent class '{namespace}' registered by plugin '{plugin.get_metadata().name}'.")
            
        # Registrar capacidades globales (si el framework las soporta)
        # Asumiendo que el framework tiene un método para esto, por ejemplo:
        # if hasattr(self.framework, 'register_capabilities'):
        #     self.framework.register_capabilities(plugin.get_capabilities())
        #     self.logger.info(f"Global capabilities from plugin '{plugin.get_metadata().name}' registered.")
        
    async def unload_plugin(self, plugin_name: str):
        """
        Descarga un plugin específico, llamando a su método shutdown.
        
        Args:
            plugin_name (str): El nombre del plugin a descargar.
        """
        if plugin_name in self.loaded_plugins:
            plugin = self.loaded_plugins[plugin_name]
            self.logger.info(f"Shutting down plugin '{plugin_name}'...")
            try:
                if await plugin.shutdown():
                    del self.loaded_plugins[plugin_name]
                    # Aquí se debería también desregistrar agentes y capacidades si es posible
                    self.logger.info(f"Plugin '{plugin_name}' unloaded successfully.")
                else:
                    self.logger.error(f"Plugin '{plugin_name}' shutdown failed.")
            except Exception as e:
                self.logger.error(f"Error during shutdown of plugin '{plugin_name}': {e}", exc_info=True)
        else:
            self.logger.warning(f"Attempted to unload non-existent plugin: '{plugin_name}'.")
            
    def get_loaded_plugins(self) -> Dict[str, PluginMetadata]:
        """
        Retorna un diccionario con los metadatos de todos los plugins actualmente cargados.
        
        Returns:
            Dict[str, PluginMetadata]: Mapa de nombres de plugins a sus metadatos.
        """
        return {
            name: plugin.get_metadata() 
            for name, plugin in self.loaded_plugins.items()
        }
        
    async def shutdown_all_plugins(self):
        """
        Inicia el proceso de apagado para todos los plugins cargados.
        """
        self.logger.info("Initiating shutdown for all loaded plugins.")
        # Creamos una copia de las claves para poder modificar el diccionario durante la iteración
        for plugin_name in list(self.loaded_plugins.keys()):
            await self.unload_plugin(plugin_name)
        self.logger.info("All plugins have been shut down.")


# --- Implementación de Plugin de Ejemplo: ExternalAPIPlugin ---

class ExternalAPIPlugin(PluginInterface):
    """
    Un plugin de ejemplo que demuestra la integración con APIs externas
    como GitHub, OpenAI y Slack, a través de sus agentes especializados.
    """
    
    def __init__(self):
        self.api_clients: Dict[str, Any] = {}
        self.framework: Optional[Any] = None
        self.plugin_config: Dict[str, Any] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="external_apis",
            version="1.0.0",
            description="Provides agents for integration with external APIs (GitHub, OpenAI, Slack).",
            author="Framework Team",
            dependencies=["aiohttp", "openai", "slack_sdk"], # Añadidos 'openai' y 'slack_sdk' como dependencias reales
            agent_types=["agent.integration.github", "agent.integration.openai", "agent.integration.slack"],
            capabilities=["github_operations", "ai_generation", "slack_messaging"]
        )
        
    async def initialize(self, framework: Any, config: Dict[str, Any] = None) -> bool:
        self.framework = framework
        self.plugin_config = config if config is not None else {}
        
        self.logger.info("Initializing ExternalAPIPlugin...")
        try:
            import aiohttp
            # Se usa una única sesión aiohttp para todas las integraciones HTTP.
            # Esta sesión debe ser cerrada en `shutdown`.
            self.api_clients['http_session'] = aiohttp.ClientSession()

            # Guardar tokens para que los agentes los usen si es necesario
            self.api_clients['github_token'] = self.plugin_config.get('github_token')
            self.api_clients['openai_api_key'] = self.plugin_config.get('openai_api_key')
            self.api_clients['slack_token'] = self.plugin_config.get('slack_token')
            
            self.logger.info("ExternalAPIPlugin initialized successfully.")
            return True
        except ImportError as e:
            self.logger.error(f"Missing dependency for ExternalAPIPlugin: {e}. Please install 'aiohttp', 'openai', 'slack_sdk'.")
            return False
        except Exception as e:
            self.logger.error(f"Error during ExternalAPIPlugin initialization: {e}", exc_info=True)
            return False
        
    async def shutdown(self) -> bool:
        self.logger.info("Shutting down ExternalAPIPlugin...")
        if 'http_session' in self.api_clients and not self.api_clients['http_session'].closed:
            await self.api_clients['http_session'].close()
            self.logger.info("aiohttp ClientSession closed.")
        self.api_clients.clear()
        self.logger.info("ExternalAPIPlugin shut down successfully.")
        return True
        
    def get_agent_classes(self) -> Dict[str, Type[BaseAgent]]:
        """
        Retorna las clases de agentes que este plugin define.
        Nota: Los tokens de API se pasan en la instanciación real del agente,
        no directamente aquí. El `PluginManager` facilita esto a través del `ExtendedAgentFactory`.
        """
        return {
            "agent.integration.github": GitHubAgent,
            "agent.integration.openai": OpenAIAgent,
            "agent.integration.slack": SlackAgent
        }
        
    def get_capabilities(self) -> List[AgentCapability]:
        # Las capacidades específicas se definen dentro de cada clase de agente.
        # Este método es para capacidades 'globales' si el plugin las ofreciera directamente.
        return []

# --- Agentes de Integración ---

class GitHubAgent(BaseAgent):
    """
    Agente especializado en la interacción con la API de GitHub.
    Permite crear repositorios, issues y simular pushes de código.
    """
    def __init__(self, name: str, framework: Any, github_token: Optional[str] = None):
        super().__init__("agent.integration.github", name, framework)
        self.github_token = github_token
        self.api_base = "https://api.github.com"
        self.session: Optional[Any] = None # aiohttp session
        self.logger = logging.getLogger(self.id)

    async def initialize(self) -> bool:
        # Obtener la sesión HTTP del framework o del PluginManager
        # En una integración real, el framework o el PluginManager deberían proveer esto de forma centralizada.
        # Para este ejemplo, simulamos que el framework puede dar acceso a clientes compartidos.
        try:
            if self.framework and hasattr(self.framework, 'get_shared_client'):
                 self.session = await self.framework.get_shared_client('http_session') # Asumir que el framework expone esto
            
            if self.session is None: # Fallback for mock framework or if no shared client
                import aiohttp
                self.session = aiohttp.ClientSession()
                self.logger.warning(f"Agent {self.name}: Using isolated aiohttp session. Shared session not found/provided by framework.")

            # Registrar capacidades aquí, ya que el initialize es donde el agente se configura a sí mismo
            self.capabilities = [
                AgentCapability(
                    name="create_repository",
                    namespace="agent.integration.github.repo.create",
                    description="Create a new GitHub repository.",
                    input_schema={"name": "string", "description": "string", "private": "boolean"},
                    output_schema={"repo_url": "string", "clone_url": "string", "ssh_url": "string"},
                    handler=self._create_repository
                ),
                AgentCapability(
                    name="create_issue",
                    namespace="agent.integration.github.issue.create",
                    description="Create a new GitHub issue in a specified repository.",
                    input_schema={"repo": "string", "title": "string", "body": "string", "labels": "array"},
                    output_schema={"issue_url": "string", "issue_number": "integer"},
                    handler=self._create_issue
                ),
                AgentCapability(
                    name="push_code",
                    namespace="agent.integration.github.code.push",
                    description="Simulate pushing code to a GitHub repository.",
                    input_schema={"repo": "string", "files": "object", "commit_message": "string"},
                    output_schema={"commit_sha": "string", "commit_url": "string", "files_pushed": "integer", "message": "string"},
                    handler=self._push_code
                ),
                AgentCapability(
                    name="analyze_repository",
                    namespace="agent.integration.github.repo.analyze",
                    description="Retrieve basic information about a GitHub repository.",
                    input_schema={"repo": "string"},
                    output_schema={"name": "string", "description": "string", "language": "string", "stars": "integer", "forks": "integer", "open_issues": "integer", "recent_commits": "integer", "last_updated": "string"},
                    handler=self._analyze_repository
                )
            ]
            self.logger.info(f"Agent {self.name} capabilities registered.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize GitHubAgent {self.name}: {e}", exc_info=True)
            return False
            
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Despacha la acción solicitada a su método handler correspondiente.
        """
        for cap in self.capabilities:
            if cap.name == action or cap.namespace == action:
                self.logger.info(f"Agent {self.name} executing action: {action}")
                return await cap.handler(params)
        self.logger.warning(f"Agent {self.name} received unknown action: {action}")
        return {"error": f"Unknown action: {action}"}
            
    async def _make_github_request(self, method: str, endpoint: str, **kwargs: Any) -> Dict[str, Any]:
        """Método helper para realizar peticiones a la API de GitHub."""
        if not self.github_token:
            self.logger.error("GitHub token not configured for GitHubAgent.")
            return {"error": "GitHub token not configured."}
        
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        kwargs.setdefault('headers', {}).update(headers)
        
        url = f"{self.api_base}/{endpoint}"
        
        if not self.session:
            self.logger.error(f"No aiohttp session available for agent {self.name}.")
            return {"error": "Internal HTTP client not initialized."}

        try:
            async with self.session.request(method, url, **kwargs) as response:
                response_data = await response.json()
                if 200 <= response.status < 300:
                    return {"success": True, "data": response_data}
                else:
                    self.logger.error(f"GitHub API error ({response.status}) for {endpoint}: {response_data.get('message', 'No message')}")
                    return {"error": f"GitHub API error: {response_data.get('message', 'Unknown error')}", "status_code": response.status}
        except Exception as e:
            self.logger.error(f"Request to GitHub API failed for {endpoint}: {e}", exc_info=True)
            return {"error": f"Request to GitHub API failed: {str(e)}"}

    async def _create_repository(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Crea un repositorio en GitHub."""
        repo_data = {
            "name": params.get("name"),
            "description": params.get("description", "Created by Agent Framework"),
            "private": params.get("private", False),
            "auto_init": True
        }
        if not repo_data["name"]:
            return {"error": "Repository name is required."}

        response = await self._make_github_request("POST", "user/repos", json=repo_data)
        
        if response.get("success"):
            result = response["data"]
            self.logger.info(f"Repository '{result['html_url']}' created successfully.")
            return {
                "repo_url": result["html_url"],
                "clone_url": result["clone_url"],
                "ssh_url": result["ssh_url"]
            }
        else:
            return response

    async def _create_issue(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Crea un issue en un repositorio de GitHub."""
        repo_full_name = params.get("repo", "") # Expected format: "owner/repo"
        issue_data = {
            "title": params.get("title"),
            "body": params.get("body", "Created by Agent Framework."),
            "labels": params.get("labels", ["agent-generated"])
        }
        if not repo_full_name or not issue_data["title"]:
            return {"error": "Repository name and issue title are required."}

        response = await self._make_github_request("POST", f"repos/{repo_full_name}/issues", json=issue_data)
        
        if response.get("success"):
            result = response["data"]
            self.logger.info(f"Issue #{result['number']} created in '{repo_full_name}'.")
            return {
                "issue_url": result["html_url"],
                "issue_number": result["number"]
            }
        else:
            return response

    async def _push_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simula el push de código a un repositorio (sin interacción real con Git)."""
        repo = params.get("repo", "owner/repo")
        files = params.get("files", {})
        commit_message = params.get("commit_message", "Automated commit by Agent Framework")
        
        self.logger.info(f"Simulating code push to {repo} with {len(files)} files and message: '{commit_message}'")
        # En una implementación real, se usaría una biblioteca Git o la API de contenido de GitHub
        # para crear blobs, árboles y commits. Esto es solo una simulación.
        
        # Generar un SHA de commit simulado
        import hashlib
        simulated_sha = hashlib.sha1(f"{repo}{commit_message}{len(files)}{asyncio.current_task()._loop.time()}".encode()).hexdigest()
        
        return {
            "commit_sha": simulated_sha,
            "commit_url": f"https://github.com/{repo}/commit/{simulated_sha}",
            "files_pushed": len(files),
            "message": commit_message,
            "status": "simulated_success"
        }
            
    async def _analyze_repository(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza y retorna información básica de un repositorio GitHub."""
        repo_full_name = params.get("repo", "")
        if not repo_full_name:
            return {"error": "Repository name is required for analysis."}
        
        repo_info_response = await self._make_github_request("GET", f"repos/{repo_full_name}")
        if not repo_info_response.get("success"):
            return {"error": repo_info_response.get('error', "Failed to fetch repository info.")}

        repo_data = repo_info_response["data"]
        
        # También puedes añadir más llamadas API si necesitas más detalles (e.g., commits, issues)
        commits_response = await self._make_github_request("GET", f"repos/{repo_full_name}/commits?per_page=1") # Solo 1 para contar el total si es posible
        commits_count = 0
        if commits_response.get("success") and commits_response["data"]:
            # A veces la API retorna el total de páginas en los headers, o estimar por la longitud
            # Para una simulación, asumimos un número fijo si no hay forma fácil de obtener el total
            commits_count = len(commits_response["data"]) # Or parse headers for 'Link' for total count
            
        issues_response = await self._make_github_request("GET", f"repos/{repo_full_name}/issues?state=all&per_page=1")
        issues_count = 0
        if issues_response.get("success") and issues_response["data"]:
            issues_count = len(issues_response["data"]) # Similar a commits
            
        return {
            "name": repo_data.get("name"),
            "description": repo_data.get("description"),
            "language": repo_data.get("language"),
            "stars": repo_data.get("stargazers_count"),
            "forks": repo_data.get("forks_count"),
            "open_issues": repo_data.get("open_issues_count"),
            "recent_commits": commits_count, # Simplified
            "last_updated": repo_data.get("updated_at")
        }


class OpenAIAgent(BaseAgent):
    """
    Agente para interactuar con la API de OpenAI, facilitando la generación y revisión de código.
    """
    def __init__(self, name: str, framework: Any, openai_api_key: Optional[str] = None):
        super().__init__("agent.integration.openai", name, framework)
        self.openai_api_key = openai_api_key
        self.client: Optional[Any] = None # OpenAI client instance
        self.logger = logging.getLogger(self.id)

    async def initialize(self) -> bool:
        try:
            import openai
            if not self.openai_api_key:
                self.logger.error(f"OpenAI API key not provided for agent {self.name}. Initialization failed.")
                return False
            self.client = openai.AsyncOpenAI(api_key=self.openai_api_key)
            
            self.capabilities = [
                AgentCapability(
                    name="generate_code",
                    namespace="agent.integration.openai.code.generate",
                    description="Generates code snippets based on a natural language prompt.",
                    input_schema={"prompt": "string", "language": "string", "model": "string"},
                    output_schema={"code": "string", "explanation": "string", "language": "string"},
                    handler=self._generate_code
                ),
                AgentCapability(
                    name="review_code",
                    namespace="agent.integration.openai.code.review",
                    description="Provides a code review and suggestions for improvement.",
                    input_schema={"code": "string", "model": "string"},
                    output_schema={"review": "string", "suggestions": "array", "rating": "string", "complexity": "string"},
                    handler=self._review_code
                ),
                AgentCapability(
                    name="explain_concept",
                    namespace="agent.integration.openai.concept.explain",
                    description="Explains a technical concept at a specified level of detail.",
                    input_schema={"concept": "string", "level": "string", "model": "string"},
                    output_schema={"concept": "string", "explanation": "string", "examples": "array", "further_reading": "array"},
                    handler=self._explain_concept
                )
            ]
            self.logger.info(f"Agent {self.name} capabilities registered.")
            return True
        except ImportError:
            self.logger.error("OpenAI library not installed. Please install it with 'pip install openai'.")
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAIAgent {self.name}: {e}", exc_info=True)
            return False
            
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Despacha la acción solicitada a su método handler correspondiente.
        """
        for cap in self.capabilities:
            if cap.name == action or cap.namespace == action:
                self.logger.info(f"Agent {self.name} executing action: {action}")
                return await cap.handler(params)
        self.logger.warning(f"Agent {self.name} received unknown action: {action}")
        return {"error": f"Unknown action: {action}"}
            
    async def _call_openai_api(self, messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """Método helper para realizar llamadas a la API de OpenAI."""
        if not self.client:
            return {"error": "OpenAI client not initialized."}
        
        try:
            chat_completion = await self.client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=0.7,
                max_tokens=1000
            )
            content = chat_completion.choices[0].message.content
            return {"success": True, "content": content}
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}", exc_info=True)
            return {"error": f"OpenAI API call failed: {str(e)}"}

    async def _generate_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Genera código utilizando OpenAI."""
        prompt = params.get("prompt", "a simple Python function")
        language = params.get("language", "Python")
        model = params.get("model", "gpt-3.5-turbo")

        messages = [
            {"role": "system", "content": f"You are a helpful AI assistant specialized in generating {language} code."},
            {"role": "user", "content": f"Generate {language} code for: {prompt}. Provide the code in a code block and a brief explanation."}
        ]
        
        response = await self._call_openai_api(messages, model)
        if response.get("success"):
            content = response["content"]
            # Intenta extraer el bloque de código y la explicación
            code_block_start = content.find("```")
            code_block_end = content.rfind("```")

            code = ""
            explanation = content
            if code_block_start != -1 and code_block_end != -1 and code_block_end > code_block_start:
                code_lang_line = content[code_block_start+3:code_block_end].strip().split('\n', 1)
                code = code_lang_line[1] if len(code_lang_line) > 1 else code_lang_line[0] # Handle cases where lang is not specified
                explanation_parts = []
                if code_block_start > 0:
                    explanation_parts.append(content[:code_block_start].strip())
                if code_block_end + 3 < len(content):
                    explanation_parts.append(content[code_block_end+3:].strip())
                explanation = "\n".join(explanation_parts).strip() or "No specific explanation provided by AI beyond the code."
            
            return {
                "code": code,
                "explanation": explanation,
                "language": language
            }
        else:
            return response

    async def _review_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Revisa código utilizando OpenAI."""
        code = params.get("code")
        model = params.get("model", "gpt-3.5-turbo")

        if not code:
            return {"error": "Code to review is required."}

        messages = [
            {"role": "system", "content": "You are an expert code reviewer. Provide concise feedback, identify potential issues, and suggest improvements. Rate the code (A-F) and estimate its complexity (Low, Medium, High)."},
            {"role": "user", "content": f"Please review the following code and provide suggestions:\n```\n{code}\n```\nOutput format: Review summary, then a list of suggestions, then 'Rating: X' and 'Complexity: Y'."}
        ]

        response = await self._call_openai_api(messages, model)
        if response.get("success"):
            content = response["content"]
            review_lines = content.split('\n')
            
            review_summary = []
            suggestions = []
            rating = "N/A"
            complexity = "N/A"

            in_suggestions = False
            for line in review_lines:
                line_lower = line.lower().strip()
                if line_lower.startswith("suggestions:") or line_lower.startswith("-"):
                    in_suggestions = True
                
                if in_suggestions and (line.startswith("- ") or line.startswith("* ")):
                    suggestions.append(line[2:].strip())
                elif line_lower.startswith("rating:"):
                    rating = line.split(":", 1)[1].strip()
                elif line_lower.startswith("complexity:"):
                    complexity = line.split(":", 1)[1].strip()
                elif not in_suggestions and line.strip():
                    review_summary.append(line.strip())

            return {
                "review": "\n".join(review_summary),
                "suggestions": suggestions,
                "rating": rating,
                "complexity": complexity
            }
        else:
            return response

    async def _explain_concept(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Explica un concepto técnico utilizando OpenAI."""
        concept = params.get("concept")
        level = params.get("level", "intermediate")
        model = params.get("model", "gpt-3.5-turbo")

        if not concept:
            return {"error": "Concept to explain is required."}

        messages = [
            {"role": "system", "content": f"You are a helpful AI assistant. Explain complex technical concepts clearly and concisely at an {level} level."},
            {"role": "user", "content": f"Explain '{concept}' at an {level} level. Include 2-3 short examples and 2-3 resources for further reading. Format as: Explanation, Examples (bulleted), Further Reading (bulleted)."}
        ]

        response = await self._call_openai_api(messages, model)
        if response.get("success"):
            content = response["content"]
            explanation_parts = []
            examples = []
            further_reading = []
            
            current_section = "explanation"

            for line in content.split('\n'):
                line_stripped = line.strip()
                if line_stripped.lower().startswith("examples:"):
                    current_section = "examples"
                    continue
                elif line_stripped.lower().startswith("further reading:"):
                    current_section = "further_reading"
                    continue
                
                if current_section == "explanation" and line_stripped:
                    explanation_parts.append(line_stripped)
                elif current_section == "examples" and (line_stripped.startswith("- ") or line_stripped.startswith("* ")):
                    examples.append(line_stripped[2:])
                elif current_section == "further_reading" and (line_stripped.startswith("- ") or line_stripped.startswith("* ")):
                    further_reading.append(line_stripped[2:])

            return {
                "concept": concept,
                "explanation": "\n".join(explanation_parts),
                "examples": examples,
                "further_reading": further_reading
            }
        else:
            return response


class SlackAgent(BaseAgent):
    """
    Agente para interactuar con la API de Slack, permitiendo enviar mensajes
    y crear canales.
    """
    def __init__(self, name: str, framework: Any, slack_token: Optional[str] = None):
        super().__init__("agent.integration.slack", name, framework)
        self.slack_token = slack_token
        self.client: Optional[Any] = None # Slack WebClient instance
        self.logger = logging.getLogger(self.id)

    async def initialize(self) -> bool:
        try:
            from slack_sdk import WebClient
            from slack_sdk.errors import SlackApiError
            if not self.slack_token:
                self.logger.error(f"Slack token not provided for agent {self.name}. Initialization failed.")
                return False
            self.client = WebClient(token=self.slack_token)
            
            # Test connection (optional, but good practice)
            # auth_test = await self.client.auth_test()
            # self.logger.info(f"Slack agent {self.name} connected as: {auth_test['user']}")
            
            self.capabilities = [
                AgentCapability(
                    name="send_message",
                    namespace="agent.integration.slack.message.send",
                    description="Sends a message to a specified Slack channel.",
                    input_schema={"channel": "string", "message": "string", "as_user": "boolean"},
                    output_schema={"message_id": "string", "timestamp": "string", "channel": "string", "status": "string"},
                    handler=self._send_message
                ),
                AgentCapability(
                    name="create_channel",
                    namespace="agent.integration.slack.channel.create",
                    description="Creates a new public Slack channel.",
                    input_schema={"name": "string", "purpose": "string", "is_private": "boolean"},
                    output_schema={"channel_id": "string", "channel_name": "string", "purpose": "string", "created": "boolean"},
                    handler=self._create_channel
                ),
                AgentCapability(
                    name="notify_team",
                    namespace="agent.integration.slack.team.notify",
                    description="Sends a formatted notification to a channel based on event urgency.",
                    input_schema={"channel": "string", "event_type": "string", "message": "string", "urgency": "string", "emoji": "string"},
                    output_schema={"notification_sent": "boolean", "message": "string", "recipients": "array", "urgency": "string", "channel": "string"},
                    handler=self._notify_team
                )
            ]
            self.logger.info(f"Agent {self.name} capabilities registered.")
            return True
        except ImportError:
            self.logger.error("Slack SDK not installed. Please install it with 'pip install slack_sdk'.")
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize SlackAgent {self.name}: {e}", exc_info=True)
            return False
            
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Despacha la acción solicitada a su método handler correspondiente.
        """
        for cap in self.capabilities:
            if cap.name == action or cap.namespace == action:
                self.logger.info(f"Agent {self.name} executing action: {action}")
                return await cap.handler(params)
        self.logger.warning(f"Agent {self.name} received unknown action: {action}")
        return {"error": f"Unknown action: {action}"}

    async def _send_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Envía un mensaje a un canal de Slack."""
        channel = params.get("channel")
        message = params.get("message")
        as_user = params.get("as_user", True) # Default to True for simplicity with bot tokens

        if not channel or not message:
            return {"error": "Channel and message are required."}
        if not self.client:
            return {"error": "Slack client not initialized."}

        try:
            response = await self.client.chat_postMessage(channel=channel, text=message, as_user=as_user)
            if response["ok"]:
                self.logger.info(f"Message sent to channel '{channel}': {message[:50]}...")
                return {
                    "message_id": response["ts"],
                    "timestamp": response["ts"],
                    "channel": response["channel"],
                    "status": "sent"
                }
            else:
                self.logger.error(f"Slack API error sending message: {response['error']}")
                return {"error": f"Slack API error: {response['error']}"}
        except Exception as e:
            self.logger.error(f"Failed to send Slack message: {e}", exc_info=True)
            return {"error": f"Failed to send Slack message: {str(e)}"}

    async def _create_channel(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Crea un nuevo canal en Slack."""
        name = params.get("name")
        purpose = params.get("purpose", "")
        is_private = params.get("is_private", False) # True for private, False for public

        if not name:
            return {"error": "Channel name is required."}
        if not self.client:
            return {"error": "Slack client not initialized."}

        try:
            response = await self.client.conversations_create(name=name, is_private=is_private, purpose=purpose)
            if response["ok"]:
                channel_info = response["channel"]
                self.logger.info(f"Channel '{channel_info['name']}' ({'private' if is_private else 'public'}) created.")
                return {
                    "channel_id": channel_info["id"],
                    "channel_name": channel_info["name"],
                    "purpose": channel_info.get("purpose", ""),
                    "created": True
                }
            else:
                self.logger.error(f"Slack API error creating channel: {response['error']}")
                return {"error": f"Slack API error: {response['error']}"}
        except Exception as e:
            self.logger.error(f"Failed to create Slack channel: {e}", exc_info=True)
            return {"error": f"Failed to create Slack channel: {str(e)}"}

    async def _notify_team(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Envía una notificación formateada al equipo en un canal Slack."""
        channel = params.get("channel", "#general")
        event_type = params.get("event_type", "General Update")
        message_content = params.get("message", "No message provided.")
        urgency = params.get("urgency", "normal").lower()
        emoji = params.get("emoji")

        # Customize message based on urgency
        if urgency == "high":
            prefix = emoji if emoji else "🚨 URGENT: "
        elif urgency == "medium":
            prefix = emoji if emoji else "⚠️ ATTENTION: "
        else:
            prefix = emoji if emoji else "ℹ️ INFO: "
        
        formatted_message = f"{prefix} [{event_type}] {message_content}"
        
        # Call the send_message capability
        send_result = await self._send_message({"channel": channel, "message": formatted_message})
        
        if send_result.get("status") == "sent":
            self.logger.info(f"Team notification sent to '{channel}' for event type '{event_type}'.")
            return {
                "notification_sent": True,
                "message": formatted_message,
                "recipients": ["@channel" if channel.startswith('#') else channel], # Simplified recipients
                "urgency": urgency,
                "channel": channel
            }
        else:
            self.logger.error(f"Failed to send team notification: {send_result.get('error', 'Unknown error')}")
            return {"notification_sent": False, "error": send_result.get("error", "Failed to send notification.")}


# --- Fábrica de Plugins (simplificada, ahora el Manager hace el descubrimiento) ---

class PluginFactory:
    """
    Fábrica de plugins, utilizada para obtener instancias de plugins registrados.
    En este diseño refactorizado, el PluginManager es quien descubre los plugins,
    pero esta fábrica podría usarse para instanciación bajo demanda si fuera necesario.
    """
    
    # Se mantendrá para compatibilidad o si se quiere un registro manual para ciertos plugins
    # Es mejor que el PluginManager maneje el descubrimiento y el registro.
    # Por ahora, es un mock para mostrar cómo se registrarían si no fuera por el manager
    _registered_plugins: Dict[str, Type[PluginInterface]] = {
        "external_apis": ExternalAPIPlugin,
        # Otros plugins se registrarían aquí, o el PluginManager los añadiría.
    }
    
    @classmethod
    def create_plugin(cls, plugin_name: str, framework: Any, config: Dict[str, Any] = None) -> Optional[PluginInterface]:
        """
        Crea una instancia de un plugin registrado y lo inicializa.
        
        Args:
            plugin_name (str): El nombre del plugin a crear.
            framework (Any): La instancia del framework principal.
            config (Dict[str, Any], optional): Configuración específica del plugin.
        
        Returns:
            Optional[PluginInterface]: Una instancia inicializada del plugin o None si no se encuentra/inicializa.
        """
        plugin_class = cls._registered_plugins.get(plugin_name)
        if plugin_class:
            plugin_instance = plugin_class()
            # En un caso real, la inicialización puede ser parte de PluginManager.load_plugin
            # Aquí, para la fábrica, la inicializamos directamente.
            # await plugin_instance.initialize(framework, config) # Esto necesitaría ser awaitable
            logging.getLogger(cls.__name__).info(f"Plugin '{plugin_name}' instance created via factory (needs explicit init).")
            return plugin_instance
        logging.getLogger(cls.__name__).warning(f"Plugin '{plugin_name}' not found in factory registry.")
        return None
        
    @classmethod
    def list_available_plugins(cls) -> List[str]:
        """
        Lista los nombres de los plugins disponibles en la fábrica.
        """
        return list(cls._registered_plugins.keys())

# --- Ejemplo de Uso (Demo) ---

async def plugin_system_demo():
    """
    Función de demostración para el sistema de plugins.
    Muestra cómo cargar plugins, configurar agentes de plugins y ejecutar sus acciones.
    """
    logger.info("Starting Plugin System Demo.")

    # 1. Preparar el entorno (framework)
    # Si el core framework no está cargado, usamos el mock.
    if _CORE_FRAMEWORK_LOADED:
        from core.autonomous_agent_framework import AgentFramework
        framework = AgentFramework()
        # Mock de get_shared_client para la demo si el framework real no lo tiene
        if not hasattr(framework, 'get_shared_client'):
            framework.shared_http_session = None # Placeholder for demo
            async def get_shared_client(client_name):
                if client_name == 'http_session':
                    if framework.shared_http_session is None or framework.shared_http_session.closed:
                        import aiohttp
                        framework.shared_http_session = aiohttp.ClientSession()
                    return framework.shared_http_session
                return None
            framework.get_shared_client = get_shared_client
    else:
        framework = MockAgentFramework()
        # Necesitamos que el MockFramework también pueda simular un cliente compartido para los agentes
        framework.shared_http_session = None
        async def get_shared_client(client_name):
            if client_name == 'http_session':
                if framework.shared_http_session is None or framework.shared_http_session.closed:
                    import aiohttp
                    framework.shared_http_session = aiohttp.ClientSession()
                return framework.shared_http_session
            return None
        framework.get_shared_client = get_shared_client


    await framework.start()
    
    # 2. Configurar el directorio de plugins y crear un archivo de configuración de ejemplo
    plugins_directory = Path("plugins_test")
    plugins_directory.mkdir(exist_ok=True)
    
    # Escribir un archivo de plugin de ejemplo (ExternalAPIPlugin)
    # Normalmente, estos archivos estarían preexistentes en el directorio
    external_api_plugin_content = f"""
import os
from {__name__} import PluginInterface, PluginMetadata, BaseAgent, AgentCapability, GitHubAgent, OpenAIAgent, SlackAgent
from typing import Dict, Any, List, Type, Optional

class ExternalAPIPlugin(PluginInterface):
    def __init__(self):
        self.api_clients: Dict[str, Any] = {{}}
        self.framework: Optional[Any] = None
        self.plugin_config: Dict[str, Any] = {{}}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="external_apis",
            version="1.0.0",
            description="Provides agents for integration with external APIs (GitHub, OpenAI, Slack).",
            author="Framework Team",
            dependencies=["aiohttp", "openai", "slack_sdk"],
            agent_types=["agent.integration.github", "agent.integration.openai", "agent.integration.slack"],
            capabilities=["github_operations", "ai_generation", "slack_messaging"]
        )
        
    async def initialize(self, framework: Any, config: Dict[str, Any] = None) -> bool:
        self.framework = framework
        self.plugin_config = config if config is not None else {{}}
        
        self.logger.info("Initializing ExternalAPIPlugin from file...")
        try:
            import aiohttp
            # Try to get shared session from framework, otherwise create own
            if self.framework and hasattr(self.framework, 'get_shared_client'):
                self.api_clients['http_session'] = await self.framework.get_shared_client('http_session')
            if self.api_clients.get('http_session') is None:
                self.api_clients['http_session'] = aiohttp.ClientSession()
                self.logger.warning("ExternalAPIPlugin using isolated aiohttp session for file-based plugin.")

            self.api_clients['github_token'] = self.plugin_config.get('github_token', os.getenv('GITHUB_TOKEN'))
            self.api_clients['openai_api_key'] = self.plugin_config.get('openai_api_key', os.getenv('OPENAI_API_KEY'))
            self.api_clients['slack_token'] = self.plugin_config.get('slack_token', os.getenv('SLACK_TOKEN'))
            
            self.logger.info("ExternalAPIPlugin (file) initialized successfully.")
            return True
        except ImportError as e:
            self.logger.error(f"Missing dependency for ExternalAPIPlugin (file): {{e}}. Install 'aiohttp', 'openai', 'slack_sdk'.")
            return False
        except Exception as e:
            self.logger.error(f"Error during ExternalAPIPlugin (file) initialization: {{e}}", exc_info=True)
            return False
        
    async def shutdown(self) -> bool:
        self.logger.info("Shutting down ExternalAPIPlugin (file)...")
        if 'http_session' in self.api_clients and not self.api_clients['http_session'].closed:
            await self.api_clients['http_session'].close()
            self.logger.info("aiohttp ClientSession for file-based plugin closed.")
        self.api_clients.clear()
        self.logger.info("ExternalAPIPlugin (file) shut down successfully.")
        return True
        
    def get_agent_classes(self) -> Dict[str, Type[BaseAgent]]:
        return {{
            "agent.integration.github": type("GitHubAgentWrapper", (GitHubAgent,), {{
                "__init__": lambda self, name, framework: GitHubAgent.__init__(self, name, framework, github_token=self.framework.get_shared_client('external_apis').api_clients.get('github_token'))
            }}),
            "agent.integration.openai": type("OpenAIAgentWrapper", (OpenAIAgent,), {{
                "__init__": lambda self, name, framework: OpenAIAgent.__init__(self, name, framework, openai_api_key=self.framework.get_shared_client('external_apis').api_clients.get('openai_api_key'))
            }}),
            "agent.integration.slack": type("SlackAgentWrapper", (SlackAgent,), {{
                "__init__": lambda self, name, framework: SlackAgent.__init__(self, name, framework, slack_token=self.framework.get_shared_client('external_apis').api_clients.get('slack_token'))
            }})
        }}
        
    def get_capabilities(self) -> List[AgentCapability]:
        return []
"""

    with open(plugins_directory / "external_api_plugin.py", "w") as f:
        f.write(external_api_plugin_content)
    logger.info(f"Created dummy plugin file: {plugins_directory / 'external_api_plugin.py'}")

    # Escribir un archivo de configuración de plugins de ejemplo
    plugin_config_file = plugins_directory / "plugins_config.json"
    sample_plugin_configs = {
        "external_apis": {
            "github_token": os.getenv("GITHUB_TOKEN", "YOUR_GITHUB_TOKEN"), # Consider using env vars
            "openai_api_key": os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
            "slack_token": os.getenv("SLACK_TOKEN", "YOUR_SLACK_TOKEN")
        }
    }
    with open(plugin_config_file, "w") as f:
        json.dump(sample_plugin_configs, f, indent=2)
    logger.info(f"Created sample plugin config file: {plugin_config_file}")

    # 3. Instanciar y cargar plugins
    plugin_manager = PluginManager(plugins_dir=str(plugins_directory))
    plugin_manager.set_framework(framework) # Importante: asignar el framework
    await plugin_manager.load_plugins(config_file=str(plugin_config_file))

    # 4. Verificar plugins cargados
    print("\n--- Loaded Plugins ---")
    loaded_plugins_metadata = plugin_manager.get_loaded_plugins()
    if loaded_plugins_metadata:
        for name, metadata in loaded_plugins_metadata.items():
            print(f"  - {metadata.name} (v{metadata.version}) by {metadata.author}: {metadata.description}")
            print(f"    Agent Types: {', '.join(metadata.agent_types)}")
    else:
        print("  No plugins loaded.")
    print("----------------------")

    # 5. Instanciar agentes a través del ExtendedAgentFactory (ahora que los plugins los han registrado)
    # En un sistema real, los agentes se instanciarían basándose en la configuración del framework
    # que usaría ExtendedAgentFactory para encontrarlos.
    # Para la demo, los creamos manualmente después de cargar plugins.

    # Asegurarse de que ExtendedAgentFactory tiene los agentes registrados
    if _CORE_FRAMEWORK_LOADED:
        print("\n--- Instantiating Agents from Registered Classes ---")
        try:
            github_agent = ExtendedAgentFactory.create_agent(
                "agent.integration.github", "my_github_bot", framework
            )
            openai_agent = ExtendedAgentFactory.create_agent(
                "agent.integration.openai", "my_ai_assistant", framework
            )
            slack_agent = ExtendedAgentFactory.create_agent(
                "agent.integration.slack", "my_slack_notifier", framework
            )

            await github_agent.start()
            await openai_agent.start()
            await slack_agent.start()

            print(f"✅ GitHub Agent: {github_agent.id}")
            print(f"✅ OpenAI Agent: {openai_agent.id}")
            print(f"✅ Slack Agent: {slack_agent.id}")

            # 6. Demostrar funcionalidades de los agentes
            print("\n--- Agent Functionality Demo ---")
            
            # Crear repositorio en GitHub
            repo_result = await github_agent.execute_action("create_repository", {
                "name": "agent-framework-project",
                "description": "Project created by autonomous agents demo",
                "private": True
            })
            print(f"\n📦 Repository creation result: {repo_result}")
            
            # Generar código con OpenAI
            code_result = await openai_agent.execute_action("generate_code", {
                "prompt": "Create a simple Python function to calculate Fibonacci sequence.",
                "language": "python"
            })
            print(f"\n🤖 Generated code preview: {code_result['code'][:100]}...")
            
            # Notificar en Slack
            slack_result = await slack_agent.execute_action("notify_team", {
                "channel": "#general",
                "event_type": "project_update",
                "message": "New project setup and initial code generated!",
                "urgency": "normal"
            })
            print(f"\n💬 Slack notification sent: {slack_result.get('message', 'Failed to send message')}")

            # Colaboración: OpenAI review code -> Slack notify
            review_result = await openai_agent.execute_action("review_code", {
                "code": "def hello_world():\n    print('Hello World')\n",
                "model": "gpt-3.5-turbo"
            })
            print(f"\n📝 Code review result: {review_result.get('review', 'No review')}")
            
            await slack_agent.execute_action("send_message", {
                "channel": "#development",
                "message": f"Code review completed: {review_result.get('review', 'N/A')}. Suggestions: {', '.join(review_result.get('suggestions', []))}"
            })
            print("✅ Collaboration complete: AI Review -> Slack Notification")

            # Analizar repositorio (dummy)
            analyze_result = await github_agent.execute_action("analyze_repository", {
                "repo": "octocat/Spoon-Knife" # Un repo público de GitHub para simular
            })
            print(f"\n🔍 Repository analysis (dummy): {analyze_result}")


        except Exception as e:
            logger.error(f"Error during agent interaction in demo: {e}", exc_info=True)
            print(f"❌ An error occurred during agent interaction: {e}. Ensure API keys are correct and dependencies installed.")
    else:
        print("\n--- Agent Instantiation Skipped ---")
        print("Agent instantiation and functionality demo skipped because core framework components are mocked.")
        print("To run agent demo, ensure 'core' modules are available and remove dummy classes/imports.")


    # 7. Apagar plugins y framework
    print("\n--- Shutting down ---")
    await plugin_manager.shutdown_all_plugins()
    await framework.stop()
    logger.info("Plugin System Demo Finished.")

    # Limpiar archivos de demo
    if plugins_directory.exists():
        import shutil
        shutil.rmtree(plugins_directory)
        logger.info(f"Cleaned up dummy plugin directory: {plugins_directory}")


if __name__ == "__main__":
    # Para ejecutar esta demo:
    # 1. Asegúrate de tener las librerías necesarias (aiohttp, openai, slack_sdk) instaladas:
    #    pip install aiohttp openai slack_sdk
    # 2. Si quieres usar las APIs reales, configura tus variables de entorno:
    #    export GITHUB_TOKEN="tu_token_github"
    #    export OPENAI_API_KEY="tu_clave_openai"
    #    export SLACK_TOKEN="tu_token_slack"
    # 3. Ejecuta el script: python tu_script.py
    
    # Manejo de bucle de eventos para Windows
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(plugin_system_demo())