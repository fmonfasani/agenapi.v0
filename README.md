# agenapi# 🤖 Autonomous Agent Framework - Guía Completa

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Características Principales](#características-principales)
3. [Arquitectura del Sistema](#arquitectura-del-sistema)
4. [Requisitos del Sistema](#requisitos-del-sistema)
5. [Instalación](#instalación)
6. [Configuración Inicial](#configuración-inicial)
7. [Guía de Inicio Rápido](#guía-de-inicio-rápido)
8. [Ejemplos de Uso](#ejemplos-de-uso)
9. [API Reference](#api-reference)
10. [Deployment](#deployment)
11. [Monitoreo y Alertas](#monitoreo-y-alertas)
12. [Seguridad](#seguridad)
13. [Backup y Recovery](#backup-y-recovery)
14. [Troubleshooting](#troubleshooting)
15. [Contribución](#contribución)

## Introducción

El **Autonomous Agent Framework** es un sistema completo para crear, gestionar y orquestar agentes autónomos interoperables. Permite construir aplicaciones complejas donde múltiples agentes especializados colaboran para completar tareas de manera inteligente y autónoma.

### ¿Qué puedes hacer con este framework?

- ✅ Crear agentes especializados (planificación, desarrollo, testing, seguridad)
- ✅ Gestionar comunicación segura entre agentes
- ✅ Monitorear y alertar sobre el estado del sistema
- ✅ Desplegar en desarrollo, staging y producción
- ✅ Backup automático y recuperación ante desastres
- ✅ API REST completa y dashboard web
- ✅ Sistema de plugins extensible
- ✅ CLI para gestión administrativa

## Características Principales

### 🏗️ Arquitectura Modular

- **Core Framework**: Sistema base de agentes
- **Specialized Agents**: Agentes pre-configurados por dominio
- **Security System**: Autenticación, autorización y auditoría
- **Persistence Layer**: Almacenamiento con múltiples backends
- **Monitoring System**: Métricas, alertas y health checks
- **Backup System**: Respaldo automático y recuperación

### 🔧 Tecnologías Integradas

- **API REST**: Interface HTTP completa
- **WebSocket**: Comunicación en tiempo real
- **Web Dashboard**: Interface gráfica de monitoreo
- **CLI Tool**: Herramienta de línea de comandos
- **Docker/Kubernetes**: Despliegue containerizado
- **SQLite/PostgreSQL**: Persistencia de datos

### 🚀 Casos de Uso

- **Desarrollo Automatizado**: Generación de código, tests y documentación
- **DevOps Inteligente**: Deployment, monitoreo y troubleshooting automático
- **Procesamiento de Datos**: ETL y análisis distribuido
- **Sistemas de Recomendación**: Agentes especializados en ML/AI
- **Automatización Empresarial**: Workflows complejos multi-agente

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                     Agent Framework Core                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Agent     │  │   Message   │  │  Resource   │             │
│  │  Registry   │  │     Bus     │  │   Manager   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│                    Specialized Agents                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Planning   │  │    Build    │  │    Test     │             │
│  │   Agents    │  │   Agents    │  │   Agents    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Security   │  │  Monitor    │  │ Integration │             │
│  │   Agents    │  │   Agents    │  │   Agents    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│                      Support Systems                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Security   │  │ Persistence │  │  Monitoring │             │
│  │   System    │  │   System    │  │   System    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Backup    │  │   Plugin    │  │ Deployment  │             │
│  │   System    │  │   System    │  │   System    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│                        Interfaces                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  REST API   │  │    Web      │  │     CLI     │             │
│  │   Server    │  │  Dashboard  │  │    Tool     │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

## Requisitos del Sistema

### Requisitos Mínimos

- **Python**: 3.11 o superior
- **RAM**: 2GB disponibles
- **Disco**: 1GB de espacio libre
- **OS**: Linux, macOS, Windows 10+

### Requisitos Recomendados

- **Python**: 3.11+ con pip y venv
- **RAM**: 4GB+ disponibles
- **Disco**: 5GB+ de espacio libre
- **CPU**: 2+ cores
- **OS**: Linux Ubuntu 20.04+ o macOS 12+

### Dependencias Externas

- **Docker** (opcional): Para deployment containerizado
- **PostgreSQL** (opcional): Para persistencia avanzada
- **Redis** (opcional): Para cache distribuido

## Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/your-org/autonomous-agent-framework.git
cd autonomous-agent-framework
```

### 2. Configurar Entorno Virtual

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Linux/macOS:
source venv/bin/activate
# En Windows:
venv\Scripts\activate
```

### 3. Instalar Dependencias

```bash
# Instalar dependencias principales
pip install -r requirements.txt

# Para desarrollo (opcional)
pip install -r requirements-dev.txt
```

### 4. Configurar Variables de Entorno

```bash
# Copiar archivo de configuración
cp .env.example .env

# Editar configuración
nano .env
```

**Contenido del archivo .env:**

```bash
# Framework Configuration
FRAMEWORK_ENV=development
LOG_LEVEL=INFO

# Security
JWT_SECRET=your-super-secure-jwt-secret-change-in-production
API_SECRET_KEY=your-api-secret-key

# Database
DATABASE_URL=sqlite:///framework.db
# Para PostgreSQL: postgresql://user:password@localhost/agent_framework

# External APIs (opcional)
GITHUB_TOKEN=your-github-token
OPENAI_API_KEY=your-openai-key
SLACK_WEBHOOK_URL=your-slack-webhook

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090

# Backup
BACKUP_ENABLED=true
BACKUP_INTERVAL_HOURS=6
```

### 5. Inicializar Base de Datos

```bash
# Crear estructura de base de datos
python scripts/init_db.py

# Verificar instalación
python scripts/verify_installation.py
```

## Configuración Inicial

### 1. Configurar CLI

```bash
# Configurar CLI
python -m framework_cli config setup

# Crear usuario administrador
python -m framework_cli auth create-admin

# Verificar configuración
python -m framework_cli system status
```

### 2. Configurar Seguridad

```bash
# Generar claves de seguridad
python scripts/generate_keys.py

# Crear usuario API
python -m framework_cli auth create-api-key --permissions admin_access
```

### 3. Configurar Monitoreo

```bash
# Verificar métricas
python -m framework_cli system metrics

# Configurar alertas
python scripts/setup_alerts.py
```

## Guía de Inicio Rápido

### 1. Ejecutar Demo End-to-End

```bash
# Ejecutar demo completo (recomendado para primera vez)
python end_to_end_example.py
```

Este demo:

- ✅ Inicializa todo el framework
- ✅ Crea agentes especializados
- ✅ Demuestra colaboración entre agentes
- ✅ Muestra monitoreo y alertas
- ✅ Inicia API y dashboard
- ✅ Genera backups
- ✅ Crea configuraciones de deployment

### 2. Inicio Manual Paso a Paso

#### 2.1 Iniciar Framework Core

```python
import asyncio
from autonomous_agent_framework import AgentFramework

async def start_framework():
    # Crear e iniciar framework
    framework = AgentFramework()
    await framework.start()

    print("✅ Framework started")
    return framework

# Ejecutar
framework = asyncio.run(start_framework())
```

#### 2.2 Crear Agentes

```python
from specialized_agents import ExtendedAgentFactory

async def create_agents(framework):
    # Crear agente estratega
    strategist = ExtendedAgentFactory.create_agent(
        "agent.planning.strategist",
        "main_strategist",
        framework
    )
    await strategist.start()

    # Crear agente generador de código
    code_gen = ExtendedAgentFactory.create_agent(
        "agent.build.code.generator",
        "code_master",
        framework
    )
    await code_gen.start()

    return {"strategist": strategist, "code_gen": code_gen}

agents = asyncio.run(create_agents(framework))
```

#### 2.3 Iniciar API y Dashboard

```bash
# Terminal 1: API Server
python -m rest_api --host 0.0.0.0 --port 8000

# Terminal 2: Dashboard
python -m web_dashboard --host 0.0.0.0 --port 8080

# Terminal 3: Monitoring
python -m monitoring_system --enable-alerts
```

### 3. Verificar Instalación

```bash
# Verificar servicios
curl http://localhost:8000/api/health
curl http://localhost:8000/api/agents

# Verificar dashboard
open http://localhost:8080

# Verificar CLI
python -m framework_cli agents list
python -m framework_cli system status
```

## Ejemplos de Uso

### Ejemplo 1: Crear Agente Simple

```python
import asyncio
from autonomous_agent_framework import AgentFramework
from specialized_agents import ExtendedAgentFactory

async def simple_agent_example():
    # Inicializar framework
    framework = AgentFramework()
    await framework.start()

    # Crear agente
    agent = ExtendedAgentFactory.create_agent(
        "agent.planning.strategist",
        "my_strategist",
        framework
    )
    await agent.start()

    # Ejecutar acción
    result = await agent.execute_action("define.strategy", {
        "requirements": {"goal": "Build mobile app"},
        "constraints": {"budget": "low", "timeline": "2 weeks"}
    })

    print("Strategy result:", result)

    # Limpiar
    await framework.stop()

asyncio.run(simple_agent_example())
```

### Ejemplo 2: Colaboración entre Agentes

```python
async def collaboration_example():
    framework = AgentFramework()
    await framework.start()

    # Crear agentes
    strategist = ExtendedAgentFactory.create_agent(
        "agent.planning.strategist", "strategist", framework
    )
    generator = ExtendedAgentFactory.create_agent(
        "agent.build.code.generator", "generator", framework
    )

    await strategist.start()
    await generator.start()

    # Estratega solicita código al generador
    message_id = await strategist.send_message(
        generator.id,
        "action.generate.component",
        {
            "specification": {
                "name": "UserService",
                "methods": [{"name": "create_user"}, {"name": "get_user"}]
            }
        }
    )

    print(f"Message sent: {message_id}")

    # Esperar procesamiento
    await asyncio.sleep(2)

    await framework.stop()

asyncio.run(collaboration_example())
```

### Ejemplo 3: Usando la API REST

```bash
# Autenticación
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin_password"}'

# Crear agente
curl -X POST http://localhost:8000/api/agents \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "agent.planning.strategist",
    "name": "api_strategist"
  }'

# Ejecutar acción
curl -X POST http://localhost:8000/api/agents/AGENT_ID/actions \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "action": "define.strategy",
    "params": {"requirements": {"goal": "API test"}}
  }'
```

### Ejemplo 4: Usando el CLI

```bash
# Configurar CLI
framework_cli config setup --api-url http://localhost:8000

# Autenticarse
framework_cli auth login --username admin

# Listar agentes
framework_cli agents list

# Crear agente
framework_cli agents create \
  --namespace agent.build.code.generator \
  --name cli_generator

# Ejecutar acción
framework_cli agents execute AGENT_ID \
  --action generate.component \
  --params '{"specification": {"name": "TestClass"}}'

# Ver estado del sistema
framework_cli system status
```

## API Reference

### Core Framework

#### AgentFramework

```python
class AgentFramework:
    async def start() -> None
    async def stop() -> None

    @property
    def registry: AgentRegistry
    @property
    def message_bus: MessageBus
    @property
    def resource_manager: ResourceManager
```

#### BaseAgent

```python
class BaseAgent:
    def __init__(namespace: str, name: str, framework: AgentFramework)

    async def initialize() -> bool
    async def start() -> None
    async def stop() -> None
    async def execute_action(action: str, params: Dict) -> Dict
    async def send_message(receiver_id: str, action: str, payload: Dict) -> str
    async def create_agent(namespace: str, name: str, agent_class: Type) -> BaseAgent
```

### Specialized Agents

#### ExtendedAgentFactory

```python
class ExtendedAgentFactory:
    @classmethod
    def create_agent(namespace: str, name: str, framework: AgentFramework) -> BaseAgent

    @classmethod
    def list_available_namespaces() -> List[str]

    @classmethod
    async def create_full_ecosystem(framework: AgentFramework) -> Dict[str, BaseAgent]
```

#### Available Namespaces

- `agent.planning.strategist` - Strategic planning
- `agent.planning.workflow` - Workflow design
- `agent.build.code.generator` - Code generation
- `agent.build.ux.generator` - UI/UX generation
- `agent.test.generator` - Test generation
- `agent.security.sentinel` - Security scanning
- `agent.monitor.progress` - Progress monitoring

### REST API Endpoints

#### Authentication

```http
POST /api/auth/login
POST /api/auth/logout
POST /api/auth/api-keys
```

#### Agents

```http
GET    /api/agents                    # List agents
GET    /api/agents/{id}               # Get agent details
POST   /api/agents                    # Create agent
POST   /api/agents/{id}/actions       # Execute action
DELETE /api/agents/{id}               # Delete agent
```

#### System

```http
GET /api/health                       # Health check
GET /api/metrics                      # System metrics
GET /api/namespaces                   # Available namespaces
```

#### Resources

```http
GET /api/resources                    # List resources
GET /api/resources/{id}               # Get resource details
```

## Deployment

### Desarrollo Local

```bash
# Desarrollo simple
python end_to_end_example.py

# Con configuración personalizada
FRAMEWORK_ENV=development python -m rest_api --port 8000
```

### Docker

```bash
# Generar archivos Docker
framework_cli deploy generate \
  --environment development \
  --strategy docker \
  --output-dir ./deployment

# Desplegar
cd deployment
./deploy.sh
```

### Kubernetes

```bash
# Generar manifiestos K8s
framework_cli deploy generate \
  --environment production \
  --strategy kubernetes \
  --domain your-domain.com \
  --output-dir ./k8s-deployment

# Desplegar
cd k8s-deployment
./deploy.sh
```

### Docker Compose (Completo)

```yaml
version: "3.8"
services:
  framework:
    build: .
    ports:
      - "8000:8000"
      - "8080:8080"
    environment:
      - FRAMEWORK_ENV=production
      - DATABASE_URL=postgresql://user:pass@postgres:5432/framework
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: framework
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  redis_data:
  grafana_data:
```

## Monitoreo y Alertas

### Configurar Alertas

```python
from monitoring_system import MonitoringOrchestrator, AlertRule, AlertSeverity

# Inicializar monitoreo
monitoring = MonitoringOrchestrator(framework)
await monitoring.start_monitoring()

# Agregar regla personalizada
rule = AlertRule(
    name="high_cpu_usage",
    metric_name="system.cpu.usage",
    condition=">",
    threshold=80.0,
    severity=AlertSeverity.WARNING,
    description="High CPU usage detected"
)
monitoring.alert_manager.add_alert_rule(rule)

# Configurar notificaciones
from monitoring_system import SlackNotificationHandler
slack_handler = SlackNotificationHandler("YOUR_SLACK_WEBHOOK")
monitoring.add_notification_handler(slack_handler)
```

### Métricas Disponibles

| Métrica                           | Descripción                   | Tipo  |
| --------------------------------- | ----------------------------- | ----- |
| `framework.agents.total`          | Total de agentes              | Gauge |
| `framework.agents.by_status`      | Agentes por estado            | Gauge |
| `framework.resources.total`       | Total de recursos             | Gauge |
| `system.cpu.usage`                | Uso de CPU                    | Gauge |
| `system.memory.usage`             | Uso de memoria                | Gauge |
| `system.disk.usage`               | Uso de disco                  | Gauge |
| `agent.heartbeat.time_since_last` | Tiempo desde último heartbeat | Gauge |

### Dashboard de Prometheus

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "agent-framework"
    static_configs:
      - targets: ["localhost:8000"]
    metrics_path: "/metrics"
```

## Seguridad

### Configuración de Seguridad

```python
from security_system import SecurityManager, Permission

# Configurar seguridad
security_config = {
    "jwt_secret": "your-secure-secret",
    "session_max_hours": 24,
    "enable_agent_authentication": True
}

security_manager = SecurityManager(security_config)

# Autenticación
auth_result = await security_manager.authenticate_user(
    AuthenticationMethod.JWT_TOKEN,
    {"username": "user", "password": "password"}
)

# Autorización
authorized = await security_manager.authorize_action(
    session_id,
    Permission.CREATE_AGENTS,
    "agents",
    SecurityLevel.INTERNAL
)
```

### Mejores Prácticas de Seguridad

1. **Cambiar Secretos**: Cambiar `JWT_SECRET` en producción
2. **HTTPS**: Usar HTTPS en producción
3. **Firewall**: Limitar acceso a puertos específicos
4. **Backups**: Encriptar backups sensibles
5. **Auditoría**: Revisar logs de auditoría regularmente
6. **API Keys**: Rotar API keys periódicamente
7. **Permisos**: Principio de menor privilegio

### Configurar HTTPS

```bash
# Generar certificado SSL
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365

# Configurar en deployment
export SSL_CERT_PATH=/path/to/cert.pem
export SSL_KEY_PATH=/path/to/key.pem
```

## Backup y Recovery

### Configurar Backup Automático

```python
from backup_recovery_system import DisasterRecoveryOrchestrator

# Inicializar sistema de backup
dr_system = DisasterRecoveryOrchestrator(framework, persistence_manager)

# Configurar backup automático
dr_system.backup_interval_hours = 6
await dr_system.start_auto_backup()

# Crear backup manual
backup = await dr_system.backup_engine.create_full_backup(
    "Manual backup before deployment"
)
```

### Estrategias de Backup

1. **Full Backup**: Backup completo cada 24 horas
2. **Incremental**: Backup incremental cada 6 horas
3. **Snapshots**: Puntos de restauración antes de cambios importantes
4. **Off-site**: Copiar backups a almacenamiento externo

### Procedimiento de Recovery

```bash
# Via CLI
framework_cli backup list
framework_cli backup restore --backup-id BACKUP_ID

# Via Python
from backup_recovery_system import RecoveryEngine

recovery = RecoveryEngine(framework, persistence_manager, backup_engine)
await recovery.restore_from_backup("backup_id")
```

### Disaster Recovery Plans

```python
# Ejecutar plan de recovery
result = await dr_system.disaster_recovery_plan("system_crash")

# Planes disponibles:
# - agent_failure: Fallo de agentes específicos
# - data_corruption: Corrupción de datos
# - system_crash: Crash del sistema
# - complete_failure: Fallo completo del sistema
```

## Troubleshooting

### Problemas Comunes

#### Framework no inicia

```bash
# Verificar dependencias
pip install -r requirements.txt

# Verificar base de datos
python scripts/verify_db.py

# Verificar configuración
python scripts/check_config.py

# Logs detallados
FRAMEWORK_ENV=development LOG_LEVEL=DEBUG python your_script.py
```

#### Agentes no responden

```bash
# Verificar estado
framework_cli agents list --status error

# Ver logs específicos
framework_cli agents show AGENT_ID

# Reiniciar agente
framework_cli agents restart AGENT_ID
```

#### API no accesible

```bash
# Verificar puerto
netstat -tlnp | grep 8000

# Verificar firewall
sudo ufw status

# Verificar logs
tail -f logs/api.log
```

#### Dashboard no carga

```bash
# Verificar servicio
curl http://localhost:8080/api/health

# Verificar WebSocket
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
     -H "Sec-WebSocket-Key: x3JJHMbDL1EzLkh9GBhXDw==" \
     -H "Sec-WebSocket-Version: 13" \
     http://localhost:8080/ws
```

### Logs y Debugging

#### Configurar Logging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
```

#### Ubicaciones de Logs

- **Framework**: `logs/framework.log`
- **API**: `logs/api.log`
- **Security**: `logs/security_audit.log`
- **Agents**: `logs/agents/`
- **System**: `logs/system.log`

#### Debug Mode

```bash
# Activar modo debug
export FRAMEWORK_DEBUG=true
export LOG_LEVEL=DEBUG

# Ejecutar con debugging
python -u your_script.py 2>&1 | tee debug_output.log
```

### Performance Issues

#### Optimización de Performance

```python
# Configurar pool de conexiones
FRAMEWORK_CONFIG = {
    "max_concurrent_agents": 50,
    "message_queue_size": 10000,
    "resource_cache_size": 1000,
    "heartbeat_interval": 60
}

# Monitorear performance
framework_cli system metrics --format json | jq '.cpu_usage'
```

#### Scaling Horizontal

```yaml
# kubernetes/deployment.yaml
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
```

## Contribución

### Setup de Desarrollo

```bash
# Clonar repo
git clone https://github.com/your-org/autonomous-agent-framework.git
cd autonomous-agent-framework

# Setup entorno de desarrollo
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Pre-commit hooks
pre-commit install

# Ejecutar tests
pytest tests/ -v

# Verificar cobertura
pytest --cov=framework tests/
```

### Estructura del Proyecto

```
autonomous-agent-framework/
├── autonomous_agent_framework.py    # Core framework
├── specialized_agents.py            # Agentes especializados
├── security_system.py              # Sistema de seguridad
├── persistence_system.py           # Sistema de persistencia
├── rest_api.py                     # API REST
├── web_dashboard.py                # Dashboard web
├── monitoring_system.py            # Sistema de monitoreo
├── backup_recovery_system.py       # Sistema de backup
├── deployment_system.py            # Sistema de deployment
├── plugin_system.py                # Sistema de plugins
├── framework_cli.py                # Herramienta CLI
├── end_to_end_example.py           # Ejemplo completo
├── tests/                          # Tests unitarios
├── docs/                           # Documentación
├── scripts/                        # Scripts de utilidad
├── deployment/                     # Configuraciones de deployment
└── requirements.txt                # Dependencias
```

### Guidelines de Contribución

1. **Fork** el repositorio
2. **Crear branch** para tu feature: `git checkout -b feature/nueva-funcionalidad`
3. **Commit** tus cambios: `git commit -am 'Add nueva funcionalidad'`
4. **Push** a tu branch: `git push origin feature/nueva-funcionalidad`
5. **Crear Pull Request**

### Coding Standards

- **PEP 8**: Seguir estándares de Python
- **Type Hints**: Usar type hints en funciones públicas
- **Docstrings**: Documentar clases y métodos importantes
- **Tests**: Escribir tests para nueva funcionalidad
- **Async/Await**: Usar patrones async cuando sea apropiado

### Testing

```bash
# Ejecutar todos los tests
pytest

# Tests específicos
pytest tests/test_framework.py

# Tests con cobertura
pytest --cov=framework --cov-report=html

# Tests de integración
pytest tests/integration/ -v

# Tests de performance
pytest tests/performance/ --benchmark-only
```

---

## 📞 Soporte

- **GitHub Issues**: [Reportar bugs](https://github.com/your-org/autonomous-agent-framework/issues)
- **Documentación**: [Wiki completa](https://github.com/your-org/autonomous-agent-framework/wiki)
- **Ejemplos**: [Repositorio de ejemplos](https://github.com/your-org/agent-framework-examples)
- **Discord**: [Comunidad de desarrolladores](https://discord.gg/agent-framework)

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🙏 Agradecimientos

- Comunidad de desarrolladores de Python
- Proyectos open source que inspiraron este framework
- Contributors y testers del framework

---

**¡Gracias por usar el Autonomous Agent Framework!** 🚀

Para comenzar rápidamente, ejecuta:

```bash
python end_to_end_example.py
```

¡Y visita `http://localhost:8080` para ver el dashboard en acción!
