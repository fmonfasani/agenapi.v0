"""
monitoring_system.py - Sistema avanzado de monitoreo y alertas
"""

import asyncio
import json
import logging
import psutil
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import statistics
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import aiohttp

# Importaciones actualizadas
from core.autonomous_agent_framework import AgentFramework, BaseAgent
from core.models import AgentStatus # <-- CAMBIO AQUI

# ================================\
# MONITORING MODELS
# ================================\

class MetricType(Enum):
    """Tipos de métricas"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertSeverity(Enum):
    """Severidad de alertas"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"

class AlertStatus(Enum):
    """Estado de alertas"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"

@dataclass
class Metric:
    """Métrica del sistema"""
    name: str
    type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""

@dataclass
class Alert:
    """Alerta del sistema"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    entity_id: Optional[str] = None # ID del agente o componente afectado
    metric_name: Optional[str] = None # Métrica que disparó la alerta

@dataclass
class MonitoringConfig:
    """Configuración del sistema de monitoreo"""
    enable_system_metrics: bool = True
    system_metrics_interval: int = 5 # segundos
    enable_agent_heartbeat_monitor: bool = True
    agent_heartbeat_timeout: int = 60 # segundos
    alert_rules: List[Dict[str, Any]] = field(default_factory=list)
    notification_channels: Dict[str, Any] = field(default_factory=dict) # e.g., {"email": {"to": "admin@example.com"}}

# ================================\
# METRICS COLLECTOR
# ================================\

class MetricsCollector:
    """Colecta métricas del sistema y de los agentes."""
    def __init__(self, framework: AgentFramework, config: Optional[MonitoringConfig] = None):
        self.framework = framework
        self.config = config or MonitoringConfig()
        self.metrics: Dict[str, List[Metric]] = {}
        self.latest_metrics: Dict[str, Metric] = {}
        self.logger = logging.getLogger("MetricsCollector")
        self._collection_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self.logger.info("MetricsCollector initialized.")

    async def start(self):
        """Inicia la recolección de métricas."""
        if not self._collection_task:
            self._stop_event.clear()
            self._collection_task = asyncio.create_task(self._collect_metrics_loop())
            self.logger.info("Metrics collection started.")

    async def stop(self):
        """Detiene la recolección de métricas."""
        if self._collection_task:
            self._stop_event.set()
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                self.logger.info("Metrics collection task cancelled.")
            self._collection_task = None
            self.logger.info("Metrics collection stopped.")

    async def _collect_metrics_loop(self):
        """Bucle principal de recolección de métricas."""
        while not self._stop_event.is_set():
            await self.collect_system_metrics()
            await self.collect_agent_metrics()
            try:
                await asyncio.sleep(self.config.system_metrics_interval)
            except asyncio.CancelledError:
                break
        self.logger.info("Metrics collection loop stopped.")

    async def collect_system_metrics(self):
        """Colecta métricas de uso del sistema (CPU, memoria, disco)."""
        if not self.config.enable_system_metrics:
            return

        timestamp = datetime.now()
        self.add_metric(Metric("system.cpu.usage", MetricType.GAUGE, psutil.cpu_percent(interval=None), timestamp, unit="%"))
        self.add_metric(Metric("system.memory.usage", MetricType.GAUGE, psutil.virtual_memory().percent, timestamp, unit="%"))
        self.add_metric(Metric("system.disk.usage", MetricType.GAUGE, psutil.disk_usage('/').percent, timestamp, unit="%"))
        self.add_metric(Metric("system.network.bytes_sent", MetricType.COUNTER, psutil.net_io_counters().bytes_sent, timestamp, unit="bytes"))
        self.add_metric(Metric("system.network.bytes_recv", MetricType.COUNTER, psutil.net_io_counters().bytes_recv, timestamp, unit="bytes"))
        self.add_metric(Metric("system.process_count", MetricType.GAUGE, len(psutil.pids()), timestamp, unit="count"))
        self.logger.debug("System metrics collected.")

    async def collect_agent_metrics(self):
        """Colecta métricas de los agentes (estado, número de mensajes en cola)."""
        timestamp = datetime.now()
        agents = self.framework.registry.list_all_agents()
        self.add_metric(Metric("agent.total_agents", MetricType.GAUGE, len(agents), timestamp))

        status_counts = self.framework.registry.count_agents_by_status()
        for status, count in status_counts.items():
            self.add_metric(Metric(f"agent.status.{status.value}", MetricType.GAUGE, count, timestamp, tags={"status": status.value}))

        for agent in agents:
            # Asumiendo que el agente tiene una queue con un .qsize()
            try:
                message_queue_size = agent.message_queue.qsize()
                self.add_metric(Metric(f"agent.queue_size", MetricType.GAUGE, message_queue_size, timestamp, tags={"agent_id": agent.id, "agent_name": agent.name}))
            except Exception:
                self.logger.warning(f"Could not get queue size for agent {agent.name}")
            
            # Recopilar la última marca de tiempo de actividad del agente
            # Esto es más útil para monitorear el pulso de los agentes
            self.add_metric(Metric(f"agent.last_heartbeat", MetricType.GAUGE, agent.last_heartbeat.timestamp(), timestamp, tags={"agent_id": agent.id, "agent_name": agent.name}))


        self.logger.debug("Agent metrics collected.")

    def add_metric(self, metric: Metric):
        """Añade una métrica a la colección."""
        if metric.name not in self.metrics:
            self.metrics[metric.name] = []
        self.metrics[metric.name].append(metric)
        self.latest_metrics[metric.name] = metric # Mantener solo la última métrica
        self.logger.debug(f"Added metric: {metric.name}={metric.value}")

    def get_metrics(self, name: Optional[str] = None, limit: int = 100) -> Union[List[Metric], Dict[str, List[Metric]]]:
        """Obtiene métricas por nombre o todas las métricas."""
        if name:
            return self.metrics.get(name, [])[-limit:]
        return {k: v[-limit:] for k, v in self.metrics.items()}

    def get_latest_metrics(self) -> Dict[str, Metric]:
        """Obtiene la última métrica para cada nombre de métrica."""
        return self.latest_metrics

    def calculate_metric_statistics(self, metric_name: str) -> Optional[Dict[str, float]]:
        """Calcula estadísticas para una métrica específica."""
        if metric_name not in self.metrics:
            return None
        
        values = [m.value for m in self.metrics[metric_name] if m.value is not None]
        if not values:
            return None
        
        stats = {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0
        }
        return stats
    
    def export_metrics(self) -> Dict[str, Any]:
        """Exporta todas las métricas recolectadas."""
        exported_data = {}
        for name, metrics_list in self.metrics.items():
            exported_data[name] = [asdict(m) for m in metrics_list]
        return exported_data

# ================================\
# HEALTH CHECKER
# ================================\

class HealthChecker:
    """Realiza comprobaciones de salud del sistema y de los agentes."""
    def __init__(self, framework: AgentFramework, config: Optional[MonitoringConfig] = None):
        self.framework = framework
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger("HealthChecker")
        self.unresponsive_agents: Dict[str, datetime] = {} # agent_id -> last_seen_ok
        self._health_check_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self.logger.info("HealthChecker initialized.")

    async def start(self):
        """Inicia el chequeo de salud."""
        if not self._health_check_task:
            self._stop_event.clear()
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self.logger.info("Health checks started.")

    async def stop(self):
        """Detiene el chequeo de salud."""
        if self._health_check_task:
            self._stop_event.set()
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                self.logger.info("Health check task cancelled.")
            self._health_check_task = None
            self.logger.info("HealthChecker stopped.")

    async def _health_check_loop(self):
        """Bucle principal de comprobación de salud."""
        while not self._stop_event.is_set():
            await self.check_framework_health()
            await self.check_agent_health()
            try:
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
        self.logger.info("Health check loop stopped.")

    async def check_framework_health(self) -> Dict[str, Any]:
        """Comprueba la salud de los componentes principales del framework."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "components": {
                "message_bus": "healthy", # Asumimos saludable si no hay errores reportados
                "resource_manager": "healthy",
                "agent_registry": "healthy"
            },
            "metrics": {
                "cpu_usage_percent": psutil.cpu_percent(interval=None),
                "memory_usage_percent": psutil.virtual_memory().percent
            }
        }
        # Podríamos añadir cheques más detallados, por ejemplo, si el message_bus tiene mensajes estancados
        return status

    async def check_agent_health(self):
        """Comprueba la salud de los agentes registrados."""
        if not self.config.enable_agent_heartbeat_monitor:
            return

        agents = self.framework.registry.list_all_agents()
        for agent in agents:
            if agent.status == AgentStatus.TERMINATED or agent.status == AgentStatus.ERROR:
                if agent.id in self.unresponsive_agents:
                    del self.unresponsive_agents[agent.id] # Remover si ya está en estado final
                continue
            
            time_since_heartbeat = (datetime.now() - agent.last_heartbeat).total_seconds()
            
            if time_since_heartbeat > self.config.agent_heartbeat_timeout:
                if agent.status != AgentStatus.ERROR:
                    self.logger.warning(f"Agent {agent.name} (ID: {agent.id}) is unresponsive ({time_since_heartbeat:.0f}s without heartbeat). Setting status to ERROR.")
                    self.framework.registry.update_agent_status(agent.id, AgentStatus.ERROR)
                    self.unresponsive_agents[agent.id] = datetime.now() # Marcar cuando se detectó la inactividad
            else:
                if agent.id in self.unresponsive_agents:
                    self.logger.info(f"Agent {agent.name} (ID: {agent.id}) is now responsive again.")
                    del self.unresponsive_agents[agent.id] # Volvió a la normalidad
                if agent.status == AgentStatus.ERROR:
                    # Si el agente se recuperó por sí mismo o fue reiniciado
                    self.logger.info(f"Agent {agent.name} (ID: {agent.id}) recovered from ERROR state. Setting status to ACTIVE.")
                    self.framework.registry.update_agent_status(agent.id, AgentStatus.ACTIVE)

        self.logger.debug("Agent health checked.")

    def get_health_status(self) -> Dict[str, Any]:
        """Obtiene el estado de salud consolidado."""
        framework_health = asyncio.run(self.check_framework_health()) # Sincronizar para la llamada
        
        overall_status = "healthy"
        if any(status == AgentStatus.ERROR for agent in self.framework.registry.list_all_agents()):
            overall_status = "degraded"
        if len(self.unresponsive_agents) > 0:
             overall_status = "critical" # Si hay agentes unresponsive, es más crítico

        return {
            "overall_status": overall_status,
            "framework_components": framework_health["components"],
            "system_metrics_snapshot": framework_health["metrics"],
            "agent_summary": {
                "total_agents": len(self.framework.registry.list_all_agents()),
                "active_agents": len([a for a in self.framework.registry.list_all_agents() if a.status == AgentStatus.ACTIVE]),
                "error_agents": len([a for a in self.framework.registry.list_all_agents() if a.status == AgentStatus.ERROR]),
                "unresponsive_agents_count": len(self.unresponsive_agents),
                "unresponsive_details": {aid: dt.isoformat() for aid, dt in self.unresponsive_agents.items()}
            },
            "timestamp": datetime.now().isoformat()
        }

# ================================\
# ALERT MANAGER
# ================================\

class AlertManager:
    """Gestiona la creación, resolución y notificación de alertas."""
    def __init__(self, metrics_collector: MetricsCollector, health_checker: HealthChecker, config: Optional[MonitoringConfig] = None):
        self.metrics_collector = metrics_collector
        self.health_checker = health_checker
        self.config = config or MonitoringConfig()
        self.active_alerts: Dict[str, Alert] = {} # rule_name -> Alert
        self.alert_history: List[Alert] = []
        self.logger = logging.getLogger("AlertManager")
        self._alert_check_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self.logger.info("AlertManager initialized.")

    async def start(self):
        """Inicia el gestor de alertas."""
        if not self._alert_check_task:
            self._stop_event.clear()
            self._alert_check_task = asyncio.create_task(self._check_alerts_loop())
            self.logger.info("Alert manager started.")

    async def stop(self):
        """Detiene el gestor de alertas."""
        if self._alert_check_task:
            self._stop_event.set()
            self._alert_check_task.cancel()
            try:
                await self._alert_check_task
            except asyncio.CancelledError:
                self.logger.info("Alert manager task cancelled.")
            self._alert_check_task = None
            self.logger.info("AlertManager stopped.")

    async def _check_alerts_loop(self):
        """Bucle principal para evaluar reglas de alerta."""
        while not self._stop_event.is_set():
            await self.evaluate_alert_rules()
            try:
                await asyncio.sleep(self.config.system_metrics_interval) # Usar mismo intervalo que métricas
            except asyncio.CancelledError:
                break
        self.logger.info("Alert checking loop stopped.")

    async def evaluate_alert_rules(self):
        """Evalúa todas las reglas de alerta configuradas."""
        for rule in self.config.alert_rules:
            rule_name = rule["name"]
            metric_name = rule.get("metric_name")
            threshold = rule.get("threshold")
            operator = rule.get("operator")
            severity = AlertSeverity(rule.get("severity", "warning"))
            message_template = rule.get("message", "Alert triggered for {metric_name}.")
            
            is_alert_triggered = False
            current_value = None

            if metric_name:
                latest_metric = self.metrics_collector.get_latest_metrics().get(metric_name)
                if latest_metric:
                    current_value = latest_metric.value
                    if operator == ">" and current_value > threshold:
                        is_alert_triggered = True
                    elif operator == "<" and current_value < threshold:
                        is_alert_triggered = True
                    elif operator == ">=" and current_value >= threshold:
                        is_alert_triggered = True
                    elif operator == "<=" and current_value <= threshold:
                        is_alert_triggered = True
                    elif operator == "==" and current_value == threshold:
                        is_alert_triggered = True
            
            # También podríamos tener reglas basadas en el estado general del sistema o agentes
            if rule.get("check_agent_status_error"):
                error_agents = [a for a in self.metrics_collector.framework.registry.list_all_agents() if a.status == AgentStatus.ERROR]
                if len(error_agents) > 0:
                    is_alert_triggered = True
                    message_template = f"Error state detected in {len(error_agents)} agents: {[a.name for a in error_agents]}"
            
            if is_alert_triggered:
                if rule_name not in self.active_alerts:
                    # Nueva alerta
                    alert = Alert(
                        rule_name=rule_name,
                        severity=severity,
                        status=AlertStatus.ACTIVE,
                        message=message_template.format(metric_name=metric_name, value=current_value, threshold=threshold),
                        metric_name=metric_name
                    )
                    self.active_alerts[rule_name] = alert
                    self.alert_history.append(alert)
                    self.logger.warning(f"ALERT: {alert.message} (Severity: {alert.severity.value})")
                    await self._send_notification(alert)
            else:
                if rule_name in self.active_alerts:
                    # Alerta resuelta
                    resolved_alert = self.active_alerts.pop(rule_name)
                    resolved_alert.status = AlertStatus.RESOLVED
                    resolved_alert.resolved_at = datetime.now()
                    self.logger.info(f"ALERT RESOLVED: {resolved_alert.message} (Rule: {rule_name})")
                    await self._send_notification(resolved_alert)
        self.logger.debug("Alert rules evaluated.")

    async def _send_notification(self, alert: Alert):
        """Envía notificaciones de alerta a los canales configurados."""
        for channel, settings in self.config.notification_channels.items():
            try:
                if channel == "email" and settings.get("enabled"):
                    await self._send_email_notification(alert, settings)
                elif channel == "webhook" and settings.get("enabled"):
                    await self._send_webhook_notification(alert, settings)
            except Exception as e:
                self.logger.error(f"Failed to send notification via {channel}: {e}")

    async def _send_email_notification(self, alert: Alert, settings: Dict[str, Any]):
        """Envía una notificación por correo electrónico."""
        sender_email = settings.get("sender_email")
        sender_password = settings.get("sender_password")
        receiver_email = settings.get("receiver_email")
        smtp_server = settings.get("smtp_server")
        smtp_port = settings.get("smtp_port", 587)

        if not all([sender_email, sender_password, receiver_email, smtp_server]):
            self.logger.warning("Email notification settings incomplete.")
            return

        message = MimeMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = f"[{alert.severity.value.upper()}] Agent Framework Alert: {alert.rule_name}"
        
        body = f"""
        Alert ID: {alert.id}
        Rule: {alert.rule_name}
        Severity: {alert.severity.value.upper()}
        Status: {alert.status.value.upper()}
        Message: {alert.message}
        Timestamp: {alert.timestamp.isoformat()}
        """
        message.attach(MimeText(body, "plain"))

        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, receiver_email, message.as_string())
            self.logger.info(f"Email alert sent for rule: {alert.rule_name}")
        except Exception as e:
            self.logger.error(f"Error sending email alert: {e}")

    async def _send_webhook_notification(self, alert: Alert, settings: Dict[str, Any]):
        """Envía una notificación vía webhook HTTP."""
        webhook_url = settings.get("webhook_url")
        if not webhook_url:
            self.logger.warning("Webhook URL not configured.")
            return
        
        payload = {
            "alert_id": alert.id,
            "rule_name": alert.rule_name,
            "severity": alert.severity.value,
            "status": alert.status.value,
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat(),
            "metric_name": alert.metric_name,
            "entity_id": alert.entity_id
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    response.raise_for_status() # Lanza excepción para códigos de estado HTTP 4xx/5xx
                    self.logger.info(f"Webhook alert sent for rule: {alert.rule_name} (Status: {response.status})")
        except aiohttp.ClientError as e:
            self.logger.error(f"Error sending webhook alert to {webhook_url}: {e}")

    def get_active_alerts(self) -> List[Alert]:
        """Retorna una lista de todas las alertas activas."""
        return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Retorna un historial de alertas recientes."""
        return self.alert_history[-limit:]

# ================================\
# MONITORING ORCHESTRATOR
# ================================\

class MonitoringOrchestrator:
    """Orquesta los componentes de monitoreo."""
    def __init__(self, framework: AgentFramework, config: Optional[MonitoringConfig] = None):
        self.framework = framework
        self.config = config or MonitoringConfig()
        self.metrics_collector = MetricsCollector(framework, self.config)
        self.health_checker = HealthChecker(framework, self.config)
        self.alert_manager = AlertManager(self.metrics_collector, self.health_checker, self.config)
        self.logger = logging.getLogger("MonitoringOrchestrator")
        self.logger.info("MonitoringOrchestrator initialized.")

    async def start_monitoring(self):
        """Inicia todos los subsistemas de monitoreo."""
        self.logger.info("Starting all monitoring subsystems...")
        await self.metrics_collector.start()
        await self.health_checker.start()
        await self.alert_manager.start()
        self.logger.info("All monitoring subsystems started.")

    async def stop_monitoring(self):
        """Detiene todos los subsistemas de monitoreo."""
        self.logger.info("Stopping all monitoring subsystems...")
        await self.alert_manager.stop()
        await self.health_checker.stop()
        await self.metrics_collector.stop()
        self.logger.info("All monitoring subsystems stopped.")

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Obtiene un resumen del estado del sistema de monitoreo."""
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "total_collected": len(self.metrics_collector.latest_metrics),
                "latest_snapshot": {name: asdict(m) for name, m in self.metrics_collector.latest_metrics.items()}
            },
            "health": self.health_checker.get_health_status(),
            "alerts": {
                "active_alerts": len(self.alert_manager.get_active_alerts()),
                "active_alerts_details": [asdict(a) for a in self.alert_manager.get_active_alerts()],
                "alert_history_count": len(self.alert_manager.alert_history)
            }
        }

# ================================\
# DEMO
# ================================\

async def monitoring_demo():
    """Ejemplo de uso del sistema de monitoreo."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("🚀 Starting Monitoring System Demo")
    print("="*50)

    framework = AgentFramework()
    await framework.start()

    # Añadir un agente ficticio para monitorear
    class MockAgent(BaseAgent):
        def __init__(self, agent_id: str, name: str, framework_instance: AgentFramework):
            super().__init__("demo.mock", name, framework_instance)
            self.id = agent_id # Sobrescribir ID para consistencia en la demo
            self.status = AgentStatus.ACTIVE
            self.message_queue = asyncio.Queue() # Asegurar que tenga cola

        async def initialize(self) -> bool:
            self.logger.info(f"MockAgent {self.name} initialized.")
            return True
        
        async def process_message(self, message):
            self.logger.info(f"MockAgent {self.name} processed message: {message.id}")
            # Simular trabajo
            await asyncio.sleep(0.1)

    agent1 = MockAgent("agent-001", "MockAgentAlpha", framework)
    agent2 = MockAgent("agent-002", "MockAgentBeta", framework)
    
    await agent1.initialize()
    await agent2.initialize()

    await agent1.start()
    await agent2.start()

    # Simular un agente en estado de error
    await framework.registry.update_agent_status(agent2.id, AgentStatus.ERROR)

    # Configuración de monitoreo de ejemplo
    monitoring_config = MonitoringConfig(
        system_metrics_interval=2, # Recolectar métricas cada 2 segundos
        agent_heartbeat_timeout=5, # Timeout para agentes sin heartbeat de 5 segundos (para test rápido)
        alert_rules=[
            {
                "name": "High CPU Usage",
                "metric_name": "system.cpu.usage",
                "operator": ">",
                "threshold": 10, # Disparar si CPU > 10% (para demo)
                "severity": "warning",
                "message": "System CPU usage is {value}% which is above threshold {threshold}%."
            },
            {
                "name": "Agent Error State",
                "check_agent_status_error": True,
                "severity": "critical",
                "message": "One or more agents are in ERROR state."
            }
        ],
        notification_channels={
            # "email": {
            #     "enabled": False,
            #     "sender_email": "your_email@example.com",
            #     "sender_password": "your_password",
            #     "receiver_email": "alert_recipient@example.com",
            #     "smtp_server": "smtp.example.com",
            #     "smtp_port": 587
            # },
            # "webhook": {
            #     "enabled": False,
            #     "webhook_url": "https://webhook.site/your-unique-id"
            # }
        }
    )

    monitoring = MonitoringOrchestrator(framework, monitoring_config)
    await monitoring.start_monitoring()

    print(f"\\nMonitoring system active. Will run for 15 seconds.")
    print(f"Watch for CPU usage alerts and Agent Error State alerts.")
    print(f"MockAgentBeta is in ERROR status to trigger an alert.")

    try:
        for i in range(1, 16):
            print(f"\\n--- Monitoring Cycle {i} ---")
            status = monitoring.get_monitoring_status()
            
            # Imprimir resumen de estado
            print(f"   🏥 Overall health: {status['health']['overall_status']}")
            print(f"   📊 Total agents: {status['health']['agent_summary']['total_agents']}")
            print(f"   ✅ Active agents: {status['health']['agent_summary']['active_agents']}")
            print(f"   🚨 Error agents: {status['health']['agent_summary']['error_agents']}")
            print(f"   📈 Metrics collected: {status['metrics']['total_collected']}")
            print(f"   🚨 Active alerts: {status['alerts']['active_alerts']}")
            
            # Mostrar últimas métricas
            latest_metrics = monitoring.metrics_collector.get_latest_metrics()
            for metric_key, metric in list(latest_metrics.items())[:3]:
                print(f"   📊 {metric.name}: {metric.value:.2f} {metric.unit}")
                
            await asyncio.sleep(2) # Esperar el intervalo de recolección de métricas
                
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted")
        
    # Mostrar resultados finales
    print(f"\n📋 Final Results:")
    
    # Health status
    health_status = monitoring.health_checker.get_health_status()
    print(f"   🏥 Overall health: {health_status['overall_status']}")
    
    # Métricas estadísticas
    cpu_stats = monitoring.metrics_collector.calculate_metric_statistics("system.cpu.usage")
    if cpu_stats:
        print(f"   💻 CPU usage - avg: {cpu_stats['mean']:.1f}%, max: {cpu_stats['max']:.1f}%")
        
    # Historial de alertas
    alert_history = monitoring.alert_manager.get_alert_history(limit=5)
    print(f"   🚨 Recent alerts: {len(alert_history)}")
    for alert in alert_history[:3]:
        print(f"      - {alert.rule_name}: {alert.status.value}")
    
    # Detener monitoreo
    await monitoring.stop_monitoring()
    await framework.stop()
    
    print("\nDemo finished.")

if __name__ == "__main__":
    asyncio.run(monitoring_demo())