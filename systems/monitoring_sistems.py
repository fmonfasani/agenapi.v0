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

# Importaciones actualizadas: AgentStatus ahora viene de core.models
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
    source: str = "system"
    triggered_value: Optional[float] = None
    threshold: Optional[float] = None
    agent_id: Optional[str] = None
    resource_id: Optional[str] = None

@dataclass
class HealthStatus:
    """Estado de salud general del sistema o un componente"""
    component: str
    status: AgentStatus # Usa AgentStatus
    last_check: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

# ================================\
# METRICS COLLECTOR
# ================================\

class MetricsCollector:
    """
    Colector de métricas del sistema.
    Recopila métricas del sistema operativo y del framework.
    """
    def __init__(self, framework: AgentFramework, collection_interval: int = 5):
        self.framework = framework
        self.collection_interval = collection_interval
        self.metrics: Dict[str, List[Metric]] = {}
        self._is_running = False
        self._collect_task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)

    async def start(self):
        """Inicia la recolección de métricas periódicamente."""
        if not self._is_running:
            self.logger.info("Starting metrics collection...")
            self._is_running = True
            self._collect_task = asyncio.create_task(self._collect_metrics_loop())

    async def stop(self):
        """Detiene la recolección de métricas."""
        if self._is_running:
            self.logger.info("Stopping metrics collection...")
            self._is_running = False
            if self._collect_task:
                self._collect_task.cancel()
                try:
                    await self._collect_task
                except asyncio.CancelledError:
                    self.logger.info("Metrics collection task cancelled.")

    async def _collect_metrics_loop(self):
        """Bucle principal para la recolección periódica de métricas."""
        while self._is_running:
            await self.collect_system_metrics()
            await self.collect_framework_metrics()
            await asyncio.sleep(self.collection_interval)

    async def collect_system_metrics(self):
        """Recopila métricas del sistema operativo."""
        try:
            cpu_usage = psutil.cpu_percent(interval=None) # Non-blocking
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('/')

            self.add_metric("system.cpu.usage", MetricType.GAUGE, cpu_usage, "percent", "Overall CPU usage")
            self.add_metric("system.memory.total", MetricType.GAUGE, memory_info.total, "bytes", "Total physical memory")
            self.add_metric("system.memory.available", MetricType.GAUGE, memory_info.available, "bytes", "Available physical memory")
            self.add_metric("system.memory.percent", MetricType.GAUGE, memory_info.percent, "percent", "Percentage of memory used")
            self.add_metric("system.disk.total", MetricType.GAUGE, disk_info.total, "bytes", "Total disk space")
            self.add_metric("system.disk.used", MetricType.GAUGE, disk_info.used, "bytes", "Used disk space")
            self.add_metric("system.disk.percent", MetricType.GAUGE, disk_info.percent, "percent", "Percentage of disk space used")

            # Network metrics (example)
            net_io = psutil.net_io_counters()
            self.add_metric("system.network.bytes_sent", MetricType.COUNTER, net_io.bytes_sent, "bytes")
            self.add_metric("system.network.bytes_recv", MetricType.COUNTER, net_io.bytes_recv, "bytes")

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")

    async def collect_framework_metrics(self):
        """Recopila métricas del framework de agentes."""
        try:
            total_agents = len(self.framework.registry.list_all_agents())
            active_agents = len([a for a in self.framework.registry.list_all_agents() if a.status == AgentStatus.ACTIVE])
            idle_agents = len([a for a in self.framework.registry.list_all_agents() if a.status == AgentStatus.IDLE])
            busy_agents = len([a for a in self.framework.registry.list_all_agents() if a.status == AgentStatus.BUSY])
            error_agents = len([a for a in self.framework.registry.list_all_agents() if a.status == AgentStatus.ERROR])

            self.add_metric("framework.agents.total", MetricType.GAUGE, float(total_agents), "count", "Total registered agents")
            self.add_metric("framework.agents.active", MetricType.GAUGE, float(active_agents), "count", "Active agents")
            self.add_metric("framework.agents.idle", MetricType.GAUGE, float(idle_agents), "count", "Idle agents")
            self.add_metric("framework.agents.busy", MetricType.GAUGE, float(busy_agents), "count", "Busy agents")
            self.add_metric("framework.agents.error", MetricType.GAUGE, float(error_agents), "count", "Agents in error state")
            self.add_metric("framework.messages.queue_size", MetricType.GAUGE, float(self.framework.message_bus.queue.qsize()), "count", "Current message queue size")
            self.add_metric("framework.resources.total", MetricType.GAUGE, float(len(self.framework.resource_manager.list_all_resources())), "count", "Total managed resources")
            
            # Metrics on agent-specific tasks/messages could be added here
            for agent_info in self.framework.registry.list_all_agents():
                self.add_metric(f"agent.{agent_info.namespace}.status", MetricType.GAUGE, float(agent_info.status.value == AgentStatus.ACTIVE), "boolean", tags={"agent_id": agent_info.id, "agent_name": agent_info.name})

        except Exception as e:
            self.logger.error(f"Error collecting framework metrics: {e}")

    def add_metric(self, name: str, m_type: MetricType, value: float, unit: str = "", description: str = "", tags: Dict[str, str] = None):
        """Añade una métrica al colector."""
        metric = Metric(name=name, type=m_type, value=value, timestamp=datetime.now(), tags=tags or {}, unit=unit, description=description)
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(metric)
        # Mantener un historial limitado de métricas (ej. últimas 100)
        self.metrics[name] = self.metrics[name][-100:]

    def get_latest_metrics(self) -> Dict[str, Metric]:
        """Retorna la última métrica de cada tipo."""
        latest = {}
        for name, metrics_list in self.metrics.items():
            if metrics_list:
                latest[name] = metrics_list[-1] # La más reciente
        return latest
    
    def get_metric_history(self, name: str, limit: int = 100) -> List[Metric]:
        """Retorna el historial de una métrica específica."""
        return self.metrics.get(name, [])[-limit:]

    def calculate_metric_statistics(self, name: str) -> Optional[Dict[str, float]]:
        """Calcula estadísticas básicas para una métrica (solo para GAUGES/COUNTERS)."""
        history = self.get_metric_history(name)
        if not history:
            return None
        
        values = [m.value for m in history if m.type in [MetricType.GAUGE, MetricType.COUNTER]]
        if not values:
            return None
            
        return {
            "count": len(values),
            "sum": sum(values),
            "mean": statistics.mean(values) if values else 0,
            "min": min(values),
            "max": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0
        }

# ================================\
# ALERT MANAGER
# ================================\

class AlertManager:
    """
    Gestor de alertas del sistema.
    Define reglas, evalúa métricas y envía notificaciones.
    """
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: List[Dict[str, Any]] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self._is_running = False
        self._check_task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)

    async def start(self):
        """Inicia el chequeo de alertas periódicamente."""
        if not self._is_running:
            self.logger.info("Starting alert manager...")
            self._is_running = True
            self._check_task = asyncio.create_task(self._check_alerts_loop())
            
    async def stop(self):
        """Detiene el chequeo de alertas."""
        if self._is_running:
            self.logger.info("Stopping alert manager...")
            self._is_running = False
            if self._check_task:
                self._check_task.cancel()
                try:
                    await self._check_task
                except asyncio.CancelledError:
                    self.logger.info("Alert check task cancelled.")

    async def _check_alerts_loop(self):
        """Bucle principal para el chequeo periódico de alertas."""
        while self._is_running:
            await self.check_all_rules()
            await asyncio.sleep(self.metrics_collector.collection_interval) # Usar el mismo intervalo o uno propio

    def add_alert_rule(self, rule_name: str, metric_name: str, threshold: float, severity: AlertSeverity,
                       operator: str = ">", description: str = "", cooldown_seconds: int = 300):
        """Añade una regla de alerta."""
        self.alert_rules.append({
            "name": rule_name,
            "metric": metric_name,
            "threshold": threshold,
            "severity": severity,
            "operator": operator,
            "description": description,
            "cooldown": timedelta(seconds=cooldown_seconds),
            "last_triggered": datetime.min # Para el cooldown
        })
        self.logger.info(f"Added alert rule: {rule_name} for {metric_name} {operator} {threshold}")

    async def check_all_rules(self):
        """Evalúa todas las reglas de alerta configuradas."""
        latest_metrics = self.metrics_collector.get_latest_metrics()
        for rule in self.alert_rules:
            metric_name = rule["metric"]
            if metric_name in latest_metrics:
                current_metric = latest_metrics[metric_name]
                await self._evaluate_rule(rule, current_metric)

    async def _evaluate_rule(self, rule: Dict[str, Any], metric: Metric):
        """Evalúa una única regla de alerta."""
        
        # Cooldown check
        if datetime.now() - rule["last_triggered"] < rule["cooldown"]:
            return # Aún en cooldown

        threshold = rule["threshold"]
        operator = rule["operator"]
        triggered = False

        if operator == ">" and metric.value > threshold:
            triggered = True
        elif operator == "<" and metric.value < threshold:
            triggered = True
        elif operator == ">=" and metric.value >= threshold:
            triggered = True
        elif operator == "<=" and metric.value <= threshold:
            triggered = True
        elif operator == "==" and metric.value == threshold:
            triggered = True
        elif operator == "!=" and metric.value != threshold:
            triggered = True
        
        alert_id = f"alert_{rule['name']}_{metric.name}"

        if triggered:
            if alert_id not in self.active_alerts:
                self.logger.warning(f"ALERT TRIGGERED: {rule['name']} ({metric.name}: {metric.value} {operator} {threshold})")
                new_alert = Alert(
                    id=alert_id,
                    rule_name=rule["name"],
                    severity=rule["severity"],
                    status=AlertStatus.ACTIVE,
                    message=f"Metric '{metric.name}' ({metric.value:.2f} {metric.unit}) violated threshold {operator} {threshold}.",
                    source=f"metric:{metric.name}",
                    triggered_value=metric.value,
                    threshold=threshold,
                    agent_id=metric.tags.get("agent_id")
                )
                self.active_alerts[alert_id] = new_alert
                self.alert_history.append(new_alert)
                rule["last_triggered"] = datetime.now()
                await self.notify_alert(new_alert)
        else:
            if alert_id in self.active_alerts:
                self.logger.info(f"ALERT RESOLVED: {rule['name']} ({metric.name} back to normal)")
                resolved_alert = self.active_alerts.pop(alert_id)
                resolved_alert.status = AlertStatus.RESOLVED
                resolved_alert.completed_at = datetime.now() # Add completed_at field to Alert
                # Update alert in history if possible, otherwise add a new resolved entry
                for i, hist_alert in enumerate(self.alert_history):
                    if hist_alert.id == resolved_alert.id:
                        self.alert_history[i] = resolved_alert
                        break
                else: # if not found (e.g. if history is limited)
                    self.alert_history.append(resolved_alert)

    async def notify_alert(self, alert: Alert):
        """Envía notificaciones para una alerta (simulado)."""
        self.logger.info(f"Sending notification for alert: {alert.rule_name} (Severity: {alert.severity.value})")
        # Aquí se integrarían sistemas de notificación reales (ej. email, Slack, PagerDuty)
        await self._send_email_notification(alert)
        await self._send_webhook_notification(alert)

    async def _send_email_notification(self, alert: Alert):
        """Simula el envío de una notificación por correo electrónico."""
        # Esto es un placeholder; se necesitaría un servidor SMTP real
        sender_email = "alerts@agent-framework.com"
        receiver_email = "admin@agent-framework.com"
        password = "your_email_password" # Esto no debería estar aquí en producción
        
        message = MIMEMultipart("alternative")
        message["Subject"] = f"AGENT FRAMEWORK ALERT: {alert.rule_name} ({alert.severity.value})"
        message["From"] = sender_email
        message["To"] = receiver_email

        text = f"""
        Alert ID: {alert.id}
        Rule: {alert.rule_name}
        Severity: {alert.severity.value}
        Status: {alert.status.value}
        Message: {alert.message}
        Timestamp: {alert.timestamp.isoformat()}
        Source: {alert.source}
        Triggered Value: {alert.triggered_value}
        Threshold: {alert.threshold}
        Agent ID: {alert.agent_id}
        """
        html = f"""
        <html>
            <body>
                <h3>Agent Framework Alert</h3>
                <p><b>Alert ID:</b> {alert.id}</p>
                <p><b>Rule:</b> {alert.rule_name}</p>
                <p><b>Severity:</b> <span style="color: {'red' if alert.severity == AlertSeverity.CRITICAL else 'orange' if alert.severity == AlertSeverity.WARNING else 'blue'}">{alert.severity.value}</span></p>
                <p><b>Status:</b> {alert.status.value}</p>
                <p><b>Message:</b> {alert.message}</p>
                <p><b>Timestamp:</b> {alert.timestamp.isoformat()}</p>
                <p><b>Source:</b> {alert.source}</p>
                <p><b>Triggered Value:</b> {alert.triggered_value}</p>
                <p><b>Threshold:</b> {alert.threshold}</p>
                <p><b>Agent ID:</b> {alert.agent_id}</p>
            </body>
        </html>
        """
        
        part1 = MimeText(text, "plain")
        part2 = MimeText(html, "html")
        message.attach(part1)
        message.attach(part2)

        try:
            # Descomentar y configurar para un servidor SMTP real
            # with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            #     server.login(sender_email, password)
            #     server.sendmail(sender_email, receiver_email, message.as_string())
            self.logger.debug(f"Simulated email sent for alert {alert.id}")
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")

    async def _send_webhook_notification(self, alert: Alert):
        """Simula el envío de una notificación vía webhook."""
        webhook_url = "http://localhost:9090/webhook_receiver" # Ejemplo de URL
        headers = {"Content-Type": "application/json"}
        payload = asdict(alert)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, headers=headers, timeout=5) as response:
                    response.raise_for_status()
                    self.logger.debug(f"Simulated webhook sent for alert {alert.id}. Status: {response.status}")
        except aiohttp.ClientError as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout sending webhook for alert {alert.id}")


    def get_active_alerts(self) -> List[Alert]:
        """Retorna una lista de alertas activas."""
        return list(self.active_alerts.values())
        
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Retorna el historial de alertas."""
        return self.alert_history[-limit:]

# ================================\
# HEALTH CHECKER
# ================================\

class HealthChecker:
    """
    Sistema de chequeo de salud del framework y sus componentes.
    """
    def __init__(self, framework: AgentFramework, check_interval: int = 10):
        self.framework = framework
        self.check_interval = check_interval
        self._is_running = False
        self._check_task: Optional[asyncio.Task] = None
        self.health_statuses: Dict[str, HealthStatus] = {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)

    async def start(self):
        """Inicia el chequeo de salud periódicamente."""
        if not self._is_running:
            self.logger.info("Starting health checker...")
            self._is_running = True
            self._check_task = asyncio.create_task(self._health_check_loop())

    async def stop(self):
        """Detiene el chequeo de salud."""
        if self._is_running:
            self.logger.info("Stopping health checker...")
            self._is_running = False
            if self._check_task:
                self._check_task.cancel()
                try:
                    await self._check_task
                except asyncio.CancelledError:
                    self.logger.info("Health check task cancelled.")

    async def _health_check_loop(self):
        """Bucle principal para el chequeo periódico de salud."""
        while self._is_running:
            await self.perform_health_check()
            await asyncio.sleep(self.check_interval)

    async def perform_health_check(self):
        """Realiza un chequeo de salud de todos los componentes clave."""
        self.logger.debug("Performing system health check...")
        
        # Check Framework Core
        framework_status = self.framework.get_status() # Asumiendo que AgentFramework tiene un método get_status
        self.update_health_status("framework_core", framework_status, details={"uptime_seconds": (datetime.now() - self.framework.start_time).total_seconds()})

        # Check Agents
        agents = self.framework.registry.list_all_agents()
        for agent_info in agents:
            status = AgentStatus.ACTIVE if agent_info.is_alive else AgentStatus.ERROR # Simplified check
            details = {"last_heartbeat": agent_info.last_heartbeat.isoformat() if agent_info.last_heartbeat else "N/A"}
            self.update_health_status(f"agent.{agent_info.name}", status, details=details,
                                      errors=[f"Agent {agent_info.name} is not alive"] if not agent_info.is_alive else [])

        # Check Message Bus
        message_bus_status = AgentStatus.ACTIVE if self.framework.message_bus.is_running else AgentStatus.ERROR
        queue_size = self.framework.message_bus.queue.qsize() if hasattr(self.framework.message_bus, 'queue') else 0
        self.update_health_status("message_bus", message_bus_status, details={"queue_size": queue_size},
                                  errors=["Message Bus not running"] if not self.framework.message_bus.is_running else [])
        
        # Check Resource Manager
        resource_manager_status = AgentStatus.ACTIVE # Asumiendo que siempre está activo si el framework lo está
        total_resources = len(self.framework.resource_manager.list_all_resources())
        self.update_health_status("resource_manager", resource_manager_status, details={"total_resources": total_resources})

        # Check Persistence Manager (if integrated)
        if hasattr(self.framework, 'persistence_manager') and self.framework.persistence_manager:
            persistence_status = await self.framework.persistence_manager.get_status() # Assuming a get_status method
            self.update_health_status("persistence_manager", persistence_status.get('status', AgentStatus.ERROR), details=persistence_status)

        self.logger.debug("System health check completed.")

    def update_health_status(self, component: str, status: AgentStatus, details: Dict[str, Any] = None, warnings: List[str] = None, errors: List[str] = None):
        """Actualiza el estado de salud de un componente."""
        self.health_statuses[component] = HealthStatus(
            component=component,
            status=status,
            details=details or {},
            warnings=warnings or [],
            errors=errors or []
        )
        if status == AgentStatus.ERROR:
            self.logger.error(f"Health check: Component '{component}' is in ERROR state. Details: {errors or 'N/A'}")
        elif status == AgentStatus.WARNING:
            self.logger.warning(f"Health check: Component '{component}' is in WARNING state. Details: {warnings or 'N/A'}")
        else:
            self.logger.debug(f"Health check: Component '{component}' is {status.value}")

    def get_health_status(self, component: Optional[str] = None) -> Dict[str, Any]:
        """
        Retorna el estado de salud de un componente específico o el estado general del sistema.
        """
        if component:
            health = self.health_statuses.get(component)
            return asdict(health) if health else {"error": "Component not found", "status": "unknown"}
        
        overall_status = AgentStatus.ACTIVE
        total_errors = 0
        total_warnings = 0

        for status_obj in self.health_statuses.values():
            if status_obj.status == AgentStatus.ERROR:
                overall_status = AgentStatus.ERROR
                total_errors += 1
            elif status_obj.status == AgentStatus.WARNING and overall_status != AgentStatus.ERROR:
                overall_status = AgentStatus.WARNING
                total_warnings += 1
        
        return {
            "overall_status": overall_status.value,
            "components": {name: asdict(status_obj) for name, status_obj in self.health_statuses.items()},
            "summary": {
                "total_components": len(self.health_statuses),
                "error_components": total_errors,
                "warning_components": total_warnings,
                "active_components": len([s for s in self.health_statuses.values() if s.status == AgentStatus.ACTIVE])
            }
        }

# ================================\
# MONITORING ORCHESTRATOR
# ================================\

class MonitoringOrchestrator:
    """
    Orquestador principal del sistema de monitoreo.
    Integra el colector de métricas, el gestor de alertas y el chequeador de salud.
    """
    def __init__(self, framework: AgentFramework, config: Dict[str, Any] = None):
        self.framework = framework
        self.config = config or {}
        
        self.metrics_collector = MetricsCollector(
            framework,
            collection_interval=self.config.get("metrics_collection_interval", 5)
        )
        self.alert_manager = AlertManager(self.metrics_collector)
        self.health_checker = HealthChecker(
            framework,
            check_interval=self.config.get("health_check_interval", 10)
        )
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)
        
        self._configure_alerts()

    def _configure_alerts(self):
        """Carga reglas de alerta predefinidas o desde configuración."""
        # Ejemplos de reglas de alerta
        self.alert_manager.add_alert_rule(
            rule_name="High CPU Usage",
            metric_name="system.cpu.usage",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            operator=">",
            description="System CPU usage is consistently above 80%.",
            cooldown_seconds=300
        )
        self.alert_manager.add_alert_rule(
            rule_name="Agent Error State",
            metric_name="framework.agents.error",
            threshold=0.0,
            severity=AlertSeverity.CRITICAL,
            operator=">",
            description="One or more agents are in an error state.",
            cooldown_seconds=60
        )
        self.alert_manager.add_alert_rule(
            rule_name="Message Queue Full",
            metric_name="framework.messages.queue_size",
            threshold=900.0,
            severity=AlertSeverity.WARNING,
            operator=">",
            description="Message bus queue is nearly full, indicating backlogs.",
            cooldown_seconds=120
        )
        self.logger.info("Default alert rules configured.")

    async def start_monitoring(self):
        """Inicia todos los subsistemas de monitoreo."""
        self.logger.info("Starting all monitoring subsystems...")
        await self.metrics_collector.start()
        await self.alert_manager.start()
        await self.health_checker.start()
        self.logger.info("Monitoring subsystems started.")

    async def stop_monitoring(self):
        """Detiene todos los subsistemas de monitoreo."""
        self.logger.info("Stopping all monitoring subsystems...")
        await self.health_checker.stop()
        await self.alert_manager.stop()
        await self.metrics_collector.stop()
        self.logger.info("Monitoring subsystems stopped.")

    def get_full_status(self) -> Dict[str, Any]:
        """Obtiene el estado completo del sistema de monitoreo."""
        return {
            "health_status": self.health_checker.get_health_status(),
            "latest_metrics": {name: asdict(m) for name, m in self.metrics_collector.get_latest_metrics().items()},
            "active_alerts": [asdict(a) for a in self.alert_manager.get_active_alerts()],
            "alerts_history_count": len(self.alert_manager.get_alert_history())
        }

# ================================\
# DEMO / USAGE EXAMPLE
# (This will eventually be moved to end_to_end_example.py)
# ================================\

async def monitoring_demo():
    """Ejemplo de uso del sistema de monitoreo."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("MonitoringDemo")

    framework = AgentFramework()
    monitoring = MonitoringOrchestrator(framework)
    
    try:
        await framework.start()
        await monitoring.start_monitoring()
        
        # Simular algunos agentes y su actividad para generar métricas
        class DemoAgent(BaseAgent):
            def __init__(self, name: str, framework: AgentFramework):
                super().__init__("demo.agent", name, framework)
                self.counter = 0
            async def initialize(self) -> bool: return True
            async def run(self):
                while self.status == AgentStatus.ACTIVE:
                    self.counter += 1
                    # self.logger.info(f"{self.name} is working... {self.counter}")
                    await asyncio.sleep(1) # Simular trabajo
                    
            async def stop(self):
                self.set_status(AgentStatus.TERMINATED)
                self.logger.info(f"{self.name} stopped.")

        agent1 = DemoAgent("AgentAlpha", framework)
        await agent1.initialize()
        await agent1.start()
        
        agent2 = DemoAgent("AgentBeta", framework)
        await agent2.initialize()
        await agent2.start()

        # Simular un agente en estado de error para probar alertas
        error_agent = DemoAgent("ErrorAgent", framework)
        await error_agent.initialize()
        await error_agent.start()
        error_agent.set_status(AgentStatus.ERROR) # Forzar estado de error

        logger.info("Monitoring system active. Simulating agent activity for 30 seconds...")
        logger.info("Check status, metrics, and alerts.")

        for i in range(6): # Ejecutar por 30 segundos (5s interval * 6)
            await asyncio.sleep(5)
            status = monitoring.get_full_status()
            logger.info(f"\n--- Monitoring Report (Cycle {i+1}) ---")
            logger.info(f"   Overall Health: {status['health_status']['overall_status']}")
            
            # Show agent specific health
            for comp_name, comp_status in status['health_status']['components'].items():
                if comp_name.startswith("agent."):
                    logger.info(f"     {comp_name}: {comp_status['status']}")

            if status['active_alerts']:
                logger.warning(f"   🚨 Active Alerts ({len(status['active_alerts'])}):")
                for alert in status['active_alerts']:
                    logger.warning(f"      - [{alert['severity'].value}] {alert['rule_name']}: {alert['message']}")
            else:
                logger.info("   ✅ No active alerts.")
                
            # Mostrar algunas métricas recolectadas
            if status['latest_metrics']:
                logger.info("   📈 Metrics collected:")
                # print(f"   📈 Metrics collected: {status['metrics']['total_collected']}") # This line was an error in previous version
                for metric_key, metric_data in list(status['latest_metrics'].items())[:3]:
                    logger.info(f"      📊 {metric_data['name']}: {metric_data['value']:.2f} {metric_data.get('unit', '')}")
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Demo interrupted")
        
    finally:
        # Detener agentes de demo
        await agent1.stop()
        await agent2.stop()
        await error_agent.stop()
        
        # Mostrar resultados finales
        logger.info(f"\n📋 Final Results:")
        
        # Health status
        health_status = monitoring.health_checker.get_health_status()
        logger.info(f"   🏥 Overall health: {health_status['overall_status']}")
        
        # Métricas estadísticas
        cpu_stats = monitoring.metrics_collector.calculate_metric_statistics("system.cpu.usage")
        if cpu_stats:
            logger.info(f"   💻 CPU usage - avg: {cpu_stats['mean']:.1f}%, max: {cpu_stats['max']:.1f}%")
            
        # Historial de alertas
        alert_history = monitoring.alert_manager.get_alert_history(limit=5)
        logger.info(f"   🚨 Recent alerts: {len(alert_history)}")
        for alert in alert_history[:3]:
            logger.info(f"      - {alert.rule_name}: {alert.status.value}")
        
        # Detener monitoreo y framework
        await monitoring.stop_monitoring()
        await framework.stop()
        logger.info("Monitoring demo finished.")

if __name__ == "__main__":
    asyncio.run(monitoring_demo())