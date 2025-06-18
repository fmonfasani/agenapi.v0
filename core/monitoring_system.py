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

from core.autonomous_agent_framework import AgentFramework, BaseAgent, AgentStatus

# CORE FRAMEWORK CLASSES

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
    id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    triggered_at: datetime
    last_updated_at: datetime
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None

@dataclass
class HealthStatus:
    """Estado de salud de un componente o del sistema"""
    component: str
    status: str
    last_check: datetime
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

class MetricsCollector:
    """Colector de métricas del sistema y de agentes."""
    def __init__(self, framework: AgentFramework):
        self.framework = framework
        self._metrics: List[Metric] = []
        self._metrics_by_name: Dict[str, List[Metric]] = {}
        self.logger = logging.getLogger("MetricsCollector")

    async def collect_system_metrics(self):
        """Colecta métricas del sistema operativo."""
        timestamp = datetime.utcnow()
        try:
            cpu_percent = psutil.cpu_percent(interval=None) # Non-blocking
            mem_info = psutil.virtual_memory()
            disk_usage = psutil.disk_usage('/')

            self.record_metric(Metric("system.cpu.usage", MetricType.GAUGE, cpu_percent, timestamp, unit="%"))
            self.record_metric(Metric("system.memory.total", MetricType.GAUGE, mem_info.total, timestamp, unit="bytes"))
            self.record_metric(Metric("system.memory.available", MetricType.GAUGE, mem_info.available, timestamp, unit="bytes"))
            self.record_metric(Metric("system.memory.percent", MetricType.GAUGE, mem_info.percent, timestamp, unit="%"))
            self.record_metric(Metric("system.disk.total", MetricType.GAUGE, disk_usage.total, timestamp, unit="bytes", tags={"path": "/"}))
            self.record_metric(Metric("system.disk.used", MetricType.GAUGE, disk_usage.used, timestamp, unit="bytes", tags={"path": "/"}))
            self.record_metric(Metric("system.disk.percent", MetricType.GAUGE, disk_usage.percent, timestamp, unit="%", tags={"path": "/"}))

            net_io = psutil.net_io_counters()
            self.record_metric(Metric("system.network.bytes_sent", MetricType.COUNTER, net_io.bytes_sent, timestamp, unit="bytes"))
            self.record_metric(Metric("system.network.bytes_recv", MetricType.COUNTER, net_io.bytes_recv, timestamp, unit="bytes"))

            self.logger.debug("System metrics collected.")
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")

    async def collect_agent_metrics(self):
        """Colecta métricas de los agentes del framework."""
        timestamp = datetime.utcnow()
        total_agents = len(self.framework.registry.list_all_agents())
        active_agents = sum(1 for a in self.framework.registry.list_all_agents() if a.status == AgentStatus.ACTIVE)
        error_agents = sum(1 for a in self.framework.registry.list_all_agents() if a.status == AgentStatus.ERROR)
        busy_agents = sum(1 for a in self.framework.registry.list_all_agents() if a.status == AgentStatus.BUSY)

        self.record_metric(Metric("agents.total", MetricType.GAUGE, total_agents, timestamp, description="Total number of agents"))
        self.record_metric(Metric("agents.active", MetricType.GAUGE, active_agents, timestamp, description="Number of active agents"))
        self.record_metric(Metric("agents.error", MetricType.GAUGE, error_agents, timestamp, description="Number of agents in error state"))
        self.record_metric(Metric("agents.busy", MetricType.GAUGE, busy_agents, timestamp, description="Number of busy agents"))

        for agent in self.framework.registry.list_all_agents():
            self.record_metric(Metric(f"agent.{agent.namespace}.{agent.name}.status", MetricType.GAUGE, 1 if agent.status == AgentStatus.ACTIVE else 0, timestamp, tags={"agent_id": agent.id, "status": agent.status.value}))
            # Podríamos añadir métricas específicas de cada agente si los agentes las exponen

        self.logger.debug("Agent metrics collected.")

    def record_metric(self, metric: Metric):
        """Registra una métrica."""
        self._metrics.append(metric)
        if metric.name not in self._metrics_by_name:
            self._metrics_by_name[metric.name] = []
        self._metrics_by_name[metric.name].append(metric)
        self.logger.debug(f"Recorded metric: {metric.name}={metric.value}")

    def get_metrics_in_range(self, metric_name: str, start_time: datetime, end_time: datetime) -> List[Metric]:
        """Obtiene métricas por nombre dentro de un rango de tiempo."""
        if metric_name not in self._metrics_by_name:
            return []
        return [m for m in self._metrics_by_name[metric_name] if start_time <= m.timestamp <= end_time]

    def get_latest_metric(self, metric_name: str) -> Optional[Metric]:
        """Obtiene la última métrica registrada por nombre."""
        if metric_name not in self._metrics_by_name or not self._metrics_by_name[metric_name]:
            return None
        return self._metrics_by_name[metric_name][-1] # Asume que las métricas se añaden cronológicamente

    def get_latest_metrics(self) -> Dict[str, Metric]:
        """Obtiene la última métrica de cada tipo registrado."""
        latest: Dict[str, Metric] = {}
        for name, metrics_list in self._metrics_by_name.items():
            if metrics_list:
                latest[name] = metrics_list[-1]
        return latest

    def calculate_metric_statistics(self, metric_name: str, window_seconds: int = 300) -> Optional[Dict[str, float]]:
        """Calcula estadísticas básicas para una métrica en una ventana de tiempo."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(seconds=window_seconds)
        relevant_metrics = self.get_metrics_in_range(metric_name, start_time, end_time)

        if not relevant_metrics:
            return None

        values = [m.value for m in relevant_metrics]
        if not values:
            return None

        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0
        }

    async def clear_old_metrics(self, retention_days: int = 7):
        """Elimina métricas antiguas para liberar memoria."""
        cutoff_time = datetime.utcnow() - timedelta(days=retention_days)
        initial_count = len(self._metrics)
        self._metrics = [m for m in self._metrics if m.timestamp >= cutoff_time]
        
        for metric_name in list(self._metrics_by_name.keys()):
            self._metrics_by_name[metric_name] = [m for m in self._metrics_by_name[metric_name] if m.timestamp >= cutoff_time]
            if not self._metrics_by_name[metric_name]:
                del self._metrics_by_name[metric_name]
        
        removed_count = initial_count - len(self._metrics)
        if removed_count > 0:
            self.logger.info(f"Cleared {removed_count} old metrics (older than {retention_days} days).")


class AlertManager:
    """Gestiona la creación, estado y notificaciones de alertas."""
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self._alert_rules: Dict[str, Dict[str, Any]] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self.logger = logging.getLogger("AlertManager")
        self._notification_handlers: List[Callable[[Alert], Any]] = []

    def define_alert_rule(self, rule_name: str, metric_name: str, threshold: float, severity: AlertSeverity,
                          operator: str = ">", description: str = "", cooldown_seconds: int = 300):
        """Define una nueva regla de alerta."""
        self._alert_rules[rule_name] = {
            "metric_name": metric_name,
            "threshold": threshold,
            "severity": severity,
            "operator": operator,
            "description": description,
            "cooldown_seconds": cooldown_seconds,
            "last_triggered": None
        }
        self.logger.info(f"Alert rule '{rule_name}' defined for metric '{metric_name}'")

    async def evaluate_rules(self):
        """Evalúa todas las reglas de alerta basadas en las últimas métricas."""
        for rule_name, rule in self._alert_rules.items():
            metric = self.metrics_collector.get_latest_metric(rule["metric_name"])
            if not metric:
                continue

            value = metric.value
            threshold = rule["threshold"]
            operator = rule["operator"]
            triggered = False

            if operator == ">" and value > threshold:
                triggered = True
            elif operator == "<" and value < threshold:
                triggered = True
            elif operator == ">=" and value >= threshold:
                triggered = True
            elif operator == "<=" and value <= threshold:
                triggered = True
            elif operator == "==" and value == threshold:
                triggered = True
            elif operator == "!=" and value != threshold:
                triggered = True

            if triggered:
                await self._trigger_alert(rule_name, rule, metric)
            else:
                await self._resolve_alert(rule_name)

    async def _trigger_alert(self, rule_name: str, rule: Dict[str, Any], metric: Metric):
        """Dispara una alerta si se cumplen las condiciones y el cooldown lo permite."""
        now = datetime.utcnow()
        last_triggered = rule.get("last_triggered")
        if last_triggered and (now - last_triggered).total_seconds() < rule["cooldown_seconds"]:
            self.logger.debug(f"Alert rule {rule_name} is in cooldown. Skipping trigger.")
            return

        message = f"ALERTA: {rule['description']} - {rule['metric_name']} ({metric.value}{metric.unit}) {rule['operator']} {rule['threshold']}"
        alert_id = f"alert-{rule_name}"

        if alert_id not in self._active_alerts:
            new_alert = Alert(
                id=alert_id,
                rule_name=rule_name,
                severity=rule["severity"],
                status=AlertStatus.ACTIVE,
                triggered_at=now,
                last_updated_at=now,
                message=message,
                details={"metric_value": metric.value, "threshold": rule["threshold"], "operator": rule["operator"]}
            )
            self._active_alerts[alert_id] = new_alert
            self._alert_history.append(new_alert)
            self.logger.warning(f"ALERT TRIGGERED: {message}")
            await self._notify_alert(new_alert)
        else:
            # Update existing active alert
            active_alert = self._active_alerts[alert_id]
            if active_alert.status != AlertStatus.ACTIVE:
                # Re-activate if it was resolved/acknowledged and re-triggered
                active_alert.status = AlertStatus.ACTIVE
                self.logger.warning(f"ALERT REACTIVATED: {message}")
                await self._notify_alert(active_alert)
            active_alert.last_updated_at = now
            active_alert.message = message
            active_alert.details = {"metric_value": metric.value, "threshold": rule["threshold"], "operator": rule["operator"]}
            self.logger.debug(f"Alert {alert_id} re-triggered (still active).")

        rule["last_triggered"] = now

    async def _resolve_alert(self, rule_name: str):
        """Resuelve una alerta si ya no se cumplen las condiciones."""
        alert_id = f"alert-{rule_name}"
        if alert_id in self._active_alerts and self._active_alerts[alert_id].status == AlertStatus.ACTIVE:
            resolved_alert = self._active_alerts[alert_id]
            resolved_alert.status = AlertStatus.RESOLVED
            resolved_alert.resolved_at = datetime.utcnow()
            resolved_alert.last_updated_at = datetime.utcnow()
            self.logger.info(f"ALERT RESOLVED: {resolved_alert.message}")
            await self._notify_alert(resolved_alert)
            # No eliminar de _active_alerts inmediatamente para mantener un rastro reciente
            # Podría ser eliminado por una tarea de limpieza posterior
            del self._active_alerts[alert_id] # Eliminar para que pueda ser re-activada si las condiciones vuelven

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Marca una alerta como reconocida."""
        if alert_id in self._active_alerts:
            self._active_alerts[alert_id].status = AlertStatus.ACKNOWLEDGED
            self._active_alerts[alert_id].acknowledged_by = acknowledged_by
            self._active_alerts[alert_id].last_updated_at = datetime.utcnow()
            self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}.")

    def get_active_alerts(self) -> List[Alert]:
        """Retorna todas las alertas activas."""
        return list(self._active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Retorna el historial de alertas (las más recientes primero)."""
        return sorted(self._alert_history, key=lambda x: x.triggered_at, reverse=True)[:limit]

    def register_notification_handler(self, handler: Callable[[Alert], Any]):
        """Registra una función para manejar notificaciones de alerta."""
        self._notification_handlers.append(handler)
        self.logger.info(f"Registered notification handler: {handler.__name__}")

    async def _notify_alert(self, alert: Alert):
        """Envía la alerta a todos los handlers registrados."""
        for handler in self._notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert notification handler {handler.__name__}: {e}", exc_info=True)

    async def email_notifier(self, alert: Alert, to_email: str, from_email: str, smtp_server: str, smtp_port: int, smtp_user: str, smtp_password: str):
        """Ejemplo de handler de notificación por correo electrónico."""
        subject = f"[{alert.severity.value.upper()}] ALERTA: {alert.rule_name}"
        body = f"""
        Alerta ID: {alert.id}
        Regla: {alert.rule_name}
        Severidad: {alert.severity.value.upper()}
        Estado: {alert.status.value.upper()}
        Mensaje: {alert.message}
        Activada: {alert.triggered_at.isoformat()}
        Última actualización: {alert.last_updated_at.isoformat()}
        Detalles: {json.dumps(alert.details, indent=2)}
        """

        msg = MimeMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MimeText(body, 'plain'))

        try:
            with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
            self.logger.info(f"Email notification sent for alert {alert.id} to {to_email}")
        except Exception as e:
            self.logger.error(f"Failed to send email notification for alert {alert.id}: {e}")

    async def webhook_notifier(self, alert: Alert, webhook_url: str):
        """Ejemplo de handler de notificación por webhook (ej. Slack, Teams)."""
        payload = {
            "alert_id": alert.id,
            "rule_name": alert.rule_name,
            "severity": alert.severity.value,
            "status": alert.status.value,
            "message": alert.message,
            "timestamp": alert.triggered_at.isoformat(),
            "details": alert.details
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    response.raise_for_status()
                    self.logger.info(f"Webhook notification sent for alert {alert.id} to {webhook_url}")
        except aiohttp.ClientError as e:
            self.logger.error(f"Failed to send webhook notification for alert {alert.id} to {webhook_url}: {e}")


class HealthChecker:
    """Realiza chequeos de salud de los componentes del framework y agentes."""
    def __init__(self, framework: AgentFramework):
        self.framework = framework
        self._component_health: Dict[str, HealthStatus] = {}
        self.logger = logging.getLogger("HealthChecker")

    async def check_framework_health(self):
        """Chequea la salud de los componentes internos del framework."""
        timestamp = datetime.utcnow()
        # Verificar Message Bus
        msg_bus_status = "healthy"
        msg_bus_message = "Message bus is responsive."
        try:
            # Simular un envío de mensaje (no intrusivo) para verificar actividad
            test_message = BaseAgent("test.namespace", "test_agent", self.framework)
            test_message.id = "test_sender"
            await self.framework.message_bus.publish(
                AgentMessage(sender_id="health_check", receiver_id="non_existent_agent", message_type="heartbeat", payload={})
            )
        except Exception as e:
            msg_bus_status = "unhealthy"
            msg_bus_message = f"Message bus error: {e}"
            self.logger.error(msg_bus_message)
        self._component_health["message_bus"] = HealthStatus("Message Bus", msg_bus_status, timestamp, msg_bus_message)

        # Verificar Agent Registry
        registry_status = "healthy"
        registry_message = f"Registry contains {len(self.framework.registry.list_all_agents())} agents."
        try:
            if not isinstance(self.framework.registry.list_all_agents(), list):
                raise TypeError("Agent registry did not return a list.")
        except Exception as e:
            registry_status = "unhealthy"
            registry_message = f"Agent Registry error: {e}"
            self.logger.error(registry_message)
        self._component_health["agent_registry"] = HealthStatus("Agent Registry", registry_status, timestamp, registry_message)

        # Verificar Resource Manager
        resource_manager_status = "healthy"
        resource_manager_message = f"Resource Manager tracks {len(self.framework.resource_manager.list_all_resources())} resources."
        try:
            if not isinstance(self.framework.resource_manager.list_all_resources(), list):
                raise TypeError("Resource Manager did not return a list.")
        except Exception as e:
            resource_manager_status = "unhealthy"
            resource_manager_message = f"Resource Manager error: {e}"
            self.logger.error(resource_manager_message)
        self._component_health["resource_manager"] = HealthStatus("Resource Manager", resource_manager_status, timestamp, resource_manager_message)

        self.logger.debug("Framework health checked.")

    async def check_agent_health(self):
        """Chequea la salud de cada agente individualmente."""
        timestamp = datetime.utcnow()
        for agent in self.framework.registry.list_all_agents():
            status = "healthy"
            message = "Agent is responding and active."
            if agent.status == AgentStatus.ERROR:
                status = "unhealthy"
                message = "Agent is in ERROR state."
            elif agent.status == AgentStatus.TERMINATED:
                status = "unhealthy"
                message = "Agent is TERMINATED."
            elif agent.status == AgentStatus.INITIALIZING:
                status = "degraded"
                message = "Agent is still INITIALIZING."
            
            # Podríamos enviar un mensaje de heartbeat al agente y esperar una respuesta
            # Para una demo simplificada, nos basamos en el estado reportado
            
            self._component_health[f"agent.{agent.id}"] = HealthStatus(
                f"Agent {agent.name}", status, timestamp, message, {"agent_status": agent.status.value}
            )
        self.logger.debug("Agent health checked.")

    def get_health_status(self) -> Dict[str, Any]:
        """Retorna un resumen del estado de salud general."""
        overall_status = "healthy"
        for component, status_obj in self._component_health.items():
            if status_obj.status == "unhealthy":
                overall_status = "unhealthy"
                break
            elif status_obj.status == "degraded" and overall_status == "healthy":
                overall_status = "degraded"
        
        return {
            "overall_status": overall_status,
            "components": {k: asdict(v) for k, v in self._component_health.items()}
        }

class MonitoringSystem:
    """Sistema de Monitoreo principal que orquesta la recolección, alerta y chequeo de salud."""
    def __init__(self, framework: AgentFramework):
        self.framework = framework
        self.metrics_collector = MetricsCollector(framework)
        self.alert_manager = AlertManager(self.metrics_collector)
        self.health_checker = HealthChecker(framework)
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_running = False
        self.logger = logging.getLogger("MonitoringSystem")

        # Configurar reglas de alerta por defecto
        self._setup_default_alert_rules()

    def _setup_default_alert_rules(self):
        """Configura algunas reglas de alerta básicas."""
        self.alert_manager.define_alert_rule(
            rule_name="high_cpu_usage",
            metric_name="system.cpu.usage",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            operator=">",
            description="System CPU usage is high"
        )
        self.alert_manager.define_alert_rule(
            rule_name="critical_cpu_usage",
            metric_name="system.cpu.usage",
            threshold=95.0,
            severity=AlertSeverity.CRITICAL,
            operator=">",
            description="System CPU usage is critically high"
        )
        self.alert_manager.define_alert_rule(
            rule_name="low_memory_available",
            metric_name="system.memory.percent",
            threshold=90.0,
            severity=AlertSeverity.WARNING,
            operator=">",
            description="System memory usage is high"
        )
        self.alert_manager.define_alert_rule(
            rule_name="agent_in_error_state",
            metric_name="agents.error",
            threshold=0.0,
            severity=AlertSeverity.CRITICAL,
            operator=">",
            description="One or more agents are in an error state"
        )
        self.logger.info("Default alert rules set up.")

    async def start_monitoring(self, interval_seconds: int = 5):
        """Inicia el proceso de monitoreo en un bucle continuo."""
        if self._is_running:
            self.logger.warning("Monitoring system is already running.")
            return

        self.logger.info("Starting monitoring system...")
        self._is_running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop(interval_seconds))
        self.logger.info("Monitoring system started.")

    async def _monitoring_loop(self, interval_seconds: int):
        """Bucle principal de monitoreo."""
        while self._is_running:
            start_time = time.time()
            try:
                await self.metrics_collector.collect_system_metrics()
                await self.metrics_collector.collect_agent_metrics()
                await self.alert_manager.evaluate_rules()
                await self.health_checker.check_framework_health()
                await self.health_checker.check_agent_health()
                await self.metrics_collector.clear_old_metrics(retention_days=7)
            except asyncio.CancelledError:
                self.logger.info("Monitoring loop cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}", exc_info=True)

            elapsed_time = time.time() - start_time
            sleep_time = interval_seconds - elapsed_time
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                self.logger.warning(f"Monitoring loop took longer than interval ({elapsed_time:.2f}s vs {interval_seconds}s)")

    async def stop_monitoring(self):
        """Detiene el proceso de monitoreo."""
        if not self._is_running:
            self.logger.warning("Monitoring system is not running.")
            return

        self.logger.info("Stopping monitoring system...")
        self._is_running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Monitoring system stopped.")

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Obtiene un resumen del estado del sistema de monitoreo."""
        return {
            "system_status": "Running" if self._is_running else "Stopped",
            "metrics": {
                "total_collected": len(self.metrics_collector._metrics),
                "unique_metric_types": len(self.metrics_collector._metrics_by_name)
            },
            "alerts": {
                "active_alerts": len(self.alert_manager.get_active_alerts()),
                "total_rules": len(self.alert_manager._alert_rules)
            },
            "health_checks": self.health_checker.get_health_status()
        }

    def register_alert_notification_channel(self, handler: Callable[[Alert], Any]):
        """Registra un canal de notificación de alertas."""
        self.alert_manager.register_notification_handler(handler)

async def demo():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("Monitoring_Demo")

    # Inicializar el framework de agentes (simulado para la demo)
    framework = AgentFramework(log_level=logging.WARNING) # Usar WARNING para reducir logs de framework
    await framework.start()

    # Inicializar un agente dummy para la demo
    class DummyAgent(BaseAgent):
        def __init__(self, name: str, framework_ref):
            super().__init__("agent.dummy", name, framework_ref)
            self._status_cycle = [AgentStatus.ACTIVE, AgentStatus.BUSY, AgentStatus.ERROR, AgentStatus.IDLE]
            self._status_index = 0

        async def initialize(self) -> bool:
            self.logger.info(f"DummyAgent {self.name} initialized.")
            return True

        async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
            if action == "cycle_status":
                self._status_index = (self._status_index + 1) % len(self._status_cycle)
                self.status = self._status_cycle[self._status_index]
                self.logger.info(f"DummyAgent {self.name} status changed to {self.status.value}")
                return {"new_status": self.status.value}
            return {"error": "Unknown action"}

    # Registrar el tipo de agente en el AgentFactory
    from core.autonomous_agent_framework import AgentFactory
    AgentFactory.register_agent_type("agent.dummy", DummyAgent)

    # Crear algunos agentes
    await framework.agent_manager.create_agent("agent.dummy", "agent_X", DummyAgent)
    await framework.agent_manager.create_agent("agent.dummy", "agent_Y", DummyAgent)

    # Inicializar el sistema de monitoreo
    monitoring = MonitoringSystem(framework)

    # Registrar un handler de notificación simple para la demo
    async def simple_notification(alert: Alert):
        logger.info(f"🔔 NOTIFICACIÓN: {alert.message} (Severidad: {alert.severity.value})")

    monitoring.register_alert_notification_channel(simple_notification)

    # Demo
    print("--- Monitoring System Demo ---")

    # 1. Iniciar monitoreo
    print("\n1. Starting monitoring system (collecting metrics, evaluating alerts, checking health)...")
    await monitoring.start_monitoring(interval_seconds=1) # Monitoreo rápido para la demo

    # Dar tiempo para que se recolecten métricas y se evalúen alertas
    print("   Waiting for initial metrics and alerts...")
    await asyncio.sleep(3) # Permite 3 ciclos de monitoreo

    # 2. Obtener estado actual
    print("\n2. Current Monitoring Status:")
    status = monitoring.get_monitoring_status()
    print(f"   System Status: {status['system_status']}")
    print(f"   Metrics collected: {status['metrics']['total_collected']}")
    print(f"   Active alerts: {status['alerts']['active_alerts']}")
    print(f"   Overall health: {status['health_checks']['overall_status']}")

    # Mostrar últimas métricas
    latest_metrics = monitoring.metrics_collector.get_latest_metrics()
    print("\n   Latest Metrics:")
    for metric_key, metric in list(latest_metrics.items())[:5]: # Mostrar solo las primeras 5
        print(f"     - {metric.name}: {metric.value:.2f} {metric.unit}")

    # Mostrar alertas activas
    active_alerts = monitoring.alert_manager.get_active_alerts()
    print(f"\n   Active Alerts ({len(active_alerts)}):")
    for alert in active_alerts:
        print(f"     - [{alert.severity.value.upper()}] {alert.rule_name}: {alert.message}")

    # 3. Simular un cambio de estado en un agente para activar una alerta (si hay agentes en error)
    print("\n3. Simulating agent error state to trigger alert...")
    agent_x = framework.registry.get_agent(next(iter(framework.registry._agents.keys())))
    if agent_x:
        # Forzar al agente a entrar en estado de error
        agent_x.status = AgentStatus.ERROR
        print(f"   Agent {agent_x.name} manually set to ERROR state.")

    print("   Waiting for monitoring system to detect agent error and trigger alert...")
    await asyncio.sleep(2) # Dar tiempo para que el monitoreo detecte el cambio

    print("\n   Checking active alerts after simulating error:")
    active_alerts_after_error = monitoring.alert_manager.get_active_alerts()
    for alert in active_alerts_after_error:
        print(f"     - [{alert.severity.value.upper()}] {alert.rule_name}: {alert.message} (Status: {alert.status.value})")

    # 4. Simular resolución de alerta
    print("\n4. Simulating alert resolution by setting agent back to active...")
    if agent_x:
        agent_x.status = AgentStatus.ACTIVE
        print(f"   Agent {agent_x.name} manually set back to ACTIVE state.")

    print("   Waiting for monitoring system to resolve alert...")
    await asyncio.sleep(2)

    print("\n   Checking active alerts after simulating resolution:")
    active_alerts_after_resolution = monitoring.alert_manager.get_active_alerts()
    if not active_alerts_after_resolution:
        print("     No active alerts. Alert resolved successfully.")
    for alert in active_alerts_after_resolution:
         print(f"     - [{alert.severity.value.upper()}] {alert.rule_name}: {alert.message} (Status: {alert.status.value})")

    # 5. Obtener historial de alertas
    print("\n5. Recent Alert History:")
    alert_history = monitoring.alert_manager.get_alert_history(limit=5)
    if not alert_history:
        print("   No alerts in history.")
    for i, alert in enumerate(alert_history, 1):
        print(f"   {i}. ID: {alert.id[:8]}..., Rule: {alert.rule_name}, Severity: {alert.severity.value}, Status: {alert.status.value}, Triggered: {alert.triggered_at.strftime('%H:%M:%S')}")

    # Cleanup
    print("\n--- Demo Cleanup ---")
    await monitoring.stop_monitoring()
    await framework.stop()
    print("Monitoring system and framework stopped.")

if __name__ == "__main__":
    asyncio.run(demo())