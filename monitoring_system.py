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

from autonomous_agent_framework import AgentFramework, BaseAgent, AgentStatus

# ================================
# MONITORING MODELS
# ================================

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
    message: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlertRule:
    """Regla de alerta"""
    name: str
    metric_name: str
    condition: str  # e.g., "> 0.8", "< 10", "== 0"
    threshold: float
    severity: AlertSeverity
    duration: int = 60  # segundos antes de activar
    description: str = ""
    enabled: bool = True
    cooldown: int = 300  # tiempo mínimo entre alertas (segundos)

# ================================
# METRICS COLLECTOR
# ================================

class AdvancedMetricsCollector:
    """Recolector avanzado de métricas"""
    
    def __init__(self, framework: AgentFramework):
        self.framework = framework
        self.metrics_history: List[Metric] = []
        self.max_history_size = 10000
        self.collection_interval = 30  # segundos
        self.running = False
        self.collection_task = None
        
    async def start_collection(self):
        """Iniciar recolección automática de métricas"""
        self.running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logging.info("Metrics collection started")
        
    async def stop_collection(self):
        """Detener recolección de métricas"""
        self.running = False
        if self.collection_task:
            self.collection_task.cancel()
        logging.info("Metrics collection stopped")
        
    async def _collection_loop(self):
        """Loop principal de recolección"""
        while self.running:
            try:
                await self._collect_all_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Metrics collection error: {e}")
                await asyncio.sleep(5)
                
    async def _collect_all_metrics(self):
        """Recopilar todas las métricas"""
        timestamp = datetime.now()
        
        # Métricas del framework
        framework_metrics = await self._collect_framework_metrics(timestamp)
        self._add_metrics(framework_metrics)
        
        # Métricas del sistema
        system_metrics = await self._collect_system_metrics(timestamp)
        self._add_metrics(system_metrics)
        
        # Métricas de agentes
        agent_metrics = await self._collect_agent_metrics(timestamp)
        self._add_metrics(agent_metrics)
        
        # Métricas de rendimiento
        performance_metrics = await self._collect_performance_metrics(timestamp)
        self._add_metrics(performance_metrics)
        
    async def _collect_framework_metrics(self, timestamp: datetime) -> List[Metric]:
        """Recopilar métricas del framework"""
        metrics = []
        
        agents = self.framework.registry.list_all_agents()
        
        # Total de agentes
        metrics.append(Metric(
            name="framework.agents.total",
            type=MetricType.GAUGE,
            value=len(agents),
            timestamp=timestamp,
            unit="count",
            description="Total number of agents"
        ))
        
        # Agentes por estado
        status_counts = {}
        for agent in agents:
            status = agent.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
        for status, count in status_counts.items():
            metrics.append(Metric(
                name="framework.agents.by_status",
                type=MetricType.GAUGE,
                value=count,
                timestamp=timestamp,
                tags={"status": status},
                unit="count",
                description=f"Number of agents in {status} status"
            ))
            
        # Recursos totales
        total_resources = 0
        for agent in agents:
            agent_resources = self.framework.resource_manager.find_resources_by_owner(agent.id)
            total_resources += len(agent_resources)
            
        metrics.append(Metric(
            name="framework.resources.total",
            type=MetricType.GAUGE,
            value=total_resources,
            timestamp=timestamp,
            unit="count",
            description="Total number of resources"
        ))
        
        return metrics
        
    async def _collect_system_metrics(self, timestamp: datetime) -> List[Metric]:
        """Recopilar métricas del sistema"""
        metrics = []
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(Metric(
            name="system.cpu.usage",
            type=MetricType.GAUGE,
            value=cpu_percent,
            timestamp=timestamp,
            unit="percent",
            description="CPU usage percentage"
        ))
        
        # Memoria
        memory = psutil.virtual_memory()
        metrics.append(Metric(
            name="system.memory.usage",
            type=MetricType.GAUGE,
            value=memory.percent,
            timestamp=timestamp,
            unit="percent",
            description="Memory usage percentage"
        ))
        
        metrics.append(Metric(
            name="system.memory.available",
            type=MetricType.GAUGE,
            value=memory.available / (1024**3),  # GB
            timestamp=timestamp,
            unit="GB",
            description="Available memory in GB"
        ))
        
        # Disco
        disk = psutil.disk_usage('/')
        metrics.append(Metric(
            name="system.disk.usage",
            type=MetricType.GAUGE,
            value=(disk.used / disk.total) * 100,
            timestamp=timestamp,
            unit="percent",
            description="Disk usage percentage"
        ))
        
        # Red
        net_io = psutil.net_io_counters()
        metrics.append(Metric(
            name="system.network.bytes_sent",
            type=MetricType.COUNTER,
            value=net_io.bytes_sent,
            timestamp=timestamp,
            unit="bytes",
            description="Total bytes sent"
        ))
        
        metrics.append(Metric(
            name="system.network.bytes_recv",
            type=MetricType.COUNTER,
            value=net_io.bytes_recv,
            timestamp=timestamp,
            unit="bytes",
            description="Total bytes received"
        ))
        
        return metrics
        
    async def _collect_agent_metrics(self, timestamp: datetime) -> List[Metric]:
        """Recopilar métricas específicas de agentes"""
        metrics = []
        
        agents = self.framework.registry.list_all_agents()
        
        for agent in agents:
            agent_tags = {
                "agent_id": agent.id,
                "agent_name": agent.name,
                "namespace": agent.namespace
            }
            
            # Tiempo desde último heartbeat
            time_since_heartbeat = (timestamp - agent.last_heartbeat).total_seconds()
            metrics.append(Metric(
                name="agent.heartbeat.time_since_last",
                type=MetricType.GAUGE,
                value=time_since_heartbeat,
                timestamp=timestamp,
                tags=agent_tags,
                unit="seconds",
                description="Time since last agent heartbeat"
            ))
            
            # Número de capacidades
            metrics.append(Metric(
                name="agent.capabilities.count",
                type=MetricType.GAUGE,
                value=len(agent.capabilities),
                timestamp=timestamp,
                tags=agent_tags,
                unit="count",
                description="Number of agent capabilities"
            ))
            
            # Recursos del agente
            agent_resources = self.framework.resource_manager.find_resources_by_owner(agent.id)
            metrics.append(Metric(
                name="agent.resources.count",
                type=MetricType.GAUGE,
                value=len(agent_resources),
                timestamp=timestamp,
                tags=agent_tags,
                unit="count",
                description="Number of resources owned by agent"
            ))
            
        return metrics
        
    async def _collect_performance_metrics(self, timestamp: datetime) -> List[Metric]:
        """Recopilar métricas de rendimiento"""
        metrics = []
        
        # Tiempo de respuesta simulado (en implementación real, medirías operaciones reales)
        import random
        response_time = random.uniform(0.1, 2.0)  # Simular tiempo de respuesta
        
        metrics.append(Metric(
            name="framework.response_time",
            type=MetricType.TIMER,
            value=response_time,
            timestamp=timestamp,
            unit="seconds",
            description="Framework response time"
        ))
        
        # Throughput simulado
        throughput = random.uniform(10, 100)  # Operaciones por segundo
        metrics.append(Metric(
            name="framework.throughput",
            type=MetricType.GAUGE,
            value=throughput,
            timestamp=timestamp,
            unit="ops/sec",
            description="Framework throughput"
        ))
        
        return metrics
        
    def _add_metrics(self, metrics: List[Metric]):
        """Añadir métricas al historial"""
        self.metrics_history.extend(metrics)
        
        # Mantener límite de historial
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
            
    def get_metric_history(self, metric_name: str, duration_minutes: int = 60) -> List[Metric]:
        """Obtener historial de una métrica específica"""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        
        return [
            metric for metric in self.metrics_history
            if metric.name == metric_name and metric.timestamp >= cutoff_time
        ]
        
    def get_latest_metrics(self) -> Dict[str, Metric]:
        """Obtener las métricas más recientes"""
        latest_metrics = {}
        
        for metric in reversed(self.metrics_history):
            key = f"{metric.name}:{json.dumps(metric.tags, sort_keys=True)}"
            if key not in latest_metrics:
                latest_metrics[key] = metric
                
        return latest_metrics
        
    def calculate_metric_statistics(self, metric_name: str, duration_minutes: int = 60) -> Dict[str, float]:
        """Calcular estadísticas de una métrica"""
        history = self.get_metric_history(metric_name, duration_minutes)
        
        if not history:
            return {}
            
        values = [metric.value for metric in history]
        
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "count": len(values)
        }

# ================================
# ALERTING SYSTEM
# ================================

class AlertManager:
    """Gestor de alertas"""
    
    def __init__(self, metrics_collector: AdvancedMetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: List[Callable] = []
        self.evaluation_interval = 30  # segundos
        self.running = False
        self.evaluation_task = None
        self.last_alert_times: Dict[str, datetime] = {}
        
    def add_alert_rule(self, rule: AlertRule):
        """Añadir regla de alerta"""
        self.alert_rules.append(rule)
        logging.info(f"Added alert rule: {rule.name}")
        
    def remove_alert_rule(self, rule_name: str) -> bool:
        """Eliminar regla de alerta"""
        self.alert_rules = [rule for rule in self.alert_rules if rule.name != rule_name]
        return True
        
    def add_notification_handler(self, handler: Callable):
        """Añadir handler de notificaciones"""
        self.notification_handlers.append(handler)
        
    async def start_monitoring(self):
        """Iniciar monitoreo de alertas"""
        self.running = True
        self.evaluation_task = asyncio.create_task(self._evaluation_loop())
        logging.info("Alert monitoring started")
        
    async def stop_monitoring(self):
        """Detener monitoreo de alertas"""
        self.running = False
        if self.evaluation_task:
            self.evaluation_task.cancel()
        logging.info("Alert monitoring stopped")
        
    async def _evaluation_loop(self):
        """Loop principal de evaluación de alertas"""
        while self.running:
            try:
                await self._evaluate_all_rules()
                await asyncio.sleep(self.evaluation_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Alert evaluation error: {e}")
                await asyncio.sleep(5)
                
    async def _evaluate_all_rules(self):
        """Evaluar todas las reglas de alerta"""
        current_metrics = self.metrics_collector.get_latest_metrics()
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
                
            await self._evaluate_rule(rule, current_metrics)
            
    async def _evaluate_rule(self, rule: AlertRule, current_metrics: Dict[str, Metric]):
        """Evaluar una regla específica"""
        # Buscar métricas que coincidan con la regla
        matching_metrics = [
            metric for metric in current_metrics.values()
            if metric.name == rule.metric_name
        ]
        
        for metric in matching_metrics:
            if self._check_condition(metric.value, rule.condition, rule.threshold):
                await self._trigger_alert(rule, metric)
            else:
                await self._resolve_alert(rule, metric)
                
    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Verificar condición de alerta"""
        try:
            if condition.startswith(">="):
                return value >= threshold
            elif condition.startswith("<="):
                return value <= threshold
            elif condition.startswith(">"):
                return value > threshold
            elif condition.startswith("<"):
                return value < threshold
            elif condition.startswith("=="):
                return value == threshold
            elif condition.startswith("!="):
                return value != threshold
            else:
                return eval(f"{value} {condition} {threshold}")
        except Exception:
            return False
            
    async def _trigger_alert(self, rule: AlertRule, metric: Metric):
        """Activar alerta"""
        alert_key = f"{rule.name}:{json.dumps(metric.tags, sort_keys=True)}"
        
        # Verificar cooldown
        if alert_key in self.last_alert_times:
            time_since_last = (datetime.now() - self.last_alert_times[alert_key]).total_seconds()
            if time_since_last < rule.cooldown:
                return
                
        # Verificar si ya existe alerta activa
        if alert_key in self.active_alerts:
            return
            
        # Crear nueva alerta
        alert = Alert(
            id=f"alert_{int(time.time())}_{hash(alert_key) % 10000}",
            rule_name=rule.name,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            message=f"{rule.description or rule.name}: {metric.name} = {metric.value} {rule.condition} {rule.threshold}",
            triggered_at=datetime.now(),
            tags=metric.tags,
            metadata={
                "metric_name": metric.name,
                "metric_value": metric.value,
                "threshold": rule.threshold,
                "condition": rule.condition
            }
        )
        
        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)
        self.last_alert_times[alert_key] = datetime.now()
        
        # Enviar notificaciones
        await self._send_notifications(alert)
        
        logging.warning(f"Alert triggered: {alert.message}")
        
    async def _resolve_alert(self, rule: AlertRule, metric: Metric):
        """Resolver alerta"""
        alert_key = f"{rule.name}:{json.dumps(metric.tags, sort_keys=True)}"
        
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            
            del self.active_alerts[alert_key]
            
            # Enviar notificación de resolución
            await self._send_notifications(alert)
            
            logging.info(f"Alert resolved: {alert.message}")
            
    async def _send_notifications(self, alert: Alert):
        """Enviar notificaciones de alerta"""
        for handler in self.notification_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logging.error(f"Notification handler error: {e}")
                
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Reconocer alerta"""
        for alert in self.active_alerts.values():
            if alert.id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now()
                alert.acknowledged_by = acknowledged_by
                return True
        return False
        
    def get_active_alerts(self, severity: AlertSeverity = None) -> List[Alert]:
        """Obtener alertas activas"""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
            
        return sorted(alerts, key=lambda x: x.triggered_at, reverse=True)
        
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Obtener historial de alertas"""
        return sorted(self.alert_history[-limit:], key=lambda x: x.triggered_at, reverse=True)

# ================================
# NOTIFICATION HANDLERS
# ================================

class EmailNotificationHandler:
    """Handler de notificaciones por email"""
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str, 
                 password: str, from_email: str, to_emails: List[str]):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
        
    async def __call__(self, alert: Alert):
        """Enviar notificación por email"""
        try:
            subject = f"[{alert.severity.value.upper()}] {alert.rule_name}"
            
            if alert.status == AlertStatus.ACTIVE:
                body = f"""
Alert Triggered

Rule: {alert.rule_name}
Severity: {alert.severity.value}
Message: {alert.message}
Triggered At: {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S')}

Tags: {json.dumps(alert.tags, indent=2)}
Metadata: {json.dumps(alert.metadata, indent=2)}
                """
            else:
                body = f"""
Alert Resolved

Rule: {alert.rule_name}
Message: {alert.message}
Triggered At: {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S')}
Resolved At: {alert.resolved_at.strftime('%Y-%m-%d %H:%M:%S') if alert.resolved_at else 'N/A'}
                """
                
            # Crear mensaje
            msg = MimeMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = subject
            msg.attach(MimeText(body, 'plain'))
            
            # Enviar email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
                
            logging.info(f"Email notification sent for alert: {alert.id}")
            
        except Exception as e:
            logging.error(f"Failed to send email notification: {e}")

class SlackNotificationHandler:
    """Handler de notificaciones a Slack"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        
    async def __call__(self, alert: Alert):
        """Enviar notificación a Slack"""
        try:
            # Color según severidad
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9500", 
                AlertSeverity.CRITICAL: "#ff0000",
                AlertSeverity.FATAL: "#8B0000"
            }
            
            color = color_map.get(alert.severity, "#36a64f")
            
            # Emoji según estado
            emoji = "🚨" if alert.status == AlertStatus.ACTIVE else "✅"
            
            # Crear payload
            payload = {
                "text": f"{emoji} Alert {alert.status.value.title()}",
                "attachments": [
                    {
                        "color": color,
                        "fields": [
                            {
                                "title": "Rule",
                                "value": alert.rule_name,
                                "short": True
                            },
                            {
                                "title": "Severity", 
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Message",
                                "value": alert.message,
                                "short": False
                            },
                            {
                                "title": "Triggered At",
                                "value": alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S'),
                                "short": True
                            }
                        ]
                    }
                ]
            }
            
            # Enviar a Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        logging.info(f"Slack notification sent for alert: {alert.id}")
                    else:
                        logging.error(f"Failed to send Slack notification: {response.status}")
                        
        except Exception as e:
            logging.error(f"Failed to send Slack notification: {e}")

class WebhookNotificationHandler:
    """Handler de notificaciones por webhook"""
    
    def __init__(self, webhook_url: str, auth_token: str = None):
        self.webhook_url = webhook_url
        self.auth_token = auth_token
        
    async def __call__(self, alert: Alert):
        """Enviar notificación por webhook"""
        try:
            headers = {"Content-Type": "application/json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
                
            payload = {
                "alert_id": alert.id,
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "message": alert.message,
                "triggered_at": alert.triggered_at.isoformat(),
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                "tags": alert.tags,
                "metadata": alert.metadata
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        logging.info(f"Webhook notification sent for alert: {alert.id}")
                    else:
                        logging.error(f"Failed to send webhook notification: {response.status}")
                        
        except Exception as e:
            logging.error(f"Failed to send webhook notification: {e}")

# ================================
# HEALTH CHECK SYSTEM
# ================================

class HealthChecker:
    """Sistema de health checks"""
    
    def __init__(self, framework: AgentFramework):
        self.framework = framework
        self.health_checks: Dict[str, Callable] = {}
        self.check_interval = 60  # segundos
        self.running = False
        self.check_task = None
        self.last_results: Dict[str, Dict[str, Any]] = {}
        
    def register_health_check(self, name: str, check_function: Callable):
        """Registrar health check"""
        self.health_checks[name] = check_function
        
    async def start_health_checks(self):
        """Iniciar health checks automáticos"""
        self.running = True
        self.check_task = asyncio.create_task(self._health_check_loop())
        logging.info("Health checks started")
        
    async def stop_health_checks(self):
        """Detener health checks"""
        self.running = False
        if self.check_task:
            self.check_task.cancel()
        logging.info("Health checks stopped")
        
    async def _health_check_loop(self):
        """Loop de health checks"""
        while self.running:
            try:
                await self._run_all_health_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Health check loop error: {e}")
                await asyncio.sleep(5)
                
    async def _run_all_health_checks(self):
        """Ejecutar todos los health checks"""
        for name, check_function in self.health_checks.items():
            try:
                result = await self._run_health_check(name, check_function)
                self.last_results[name] = result
            except Exception as e:
                logging.error(f"Health check {name} failed: {e}")
                self.last_results[name] = {
                    "status": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                
    async def _run_health_check(self, name: str, check_function: Callable) -> Dict[str, Any]:
        """Ejecutar un health check específico"""
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(check_function):
                result = await check_function()
            else:
                result = check_function()
                
            duration = time.time() - start_time
            
            return {
                "status": "healthy" if result else "unhealthy",
                "duration": duration,
                "timestamp": datetime.now().isoformat(),
                "details": result if isinstance(result, dict) else {"result": result}
            }
            
        except Exception as e:
            duration = time.time() - start_time
            return {
                "status": "error",
                "duration": duration,
                "timestamp": datetime.now().isoformat(),
                "message": str(e)
            }
            
    def get_health_status(self) -> Dict[str, Any]:
        """Obtener estado general de salud"""
        overall_status = "healthy"
        
        for result in self.last_results.values():
            if result.get("status") != "healthy":
                overall_status = "unhealthy"
                break
                
        return {
            "overall_status": overall_status,
            "checks": self.last_results,
            "timestamp": datetime.now().isoformat()
        }

# ================================
# MONITORING ORCHESTRATOR
# ================================

class MonitoringOrchestrator:
    """Orquestador principal de monitoreo"""
    
    def __init__(self, framework: AgentFramework):
        self.framework = framework
        self.metrics_collector = AdvancedMetricsCollector(framework)
        self.alert_manager = AlertManager(self.metrics_collector)
        self.health_checker = HealthChecker(framework)
        
        # Configurar reglas de alerta por defecto
        self._setup_default_alert_rules()
        
        # Configurar health checks por defecto
        self._setup_default_health_checks()
        
    def _setup_default_alert_rules(self):
        """Configurar reglas de alerta por defecto"""
        rules = [
            AlertRule(
                name="high_cpu_usage",
                metric_name="system.cpu.usage",
                condition=">",
                threshold=80.0,
                severity=AlertSeverity.WARNING,
                description="High CPU usage detected"
            ),
            AlertRule(
                name="high_memory_usage", 
                metric_name="system.memory.usage",
                condition=">",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                description="High memory usage detected"
            ),
            AlertRule(
                name="critical_memory_usage",
                metric_name="system.memory.usage", 
                condition=">",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL,
                description="Critical memory usage detected"
            ),
            AlertRule(
                name="agent_heartbeat_timeout",
                metric_name="agent.heartbeat.time_since_last",
                condition=">", 
                threshold=120.0,  # 2 minutos
                severity=AlertSeverity.WARNING,
                description="Agent heartbeat timeout"
            ),
            AlertRule(
                name="no_active_agents",
                metric_name="framework.agents.total",
                condition="==",
                threshold=0.0,
                severity=AlertSeverity.CRITICAL,
                description="No active agents in framework"
            )
        ]
        
        for rule in rules:
            self.alert_manager.add_alert_rule(rule)
            
    def _setup_default_health_checks(self):
        """Configurar health checks por defecto"""
        
        async def framework_health():
            """Health check del framework"""
            agents = self.framework.registry.list_all_agents()
            active_agents = [a for a in agents if a.status == AgentStatus.ACTIVE]
            
            return {
                "total_agents": len(agents),
                "active_agents": len(active_agents),
                "healthy": len(active_agents) > 0
            }
            
        async def system_resources_health():
            """Health check de recursos del sistema"""
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            healthy = (cpu_percent < 90 and 
                      memory.percent < 90 and 
                      (disk.used / disk.total) * 100 < 90)
            
            return {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "disk_usage": (disk.used / disk.total) * 100,
                "healthy": healthy
            }
            
        self.health_checker.register_health_check("framework", framework_health)
        self.health_checker.register_health_check("system_resources", system_resources_health)
        
    async def start_monitoring(self):
        """Iniciar monitoreo completo"""
        await self.metrics_collector.start_collection()
        await self.alert_manager.start_monitoring()
        await self.health_checker.start_health_checks()
        
        logging.info("Complete monitoring started")
        
    async def stop_monitoring(self):
        """Detener monitoreo completo"""
        await self.metrics_collector.stop_collection()
        await self.alert_manager.stop_monitoring()
        await self.health_checker.stop_health_checks()
        
        logging.info("Complete monitoring stopped")
        
    def add_notification_handler(self, handler):
        """Añadir handler de notificaciones"""
        self.alert_manager.add_notification_handler(handler)
        
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Obtener estado completo del monitoreo"""
        return {
            "metrics": {
                "total_collected": len(self.metrics_collector.metrics_history),
                "collection_running": self.metrics_collector.running
            },
            "alerts": {
                "total_rules": len(self.alert_manager.alert_rules),
                "active_alerts": len(self.alert_manager.active_alerts),
                "monitoring_running": self.alert_manager.running
            },
            "health_checks": {
                "total_checks": len(self.health_checker.health_checks),
                "last_results": self.health_checker.last_results,
                "checking_running": self.health_checker.running
            }
        }

# ================================
# EXAMPLE USAGE
# ================================

async def monitoring_demo():
    """Demo del sistema de monitoreo"""
    
    logging.basicConfig(level=logging.INFO)
    
    print("📊 Advanced Monitoring System Demo")
    print("="*60)
    
    # Crear framework y agentes
    from autonomous_agent_framework import AgentFramework
    from specialized_agents import ExtendedAgentFactory
    
    framework = AgentFramework()
    await framework.start()
    
    # Crear algunos agentes
    strategist = ExtendedAgentFactory.create_agent("agent.planning.strategist", "strategist", framework)
    generator = ExtendedAgentFactory.create_agent("agent.build.code.generator", "generator", framework)
    
    await strategist.start()
    await generator.start()
    
    # Crear sistema de monitoreo
    monitoring = MonitoringOrchestrator(framework)
    
    # Configurar notificaciones (simuladas)
    class ConsoleNotificationHandler:
        async def __call__(self, alert: Alert):
            status_emoji = "🚨" if alert.status == AlertStatus.ACTIVE else "✅"
            print(f"{status_emoji} ALERT [{alert.severity.value.upper()}]: {alert.message}")
            
    monitoring.add_notification_handler(ConsoleNotificationHandler())
    
    # Iniciar monitoreo
    await monitoring.start_monitoring()
    
    print(f"\n✅ Monitoring started")
    print(f"   Agents: {len(framework.registry.list_all_agents())}")
    print(f"   Alert rules: {len(monitoring.alert_manager.alert_rules)}")
    print(f"   Health checks: {len(monitoring.health_checker.health_checks)}")
    
    # Simular actividad durante 2 minutos
    print(f"\n⏳ Running monitoring for 2 minutes...")
    print(f"   Watch for metrics collection and alerts...")
    
    try:
        for i in range(24):  # 2 minutos en intervalos de 5 segundos
            await asyncio.sleep(5)
            
            # Mostrar progreso cada 30 segundos
            if i % 6 == 0:
                status = monitoring.get_monitoring_status()
                print(f"   📈 Metrics collected: {status['metrics']['total_collected']}")
                print(f"   🚨 Active alerts: {status['alerts']['active_alerts']}")
                
                # Mostrar últimas métricas
                latest_metrics = monitoring.metrics_collector.get_latest_metrics()
                for metric_key, metric in list(latest_metrics.items())[:3]:
                    print(f"   📊 {metric.name}: {metric.value:.2f} {metric.unit}")
                    
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
    
    print(f"\n✅ Monitoring demo completed!")

if __name__ == "__main__":
    asyncio.run(monitoring_demo())