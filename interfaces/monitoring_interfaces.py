# agentapi/interfaces/monitoring_interfaces.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from agentapi.models.framework_models import MonitoringConfig
from agentapi.models.general_models import Metric, Alert, AlertSeverity, AlertStatus

class IMonitoringSystem(ABC):
    """
    Interfaz abstracta para el sistema de monitoreo.
    Define las operaciones de recolección de métricas, chequeo de salud y gestión de alertas.
    """

    @abstractmethod
    async def start_monitoring(self, config: MonitoringConfig) -> bool:
        """Inicia los subsistemas de monitoreo."""
        pass

    @abstractmethod
    async def stop_monitoring(self) -> None:
        """Detiene los subsistemas de monitoreo."""
        pass

    @property
    @abstractmethod
    def metrics_collector(self) -> Any: # Retorna una interfaz de IMetricsCollector
        """Obtiene la instancia del recolector de métricas."""
        pass

    @property
    @abstractmethod
    def health_checker(self) -> Any: # Retorna una interfaz de IHealthChecker
        """Obtiene la instancia del verificador de salud."""
        pass

    @property
    @abstractmethod
    def alert_manager(self) -> Any: # Retorna una interfaz de IAlertManager
        """Obtiene la instancia del gestor de alertas."""
        pass

    @abstractmethod
    async def get_system_status(self) -> Dict[str, Any]:
        """Obtiene un resumen del estado general del sistema de monitoreo."""
        pass


class IMetricsCollector(ABC):
    """Interfaz abstracta para el recolector de métricas."""

    @abstractmethod
    def add_metric(self, metric: Metric) -> None:
        """Agrega una métrica al recolector."""
        pass

    @abstractmethod
    def add_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Incrementa un contador o agrega un valor a una métrica de tipo contador."""
        pass

    @abstractmethod
    def add_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Establece el valor de una métrica de tipo gauge."""
        pass

    @abstractmethod
    def get_metric(self, name: str) -> Optional[Metric]:
        """Obtiene la última métrica por nombre."""
        pass

    @abstractmethod
    def get_all_metrics(self) -> List[Metric]:
        """Obtiene todas las métricas recolectadas."""
        pass

    @abstractmethod
    def get_latest_metrics(self) -> Dict[str, Metric]:
        """Obtiene las últimas métricas por nombre, en un diccionario."""
        pass

    @abstractmethod
    def calculate_metric_statistics(self, name: str) -> Optional[Dict[str, float]]:
        """Calcula estadísticas (min, max, avg, etc.) para una métrica."""
        pass

    @abstractmethod
    async def export_metrics(self) -> Dict[str, Any]:
        """Exporta las métricas actuales en un formato procesable."""
        pass


class IHealthChecker(ABC):
    """Interfaz abstracta para el verificador de salud del sistema."""

    @abstractmethod
    async def perform_health_check(self) -> Dict[str, Any]:
        """Realiza un chequeo de salud completo de los componentes del framework."""
        pass

    @abstractmethod
    def report_component_status(self, component_name: str, status: str, message: str) -> None:
        """Reporta el estado de un componente específico."""
        pass

    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Obtiene el estado de salud actual de todos los componentes."""
        pass


class IAlertManager(ABC):
    """Interfaz abstracta para el gestor de alertas."""

    @abstractmethod
    async def raise_alert(self, rule_name: str, severity: AlertSeverity, message: str,
                          details: Optional[Dict[str, Any]] = None) -> Alert:
        """Levanta una nueva alerta."""
        pass

    @abstractmethod
    async def resolve_alert(self, alert_id: str, resolved_by: Optional[str] = None) -> Optional[Alert]:
        """Resuelve una alerta existente."""
        pass

    @abstractmethod
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: Optional[str] = None) -> Optional[Alert]:
        """Reconoce una alerta existente."""
        pass

    @abstractmethod
    def get_active_alerts(self) -> List[Alert]:
        """Obtiene todas las alertas activas."""
        pass

    @abstractmethod
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Obtiene el historial de alertas."""
        pass

    @abstractmethod
    async def notify_alert(self, alert: Alert) -> bool:
        """Envía notificaciones para una alerta."""
        pass