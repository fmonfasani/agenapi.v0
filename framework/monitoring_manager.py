# framework/monitoring_manager.py

import asyncio
import logging
from typing import List, Optional
from datetime import datetime

from agentapi.models.general_models import Metric, Alert, AlertSeverity, AlertStatus

class MonitoringManager: # Renamed from MonitoringOrchestrator for consistency with other managers
    def __init__(self):
        self.metrics: List[Metric] = []
        self.alerts: List[Alert] = []
        self._background_tasks: List[asyncio.Task] = [] # To manage internal monitoring tasks
        self._running = False
        self.logger = logging.getLogger("MonitoringManager")

    async def initialize(self):
        """Initializes monitoring components and starts background tasks."""
        if self._running:
            return
        self._running = True
        self.logger.info("MonitoringManager initializing...")
        
        # Example: Start a task for processing/flushing metrics or evaluating alerts
        # self._background_tasks.append(asyncio.create_task(self._metric_processing_loop()))
        # self._background_tasks.append(asyncio.create_task(self._alert_evaluation_loop()))
        
        self.logger.info("MonitoringManager initialized.")

    async def _metric_processing_loop(self):
        """Example background task for processing/flushing metrics."""
        while self._running:
            try:
                # Process self.metrics, flush to external system, etc.
                if self.metrics:
                    # For demo, just log the count and clear
                    # self.logger.debug(f"Processing {len(self.metrics)} collected metrics.")
                    self.metrics.clear()
                await asyncio.sleep(5) # Process every 5 seconds
            except asyncio.CancelledError:
                self.logger.info("Metric processing loop cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in metric processing loop: {e}", exc_info=True)

    async def _alert_evaluation_loop(self):
        """Example background task for evaluating alert rules."""
        while self._running:
            try:
                # Evaluate rules based on self.metrics or other system state
                # Example: If a certain metric exceeds a threshold, raise an alert
                # self.logger.debug("Evaluating alert rules...")
                await asyncio.sleep(10) # Evaluate every 10 seconds
            except asyncio.CancelledError:
                self.logger.info("Alert evaluation loop cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in alert evaluation loop: {e}", exc_info=True)


    async def collect_metric(self, metric: Metric):
        self.metrics.append(metric)
        # In a real system, you'd push this to a time-series DB or metrics system
        # self.logger.debug(f"Metric collected: {metric.name}={metric.value}")

    async def raise_alert(self, alert: Alert):
        self.alerts.append(alert)
        self.logger.warning(f"ALERT! {alert.severity.value}: {alert.message} (Rule: {alert.rule_name})")
        # Here you'd trigger notifications (email, slack, etc.)

    async def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        for alert in self.alerts:
            if alert.id == alert_id and alert.status != AlertStatus.RESOLVED:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                alert.resolved_by = resolved_by
                self.logger.info(f"Alert {alert_id} resolved by {resolved_by}.")
                return True
        return False

    async def shutdown(self):
        """Stops all monitoring background tasks and performs cleanup."""
        self.logger.info("Shutting down MonitoringManager...")
        self._running = False # Signal internal loops to stop

        for task in self._background_tasks:
            task.cancel()
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
        
        # Any final flush of metrics or alerts
        if self.metrics:
            self.logger.info(f"Flushing {len(self.metrics)} remaining metrics.")
            self.metrics.clear()
        if self.alerts:
            self.logger.info(f"Reporting {len(self.alerts)} remaining alerts.")
            # In a real system, send these to a persistent store or notification service
        
        self.logger.info("MonitoringManager shut down.")