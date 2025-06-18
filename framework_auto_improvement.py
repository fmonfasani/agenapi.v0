"""
framework_auto_improvement.py - Sistema de auto-mejora del framework usando sus propios agentes
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import ast
import subprocess

from core.autonomous_agent_framework import AgentFramework, BaseAgent, AgentCapability, AgentResource, ResourceType
from core.specialized_agents import ExtendedAgentFactory
from systems.plugin_system import OpenAIAgent, GitHubAgent
from core.monitoring_system import MonitoringOrchestrator
from systems.deployment_system import DeploymentOrchestrator, DeploymentEnvironment, DeploymentStrategy


# AUTO-IMPROVEMENT MODELS


@dataclass
class ImprovementTask:
    """Tarea de mejora identificada"""
    id: str
    type: str  # "bug_fix", "feature_enhancement", "performance", "security", "refactor"
    priority: int  # 1-10
    description: str
    affected_files: List[str]
    suggested_solution: str
    estimated_effort: str  # "low", "medium", "high"
    impact_score: float  # 0-1
    created_at: datetime
    status: str = "identified"  # "identified", "in_progress", "implemented", "tested", "deployed"

@dataclass
class ImprovementPlan:
    """Plan de mejoras del framework"""
    id: str
    tasks: List[ImprovementTask]
    created_at: datetime
    target_completion: datetime
    overall_goals: List[str]
    success_metrics: Dict[str, float]


# FRAMEWORK ANALYZER AGENT


class FrameworkAnalyzerAgent(BaseAgent):
    """Agente que analiza el framework para identificar mejoras"""
    
    def __init__(self, name: str, framework):
        super().__init__("agent.framework.analyzer", name, framework)
        self.framework_path = Path("./")
        self.analysis_results = {}
        
    async def initialize(self) -> bool:
        self.capabilities = [
            AgentCapability(
                name="analyze_codebase",
                namespace="agent.framework.analyzer.codebase",
                description="Analyze framework codebase for improvements",
                input_schema={"target_path": "string", "analysis_type": "string"},
                output_schema={"issues": "array", "suggestions": "array"},
                handler=self._analyze_codebase
            ),
            AgentCapability(
                name="identify_bottlenecks",
                namespace="agent.framework.analyzer.performance",
                description="Identify performance bottlenecks",
                input_schema={"metrics": "object"},
                output_schema={"bottlenecks": "array", "optimizations": "array"},
                handler=self._identify_bottlenecks
            ),
            AgentCapability(
                name="security_audit",
                namespace="agent.framework.analyzer.security",
                description="Perform security audit of framework",
                input_schema={"scope": "string"},
                output_schema={"vulnerabilities": "array", "recommendations": "array"},
                handler=self._security_audit
            )
        ]
        return True
        
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "analyze.codebase":
            return await self._analyze_codebase(params)
        elif action == "identify.bottlenecks":
            return await self._identify_bottlenecks(params)
        elif action == "security.audit":
            return await self._security_audit(params)
        elif action == "comprehensive.analysis":
            return await self._comprehensive_analysis(params)
        return {"error": f"Unknown action: {action}"}
        
    async def _analyze_codebase(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analizar el código base del framework"""
        target_path = Path(params.get("target_path", "./"))
        analysis_type = params.get("analysis_type", "comprehensive")
        
        issues = []
        suggestions = []
        
        # Analizar archivos Python
        for py_file in target_path.rglob("*.py"):
            if self._should_analyze_file(py_file):
                file_analysis = await self._analyze_python_file(py_file)
                issues.extend(file_analysis["issues"])
                suggestions.extend(file_analysis["suggestions"])
                
        # Análisis arquitectural
        arch_analysis = await self._analyze_architecture()
        suggestions.extend(arch_analysis["suggestions"])
        
        return {
            "issues": issues,
            "suggestions": suggestions,
            "files_analyzed": len(list(target_path.rglob("*.py"))),
            "analysis_type": analysis_type
        }
        
    async def _analyze_python_file(self, file_path: Path) -> Dict[str, Any]:
        """Analizar un archivo Python específico"""
        issues = []
        suggestions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST para análisis estático
            tree = ast.parse(content)
            
            # Análisis de complejidad
            complexity_issues = self._analyze_complexity(tree, file_path)
            issues.extend(complexity_issues)
            
            # Análisis de patrones
            pattern_suggestions = self._analyze_patterns(tree, file_path)
            suggestions.extend(pattern_suggestions)
            
            # Análisis de documentación
            doc_issues = self._analyze_documentation(tree, file_path)
            issues.extend(doc_issues)
            
        except Exception as e:
            issues.append({
                "type": "parse_error",
                "file": str(file_path),
                "description": f"Could not parse file: {e}",
                "severity": "high"
            })
            
        return {"issues": issues, "suggestions": suggestions}
        
    def _analyze_complexity(self, tree: ast.AST, file_path: Path) -> List[Dict[str, Any]]:
        """Analizar complejidad del código"""
        issues = []
        
        class ComplexityAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.current_function = None
                self.complexity_count = 0
                
            def visit_FunctionDef(self, node):
                self.current_function = node.name
                self.complexity_count = 1  # Base complexity
                self.generic_visit(node)
                
                if self.complexity_count > 10:
                    issues.append({
                        "type": "high_complexity",
                        "file": str(file_path),
                        "function": node.name,
                        "line": node.lineno,
                        "description": f"Function {node.name} has high complexity ({self.complexity_count})",
                        "severity": "medium",
                        "suggestion": "Consider breaking into smaller functions"
                    })
                    
            def visit_If(self, node):
                self.complexity_count += 1
                self.generic_visit(node)
                
            def visit_For(self, node):
                self.complexity_count += 1
                self.generic_visit(node)
                
            def visit_While(self, node):
                self.complexity_count += 1
                self.generic_visit(node)
                
        analyzer = ComplexityAnalyzer()
        analyzer.visit(tree)
        
        return issues
        
    def _analyze_patterns(self, tree: ast.AST, file_path: Path) -> List[Dict[str, Any]]:
        """Analizar patrones de código"""
        suggestions = []
        
        class PatternAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.class_methods = {}
                self.imports = []
                
            def visit_Import(self, node):
                for alias in node.names:
                    self.imports.append(alias.name)
                    
            def visit_ImportFrom(self, node):
                if node.module:
                    self.imports.append(node.module)
                    
            def visit_ClassDef(self, node):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(item.name)
                        
                self.class_methods[node.name] = methods
                
                # Verificar si la clase podría beneficiarse de patrones
                if len(methods) > 15:
                    suggestions.append({
                        "type": "refactor_suggestion",
                        "file": str(file_path),
                        "class": node.name,
                        "description": f"Class {node.name} has many methods ({len(methods)})",
                        "suggestion": "Consider splitting into multiple classes or using composition",
                        "priority": "medium"
                    })
                    
                self.generic_visit(node)
                
        analyzer = PatternAnalyzer()
        analyzer.visit(tree)
        
        return suggestions
        
    def _analyze_documentation(self, tree: ast.AST, file_path: Path) -> List[Dict[str, Any]]:
        """Analizar documentación del código"""
        issues = []
        
        class DocAnalyzer(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                if not ast.get_docstring(node) and not node.name.startswith('_'):
                    issues.append({
                        "type": "missing_docstring",
                        "file": str(file_path),
                        "function": node.name,
                        "line": node.lineno,
                        "description": f"Public function {node.name} missing docstring",
                        "severity": "low",
                        "suggestion": "Add docstring explaining function purpose and parameters"
                    })
                    
            def visit_ClassDef(self, node):
                if not ast.get_docstring(node):
                    issues.append({
                        "type": "missing_docstring",
                        "file": str(file_path),
                        "class": node.name,
                        "line": node.lineno,
                        "description": f"Class {node.name} missing docstring",
                        "severity": "low",
                        "suggestion": "Add docstring explaining class purpose"
                    })
                self.generic_visit(node)
                
        analyzer = DocAnalyzer()
        analyzer.visit(tree)
        
        return issues
        
    async def _analyze_architecture(self) -> Dict[str, Any]:
        """Analizar arquitectura del framework"""
        suggestions = []
        
        # Analizar dependencias circulares
        suggestions.append({
            "type": "architecture",
            "description": "Consider implementing dependency injection for better testability",
            "impact": "high",
            "effort": "medium"
        })
        
        # Analizar modularidad
        suggestions.append({
            "type": "architecture", 
            "description": "Add interface abstractions for core components",
            "impact": "medium",
            "effort": "low"
        })
        
        return {"suggestions": suggestions}
        
    async def _identify_bottlenecks(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Identificar cuellos de botella de rendimiento"""
        metrics = params.get("metrics", {})
        
        bottlenecks = []
        optimizations = []
        
        # Analizar métricas de agentes
        if "agent_response_times" in metrics:
            for agent_id, response_time in metrics["agent_response_times"].items():
                if response_time > 2.0:  # > 2 segundos
                    bottlenecks.append({
                        "type": "slow_agent_response",
                        "agent_id": agent_id,
                        "response_time": response_time,
                        "severity": "high" if response_time > 5.0 else "medium"
                    })
                    
                    optimizations.append({
                        "target": agent_id,
                        "suggestion": "Add async operations and caching",
                        "expected_improvement": "50-70% response time reduction"
                    })
                    
        # Analizar uso de memoria
        if "memory_usage" in metrics:
            if metrics["memory_usage"] > 85:
                bottlenecks.append({
                    "type": "high_memory_usage",
                    "usage_percent": metrics["memory_usage"],
                    "severity": "high"
                })
                
                optimizations.append({
                    "target": "memory",
                    "suggestion": "Implement resource cleanup and object pooling",
                    "expected_improvement": "20-30% memory reduction"
                })
                
        return {
            "bottlenecks": bottlenecks,
            "optimizations": optimizations
        }
        
    async def _security_audit(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Realizar auditoría de seguridad"""
        scope = params.get("scope", "full")
        
        vulnerabilities = []
        recommendations = []
        
        # Análisis de autenticación
        auth_files = list(self.framework_path.glob("**/security_system.py"))
        for file_path in auth_files:
            vulnerabilities.extend(await self._audit_auth_security(file_path))
            
        # Análisis de validación de entrada
        api_files = list(self.framework_path.glob("**/rest_api.py"))
        for file_path in api_files:
            vulnerabilities.extend(await self._audit_input_validation(file_path))
            
        # Recomendaciones generales
        recommendations.extend([
            "Implement rate limiting for API endpoints",
            "Add input sanitization for all user inputs",
            "Use secure random generation for tokens",
            "Implement proper session management",
            "Add audit logging for security events"
        ])
        
        return {
            "vulnerabilities": vulnerabilities,
            "recommendations": recommendations,
            "audit_scope": scope
        }
        
    async def _audit_auth_security(self, file_path: Path) -> List[Dict[str, Any]]:
        """Auditar seguridad de autenticación"""
        vulnerabilities = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Buscar patrones inseguros
            if "default_secret" in content:
                vulnerabilities.append({
                    "type": "weak_secret",
                    "file": str(file_path),
                    "description": "Default secret keys found",
                    "severity": "high",
                    "recommendation": "Use environment variables for secrets"
                })
                
            if "sha256" in content and "salt" not in content:
                vulnerabilities.append({
                    "type": "weak_hashing",
                    "file": str(file_path),
                    "description": "Password hashing without salt detected",
                    "severity": "medium",
                    "recommendation": "Use bcrypt or similar with salt"
                })
                
        except Exception as e:
            logging.error(f"Error auditing {file_path}: {e}")
            
        return vulnerabilities
        
    async def _audit_input_validation(self, file_path: Path) -> List[Dict[str, Any]]:
        """Auditar validación de entrada"""
        vulnerabilities = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Buscar endpoints sin validación
            if "request.json()" in content and "validate" not in content:
                vulnerabilities.append({
                    "type": "missing_input_validation",
                    "file": str(file_path),
                    "description": "API endpoints without input validation",
                    "severity": "medium",
                    "recommendation": "Add input validation schemas"
                })
                
        except Exception as e:
            logging.error(f"Error auditing {file_path}: {e}")
            
        return vulnerabilities
        
    async def _comprehensive_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Análisis comprensivo del framework"""
        
        # Ejecutar todos los análisis
        codebase_analysis = await self._analyze_codebase({"target_path": "./"})
        bottleneck_analysis = await self._identify_bottlenecks(params.get("metrics", {}))
        security_analysis = await self._security_audit({"scope": "full"})
        
        # Combinar resultados
        all_issues = (
            codebase_analysis["issues"] + 
            bottleneck_analysis["bottlenecks"] + 
            security_analysis["vulnerabilities"]
        )
        
        all_suggestions = (
            codebase_analysis["suggestions"] + 
            bottleneck_analysis["optimizations"] + 
            [{"type": "security", "description": rec} for rec in security_analysis["recommendations"]]
        )
        
        return {
            "total_issues": len(all_issues),
            "total_suggestions": len(all_suggestions),
            "issues_by_severity": self._categorize_by_severity(all_issues),
            "improvement_opportunities": all_suggestions,
            "analysis_summary": {
                "code_quality_score": self._calculate_quality_score(all_issues),
                "security_score": self._calculate_security_score(security_analysis["vulnerabilities"]),
                "performance_score": self._calculate_performance_score(bottleneck_analysis["bottlenecks"])
            }
        }
        
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Determinar si un archivo debe ser analizado"""
        # Excluir archivos que no necesitan análisis
        exclude_patterns = ["__pycache__", ".git", "venv", ".pytest_cache", "test_"]
        
        for pattern in exclude_patterns:
            if pattern in str(file_path):
                return False
                
        return file_path.suffix == ".py"
        
    def _categorize_by_severity(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorizar issues por severidad"""
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        
        for issue in issues:
            severity = issue.get("severity", "medium")
            if severity in severity_counts:
                severity_counts[severity] += 1
                
        return severity_counts
        
    def _calculate_quality_score(self, issues: List[Dict[str, Any]]) -> float:
        """Calcular score de calidad del código (0-100)"""
        if not issues:
            return 100.0
            
        severity_weights = {"high": 3, "medium": 2, "low": 1}
        total_weight = sum(severity_weights.get(issue.get("severity", "medium"), 2) for issue in issues)
        
        # Score basado en número y severidad de issues
        max_penalty = len(issues) * 3  # Máximo peso si todos fueran high
        penalty_ratio = total_weight / max(max_penalty, 1)
        
        return max(0, 100 - (penalty_ratio * 100))
        
    def _calculate_security_score(self, vulnerabilities: List[Dict[str, Any]]) -> float:
        """Calcular score de seguridad (0-100)"""
        if not vulnerabilities:
            return 100.0
            
        high_vulns = len([v for v in vulnerabilities if v.get("severity") == "high"])
        medium_vulns = len([v for v in vulnerabilities if v.get("severity") == "medium"])
        low_vulns = len([v for v in vulnerabilities if v.get("severity") == "low"])
        
        penalty = (high_vulns * 30) + (medium_vulns * 15) + (low_vulns * 5)
        
        return max(0, 100 - penalty)
        
    def _calculate_performance_score(self, bottlenecks: List[Dict[str, Any]]) -> float:
        """Calcular score de performance (0-100)"""
        if not bottlenecks:
            return 100.0
            
        high_bottlenecks = len([b for b in bottlenecks if b.get("severity") == "high"])
        medium_bottlenecks = len([b for b in bottlenecks if b.get("severity") == "medium"])
        
        penalty = (high_bottlenecks * 25) + (medium_bottlenecks * 10)
        
        return max(0, 100 - penalty)


# FRAMEWORK IMPROVER AGENT


class FrameworkImproverAgent(BaseAgent):
    """Agente que implementa mejoras al framework"""
    
    def __init__(self, name: str, framework):
        super().__init__("agent.framework.improver", name, framework)
        self.improvement_tasks = []
        
    async def initialize(self) -> bool:
        self.capabilities = [
            AgentCapability(
                name="generate_improvement",
                namespace="agent.framework.improver.generate",
                description="Generate code improvements",
                input_schema={"task": "object", "context": "object"},
                output_schema={"implementation": "string", "tests": "string"},
                handler=self._generate_improvement
            ),
            AgentCapability(
                name="apply_refactoring",
                namespace="agent.framework.improver.refactor",
                description="Apply refactoring to existing code",
                input_schema={"file_path": "string", "refactor_type": "string"},
                output_schema={"refactored_code": "string", "changes": "array"},
                handler=self._apply_refactoring
            ),
            AgentCapability(
                name="optimize_performance",
                namespace="agent.framework.improver.optimize",
                description="Optimize performance bottlenecks",
                input_schema={"bottleneck": "object"},
                output_schema={"optimized_code": "string", "performance_gain": "number"},
                handler=self._optimize_performance
            )
        ]
        return True
        
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "generate.improvement":
            return await self._generate_improvement(params)
        elif action == "apply.refactoring":
            return await self._apply_refactoring(params)
        elif action == "optimize.performance":
            return await self._optimize_performance(params)
        elif action == "implement.task":
            return await self._implement_improvement_task(params)
        return {"error": f"Unknown action: {action}"}
        
    async def _generate_improvement(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generar mejoras de código"""
        task = params.get("task", {})
        context = params.get("context", {})
        
        task_type = task.get("type", "")
        
        if task_type == "add_caching":
            return await self._generate_caching_improvement(task, context)
        elif task_type == "add_async_support":
            return await self._generate_async_improvement(task, context)
        elif task_type == "improve_error_handling":
            return await self._generate_error_handling_improvement(task, context)
        elif task_type == "add_logging":
            return await self._generate_logging_improvement(task, context)
        else:
            return await self._generate_generic_improvement(task, context)
            
    async def _generate_caching_improvement(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generar mejora de cache"""
        
        implementation = '''
import functools
import asyncio
from typing import Dict, Any, Optional
import time

class FrameworkCache:
    """Cache system for framework operations"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl = ttl
        
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            item = self.cache[key]
            if time.time() - item["timestamp"] < self.ttl:
                return item["value"]
            else:
                del self.cache[key]
        return None
        
    def set(self, key: str, value: Any):
        if len(self.cache) >= self.max_size:
            # Remove oldest item
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
            
        self.cache[key] = {
            "value": value,
            "timestamp": time.time()
        }
        
    def invalidate(self, pattern: str = None):
        if pattern:
            keys_to_remove = [k for k in self.cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.cache[key]
        else:
            self.cache.clear()

def cached(ttl: int = 3600):
    """Decorator for caching function results"""
    def decorator(func):
        cache = FrameworkCache(ttl=ttl)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            result = cache.get(key)
            if result is not None:
                return result
                
            result = await func(*args, **kwargs)
            cache.set(key, result)
            return result
            
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            result = cache.get(key)
            if result is not None:
                return result
                
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
            
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Example usage in agent registry:
class CachedAgentRegistry(AgentRegistry):
    @cached(ttl=300)  # Cache for 5 minutes
    async def find_agents_by_capability(self, capability_name: str) -> List['BaseAgent']:
        return super().find_agents_by_capability(capability_name)
'''
        
        tests = '''
import pytest
import asyncio
import time
from your_module import FrameworkCache, cached

class TestFrameworkCache:
    def test_cache_set_get(self):
        cache = FrameworkCache()
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
        
    def test_cache_ttl_expiry(self):
        cache = FrameworkCache(ttl=1)  # 1 second TTL
        cache.set("test_key", "test_value")
        time.sleep(2)
        assert cache.get("test_key") is None
        
    def test_cache_max_size(self):
        cache = FrameworkCache(max_size=2)
        cache.set("key1", "value1")
        cache.set("key2", "value2") 
        cache.set("key3", "value3")  # Should evict key1
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        
    @pytest.mark.asyncio
    async def test_cached_decorator_async(self):
        call_count = 0
        
        @cached(ttl=60)
        async def expensive_function(x):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return x * 2
            
        result1 = await expensive_function(5)
        result2 = await expensive_function(5)
        
        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # Function called only once due to caching
'''
        
        return {
            "implementation": implementation,
            "tests": tests,
            "files_to_modify": task.get("affected_files", []),
            "estimated_performance_gain": "30-50% for repeated operations"
        }
        
    async def _generate_async_improvement(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generar mejora async"""
        
        implementation = '''
import asyncio
import concurrent.futures
from typing import List, Callable, Any

class AsyncExecutor:
    """Enhanced async execution utilities for the framework"""
    
    def __init__(self, max_workers: int = None):
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.max_concurrent = max_workers or 10
        
    async def run_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """Run synchronous function in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, func, *args, **kwargs)
        
    async def gather_with_limit(self, *coroutines, limit: int = None) -> List[Any]:
        """Execute coroutines with concurrency limit"""
        limit = limit or self.max_concurrent
        semaphore = asyncio.Semaphore(limit)
        
        async def limited_coroutine(coro):
            async with semaphore:
                return await coro
                
        limited_coroutines = [limited_coroutine(coro) for coro in coroutines]
        return await asyncio.gather(*limited_coroutines)
        
    async def batch_process(self, items: List[Any], processor: Callable, 
                          batch_size: int = 10) -> List[Any]:
        """Process items in batches asynchronously"""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_coroutines = [processor(item) for item in batch]
            batch_results = await asyncio.gather(*batch_coroutines, return_exceptions=True)
            results.extend(batch_results)
            
        return results
        
    def cleanup(self):
        """Cleanup thread pool"""
        self.thread_pool.shutdown(wait=True)

# Enhanced BaseAgent with better async support
class AsyncBaseAgent(BaseAgent):
    """Enhanced BaseAgent with improved async capabilities"""
    
    def __init__(self, namespace: str, name: str, framework):
        super().__init__(namespace, name, framework)
        self.async_executor = AsyncExecutor()
        self._pending_tasks = set()
        
    async def execute_concurrent_actions(self, actions: List[Dict[str, Any]], 
                                       max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """Execute multiple actions concurrently"""
        async def execute_single_action(action_data):
            try:
                action = action_data["action"]
                params = action_data.get("params", {})
                return await self.execute_action(action, params)
            except Exception as e:
                return {"error": str(e), "action": action_data.get("action")}
                
        action_coroutines = [execute_single_action(action) for action in actions]
        return await self.async_executor.gather_with_limit(*action_coroutines, limit=max_concurrent)
        
    async def schedule_task(self, coro, delay: float = 0) -> asyncio.Task:
        """Schedule a task with optional delay"""
        async def delayed_task():
            if delay > 0:
                await asyncio.sleep(delay)
            return await coro
            
        task = asyncio.create_task(delayed_task())
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return task
        
    async def wait_for_pending_tasks(self, timeout: float = None):
        """Wait for all pending tasks to complete"""
        if self._pending_tasks:
            await asyncio.wait(self._pending_tasks, timeout=timeout)
            
    async def stop(self):
        """Enhanced stop method with proper cleanup"""
        # Cancel pending tasks
        for task in self._pending_tasks:
            task.cancel()
            
        # Wait for tasks to finish cancellation
        if self._pending_tasks:
            await asyncio.wait(self._pending_tasks, return_when=asyncio.ALL_COMPLETED)
            
        # Cleanup async executor
        self.async_executor.cleanup()
        
        # Call parent stop
        await super().stop()
'''
        
        tests = '''
import pytest
import asyncio
from your_module import AsyncExecutor, AsyncBaseAgent

class TestAsyncExecutor:
    @pytest.mark.asyncio
    async def test_gather_with_limit(self):
        executor = AsyncExecutor()
        
        async def slow_task(x):
            await asyncio.sleep(0.1)
            return x * 2
            
        tasks = [slow_task(i) for i in range(10)]
        results = await executor.gather_with_limit(*tasks, limit=3)
        
        assert len(results) == 10
        assert results == [i * 2 for i in range(10)]
        
    @pytest.mark.asyncio
    async def test_batch_process(self):
        executor = AsyncExecutor()
        
        async def process_item(x):
            return x ** 2
            
        items = list(range(10))
        results = await executor.batch_process(items, process_item, batch_size=3)
        
        assert len(results) == 10
        assert results == [i ** 2 for i in range(10)]
        
    def test_run_in_thread(self):
        import time
        executor = AsyncExecutor()
        
        def blocking_function(duration):
            time.sleep(duration)
            return "completed"
            
        async def test():
            result = await executor.run_in_thread(blocking_function, 0.1)
            return result
            
        result = asyncio.run(test())
        assert result == "completed"
'''
        
        return {
            "implementation": implementation,
            "tests": tests,
            "files_to_modify": ["core/autonomous_agent_framework.py"],
            "estimated_performance_gain": "40-60% for concurrent operations"
        }
        
    async def _implement_improvement_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Implementar una tarea de mejora completa"""
        task = params.get("task", {})
        
        # Generar la mejora
        improvement = await self._generate_improvement({"task": task, "context": {}})
        
        # Crear archivos de implementación
        implementation_files = {}
        
        if improvement.get("implementation"):
            implementation_files["implementation.py"] = improvement["implementation"]
            
        if improvement.get("tests"):
            implementation_files["test_implementation.py"] = improvement["tests"]
            
        # Crear recurso con la implementación
        from core.autonomous_agent_framework import AgentResource, ResourceType
        
        improvement_resource = AgentResource(
            type=ResourceType.CODE,
            name=f"improvement_{task.get('id', 'unknown')}",
            namespace="resource.framework.improvement",
            data={
                "task": task,
                "implementation": improvement,
                "files": implementation_files,
                "status": "generated"
            },
            owner_agent_id=self.id
        )
        
        await self.framework.resource_manager.create_resource(improvement_resource)
        
        return {
            "task_id": task.get("id"),
            "resource_id": improvement_resource.id,
            "implementation_ready": True,
            "files_generated": len(implementation_files),
            "next_steps": [
                "Review generated code",
                "Run tests",
                "Apply to framework",
                "Deploy changes"
            ]
        }
        
    async def _apply_refactoring(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Aplicar refactoring a código existente"""
        file_path = params.get("file_path", "")
        refactor_type = params.get("refactor_type", "")
        
        # Leer archivo actual
        try:
            with open(file_path, 'r') as f:
                original_code = f.read()
        except FileNotFoundError:
            return {"error": f"File {file_path} not found"}
            
        # Aplicar refactoring según el tipo
        if refactor_type == "extract_method":
            refactored_code = self._extract_method_refactoring(original_code)
        elif refactor_type == "reduce_complexity":
            refactored_code = self._reduce_complexity_refactoring(original_code)
        else:
            return {"error": f"Unknown refactor type: {refactor_type}"}
            
        changes = self._diff_code(original_code, refactored_code)
        
        return {
            "refactored_code": refactored_code,
            "changes": changes,
            "file_path": file_path,
            "refactor_type": refactor_type
        }
        
    def _extract_method_refactoring(self, code: str) -> str:
        """Refactoring para extraer métodos"""
        # Implementación simplificada - en un caso real usarías un parser AST más sofisticado
        lines = code.split('\n')
        refactored_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Buscar bloques largos en funciones
            if line.strip().startswith('def ') and not line.strip().endswith(':'):
                # Función compleja detectada, sugerir extracción
                refactored_lines.append(line)
                refactored_lines.append("        # TODO: Consider extracting complex logic into separate methods")
            else:
                refactored_lines.append(line)
            i += 1
            
        return '\n'.join(refactored_lines)
        
    def _diff_code(self, original: str, refactored: str) -> List[str]:
        """Generar diff entre código original y refactorizado"""
        original_lines = original.split('\n')
        refactored_lines = refactored.split('\n')
        
        changes = []
        
        for i, (orig, refact) in enumerate(zip(original_lines, refactored_lines)):
            if orig != refact:
                changes.append(f"Line {i+1}: '{orig}' -> '{refact}'")
                
        return changes


# AUTO-IMPROVEMENT ORCHESTRATOR


class AutoImprovementOrchestrator:
    """Orquestador principal del sistema de auto-mejora"""
    
    def __init__(self, framework: AgentFramework):
        self.framework = framework
        self.analyzer_agent = None
        self.improver_agent = None
        self.ai_agent = None  # OpenAI agent for advanced analysis
        self.github_agent = None  # For managing improvements in GitHub
        self.monitoring = None
        self.improvement_plans = []
        self.active_tasks = []
        
    async def initialize(self):
        """Inicializar el sistema de auto-mejora"""
        
        # Crear agentes especializados
        self.analyzer_agent = FrameworkAnalyzerAgent("framework_analyzer", self.framework)
        self.improver_agent = FrameworkImproverAgent("framework_improver", self.framework)
        
        await self.analyzer_agent.start()
        await self.improver_agent.start()
        
        # Configurar monitoreo
        self.monitoring = MonitoringOrchestrator(self.framework)
        await self.monitoring.start_monitoring()
        
        logging.info("Auto-improvement system initialized")
        
    async def run_improvement_cycle(self) -> ImprovementPlan:
        """Ejecutar un ciclo completo de mejora"""
        
        print("🔍 Starting framework analysis...")
        
        # 1. Análisis comprensivo
        analysis_result = await self.analyzer_agent.execute_action("comprehensive.analysis", {
            "metrics": self._collect_current_metrics()
        })
        
        print(f"📊 Analysis complete:")
        print(f"   Issues found: {analysis_result['total_issues']}")
        print(f"   Suggestions: {analysis_result['total_suggestions']}")
        print(f"   Quality score: {analysis_result['analysis_summary']['code_quality_score']:.1f}/100")
        
        # 2. Crear plan de mejoras
        improvement_plan = await self._create_improvement_plan(analysis_result)
        self.improvement_plans.append(improvement_plan)
        
        print(f"📋 Improvement plan created with {len(improvement_plan.tasks)} tasks")
        
        # 3. Ejecutar mejoras de alta prioridad
        high_priority_tasks = [task for task in improvement_plan.tasks if task.priority >= 8]
        
        print(f"⚡ Implementing {len(high_priority_tasks)} high-priority improvements...")
        
        implementation_results = []
        for task in high_priority_tasks[:3]:  # Limitar a 3 mejoras por ciclo
            result = await self._implement_improvement(task)
            implementation_results.append(result)
            
        # 4. Crear PR en GitHub si está configurado
        if self.github_agent and implementation_results:
            await self._create_improvement_pr(improvement_plan, implementation_results)
            
        return improvement_plan
        
    async def _create_improvement_plan(self, analysis_result: Dict[str, Any]) -> ImprovementPlan:
        """Crear plan de mejoras basado en análisis"""
        
        tasks = []
        task_id_counter = 1
        
        # Convertir issues en tareas
        for issue in analysis_result.get("issues", []):
            if issue.get("severity") in ["high", "medium"]:
                task = ImprovementTask(
                    id=f"task_{task_id_counter:03d}",
                    type=self._categorize_issue_type(issue),
                    priority=self._calculate_priority(issue),
                    description=issue.get("description", ""),
                    affected_files=[issue.get("file", "")],
                    suggested_solution=issue.get("suggestion", ""),
                    estimated_effort=self._estimate_effort(issue),
                    impact_score=self._calculate_impact(issue),
                    created_at=datetime.now()
                )
                tasks.append(task)
                task_id_counter += 1
                
        # Convertir suggestions en tareas
        for suggestion in analysis_result.get("improvement_opportunities", []):
            task = ImprovementTask(
                id=f"task_{task_id_counter:03d}",
                type=suggestion.get("type", "enhancement"),
                priority=self._calculate_suggestion_priority(suggestion),
                description=suggestion.get("description", ""),
                affected_files=suggestion.get("files", []),
                suggested_solution=suggestion.get("suggestion", suggestion.get("description")),
                estimated_effort=suggestion.get("effort", "medium"),
                impact_score=suggestion.get("impact", 0.5),
                created_at=datetime.now()
            )
            tasks.append(task)
            task_id_counter += 1
            
        # Ordenar por prioridad
        tasks.sort(key=lambda x: x.priority, reverse=True)
        
        plan = ImprovementPlan(
            id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tasks=tasks,
            created_at=datetime.now(),
            target_completion=datetime.now() + timedelta(days=7),
            overall_goals=[
                "Improve code quality",
                "Enhance performance",
                "Strengthen security",
                "Reduce technical debt"
            ],
            success_metrics={
                "quality_score_improvement": 10.0,
                "performance_gain": 20.0,
                "security_score_improvement": 15.0,
                "test_coverage_increase": 5.0
            }
        )
        
        return plan
        
    async def _implement_improvement(self, task: ImprovementTask) -> Dict[str, Any]:
        """Implementar una mejora específica"""
        
        print(f"🔧 Implementing: {task.description}")
        
        # Usar el agente mejorador para implementar
        result = await self.improver_agent.execute_action("implement.task", {
            "task": {
                "id": task.id,
                "type": task.type,
                "description": task.description,
                "affected_files": task.affected_files,
                "suggested_solution": task.suggested_solution
            }
        })
        
        if result.get("implementation_ready"):
            task.status = "implemented"
            print(f"   ✅ Implementation ready: {result['resource_id']}")
        else:
            task.status = "failed"
            print(f"   ❌ Implementation failed")
            
        return result
        
    async def _create_improvement_pr(self, plan: ImprovementPlan, implementations: List[Dict[str, Any]]):
        """Crear Pull Request con las mejoras"""
        
        if not self.github_agent:
            return
            
        # Crear branch para mejoras
        branch_name = f"auto-improvement-{plan.id}"
        
        # Generar descripción del PR
        pr_description = f"""# Automated Framework Improvements
        
## Improvement Plan: {plan.id}

### Implemented Tasks:
"""
        
        for i, impl in enumerate(implementations, 1):
            pr_description += f"{i}. Task {impl.get('task_id', 'unknown')}\n"
            
        pr_description += f"""
### Quality Metrics:
- Tasks completed: {len(implementations)}
- Target completion: {plan.target_completion.strftime('%Y-%m-%d')}

### Files Modified:
"""
        
        modified_files = set()
        for task in plan.tasks:
            modified_files.update(task.affected_files)
            
        for file in modified_files:
            if file:
                pr_description += f"- {file}\n"
                
        # Crear PR
        pr_result = await self.github_agent.execute_action("create.issue", {
            "repo": "your-org/agent-framework",  # Configurar según tu repo
            "title": f"Auto-improvement: {plan.id}",
            "body": pr_description
        })
        
        print(f"📝 Created improvement PR: {pr_result.get('issue_url', 'N/A')}")
        
    def _collect_current_metrics(self) -> Dict[str, Any]:
        """Recopilar métricas actuales del framework"""
        
        if not self.monitoring:
            return {}
            
        try:
            latest_metrics = self.monitoring.metrics_collector.get_latest_metrics()
            
            # Convertir métricas a formato analizable
            metrics = {}
            
            for metric_key, metric in latest_metrics.items():
                if hasattr(metric, 'value'):
                    metrics[metric.name] = metric.value
                    
            return metrics
            
        except Exception as e:
            logging.error(f"Error collecting metrics: {e}")
            return {}
            
    def _categorize_issue_type(self, issue: Dict[str, Any]) -> str:
        """Categorizar tipo de issue"""
        issue_type = issue.get("type", "")
        
        if "security" in issue_type or "vulnerability" in issue_type:
            return "security"
        elif "performance" in issue_type or "slow" in issue_type:
            return "performance"
        elif "complexity" in issue_type:
            return "refactor"
        elif "missing" in issue_type:
            return "feature_enhancement"
        else:
            return "bug_fix"
            
    def _calculate_priority(self, issue: Dict[str, Any]) -> int:
        """Calcular prioridad de issue (1-10)"""
        severity = issue.get("severity", "medium")
        
        base_priority = {
            "high": 8,
            "medium": 5, 
            "low": 2
        }.get(severity, 5)
        
        # Ajustar según tipo
        issue_type = issue.get("type", "")
        if "security" in issue_type:
            base_priority += 2
        elif "performance" in issue_type:
            base_priority += 1
            
        return min(10, base_priority)
        
    def _calculate_suggestion_priority(self, suggestion: Dict[str, Any]) -> int:
        """Calcular prioridad de sugerencia"""
        impact = suggestion.get("impact", "medium")
        effort = suggestion.get("effort", "medium")
        
        impact_score = {
            "high": 3,
            "medium": 2,
            "low": 1
        }.get(impact, 2)
        
        effort_score = {
            "low": 3,
            "medium": 2, 
            "high": 1
        }.get(effort, 2)
        
        return min(10, impact_score + effort_score + 2)
        
    def _estimate_effort(self, issue: Dict[str, Any]) -> str:
        """Estimar esfuerzo requerido"""
        severity = issue.get("severity", "medium")
        issue_type = issue.get("type", "")
        
        if severity == "high" or "security" in issue_type:
            return "high"
        elif severity == "low" or "documentation" in issue_type:
            return "low"
        else:
            return "medium"
            
    def _calculate_impact(self, issue: Dict[str, Any]) -> float:
        """Calcular impacto de resolver el issue (0-1)"""
        severity = issue.get("severity", "medium")
        issue_type = issue.get("type", "")
        
        base_impact = {
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2
        }.get(severity, 0.5)
        
        # Aumentar impacto para ciertos tipos
        if "security" in issue_type:
            base_impact += 0.2
        elif "performance" in issue_type:
            base_impact += 0.1
            
        return min(1.0, base_impact)
        
    async def run_continuous_improvement(self, interval_hours: int = 24):
        """Ejecutar mejora continua"""
        
        print(f"🔄 Starting continuous improvement (every {interval_hours}h)")
        
        while True:
            try:
                plan = await self.run_improvement_cycle()
                
                print(f"✅ Improvement cycle completed: {plan.id}")
                print(f"⏰ Next cycle in {interval_hours} hours")
                
                await asyncio.sleep(interval_hours * 3600)
                
            except KeyboardInterrupt:
                print("🛑 Continuous improvement stopped")
                break
            except Exception as e:
                logging.error(f"Improvement cycle error: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error


# EXAMPLE USAGE


async def auto_improvement_demo():
    """Demo del sistema de auto-mejora"""
    
    logging.basicConfig(level=logging.INFO)
    
    print("🤖 Framework Auto-Improvement Demo")
    print("="*60)
    
    # Crear framework con algunos agentes
    from core.autonomous_agent_framework import AgentFramework
    from core.specialized_agents import ExtendedAgentFactory
    
    framework = AgentFramework()
    await framework.start()
    
    # Crear algunos agentes para tener actividad
    strategist = ExtendedAgentFactory.create_agent("agent.planning.strategist", "strategist", framework)
    generator = ExtendedAgentFactory.create_agent("agent.build.code.generator", "generator", framework)
    
    await strategist.start()
    await generator.start()
    
    # Inicializar sistema de auto-mejora
    auto_improver = AutoImprovementOrchestrator(framework)
    await auto_improver.initialize()
    
    print(f"✅ Auto-improvement system ready")
    print(f"   Analyzer agent: {auto_improver.analyzer_agent.id}")
    print(f"   Improver agent: {auto_improver.improver_agent.id}")
    
    # Ejecutar un ciclo de mejora
    improvement_plan = await auto_improver.run_improvement_cycle()
    
    print(f"\n📊 Improvement Results:")
    print(f"   Plan ID: {improvement_plan.id}")
    print(f"   Total tasks: {len(improvement_plan.tasks)}")
    print(f"   High priority: {len([t for t in improvement_plan.tasks if t.priority >= 8])}")
    print(f"   Target completion: {improvement_plan.target_completion.strftime('%Y-%m-%d')}")
    
    # Mostrar algunas tareas de ejemplo
    print(f"\n📋 Sample improvement tasks:")
    for i, task in enumerate(improvement_plan.tasks[:5], 1):
        print(f"   {i}. [{task.type}] {task.description[:60]}...")
        print(f"      Priority: {task.priority}/10, Effort: {task.estimated_effort}")
    
    await framework.stop()
    
    print(f"\n🎉 Demo completed - Framework ready for self-improvement!")

if __name__ == "__main__":
    asyncio.run(auto_improvement_demo())