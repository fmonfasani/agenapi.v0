#!/usr/bin/env python3
"""
auto_improve_framework.py - Sistema de Auto-Mejora del Framework
================================================================

SCRIPT COMPLETAMENTE FUNCIONAL - No requiere imports complejos
"""

import os
import sys
import asyncio
from pathlib import Path
import json
from datetime import datetime
import ast
import re


# ANALIZADOR DE FRAMEWORK SIMPLE


class FrameworkAnalyzer:
    """Analizador simple pero efectivo del framework"""
    
    def __init__(self, base_path="."):
        self.base_path = Path(base_path)
        self.results = {
            "files_analyzed": 0,
            "total_lines": 0,
            "issues": [],
            "suggestions": [],
            "stats": {}
        }
        
    def analyze(self):
        """Ejecutar análisis completo"""
        print("🔍 ANALIZANDO FRAMEWORK...")
        print("=" * 50)
        
        # Encontrar archivos Python
        python_files = self._find_python_files()
        print(f"📁 Encontrados {len(python_files)} archivos Python")
        
        # Analizar cada archivo
        for file_path in python_files:
            self._analyze_file(file_path)
            
        # Calcular estadísticas
        self._calculate_stats()
        
        # Mostrar resultados
        self._display_results()
        
        return self.results
        
    def _find_python_files(self):
        """Encontrar archivos Python en el proyecto"""
        python_files = []
        
        for file_path in self.base_path.rglob("*.py"):
            # Excluir directorios que no queremos analizar
            if self._should_analyze_file(file_path):
                python_files.append(file_path)
                
        return python_files
        
    def _should_analyze_file(self, file_path):
        """Determinar si debemos analizar este archivo"""
        exclude_patterns = [
            "__pycache__",
            ".git",
            "venv",
            "env",
            ".pytest_cache",
            "node_modules",
            "build",
            "dist"
        ]
        
        path_str = str(file_path)
        return not any(pattern in path_str for pattern in exclude_patterns)
        
    def _analyze_file(self, file_path):
        """Analizar un archivo específico"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            self.results["files_analyzed"] += 1
            lines = content.split('\n')
            self.results["total_lines"] += len(lines)
            
            # Análisis básico de líneas
            self._analyze_lines(file_path, lines)
            
            # Análisis AST si es posible
            try:
                tree = ast.parse(content)
                self._analyze_ast(file_path, tree)
            except SyntaxError:
                self.results["issues"].append({
                    "type": "syntax_error",
                    "file": str(file_path),
                    "severity": "high",
                    "description": f"Error de sintaxis en {file_path.name}"
                })
                
        except Exception as e:
            self.results["issues"].append({
                "type": "read_error",
                "file": str(file_path),
                "severity": "medium",
                "description": f"No se pudo leer el archivo: {e}"
            })
            
    def _analyze_lines(self, file_path, lines):
        """Análisis línea por línea"""
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # 1. Buscar print() en lugar de logging
            if 'print(' in line and not stripped.startswith('#'):
                self.results["issues"].append({
                    "type": "print_usage",
                    "file": str(file_path),
                    "line": i,
                    "severity": "low",
                    "description": f"Uso de print() en línea {i}",
                    "suggestion": "Usar logging en lugar de print()"
                })
                
            # 2. Buscar TODO/FIXME
            if any(keyword in stripped.upper() for keyword in ['TODO', 'FIXME', 'HACK']):
                self.results["suggestions"].append({
                    "type": "todo_found",
                    "file": str(file_path),
                    "line": i,
                    "description": f"TODO/FIXME encontrado: {stripped}",
                    "priority": "medium"
                })
                
            # 3. Líneas muy largas
            if len(line) > 120:
                self.results["issues"].append({
                    "type": "long_line",
                    "file": str(file_path),
                    "line": i,
                    "severity": "low",
                    "description": f"Línea muy larga ({len(line)} caracteres)",
                    "suggestion": "Dividir en múltiples líneas"
                })
                
            # 4. Importaciones innecesarias
            if stripped.startswith('import ') and '*' in stripped:
                self.results["issues"].append({
                    "type": "wildcard_import",
                    "file": str(file_path),
                    "line": i,
                    "severity": "medium",
                    "description": "Importación con wildcard (*)",
                    "suggestion": "Importar elementos específicos"
                })
                
    def _analyze_ast(self, file_path, tree):
        """Análisis usando AST (Abstract Syntax Tree)"""
        
        class CodeAnalyzer(ast.NodeVisitor):
            def __init__(self, file_path, results):
                self.file_path = file_path
                self.results = results
                self.current_class = None
                self.current_function = None
                
            def visit_FunctionDef(self, node):
                # Verificar funciones muy largas
                if len(node.body) > 30:
                    self.results["issues"].append({
                        "type": "long_function",
                        "file": str(self.file_path),
                        "line": node.lineno,
                        "function": node.name,
                        "severity": "medium",
                        "description": f"Función {node.name} muy larga ({len(node.body)} statements)",
                        "suggestion": "Considerar dividir en funciones más pequeñas"
                    })
                    
                # Verificar docstrings
                if not ast.get_docstring(node) and not node.name.startswith('_'):
                    self.results["issues"].append({
                        "type": "missing_docstring",
                        "file": str(self.file_path),
                        "line": node.lineno,
                        "function": node.name,
                        "severity": "low",
                        "description": f"Función pública {node.name} sin docstring",
                        "suggestion": "Añadir docstring explicando propósito y parámetros"
                    })
                    
                self.generic_visit(node)
                
            def visit_ClassDef(self, node):
                # Verificar docstring de clase
                if not ast.get_docstring(node):
                    self.results["issues"].append({
                        "type": "missing_class_docstring",
                        "file": str(self.file_path),
                        "line": node.lineno,
                        "class": node.name,
                        "severity": "low",
                        "description": f"Clase {node.name} sin docstring",
                        "suggestion": "Añadir docstring explicando propósito de la clase"
                    })
                    
                # Verificar clases muy grandes
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                if len(methods) > 20:
                    self.results["suggestions"].append({
                        "type": "large_class",
                        "file": str(self.file_path),
                        "line": node.lineno,
                        "class": node.name,
                        "description": f"Clase {node.name} tiene muchos métodos ({len(methods)})",
                        "suggestion": "Considerar dividir responsabilidades"
                    })
                    
                self.generic_visit(node)
                
        analyzer = CodeAnalyzer(file_path, self.results)
        analyzer.visit(tree)
        
    def _calculate_stats(self):
        """Calcular estadísticas del análisis"""
        
        # Contar por severidad
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        for issue in self.results["issues"]:
            severity = issue.get("severity", "medium")
            severity_counts[severity] += 1
            
        # Contar por tipo
        type_counts = {}
        for issue in self.results["issues"]:
            issue_type = issue.get("type", "unknown")
            type_counts[issue_type] = type_counts.get(issue_type, 0) + 1
            
        # Calcular score de calidad
        total_issues = len(self.results["issues"])
        high_weight = severity_counts["high"] * 3
        medium_weight = severity_counts["medium"] * 2
        low_weight = severity_counts["low"] * 1
        
        penalty = high_weight + medium_weight + low_weight
        max_penalty = max(penalty, 1)
        
        # Score sobre 100, penalizando por issues
        quality_score = max(0, 100 - (penalty * 2))
        
        self.results["stats"] = {
            "quality_score": quality_score,
            "total_issues": total_issues,
            "severity_counts": severity_counts,
            "type_counts": type_counts,
            "avg_issues_per_file": total_issues / max(self.results["files_analyzed"], 1)
        }
        
    def _display_results(self):
        """Mostrar resultados del análisis"""
        stats = self.results["stats"]
        
        print(f"\n📊 RESULTADOS DEL ANÁLISIS")
        print("=" * 30)
        print(f"📁 Archivos analizados: {self.results['files_analyzed']}")
        print(f"📄 Total de líneas: {self.results['total_lines']:,}")
        print(f"❗ Issues encontrados: {stats['total_issues']}")
        print(f"💡 Sugerencias: {len(self.results['suggestions'])}")
        print(f"📈 Score de calidad: {stats['quality_score']:.1f}/100")
        
        print(f"\n🚨 Issues por severidad:")
        for severity, count in stats['severity_counts'].items():
            if count > 0:
                print(f"   {severity.upper()}: {count}")
                
        print(f"\n🔍 Tipos de issues más comunes:")
        sorted_types = sorted(stats['type_counts'].items(), key=lambda x: x[1], reverse=True)
        for issue_type, count in sorted_types[:5]:
            print(f"   {issue_type}: {count}")


# GENERADOR DE MEJORAS


class ImprovementGenerator:
    """Generador de mejoras específicas"""
    
    def __init__(self, analysis_results):
        self.analysis = analysis_results
        self.improvements = []
        
    def generate_improvements(self):
        """Generar mejoras basadas en análisis"""
        print(f"\n🔧 GENERANDO MEJORAS...")
        print("=" * 30)
        
        # Generar mejoras por tipo de issue
        self._generate_logging_improvements()
        self._generate_docstring_improvements()
        self._generate_refactoring_improvements()
        self._generate_code_quality_improvements()
        
        # Ordenar por prioridad
        self.improvements.sort(key=lambda x: x.get('priority_score', 0), reverse=True)
        
        self._display_improvements()
        return self.improvements
        
    def _generate_logging_improvements(self):
        """Mejorar uso de logging"""
        print_issues = [i for i in self.analysis["issues"] if i.get("type") == "print_usage"]
        
        if print_issues:
            improvement = {
                "type": "logging_system",
                "title": "Implementar Sistema de Logging",
                "description": f"Reemplazar {len(print_issues)} usos de print() con logging",
                "priority_score": len(print_issues) * 2,
                "files_affected": list(set(i["file"] for i in print_issues)),
                "implementation": self._get_logging_implementation(),
                "benefits": [
                    "Control de niveles de log (DEBUG, INFO, WARNING, ERROR)",
                    "Posibilidad de enviar logs a archivos",
                    "Mejor debugging en producción",
                    "Logs estructurados y timestamps automáticos"
                ]
            }
            self.improvements.append(improvement)
            
    def _generate_docstring_improvements(self):
        """Mejorar documentación"""
        docstring_issues = [i for i in self.analysis["issues"] 
                          if i.get("type") in ["missing_docstring", "missing_class_docstring"]]
        
        if docstring_issues:
            improvement = {
                "type": "documentation",
                "title": "Mejorar Documentación del Código",
                "description": f"Añadir docstrings a {len(docstring_issues)} funciones/clases",
                "priority_score": len(docstring_issues),
                "files_affected": list(set(i["file"] for i in docstring_issues)),
                "implementation": self._get_docstring_implementation(),
                "benefits": [
                    "Mejor comprensión del código",
                    "Facilita mantenimiento",
                    "Genera documentación automática",
                    "Mejora experiencia de desarrollo"
                ]
            }
            self.improvements.append(improvement)
            
    def _generate_refactoring_improvements(self):
        """Generar mejoras de refactoring"""
        long_functions = [i for i in self.analysis["issues"] if i.get("type") == "long_function"]
        
        if long_functions:
            improvement = {
                "type": "refactoring",
                "title": "Refactorizar Funciones Complejas",
                "description": f"Simplificar {len(long_functions)} funciones largas",
                "priority_score": len(long_functions) * 3,  # Alta prioridad
                "files_affected": list(set(i["file"] for i in long_functions)),
                "implementation": self._get_refactoring_implementation(),
                "benefits": [
                    "Código más legible y mantenible",
                    "Funciones más testeable",
                    "Menor complejidad ciclomática",
                    "Facilita reutilización de código"
                ]
            }
            self.improvements.append(improvement)
            
    def _generate_code_quality_improvements(self):
        """Mejoras generales de calidad"""
        quality_score = self.analysis["stats"]["quality_score"]
        
        if quality_score < 80:
            improvement = {
                "type": "quality_boost",
                "title": "Mejoras Generales de Calidad",
                "description": f"Elevar score de calidad de {quality_score:.1f} a 85+",
                "priority_score": (80 - quality_score) if quality_score < 80 else 0,
                "implementation": self._get_quality_implementation(),
                "benefits": [
                    "Código más robusto y confiable",
                    "Menos bugs en producción",
                    "Mejor performance",
                    "Facilita escalabilidad"
                ]
            }
            self.improvements.append(improvement)
            
    def _get_logging_implementation(self):
        """Implementación del sistema de logging"""
        return '''
# 1. Crear archivo: framework_logger.py
import logging
import sys
from datetime import datetime

def setup_framework_logging(level=logging.INFO, log_file=None):
    """Configurar logging para el framework"""
    
    # Formato de logs
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Handler para archivo (opcional)
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configurar logger raíz
    logging.basicConfig(
        level=level,
        handlers=handlers
    )
    
    return logging.getLogger('framework')

# 2. Reemplazar print() por logging:
# ANTES:
print("Agent started")

# DESPUÉS:
logger = logging.getLogger('framework')
logger.info("Agent started")

# 3. Diferentes niveles:
logger.debug("Información detallada para debugging")
logger.info("Información general del funcionamiento")
logger.warning("Algo inesperado pero no crítico")
logger.error("Error que impide operación")
logger.critical("Error crítico del sistema")
        '''
        
    def _get_docstring_implementation(self):
        """Implementación de docstrings"""
        return '''
# Formato estándar para docstrings:

def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ejecutar una acción específica del agente.
    
    Args:
        action (str): Nombre de la acción a ejecutar
        params (Dict[str, Any]): Parámetros para la acción
        
    Returns:
        Dict[str, Any]: Resultado de la ejecución con formato:
            - success (bool): Si la acción fue exitosa
            - data (Any): Datos resultado de la acción
            - error (str, opcional): Mensaje de error si falló
            
    Raises:
        ValueError: Si la acción no es válida
        RuntimeError: Si el agente no está activo
        
    Example:
        >>> result = agent.execute_action("generate.code", {"language": "python"})
        >>> print(result["data"]["code"])
    """
    pass

class FrameworkAgent:
    """
    Agente base del framework de agentes autónomos.
    
    Esta clase proporciona la funcionalidad básica que todos los agentes
    deben implementar, incluyendo comunicación, gestión de estado y
    ejecución de acciones.
    
    Attributes:
        id (str): Identificador único del agente
        name (str): Nombre descriptivo del agente
        status (AgentStatus): Estado actual del agente
        capabilities (List[AgentCapability]): Capacidades disponibles
        
    Example:
        >>> agent = FrameworkAgent("agent.example", "mi_agente", framework)
        >>> await agent.start()
        >>> result = await agent.execute_action("ping", {})
    """
    pass
        '''
        
    def _get_refactoring_implementation(self):
        """Implementación de refactoring"""
        return '''
# Técnicas de refactoring para funciones largas:

# 1. EXTRAER MÉTODOS
# ANTES - Función larga:
def process_agent_request(self, request):
    # Validar request (10 líneas)
    if not request:
        return {"error": "Empty request"}
    # ... más validaciones
    
    # Procesar datos (15 líneas) 
    data = request.get("data", {})
    # ... procesamiento complejo
    
    # Generar respuesta (20 líneas)
    response = {"status": "success"}
    # ... construcción de respuesta
    
    return response

# DESPUÉS - Funciones pequeñas:
def process_agent_request(self, request):
    """Procesar request de agente - función coordinadora"""
    validation_result = self._validate_request(request)
    if not validation_result["valid"]:
        return validation_result["error_response"]
        
    processed_data = self._process_request_data(request["data"])
    response = self._generate_response(processed_data)
    
    return response
    
def _validate_request(self, request):
    """Validar estructura y contenido del request"""
    # Lógica de validación concentrada
    pass
    
def _process_request_data(self, data):
    """Procesar datos del request"""
    # Lógica de procesamiento concentrada
    pass
    
def _generate_response(self, processed_data):
    """Generar respuesta estructurada"""
    # Lógica de respuesta concentrada
    pass

# 2. EXTRAER CLASES para responsabilidades separadas
# 3. USAR PATRONES como Strategy, Command, Factory
# 4. ELIMINAR CÓDIGO DUPLICADO
        '''
        
    def _get_quality_implementation(self):
        """Implementación de mejoras de calidad"""
        return '''
# Mejoras generales de calidad del código:

# 1. AÑADIR TYPE HINTS
from typing import Dict, List, Optional, Union, Any

def create_agent(name: str, config: Dict[str, Any]) -> Optional['BaseAgent']:
    """Crear agente con tipos claros"""
    pass

# 2. VALIDACIÓN DE ENTRADA
def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Ejecutar acción con validación"""
    # Validar tipos
    if not isinstance(action, str):
        raise TypeError("Action must be string")
        
    if not isinstance(params, dict):
        raise TypeError("Params must be dictionary")
        
    # Validar valores
    if not action.strip():
        raise ValueError("Action cannot be empty")
        
    # Continuar con lógica...

# 3. MANEJO DE ERRORES ROBUSTO
async def safe_agent_operation(self, operation):
    """Operación segura con manejo de errores"""
    try:
        result = await operation()
        return {"success": True, "data": result}
        
    except ConnectionError as e:
        logger.error(f"Connection failed: {e}")
        return {"success": False, "error": "Network error", "retry": True}
        
    except ValueError as e:
        logger.error(f"Invalid value: {e}")
        return {"success": False, "error": "Invalid input", "retry": False}
        
    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
        return {"success": False, "error": "Internal error", "retry": False}

# 4. TESTS UNITARIOS
import pytest

def test_agent_creation():
    """Test creación de agente"""
    agent = create_agent("test_agent", {"type": "basic"})
    assert agent is not None
    assert agent.name == "test_agent"

# 5. CONFIGURACIÓN CENTRALIZADA
class FrameworkConfig:
    """Configuración centralizada del framework"""
    
    def __init__(self, config_file="config.yaml"):
        self.load_config(config_file)
        
    def get(self, key: str, default=None):
        """Obtener valor de configuración"""
        return self._config.get(key, default)
        '''
        
    def _display_improvements(self):
        """Mostrar mejoras generadas"""
        if not self.improvements:
            print("✅ No se encontraron mejoras necesarias")
            return
            
        print(f"💡 Generadas {len(self.improvements)} mejoras:")
        
        for i, improvement in enumerate(self.improvements, 1):
            print(f"\n{i}. {improvement['title']}")
            print(f"   📝 {improvement['description']}")
            print(f"   🎯 Prioridad: {improvement.get('priority_score', 0)}")
            if 'files_affected' in improvement:
                print(f"   📁 Archivos: {len(improvement['files_affected'])}")
            
            if improvement.get('benefits'):
                print("   ✨ Beneficios:")
                for benefit in improvement['benefits'][:2]:  # Mostrar solo los primeros 2
                    print(f"      • {benefit}")


# GENERADOR DE REPORTES


class ReportGenerator:
    """Generador de reportes de mejoras"""
    
    def __init__(self, analysis_results, improvements):
        self.analysis = analysis_results
        self.improvements = improvements
        
    def generate_report(self):
        """Generar reporte completo"""
        print(f"\n📄 GENERANDO REPORTE...")
        
        # Crear contenido del reporte
        content = self._build_report_content()
        
        # Guardar archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"framework_improvement_report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"✅ Reporte guardado: {report_file}")
        
        # Generar reporte JSON también
        json_file = f"framework_analysis_{timestamp}.json"
        self._save_json_report(json_file)
        print(f"✅ Datos JSON guardados: {json_file}")
        
        return report_file, json_file
        
    def _build_report_content(self):
        """Construir contenido del reporte en Markdown"""
        stats = self.analysis["stats"]
        
        content = f"""# 🤖 Reporte de Auto-Mejora del Framework

**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Score de Calidad:** {stats['quality_score']:.1f}/100

## 📊 Resumen Ejecutivo

- **Archivos analizados:** {self.analysis['files_analyzed']}
- **Líneas de código:** {self.analysis['total_lines']:,}
- **Issues encontrados:** {stats['total_issues']}
- **Mejoras propuestas:** {len(self.improvements)}

### 🚨 Distribución de Issues

| Severidad | Cantidad |
|-----------|----------|
"""
        
        for severity, count in stats['severity_counts'].items():
            content += f"| {severity.upper()} | {count} |\n"
            
        content += f"""
### 🎯 Score de Calidad: {stats['quality_score']:.1f}/100

"""
        
        if stats['quality_score'] >= 90:
            content += "🏆 **EXCELENTE** - Framework en excelente estado\n"
        elif stats['quality_score'] >= 75:
            content += "✅ **BUENO** - Framework en buen estado con pocas mejoras\n"
        elif stats['quality_score'] >= 60:
            content += "⚠️ **ACEPTABLE** - Framework necesita algunas mejoras\n"
        else:
            content += "🚨 **NECESITA ATENCIÓN** - Framework requiere mejoras significativas\n"
            
        # Issues detallados
        content += "\n## ❗ Issues Encontrados\n\n"
        
        if self.analysis["issues"]:
            for i, issue in enumerate(self.analysis["issues"][:10], 1):  # Top 10
                content += f"""### {i}. {issue.get('description', 'Sin descripción')}

- **Archivo:** `{issue.get('file', 'N/A')}`
- **Línea:** {issue.get('line', 'N/A')}
- **Tipo:** {issue.get('type', 'unknown')}
- **Severidad:** {issue.get('severity', 'medium').upper()}
"""
                if 'suggestion' in issue:
                    content += f"- **Sugerencia:** {issue['suggestion']}\n"
                content += "\n"
        else:
            content += "✅ No se encontraron issues significativos\n\n"
            
        # Mejoras propuestas
        content += "## 🔧 Mejoras Propuestas\n\n"
        
        for i, improvement in enumerate(self.improvements, 1):
            content += f"""### {i}. {improvement['title']}

**Descripción:** {improvement['description']}  
**Prioridad:** {improvement.get('priority_score', 0)}/10

**Beneficios:**
"""
            for benefit in improvement.get('benefits', []):
                content += f"- {benefit}\n"
                
            content += f"""
**Implementación:**

```python
{improvement.get('implementation', 'Ver documentación específica')}
```

---

"""
        
        # Próximos pasos
        content += """## 🚀 Próximos Pasos Recomendados

1. **Implementar mejoras de alta prioridad** (score > 7)
2. **Establecer estándares de código** para prevenir regresiones
3. **Configurar CI/CD** con análisis automático de calidad
4. **Añadir tests** para cubrir funcionalidad crítica
5. **Documentar APIs** para facilitar uso y mantenimiento

## 📈 Métricas de Seguimiento

Para medir el progreso de las mejoras:

- **Score de calidad objetivo:** 85+
- **Reducción de issues críticos:** 100%
- **Cobertura de tests:** 80%+
- **Documentación de APIs:** 90%+

---

*Reporte generado automáticamente por el Sistema de Auto-Mejora del Framework*
"""
        
        return content
        
    def _save_json_report(self, filename):
        """Guardar reporte en formato JSON"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "analysis": self.analysis,
            "improvements": self.improvements,
            "summary": {
                "quality_score": self.analysis["stats"]["quality_score"],
                "total_files": self.analysis["files_analyzed"],
                "total_issues": len(self.analysis["issues"]),
                "total_improvements": len(self.improvements)
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# FUNCIÓN PRINCIPAL


async def main():
    """Función principal del sistema de auto-mejora"""
    
    print("🚀 SISTEMA DE AUTO-MEJORA DEL FRAMEWORK")
    print("=" * 60)
    print(f"⏰ Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 1. Verificar entorno
        print("1️⃣ Verificando entorno...")
        current_dir = Path.cwd()
        print(f"   📁 Directorio actual: {current_dir}")
        
        python_files = list(current_dir.rglob("*.py"))
        if not python_files:
            print("   ❌ No se encontraron archivos Python")
            return
        print(f"   ✅ Encontrados {len(python_files)} archivos Python")
        
        # 2. Análisis del framework
        print("\n2️⃣ Ejecutando análisis...")
        analyzer = FrameworkAnalyzer()
        analysis_results = analyzer.analyze()
        
        # 3. Generar mejoras
        print("\n3️⃣ Generando mejoras...")
        improver = ImprovementGenerator(analysis_results)
        improvements = improver.generate_improvements()
        
        # 4. Crear reporte
        print("\n4️⃣ Creando reporte...")
        reporter = ReportGenerator(analysis_results, improvements)
        report_file, json_file = reporter.generate_report()
        
        # 5. Resumen final
        print(f"\n🎉 AUTO-MEJORA COMPLETADA")
        print("=" * 40)
        stats = analysis_results["stats"]
        print(f"📊 Score de calidad: {stats['quality_score']:.1f}/100")
        print(f"❗ Issues encontrados: {stats['total_issues']}")
        print(f"💡 Mejoras propuestas: {len(improvements)}")
        print(f"📄 Reporte: {report_file}")
        
        # Mostrar top 3 mejoras
        if improvements:
            print(f"\n🔥 TOP 3 MEJORAS RECOMENDADAS:")
            for i, improvement in enumerate(improvements[:3], 1):
                print(f"   {i}. {improvement['title']}")
                print(f"      🎯 Prioridad: {improvement.get('priority_score', 0)}")
                
        # Consejos finales
        print(f"\n💡 PRÓXIMOS PASOS:")
        print(f"   1. Revisar reporte detallado: {report_file}")
        print(f"   2. Implementar mejoras de alta prioridad")
        print(f"   3. Ejecutar análisis regularmente")
        print(f"   4. Configurar CI/CD para calidad automática")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Proceso interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"\n👋 Análisis completado")


# PUNTO DE ENTRADA


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║            🤖 FRAMEWORK AUTO-IMPROVEMENT SYSTEM          ║
║                                                          ║
║  Este script analizará tu framework y generará          ║
║  sugerencias específicas de mejora automáticamente      ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝

Presiona ENTER para comenzar el análisis...
    """)
    
    try:
        input()  # Esperar confirmación del usuario
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 ¡Hasta luego!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Asegúrate de ejecutar desde el directorio del proyecto")