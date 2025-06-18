# test_auto_improvement.py - ARCHIVO PARA PROBAR RÁPIDAMENTE

import asyncio
import sys
from pathlib import Path

# Verificar que estamos en el directorio correcto
if not Path("core").exists():
    print("❌ No se encuentra la carpeta 'core'")
    print("💡 Ejecuta desde el directorio raíz del proyecto")
    exit(1)

try:
    # Importar módulos del framework
    from core.autonomous_agent_framework import AgentFramework
    from core.specialized_agents import ExtendedAgentFactory
    print("✅ Módulos del framework importados correctamente")
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    print("💡 Asegúrate de que todos los archivos del framework existan")
    exit(1)

async def test_auto_improvement():   
    
    print("🧪 EJECUTANDO TEST DE AUTO-MEJORA")
    print("=" * 50)
    
    # 1. Crear framework
    print("1️⃣ Creando framework...")
    framework = AgentFramework()
    await framework.start()
    print("   ✅ Framework iniciado")
    
    # 2. Crear agente de prueba
    print("2️⃣ Creando agente de prueba...")
    try:
        strategist = ExtendedAgentFactory.create_agent(
            "agent.planning.strategist", "test_strategist", framework
        )
        await strategist.start()
        print("   ✅ Agente creado y iniciado")
    except Exception as e:
        print(f"   ❌ Error creando agente: {e}")
        return
    
    # 3. Análisis básico de archivos
    print("3️⃣ Analizando archivos del proyecto...")
    
    python_files = list(Path(".").rglob("*.py"))
    python_files = [f for f in python_files if "pycache" not in str(f)]
    
    print(f"   📁 Encontrados {len(python_files)} archivos Python")
    
    # Buscar issues simples
    issues_found = 0
    for file_path in python_files[:5]:  # Solo los primeros 5 para el test
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Buscar print() en código
            if 'print(' in content and not str(file_path).endswith('test_auto_improvement.py'):
                print(f"   📝 {file_path}: Encontrado uso de print()")
                issues_found += 1
                
            # Buscar funciones sin docstring
            if 'def ' in content and '' not in content:
                print(f"   📝 {file_path}: Funciones sin docstring")
                issues_found += 1
                
        except Exception as e:
            print(f"   ⚠️ Error leyendo {file_path}: {e}")
    
    print(f"   🔍 Issues encontrados: {issues_found}")
    
    # 4. Generar sugerencia de mejora
    print("4️⃣ Generando sugerencias de mejora...")
    
    suggestions = [
        "🔧 Reemplazar print() con logging para mejor control de logs",
        "📚 Añadir docstrings a funciones públicas",
        "⚡ Considerar usar async/await para operaciones I/O",
        "🧪 Añadir más tests unitarios",
        "🔒 Implementar validación de entrada en APIs"
    ]
    
    for suggestion in suggestions:
        print(f"   {suggestion}")
    
    # 5. Crear mini-reporte
    print("5️⃣ Generando reporte...")
    
    report_content = f# Test de Auto-Mejora del Framework

## Resultados del Test

- **Archivos analizados:** {len(python_files)}
- **Issues encontrados:** {issues_found}
- **Agentes activos:** {len(framework.registry.list_all_agents())}

## Sugerencias de Mejora

{chr(10).join(f"- {s}" for s in suggestions)}

## Próximos Pasos

1. Implementar sistema completo de análisis
2. Añadir generación automática de código
3. Integrar con sistema de deployment
4. Configurar mejora continua

    
    with open("test_improvement_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print("   📄 Reporte guardado en: test_improvement_report.md")
    
    # 6. Limpiar
    await framework.stop()
    
    print("\n🎉 TEST COMPLETADO")
    print("📄 Revisa el archivo: test_improvement_report.md")
    print("\n💡 Para el sistema completo, ejecuta: python auto_improve_framework.py")

if __name__ == "__main__":
    print("🚀 Iniciando test de auto-mejora...")
    print("⏳ Esto tomará unos segundos...")
    print()
    
    try:
        asyncio.run(test_auto_improvement())
    except KeyboardInterrupt:
        print("\n🛑 Test interrumpido")
    except Exception as e:
        print(f"\n❌ Error en test: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n👋 Test finalizado")