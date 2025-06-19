import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

# Para la integración LLM
from openai import AsyncOpenAI
# Puedes añadir otras librerías aquí como anthropic, o un sistema de plugin para LLMs

# Para el sistema de memoria
from chromadb import Client, Settings
from chromadb.utils import embedding_functions
import numpy as np # Para manejar embeddings
from sklearn.metrics.pairwise import cosine_similarity # Para búsqueda de similitud

# Modelos del framework (asumiendo que están en agentapi.models o core.models)
# Si no existen estos modelos, los crearemos o ajustaremos.
try:
    from agentapi.models.general_models import Metric, Alert
    from agentapi.models.agent_models import AgentMessage, AgentResource
except ImportError:
    logging.warning("Models not found in agentapi.models. Using dummy classes for standalone demo.")
    # Dummy classes for demonstration if models are not available
    class Metric:
        def __init__(self, name, value, timestamp=None, tags=None, unit=""):
            self.name = name
            self.value = value
            self.timestamp = timestamp or datetime.now()
            self.tags = tags or {}
            self.unit = unit

    class Alert:
        def __init__(self, rule_name, severity, message, timestamp=None):
            self.rule_name = rule_name
            self.severity = severity
            self.message = message
            self.timestamp = timestamp or datetime.now()

    class AgentMessage:
        def __init__(self, sender_id, receiver_id, content, message_type="COMMAND"):
            self.id = str(uuid.uuid4())
            self.sender_id = sender_id
            self.receiver_id = receiver_id
            self.message_type = message_type
            self.content = content
            self.timestamp = datetime.now()

    class AgentResource:
        def __init__(self, name, data, resource_type="DATA", owner_agent_id=None):
            self.id = str(uuid.uuid4())
            self.name = name
            self.data = data
            self.type = resource_type
            self.owner_agent_id = owner_agent_id
            self.created_at = datetime.now()


# --- Enumeraciones para tipos de memoria ---
class MemoryType(Enum):
    EPISODIC = "episodic" # Experiencias específicas, eventos
    SEMANTIC = "semantic" # Conocimiento general, hechos, conceptos
    PROCEDURAL = "procedural" # Cómo hacer cosas, habilidades
    SENSORY = "sensory"   # Datos brutos de entrada

@dataclass
class Memory:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    type: MemoryType
    content: Any # Puede ser texto, JSON, referencia a un recurso
    timestamp: datetime = field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict) # Source, relevancy, etc.

class EmbeddingModel(Enum):
    OPENAI_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    SENTENCE_TRANSFORMERS_ALL_MINILM_L6_V2 = "all-MiniLM-L6-v2" # Para uso local con sentence-transformers

class AgentMemorySystem:
    """
    Sistema de memoria para agentes, utilizando ChromaDB para la memoria a largo plazo
    y un buffer para la memoria a corto plazo.
    """
    def __init__(self, agent_id: str, config: Dict[str, Any], persistence_manager=None):
        self.agent_id = agent_id
        self.config = config
        self.logger = logging.getLogger(f"AgentMemorySystem.{agent_id}")
        self.persistence_manager = persistence_manager # Para persistencia avanzada si se necesita
        self.short_term_memory_buffer: List[Memory] = []
        self.max_short_term_memory: int = self.config.get("max_short_term_memory", 20)

        # Configuración del cliente ChromaDB
        self.chroma_client = Client(Settings(
            persist_directory=self.config.get("chroma_persist_dir", "./chroma_db"),
            is_persistent=True
        ))
        self.collection_name = f"agent_{self.agent_id}_memory"
        self.embedding_model_name = self.config.get("embedding_model", EmbeddingModel.OPENAI_EMBEDDING_ADA_002.value)

        # Inicializar función de embedding
        if self.embedding_model_name == EmbeddingModel.OPENAI_EMBEDDING_ADA_002.value:
            self.embed_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"), # Asegúrate de que esto esté configurado
                model_name=self.embedding_model_name
            )
            self.logger.info(f"Using OpenAI embedding model: {self.embedding_model_name}")
        elif self.embedding_model_name == EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MINILM_L6_V2.value:
            from sentence_transformers import SentenceTransformer
            self.sentence_transformer_model = SentenceTransformer(self.embedding_model_name)
            self.embed_function = lambda texts: self.sentence_transformer_model.encode(texts).tolist()
            self.logger.info(f"Using Sentence Transformers embedding model: {self.embedding_model_name}")
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model_name}")


        self.long_term_memory_collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embed_function # Asigna la función de embedding aquí
        )
        self.logger.info(f"Initialized AgentMemorySystem for agent {agent_id} with ChromaDB collection {self.collection_name}.")


    async def add_memory(self, memory_type: MemoryType, content: Any, metadata: Optional[Dict[str, Any]] = None) -> Memory:
        """Añade una nueva entrada de memoria."""
        if isinstance(content, dict) or isinstance(content, list):
            content_str = json.dumps(content)
        else:
            content_str = str(content)

        # Generar embedding si no se proporciona (ChromaDB lo hace si se le pasa el texto y una embedding_function)
        memory = Memory(agent_id=self.agent_id, type=memory_type, content=content_str, metadata=metadata or {})

        # Añadir a la memoria a corto plazo
        self.short_term_memory_buffer.append(memory)
        if len(self.short_term_memory_buffer) > self.max_short_term_memory:
            self.short_term_memory_buffer.pop(0) # Elimina la más antigua

        # Añadir a la memoria a largo plazo (ChromaDB)
        # ChromaDB genera embeddings automáticamente si se configura la embedding_function
        try:
            await asyncio.to_thread(
                self.long_term_memory_collection.add,
                documents=[content_str],
                metadatas=[{"agent_id": self.agent_id, "type": memory_type.value, **(metadata or {}) }],
                ids=[memory.id]
            )
            self.logger.debug(f"Memory '{memory.id}' added to long-term memory (ChromaDB).")
        except Exception as e:
            self.logger.error(f"Failed to add memory to ChromaDB: {e}", exc_info=True)

        self.logger.debug(f"Memory '{memory.id}' (type: {memory_type.value}) added to short-term memory.")
        return memory

    async def retrieve_memories(self, query: str, num_results: int = 5, memory_types: Optional[List[MemoryType]] = None) -> List[Memory]:
        """
        Recupera memorias relevantes basadas en una consulta.
        Prioriza la memoria a corto plazo y luego la memoria a largo plazo.
        """
        relevant_memories: List[Memory] = []

        # 1. Buscar en memoria a corto plazo (buffer reciente)
        # Una búsqueda simple basada en palabras clave o relevancia heurística
        short_term_matches = [
            m for m in self.short_term_memory_buffer
            if query.lower() in str(m.content).lower() and (not memory_types or m.type in memory_types)
        ]
        # Podríamos hacer una búsqueda de similitud también aquí si tuviéramos embeddings para STM
        relevant_memories.extend(short_term_matches)

        # 2. Buscar en memoria a largo plazo (ChromaDB)
        try:
            # ChromaDB handles embedding the query if a function is provided
            chroma_results = await asyncio.to_thread(
                self.long_term_memory_collection.query,
                query_texts=[query],
                n_results=num_results,
                where={"type": {"$in": [mt.value for mt in memory_types]}} if memory_types else {}
            )
            for i in range(len(chroma_results['ids'][0])):
                doc_id = chroma_results['ids'][0][i]
                content = chroma_results['documents'][0][i]
                metadata = chroma_results['metadatas'][0][i]
                distance = chroma_results['distances'][0][i]

                # Reconstruir el objeto Memory
                mem = Memory(
                    id=doc_id,
                    agent_id=metadata.get("agent_id", self.agent_id),
                    type=MemoryType(metadata.get("type", "episodic")),
                    content=content,
                    timestamp=datetime.fromisoformat(metadata["timestamp"]) if "timestamp" in metadata else datetime.now(),
                    metadata=metadata
                )
                relevant_memories.append(mem)
                self.logger.debug(f"Retrieved LTM: {mem.id} (Distance: {distance:.2f})")

        except Exception as e:
            self.logger.error(f"Failed to retrieve memories from ChromaDB: {e}", exc_info=True)

        # Eliminar duplicados si alguna memoria a corto plazo también está en largo plazo
        unique_memories = {m.id: m for m in relevant_memories}
        return list(unique_memories.values())

    async def get_all_memories(self, limit: int = 100) -> List[Memory]:
        """Recupera todas las memorias (o un límite) del agente."""
        all_chroma_memories = await asyncio.to_thread(self.long_term_memory_collection.get, limit=limit)
        memories = []
        for i in range(len(all_chroma_memories['ids'])):
            doc_id = all_chroma_memories['ids'][i]
            content = all_chroma_memories['documents'][i]
            metadata = all_chroma_memories['metadatas'][i]
            memories.append(Memory(
                id=doc_id,
                agent_id=metadata.get("agent_id", self.agent_id),
                type=MemoryType(metadata.get("type", "episodic")),
                content=content,
                timestamp=datetime.fromisoformat(metadata["timestamp"]) if "timestamp" in metadata else datetime.now(),
                metadata=metadata
            ))
        return self.short_term_memory_buffer + memories

    async def clear_short_term_memory(self):
        """Limpia el buffer de memoria a corto plazo."""
        self.short_term_memory_buffer.clear()
        self.logger.info(f"Short-term memory cleared for agent {self.agent_id}.")

    async def delete_memory(self, memory_id: str) -> bool:
        """Elimina una memoria específica por ID."""
        # Intentar eliminar de STM
        original_len_stm = len(self.short_term_memory_buffer)
        self.short_term_memory_buffer = [m for m in self.short_term_memory_buffer if m.id != memory_id]
        deleted_from_stm = (len(self.short_term_memory_buffer) < original_len_stm)

        # Intentar eliminar de LTM
        deleted_from_ltm = False
        try:
            await asyncio.to_thread(self.long_term_memory_collection.delete, ids=[memory_id])
            deleted_from_ltm = True
            self.logger.info(f"Memory {memory_id} deleted from long-term memory (ChromaDB).")
        except Exception as e:
            self.logger.error(f"Failed to delete memory {memory_id} from ChromaDB: {e}")

        if deleted_from_stm or deleted_from_ltm:
            self.logger.info(f"Memory {memory_id} deleted.")
            return True
        self.logger.warning(f"Memory {memory_id} not found for deletion.")
        return False

    async def shutdown(self):
        """Realiza cualquier limpieza necesaria al apagar el sistema de memoria."""
        self.logger.info(f"Shutting down AgentMemorySystem for agent {self.agent_id}.")
        # ChromaDB automáticamente persiste al cerrar el cliente si es persistente.
        # No hay una función de "cerrar" explícita para el cliente de persistencia en las últimas versiones.
        # self.chroma_client.persist() # Si necesitas forzar la persistencia, aunque se guarda en cada operación
        self.logger.info("AgentMemorySystem shutdown complete.")

# --- Demo de uso (para probar cognitive_system.py directamente) ---
async def demo_memory_system():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DemoMemorySystem")

    # Configuración de ejemplo
    memory_config = {
        "chroma_persist_dir": "./demo_chroma_db",
        "max_short_term_memory": 5,
        "embedding_model": EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MINILM_L6_V2.value
        # Para OpenAI, asegúrate de tener OPENAI_API_KEY en tus variables de entorno
        # "embedding_model": EmbeddingModel.OPENAI_EMBEDDING_ADA_002.value
    }
    agent_id = "demo_agent_123"
    memory_system = AgentMemorySystem(agent_id, memory_config)
    
    try:
        logger.info("\n--- Demo: AgentMemorySystem ---")

        # Añadir memorias episódicas
        logger.info("\n1. Adding episodic memories...")
        await memory_system.add_memory(MemoryType.EPISODIC, "Today I processed 10 user requests successfully.", {"source": "log_event"})
        await memory_system.add_memory(MemoryType.EPISODIC, "Received an error message from external API 'payment-service'.")
        await memory_system.add_memory(MemoryType.EPISODIC, "Assisted user 'Alice' with password reset.", {"user_id": "Alice"})
        
        # Añadir memoria semántica
        logger.info("\n2. Adding semantic memory (knowledge)...")
        await memory_system.add_memory(MemoryType.SEMANTIC, "The main goal of the framework is to enable autonomous agents.", {"source": "documentation"})
        await memory_system.add_memory(MemoryType.SEMANTIC, "Error code 404 means 'Not Found' in HTTP.", {"source": "knowledge_base"})
        
        # Añadir más memorias episódicas para ver el buffer de STM
        for i in range(7):
            await memory_system.add_memory(MemoryType.EPISODIC, f"Processed transaction {i+1}.")
        
        logger.info(f"\nShort-term memory buffer size: {len(memory_system.short_term_memory_buffer)}")
        for i, mem in enumerate(memory_system.short_term_memory_buffer):
            logger.info(f"  STM {i+1}: {mem.content[:50]}...")

        # Recuperar memorias
        logger.info("\n3. Retrieving memories related to 'error messages'...")
        retrieved_errors = await memory_system.retrieve_memories("What kind of error messages did I receive?", num_results=2)
        for i, mem in enumerate(retrieved_errors):
            logger.info(f"  Retrieved {i+1}: {mem.content[:70]}... (Type: {mem.type.value})")

        logger.info("\n4. Retrieving semantic memories about the framework...")
        retrieved_semantic = await memory_system.retrieve_memories("What is the purpose of this framework?", num_results=1, memory_types=[MemoryType.SEMANTIC])
        for i, mem in enumerate(retrieved_semantic):
            logger.info(f"  Retrieved {i+1}: {mem.content[:70]}... (Type: {mem.type.value})")

        logger.info("\n5. Retrieving all memories (up to limit)...")
        all_memories = await memory_system.get_all_memories(limit=3)
        for i, mem in enumerate(all_memories):
            logger.info(f"  All Mem {i+1}: {mem.content[:70]}... (Type: {mem.type.value})")
            
        # Limpiar STM
        logger.info("\n6. Clearing short-term memory...")
        await memory_system.clear_short_term_memory()
        logger.info(f"Short-term memory buffer size after clear: {len(memory_system.short_term_memory_buffer)}")

        # Eliminar una memoria (primero la recuperamos para obtener el ID)
        logger.info("\n7. Deleting a specific memory (the 'payment-service' error)...")
        error_memories = await memory_system.retrieve_memories("payment-service error", num_results=1)
        if error_memories:
            memory_to_delete = error_memories[0]
            logger.info(f"Attempting to delete memory ID: {memory_to_delete.id}")
            delete_success = await memory_system.delete_memory(memory_to_delete.id)
            logger.info(f"Deletion successful: {delete_success}")
            # Verificar si se eliminó
            retrieved_after_delete = await memory_system.retrieve_memories("payment-service error", num_results=1)
            if not retrieved_after_delete or retrieved_after_delete[0].id != memory_to_delete.id:
                 logger.info("   ✅ Memory successfully deleted from LTM.")
            else:
                 logger.warning("   ❌ Memory still found after deletion (ChromaDB might not have flushed immediately or ID mismatch).")
        else:
            logger.warning("Memory 'payment-service' not found for deletion attempt.")


    except Exception as e:
        logger.critical(f"An error occurred during demo: {e}", exc_info=True)
    finally:
        await memory_system.shutdown()
        # Limpiar directorio de ChromaDB para futuras ejecuciones de demo
        chroma_path = Path(memory_config["chroma_persist_dir"])
        if chroma_path.exists():
            import shutil
            shutil.rmtree(chroma_path)
            logger.info(f"Cleaned up ChromaDB directory: {chroma_path}")

if __name__ == "__main__":
    import os
    # Establecer la variable de entorno para OpenAI si se usa
    # os.environ["OPENAI_API_KEY"] = "sk-..." # ¡Reemplaza con tu clave real para probar OpenAI!
    
    # Asegúrate de instalar las dependencias:
    # pip install chromadb openai sentence-transformers scikit-learn
    asyncio.run(demo_memory_system())
    