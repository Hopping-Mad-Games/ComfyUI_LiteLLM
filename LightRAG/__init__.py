from .lightrag_nodes import (
  DocumentProcessorNode,
  QueryNode,
  LinuxMemoryDirectoryNode
)
from .agent_memory import AgentMemoryProviderNode

NODE_CLASS_MAPPINGS = {
  "DocumentProcessor": DocumentProcessorNode,
  "QueryNode": QueryNode,
  "LinuxMemoryDirectory": LinuxMemoryDirectoryNode,
  "AgentMemoryProvider": AgentMemoryProviderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
  "DocumentProcessor": "LightRAG Document Processor",
  "QueryNode": "LightRAG Query Node",
  "LinuxMemoryDirectory": "Linux Memory Directory",
  "AgentMemoryProvider": "Agent Memory Provider"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'AgentMemoryProviderNode']