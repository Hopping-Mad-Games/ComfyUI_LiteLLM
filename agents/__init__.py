from .base import (
    AgentBaseNode,
)
from .nodes import (
    AgentNode,
    BasicRecursionFilterNode,
    DocumentChunkRecursionFilterNode,
)

NODE_CLASS_MAPPINGS = {
    "AgentNode": AgentNode,
    "BasicRecursionFilterNode": BasicRecursionFilterNode,
    "DocumentChunkRecursionFilterNode": DocumentChunkRecursionFilterNode,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Agent Node": "AgentNode",
    "Basic Recursion Filter Node": "BasicRecursionFilterNode",
    "Document Chunk Recursion Filter Node": "DocumentChunkRecursionFilterNode",

}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
