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
    "Iterative Completion Agent": "AgentNode",
    "Completion Enhancement Filter": "BasicRecursionFilterNode",
    "Document Chunk Processor": "DocumentChunkRecursionFilterNode",

}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
