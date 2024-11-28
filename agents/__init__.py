from .base import (
    AgentBaseNode,
)
from .nodes import (
    AgentNode,
    BasicRecursionFilterNode,
)

NODE_CLASS_MAPPINGS = {
    "AgentNode": AgentNode,
    "BasicRecursionFilterNode": BasicRecursionFilterNode,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Agent Node": "AgentNode",
    "Basic Recursion Filter Node": "BasicRecursionFilterNode",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
