"""
@title: ComfyUI_LiteLLM
@nickname: Tasha
@description: Nodes for interfacing with LiteLLM
"""
import os
import sys
from . import utils
from . import config

CustomDict = utils.custom_dict.CustomDict

python = sys.executable
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_CLASS_MAPPINGS = {}
WEB_DIRECTORY = "js"

from . import litellmnodes

try:
    from .lightrag import NODE_CLASS_MAPPINGS as LR_NCM
    from .lightrag import NODE_DISPLAY_NAME_MAPPINGS as LR_NDNM
    NODE_CLASS_MAPPINGS.update(LR_NCM)
    NODE_DISPLAY_NAME_MAPPINGS.update(LR_NDNM)
except ImportError:
    # LightRAG not available - skip these nodes
    pass

from .agents import NODE_CLASS_MAPPINGS as AG_NCM
from .agents import NODE_DISPLAY_NAME_MAPPINGS as AG_NDNM

NODE_CLASS_MAPPINGS.update(AG_NCM)
NODE_DISPLAY_NAME_MAPPINGS.update(AG_NDNM)

NODE_CLASS_MAPPINGS.update(litellmnodes.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(litellmnodes.NODE_DISPLAY_NAME_MAPPINGS)



__all__ = ['NODE_CLASS_MAPPINGS', "WEB_DIRECTORY", "NODE_DISPLAY_NAME_MAPPINGS","CustomDict"]
