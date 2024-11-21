"""
@title: ComfyUI_LiteLLM
@nickname: Tasha
@description: Nodes for interfacing with LiteLLM
"""
import os
import sys
import __main__

from . import config

python = sys.executable
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_CLASS_MAPPINGS = {}
WEB_DIRECTORY = "js"

from . import litellmnodes

from .LightRAG import NODE_CLASS_MAPPINGS as LR_NCM
from .LightRAG import NODE_DISPLAY_NAME_MAPPINGS as LR_NDNM

NODE_CLASS_MAPPINGS.update(LR_NCM)
NODE_DISPLAY_NAME_MAPPINGS.update(LR_NDNM)

from .Agents import NODE_CLASS_MAPPINGS as AG_NCM
from .Agents import NODE_DISPLAY_NAME_MAPPINGS as AG_NDNM

NODE_CLASS_MAPPINGS.update(AG_NCM)
NODE_DISPLAY_NAME_MAPPINGS.update(AG_NDNM)

NODE_CLASS_MAPPINGS.update(litellmnodes.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(litellmnodes.NODE_DISPLAY_NAME_MAPPINGS)



__all__ = ['NODE_CLASS_MAPPINGS', "WEB_DIRECTORY", "NODE_DISPLAY_NAME_MAPPINGS"]
