"""
@author: TashaSkyUp
@title: ComfyUI_LiteLLM
@nickname: Tasha
@description: Nodes for interfacing with LiteLLM
"""

import config

NODE_CLASS_MAPPINGS = {}
WEB_DIRECTORY = None

import litellmnodes

NODE_CLASS_MAPPINGS.update(litellmnodes.NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS']
