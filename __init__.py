"""
@author: TashaSkyUp
@title: ComfyUI_LiteLLM
@nickname: Tasha
@description: Nodes for interfacing with LiteLLM
"""
import os
import sys
import __main__

from . import config

python = sys.executable

extentions_folder = os.path.join(os.path.dirname(os.path.realpath(__main__.__file__)),
                                 "web" + os.sep + "extensions" + os.sep + "Comfy-MK")
javascript_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "js")

NODE_CLASS_MAPPINGS = {}
WEB_DIRECTORY = "js"

from . import litellmnodes

NODE_CLASS_MAPPINGS.update(litellmnodes.NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS']
