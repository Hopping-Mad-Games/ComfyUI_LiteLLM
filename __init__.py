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
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_CLASS_MAPPINGS = {}
WEB_DIRECTORY = "js"

from . import litellmnodes

NODE_CLASS_MAPPINGS.update(litellmnodes.NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS', "WEB_DIRECTORY", "NODE_DISPLAY_NAME_MAPPINGS"]

# # List of JavaScript files
# js_files = [
#     "genericLiteGraphTextNode.js",
#     "htmlRenderer.js"
# ]
#
#
# # Update the extensions_folder path
# extensions_folder = os.path.join(os.path.dirname(os.path.realpath(__main__.__file__)),
#                                  "web", "extensions", "HMG")
# javascript_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "js")
#
# # Ensure the HMG directory exists
# os.makedirs(extensions_folder, exist_ok=True)
#
# for js_file in js_files:
#     src = os.path.join(javascript_folder, js_file)
#     if os.path.exists(src):
#         with open(src, "r") as f:
#             content = f.read()
#         dst = os.path.join(extensions_folder, js_file)
#         # Ensure the directory for the destination file exists
#         os.makedirs(os.path.dirname(dst), exist_ok=True)
#         with open(dst, "w") as f:
#             f.write(content)
