# lightrag_base.py
import torch
import numpy as np
from typing import Dict, List, Union


class LightRAGBaseNode:
    def __init__(self):
        self.CATEGORY = "LightRAG"

    @staticmethod
    def INPUT_TYPES():
        return {}

    RETURN_TYPES = ()
    FUNCTION = "execute"

    def execute(self, *args, **kwargs):
        raise NotImplementedError