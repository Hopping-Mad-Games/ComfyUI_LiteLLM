# lightrag_base.py
import torch
import numpy as np
from typing import Dict, List, Union


class AgentBaseNode:
    def __init__(self):
        self.CATEGORY = "ETK/LLLM_Agents"

    FUNCTION = "handler"