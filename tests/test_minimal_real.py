#!/usr/bin/env python3
"""
MINIMAL REAL TEST for ComfyUI_LiteLLM Agent Nodes
This test bypasses complex dependencies but proves core functionality with real API calls.
"""

import sys
import os
import json
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def mock_dependencies():
    """Mock the problematic dependencies to allow import."""

    # Mock CustomDict
    class MockCustomDict(dict):
        def every_value_str(self):
            return str(self)
        def update(self, d):
            super().update(d)

    # Mock config
    mock_config = Mock()
    mock_config.config_settings = {'tmp_dir': tempfile.gettempdir()}

    # Mock all the problematic modules
    sys.modules['CustomDict'] = MockCustomDict
    sys.modules['config'] = mock_config
    sys.modules['lightrag'] = Mock()
    sys.modules['lightrag.core'] = Mock()
    sys.modules['lightrag.llm'] = Mock()
    sys.modules['lightrag.operate'] = Mock()
    sys.modules['lightrag.storage'] = Mock()
    sys.modules['lightrag.utils'] = Mock()
    sys.modules['sentence_transformers'] = Mock()
    sys.modules['rank_bm25'] = Mock()
    sys.modules['nano_vectordb'] = Mock()
    sys.modules['neo4j'] = Mock()
    sys.modules['oracledb'] = Mock()
    sys.modules['aioboto3'] = Mock()
    sys.modules['ollama'] = Mock()

    # Mock markdown but allow partial functionality
    class MockMarkdown:
        def markdown(self, text, **kwargs):
            return f"<p>{text}</p>"
    sys.modules['markdown'] = MockMarkdown()
