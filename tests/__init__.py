"""
ComfyUI_LiteLLM Test Suite

This package contains tests for the ComfyUI_LiteLLM agent nodes and functionality.

Test files:
- smoke_test.py: Basic smoke tests
- test_simple_real.py: Comprehensive agent improvement tests
- test_kluster_real.py: Real API integration tests
- test_hash_utils.py: Utility function tests
- test_agent_structure.py: Code structure and documentation tests
"""

__version__ = "1.0.0"
__author__ = "ComfyUI_LiteLLM Team"

# Test configuration
TEST_TIMEOUT = 30  # seconds
API_ENDPOINT = "https://api.kluster.ai/v1"
TEST_MODEL = "mistralai/Mistral-Nemo-Instruct-2407"
