#!/usr/bin/env python3
"""
Robust Test Runner for ComfyUI_LiteLLM
Handles dependency issues, import problems, and runs tests successfully.
"""

import sys
import os
import time
import tempfile
import importlib
from pathlib import Path
from unittest.mock import Mock, MagicMock
import subprocess
import traceback

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def print_colored(text, color):
    """Print colored text."""
    print(f"{color}{text}{Colors.RESET}")

def setup_mock_environment():
    """Set up comprehensive mock environment to handle all dependency issues."""
    print_colored("🔧 Setting up mock environment...", Colors.BLUE)

    # Mock LightRAG package and all its components
    mock_lightrag = MagicMock()
    mock_lightrag.LightRAG = MagicMock()
    mock_lightrag.QueryParam = MagicMock()
    mock_lightrag.llm = MagicMock()
    mock_lightrag.llm.gpt_4o_mini_complete = MagicMock()
    mock_lightrag.llm.gpt_4o_complete = MagicMock()
    sys.modules['lightrag'] = mock_lightrag
    sys.modules['lightrag.llm'] = mock_lightrag.llm

    # Mock config module
    mock_config = Mock()
    mock_config.config_settings = {
        'tmp_dir': tempfile.gettempdir(),
        'cache_dir': tempfile.gettempdir()
    }
    sys.modules['config'] = mock_config

    # Mock utils
    mock_utils = Mock()
    mock_custom_dict = Mock()
    mock_custom_dict.CustomDict = dict  # Use regular dict as mock
    mock_utils.custom_dict = mock_custom_dict
    sys.modules['utils'] = mock_utils
    sys.modules['utils.custom_dict'] = mock_custom_dict

    # Mock litellm
    mock_litellm = MagicMock()
    mock_litellm.completion = MagicMock()
    mock_litellm.api_key = "mock_key"
    mock_litellm.api_base = "mock_base"
    sys.modules['litellm'] = mock_litellm

    # Mock ComfyUI specific modules
    sys.modules['folder_paths'] = Mock()
    sys.modules['execution'] = Mock()
    sys.modules['server'] = Mock()

    print_colored("✓ Mock environment ready", Colors.GREEN)
    return True

def test_direct_api_connectivity():
    """Test direct API connectivity with Kluster.ai."""
    print_colored("\n🔌 Testing API Connectivity...", Colors.BLUE)

    try:
        import requests

        # Test basic connectivity
        response = requests.get("https://api.kluster.ai", timeout=10)
        if response.status_code in [200, 404, 403]:  # Any response indicates connectivity
            print_colored("✓ Network connectivity to Kluster.ai confirmed", Colors.GREEN)

        # Test actual API call
        from openai import OpenAI

        # Get API configuration from environment
        api_key = os.environ.get('KLUSTER_API_KEY')
        if not api_key:
            print("❌ Kluster API key not found in environment")
            print("Set KLUSTER_API_KEY environment variable or create .env file")
            return False

        client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get('KLUSTER_BASE_URL', 'https://api.kluster.ai/v1')
        )

        response = client.chat.completions.create(
            model="mistralai/Mistral-Nemo-Instruct-2407",
            messages=[{"role": "user", "content": "Say 'API_TEST_SUCCESS' only."}],
            max_tokens=10,
            temperature=0
        )

        result = response.choices[0].message.content
        print_colored(f"✓ API call successful: {result}", Colors.GREEN)
        return True

    except Exception as e:
        print_colored(f"⚠️ API connectivity issue: {e}", Colors.YELLOW)
        return False

def test_agent_imports():
    """Test that agent modules can be imported successfully."""
    print_colored("\n📦 Testing Agent Imports...", Colors.BLUE)

    try:
        # Setup mock environment first
        setup_mock_environment()

        # Test imports
        from agents.nodes import AgentNode, BasicRecursionFilterNode, DocumentChunkRecursionFilterNode
        from agents import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

        print_colored("✓ AgentNode imported successfully", Colors.GREEN)
        print_colored("✓ BasicRecursionFilterNode imported successfully", Colors.GREEN)
        print_colored("✓ DocumentChunkRecursionFilterNode imported successfully", Colors.GREEN)
        print_colored(f"✓ Found {len(NODE_CLASS_MAPPINGS)} node class mappings", Colors.GREEN)
        print_colored(f"✓ Found {len(NODE_DISPLAY_NAME_MAPPINGS)} display name mappings", Colors.GREEN)

        return True

    except Exception as e:
        print_colored(f"❌ Import failed: {e}", Colors.RED)
        traceback.print_exc()
        return False

def test_input_validation():
    """Test comprehensive input validation."""
    print_colored("\n🛡️ Testing Input Validation...", Colors.BLUE)

    try:
        setup_mock_environment()
        from agents.nodes import AgentNode, BasicRecursionFilterNode, DocumentChunkRecursionFilterNode

        agent = AgentNode()

        # Test validation cases
        validation_tests = [
            ("missing_model", {"prompt": "test"}, "Model is required"),
            ("missing_prompt", {"model": "test-model"}, "Either 'prompt' or 'List_prompts' must be provided"),
            ("invalid_iterations", {"model": "test", "prompt": "test", "max_iterations": 0}, "max_iterations must be at least 1"),
            ("empty_list", {"model": "test", "List_prompts": []}, "List_prompts cannot be empty"),
            ("invalid_list_types", {"model": "test", "List_prompts": ["valid", 123]}, "All items in List_prompts must be strings")
        ]

        passed = 0
        for test_name, kwargs, expected_error in validation_tests:
            try:
                agent.handler(**kwargs)
                print_colored(f"  ❌ {test_name}: Should have failed", Colors.RED)
            except ValueError as e:
                if expected_error in str(e):
                    print_colored(f"  ✓ {test_name}: Correctly caught error", Colors.GREEN)
                    passed += 1
                else:
                    print_colored(f"  ? {test_name}: Wrong error message", Colors.YELLOW)
            except Exception as e:
                print_colored(f"  ? {test_name}: Unexpected error type: {type(e).__name__}", Colors.YELLOW)

        # Test filter validation
        filter_node = BasicRecursionFilterNode()
        try:
            filter_node.handler(max_depth=2, LLLM_provider=None)
            print_colored("  ❌ filter_validation: Should have failed", Colors.RED)
        except ValueError as e:
            if "LLLM_provider is required" in str(e):
                print_colored("  ✓ filter_validation: Correctly caught error", Colors.GREEN)
                passed += 1

        # Test document filter validation
        doc_filter = DocumentChunkRecursionFilterNode()
        try:
            doc_filter.handler(LLLM_provider=lambda x: "test", document="", chunk_size=10)
            print_colored("  ❌ doc_validation: Should have failed", Colors.RED)
        except ValueError as e:
            if "Document text cannot be empty" in str(e):
                print_colored("  ✓ doc_validation: Correctly caught error", Colors.GREEN)
                passed += 1

        print_colored(f"✓ Input validation: {passed}/7 tests passed", Colors.GREEN if passed >= 6 else Colors.YELLOW)
        return passed >= 6

    except Exception as e:
        print_colored(f"❌ Validation test failed: {e}", Colors.RED)
        return False

def test_naming_improvements():
    """Test that display names were improved."""
    print_colored("\n🏷️ Testing Naming Improvements...", Colors.BLUE)

    try:
        setup_mock_environment()
        from agents import NODE_DISPLAY_NAME_MAPPINGS

        expected_names = {
            "AgentNode": "Iterative Completion Agent",
            "BasicRecursionFilterNode": "Completion Enhancement Filter",
            "DocumentChunkRecursionFilterNode": "Document Chunk Processor"
        }

        passed = 0
        for node_name, expected_display_name in expected_names.items():
            if node_name in NODE_DISPLAY_NAME_MAPPINGS:
                if NODE_DISPLAY_NAME_MAPPINGS[node_name] == expected_display_name:
                    print_colored(f"  ✓ {node_name} → {expected_display_name}", Colors.GREEN)
                    passed += 1
                else:
                    print_colored(f"  ❌ {node_name} maps to wrong display name", Colors.RED)
            else:
                print_colored(f"  ❌ Missing display name for: {node_name}", Colors.RED)

        print_colored(f"✓ Naming: {passed}/3 improvements verified", Colors.GREEN if passed == 3 else Colors.YELLOW)
        return passed == 3

    except Exception as e:
        print_colored(f"❌ Naming test failed: {e}", Colors.RED)
        return False

def test_documentation():
    """Test that documentation was created."""
    print_colored("\n📚 Testing Documentation...", Colors.BLUE)

    try:
        base_path = Path(__file__).parent / "agents"

        docs = [
            ("USAGE_GUIDE.md", 3000),
            ("EXAMPLES.md", 2000)
        ]

        passed = 0
        for doc_file, min_size in docs:
            doc_path = base_path / doc_file
            if doc_path.exists():
                content = doc_path.read_text()
                if len(content) >= min_size:
                    print_colored(f"  ✓ {doc_file}: {len(content)} chars", Colors.GREEN)
                    passed += 1
                else:
                    print_colored(f"  ❌ {doc_file}: Too short ({len(content)} chars)", Colors.RED)
            else:
                print_colored(f"  ❌ {doc_file}: Missing", Colors.RED)

        print_colored(f"✓ Documentation: {passed}/2 files verified", Colors.GREEN if passed == 2 else Colors.YELLOW)
        return passed == 2

    except Exception as e:
        print_colored(f"❌ Documentation test failed: {e}", Colors.RED)
        return False

def test_mock_functionality():
    """Test basic functionality with mocked responses."""
    print_colored("\n🤖 Testing Mock Functionality...", Colors.BLUE)

    try:
        setup_mock_environment()
        from agents.nodes import AgentNode, BasicRecursionFilterNode

        # Mock a simple completion provider
        def mock_provider(prompt):
            if "enhance" in prompt.lower():
                return "Enhanced: " + prompt[:50]
            return "Mock response to: " + prompt[:30]

        # Mock base handler
        def mock_base_handler(*args, **kwargs):
            prompt = kwargs.get('prompt', 'test')
            messages = kwargs.get('messages', [])
            messages.append({"role": "user", "content": prompt})
            completion = f"Mock completion for: {prompt[:30]}"
            messages.append({"role": "assistant", "content": completion})

            return (
                Mock(model=kwargs.get('model', 'test')),
                messages,
                completion,
                [completion],
                [messages],
                "mock_usage"
            )

        # Test basic agent
        agent = AgentNode()
        agent.base_handler = mock_base_handler

        result = agent.handler(
            model="test-model",
            prompt="Test basic functionality",
            max_iterations=1
        )

        assert len(result) == 6, "Agent should return 6 values"
        assert result[2] is not None, "Should have completion"
        print_colored("  ✓ Basic agent works", Colors.GREEN)

        # Test recursion filter
        filter_node = BasicRecursionFilterNode()
        recursion_filter = filter_node.handler(
            max_depth=1,
            LLLM_provider=mock_provider
        )[0]

        filter_result = recursion_filter("test prompt", "initial completion")
        assert filter_result is not None, "Filter should return result"
        print_colored("  ✓ Recursion filter works", Colors.GREEN)

        # Test agent with filter
        enhanced_result = agent.handler(
            model="test-model",
            prompt="Test with filter",
            max
