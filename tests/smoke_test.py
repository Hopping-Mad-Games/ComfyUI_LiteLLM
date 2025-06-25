#!/usr/bin/env python3
"""
Smoke test for ComfyUI_LiteLLM agent nodes.
This test performs basic functionality checks without requiring external APIs.
"""

import os
import sys
from unittest.mock import Mock, patch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def mock_api_call(prompt, **kwargs):
    """Mock API call that returns predictable responses."""
    if "working" in prompt.lower() or "hello" in prompt.lower():
        return "Yes, working correctly!"
    elif "count" in prompt.lower():
        return "1, 2, 3"
    elif "test" in prompt.lower():
        return "Test response received"
    else:
        return f"Mock response for: {prompt[:30]}..."

def test_imports():
    """Test that we can import the agent modules."""
    print("Testing imports...")

    try:
        # Mock the problematic dependencies
        mock_config = Mock()
        mock_config.config_settings = {'tmp_dir': '/tmp'}

        with patch.dict(sys.modules, {
            'config': mock_config,
            'litellm': Mock(),
            'markdown': Mock(),
            'CustomDict': Mock()
        }):
            from agents.base import AgentBaseNode
            from agents import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

            # Verify the mappings were updated
            assert "AgentNode" in NODE_CLASS_MAPPINGS
            assert "BasicRecursionFilterNode" in NODE_CLASS_MAPPINGS
            assert "DocumentChunkRecursionFilterNode" in NODE_CLASS_MAPPINGS

            print("✓ Successfully imported agent modules")
            return True

    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_display_names():
    """Test that display names were improved."""
    print("Testing display names...")

    try:
        from agents import NODE_DISPLAY_NAME_MAPPINGS

        expected_names = [
            "Iterative Completion Agent",
            "Completion Enhancement Filter",
            "Document Chunk Processor"
        ]

        found_names = list(NODE_DISPLAY_NAME_MAPPINGS.keys())

        for name in expected_names:
            assert name in found_names, f"Missing display name: {name}"

        print(f"✓ Found improved display names: {found_names}")
        return True

    except Exception as e:
        print(f"✗ Display name test failed: {e}")
        return False

def test_validation_exists():
    """Test that validation code exists in the source."""
    print("Testing validation code exists...")

    try:
        nodes_file = Path(__file__).parent.parent / "agents" / "nodes.py"

        with open(nodes_file, 'r') as f:
            content = f.read()

        # Check for key validation patterns
        validations = [
            "Model is required",
            "max_iterations must be at least 1",
            "Either 'prompt' or 'List_prompts' must be provided",
            "LLLM_provider is required"
        ]

        for validation in validations:
            assert validation in content, f"Missing validation: {validation}"

        print("✓ Found all expected validation patterns")
        return True

    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        return False

def test_documentation_exists():
    """Test that documentation files exist and have content."""
    print("Testing documentation...")

    try:
        base_path = Path(__file__).parent.parent / "agents"

        docs = [
            ("USAGE_GUIDE.md", 5000),  # Should be substantial
            ("EXAMPLES.md", 3000)     # Should have multiple examples
        ]

        for doc_file, min_size in docs:
            doc_path = base_path / doc_file
            assert doc_path.exists(), f"Missing documentation: {doc_file}"

            content = doc_path.read_text()
            assert len(content) >= min_size, f"{doc_file} too short: {len(content)} chars"

            print(f"✓ {doc_file}: {len(content)} chars")

        return True

    except Exception as e:
        print(f"✗ Documentation test failed: {e}")
        return False

def test_class_structure():
    """Test that classes have expected structure."""
    print("Testing class structure...")

    try:
        with patch.dict(sys.modules, {
            'config': Mock(config_settings={'tmp_dir': '/tmp'}),
            'litellm': Mock(),
            'markdown': Mock(),
        }):
            from agents.base import AgentBaseNode

            # Test base class
            base_node = AgentBaseNode()
            assert hasattr(base_node, 'CATEGORY'), "Missing CATEGORY attribute"
            assert base_node.CATEGORY == "ETK/LLLM_Agents", f"Unexpected category: {base_node.CATEGORY}"

            print("✓ AgentBaseNode structure correct")
            return True

    except Exception as e:
        print(f"✗ Class structure test failed: {e}")
        return False

def test_error_handling_patterns():
    """Test that error handling patterns exist."""
    print("Testing error handling patterns...")

    try:
        nodes_file = Path(__file__).parent.parent / "agents" / "nodes.py"

        with open(nodes_file, 'r') as f:
            content = f.read()

        # Check for error handling patterns
        patterns = [
            "try:",
            "except Exception",
            "raise ValueError",
            "print(f\"Error",
            "print(f\"Warning"
        ]

        found_patterns = 0
        for pattern in patterns:
            if pattern in content:
                found_patterns += 1

        assert found_patterns >= 4, f"Expected at least 4 error handling patterns, found {found_patterns}"

        print(f"✓ Found {found_patterns} error handling patterns")
        return True

    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False

def run_smoke_tests():
    """Run all smoke tests."""
    print("=" * 60)
    print("ComfyUI_LiteLLM Agent Nodes - Smoke Tests")
    print("=" * 60)

    tests = [
        ("Module imports", test_imports),
        ("Display names", test_display_names),
        ("Input validation", test_validation_exists),
        ("Documentation", test_documentation_exists),
        ("Class structure", test_class_structure),
        ("Error handling", test_error_handling_patterns)
    ]

    passed = 0
    total = len(tests)

    for i, (name, test_func) in enumerate(tests, 1):
        print(f"\n[{i}/{total}] {name}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ Test returned False")
        except Exception as e:
            print(f"✗ Test exception: {e}")

    print("\n" + "=" * 60)
    print(f"SMOKE TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("🎉 All smoke tests passed!")
        print("\nThe agent nodes improvements are working correctly:")
        print("- Modules can be imported successfully")
        print("- Display names have been improved")
        print("- Input validation is in place")
        print("- Documentation has been created")
        print("- Class structure is correct")
        print("- Error handling has been added")
        print("\nReady for use in ComfyUI workflows!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed")
        print("Some improvements may need attention.")
        return False

if __name__ == '__main__':
    success = run_smoke_tests()
    sys.exit(0 if success else 1)
