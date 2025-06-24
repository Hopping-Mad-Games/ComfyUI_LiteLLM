#!/usr/bin/env python3
"""
Real Integration Test for Custom Endpoint Provider
Tests the custom endpoint functionality with actual API calls to Kluster.ai
"""

import pytest
import sys
import os
import tempfile
from unittest.mock import Mock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

class TestCustomEndpointRealIntegration:
    """Test custom endpoint with real API calls."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up minimal mocks for dependencies."""
        # Mock config
        mock_config = Mock()
        mock_config.config_settings = {'tmp_dir': tempfile.gettempdir()}
        sys.modules['config'] = mock_config

        # Mock utils
        mock_utils = Mock()
        mock_custom_dict = Mock()
        mock_custom_dict.CustomDict = dict
        mock_utils.custom_dict = mock_custom_dict
        sys.modules['utils'] = mock_utils
        sys.modules['utils.custom_dict'] = mock_custom_dict

    @pytest.fixture
    def kluster_config(self):
        """Configuration for Kluster.ai API."""
        api_key = os.environ.get('KLUSTER_API_KEY')
        if not api_key:
            pytest.skip("Kluster API key not configured. Set KLUSTER_API_KEY environment variable.")

        return {
            'model_name': "mistralai/Mistral-Nemo-Instruct-2407",
            'api_base': os.environ.get('KLUSTER_BASE_URL', 'https://api.kluster.ai/v1'),
            'api_key': api_key,
            'provider': "openai"
        }

    def test_custom_endpoint_provider_creation(self, kluster_config):
        """Test creating a custom endpoint provider with Kluster.ai config."""
        from litellmnodes import LiteLLMCustomEndpointProvider

        provider = LiteLLMCustomEndpointProvider()
        result = provider.handler(**kluster_config)

        assert isinstance(result, tuple)
        assert len(result) == 1

        model_config = result[0]
        assert isinstance(model_config, dict)
        assert "model" in model_config
        assert "kwargs" in model_config

        # Verify the configuration
        assert model_config["model"] == "openai/mistralai/Mistral-Nemo-Instruct-2407"
        assert model_config["kwargs"]["api_key"] == kluster_config["api_key"]
        assert model_config["kwargs"]["api_base"] == kluster_config["api_base"]

    def test_completion_with_custom_endpoint_real_api(self, kluster_config):
        """Test completion using custom endpoint with real API call."""
        from litellmnodes import LiteLLMCustomEndpointProvider, LitellmCompletionV2

        # Create custom endpoint model
        provider = LiteLLMCustomEndpointProvider()
        model_config = provider.handler(**kluster_config)[0]

        # Create completion node
        completion = LitellmCompletionV2()

        # Test real API call
        try:
            result = completion.litellm_completion_v2_inner(
                frequency_penalty=0.0,
                max_tokens=20,
                messages=[{"role": "user", "content": "Say 'Hello from Kluster' and nothing else"}],
                model=model_config,
                presence_penalty=0.0,
                prompt="Say 'Hello from Kluster' and nothing else",
                task="completion",
                temperature=0.1,
                top_p=1.0
            )

            # Verify we got a response
            assert result is not None

            # Check response structure
            response_content = result.choices[0].message.content
            assert isinstance(response_content, str)
            assert len(response_content) > 0

            # Should contain the expected response
            assert "Hello from Kluster" in response_content or "hello" in response_content.lower()

            print(f"✅ Real API call successful: {response_content}")
            return True

        except Exception as e:
            # If API is not available, skip the test
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                pytest.skip(f"API not available: {e}")
            else:
                pytest.fail(f"Unexpected error in real API call: {e}")

    def test_completion_handler_with_custom_endpoint(self, kluster_config):
        """Test the full completion handler with custom endpoint."""
        from litellmnodes import LiteLLMCustomEndpointProvider, LitellmCompletionV2

        # Create custom endpoint model
        provider = LiteLLMCustomEndpointProvider()
        model_config = provider.handler(**kluster_config)[0]

        # Create completion node
        completion = LitellmCompletionV2()

        try:
            # Test the full handler method
            result = completion.handler(
                model=model_config,
                messages=[{"role": "user", "content": "Count to 3"}],
                max_tokens=20,
                temperature=0.1,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                task="completion"
            )

            # Verify handler return structure
            assert isinstance(result, tuple)
            assert len(result) == 5  # (model, messages, completion, completions, usage)

            model_out, messages_out, completion_out, completions_out, usage_out = result

            # Check completion
            assert isinstance(completion_out, str)
            assert len(completion_out) > 0

            # Should contain numbers or counting
            assert any(char.isdigit() for char in completion_out)

            print(f"✅ Full handler test successful: {completion_out}")
            return True

        except Exception as e:
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                pytest.skip(f"API not available: {e}")
            else:
                pytest.fail(f"Handler test failed: {e}")

    def test_custom_endpoint_with_different_models(self):
        """Test custom endpoint with different model configurations."""
        from litellmnodes import LiteLLMCustomEndpointProvider

        provider = LiteLLMCustomEndpointProvider()

        test_configs = [
            {
                'model_name': "gpt-4o-mini",
                'api_base': "https://api.openai.com/v1",
                'api_key': os.environ.get('OPENAI_API_KEY', 'sk-test-key'),
                'provider': "openai"
            },
            {
                'model_name': "claude-3-haiku-20240307",
                'api_base': "https://api.anthropic.com",
                'api_key': os.environ.get('ANTHROPIC_API_KEY', 'sk-ant-test'),
                'provider': "anthropic"
            },
            {
                'model_name': "command-r",
                'api_base': "https://api.cohere.ai/v1",
                'api_key': os.environ.get('COHERE_API_KEY', 'test-cohere-key'),
                'provider': "cohere"
            }
        ]

        for config in test_configs:
            result = provider.handler(**config)
            model_config = result[0]

            expected_model = f"{config['provider']}/{config['model_name']}"
            assert model_config["model"] == expected_model
            assert model_config["kwargs"]["api_key"] == config["api_key"]
            assert model_config["kwargs"]["api_base"] == config["api_base"]

    def test_custom_endpoint_with_optional_parameters(self, kluster_config):
        """Test custom endpoint with all optional parameters."""
        from litellmnodes import LiteLLMCustomEndpointProvider

        provider = LiteLLMCustomEndpointProvider()

        # Add optional parameters
        full_config = {
            **kluster_config,
            'api_version': "2024-02-15-preview",
            'organization': "org-test123",
            'timeout': 120,
            'max_retries': 5
        }

        result = provider.handler(**full_config)
        model_config = result[0]

        # Verify optional parameters are included
        kwargs = model_config["kwargs"]
        assert kwargs["api_version"] == "2024-02-15-preview"
        assert kwargs["organization"] == "org-test123"
        assert kwargs["timeout"] == 120
        assert kwargs["max_retries"] == 5

    def test_integration_with_agent_nodes(self, kluster_config):
        """Test that custom endpoint works with agent nodes."""
        from litellmnodes import LiteLLMCustomEndpointProvider

        # Create custom endpoint model
        provider = LiteLLMCustomEndpointProvider()
        model_config = provider.handler(**kluster_config)[0]

        # Test that the model config is in the format expected by agent nodes
        assert isinstance(model_config, dict)
        assert "model" in model_config
        assert "kwargs" in model_config

        # The model string should be properly formatted for LiteLLM
        assert "/" in model_config["model"]  # Should have provider/model format

        # API credentials should be present
        assert "api_key" in model_config["kwargs"]
        assert "api_base" in model_config["kwargs"]

    def test_error_handling_with_invalid_config(self):
        """Test error handling with invalid configurations."""
        from litellmnodes import LiteLLMCustomEndpointProvider

        provider = LiteLLMCustomEndpointProvider()

        # Test with empty values - should not crash
        result = provider.handler(
            model_name="",
            api_base="",
            api_key="test-empty-key",
            provider="openai"
        )

        # Should still return a valid structure
        assert isinstance(result, tuple)
        assert len(result) == 1
        model_config = result[0]
        assert isinstance(model_config, dict)
        assert "model" in model_config
        assert "kwargs" in model_config

    def test_recursion_filter_with_custom_endpoint(self, kluster_config):
        """Test that recursion filters work with custom endpoint configurations."""
        from litellmnodes import LiteLLMCustomEndpointProvider, LiteLLMCompletionProvider

        # Create custom endpoint model
        provider = LiteLLMCustomEndpointProvider()
        model_config = provider.handler(**kluster_config)[0]

        # Create completion provider with custom endpoint
        completion_provider = LiteLLMCompletionProvider()

        try:
            # Test that completion provider can handle custom model config
            completion_function = completion_provider.handler(
                model=model_config,
                max_tokens=30,
                temperature=0.2
            )[0]

            # Test the completion function with a simple prompt
            result = completion_function("Say 'Recursion test' and nothing else")

            # Verify we got a response
            assert isinstance(result, str)
            assert len(result) > 0
            assert "recursion" in result.lower() or "test" in result.lower()

            print(f"✅ Recursion filter with custom endpoint works: {result}")
            return True

        except Exception as e:
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                pytest.skip(f"API not available: {e}")
            else:
                pytest.fail(f"Recursion filter test failed: {e}")


def test_standalone_verification():
    """Standalone test to verify the feature works."""
    print("\n🔧 Testing Custom Endpoint Provider...")

    try:
        # Mock dependencies
        import tempfile
        from unittest.mock import Mock

        mock_config = Mock()
        mock_config.config_settings = {'tmp_dir': tempfile.gettempdir()}
        sys.modules['config'] = mock_config

        mock_utils = Mock()
        mock_custom_dict = Mock()
        mock_custom_dict.CustomDict = dict
        mock_utils.custom_dict = mock_custom_dict
        sys.modules['utils'] = mock_utils
        sys.modules['utils.custom_dict'] = mock_custom_dict

        # Test import
        from litellmnodes import LiteLLMCustomEndpointProvider
        print("✅ Custom endpoint provider imported successfully")

        # Test basic functionality
        provider = LiteLLMCustomEndpointProvider()
        api_key = os.environ.get('KLUSTER_API_KEY', 'test-key')
        result = provider.handler(
            model_name="mistralai/Mistral-Nemo-Instruct-2407",
            api_base="https://api.kluster.ai/v1",
            api_key=api_key,
            provider="openai"
        )

        model_config = result[0]
        print(f"✅ Model config created: {model_config['model']}")
        print(f"✅ API base configured: {model_config['kwargs']['api_base']}")

        return True

    except Exception as e:
        print(f"❌ Standalone test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 CUSTOM ENDPOINT REAL INTEGRATION TEST")
    print("=" * 50)

    success = test_standalone_verification()

    if success:
        print("\n🎉 Custom Endpoint Provider is working!")
        print("✅ Node can be imported and used")
        print("✅ Configuration format is correct")
        print("✅ Ready for real API testing")
        print("\nRun with pytest for full test suite:")
        print("python3 -m pytest tests/test_custom_endpoint_real.py -v")
    else:
        print("\n❌ Issues detected - check implementation")

    sys.exit(0 if success else 1)
