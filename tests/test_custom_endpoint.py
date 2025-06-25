#!/usr/bin/env python3
"""
TDD Tests for Custom Endpoint Provider
Tests the functionality for configuring custom API endpoints like Kluster.ai
"""

import pytest
import sys
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

class TestCustomEndpointProvider:
    """Test the LiteLLMCustomEndpointProvider node."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks for dependencies."""
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

    def test_custom_endpoint_provider_exists(self):
        """Test that LiteLLMCustomEndpointProvider class exists and can be imported."""
        try:
            from litellmnodes import LiteLLMCustomEndpointProvider
            assert LiteLLMCustomEndpointProvider is not None
        except ImportError:
            pytest.fail("LiteLLMCustomEndpointProvider class should exist in litellmnodes")

    def test_custom_endpoint_provider_input_types(self):
        """Test that the provider has correct input types."""
        from litellmnodes import LiteLLMCustomEndpointProvider

        input_types = LiteLLMCustomEndpointProvider.INPUT_TYPES()

        # Check required inputs
        assert "required" in input_types
        required = input_types["required"]

        assert "model_name" in required
        assert "api_base" in required
        assert "api_key" in required
        assert "provider" in required

        # Check optional inputs
        assert "optional" in input_types
        optional = input_types["optional"]

        assert "api_version" in optional
        assert "organization" in optional
        assert "timeout" in optional
        assert "max_retries" in optional

    def test_custom_endpoint_provider_return_types(self):
        """Test that the provider has correct return types."""
        from litellmnodes import LiteLLMCustomEndpointProvider

        assert hasattr(LiteLLMCustomEndpointProvider, 'RETURN_TYPES')
        assert LiteLLMCustomEndpointProvider.RETURN_TYPES == ("LITELLM_MODEL",)

        assert hasattr(LiteLLMCustomEndpointProvider, 'RETURN_NAMES')
        assert LiteLLMCustomEndpointProvider.RETURN_NAMES == ("model",)

    def test_custom_endpoint_provider_basic_functionality(self):
        """Test basic functionality of the custom endpoint provider."""
        from litellmnodes import LiteLLMCustomEndpointProvider

        provider = LiteLLMCustomEndpointProvider()

        # Test with Kluster.ai configuration
        result = provider.handler(
            model_name="mistralai/Mistral-Nemo-Instruct-2407",
            api_base="https://api.kluster.ai/v1",
            api_key="test-api-key-123",
            provider="openai"
        )

        # Should return a tuple with one element (the model config)
        assert isinstance(result, tuple)
        assert len(result) == 1

        model_config = result[0]
        assert isinstance(model_config, dict)

        # Check model configuration structure
        assert "model" in model_config
        assert "kwargs" in model_config

        # Check model string format
        assert model_config["model"] == "openai/mistralai/Mistral-Nemo-Instruct-2407"

        # Check kwargs
        kwargs = model_config["kwargs"]
        assert kwargs["api_key"] == "test-api-key-123"
        assert kwargs["api_base"] == "https://api.kluster.ai/v1"
        assert "timeout" in kwargs
        assert "max_retries" in kwargs

    def test_custom_endpoint_provider_with_optional_params(self):
        """Test provider with optional parameters."""
        from litellmnodes import LiteLLMCustomEndpointProvider

        provider = LiteLLMCustomEndpointProvider()

        result = provider.handler(
            model_name="gpt-4o-mini",
            api_base="https://custom-api.example.com/v1",
            api_key="custom-key",
            provider="openai",
            api_version="2023-12-01",
            organization="org-123",
            timeout=120,
            max_retries=5
        )

        model_config = result[0]
        kwargs = model_config["kwargs"]

        assert kwargs["api_version"] == "2023-12-01"
        assert kwargs["organization"] == "org-123"
        assert kwargs["timeout"] == 120
        assert kwargs["max_retries"] == 5

    def test_custom_endpoint_provider_different_providers(self):
        """Test provider with different provider types."""
        from litellmnodes import LiteLLMCustomEndpointProvider

        provider = LiteLLMCustomEndpointProvider()

        # Test with anthropic provider
        result = provider.handler(
            model_name="claude-3-sonnet-20240229",
            api_base="https://api.anthropic.com",
            api_key="sk-ant-test-key",
            provider="anthropic"
        )

        model_config = result[0]
        assert model_config["model"] == "anthropic/claude-3-sonnet-20240229"

    def test_node_mappings_updated(self):
        """Test that the new node is added to NODE_CLASS_MAPPINGS."""
        try:
            from litellmnodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
            from litellmnodes import LiteLLMCustomEndpointProvider

            assert "LiteLLMCustomEndpointProvider" in NODE_CLASS_MAPPINGS
            assert NODE_CLASS_MAPPINGS["LiteLLMCustomEndpointProvider"] == LiteLLMCustomEndpointProvider

            # Check that the display name exists (it uses DISPLAY_NAME, not class name)
            assert "Custom Endpoint Provider" in NODE_DISPLAY_NAME_MAPPINGS
            assert isinstance(NODE_DISPLAY_NAME_MAPPINGS["Custom Endpoint Provider"], str)

        except ImportError:
            pytest.fail("NODE_CLASS_MAPPINGS or NODE_DISPLAY_NAME_MAPPINGS not properly updated")


class TestCustomEndpointIntegration:
    """Test integration of custom endpoint with existing completion nodes."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks for dependencies."""
        mock_config = Mock()
        mock_config.config_settings = {'tmp_dir': tempfile.gettempdir()}
        sys.modules['config'] = mock_config

        mock_utils = Mock()
        mock_custom_dict = Mock()
        mock_custom_dict.CustomDict = dict
        mock_utils.custom_dict = mock_custom_dict
        sys.modules['utils'] = mock_utils
        sys.modules['utils.custom_dict'] = mock_custom_dict

    def test_completion_node_accepts_custom_model(self):
        """Test that completion nodes can accept and use custom model configs."""
        from litellmnodes import LiteLLMCustomEndpointProvider, LitellmCompletionV2

        # Create custom endpoint model
        provider = LiteLLMCustomEndpointProvider()
        model_config = provider.handler(
            model_name="mistralai/Mistral-Nemo-Instruct-2407",
            api_base="https://api.kluster.ai/v1",
            api_key="test-key",
            provider="openai"
        )[0]

        # Test that completion node can accept this model
        completion = LitellmCompletionV2()

        # Mock the litellm.completion call to verify it receives correct params
        with patch('litellm.completion') as mock_completion:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = "Test response"
            mock_response.choices[0].message.role = "assistant"
            mock_response.usage = Mock()
            mock_response.usage.model_extra = {}
            mock_completion.return_value = mock_response

            try:
                # This should work without errors
                result = completion.litellm_completion_v2_inner(
                    frequency_penalty=0.0,
                    max_tokens=100,
                    messages=[{"role": "user", "content": "Hello"}],
                    model=model_config,
                    presence_penalty=0.0,
                    prompt="Hello",
                    task="completion",
                    temperature=1.0,
                    top_p=1.0
                )

                # Verify litellm.completion was called with correct parameters
                assert mock_completion.called
                call_kwargs = mock_completion.call_args[1]

                # Should have custom endpoint parameters
                assert call_kwargs["api_key"] == "test-key"
                assert call_kwargs["api_base"] == "https://api.kluster.ai/v1"
                assert call_kwargs["model"] == "openai/mistralai/Mistral-Nemo-Instruct-2407"

            except Exception as e:
                pytest.fail(f"Completion node should handle custom model config: {e}")

    def test_agent_node_integration(self):
        """Test that agent nodes can use custom endpoint configurations."""
        # This will depend on how agent nodes integrate with the model provider
        # For now, we'll test that the model config format is compatible
        from litellmnodes import LiteLLMCustomEndpointProvider

        provider = LiteLLMCustomEndpointProvider()
        model_config = provider.handler(
            model_name="mistralai/Mistral-Nemo-Instruct-2407",
            api_base="https://api.kluster.ai/v1",
            api_key="test-key",
            provider="openai"
        )[0]

        # Model config should be in the format expected by agent nodes
        assert isinstance(model_config, dict)
        assert "model" in model_config
        assert "kwargs" in model_config
        assert isinstance(model_config["kwargs"], dict)


class TestCustomEndpointValidation:
    """Test validation and error handling for custom endpoint provider."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks for dependencies."""
        mock_config = Mock()
        mock_config.config_settings = {'tmp_dir': tempfile.gettempdir()}
        sys.modules['config'] = mock_config

        mock_utils = Mock()
        mock_custom_dict = Mock()
        mock_custom_dict.CustomDict = dict
        mock_utils.custom_dict = mock_custom_dict
        sys.modules['utils'] = mock_utils
        sys.modules['utils.custom_dict'] = mock_custom_dict

    def test_empty_model_name_validation(self):
        """Test validation when model name is empty."""
        from litellmnodes import LiteLLMCustomEndpointProvider

        provider = LiteLLMCustomEndpointProvider()

        # Should handle empty model name gracefully
        result = provider.handler(
            model_name="",
            api_base="https://api.example.com",
            api_key="test-key",
            provider="openai"
        )

        # Should still create a model config, even if model name is empty
        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_empty_api_base_validation(self):
        """Test validation when API base is empty."""
        from litellmnodes import LiteLLMCustomEndpointProvider

        provider = LiteLLMCustomEndpointProvider()

        # Should handle empty API base
        result = provider.handler(
            model_name="gpt-4",
            api_base="",
            api_key="test-key",
            provider="openai"
        )

        # Should still create a model config
        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_provider_formats_model_correctly(self):
        """Test that different provider types format model strings correctly."""
        from litellmnodes import LiteLLMCustomEndpointProvider

        provider = LiteLLMCustomEndpointProvider()

        test_cases = [
            ("openai", "gpt-4", "openai/gpt-4"),
            ("anthropic", "claude-3-sonnet", "anthropic/claude-3-sonnet"),
            ("cohere", "command-r", "cohere/command-r"),
            ("custom", "local-model", "custom/local-model"),
        ]

        for provider_name, model_name, expected_model in test_cases:
            result = provider.handler(
                model_name=model_name,
                api_base="https://api.example.com",
                api_key="test-key",
                provider=provider_name
            )

            model_config = result[0]
            assert model_config["model"] == expected_model


class TestKlusterIntegration:
    """Specific tests for Kluster.ai integration."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks for dependencies."""
        mock_config = Mock()
        mock_config.config_settings = {'tmp_dir': tempfile.gettempdir()}
        sys.modules['config'] = mock_config

        mock_utils = Mock()
        mock_custom_dict = Mock()
        mock_custom_dict.CustomDict = dict
        mock_utils.custom_dict = mock_custom_dict
        sys.modules['utils'] = mock_utils
        sys.modules['utils.custom_dict'] = mock_custom_dict

    def test_kluster_configuration(self):
        """Test specific Kluster.ai configuration."""
        from litellmnodes import LiteLLMCustomEndpointProvider

        provider = LiteLLMCustomEndpointProvider()

        # Get API key from environment
        api_key = os.environ.get('KLUSTER_API_KEY', 'test-kluster-key')

        # Test exact Kluster.ai configuration
        result = provider.handler(
            model_name="mistralai/Mistral-Nemo-Instruct-2407",
            api_base="https://api.kluster.ai/v1",
            api_key=api_key,
            provider="openai"
        )

        model_config = result[0]

        # Verify Kluster.ai specific configuration
        assert model_config["model"] == "openai/mistralai/Mistral-Nemo-Instruct-2407"
        assert model_config["kwargs"]["api_base"] == "https://api.kluster.ai/v1"
        assert model_config["kwargs"]["api_key"] == api_key

    @patch('litellm.completion')
    def test_kluster_api_call_format(self, mock_completion):
        """Test that the API call to Kluster.ai is formatted correctly."""
        from litellmnodes import LiteLLMCustomEndpointProvider, LitellmCompletionV2

        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.role = "assistant"
        mock_response.usage = Mock()
        mock_response.usage.model_extra = {}
        mock_completion.return_value = mock_response

        # Create Kluster model config
        provider = LiteLLMCustomEndpointProvider()
        api_key = os.environ.get('KLUSTER_API_KEY', 'test-key')
        model_config = provider.handler(
            model_name="mistralai/Mistral-Nemo-Instruct-2407",
            api_base="https://api.kluster.ai/v1",
            api_key=api_key,
            provider="openai"
        )[0]

        # Use in completion
        completion = LitellmCompletionV2()
        completion.litellm_completion_v2_inner(
            frequency_penalty=0.0,
            max_tokens=50,
            messages=[{"role": "user", "content": "Hello"}],
            model=model_config,
            presence_penalty=0.0,
            prompt="Hello",
            task="completion",
            temperature=0.7,
            top_p=1.0
        )

        # Verify the call was made with correct Kluster.ai parameters
        assert mock_completion.called
        call_kwargs = mock_completion.call_args[1]

        assert call_kwargs["model"] == "openai/mistralai/Mistral-Nemo-Instruct-2407"
        assert call_kwargs["api_base"] == "https://api.kluster.ai/v1"
        assert call_kwargs["api_key"] == api_key
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]
        assert call_kwargs["max_tokens"] == 50
        assert call_kwargs["temperature"] == 0.7


if __name__ == "__main__":
    # Run tests standalone for quick verification
    print("Running TDD tests for Custom Endpoint Provider...")

    # Basic smoke test
    try:
        # This will fail until we implement the feature
        from litellmnodes import LiteLLMCustomEndpointProvider
        print("✅ LiteLLMCustomEndpointProvider exists")
    except ImportError:
        print("❌ LiteLLMCustomEndpointProvider not implemented yet - this is expected for TDD")

    print("\nRun with pytest for full test suite:")
    print("python3 -m pytest tests/test_custom_endpoint.py -v")
