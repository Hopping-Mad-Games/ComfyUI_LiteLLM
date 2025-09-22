import importlib

import pytest


def _reload_litellmnodes(monkeypatch, tmp_path, api_base="https://api.openai.com"):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", api_base)
    import sys
    for module_name in [
        "config",
        "utils",
        "utils.custom_dict",
        "utils.env_config",
        "litellmnodes",
    ]:
        sys.modules.pop(module_name, None)

    import litellmnodes

    # Ensure environment-driven configuration is re-read for each test.
    module = importlib.reload(litellmnodes)
    monkeypatch.setitem(module.config.config_settings, "tmp_dir", str(tmp_path))
    return module


@pytest.fixture
def litellm_module(monkeypatch, tmp_path):
    return _reload_litellmnodes(monkeypatch, tmp_path)


def test_provider_wraps_openai_models_with_kwargs(litellm_module):
    provider = litellm_module.LiteLLMModelProviderAdv()

    model_config = provider._build_litellm_model("openai/gpt-4o")

    assert isinstance(model_config, dict)
    assert model_config["model"] == "openai/gpt-4o"
    assert model_config["type"] == "kwargs"
    assert model_config["kwargs"]["api_key"] == "sk-test"
    # Normalized base URL should include the version suffix exactly once.
    assert model_config["kwargs"]["api_base"] == "https://api.openai.com/v1"


def test_completion_v2_uses_provider_kwargs(litellm_module, monkeypatch, tmp_path):
    provider = litellm_module.LiteLLMModelProviderAdv()
    model_config = provider._build_litellm_model("openai/gpt-4o")

    captured = {}

    class DummyMessage:
        def __init__(self):
            self.content = "ok"
            self.role = "assistant"

    class DummyChoice:
        def __init__(self):
            self.message = DummyMessage()

    class DummyUsage:
        def __init__(self):
            self.model_extra = {"total_tokens": 1}

    class DummyResponse:
        def __init__(self):
            self.choices = [DummyChoice()]
            self.usage = DummyUsage()

    def fake_completion(**kwargs):
        captured["kwargs"] = kwargs
        return DummyResponse()

    monkeypatch.setattr(litellm_module.litellm, "completion", fake_completion)

    node = litellm_module.LitellmCompletionV2()

    result = node.handler(
        model=model_config,
        prompt="Hello",
        max_tokens=16,
        temperature=0.1,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        task="completion",
    )

    assert captured["kwargs"]["model"] == "openai/gpt-4o"
    assert captured["kwargs"]["api_key"] == "sk-test"
    assert captured["kwargs"]["api_base"] == "https://api.openai.com/v1"

    # Ensure the node still returns the structured tuple with the model payload intact.
    assert isinstance(result, tuple)
    assert result[0] == model_config
    assert isinstance(result[2], str)


def test_completion_v2_switches_to_max_completion_tokens(monkeypatch, litellm_module, tmp_path):
    provider_cls = litellm_module.LiteLLMModelProviderAdv
    monkeypatch.setattr(
        provider_cls,
        "_openai_model_metadata",
        {
            "openai/gpt-4.1": {"capabilities": None, "requires_max_completion_tokens": True},
            "gpt-4.1": {"capabilities": None, "requires_max_completion_tokens": True},
        },
    )

    provider = provider_cls()
    model_config = provider._build_litellm_model("openai/gpt-4.1")

    captured = {}

    class DummyMessage:
        def __init__(self):
            self.content = "ok"
            self.role = "assistant"

    class DummyChoice:
        def __init__(self):
            self.message = DummyMessage()

    class DummyUsage:
        def __init__(self):
            self.model_extra = {"total_tokens": 1}

    class DummyResponse:
        def __init__(self):
            self.choices = [DummyChoice()]
            self.usage = DummyUsage()

    def fake_completion(**kwargs):
        captured["kwargs"] = kwargs
        return DummyResponse()

    monkeypatch.setattr(litellm_module.litellm, "completion", fake_completion)

    node = litellm_module.LitellmCompletionV2()

    node.handler(
        model=model_config,
        prompt="Hello",
        max_tokens=32,
        temperature=0.1,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        task="completion",
    )

    kwargs = captured["kwargs"]
    assert "max_tokens" not in kwargs
    assert kwargs["max_completion_tokens"] == 32
