"""
pytest configuration for ComfyUI_LiteLLM test suite.
Essential configurations for test isolation and mock setup.
"""

import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock
import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Disable warnings for cleaner output
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

@pytest.fixture(scope="session", autouse=True)
def mock_dependencies():
    """Mock problematic dependencies before any imports."""

    # Mock LightRAG package
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

    # Mock ComfyUI specific modules
    sys.modules['folder_paths'] = Mock()
    sys.modules['execution'] = Mock()
    sys.modules['server'] = Mock()

    return {
        'lightrag': mock_lightrag,
        'config': mock_config,
        'utils': mock_utils
    }

@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for tests."""
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp(prefix="comfyui_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def kluster_api_config():
    try:
        from utils.env_config import get_kluster_config
        return get_kluster_config()
    except (ImportError, ValueError):
        # Fallback for tests when env config is not available
        api_key = os.environ.get('KLUSTER_API_KEY')
        if not api_key:
            pytest.skip("Kluster API key not configured. Set KLUSTER_API_KEY environment variable.")

        return {
            'api_key': api_key,
            'base_url': os.environ.get('KLUSTER_BASE_URL', 'https://api.kluster.ai/v1'),
            'model': "mistralai/Mistral-Nemo-Instruct-2407"
        }

@pytest.fixture(autouse=True)
def prevent_infinite_loops():
    """Prevent infinite loops in tests."""
    import signal
    import time

    def timeout_handler(signum, frame):
        raise TimeoutError("Test exceeded 30 second timeout - possible infinite loop")

    # Set up timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 second timeout

    yield

    # Clean up
    signal.alarm(0)
    signal.signal(signal.SIGALRM, old_handler)

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark API tests
        if "api" in item.nodeid.lower() or "kluster" in item.nodeid.lower():
            item.add_marker(pytest.mark.api)

        # Mark integration tests
        if "real" in item.nodeid.lower() or "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)

# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.api = pytest.mark.api
