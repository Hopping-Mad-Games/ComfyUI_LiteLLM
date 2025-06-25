"""
Environment configuration utility for ComfyUI_LiteLLM.

This module provides utilities for loading environment variables
from .env files and providing defaults for testing.
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any


def load_env_file(env_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load environment variables from a .env file.

    Args:
        env_path: Path to the .env file. If None, looks for .env in the project root.

    Returns:
        Dictionary of environment variables loaded from the file.
    """
    if env_path is None:
        # Look for .env file in the project root
        current_dir = Path(__file__).parent.parent
        env_path = current_dir / ".env"

    env_vars = {}

    if not os.path.exists(env_path):
        return env_vars

    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                # Parse key=value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]

                    env_vars[key] = value
    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")

    return env_vars


def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Get an environment variable with fallback to .env file.

    Args:
        key: Environment variable name
        default: Default value if not found
        required: If True, raises ValueError if not found and no default

    Returns:
        Environment variable value or default

    Raises:
        ValueError: If required=True and variable not found
    """
    # First check actual environment variables
    value = os.environ.get(key)

    if value is None:
        # Load from .env file if not in environment
        env_vars = load_env_file()
        value = env_vars.get(key)

    if value is None:
        if required and default is None:
            raise ValueError(f"Required environment variable '{key}' not found")
        value = default

    return value


def get_api_config() -> Dict[str, Any]:
    """
    Get API configuration from environment variables.

    Returns:
        Dictionary containing API configuration
    """
    return {
        'kluster': {
            'api_key': get_env_var('KLUSTER_API_KEY'),
            'base_url': get_env_var('KLUSTER_BASE_URL', 'https://api.kluster.ai/v1'),
        },
        'openai': {
            'api_key': get_env_var('OPENAI_API_KEY'),
            'base_url': get_env_var('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
        },
        'anthropic': {
            'api_key': get_env_var('ANTHROPIC_API_KEY'),
        },
        'cohere': {
            'api_key': get_env_var('COHERE_API_KEY'),
        },
    }


def should_run_integration_tests() -> bool:
    """
    Check if integration tests should be run based on environment configuration.

    Returns:
        True if integration tests should run, False otherwise
    """
    run_tests = get_env_var('RUN_INTEGRATION_TESTS', 'false').lower()
    return run_tests in ('true', '1', 'yes', 'on')


def get_kluster_config() -> Dict[str, str]:
    """
    Get Kluster.ai API configuration.

    Returns:
        Dictionary with Kluster API configuration

    Raises:
        ValueError: If Kluster API key is not configured
    """
    api_key = get_env_var('KLUSTER_API_KEY')
    base_url = get_env_var('KLUSTER_BASE_URL', 'https://api.kluster.ai/v1')

    if not api_key:
        raise ValueError(
            "Kluster API key not found. Please set KLUSTER_API_KEY in your environment "
            "or .env file. See .env.example for template."
        )

    return {
        'api_key': api_key,
        'base_url': base_url,
        'model': 'mistralai/Mistral-Nemo-Instruct-2407'
    }


# Initialize environment on import
_env_vars = load_env_file()
