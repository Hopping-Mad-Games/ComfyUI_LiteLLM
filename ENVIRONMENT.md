# Environment Configuration for ComfyUI_LiteLLM

This document explains how to configure environment variables and API keys for ComfyUI_LiteLLM.

## Quick Setup

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your actual API keys:**
   ```bash
   nano .env  # or use your preferred editor
   ```

3. **Set your API keys in the `.env` file:**
   ```env
   KLUSTER_API_KEY=your-actual-kluster-api-key-here
   OPENAI_API_KEY=your-actual-openai-api-key-here
   ```

## Environment Variables

### Required for Testing
- `KLUSTER_API_KEY` - Your Kluster.ai API key (required for integration tests)
- `KLUSTER_BASE_URL` - Kluster.ai API base URL (default: https://api.kluster.ai/v1)

### Optional API Keys
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic (Claude) API key  
- `COHERE_API_KEY` - Cohere API key

### Test Configuration
- `RUN_INTEGRATION_TESTS` - Set to `true` to run integration tests (default: false)

### Development Settings
- `DEBUG` - Enable debug mode (default: false)
- `LOG_LEVEL` - Set logging level (default: INFO)

## Security Best Practices

### ✅ DO
- Use environment variables for all API keys
- Keep your `.env` file local and never commit it
- Use the `.env.example` file as a template
- Rotate API keys regularly
- Use different keys for development and production

### ❌ DON'T
- Hardcode API keys in source code
- Commit `.env` files to version control
- Share API keys in plain text
- Use production keys for testing

## File Structure

```
ComfyUI_LiteLLM/
├── .env.example          # Template for environment variables
├── .env                  # Your actual environment variables (gitignored)
├── .gitignore           # Ensures .env files are not committed
├── config.yaml          # Application configuration
└── utils/
    └── env_config.py    # Environment configuration utility
```

## Usage in Code

### Loading Environment Variables

```python
from utils.env_config import get_env_var, get_api_config

# Get a single environment variable
api_key = get_env_var('KLUSTER_API_KEY', required=True)

# Get all API configuration
config = get_api_config()
kluster_key = config['kluster']['api_key']
```

### Kluster.ai Configuration

```python
from utils.env_config import get_kluster_config

try:
    config = get_kluster_config()
    # Use config['api_key'], config['base_url'], config['model']
except ValueError as e:
    print(f"Kluster configuration error: {e}")
```

## Testing

### Running Tests with Environment Variables

```bash
# Set environment variables and run tests
export KLUSTER_API_KEY="your-key-here"
python -m pytest tests/

# Or use .env file (recommended)
echo "KLUSTER_API_KEY=your-key-here" > .env
python -m pytest tests/
```

### Skipping Tests Without API Keys

Tests will automatically skip if required API keys are not configured:

```
SKIPPED [1] Kluster API key not configured. Set KLUSTER_API_KEY environment variable.
```

### Integration Tests

Set `RUN_INTEGRATION_TESTS=true` in your `.env` file to enable tests that make real API calls:

```env
RUN_INTEGRATION_TESTS=true
KLUSTER_API_KEY=your-actual-key
```

## Troubleshooting

### Common Issues

1. **"Kluster API key not configured"**
   - Ensure `KLUSTER_API_KEY` is set in your environment or `.env` file
   - Check that the `.env` file is in the project root directory

2. **"Tests are skipped"**
   - This is normal behavior when API keys are not configured
   - Add the required API keys to run integration tests

3. **"Environment file not found"**
   - Create a `.env` file based on `.env.example`
   - Ensure the file is in the correct location (project root)

### Debug Environment Loading

```python
from utils.env_config import load_env_file
import os

# Check what's loaded from .env file
env_vars = load_env_file()
print("Loaded from .env:", env_vars)

# Check what's in the actual environment
print("KLUSTER_API_KEY in env:", os.environ.get('KLUSTER_API_KEY'))
```

## Migration from Hardcoded Keys

If you're migrating from an older version with hardcoded API keys:

1. **Find hardcoded keys in your code:**
   ```bash
   grep -r "46b1a513-72b6-4549-aac3-0d2f8ae16db9" .
   ```

2. **Replace with environment variables:**
   ```python
   # Old (insecure)
   api_key = "46b1a513-72b6-4549-aac3-0d2f8ae16db9"
   
   # New (secure)
   api_key = os.environ.get('KLUSTER_API_KEY')
   ```

3. **Add keys to .env file:**
   ```env
   KLUSTER_API_KEY=46b1a513-72b6-4549-aac3-0d2f8ae16db9
   ```

4. **Test the migration:**
   ```bash
   python -m pytest tests/ -v
   ```

## Production Deployment

For production environments:

1. **Don't use `.env` files** - Set environment variables directly
2. **Use secrets management** - Tools like HashiCorp Vault, AWS Secrets Manager, etc.
3. **Monitor key usage** - Set up alerts for API key usage and errors
4. **Implement key rotation** - Regularly rotate API keys

Example production setup:
```bash
# Set environment variables in your deployment
export KLUSTER_API_KEY="prod-key-here"
export OPENAI_API_KEY="prod-openai-key"
export RUN_INTEGRATION_TESTS="false"
export LOG_LEVEL="WARNING"
```
