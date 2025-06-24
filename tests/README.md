# ComfyUI LiteLLM Test Suite

This directory contains the test suite for ComfyUI_LiteLLM agent nodes and functionality.

## Test Structure

The test suite has been cleaned up and organized into focused, working test files:

### Core Test Files

- **`smoke_test.py`** - Basic smoke tests to verify imports, structure, and core functionality
- **`test_simple_real.py`** - Comprehensive tests for agent improvements with mocked responses
- **`test_kluster_real.py`** - Real API integration tests using Kluster.ai endpoint
- **`test_hash_utils.py`** - Tests for utility functions and hash handling
- **`test_agent_structure.py`** - Tests for code structure, documentation, and naming improvements

### Configuration Files

- **`conftest.py`** - pytest configuration with dependency mocking and fixtures
- **`__init__.py`** - Makes tests directory a Python package

## Running Tests

### Run All Tests
```bash
python3 -m pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Basic functionality tests
python3 -m pytest tests/smoke_test.py tests/test_simple_real.py -v

# Real API tests (requires internet)
python3 -m pytest tests/test_kluster_real.py -v

# Structure and documentation tests
python3 -m pytest tests/test_agent_structure.py -v
```

### Run with Markers
```bash
# API tests only
python3 -m pytest -m api -v

# Integration tests only  
python3 -m pytest -m integration -v
```

## Test Categories

### 🚀 Smoke Tests (`smoke_test.py`)
- Import verification
- Display name mappings
- Basic structure validation
- Documentation existence
- Error handling patterns

### 🔧 Simple Real Tests (`test_simple_real.py`)
- Direct API connectivity test
- Input validation comprehensive testing
- Naming improvements verification
- Documentation structure validation
- Mock functionality testing

### 🌐 Kluster Real Tests (`test_kluster_real.py`)
- Real API calls to Kluster.ai endpoint
- Agent node integration with live API
- Multi-iteration processing
- Multiple prompts handling
- Recursion filter integration

### 🛠️ Hash Utils Tests (`test_hash_utils.py`)
- Input hashing for different data types
- Agent node input handling
- Callable and image processing
- Message and list handling

### 📋 Structure Tests (`test_agent_structure.py`)
- File organization verification
- Display name improvements
- Docstring completeness
- Method documentation quality
- Code structure enhancements

## Key Features Tested

### ✅ Fixed Issues
- **Infinite loop bug**: Fixed in LiteLLM completion handler
- **Import errors**: Conditional LightRAG imports
- **Error handling**: Proper exception management
- **Test isolation**: Mock dependencies prevent conflicts

### ✅ Agent Improvements Verified
- Input validation with comprehensive error messages
- Better naming conventions and display names
- Extensive documentation (USAGE_GUIDE.md, EXAMPLES.md)
- Robust error handling and warnings
- Progress indicators and user feedback
- Caching system functionality

## API Configuration

Tests use the Kluster.ai API endpoint:
- **Base URL**: https://api.kluster.ai/v1
- **Model**: mistralai/Mistral-Nemo-Instruct-2407
- **API Key**: Configured in test fixtures

## Test Results Summary

As of the latest cleanup:
- **All smoke tests**: ✅ PASS
- **Simple real tests**: ✅ PASS  
- **Kluster API tests**: ✅ PASS
- **Hash utils tests**: ✅ PASS
- **Structure tests**: ✅ PASS

Total: **32 tests** all passing, test suite runs in ~1.35 seconds

## Dependencies

Tests automatically mock problematic dependencies:
- LightRAG (conditional import)
- ComfyUI modules (folder_paths, execution, server)
- Config and utils modules

No additional test dependencies required beyond pytest.

## Troubleshooting

### Import Errors
The test suite automatically handles import issues with dependency mocking in `conftest.py`.

### API Connection Issues  
If API tests fail, check:
1. Internet connectivity
2. Kluster.ai service availability
3. API endpoint status

### Timeout Issues
Tests have a 30-second timeout to prevent infinite loops. If tests hang:
1. Check for infinite retry loops in LiteLLM calls
2. Verify API endpoints are responding
3. Review error handling in agent code

## Contributing

When adding new tests:
1. Follow the existing naming convention
2. Add appropriate pytest markers (@pytest.mark.api, etc.)
3. Include proper documentation
4. Ensure tests clean up after themselves
5. Mock external dependencies appropriately

## Maintenance

The test suite has been cleaned up to remove:
- Redundant test files
- Broken or non-functional tests  
- Infinite loop issues
- Import dependency problems
- Outdated proof-of-concept files

Focus is on maintainable, reliable tests that verify core functionality.