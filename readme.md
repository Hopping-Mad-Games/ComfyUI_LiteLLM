# ComfyUI_LiteLLM

ComfyUI_LiteLLM is an addon for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that provides nodes for interacting with [LiteLLM](https://github.com/Lightning-AI/lit-llama) models directly within the ComfyUI interface.

## Features

- Various nodes for working with LiteLLM models:
  - `LiteLLMModelProvider`: Provides LiteLLM models for use in other nodes
  - `LiteLLMCompletion`: Generates completions using LiteLLM models
  - `LiteLLMCompletionListOfPrompts`: Generates completions for a list of prompts
  - `CreateReflectionFilter` and `FirstCodeBlockReflectionFilter`: Creates reflection filters for use with `LiteLLMCompletionWithReflectionFilter`
  - `LiteLLMCompletionWithReflectionFilter`: Generates completions with a reflection filter
  - `LiteLLMCompletionProvider`: Provides a completion function for use in other nodes
  - `LiteLLMMessage`: Creates LiteLLM messages
  - `ListToMessages`: Converts a list to LiteLLM messages

## Installation

### Manual Installation

1. Clone this repository into your `custom_nodes` directory in your ComfyUI installation.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Restart ComfyUI.

### Installation with ComfyUI-Manager

1. Add this addon's repository URL to your ComfyUI-Manager configuration.
2. Install the addon through ComfyUI-Manager.
3. Restart ComfyUI.

## Configuration

The addon reads configuration settings from `config.yaml`. You can customize the settings by modifying this file.

Please ensure that your API keys are stored in environment variables and not saved to the graph. You can, at your discretion, put them in the `config.yaml` file, but it is not recommended.

## Usage

After installation, the LiteLLM nodes will be available in the ComfyUI node editor under the "ETK/LLM/LiteLLM" category. You can use these nodes to integrate LiteLLM models into your ComfyUI workflows.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This addon is released under the [MIT License](LICENSE).