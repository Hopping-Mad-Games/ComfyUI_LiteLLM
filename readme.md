# ComfyUI_LiteLLM

ComfyUI_LiteLLM is an addon for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that provides nodes for interacting with various language models using the [LiteLLM](https://github.com/BerriAI/litellm) library directly within the ComfyUI interface. This addon serves as a bridge between ComfyUI and LiteLLM, allowing you to leverage any model supported by LiteLLM in your ComfyUI workflows.

## Features

This addon provides ComfyUI nodes that interface with LiteLLM's functionality, including:

- `LiteLLMModelProvider`: Selects LiteLLM-supported models for use in other nodes
- `LiteLLMCompletion`: Generates completions using any LiteLLM-supported model
- `LiteLLMCompletionPrePend`: Similar to LiteLLMCompletion but with a pre-pended prompt
- `LiteLLMCompletionListOfPrompts`: Generates completions for a list of prompts
- `CreateReflectionFilter` and `FirstCodeBlockReflectionFilter`: Creates reflection filters for use with `LiteLLMCompletionWithReflectionFilter`
- `LiteLLMCompletionWithReflectionFilter`: Generates completions with a reflection filter
- `LiteLLMCompletionProvider`: Provides a completion function for use in other nodes
- `LiteLLMMessage`: Creates LiteLLM messages
- `ShowLastMessage` and `ShowMessages`: Display message content in the UI
- `ListToMessages` and `MessagesToList`: Convert between list and message formats
- `AppendMessages`: Combine multiple message sets
- `MessagesToText` and `TextToMessages`: Convert between text and message formats
- `LLLMInterpretUrl`: Fetch and process content from URLs

## Supported Models

This addon supports all models that LiteLLM can interface with. This includes, but is not limited to:

- OpenAI models
- Anthropic models (Claude)
- Google models (PaLM, Gemini)
- Azure OpenAI
- AWS Bedrock models
- Cohere models
- AI21 models
- And many more

For the most up-to-date list of supported models, please refer to the [LiteLLM documentation](https://docs.litellm.ai/docs/).

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

Please ensure that your API keys are stored in environment variables and not saved to the graph. You can, at your discretion, put them in the `config.yaml` file, but it is not recommended for security reasons.

## Usage

After installation, the LiteLLM nodes will be available in the ComfyUI node editor under the "ETK/LLM/LiteLLM" category. You can use these nodes to integrate any LiteLLM-supported language model into your ComfyUI workflows.

To use a specific model, simply select it in the `LiteLLMModelProvider` node, and ensure you have the necessary API keys and configurations set up as per LiteLLM's requirements.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This addon is released under the [MIT License](LICENSE).

## Dependencies

- pyyaml
- addict
- litellm
- boto3

For the full list of dependencies, please refer to the `requirements.txt` file.
