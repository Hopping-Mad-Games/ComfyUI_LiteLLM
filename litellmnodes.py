

try:
    from . import config
    from .utils.custom_dict import CustomDict
    from .utils.env_config import get_env_var
except ImportError:
    import config
    from utils.custom_dict import CustomDict
    from utils.env_config import get_env_var

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def litellm_base(cls):
    cls.CATEGORY = "ETK/LLM/LiteLLM"
    cls.FUNCTION = "handler"

    if "RETURN_TYPES" not in cls.__dict__:
        cls.RETURN_TYPES = ("STRING",)

    # Add spaces to the camel case class name
    pretty_name = cls.__name__
    for i in range(1, len(pretty_name)):
        if pretty_name[i].isupper():
            pretty_name = pretty_name[:i] + " " + pretty_name[i:]
    cls.DISPLAY_NAME = pretty_name
    NODE_CLASS_MAPPINGS[cls.__name__] = cls
    NODE_DISPLAY_NAME_MAPPINGS[cls.DISPLAY_NAME] = pretty_name
    return cls


@litellm_base
class HTMLRenderer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "html_content": ("STRING", {"multiline": True}),
                "iframe_height": ("STRING", {"default": "800px"}),
                "iframe_width": ("STRING", {"default": "100%"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "handler"
    OUTPUT_NODE = True

    def handler(self, html_content, iframe_height, iframe_width):
        new_html_content = html_content.replace("\"", "'")
        new_html_content = f"""<iframe srcdoc="{new_html_content}" style="width: {iframe_width}; height: {iframe_height};border: none;"></iframe>"""

        # Replace newlines and double quotes in the raw HTML

        ret = {"ui": {"string": [new_html_content]}, "result": (html_content,)}
        return ret



import io
import numpy as np

@litellm_base
class HTMLRendererScreenshot:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "html_content": ("STRING", {"multiline": True, "default": "<h2>Hello World!</h2>"}),
                "screenshot_base64": ("STRING", {"default": ""}),
            }
        }

    # We will return two things:
    #   1) An IMAGE => your screenshot as a float BHWC tensor
    #   2) A STRING => the actual iframe code (so the client can display the HTML)
    RETURN_TYPES = ("IMAGE", "STRING",)
    FUNCTION = "handler"
    OUTPUT_NODE = True

    NAME = "HTMLRendererScreenshot"

    def handler(self, html_content, screenshot_base64):
        magic = """setTimeout(() => {
        html2canvas(document.body, {
            allowTaint : true,
            logging: true,
            profile: true,
            useCORS: true
            }).then(function(canvas) {
            document.getElementById('screen').appendChild(canvas);
        }); }000);"""

        # 1) Create the iframe code from the user's HTML
        #    We replace " to ' in the HTML to avoid messing up the srcdoc attribute.
        # safe_html = html_content.replace("'", '"')
        safe_html = html_content.replace('"', "'")
        iframe_code = """<iframe id="htmlRendererScreenshotFrame" srcdoc="{safe_html}" style="width:100%; height:800px; border:none;"> </iframe> """
        # iframe_code = iframe_code.replace("\n", " ")
        # iframe_code = iframe_code.replace("\r", " ")
        # iframe_code = iframe_code.replace("\t", " ")
        iframe_code = iframe_code.replace("{safe_html}", safe_html)

        # 2) If we have no screenshot, return a 1×1 black image as a placeholder
        if not screenshot_base64.strip():
            # shape = (batch=1, height=1, width=1, channels=3)
            arr = np.zeros((1, 1, 1, 3), dtype=np.float32)
            tensor = torch.from_numpy(arr)
            ret = {"ui": {"string": ["no image"]}, "result": (tensor, html_content,)}
            return ret

        # Otherwise, decode the base64 screenshot into a float [0..1] BHWC tensor
        raw_bytes = base64.b64decode(screenshot_base64)
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        arr = np.array(pil_img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
        tensor = torch.from_numpy(arr).contiguous()

        # Return both the image and the iframe code
        # return (tensor, iframe_code)
        ret = {"ui": {"string": [iframe_code]}, "result": (tensor, html_content,)}
        return ret


import markdown


@litellm_base
class MarkdownNode:
    """
    A ComfyUI node that renders Markdown input as HTML in the content area while
    passing the raw Markdown string as output.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "markdown_input": ("STRING", {"multiline": True, "default": "Enter Markdown text here..."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Markdown Output",)
    CATEGORY = "ETK\LLM\Markdown"

    def handler(self, markdown_input):
        # Render the Markdown to HTML
        rendered_html = markdown.markdown(markdown_input)

        # Return the required framework structure
        ret = {
            "ui": {"string": [rendered_html]},  # Rendered HTML for the UI
            "result": (markdown_input,),  # Raw Markdown output
        }
        return ret


@litellm_base
class LiteLLMModelProviderAdv:
    # Hard-coded pricing (per 1M tokens) for known modern OpenAI models.
    # Pricing values are based on your provided table.
    OPENAI_MODEL_COSTS = {
        # GPT-4o and variants:
        "openai/gpt-4o": "Input: $2.50/1M, Cached: $1.25/1M, Output: $10.00/1M tokens",
        "openai/gpt-4o-2024-08-06": "Input: $2.50/1M, Cached: $1.25/1M, Output: $10.00/1M tokens",
        # GPT-4o audio-preview variants:
        "openai/gpt-4o-audio-preview": "Input: $2.50/1M, Output: $10.00/1M tokens",
        "openai/gpt-4o-audio-preview-2024-12-17": "Input: $2.50/1M, Output: $10.00/1M tokens",
        # GPT-4o realtime-preview variants:
        "openai/gpt-4o-realtime-preview": "Input: $5.00/1M, Cached: $2.50/1M, Output: $20.00/1M tokens",
        "openai/gpt-4o-realtime-preview-2024-12-17": "Input: $5.00/1M, Cached: $2.50/1M, Output: $20.00/1M tokens",
        # GPT-4o mini variants:
        "openai/gpt-4o-mini": "Input: $0.15/1M, Cached: $0.075/1M, Output: $0.60/1M tokens",
        "openai/gpt-4o-mini-2024-07-18": "Input: $0.15/1M, Cached: $0.075/1M, Output: $0.60/1M tokens",
        # GPT-4o mini audio-preview variants:
        "openai/gpt-4o-mini-audio-preview": "Input: $0.15/1M, Output: $0.60/1M tokens",
        "openai/gpt-4o-mini-audio-preview-2024-12-17": "Input: $0.15/1M, Output: $0.60/1M tokens",
        # GPT-4o mini realtime-preview variants:
        "openai/gpt-4o-mini-realtime-preview": "Input: $0.60/1M, Cached: $0.30/1M, Output: $2.40/1M tokens",
        "openai/gpt-4o-mini-realtime-preview-2024-12-17": "Input: $0.60/1M, Cached: $0.30/1M, Output: $2.40/1M tokens",
        # o1 variants:
        "openai/o1": "Input: $15.00/1M, Cached: $7.50/1M, Output: $60.00/1M tokens",
        "openai/o1-2024-12-17": "Input: $15.00/1M, Cached: $7.50/1M, Output: $60.00/1M tokens",
        # o1-mini variants:
        "openai/o1-mini": "Input: $1.10/1M, Cached: $0.55/1M, Output: $4.40/1M tokens",
        "openai/o1-mini-2024-09-12": "Input: $1.10/1M, Cached: $0.55/1M, Output: $4.40/1M tokens",
        # o3-mini variants:
        "openai/o3-mini": "Input: $1.10/1M, Cached: $0.55/1M, Output: $4.40/1M tokens",
        "openai/o3-mini-2025-01-31": "Input: $1.10/1M, Cached: $0.55/1M, Output: $4.40/1M tokens",
        # Also include GPT-3.5 Turbo and GPT-4 for completeness:
        "openai/gpt-3.5-turbo": "$2.00/1M tokens",
        "openai/gpt-3.5-turbo-16k": "Input: $3.00/1M, Output: $4.00/1M tokens",
        "openai/gpt-4": "Input: $30/1M, Output: $60/1M tokens (8K)",
        "openai/gpt-4-32k": "Input: $60/1M, Output: $120/1M tokens (32K)",
    }

    # Define the input types for the node; the displayed list is a list of strings with pricing info.
    @classmethod
    def INPUT_TYPES(cls):
        models = cls.get_all_model_display_strings()
        default = models[0] if models else ""
        return {
            "required": {
                "name": (models, {"default": default}),
            }
        }

    # Merge models fetched from OpenAI with hard-coded provider models.
    @classmethod
    def get_all_model_display_strings(cls):
        openai_ids = cls.get_openai_models()  # Fetched OpenAI model IDs (list of strings)
        other_ids = cls.get_other_models()  # Hard-coded models from other providers
        openai_display = [cls.format_model_display(model) for model in openai_ids]
        model_set = set(openai_ids)
        other_display = []
        for model in other_ids:
            if model not in model_set:
                other_display.append(cls.format_model_display(model))
        return openai_display + other_display

    # Format the display string for a model.
    @classmethod
    def format_model_display(cls, model):
        if model.startswith("openai/"):
            cost = cls.OPENAI_MODEL_COSTS.get(model, "cost unknown")
            return f"{model} ({cost})"
        else:
            return model

    # Fetch all OpenAI models using the new client instance.
    @classmethod
    def get_openai_models(cls):
        # Check if OpenAI API key is available and configured
        api_key = get_env_var("OPENAI_API_KEY")
        if not api_key or api_key.strip() == "" or api_key == "your-api-key-here":
            # Return fallback list of common OpenAI models if no API key configured
            return [
                "openai/gpt-4o-mini",
                "openai/gpt-4o",
                "openai/gpt-4o-2024-08-06",
                "openai/gpt-3.5-turbo",
                "openai/o1-mini",
                "openai/o1"
            ]

        import os
        from openai import OpenAI
        # Use configured OPENAI_BASE_URL from config settings
        base_url = config.config_settings.get("OPENAI_BASE_URL")
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        try:
            response = client.models.list()
            # Only include models whose IDs contain one of these substrings.
            valid_substrings = ["gpt-3.5", "gpt-4", "gpt-4o", "o1", "o3"]
            q = [
                "openai/" + model.id for model in response.data
                if any(sub in model.id for sub in valid_substrings)
            ]
            return q if q else ["openai/gpt-4o-mini"]  # Fallback if no models found
        except Exception as e:
            print(f"Warning: Could not fetch OpenAI models: {str(e)}")
            # Return fallback list of common OpenAI models on error
            return [
                "openai/gpt-4o-mini",
                "openai/gpt-4o",
                "openai/gpt-4o-2024-08-06",
                "openai/gpt-3.5-turbo"
            ]

    # Hard-coded list of models from other providers.
    @classmethod
    def get_other_models(cls):
        return [
            "anthropic/claude-3-haiku-20240307",
            "anthropic/claude-3-sonnet-20240229",
            "anthropic/claude-3-opus-20240229",
            "anthropic/claude-3-5-sonnet-20240620",
            "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            "bedrock/anthropic.claude-3-opus-20240229-v1:0",
            "bedrock/anthropic.claude-3-5-sonnet-20240620",
            "vertex_ai/gemini-1.5-pro-preview-0514",
            "vertex_ai/gemini-1.5-flash-preview-0514",
            "vertex_ai/meta/llama3-405b-instruct-maas",
            "together_ai/deepseek-ai/DeepSeek-V3",
            "together_ai/meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        ]

    # Define the return types for the node.
    RETURN_TYPES = ("LITELLM_MODEL",)
    RETURN_NAMES = ("Litellm model",)

    # Handler: Extract the actual model ID (by removing the pricing info) and return it.
    def handler(self, **kwargs):
        selected = kwargs.get("name", None)
        if selected and " (" in selected:
            model_id = selected.split(" (")[0]
        else:
            model_id = selected
        return (model_id,)


@litellm_base
class LiteLLMModelProvider:
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        # Inputs can be used to customize the default architecture
        return {
            "required": {
                # ([True, False], {"default": True})
                "name": ([
                             "anthropic/claude-3-haiku-20240307",
                             "anthropic/claude-3-sonnet-20240229",
                             "anthropic/claude-3-opus-20240229",
                             "anthropic/claude-3-5-sonnet-20240620",
                             "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
                             "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                             "bedrock/anthropic.claude-3-opus-20240229-v1:0",
                             "bedrock/anthropic.claude-3-5-sonnet-20240620",
                             "vertex_ai/gemini-1.5-pro-preview-0514",
                             "vertex_ai/gemini-1.5-flash-preview-0514",
                             "vertex_ai/meta/llama3-405b-instruct-maas",
                             "openai/gpt-4o-mini",
                             "openai/gpt-4o",
                             "openai/gpt-4o-2024-08-06",
                             "together_ai/deepseek-ai/DeepSeek-V3",
                             "together_ai/meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",

                         ],

                         {"default": "anthropic/claude-3-haiku-20240307"}),
                # "norm_zero_to_1": ([True, False], {"default": True}),
            }
        }

    # Define the return types
    RETURN_TYPES = ("LITELLM_MODEL",)
    RETURN_NAMES = ("Litellm model",)

    # Method to provide the default LiteLLM model
    def handler(self, **kwargs):
        # Get the model name from the input
        model_name = kwargs.get("name", None)
        return (model_name,)


@litellm_base
class LiteLLMCustomEndpointProvider:
    """
    Custom Endpoint Provider for LiteLLM

    Allows configuration of custom API endpoints, API keys, and models
    for platforms like Kluster.ai, local servers, or other OpenAI-compatible APIs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "mistralai/Mistral-Nemo-Instruct-2407"}),
                "api_base": ("STRING", {"default": "https://api.kluster.ai/v1"}),
                "api_key": ("STRING", {"default": "your-api-key-here"}),
                "provider": (["openai", "anthropic", "cohere", "custom"], {"default": "openai"}),
            },
            "optional": {
                "api_version": ("STRING", {"default": ""}),
                "organization": ("STRING", {"default": ""}),
                "timeout": ("INT", {"default": 60, "min": 1, "max": 300}),
                "max_retries": ("INT", {"default": 3, "min": 0, "max": 10}),
            }
        }

    RETURN_TYPES = ("LITELLM_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "handler"
    CATEGORY = "ETK/LLM"

    def handler(self, model_name, api_base, api_key, provider, **kwargs):
        # Create model configuration with custom endpoint
        model_config = {
            "model": f"{provider}/{model_name}",
            "kwargs": {
                "api_key": api_key,
                "api_base": api_base,
                "timeout": kwargs.get("timeout", 60),
                "max_retries": kwargs.get("max_retries", 3),
            }
        }

        # Add optional parameters if provided
        if kwargs.get("api_version"):
            model_config["kwargs"]["api_version"] = kwargs["api_version"]
        if kwargs.get("organization"):
            model_config["kwargs"]["organization"] = kwargs["organization"]

        return (model_config,)


# Fix the display name after the decorator has processed it
LiteLLMCustomEndpointProvider.DISPLAY_NAME = "Custom Endpoint Provider"
NODE_DISPLAY_NAME_MAPPINGS[LiteLLMCustomEndpointProvider.DISPLAY_NAME] = "Custom Endpoint Provider"


@litellm_base
class AddDataModelToLLLm:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LITELLM_MODEL", {"default": "anthropic/claude-3-haiku-20240307"}),
            },
            "optional": {
                "code": (
                    "STRING",
                    {"default": "class UserModel(BaseModel):\n    name: str= Field(..., description='The name of the "
                                "person')\n    age: int\n", "multiline": True}),
                "data_model": ("DATA_MODEL", {"default": None}),
            }
        }

    RETURN_TYPES = ("LITELLM_MODEL",)
    RETURN_NAMES = ("Litellm model",)

    def restricted_exec(self, code: str):
        from pydantic import BaseModel, Field, conlist
        from typing import Tuple, List, Dict, Set, Any, Callable, Union, Optional, Iterable, Iterator, Generator

        # Define a restricted global namespace
        restricted_globals = {
            "__builtins__": {
                # Basic functions
                "print": print,
                "range": range,
                "len": len,
                "__build_class__": __build_class__,  # For class definitions

                # Primitive types
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "complex": complex,

                # Collection types
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "frozenset": frozenset,
                "bytes": bytes,
                "bytearray": bytearray,

                # Typing counterparts
                "Tuple": Tuple,
                "List": List,
                "Dict": Dict,
                "Set": Set,
                "Any": Any,
                "Callable": Callable,
                "Union": Union,
                "Optional": Optional,
                "Iterable": Iterable,
                "Iterator": Iterator,
                "Generator": Generator,
            },
            "__name__": "__main__",  # Define __name__ to simulate normal execution context
            "BaseModel": BaseModel,  # Explicitly allow BaseModel from pydantic
            "Field": Field,  # Explicitly allow Field to define field descriptions
            "conlist": conlist  # Explicitly allow conlist to define list constraints
        }

        # Execute the code with restricted globals
        exec(code, restricted_globals)

        # Extract the created model from the restricted_globals
        # Assuming the user will name their model 'UserModel'
        user_model = restricted_globals.get("UserModel")
        if user_model is None:
            raise ValueError("No model named 'UserModel' was defined in the provided code.")
        return user_model

    def handler(self, model, code=None, data_model=None):
        use_data_model = None
        # first figure out the data_model
        if data_model is not None:
            use_data_model = data_model
        elif isinstance(code, str):
            use_data_model = self.restricted_exec(code)

        if isinstance(model, dict):
            model["kwargs"]["response_format"] = use_data_model
        elif isinstance(model, str):
            model = {"model": model,
                     "type": "kwargs",
                     "kwargs": {"response_format": use_data_model}
                     }

        return (model,)


@litellm_base
class ModifyModelKwargs:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LITELLM_MODEL", {"default": "anthropic/claude-3-haiku-20240307"}),
            },
            "optional": {
                "kwargs": ("DICT", {"default": {}}),
                "json_str_of_kwargs": ("STRING", {"default": "{}", "multiline": True}),
            }
        }

    RETURN_TYPES = ("LITELLM_MODEL",)
    RETURN_NAMES = ("Litellm model",)

    def handler(self, model, kwargs={}, json_str_of_kwargs={}, data_model=None):
        import json

        if json_str_of_kwargs != "{}":
            kwargs = json.loads(json_str_of_kwargs)

        if "logit_bias" in kwargs:
            if isinstance(kwargs["logit_bias"], list):
                l_logits = kwargs["logit_bias"]
                set_logits = dict()
                for id, p in l_logits:
                    set_logits[str(id)] = int(p)
                kwargs["logit_bias"] = set_logits

        if isinstance(model, dict):
            model["kwargs"].update(kwargs)
        elif isinstance(model, str):
            model = {"model": model, "type": "kwargs", "kwargs": kwargs}

        return (model,)


@litellm_base
class LiteLLMCompletion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LITELLM_MODEL", {"default": "anthropic/claude-3-haiku-20240307"}),
                "max_tokens": ("INT", {"default": 250, "min": 1, "max": 1e10, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frequency_penalty": ("FLOAT", {"default": 0}),
                "presence_penalty": ("FLOAT", {"default": 0}),
                "prompt": ("STRING", {"default": "Hello World!", "multiline": True}),
                "reasoning_effort": (["low", "medium", "high"], {"default": "low"}),
            },
            "optional": {
                "messages": ("LLLM_MESSAGES", {"default": None}),
                "use_cached_response": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LITELLM_MODEL", "LLLM_MESSAGES", "STRING", "STRING",)
    RETURN_NAMES = ("Model", "Messages", "Completion", "Usage",)

    def handler(self, **kwargs):
        from copy import deepcopy
        litellm.drop_params = True
        kwargs = deepcopy(kwargs)
        model = kwargs.get('model', 'anthropic/claude-3-haiku-20240307')
        out_model = deepcopy(kwargs.get('model', 'anthropic/claude-3-haiku-20240307'))

        if kwargs["top_p"] == 0:
            kwargs["top_p"] = None
        if kwargs["temperature"] == 0:
            kwargs["temperature"] = None
        if kwargs["frequency_penalty"] == 0:
            kwargs["frequency_penalty"] = None
        if kwargs["presence_penalty"] == 0:
            kwargs["presence_penalty"] = None
        reasoning_effort = kwargs.get('reasoning_effort', "low")
        if isinstance(kwargs["model"], dict):
            _model = model.get("model", "anthropic/claude-3-haiku-20240307")
            _type = model.get("type", None)
            if _type:
                if _type == "kwargs":
                    _kwargs = model.get("kwargs", {})
                    kwargs.update(_kwargs)
                    kwargs["model"] = _model
            model = _model

        messages = kwargs.get('messages', [])
        messages = deepcopy(messages)
        max_tokens = kwargs.get('max_tokens', None)
        temperature = kwargs.get('temperature', None)
        top_p = kwargs.get('top_p', None)
        frequency_penalty = kwargs.get('frequency_penalty', None)
        presence_penalty = kwargs.get('presence_penalty', None)
        prompt = kwargs.pop('prompt', "Hello World!")
        use_cached_response = kwargs.get('use_cached_response', False)

        messages.append({"content": prompt, "role": "user"})
        kwargs["messages"] = messages

        import hashlib
        kwargs.pop('use_cached_response', None)
        uid_kwargs = CustomDict()
        uid_kwargs.update(kwargs.copy())
        uid_kwargs["prompt"] = prompt
        uid_kwargs = str(uid_kwargs.every_value_str())

        unique_id = str(hashlib.sha256(json.dumps(uid_kwargs).encode()).hexdigest())
        del uid_kwargs

        cache_file_name = f'cached_response_{unique_id}.json'
        cache_file_path = os.path.join(config.config_settings['tmp_dir'], cache_file_name)

        response = None
        cached_completion = False

        if use_cached_response and os.path.exists(cache_file_path):
            with open(cache_file_path, 'r') as file:
                response = json.load(file)
                response = litellm.ModelResponse(**response)
                cached_completion = True

        if not response:
            while not response:
                try:
                    litellm.set_verbose: True

                    n_kwargs = {
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "frequency_penalty": frequency_penalty,
                        "presence_penalty": presence_penalty,
                        "reasoning_effort": reasoning_effort,
                    }
                    n_kwargs.update(kwargs)

                    if ("o1" in model) or ("o3" in model):
                        n_kwargs["max_completion_tokens"] = n_kwargs.pop("max_tokens")
                        n_kwargs.pop("temperature", None)
                        n_kwargs.pop("top_p", None)
                    else:
                        n_kwargs.pop("reasoning_effort", None)

                    response = litellm.completion(
                        **n_kwargs
                    )

                except litellm.exceptions.RateLimitError as e:
                    print(f"Rate limit error: {e}")

                    time.sleep(5)
                    continue
                except Exception as e:
                    print(f"An error occurred: {e}")

                    raise

        response_choices = response.choices
        response_first_choice = response_choices[0]
        response_first_choice_message = response_first_choice.message
        response_content = response_first_choice_message.content or ""

        if not os.path.exists(config.config_settings['tmp_dir']):
            os.makedirs(config.config_settings['tmp_dir'])

        if os.path.exists(cache_file_path):
            os.remove(cache_file_path)
        with open(cache_file_path, 'w') as file:
            json.dump(response.json(), file, indent=4)

        messages.append({"content": response_content, "role": response_first_choice_message.role})

        try:
            d = response.usage.model_extra
            lines = [f"{k}:{v}" for k, v in d.items()]
            usage = "\n".join(lines)
        except Exception as e:
            usage = ""

        if cached_completion:
            top = "cached response used\n"
            top = f"\033[1;32;40m{top}\033[0m"
            usage = top + usage
        else:
            top = "new result generated\n"
            top = f"\033[1;31;40m{top}\033[0m"
            usage = top + usage

        return (out_model, messages, response_content, usage,)


import os
import json
import hashlib
import time
import base64
from copy import deepcopy

import litellm
api_base = config.config_settings.get("OPENAI_BASE_URL", "https://api.openai.com/v1/")
key = get_env_var("BASE_API_KEY")
if api_base:
    litellm.api_base = api_base
if key:
    litellm.api_key = key
litellm.set_verbose=True

import torch
from PIL import Image
from io import BytesIO


@litellm_base
class LitellmCompletionV2:

    @classmethod
    def get_valid_tasks(cls):
        # Ideally, fetch this list dynamically from LiteLLM documentation or API
        return ["transcription", "classification", "completion", "translation", "summarization", "image_captioning",
                "object_detection"]

    @classmethod
    def INPUT_TYPES(cls):
        valid_tasks = cls.get_valid_tasks()
        default = LiteLLMCompletion.INPUT_TYPES()["required"]
        default["task"] = (valid_tasks, {"default": "completion"})

        out = {
            "required": default,
            "optional": {
                "image": ("IMAGE", {"default": None}),  # Assuming image_tensor is the PyTorch tensor
                "messages": ("LLLM_MESSAGES", {"default": None}),
                "use_cached_response": ("BOOLEAN", {"default": False}),
            }
        }
        return out

    RETURN_TYPES = ("LITELLM_MODEL", "LLLM_MESSAGES", "STRING", "LIST", "STRING",)
    RETURN_NAMES = ("Model", "Messages", "Completion", "[Completions]", "Usage",)

    def tensor_image_to_base64(self, tensor):
        tensor = tensor.mul(255).byte()  # Convert to 0-255
        image = Image.fromarray(tensor.numpy())
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=100)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def tensor_image_stack_to_base64(self, tensor):
        b, h, w, c = tensor.shape
        out = []
        for i in range(b):
            img = self.tensor_image_to_base64(tensor[i])
            img_data = self.get_image_data(img)
            out.append(img_data)

        return out

    def get_image_data(self, base64_image):
        image_data = {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64," + base64_image
            }
        }
        return image_data

    def handler(self, **kwargs):
        from pydantic import BaseModel
        from copy import deepcopy
        kwargs = deepcopy(kwargs)
        litellm.drop_params = True
        single_response = True
        task = kwargs.get('task', "transcription")
        image = kwargs.get('image', None)

        # Validate the task
        valid_tasks = self.get_valid_tasks()
        if task not in valid_tasks:
            raise ValueError(f"Invalid task '{task}'. Valid tasks are: {valid_tasks}")

            # Optional parameters handling
        if kwargs.get("top_p", 0) == 0:
            kwargs["top_p"] = 0
        if kwargs.get("temperature", 0) == 0:
            kwargs["temperature"] = 0
        if kwargs.get("frequency_penalty", 0) == 0:
            kwargs["frequency_penalty"] = 0
        if kwargs.get("presence_penalty", 0) == 0:
            kwargs["presence_penalty"] = 0

        prompt = kwargs.get('prompt', "Hello World!")

        # Get input parameters
        model = kwargs.get('model', 'anthropic/claude-3.5-sonnet')
        messages = kwargs.get('messages', [])
        messages.append({"role": "user", "content": prompt})
        messages = deepcopy(messages)

        max_tokens = kwargs.get('max_tokens', None)
        temperature = kwargs.get('temperature', None)
        top_p = kwargs.get('top_p', None)
        frequency_penalty = kwargs.get('frequency_penalty', None)
        presence_penalty = kwargs.get('presence_penalty', None)

        use_cached_response = kwargs.get('use_cached_response', False)

        # Cache handling setup
        kwargs.pop('use_cached_response', None)
        kwargs.pop('image', None)
        mdl_cls = None
        if isinstance(kwargs["model"], dict):
            if "response_format" in kwargs["model"]["kwargs"]:
                # if kwargs["model"]["kwargs"]["response_format"].__name__ == "UserModel":
                mdl_cls = kwargs["model"]["kwargs"]["response_format"]
                if isinstance(mdl_cls, dict):
                    kwargs["model"]["kwargs"]["response_format"] = mdl_cls
                else:
                    kwargs["model"]["kwargs"]["response_format"] = mdl_cls.schema_json()

        unique_id = str(hashlib.sha256(json.dumps(kwargs).encode()).hexdigest())

        if mdl_cls:
            kwargs["model"]["kwargs"]["response_format"] = mdl_cls

        cache_file_name = f'cached_response_{unique_id}.json'
        cache_file_path = os.path.join(config.config_settings['tmp_dir'], cache_file_name)

        response = None
        cached_completion = False

        responses = None
        # Check for cached response
        if use_cached_response and os.path.exists(cache_file_path):
            try:
                with open(cache_file_path, 'r') as file:
                    responses = json.load(file)
                    responses_out = []
                    if isinstance(responses, list):
                        for r in responses:
                            responses_out.append(litellm.ModelResponse(**r))
                        responses = responses_out
                        cached_completion = True
                    else:
                        responses = None
                        print("cached response not a list.")
            except Exception as e:
                print(f"Error loading cached response: {e}")
                responses = None
        base64_images = [None]
        # Encode image to base64 if the task requires image processing
        if task in ["transcription", "classification", "image_captioning", "object_detection"]:
            base64_images = self.tensor_image_stack_to_base64(image)
            if len(base64_images) > 1:
                single_response = False

        if not responses:
            responses = []
            for id in base64_images:
                response = None
                response_content = ""
                response_first_choice_message = None
                while not response:
                    try:
                        if id:
                            if id is not None:
                                response = self.litellm_completion_v2_inner(
                                    frequency_penalty = frequency_penalty,
                                    max_tokens = max_tokens,
                                    messages = messages,
                                    model = model,
                                    presence_penalty = presence_penalty,
                                    prompt=prompt,
                                    task=task,
                                    temperature=temperature,
                                    top_p=top_p,
                                    image_data=id
                                    )
                        else:
                                response = self.litellm_completion_v2_inner(
                                    frequency_penalty,
                                    max_tokens,
                                    messages,
                                    model,
                                     presence_penalty,
                                    prompt,
                                    task,
                                    temperature,
                                    top_p
                                )

                        if response:
                            response_choices = response.choices
                            response_first_choice = response_choices[0]
                            response_first_choice_message = response_first_choice.message
                            response_content = response_first_choice_message.content or ""
                            responses.append(response_content)
                    except litellm.BadRequestError as e:
                        print(f"Bad request error: {e}")
                        response_content = f"Error: {e}"
                        responses.append(response_content)
                        break
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        response_content = f"Error: {e}"
                        responses.append(response_content)
                        break

        if not os.path.exists(config.config_settings['tmp_dir']):
            os.makedirs(config.config_settings['tmp_dir'])

        if os.path.exists(cache_file_path):
            os.remove(cache_file_path)
        with open(cache_file_path, 'w') as file:
            json.dump(responses, file, indent=4)

        usage = ""
        if single_response and response_first_choice_message:
            messages.append({"content": response_content, "role": response_first_choice_message.role})

            try:
                d = response.usage.model_extra
                lines = [f"{k}:{v}" for k, v in d.items()]
                usage = "\n".join(lines)
            except Exception as e:
                usage = ""

        if cached_completion:
            top = "cached response used\n"
            top = f"\033[1;32;40m{top}\033[0m"
            usage = top + usage
        else:
            top = "new result generated\n"
            top = f"\033[1;31;40m{top}\033[0m"
            usage = top + usage

        return (model, messages, responses[0], responses, usage,)

    def litellm_completion_v2_inner(self, frequency_penalty, max_tokens, messages, model, presence_penalty,
                                    prompt, reasoning_effort="low", task="completion", temperature=1.0, top_p=1.0, image_data=None):
        from copy import deepcopy
        import json
        # deepcopy all the inputs
        frequency_penalty = deepcopy(frequency_penalty)
        presence_penalty = deepcopy(presence_penalty)
        max_tokens = deepcopy(max_tokens)
        messages = deepcopy(messages)
        model = deepcopy(model)
        presence_penalty = deepcopy(presence_penalty)
        prompt = deepcopy(prompt)
        task = deepcopy(task)
        temperature = deepcopy(temperature)
        top_p = deepcopy(top_p)

        try:
            litellm.set_verbose = True
            if task in ["transcription", "classification", "image_captioning",
                        "object_detection"]:  # Tasks that require vision API

                if image_data:
                    last_message = messages[-1]
                    if isinstance(last_message["content"], str):
                        message_type = "text"
                        new_message = {"type": "text", "text": last_message["content"]}
                        last_message["content"] = [new_message]
                    else:
                        message_type = last_message["content"][-1]["type"]

                    if message_type == "text":
                        last_message["content"].append(image_data)

                use_kwargs = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "reasoning_effort": reasoning_effort,
                }

                # if isinstance(model, dict):
                #     if model[model]
                # else:
                #     if ("o1" in model) or ("o3" in model):
                #         o_model=True
                #     else:
                #         o_model=False

                if ("o1" in model) or ("o3" in model):
                    use_kwargs["max_completion_tokens"] = use_kwargs.pop("max_tokens")
                    use_kwargs.pop("temperature", None)
                    use_kwargs.pop("top_p", None)
                    use_kwargs.pop("logit_bias", None)
                    print("o1 or o3")
                else:
                    use_kwargs.pop("reasoning_effort", None)
                    print("NOT o1 or o3")
                print(list(use_kwargs.keys()))
                response = litellm.completion(**use_kwargs)


            else:  # Other tasks
                base_schema = {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "user_model",
                            "strict": True,
                            "schema": {
                                "$schema": "http://json-schema.org/draft-07/schema#",
                                "type": "object",
                                "additionalProperties": False
                            }
                        }
                    }
                }

                use_kwargs = {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "prompt": prompt
                }

                if isinstance(model, dict):
                    if "kwargs" in model:
                        # Extract and update with custom kwargs (api_key, api_base, etc.)
                        use_kwargs.update(model["kwargs"])

                        # Always fix model string when we have a dict with kwargs
                        use_kwargs["model"] = model["model"]

                        if "response_format" in use_kwargs:
                            base_scm_use = deepcopy(base_schema)
                            if hasattr(use_kwargs["response_format"], "schema"):
                                scm = use_kwargs["response_format"].schema()
                                base_scm_use["response_format"]["json_schema"]["schema"].update(scm)
                                use_kwargs["response_format"] = json.dumps(base_scm_use["response_format"])

                            if isinstance(use_kwargs["response_format"], str):
                                ob = json.loads(use_kwargs["response_format"])
                            else:
                                ob = use_kwargs["response_format"]

                            if "$defs" in ob["json_schema"]["schema"]:
                                defs = ob["json_schema"]["schema"]["$defs"]
                                for k, v in defs.items():
                                    v["additionalProperties"] = False

                            use_kwargs["response_format"] = ob

                use_kwargs.pop("prompt", None)

                if ("o1" in use_kwargs["model"]) or ("o3" in use_kwargs["model"]):
                    use_kwargs["max_completion_tokens"] = use_kwargs.pop("max_tokens")
                    use_kwargs.pop("temperature", None)
                    use_kwargs.pop("top_p", None)
                    use_kwargs.pop("logit_bias", None)
                    print("o1 or o3")
                else:
                    use_kwargs.pop("reasoning_effort", None)
                    print("NOT o1 or o3")
                print(list(use_kwargs.keys()))

                response = litellm.completion(
                    **use_kwargs
                )
        except litellm.exceptions.RateLimitError as e:
            print(f"Rate limit error: {e}")
            time.sleep(5)
        except Exception as e:
            print(f"An error occurred: {e}")
            raise
        return response


@litellm_base
class PDFToImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pdf_path": ("STRING", {"default": "path/to/your/pdf/file.pdf"}),
                "dpi": ("INT", {"default": 200, "min": 72, "max": 600}),
                 # Page numbers are 1-based for user input consistency
                "first_page": ("INT", {"default": 1, "min": 1}),
                 # -1 or 0 means process until the end
                "last_page": ("INT", {"default": -1}),
            },
        }

    # Output is a batch of images suitable for ComfyUI IMAGE type
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "handler" # Function to execute

    def handler(self, pdf_path, dpi, first_page, last_page):
        import pymupdf,sys
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        page_tensors = []
        doc = None # Initialize doc to None for finally block

        try:
            # 1. Open PDF with PyMuPDF
            doc = pymupdf.open(pdf_path)
            num_pages = doc.page_count

            # 2. Determine page range (0-based index for PyMuPDF)
            start_index = max(0, first_page - 1)
            # Handle last_page: <=0 means 'to the end'
            if last_page <= 0 or last_page > num_pages:
                end_index = num_pages - 1
            else:
                end_index = last_page - 1 # Convert 1-based to 0-based

            # Check for invalid range
            if start_index >= num_pages or start_index > end_index:
                raise ValueError(f"Invalid page range ({first_page}-{last_page}) for PDF with {num_pages} pages.")

            # 3. Calculate zoom factor from DPI (base is 72 DPI for PDFs)
            zoom_factor = dpi / 72.0
            render_matrix = pymupdf.Matrix(zoom_factor, zoom_factor)

            # 4. Iterate through the specified page range
            for i in range(start_index, end_index + 1):
                page = doc.load_page(i)

                # 5. Render page to RGB pixmap using get_pixmap
                #    alpha=False ensures 3 channels (RGB)
                pix = page.get_pixmap(matrix=render_matrix, alpha=False)
                height, width, channels = pix.height, pix.width, pix.n

                # Ensure we got an RGB image (3 channels)
                if channels != 3:
                    print(f"Warning: Page {i+1} rendered with {channels} channels (expected 3 for RGB). Skipping page.", file=sys.stderr)
                    # Attempt conversion if RGBA? For simplicity, we skip non-RGB here.
                    # If RGBA (channels=4), you could potentially slice: np_array = np_array[:, :, :3]
                    continue

                # 6. Convert pixmap samples to NumPy array
                #    samples are bytes: height * width * channels
                np_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(height, width, channels)

                # 7. Convert NumPy array to PyTorch Tensor (float32, normalized 0-1)
                #    from_numpy avoids data copy where possible
                page_tensor = torch.from_numpy(np_array).float().div(255.0) # Shape: (H, W, C)

                page_tensors.append(page_tensor)

        except Exception as e:
            raise RuntimeError(f"Error processing PDF '{pdf_path}': {e}") from e
        finally:
             # Ensure the document is closed even if errors occur
             if doc:
                 doc.close()

        # Check if any pages were processed
        if not page_tensors:
            # Raise error or return specific empty tensor? Framework dependent.
            # Raising error is often clearer than returning ambiguous empty tensor.
            raise ValueError(f"No pages were successfully rendered for PDF '{pdf_path}' in the specified range.")

        # 8. Stack page tensors into a batch tensor (B, H, W, C)
        try:
            # torch.stack adds the batch dimension (B)
            batch_images = torch.stack(page_tensors, dim=0)
        except RuntimeError as stack_error:
            # This error typically occurs if pages have different dimensions
            # even when rendered at the same DPI (e.g., different aspect ratios).
             raise RuntimeError(
                 f"Failed to stack page images for '{pdf_path}'. "
                 f"Pages might have different dimensions after rendering at {dpi} DPI. Error: {stack_error}"
             ) from stack_error


        # Return the stacked tensor in a tuple, as expected by ComfyUI
        return (batch_images,)


@litellm_base
class ShowLastMessage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "messages": ("LLLM_MESSAGES", {"default": "Something to show..."}),
            },
            "optional": {
                "list_display": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    INPUT_IS_LIST = False
    RETURN_TYPES = ("LLLM_MESSAGES",)
    RETURN_NAMES = ("messages",)
    # INTERNAL_STATE_DISPLAY = "list_display"

    # THIS IS BRYTHON CODE
    # INTERNAL_STATE = ""
    FUNCTION = "handler"
    OUTPUT_NODE = True
    # OUTPUT_IS_LIST = (True,)
    description = "Show list in browser"
    CATEGORY = "utils"

    def handler(self, **kwargs):
        from copy import deepcopy
        kwargs = deepcopy(kwargs)
        lst = kwargs.get("messages", None)

        text_d = []
        d = lst[-1]
        who = d.get("role")
        content = d.get("content")

        s = ""
        s += f"{who.upper()}:\n"
        s += f"{content}\n"
        s += "\n"

        text_d.append(s)

        res = lst
        out_list = ["\n".join(text_d)]
        ret = {"ui": {"string": out_list}, "result": (res,)}
        return ret


@litellm_base
class ShowMessages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "messages": ("LLLM_MESSAGES", {"default": "Something to show..."}),
            },
            "optional": {
                "list_display": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    INPUT_IS_LIST = False
    RETURN_TYPES = ("LLLM_MESSAGES",)
    RETURN_NAMES = ("messages",)
    INTERNAL_STATE_DISPLAY = "list_display"

    # THIS IS BRYTHON CODE
    INTERNAL_STATE = ""
    FUNCTION = "handler"
    OUTPUT_NODE = True
    # OUTPUT_IS_LIST = (True,)
    description = "Show list in browser"
    CATEGORY = "utils"

    def handler(self, **kwargs):
        from copy import deepcopy
        kwargs = deepcopy(kwargs)
        """
        input is list : List[List[str]]
        Returns a dict to be displayed by the UI and passed to the next function
        output like ret["ui"]["list"] needs to be a list of strings
        output like ret["result"] should mirror the input

        >>> ShowList.notify([["Hello", "world"], ["How", "are", "you?"]])
        {'ui': {'list': ["Hello", "world", "How", "are", "you?"]}, 'result': [["Hello", "world"], ["How", "are", "you?"]]}

        >>> ShowList.notify([])
        {'ui': {'list': []}, 'result': []}
        """
        lst = kwargs.get("messages", None)
        text_d = []
        for d in lst:
            who = d.get("role")
            content = d.get("content")

            s = ""
            s += f"{who.upper()}:\n"
            s += f"{content}\n"
            s += "\n"

            text_d.append(s)

        res = lst

        out_list = ["\n".join(text_d)]
        ret = {"ui": {"string": out_list}, "result": (res,)}

        return ret


@litellm_base
class LiteLLMCompletionPrePend(LiteLLMCompletion):
    """just like LiteLLMCompletion but with a pre-pended prompt"""

    @classmethod
    def INPUT_TYPES(cls):
        base = LiteLLMCompletion.INPUT_TYPES()
        base["required"]["pre_prompt"] = ("STRING", {"default": "pre", "multiline": True})
        _ = base["required"].pop("prompt")
        base["required"]["prompt"] = ("STRING", {"default": "prompt", "multiline": True})
        return base

    RETURN_TYPES = LiteLLMCompletion.RETURN_TYPES

    def handler(self, **kwargs):
        pp = kwargs.pop("pre_prompt", None)
        kwargs["prompt"] = f"{pp}\n{kwargs['prompt']}"
        return super().handler(**kwargs)


@litellm_base
@litellm_base
class LiteLLMCompletionListOfPrompts:
    """Calls LiteLLMCompletion for each item.
       If 'prompts' list is provided, it iterates through prompts, using corresponding
       message history (optional) and prepending pre_prompt to each prompt.
       If 'prompts' is empty/None, it iterates through the 'messages' list,
       using pre_prompt as the new user message content for each history.
       Returns completions and resulting messages lists."""

    @classmethod
    def INPUT_TYPES(cls):
        # Use the definition from the provided LiteLLMCompletion
        base = LiteLLMCompletion.INPUT_TYPES()

        # Modify for list processing: Set up required fields
        if "required" in base and "prompt" in base["required"]:
             base["required"]["pre_prompt"] = base["required"].pop("prompt")
        elif "required" not in base:
             base["required"] = {}
             base["required"]["pre_prompt"] = ("STRING", {"multiline": True, "default": "Hello World!"})

        base["required"]["prompts"] = ("LIST", {"default": None})
        base["required"]["async"] = ("BOOLEAN", {"default": True})
        base["optional"]["messages"] =("LIST", {"default": None})

        return base

    # Return types remain the same
    RETURN_TYPES = ("LITELLM_MODEL", "LIST", "LIST", "STRING",)
    RETURN_NAMES = ("Model", "Completions", "Messages", "Usage",)

    # Helper function remains the same
    def wrapped_llm_call(self, index, **kwargs):
        return (index, LiteLLMCompletion().handler(**kwargs))

    # Core async processing function remains the same
    async def async_process_prompt(self, index, prompt, pre_prompt, message_history, **kwargs):
        import asyncio
        import functools
        combined_prompt = f"{pre_prompt}\n{prompt}" if pre_prompt else prompt
        kwargs["messages"] = message_history
        kwargs["prompt"] = combined_prompt
        loop = asyncio.get_running_loop()
        partial_func = functools.partial(self.wrapped_llm_call, index, **kwargs)
        idx, the_rest = await loop.run_in_executor(None, partial_func)
        model, resulting_messages, completion, usage = the_rest
        return idx, resulting_messages, completion

    # Core sync processing function remains the same
    def process_prompt(self, prompt, pre_prompt, message_history, **kwargs):
        combined_prompt = f"{pre_prompt}\n{prompt}" if pre_prompt else prompt
        kwargs["messages"] = message_history
        kwargs["prompt"] = combined_prompt
        model, resulting_messages, completion, usage = LiteLLMCompletion().handler(**kwargs)
        return resulting_messages, completion

    # Modified async aggregation to handle the two modes
    async def process_prompts(self, prompts, pre_prompt, messages_input_list, use_pre_prompt_as_prompt=False, **kwargs):
        import asyncio

        # Clean kwargs before passing down
        kwargs.pop("pre_prompt", None)
        kwargs.pop("prompts", None)
        kwargs.pop("messages", None)
        kwargs.pop("async", None)

        tasks = []
        if use_pre_prompt_as_prompt:
            # Mode 2: Iterate through messages, use pre_prompt as the new prompt content
            print("Info: No prompts provided, iterating through message histories using pre_prompt.")
            for i, message_history in enumerate(messages_input_list):
                 tasks.append(
                     self.async_process_prompt(
                         i,
                         prompt=pre_prompt,  # Use pre_prompt as the main content
                         pre_prompt="",      # No additional pre_prompt needed here
                         message_history=message_history,
                         **kwargs
                     )
                 )
        else:
            # Mode 1: Original logic - iterate through prompts
             print(f"Info: Processing {len(prompts)} prompts.")
             for i, prompt in enumerate(prompts):
                 tasks.append(
                     self.async_process_prompt(
                         i,
                         prompt=prompt,
                         pre_prompt=pre_prompt,
                         message_history=messages_input_list[i], # Assumes validation happened in handler
                         **kwargs
                     )
                 )

        if not tasks: return [], [] # Return empty if no tasks generated

        results = await asyncio.gather(*tasks) # List of (idx, resulting_messages, completion)
        sorted_results = sorted(results, key=lambda x: x[0])

        completions_list = [completion for idx, messages, completion in sorted_results]
        messages_list = [messages for idx, messages, completion in sorted_results]
        return completions_list, messages_list


    def handler(self, **kwargs):
        import asyncio
        import sys

        prompts = kwargs.pop("prompts", None) # Pop prompts, default None
        pre_prompt = kwargs.pop("pre_prompt", "")
        messages_input_list = kwargs.pop("messages", None) # Pop optional messages
        run_async = kwargs.pop("async", True)

        completions_list = []
        messages_output_list = []
        usage_str = ""
        model_param = kwargs.get("model")

        # --- Determine Mode and Validate Inputs ---
        process_using_prompts = bool(prompts and isinstance(prompts, list) and len(prompts) > 0)
        process_using_messages = bool(messages_input_list and isinstance(messages_input_list, list) and len(messages_input_list) > 0)

        iterator_list = [] # The list we will actually loop over
        mode_kwargs = {}   # kwargs specific to the mode for process_prompts
        message_histories_to_use = []

        if process_using_prompts:
            iterator_list = prompts
            mode_kwargs['use_pre_prompt_as_prompt'] = False
            # Validate messages length against prompts length or default
            if not process_using_messages or len(messages_input_list) != len(prompts):
                 print(f"Warning: Input 'messages' list is None or length mismatch with 'prompts'. Using empty history for prompts.", file=sys.stderr)
                 message_histories_to_use = [[] for _ in prompts]
            else:
                 message_histories_to_use = [hist if isinstance(hist, list) else [] for hist in messages_input_list]

        elif process_using_messages: # No valid prompts, but valid messages list
            iterator_list = messages_input_list # We iterate based on message histories
            mode_kwargs['use_pre_prompt_as_prompt'] = True
            message_histories_to_use = [hist if isinstance(hist, list) else [] for hist in messages_input_list] # Validate inner lists
            if not pre_prompt:
                 print("Warning: No prompts provided, iterating message histories, but 'pre_prompt' is empty. LLM calls might lack new input.", file=sys.stderr)

        else: # Neither prompts nor messages are valid lists
            print("Error: No valid 'prompts' list or 'messages' list provided. Nothing to process.", file=sys.stderr)
            return (model_param, [], [], usage_str,) # Return empty results

        # --- End Mode Determination ---


        if run_async:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Pass necessary lists and the mode flag
                completions_list, messages_output_list = loop.run_until_complete(
                    self.process_prompts(
                        prompts if process_using_prompts else [], # Pass original prompts if relevant
                        pre_prompt,
                        message_histories_to_use, # Pass the validated/defaulted list
                        **mode_kwargs, # Contains use_pre_prompt_as_prompt flag
                        **kwargs      # Pass remaining LLM params
                    )
                )
            finally:
                loop.close()
        else:
            # Sync Path
            if process_using_prompts:
                 # Iterate through prompts
                 for i, prompt in enumerate(iterator_list):
                     message_history = message_histories_to_use[i]
                     resulting_messages, completion = self.process_prompt(
                         prompt=prompt,
                         pre_prompt=pre_prompt,
                         message_history=message_history,
                         **kwargs
                     )
                     messages_output_list.append(resulting_messages)
                     completions_list.append(completion)
            elif process_using_messages:
                 # Iterate through message histories, use pre_prompt as prompt content
                 for i, message_history in enumerate(iterator_list):
                     resulting_messages, completion = self.process_prompt(
                         prompt=pre_prompt, # Use pre_prompt as the new content
                         pre_prompt="",     # No secondary pre_prompt needed
                         message_history=message_history,
                         **kwargs
                     )
                     messages_output_list.append(resulting_messages)
                     completions_list.append(completion)

        return (model_param, completions_list, messages_output_list, usage_str,)



@litellm_base
class CreateReflectionFilter:
    """
    this creates a callable that can be used in the LiteLLMCompletionWithReflectionFilter node
    the inputs are another callable and args and kwargs
    the returned callable will call the input callable with the args and kwargs
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "CALLABLE": ("CALLABLE", {"default": None}),
            },
            "optional": {
                "args": ("LIST", {"default": []}),
                "kwargs": ("DICT", {"default": {}}),
            },
        }

    RETURN_TYPES = ("CALLABLE",)
    RETURN_NAMES = ("Reflection filter",)

    def handler(self, CALLABLE: callable, args: list = [], kwargs: dict = {}):
        _args = args

        def reflection_filter(completion, *__args, **kwargs):
            __args = [completion] + _args
            return CALLABLE(completion, *__args, **kwargs)

        return (reflection_filter,)


@litellm_base
class FirstCodeBlockReflectionFilter:
    """
    this is designed to work as a CALLABLE that can be chained with the CreateReflectionFilter node
    it proveds a callable that will return the first code block in its first input
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "CALLABLE": ("CALLABLE", {"default": None}),
            },
            "optional": {
                "args": ("LIST", {"default": []}),
                "kwargs": ("DICT", {"default": {}}),
            },
        }

    RETURN_TYPES = ("CALLABLE",)
    RETURN_NAMES = ("callable",)

    def handler(self, CALLABLE: callable, args: list = [], kwargs: dict = {}):
        _args = args

        def code_block_filter(completion, *__args, **kwargs):
            if "```" not in completion:
                return completion

            completion = completion.split("```")[1]
            if completion.startswith("python"):
                completion = completion.split("python")[1]
            completion = completion.split("```")[0]
            __args = [completion] + _args
            return CALLABLE(completion, *__args, **kwargs)

        return (code_block_filter,)


@litellm_base
class LiteLLMCompletionWithReflectionFilter:
    """
    this calls LiteLLMCompletion but takes an optional parameter that is a callable
    the callable is called with its only input being the response from the completion
    if the callable returns None, the response is returned as is
    if the callable returns a string, the string is used as the prompt for the next completion
    this continues until the required paramater max_iterations is reached or the callable returns None

    """

    @classmethod
    def INPUT_TYPES(cls):
        # grab the input types from the LiteLLMCompletion node
        input_types = LiteLLMCompletion.INPUT_TYPES()
        # add the reflection_filter input
        input_types["optional"]["reflection_filter"] = ("CALLABLE", {"default": None})
        # add the max_iterations input
        input_types["required"]["max_iterations"] = ("INT", {"default": 10})
        return input_types

    RETURN_TYPES = ("LITELLM_MODEL", "LLLM_MESSAGES", "STRING", "STRING",)
    RETURN_NAMES = ("Model", "Messages", "Completion", "Usage",)

    # Method to provide the default LiteLLM model
    def handler(self, **kwargs):
        # grab the handler from the LiteLLMCompletion node
        completion_handler = LiteLLMCompletion().handler

        # pop the reflection_filter from the kwargs
        reflection_filter = kwargs.pop("reflection_filter", None)
        # pop the max_iterations from the kwargs
        max_iterations = kwargs.pop("max_iterations", 10)

        # call the handler collect the resutls
        model, messages, completion, usage = completion_handler(**kwargs)

        # set the current iteration to 0
        current_iteration = 0
        # while the current iteration is less than the max_iterations
        while current_iteration < max_iterations:
            # if there is no reflection filter, break the loop
            if not reflection_filter:
                break
            # call the reflection filter with the completion
            new_prompt = reflection_filter(completion)[0]
            # if the prompt is None, break the loop
            if new_prompt is None:
                break
            # increment the current iteration
            current_iteration += 1
            # call the completion handler with the new prompt

            kwargs["messages"] = messages
            kwargs["prompt"] = new_prompt
            model, messages, completion, usage = completion_handler(**kwargs)

        return (model, messages, completion, usage,)


@litellm_base
class LiteLLMCompletionProvider:
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        # Inputs can be used to customize the default architecture
        return {
            "required": {
                "model": ("LITELLM_MODEL", {"default": "anthropic/claude-3-haiku-20240307"}),
                "max_tokens": ("INT", {"default": 250}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frequency_penalty": ("FLOAT", {"default": 0}),
                "presence_penalty": ("FLOAT", {"default": 0}),
                "prompt": ("STRING", {"default": "Hello World!", "multiline": True}),
            },
            "optional": {
                "messages": ("LLLM_MESSAGES", {"default": None}),
            }
        }

    RETURN_TYPES = ("CALLABLE",)
    RETURN_NAMES = ("Completion function",)

    # Method to provide the default LiteLLM model
    def handler(self, **kwargs):

        from copy import deepcopy

        litellm.drop_params = True

        model = kwargs.get('model', 'anthropic/claude-3-haiku-20240307')
        messages = kwargs.get('messages', [])
        messages = deepcopy(messages)
        max_tokens = kwargs.get('max_tokens', None)
        temperature = kwargs.get('temperature', None)
        top_p = kwargs.get('top_p', None)
        frequency_penalty = kwargs.get('frequency_penalty', None)
        presence_penalty = kwargs.get('presence_penalty', None)

        base_messages = messages.copy()

        # here we need to compose a callable that we will return
        def completion_function(prompt):
            messages = base_messages.copy()
            # append the prompt to the messages
            messages.append({"content": prompt, "role": "user"})

            use_kwargs = {
                "model": model,
                "messages": messages,
                "stream": False,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            }

            # Handle custom endpoint model configurations
            if isinstance(model, dict) and "kwargs" in model:
                # Extract and update with custom kwargs (api_key, api_base, etc.)
                use_kwargs.update(model["kwargs"])
                # Fix model string when we have a dict with kwargs
                use_kwargs["model"] = model["model"]

            if ("o1" in use_kwargs["model"]) or ("o3" in use_kwargs["model"]):
                use_kwargs["max_completion_tokens"] = use_kwargs.pop("max_tokens")
                use_kwargs.pop("temperature", None)
                use_kwargs.pop("top_p", None)
                use_kwargs.pop("logit_bias", None)


            ret = litellm.completion(**use_kwargs)
            # now extract the actual content
            response_choices = ret.choices
            response_first_choice = response_choices[0]
            response_first_choice_message = response_first_choice.message
            response_first_choice_message_content = response_first_choice_message.content

            response_content = response_first_choice_message_content or ""
            return response_content

        return (completion_function,)


@litellm_base
class LiteLLMImageCaptioningProvider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LITELLM_MODEL", {"default": "anthropic/claude-3.5-sonnet"}),
                "max_tokens": ("INT", {"default": 250, "min": 1, "max": 1e10, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frequency_penalty": ("FLOAT", {"default": 0}),
                "presence_penalty": ("FLOAT", {"default": 0}),
            },
            "optional": {
                "image": ("IMAGE", {"default": None}),  # Optional image input
                "prompt": ("STRING", {"default": "Describe the image:", "multiline": True}),
                "messages": ("LLLM_MESSAGES", {"default": None}),
            }
        }

    RETURN_TYPES = ("CALLABLE",)
    RETURN_NAMES = ("Captioning function",)

    def tensor_image_to_base64(self, tensor):
        """
        Convert a tensor in the format (batch_size, height, width, channels) with float values [0, 1]
        to a base64-encoded JPEG image.
        """
        from PIL import Image
        from io import BytesIO
        import base64

        # Ensure the tensor is in the correct shape and format
        if tensor.dim() == 4:  # Batch of images
            tensor = tensor[0]  # Take the first image in the batch
        elif tensor.dim() != 3:
            raise ValueError(f"Expected tensor with 3 or 4 dimensions, got {tensor.dim()}")

        # Convert from float [0, 1] to uint8 [0, 255]
        tensor = tensor.mul(255).byte()

        # Convert to numpy array and then to PIL image
        image = Image.fromarray(tensor.numpy(), mode="RGB")

        # Encode to base64
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=100)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_image_data(self, base64_image):
        image_data = {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64," + base64_image
            }
        }
        return image_data

    def handler(self, **kwargs):
        from copy import deepcopy

        model = kwargs.get('model', 'anthropic/claude-3.5-sonnet')
        messages = kwargs.get('messages', [])
        messages = deepcopy(messages)
        max_tokens = kwargs.get('max_tokens', None)
        temperature = kwargs.get('temperature', None)
        top_p = kwargs.get('top_p', None)
        frequency_penalty = kwargs.get('frequency_penalty', None)
        presence_penalty = kwargs.get('presence_penalty', None)
        image = kwargs.get('image', None)
        prompt = kwargs.get('prompt', "Describe the image:")

        # Encode image to base64 if provided
        base64_image = None
        if image is not None:
            base64_image = self.tensor_image_to_base64(image)

        # Append the initial prompt and image (if provided) to the messages
        base_messages = messages.copy()
        if base64_image:
            image_data = self.get_image_data(base64_image)
            base_messages.append({"role": "user", "content": [{"type": "text", "text": prompt}, image_data]})
        else:
            base_messages.append({"role": "user", "content": prompt})

        # Define the callable function for captioning
        def captioning_function(new_prompt=None, new_image=None):

            litellm.set_verbose = False
            litellm.suppress_debug_info = True

            messages = base_messages.copy()

            # Update the prompt if a new one is provided
            if new_prompt:
                if isinstance(messages[-1]["content"], list):
                    # If the last message contains an image, update the text part
                    messages[-1]["content"][0]["text"] = new_prompt
                else:
                    # If it's a text-only message, update the content
                    messages[-1]["content"] = new_prompt

            # Update the image if a new one is provided
            if new_image is not None:
                new_base64_image = self.tensor_image_to_base64(new_image)
                new_image_data = self.get_image_data(new_base64_image)
                if isinstance(messages[-1]["content"], list):
                    # If the last message contains an image, replace it
                    messages[-1]["content"][1] = new_image_data
                else:
                    # If it's a text-only message, append the image
                    messages[-1]["content"] = [{"type": "text", "text": new_prompt or prompt}, new_image_data]

            use_kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            }

            if ("o1" in use_kwargs["model"]) or ("o3" in use_kwargs["model"]):
                use_kwargs["max_completion_tokens"] = use_kwargs.pop("max_tokens")
                use_kwargs.pop("temperature", None)
                use_kwargs.pop("top_p", None)
                use_kwargs.pop("logit_bias", None)

            # Call LiteLLM for captioning
            response = litellm.completion(**use_kwargs)

            # Extract the caption from the response
            response_choices = response.choices
            response_first_choice = response_choices[0]
            response_first_choice_message = response_first_choice.message
            caption = response_first_choice_message.content or ""
            return caption

        return (captioning_function,)


@litellm_base
class LiteLLMMessage:
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        # Inputs can be used to customize the default architecture
        return {
            "required": {
                "content": ("STRING", {"default": "Hello World!", "multiline": True}),
                "role": (["user", "assistant", "system"], {"default": "user"}),
            },
            "optional": {
                "messages": ("LLLM_MESSAGES", {"default": None}),
            }
        }

    # Define the return types
    RETURN_TYPES = ("LLLM_MESSAGES",)
    RETURN_NAMES = ("Message(s)",)

    # Method to provide the default LiteLLM model
    def handler(self, **kwargs):
        from copy import deepcopy
        existing_messages = kwargs.get('messages', [])
        existing_messages = deepcopy(existing_messages)
        content = kwargs.get('content', "")
        if content == "":
            return (existing_messages,)

        role = kwargs.get('role', "user")
        new_message = {"content": content, "role": role}
        new_messages = existing_messages + [new_message]
        return (new_messages,)


@litellm_base
class LLLMInterpretUrl:

    @classmethod
    def INPUT_TYPES(cls):
        ret = {"required": {
            "url": ("STRING", {"multiline": False, "default": "http://127.0.0.1:3389/infer"}),
            "get_what": (["text", "links", "urls", "article"],)
        },
            "optional": {
                "headers": ("STRING", {"multiline": True, "default": "Content-Type:application/json"}),
                "url_list": ("LIST", {"default": []})
            }
        }
        return ret

    RETURN_TYPES = ("STRING", "LIST",)
    # FUNCTION = "http_get"

    CATEGORY = "network"

    def handler(self, url, headers=None, get_what="text", url_list=[]):
        import asyncio
        from aiohttp import ClientSession
        from bs4 import BeautifulSoup
        import time

        dailymail = '<div itemprop="articleBody">'

        async def fetch(url, session):
            try:
                async with session.get(url, timeout=10) as response:
                    return await response.text()
            except Exception as e:
                return f"An error occurred: {e}"

        async def get_all_links(url, session):
            html = await fetch(url, session)
            soup = BeautifulSoup(html, 'html.parser')
            links = [f"(\"{a.string}\", \"{a['href']}\")" for a in soup.find_all('a', href=True)]
            return "[\n" + ",\n".join(links) + "\n]"

        async def get_all_visible_text(url, session):
            html = await fetch(url, session)
            soup = BeautifulSoup(html, 'html.parser')
            try:
                text = soup.body.get_text(separator=' ', strip=True)
            except AttributeError:
                text = soup.get_text(separator=' ', strip=True)

            # also try to get the article text from the daily mail
            if "dailymail.co.uk" in url:
                try:
                    text = soup.find("div", {"itemprop": "articleBody"}).get_text(separator=' ', strip=True)
                except AttributeError:
                    pass

            return text

        async def get_article_text(url, session):
            import article_parser
            html = await fetch(url, session)
            title, content = article_parser.parse(html=html, output="markdown")
            return f"{title}\n\n{content}"

        async def bounded_fetch(sem, url, session):
            async with sem:
                if get_what == "text":
                    return await get_all_visible_text(url, session)
                elif get_what == "links":
                    t = await get_all_links(url, session)
                    return t
                elif get_what == "article":
                    return await get_article_text(url, session)
                else:
                    return "Invalid option"

        async def run():
            tasks = []
            sem = asyncio.Semaphore(5)
            async with ClientSession() as session:
                for url in url_list:
                    task = asyncio.ensure_future(bounded_fetch(sem, url, session))
                    tasks.append(task)
                responses = await asyncio.gather(*tasks)
                return responses

        if url_list == []:
            url_list = [url]
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ret = loop.run_until_complete(run())
        loop.close()

        for _ in range(0, len(url_list), 5):
            time.sleep(10)  # sleep for 10 seconds

        if len(ret) == 1:
            if get_what == "text":
                return (ret[0], ret,)
            elif get_what == "links":
                return (ret[0], eval(ret[0]),)
            return (ret[0], ret,)
        else:
            if get_what == "text":
                return ("\n".join(ret), ret,)
            elif get_what == "links":
                raise NotImplementedError
            return ("\n".join(ret), ret,)


@litellm_base
class ListToMessages:
    """simply convert any list of messages check the for the correct keys in each message in the list"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "messages": ("LIST", {"default": []}),
            },
        }

    RETURN_TYPES = ("LLLM_MESSAGES",)
    RETURN_NAMES = ("Messages",)

    def handler(self, messages):
        if not messages:
            return (messages,)
        for message in messages:
            if "content" not in message:
                raise ValueError("Each message in the list must have a 'content' key")
            if "role" not in message:
                raise ValueError("Each message in the list must have a 'role' key")
        return (messages,)


@litellm_base
class MessagesToList:
    """converts LLLM_MESSAGES to a list of messages"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "messages": ("LLLM_MESSAGES", {"default": []}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("Messages",)

    def handler(self, messages):
        if not messages:
            return (messages,)
        for message in messages:
            if "content" not in message:
                raise ValueError("Each message in the list must have a 'content' key")
            if "role" not in message:
                raise ValueError("Each message in the list must have a 'role' key")
        return (messages,)


@litellm_base
class AppendMessages:
    """
    Appends messages to the existing messages
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "existing_messages": ("LLLM_MESSAGES", {"default": []}),
                "new_messages": ("LLLM_MESSAGES", {"default": []}),
            },
        }

    RETURN_TYPES = ("LLLM_MESSAGES",)
    RETURN_NAMES = ("Messages",)

    def handler(self, existing_messages, new_messages):
        return (existing_messages + new_messages,)


@litellm_base
class MessagesToText:
    """
    Converts LLLM_MESSAGES to a single string
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "messages": ("LLLM_MESSAGES", {"default": []}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Text",)

    def handler(self, messages):
        delimiter = "\uFF1A"  # Fullwidth Colon (U+FF1A)
        # Check if any message content contains the unique Unicode delimiter
        for message in messages:
            if delimiter in message['content']:
                raise ValueError(f"Message content contains the delimiter {delimiter}")

        # Convert messages to a single string
        return ("\n".join([f"{m['role'].upper()}{delimiter} {m['content']}" for m in messages]),)

    def handler(self, messages):
        return ("\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]),)


@litellm_base
class TextToMessages:
    """
    Converts a string to a list of messages
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("LLLM_MESSAGES",)
    RETURN_NAMES = ("Messages",)

    def handler(self, text):
        import re
        delimiter = "\uFF1A"  # Fullwidth Colon (U+FF1A)
        pattern = re.compile(rf'^(user|assistant|system){delimiter}\s*(.*)$', re.IGNORECASE | re.MULTILINE)

        ret = []
        for match in pattern.finditer(text):
            role, content = match.groups()
            role = role.strip().lower()
            content = content.strip()
            if not content:
                raise ValueError("Content cannot be empty")
            ret.append({"role": role, "content": content})

        if not ret:
            raise ValueError("Invalid message format")

        return (ret,)
