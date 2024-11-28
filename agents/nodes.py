import importlib
import os
import json
from pathlib import Path
from pydantic import BaseModel, create_model

try:
    from .. import config
    from ..utils.hash_utils import get_input_hash
    from .. import litellmnodes
except ImportError:
    import config
    from utils.hash_utils import get_input_hash
    import litellmnodes

try:
    from .base import AgentBaseNode
except ImportError:
    from base import AgentBaseNode

default_expansion_prompt = """
the date is {date}

given the prompt:
<prompt>
{prompt}
</prompt>

and completion:
<completion>
{completion}
</completion>

Please expand on the completion by.
- considering the prompt more thoroughly
 - why would the user have entered this as their prompt?
 - based on the language the user has used, what can i assume about the user in order to help them?
- considering what the completion might have missed.

use tags:
<consideration></consideration>
<completion></completion>
for your response
please understand that what you tag as completion will go directly to the user.
please completely re-write and re-tag everything, but include everything useful from the given completion and prompt
"""


class AgentNode(AgentBaseNode):
    __package__ = globals().get("__package__")
    __package__ = __package__ or "custom_nodes.ComfyUI_LiteLLM.Agents"

    try:
        litellmnodes = importlib.import_module("..litellmnodes", __package__)
    except ImportError:
        litellmnodes = importlib.import_module("litellmnodes")

    rough_handler = litellmnodes.LitellmCompletionV2().handler

    base_handler = rough_handler
    base_input_types = litellmnodes.LitellmCompletionV2.INPUT_TYPES

    RETURN_TYPES = ("LITELLM_MODEL", "LLLM_MESSAGES", "STRING", "LIST", "LIST", "STRING",)
    RETURN_NAMES = ("Model", "Messages", "Completion", "List_Completions", "List_messages", "Usage",)

    @classmethod
    def get_cache_dir(cls):
        # Use the temporary directory from config
        tmp_dir = Path(config.config_settings['tmp_dir'])
        cache_dir = tmp_dir / "agent_response_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def save_response(self, response_hash, response_data):
        # Create a copy to avoid modifying the original data
        data_to_save = response_data.copy()

        # Handle non-serializable model kwargs
        if "model" in data_to_save and "kwargs" in data_to_save["model"]:
            model_kwargs = data_to_save["model"]["kwargs"].copy()
            # Convert any non-serializable objects to a dict with type info
            for key, value in model_kwargs.items():
                if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    if isinstance(value, type) and hasattr(value, 'model_json_schema'):  # It's a Pydantic model class
                        model_kwargs[key] = {
                            "_type": "pydantic_model_definition",
                            "_schema": value.model_json_schema(),
                            "_model_name": value.__name__
                        }
                    elif hasattr(value, 'model_dump'):  # It's a Pydantic model instance
                        model_kwargs[key] = {
                            "_type": f"{value.__class__.__module__}.{value.__class__.__name__}",
                            "_pydantic": True,
                            "_data": value.model_dump()
                        }
                    else:
                        model_kwargs[key] = {
                            "_type": f"{value.__class__.__module__}.{value.__class__.__name__}",
                            "_repr": str(value)
                        }
            data_to_save["model"]["kwargs"] = model_kwargs

        cache_file = self.get_cache_dir() / f"{response_hash}.json"
        with open(cache_file, 'w') as f:
            json.dump(data_to_save, f)

    def load_response(self, response_hash):
        cache_file = self.get_cache_dir() / f"{response_hash}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)

            # Try to reconstruct any saved class objects
            if "model" in data and "kwargs" in data["model"]:
                model_kwargs = data["model"]["kwargs"]
                for key, value in model_kwargs.items():
                    if isinstance(value, dict) and "_type" in value:
                        try:
                            if value["_type"] == "pydantic_model_definition":
                                # Create a new Pydantic model class from the JSON schema
                                schema = value["_schema"]
                                field_definitions = {}
                                
                                # Map JSON schema types to Python types
                                type_map = {
                                    "integer": int,
                                    "number": float,
                                    "string": str,
                                    "boolean": bool,
                                    "array": list
                                }
                                
                                for field_name, field_info in schema["properties"].items():
                                    field_type = field_info["type"]
                                    if field_type == "array" and "items" in field_info:
                                        # Handle array types (e.g., list[str])
                                        item_type = field_info["items"]["type"]
                                        python_type = list[type_map.get(item_type, str)]
                                    else:
                                        python_type = type_map.get(field_type, str)
                                    
                                    field_definitions[field_name] = (
                                        python_type,
                                        ... if field_name in schema.get("required", []) else None
                                    )
                                
                                model_class = create_model(
                                    value["_model_name"],
                                    __base__=BaseModel,
                                    **field_definitions
                                )
                                model_kwargs[key] = model_class
                            elif value.get("_pydantic"):
                                # Handle Pydantic model instance
                                module_name, class_name = value["_type"].rsplit(".", 1)
                                if module_name == "__main__":
                                    if class_name in globals():
                                        cls = globals()[class_name]
                                    else:
                                        raise ImportError(f"Class {class_name} not found in globals")
                                else:
                                    module = importlib.import_module(module_name)
                                    cls = getattr(module, class_name)
                                model_kwargs[key] = cls(**value["_data"])
                            else:
                                model_kwargs[key] = str(value["_repr"])
                        except Exception as e:
                            print(f"Warning: Could not reconstruct class {value['_type']}: {e}")
                            if "_repr" in value:
                                model_kwargs[key] = str(value["_repr"])
                data["model"]["kwargs"] = model_kwargs

            return data
        return None

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = cls.base_input_types()
        # Remove use_cached_response from optional inputs if it exists
        base_inputs["optional"].pop("use_cached_response", None)
        base_inputs["required"].update({
            "max_iterations": ("INT", {"default": 2, "min": 1}),
            "List_prompts": ("LIST", {"default": None}),
        })
        base_inputs["optional"].update({
            "memory_provider": ("LLLM_AGENT_MEMORY_PROVIDER", {"default": None}),
            "recursion_filter": ("LLLM_AGENT_RECURSION_FILTER", {"default": None}),
            "use_last_response": ("BOOLEAN", {"default": False})
        })
        return base_inputs

    def handler(self, **kwargs):
        from copy import deepcopy
        from typing import Callable

        # Generate hash for the complete input state before any modifications
        input_hash = get_input_hash(**kwargs)
        use_last_response = kwargs.get("use_last_response", False)

        # Check for cached response at the node level
        if use_last_response:
            cached_response = self.load_response(input_hash)
            if cached_response:
                return (cached_response["model"],
                        cached_response["messages"],
                        cached_response["completion"],
                        cached_response["completion_list"],
                        cached_response["messages_results"],
                        cached_response.get("usage", "Usage"))

        # If no cache hit, proceed with normal processing
        if kwargs.get("List_prompts", None):
            prompts = kwargs.pop("List_prompts", None)
        else:
            prompts = [kwargs.pop("prompt", None)]

        recursion_filter: Callable | None = kwargs.pop("recursion_filter", None)
        memory_provider: Callable | None = kwargs.pop("memory_provider", None)
        kwargs.pop("use_last_response", False)  # Remove from kwargs after checking

        max_iterations = kwargs.pop("max_iterations", 1)
        if "messages" not in kwargs:
            kwargs["messages"] = []
        ret = (None, None, None, None)

        # Ensure use_cached_response is set to False in kwargs
        kwargs["use_cached_response"] = False
        frozen_kwargs = deepcopy(kwargs)

        all_results = []

        for prompt in prompts:
            kwargs = deepcopy(frozen_kwargs)
            kwargs["prompt"] = prompt

            for _ in range(max_iterations):
                kwargs, memories = self.memory_step(kwargs, memory_provider)
                recursive_completion = self.recursion_step(kwargs, ret, recursion_filter)
                # insert the recursive completion into the messages
                kwargs["messages"].append({"role": "assistant",
                                           "content": f"<ASSISTANT_THOUGHTS>{recursive_completion}</ASSISTANT_THOUGHTS>"})
                ret, ret_completion, ret_messages = self.normal_step(kwargs)

            all_results.append(ret)

        messages_results = []
        completion_list = []
        for res in all_results:
            completion = res[2]
            completion_list.append(completion)
            messages_results.extend(res[1])

        final_result = (res[0], res[1], completion, completion_list, messages_results, "Usage")

        # Save the complete node result if we're using last_response
        response_data = {
            "model": res[0],
            "messages": res[1],
            "completion": completion,
            "completion_list": completion_list,
            "messages_results": messages_results,
            "usage": "Usage"
        }
        self.save_response(input_hash, response_data)

        return final_result

    def normal_step(self, kwargs):
        ret = self.base_handler(**kwargs)
        ret_completion = ret[2]
        ret_messages = ret[1]
        return ret, ret_completion, ret_messages

    def memory_step(self, kwargs, memory_provider):
        memory_prompt = kwargs["prompt"]
        memories = memory_provider(memory_prompt) if memory_provider else None
        if memories:
            new_prompt = "<SYSTEM_RAG> \n" + "\n".join(memories) + "\n</SYSTEM_RAG>\n" + kwargs["prompt"]

        kwargs["prompt"] = new_prompt if memories else kwargs["prompt"]

        return kwargs, memories

    def recursion_step(self, kwargs, last_results, recursion_filter):
        last_completion = last_results[2]
        recursive_completion = recursion_filter(kwargs["prompt"],
                                                last_completion) if recursion_filter else last_completion
        return recursive_completion


class BasicRecursionFilterNode(AgentBaseNode):
    import importlib
    basic_kwargs = {
        "model": "openai/gpt-4o-mini",
        "task": "completion"
    }
    __package__ = globals().get("__package__")
    __package__ = __package__ or "custom_nodes.ComfyUI_LiteLLM.Agents"

    try:
        litellmnodes = importlib.import_module("..litellmnodes", __package__)
    except ImportError:
        litellmnodes = importlib.import_module("litellmnodes")

    rough_handler = litellmnodes.LitellmCompletionV2().handler

    @classmethod
    def base_handler(cls, prompt):
        new_kwargs = cls.basic_kwargs.copy()
        new_kwargs.update({"prompt": prompt})
        ret = cls.rough_handler(**new_kwargs)
        return ret

    # base_handler = base_handler

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_depth": ("INT", {"default": 2, "min": 1}),
            },
            "optional": {
                "recursion_prompt": ("STRING", {
                    "default": default_expansion_prompt, "multiline": True}),
            }
        }

    RETURN_TYPES = ("LLLM_AGENT_RECURSION_FILTER",)
    RETURN_NAMES = ("Recursion Filter",)

    @classmethod
    def handler(cls, max_depth=1, recursion_prompt=default_expansion_prompt):
        def recursion_filter(prompt, completion):
            from datetime import datetime
            for _ in range(max_depth):
                prompt = (
                    recursion_prompt.replace("{prompt}", prompt).
                    replace("{completion}", completion or "").
                    replace("{date}", datetime.now().strftime("%Y-%m-%d"))
                )
                ret = cls.base_handler(prompt=prompt)
                completion = ret[2]
            return completion

        return (recursion_filter,)


if __name__ == "__main__":
    default_kwargs = {"model": "openai/gpt-4o-mini",
                      "task": "completion"}

    test_kwargs = {"prompt": "This is a test prompt",
                   "recursion_filter": BasicRecursionFilterNode().handler()[0],
                   }
    test_kwargs.update(default_kwargs)
    AgentNode().handler(**test_kwargs)
    BasicRecursionFilterNode().handler(max_depth=5, recursion_prompt=default_expansion_prompt)
