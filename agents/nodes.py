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

inside the consideration tag it is important that you think step by step in at least 5 steps about how to adhere to the instructions.

please understand that what you tag as completion will go directly to the user.
please completely re-write and re-tag everything, but include everything useful from the given completion and prompt
"""
default_chunking_prompt = \
"""
given
prompt:
{prompt}

chunk:
{chunk}

your previous completion:
{completion}

just repeat the chunk and say that it needs to be processed.
"""

class AgentNode(AgentBaseNode):
    """
    Iterative Completion Node with Memory and Recursion Support

    This node performs iterative LLM completions with optional memory integration
    and recursion filtering. It processes multiple prompts across iterations,
    maintaining conversation state and applying custom processing filters.

    Features:
    - Multi-iteration processing with configurable max iterations
    - Memory provider integration for RAG-like functionality
    - Recursion filters for post-processing completions
    - Caching system for response persistence
    - Support for multiple prompts in a single run

    Note: Despite the name "AgentNode", this is primarily an iterative completion
    processor. Consider renaming to "IterativeCompletionNode" for clarity.
    """
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

    async def save_response(self, response_hash, response_data):
        from asyncio import to_thread

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

        async def write_json():
            with cache_file.open('w') as f:
                json.dump(data_to_save, f)

        await to_thread(write_json)

    async def load_response(self, response_hash):
        from asyncio import to_thread

        cache_file = self.get_cache_dir() / f"{response_hash}.json"
        if cache_file.exists():
            async def read_json():
                with cache_file.open('r') as f:
                    return json.load(f)

            data = await to_thread(read_json)

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

    async def process_iteration(self, kwargs, memory_provider, recursion_filter, ret):
        """
        Process a single iteration of the agent workflow.

        Args:
            kwargs: Arguments for the base handler including prompt and messages
            memory_provider: Optional function to provide memory/context
            recursion_filter: Optional function to post-process completions
            ret: Previous iteration result tuple (model, messages, completion, ...)

        Returns:
            Tuple of (updated_result, updated_kwargs)
        """

        import asyncio

        # Memory step
        if memory_provider:
            memories = await asyncio.to_thread(memory_provider, kwargs["prompt"])
            if memories:
                new_prompt = (
                        "<SYSTEM_RAG>\n" +
                        "\n".join(memories) +
                        "\n</SYSTEM_RAG>\n" +
                        kwargs["prompt"]
                )
                kwargs["prompt"] = new_prompt

        # Recursion step
        if recursion_filter:
            recursive_completion = await asyncio.to_thread(
                recursion_filter,
                kwargs["prompt"],
                ret[2] if len(ret) > 2 else None
            )
            kwargs["messages"].append({
                "role": "assistant",
                "content": f"{recursive_completion}"
            })

        # Base handling step (assuming base_handler is sync)
        ret = await asyncio.to_thread(self.base_handler, **kwargs)
        return ret, kwargs

    def handler(self, **kwargs):
        """
        Main handler for iterative completion processing.

        This method orchestrates the entire agent workflow:
        1. Checks for cached responses if use_last_response is True
        2. Processes multiple prompts across multiple iterations
        3. Applies memory and recursion filters at each step
        4. Caches the final results

        Args:
            **kwargs: All input parameters including prompts, model config, etc.

        Returns:
            Tuple of (model, messages, completion, completion_list, messages_results, usage)
        """
        import asyncio
        # from ..utils import get_input_hash
        from copy import deepcopy

        # Input validation
        if not kwargs.get("model"):
            raise ValueError("Model is required. Please connect a LiteLLMModelProvider node.")

        max_iterations = kwargs.get("max_iterations", 1)
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

        if max_iterations > 10:
            print(f"Warning: max_iterations is set to {max_iterations}. This may result in high API costs.")

        # Initial setup and cache check
        input_hash = get_input_hash(**kwargs)
        use_last_response = kwargs.get("use_last_response", False)

        if use_last_response:
            # cached_response = await self.load_response(input_hash)
            cached_response = asyncio.run(self.load_response(input_hash))
            if cached_response:
                return (cached_response["model"],
                        cached_response["messages"],
                        cached_response["completion"],
                        cached_response["completion_list"],
                        cached_response["messages_results"],
                        cached_response.get("usage", "Usage"))

        # Prompt preparation and validation
        if kwargs.get("List_prompts", None):
            prompts = kwargs.pop("List_prompts", None)
            if not prompts:
                raise ValueError("List_prompts cannot be empty")
            if not isinstance(prompts, list):
                raise ValueError("List_prompts must be a list")
            if not all(isinstance(p, str) for p in prompts):
                raise ValueError("All items in List_prompts must be strings")
        else:
            prompt = kwargs.get("prompt")
            if not prompt:
                raise ValueError("Either 'prompt' or 'List_prompts' must be provided")
            prompts = [kwargs.pop("prompt", None)]

        recursion_filter = kwargs.pop("recursion_filter", None)
        memory_provider = kwargs.pop("memory_provider", None)
        kwargs.pop("use_last_response", False)

        max_iterations = kwargs.pop("max_iterations", 1)
        if "messages" not in kwargs:
            kwargs["messages"] = []

        # Initialize ret with last assistant message if available
        initial_completion = None
        if kwargs["messages"]:
            assistant_messages = [

                msg["content"] for msg in kwargs["messages"]
                if msg["role"] == "assistant"
            ]
            if assistant_messages:
                initial_completion = assistant_messages[-1]


        ret = (None, kwargs["messages"], initial_completion, None) # initial completion is the last assistant message
        kwargs["use_cached_response"] = False
        frozen_kwargs = deepcopy(kwargs)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        all_results = []
        for iter in range(max_iterations):
            iter_all_calls = []
            for pi, prompt in enumerate(prompts):
                kwargs = deepcopy(frozen_kwargs)
                if iter > 0:
                    kwargs["messages"] = iter_all_done_messages[pi]
                kwargs["prompt"] = prompt
                iter_all_calls.append({"kwargs": kwargs,
                                       "memory_provider": memory_provider,
                                       "recursion_filter": recursion_filter,
                                       "ret": ret})
            iter_all_results = loop.run_until_complete(
                asyncio.gather(*[self.process_iteration(**call) for call in iter_all_calls]))
            iter_all_done_messages = [res[0][1] for res in iter_all_results]
            iter_all_done_responses = [res[0][2] for res in iter_all_results]
            # we may need to unpack the results and populate the kwargs["messages"] with the new messages

        # if there where multiple prompts then we dont want to worry about the singular returns
        # we will just return the last one
        ret_model = iter_all_results[-1][0][0]
        ret_messages = iter_all_done_messages[-1]
        ret_completion = iter_all_done_responses[-1]
        ret_completion_list = iter_all_done_responses
        ret_messages_results = iter_all_done_messages

        response_data = {
            "model": ret_model,
            "messages": ret_messages,
            "completion": ret_completion,
            "completion_list": ret_completion_list,
            "messages_results": ret_messages_results,
            "usage": "Usage"
        }

        # self.save_response(input_hash, response_data)
        asyncio.run(self.save_response(input_hash, response_data))
        return [response_data["model"],
                response_data["messages"],
                response_data["completion"],
                response_data["completion_list"],
                response_data["messages_results"],
                response_data.get("usage", "Usage")
                ]


class BasicRecursionFilterNode(AgentBaseNode):
    """
    Iterative Completion Enhancement Filter

    This node creates a recursion filter that enhances completions through
    multiple rounds of processing. It takes an initial completion and iteratively
    improves it using a customizable prompt template.

    The filter is designed to:
    - Expand on initial completions with deeper analysis
    - Consider user intent and context more thoroughly
    - Apply multiple rounds of enhancement up to max_depth

    Usage: Connect the output to AgentNode's recursion_filter input to enable
    iterative completion enhancement during agent processing.

    Note: Consider renaming to "CompletionEnhancementFilterNode" for clarity.
    """
    __package__ = globals().get("__package__")
    __package__ = __package__ or "custom_nodes.ComfyUI_LiteLLM.Agents"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_depth": ("INT", {"default": 2, "min": 1}),
                "LLLM_provider": ("CALLABLE",),  # Now required
            },
            "optional": {
                "recursion_prompt": ("STRING", {
                    "default": default_expansion_prompt,
                    "multiline": True
                }),
                "inner_recursion_filter": ("LLLM_AGENT_RECURSION_FILTER", {"default": None}),
            }
        }

    RETURN_TYPES = ("LLLM_AGENT_RECURSION_FILTER",)
    RETURN_NAMES = ("Recursion Filter",)

    @classmethod
    def handler(cls, max_depth: int,
                LLLM_provider: callable,
                recursion_prompt: str = default_expansion_prompt,
                inner_recursion_filter: callable = None
                ):
        """
        Create a recursion filter for iterative completion enhancement.

        Args:
            max_depth: Number of enhancement iterations to perform
            LLLM_provider: Function that takes a prompt and returns a completion
            recursion_prompt: Template for enhancement prompts (uses {prompt}, {completion}, {date})
            inner_recursion_filter: Optional nested filter to apply before each iteration

        Returns:
            Tuple containing the recursion filter function
        """
        def recursion_filter(prompt: str, completion: str) -> str:
            from datetime import datetime

            # Input validation
            if not prompt:
                raise ValueError("Prompt cannot be empty for recursion filter")

            if not LLLM_provider:
                raise ValueError("LLLM_provider is required but not provided")

            current_completion = completion or ""

            try:
                for depth in range(max_depth):
                    if inner_recursion_filter:
                        try:
                            current_completion = inner_recursion_filter(
                                prompt,
                                current_completion
                            )
                        except Exception as e:
                            print(f"Warning: Inner recursion filter failed at depth {depth}: {e}")
                            # Continue with current completion

                    formatted_prompt = (
                        recursion_prompt
                        .replace("{prompt}", prompt)
                        .replace("{completion}", current_completion)
                        .replace("{date}", datetime.now().strftime("%Y-%m-%d"))
                    )

                    try:
                        current_completion = LLLM_provider(formatted_prompt)
                    except Exception as e:
                        print(f"Error in LLLM_provider at recursion depth {depth}: {e}")
                        raise

                return current_completion

            except Exception as e:
                # in comfyui we raise errors so the user knows whats going on
                print(f"Error in recursion filter: {e}")
                raise

        return (recursion_filter,)


class DocumentChunkRecursionFilterNode(AgentBaseNode):
    """
    Document Chunk Processing Filter

    This node creates a recursion filter that processes documents in chunks.
    It's designed for handling large documents by breaking them into manageable
    pieces and processing each chunk sequentially.

    The filter:
    - Chunks documents into specified sizes
    - Processes one chunk per agent iteration
    - Maintains state across chunk processing
    - Can be combined with other recursion filters

    Usage: Connect to AgentNode's recursion_filter input for document processing.
    Set max_iterations on AgentNode to match expected number of chunks.

    Note: Consider renaming to "DocumentChunkProcessorNode" for clarity.
    """
    __package__ = globals().get("__package__")
    __package__ = __package__ or "custom_nodes.ComfyUI_LiteLLM.Agents"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LLLM_provider": ("CALLABLE",),  # Now required
                "document": ("STRING",),
                "chunk_size": ("INT", {"default": 512, "min": 1}),
            },
            "optional": {
                "recursion_prompt": ("STRING", {
                    "default": default_chunking_prompt,
                    "multiline": True
                }),
                "inner_recursion_filter": ("LLLM_AGENT_RECURSION_FILTER", {"default": None}),
            }
        }

    RETURN_TYPES = ("LLLM_AGENT_RECURSION_FILTER",)
    RETURN_NAMES = ("Recursion Filter",)

    @classmethod
    def handler(cls, LLLM_provider: callable, document:str, chunk_size: int, recursion_prompt: str = default_expansion_prompt,
                inner_recursion_filter: callable = None):
        """
        Create a document chunk processing filter.

        Args:
            LLLM_provider: Function that takes a prompt and returns a completion
            document: Full document text to be processed in chunks
            chunk_size: Size of each chunk in characters
            recursion_prompt: Template for chunk processing prompts (uses {chunk}, {prompt}, {completion}, {date})
            inner_recursion_filter: Optional nested filter to apply before chunk processing

        Returns:
            Tuple containing the document chunk recursion filter
        """

        # Input validation
        if not LLLM_provider:
            raise ValueError("LLLM_provider is required. Please connect a LiteLLMCompletionProvider node.")

        if not document:
            raise ValueError("Document text cannot be empty")

        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")

        if chunk_size > 10000:
            print(f"Warning: Large chunk_size ({chunk_size}) may exceed model token limits")

        # Calculate expected number of chunks for user information
        expected_chunks = (len(document) + chunk_size - 1) // chunk_size
        print(f"Document will be processed in {expected_chunks} chunks. Set max_iterations to {expected_chunks} in AgentNode.")

        class document_chunk_recursion_filter:
            """
            Internal class that handles document chunk processing.

            This class maintains state for chunk processing and provides
            the actual filter function that gets called during agent iterations.
            """
            def __init__(self):
                self.document = document
                self.chunked_document = self.chunk_document(self.document, chunk_size)
                self.recursion_prompt = recursion_prompt
                self.chunker = self.yield_chunk()
                self.current_chunk_index = 0
                self.total_chunks = len(self.chunked_document)

            def yield_chunk(self):
                for chunk in self.chunked_document:
                    yield chunk

            def chunk_document(self, doc, size):
                return [doc[i:i + size] for i in range(0, len(doc), size)]

            def format_prompt(self, chunk, completion, prompt):
                from datetime import datetime
                formatted_prompt = (
                    self.recursion_prompt
                    .replace("{chunk}", chunk)
                    .replace("{prompt}", prompt)
                    .replace("{completion}", completion or "")
                    .replace("{date}", datetime.now().strftime("%Y-%m-%d"))
                )
                return formatted_prompt

            def __call__(self, prompt: str, messages: list) -> str:
                try:
                    if self.current_chunk_index >= self.total_chunks:
                        print("Warning: No more chunks to process. Consider reducing max_iterations.")
                        return "No more document chunks to process."

                    chunk = next(self.chunker)
                    self.current_chunk_index += 1

                    print(f"Processing chunk {self.current_chunk_index}/{self.total_chunks}")

                    current_completion = ""

                    if inner_recursion_filter:
                        try:
                            current_completion = inner_recursion_filter(prompt, messages)
                        except Exception as e:
                            print(f"Warning: Inner recursion filter failed: {e}")

                    formatted_prompt = self.format_prompt(
                        chunk=chunk,
                        completion=current_completion,
                        prompt=prompt
                    )

                    try:
                        current_completion = LLLM_provider(formatted_prompt)
                    except Exception as e:
                        print(f"Error in LLLM_provider for chunk {self.current_chunk_index}: {e}")
                        raise

                    return current_completion

                except StopIteration:
                    print("All document chunks have been processed.")
                    return "Document processing complete."
                except Exception as e:
                    print(f"Error in document chunk processing: {e}")
                    return f"Error processing chunk: {str(e)}"

        recursion_filter = document_chunk_recursion_filter()
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
