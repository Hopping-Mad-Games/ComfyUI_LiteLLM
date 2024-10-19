import base64
import hashlib
import os
import json
import time
import litellm
from copy import deepcopy
from . import config
from .utils import safe_print, safe_error

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
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "handler"
    OUTPUT_NODE = True

    def handler(self, html_content):
        tmplt = """<iframe srcdoc="{html_content}" style="width:100%; height:800px; border:none;"></iframe>"""
        # Replace newlines and double quotes in the raw HTML
        y = tmplt.format(html_content=html_content.replace("\n", "").replace('"', '&quot;'))
        html_content = y

        ret = {"ui": {"string": [html_content]}, "result": (html_content,)}
        return ret


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
                "data_model": ("data_model", {"default": None}),
            }
        }

    RETURN_TYPES = ("LITELLM_MODEL",)
    RETURN_NAMES = ("Litellm model",)

    def restricted_exec(self, code: str):
        from pydantic import BaseModel, Field, conlist
        from typing import Tuple

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
        uid_kwargs = kwargs.copy()
        uid_kwargs["prompt"] = prompt

        unique_id = str(hashlib.sha256(json.dumps(uid_kwargs).encode()).hexdigest())
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
                        "stream": False,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "frequency_penalty": frequency_penalty,
                        "presence_penalty": presence_penalty
                    }
                    n_kwargs.update(kwargs)

                    response = litellm.completion(
                        **n_kwargs
                    )

                except litellm.exceptions.RateLimitError as e:
                    safe_print(f"Rate limit error: {safe_error(e)}")
                    time.sleep(5)
                    continue
                except Exception as e:
                    safe_print(f"An error occurred: {safe_error(e)}")
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
        return {
            "required": {
                "model": ("LITELLM_MODEL", {"default": "anthropic/claude-3.5-sonnet"}),
                "max_tokens": ("INT", {"default": 250, "min": 1, "max": 1e10, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frequency_penalty": ("FLOAT", {"default": 0}),
                "presence_penalty": ("FLOAT", {"default": 0}),
                "prompt": ("STRING", {"default": "Hello World!", "multiline": True}),
                "task": (valid_tasks, {"default": "completion"})  # Use a dropdown list for valid tasks
            },
            "optional": {
                "image": ("IMAGE", {"default": None}),  # Assuming image_tensor is the PyTorch tensor
                "messages": ("LLLM_MESSAGES", {"default": None}),
                "use_cached_response": ("BOOLEAN", {"default": False}),
            }
        }

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
        if kwargs["top_p"] == 0:
            kwargs["top_p"] = None
        if kwargs["temperature"] == 0:
            kwargs["temperature"] = None
        if kwargs["frequency_penalty"] == 0:
            kwargs["frequency_penalty"] = None
        if kwargs["presence_penalty"] == 0:
            kwargs["presence_penalty"] = None

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
                if kwargs["model"]["kwargs"]["response_format"].__name__ == "UserModel":
                    mdl_cls = kwargs["model"]["kwargs"]["response_format"]
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
                while not response:
                    try:
                        if id:
                            response = self.litellm_completion_v2_inner(frequency_penalty, max_tokens, messages, model,
                                                                        presence_penalty,
                                                                        prompt, task, temperature, top_p, id)
                        else:
                            response = self.litellm_completion_v2_inner(frequency_penalty, max_tokens, messages, model,
                                                                        presence_penalty,
                                                                        prompt, task, temperature, top_p)

                        response_choices = response.choices
                        response_first_choice = response_choices[0]
                        response_first_choice_message = response_first_choice.message
                        response_content = response_first_choice_message.content or ""
                        responses.append(response_content)
                    except litellm.BadRequestError as e:
                        print(f"Bad request error: {e}")
                        break
                    except Exception as e:
                        print(f"An error occurred: {e}")

        if not os.path.exists(config.config_settings['tmp_dir']):
            os.makedirs(config.config_settings['tmp_dir'])

        if os.path.exists(cache_file_path):
            os.remove(cache_file_path)
        with open(cache_file_path, 'w') as file:
            json.dump(responses, file, indent=4)

        usage = ""
        if single_response:
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

    def litellm_completion_v2_inner(self, frequency_penalty, max_tokens, messages, model, presence_penalty, prompt,
                                    task, temperature, top_p, image_data=None):
        import json
        # content = [{"type": "text", "text": prompt},image_data]
        # content = [{"type": "text", "text": prompt}, image_data]
        try:
            litellm.set_verbose = True
            if task in ["transcription", "classification", "image_captioning",
                        "object_detection"]:  # Tasks that require vision API
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )
            else:  # Other tasks
                base_schema = {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "video_script_structure",
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
                        use_kwargs.update(model["kwargs"])
                        if "response_format" in use_kwargs:
                            base_scm_use = base_schema.copy()
                            scm = use_kwargs["response_format"].schema()
                            base_scm_use["response_format"]["json_schema"]["schema"].update(scm)
                            use_kwargs["response_format"] = json.dumps(base_scm_use["response_format"])

                            # fix model string
                            use_kwargs["model"] = use_kwargs["model"]["model"]
                            use_kwargs.pop("prompt")
                            ob = json.loads(use_kwargs["response_format"])

                            if "$defs" in ob["json_schema"]["schema"]:
                                defs = ob["json_schema"]["schema"]["$defs"]
                                for k, v in defs.items():
                                    v["additionalProperties"] = False

                            use_kwargs["response_format"] = ob

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
                "first_page": ("INT", {"default": 1, "min": 1}),
                "last_page": ("INT", {"default": -1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    def convert_from_path(self, pdf_path, dpi=200, first_page=1, last_page=-1, output_folder=None):
        import pdfplumber

        if last_page == 0:
            last_page = -1
        out = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                if i < first_page - 1:
                    continue
                if last_page > 0 and i >= last_page:
                    break

                pil_image = page.to_image(resolution=dpi)
                out.append(pil_image)
        return out

    def handler(self, pdf_path, dpi, first_page, last_page):
        import os
        import tempfile
        import numpy as np

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        with tempfile.TemporaryDirectory() as temp_dir:
            images = self.convert_from_path(pdf_path, dpi=dpi, first_page=first_page,
                                            last_page=last_page, output_folder=temp_dir)
            images = [i.original for i in images]
            # Convert PIL Images to torch tensors
            tensor_images = []
            for img in images:
                # Convert PIL Image to RGB mode
                img_rgb = img.convert('RGB')
                # Convert to numpy array and normalize
                np_image = np.array(img_rgb).astype(np.float32) / 255.0
                # Convert to torch tensor and add batch dimension
                tensor_image = torch.from_numpy(np_image).unsqueeze(0)
                tensor_images.append(tensor_image)

            # Stack all images into a single tensor
            batch_images = torch.cat(tensor_images, dim=0)

        # The output tensor is in the format [B, H, W, C], where:
        # B: Batch size (number of pages)
        # H: Height of each image
        # W: Width of each image
        # C: Number of channels (3 for RGB)
        # Values are normalized to the range 0.0 to 1.0
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
                "list_display": ("STRING", {"multi_line": True, "default": ""}),
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
class LiteLLMCompletionListOfPrompts:
    """Calls LiteLLMCompletion for each prompt in the list asynchronously"""

    @classmethod
    def INPUT_TYPES(cls):
        base = LiteLLMCompletion.INPUT_TYPES()
        base["required"]["pre_prompt"] = base["required"]["prompt"]
        base["required"].pop("prompt")
        base["required"]["prompts"] = ("LIST", {"default": None})
        base["required"]["async"] = ("BOOLEAN", {"default": True})
        return base

    RETURN_TYPES = ("LITELLM_MODEL", "LIST", "STRING",)
    RETURN_NAMES = ("Model", "Completions", "Usage",)

    def wrapped_llm_call(self, index, **kwargs):
        return (index, LiteLLMCompletion().handler(**kwargs))

    async def async_process_prompt(self, index, prompt, pre_prompt, **kwargs):
        import asyncio
        import functools

        kwargs["prompt"] = f"{pre_prompt}\n{prompt}"
        loop = asyncio.get_running_loop()
        partial_func = functools.partial(self.wrapped_llm_call, index, **kwargs)
        idx, the_rest = await loop.run_in_executor(None, partial_func)
        model, messages, completion, usage = the_rest
        return idx, completion

    def process_prompt(self, prompt, pre_prompt, **kwargs):
        kwargs["prompt"] = f"{pre_prompt}\n{prompt}"
        model, messages, completion, usage = LiteLLMCompletion().handler(**kwargs)
        return completion

    async def process_prompts(self, prompts, pre_prompt, **kwargs):
        import asyncio

        if "pre_prompt" in kwargs:
            kwargs.pop("pre_prompt")
        if "prompt" in kwargs:
            kwargs.pop("prompt")

        tasks = [self.async_process_prompt(i, prompt, pre_prompt, **kwargs) for i, prompt in enumerate(prompts)]
        completions = await asyncio.gather(*tasks)
        return [completion for _, completion in sorted(completions, key=lambda x: x[0])]

    def handler(self, **kwargs):
        import asyncio
        prompts = kwargs.pop("prompts", ["Hello World!"])
        pre_prompt = kwargs.pop("pre_prompt", "Hello World!")
        completions = []

        if kwargs.pop("async", None):
            # Create a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Run the async code in the loop
                completions = loop.run_until_complete(self.process_prompts(prompts, pre_prompt, **kwargs))
            finally:
                # Ensure the loop is closed
                loop.close()
        else:
            for prompt in prompts:
                completion = self.process_prompt(prompt, pre_prompt, **kwargs)
                completions.append(completion)

        return (kwargs.get("model"), completions, "",)


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
        import litellm
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

            ret = litellm.completion(model=model,
                                     messages=messages,
                                     stream=False,
                                     max_tokens=max_tokens,
                                     temperature=temperature,
                                     top_p=top_p,
                                     frequency_penalty=frequency_penalty,
                                     presence_penalty=presence_penalty,
                                     )
            # now extract the actual content
            response_choices = ret.choices
            response_first_choice = response_choices[0]
            response_first_choice_message = response_first_choice.message
            response_first_choice_message_content = response_first_choice_message.content

            response_content = response_first_choice_message_content or ""
            return response_content

        return (completion_function,)


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
