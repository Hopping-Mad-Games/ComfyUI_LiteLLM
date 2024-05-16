import os
import json
from . import config

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
                             "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
                             "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                             "bedrock/anthropic.claude-3-opus-20240229-v1:0"
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
class LiteLLMCompletion:
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        # Inputs can be used to customize the default architecture
        return {
            "required": {
                "model": ("LITELLM_MODEL", {"default": "anthropic/claude-3-haiku-20240307"}),
                "max_tokens": ("INT", {"default": 250,"min": 1, "max": 1e10, "step": 1}),
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

    # Method to provide the default LiteLLM model
    def handler(self, **kwargs):
        import litellm
        from copy import deepcopy
        import time
        litellm.drop_params = True

        # Update kwargs
        if kwargs["top_p"] == 0:
            kwargs["top_p"] = None
        if kwargs["temperature"] == 0:
            kwargs["temperature"] = None
        if kwargs["frequency_penalty"] == 0:
            kwargs["frequency_penalty"] = None
        if kwargs["presence_penalty"] == 0:
            kwargs["presence_penalty"] = None

        # Extract all the necessary variables from the input
        model = kwargs.get('model', 'anthropic/claude-3-haiku-20240307')
        messages = kwargs.get('messages', [])
        messages = deepcopy(messages)
        max_tokens = kwargs.get('max_tokens', None)
        temperature = kwargs.get('temperature', None)
        top_p = kwargs.get('top_p', None)
        frequency_penalty = kwargs.get('frequency_penalty', None)
        presence_penalty = kwargs.get('presence_penalty', None)
        prompt = kwargs.get('prompt', "Hello World!")
        use_cached_response = kwargs.get('use_cached_response', False)

        # append the prompt to the messages
        messages.append({"content": prompt, "role": "user"})

        import hashlib
        # remove untracked params
        kwargs.pop('use_cached_response', None)
        # create a unique id for the cache file
        unique_id = str(hashlib.sha256(json.dumps(kwargs).encode()).hexdigest())
        # create a unique filename for the cache file
        cache_file_name = f'cached_response_{unique_id}.json'
        # Define the path for the cached response
        cache_file_path = os.path.join(config.config_settings['tmp_dir'], cache_file_name)

        response = None
        cached_completion = False

        if use_cached_response:
            if os.path.exists(cache_file_path):
                # Load the cached response
                with open(cache_file_path, 'r') as file:
                    response = json.load(file)
                    response = litellm.ModelResponse(**response)
                    cached_completion = True

        if not response:
            # Call the completion function with the extracted variables
            while not response:
                try:
                    litellm.drop_params = True
                    litellm.set_verbose = True
                    response = litellm.completion(model=model,
                                                  messages=messages,
                                                  stream=False,
                                                  max_tokens=max_tokens,
                                                  temperature=temperature,
                                                  top_p=top_p,
                                                  frequency_penalty=frequency_penalty,
                                                  presence_penalty=presence_penalty,
                                                  )
                except litellm.exceptions.RateLimitError as e:
                    print(f"Rate limit error: {e}")
                    time.sleep(5)
                    continue

        response_choices = response.choices
        response_first_choice = response_choices[0]
        response_first_choice_message = response_first_choice.message
        response_first_choice_message_content = response_first_choice_message.content

        response_content = response_first_choice_message_content or ""

        # first check if the tmp directory exists
        if not os.path.exists(config.config_settings['tmp_dir']):
            os.makedirs(config.config_settings['tmp_dir'])


        # delete the file if it exists
        if os.path.exists(cache_file_path):
            os.remove(cache_file_path)
        # Save the response to the cache file
        with open(cache_file_path, 'w') as file:
            jsn = response.json()
            json.dump(jsn, file, indent=4)

        # now update message with the response
        messages.append({"content": response_content, "role": response_first_choice_message.role})

        # Extract the usage information from the response
        d = response.usage
        lines = [f"{k}:{v}" for k, v in d]
        usage = "\n".join(lines)

        if cached_completion:
            top = "cached response used\n"
            # make top bright green
            top = f"\033[1;32;40m{top}\033[0m"
            usage = top + usage

        else:
            top = "new result generated\n"
            # text color is 30-37 for black, red, green, yellow, blue, magenta, cyan, white
            # background color is 40-47 for black, red, green, yellow, blue, magenta, cyan, white
            top = f"\033[1;31;40m{top}\033[0m"

            usage = top + usage

        #print(usage)

        return (model, messages, response_content, usage,)


@litellm_base
class LiteLLMCompletionListOfPrompts:
    """just calls LiteLLMCompletion for each prompt in the list"""

    @classmethod
    def INPUT_TYPES(cls):
        base = LiteLLMCompletion.INPUT_TYPES()
        base["required"]["pre_prompt"] = base["required"]["prompt"]
        base["required"].pop("prompt")
        base["required"]["prompts"] = ("LIST", {"default": None})
        return base

    RETURN_TYPES = ("LITELLM_MODEL", "LIST", "STRING",)
    RETURN_NAMES = ("Model", "Completions", "Usage",)

    def handler(self, **kwargs):
        import litellm
        prompts = kwargs.get("prompts", ["Hello World!"])
        completions = []
        total_usage = {}
        pre_prompt = kwargs.get("pre_prompt", "Hello World!")
        for prompt in prompts:
            kwargs["prompt"] = f"{pre_prompt}\n{prompt}"
            model, messages, completion, usage = LiteLLMCompletion().handler(**kwargs)
            completions.append(completion)

            # for k, v in usage.items():
            #     if k in total_usage:
            #         total_usage[k] += v
            #     else:
            #         total_usage[k] = v
        return (model, completions, "",)


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
        content = kwargs.get('content', "Hello World!")
        role = kwargs.get('role', "user")
        new_message = {"content": content, "role": role}
        new_messages = existing_messages + [new_message]
        return (new_messages,)


@litellm_base
class ListToMessages:
    """simply convert any list to messages check the for the correct keys in each message in the list"""

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
