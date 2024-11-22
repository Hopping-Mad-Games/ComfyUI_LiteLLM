from typing import Callable

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
    import importlib
    __package__ = globals().get("__package__")
    __package__ = __package__ or "custom_nodes.ComfyUI_LiteLLM.Agents"

    litellmnodes = importlib.import_module("..litellmnodes", __package__)
    rough_handler = litellmnodes.LitellmCompletionV2().handler

    base_handler = rough_handler
    base_input_types = litellmnodes.LitellmCompletionV2.INPUT_TYPES

    RETURN_TYPES = ("LITELLM_MODEL", "LLLM_MESSAGES", "STRING", "LIST", "LIST", "STRING",)
    RETURN_NAMES = ("Model", "Messages", "Completion", "List_Completions", "List_messages", "Usage",)

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = cls.base_input_types()
        base_inputs["required"].update({
            "max_iterations": ("INT", {"default": 100, "min": 1}),
            "List_prompts": ("LIST", {"default": None}),
        })
        base_inputs["optional"].update({
            "memory_provider": ("LLLM_AGENT_MEMORY_PROVIDER", {"default": None}),
            "recursion_filter": ("LLLM_AGENT_RECURSION_FILTER", {"default": None})
        })
        return base_inputs

    def handler(self, **kwargs):
        from copy import deepcopy
        from typing import Callable
        if kwargs.get("List_prompts", None):
            prompts = kwargs.pop("List_prompts", None)
        else:
            prompts = [kwargs.pop("prompt", None)]

        recursion_filter: Callable | None = kwargs.pop("recursion_filter", None)
        memory_provider: Callable | None = kwargs.pop("memory_provider", None)

        max_iterations = kwargs.pop("max_iterations", 1)
        if "messages" not in kwargs:
            kwargs["messages"] = []
        ret = (None, None, None, None)

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
                # for compatibility, need to add a user message
                #kwargs["messages"].append({"role": "user", "content": kwargs["prompt"]})
                ret, ret_completion, ret_messages = self.normal_step(kwargs)

            all_results.append(ret)

        messages_reults = []
        completion_list = []
        for res in all_results:
            completion = res[2]
            completion_list.append(completion)
            messages_reults.extend(res[1])

        ret = (res[0], res[1], completion, completion_list, messages_reults, "Usage",)
        return ret

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

    litellmnodes = importlib.import_module("..litellmnodes", __package__)
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
                "max_depth": ("INT", {"default": 5, "min": 1}),
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
