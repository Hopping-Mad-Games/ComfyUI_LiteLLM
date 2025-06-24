from .lightrag_base import LightRAGBaseNode
from lightrag import LightRAG
from lightrag import QueryParam

mem_default_prompt = """

Given the prompt:
<Prompt>
{prompt}
</Prompt>

Provide a very concise matching_prompt that will be used to query the RAG system. The matching_prompt should not be about the task mentioned in the prompt but should be designed to retrieve the context or data that is needed for another LLM to complete the task. The matching_prompt should specify that the ouput should also be non-conversational, provided with minimal context and when possible be direct quotes from the source. The response should be very,very inormation dense.
also the matching_prompt should include the instruction for the llm to return "None" if there is nothing relevant

Your response should be only the matching_prompt, without any additional context or content.


"""


class AgentMemoryProviderNode(LightRAGBaseNode):
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "rag": ("LIGHTRAG",),
                "query_mode": (["naive", "local", "global", "hybrid"], {"default": "global"}),
            },
            "optional": {
                "LLLM_provider": ("CALLABLE", {"default": None}),
                "Prompt": ("STRING", {"default": mem_default_prompt, "multiline": True}),
            }

        }

    RETURN_TYPES = ("LLLM_AGENT_MEMORY_PROVIDER",)
    FUNCTION = "create_memory_provider"
    CATEGORY = "LiteLLM/Memory"

    def create_memory_provider(self, rag: LightRAG,
                               query_mode: str = "global",
                               LLLM_provider=None,
                               Prompt: str = mem_default_prompt):

        from typing import Union, List

        def memory_provider(query: Union[str, List[str]]) -> [str]:
            if isinstance(query, str):
                query = [query]

            data = []
            for q in query:
                # Query the RAG system
                param = QueryParam(mode=query_mode)
                refine_prompt = Prompt.format(prompt=q)
                q_prompt=LLLM_provider(refine_prompt)

                results = rag.query(q_prompt, param=param)

                # Enhance the prompt with context from memory
                if results:
                    data.append(results)

            return data

        return (memory_provider,)
