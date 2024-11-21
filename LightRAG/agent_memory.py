from .lightrag_base import LightRAGBaseNode
from lightrag import LightRAG
from lightrag import QueryParam


class AgentMemoryProviderNode(LightRAGBaseNode):
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "rag": ("LIGHTRAG",),
                "query_mode": (["naive", "local", "global", "hybrid"], {"default": "global"}),
            }
        }

    RETURN_TYPES = ("LLLM_AGENT_MEMORY_PROVIDER",)
    FUNCTION = "create_memory_provider"
    CATEGORY = "LiteLLM/Memory"

    def create_memory_provider(self, rag: LightRAG, query_mode: str = "global"):
        from typing import Union, List
        def memory_provider(query: Union[str,List[str]]) -> [str]:
            if isinstance(query, str):
                query = [query]

            data = []
            for q in query:
                # Query the RAG system
                param = QueryParam(mode=query_mode)
                results = rag.query(q, param=param)

                # Enhance the prompt with context from memory
                if results:
                    data.append(results)

            return data

        return (memory_provider,)
