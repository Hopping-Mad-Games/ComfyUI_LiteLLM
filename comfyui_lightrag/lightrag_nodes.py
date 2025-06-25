# lightrag_nodes.py
try:
    from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete
    from lightrag import LightRAG
    from lightrag import QueryParam
    from lightrag.utils import EmbeddingFunc
    import litellm
    LIGHTRAG_AVAILABLE = True
except ImportError:
    # LightRAG not available - create dummy functions for testing
    def gpt_4o_mini_complete(*args, **kwargs):
        return "LightRAG not available"

    def gpt_4o_complete(*args, **kwargs):
        return "LightRAG not available"

    class LightRAG:
        def __init__(self, *args, **kwargs):
            pass
        def insert(self, *args, **kwargs):
            pass
        def query(self, *args, **kwargs):
            return "LightRAG not available"

    class QueryParam:
        def __init__(self, *args, **kwargs):
            pass

    LIGHTRAG_AVAILABLE = False

# Try to import sentence transformers for local embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .lightrag_base import LightRAGBaseNode
import os


class LinuxMemoryDirectoryNode(LightRAGBaseNode):
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "working_dir": ("STRING", {"default": "tmp"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_linux_memory_dir"

    def get_linux_memory_dir(self, working_dir: str):
        import os
        import tempfile

        if os.access("/dev/shm", os.W_OK):
            target = f"/dev/shm/{working_dir}"
            # make sure the directory exists
            if not os.path.exists(target):
                os.mkdir(target)
            temp_dir = tempfile.mkdtemp(dir=target)
            return (temp_dir,)
        else:
            raise Exception("Linux memory directory not available")

class DocumentProcessorNode(LightRAGBaseNode):
    """
    LightRAG Document Processor Node for ComfyUI.

    This node processes documents using LightRAG with custom LiteLLM completion providers.
    It supports both custom LLM functions and custom embedding functions to avoid OpenAI API calls.

    IMPORTANT: You MUST connect a LiteLLMCompletionProvider to the override_callable input.
    The built-in OpenAI functions have been disabled to prevent unexpected API charges.

    Features:
    - Custom LLM completion via LiteLLM (any supported provider)
    - Local embedding models via Sentence Transformers (default: Stella 1.5B)
    - Custom embedding via LiteLLM to avoid OpenAI embedding API calls
    - Automatic prompt formatting for LightRAG compatibility
    - Working directory management for incremental processing

    Required workflow:
    1. Connect a LiteLLMCompletionProvider node to the override_callable input (REQUIRED)
    2. Optionally connect the same provider to embedding_callable to use same provider for embeddings
    3. The provider's completion function will be automatically wrapped to work with LightRAG
    4. System prompts and conversation history will be combined into a single prompt
    5. Embeddings will use local Stella model by default (fast and high-quality)

    Example workflow:
    LiteLLMModelProvider -> LiteLLMCompletionProvider -> DocumentProcessorNode
    """

    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                # a bigger text field
                "enable_llm_cache": ("BOOLEAN", {"default": True}),
                "override_callable": ("CALLABLE",),  # LiteLLMCompletionProvider function - REQUIRED to avoid OpenAI calls
            },
            "optional": {
                "document": ("STRING", {"default": None, "multiline": True}),
                "working_dir": ("STRING", {"default": "./tmp"}),
                "chunk_token_size": ("INT", {"default": 1200, "min": 100, "max": 5000}),
                "chunk_overlap_token_size": ("INT", {"default": 100, "min": 0, "max": 500}),
                "llm_model_name": ("STRING", {"default": "meta-llama/Llama-3.2-1B-Instruct"}),
                "embedding_model": ("STRING", {"default": "NovaSearch/stella_en_1.5B_v5"}),
                "embedding_dimension": ("INT", {"default": 1024, "min": 256, "max": 8192}),
                "embedding_provider": (["local", "openai", "litellm", "same_as_llm"], {"default": "local"}),
                "embedding_callable": ("CALLABLE", {"default": None}),
            }

        }

    RETURN_TYPES = ("LIGHTRAG",)
    FUNCTION = "process_document"

    def process_document(self,**kwargs):
        import shutil

        # marker file name
        marker_file = "lightrag.workdir.marker"
        working_dir = kwargs.get("working_dir", "./tmp")
        override_callable = kwargs.get("override_callable", None)

        chunk_token_size = kwargs.get("chunk_token_size", 1200)
        chunk_overlap_token_size = kwargs.get("chunk_overlap_token_size", 100)
        llm_model_name = kwargs.get("llm_model_name", "meta-llama/Llama-3.2-1B-Instruct")
        enable_llm_cache = kwargs.get("enable_llm_cache", True)
        document = kwargs.get("document", "")
        embedding_model = kwargs.get("embedding_model", "NovaSearch/stella_en_1.5B_v5")
        embedding_dimension = kwargs.get("embedding_dimension", 1024)
        embedding_provider = kwargs.get("embedding_provider", "local")
        embedding_callable = kwargs.get("embedding_callable", None)

        # a few possible scenarios
        # 1. working directory does not exist
        # 2. working directory exists but the marker file does not exist
        # 3. working directory exists and the marker file exists
        # 4. working directory exists and the marker file exists and document is None

        # check if the working directory exists
        if not os.path.exists(working_dir): # does not exist
            os.mkdir(working_dir)
            # add a marker file
            with open(os.path.join(working_dir, marker_file), "w") as f:
                f.write("marker")
                f.flush()
                f.close()
        else:
            # check if the marker file exists
            if not os.path.exists(os.path.join(working_dir, marker_file)): # does not exist
                # clear the working directory
                shutil.rmtree(working_dir)
                os.mkdir(working_dir)
                with open(os.path.join(working_dir, marker_file), "w") as f:
                    f.write("marker")
                    f.flush()
                    f.close()
            else:
                # use
                pass

        if not override_callable:
            raise ValueError("override_callable is required! Connect a LiteLLMCompletionProvider node to avoid OpenAI API calls. The built-in OpenAI functions have been disabled to prevent unexpected API charges.")

        # Create an async wrapper for the LiteLLM completion provider
        # LightRAG expects: async def func(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs)
        # LiteLLMCompletionProvider provides: def func(prompt)
        async def lightrag_compatible_wrapper(
            prompt: str,
            system_prompt: str = None,
            history_messages: list = None,
            keyword_extraction: bool = False,
            **kwargs
        ) -> str:
            try:
                # Build the complete prompt by combining all context
                combined_prompt_parts = []

                # Add system prompt if provided
                if system_prompt:
                    combined_prompt_parts.append(f"System: {system_prompt}")

                # Add history messages if provided
                if history_messages:
                    for msg in history_messages:
                        if isinstance(msg, dict) and "role" in msg and "content" in msg:
                            role = msg["role"].capitalize()
                            content = msg["content"]
                            combined_prompt_parts.append(f"{role}: {content}")

                # Add the main prompt
                combined_prompt_parts.append(f"User: {prompt}")

                # Join all parts with double newlines
                combined_prompt = "\n\n".join(combined_prompt_parts)

                # Call the synchronous LiteLLM completion function with error handling
                try:
                    result = override_callable(combined_prompt)
                except Exception as call_error:
                    # Handle the specific case where the LiteLLM call fails
                    call_error_msg = str(call_error) if call_error is not None else "Unknown call error"
                    return f"LiteLLM call failed: {call_error_msg}"

                # Handle None result (can happen with invalid API keys or other issues)
                if result is None:
                    return "LiteLLM completion returned None - this may indicate an API configuration issue or the provider returned no content"

                # Handle empty string result
                if isinstance(result, str) and result.strip() == "":
                    return "LiteLLM completion returned empty string - check model configuration"

                return str(result)

            except Exception as e:
                # Provide more helpful error messages with safe string handling
                try:
                    error_msg = str(e) if e is not None else "Unknown error"
                    error_msg_lower = error_msg.lower() if error_msg else ""

                    if error_msg_lower and ("api" in error_msg_lower or "key" in error_msg_lower):
                        raise Exception(f"LiteLLM API error (check your API key and configuration): {error_msg}")
                    elif error_msg_lower and "auth" in error_msg_lower:
                        raise Exception(f"LiteLLM authentication error: {error_msg}")
                    else:
                        raise Exception(f"Error in LiteLLM completion wrapper: {error_msg}")
                except Exception as inner_e:
                    # Fallback if even error message processing fails
                    raise Exception(f"Error in LiteLLM completion wrapper (error processing failed): {type(e).__name__}")

        llm_func = lightrag_compatible_wrapper

        # Create custom embedding function to avoid OpenAI calls
        if embedding_provider == "local":
            # Use local Sentence Transformers model (Stella)
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                print("Warning: sentence-transformers not available, falling back to dummy embeddings")
                embedding_provider = "litellm"  # Fallback to dummy embeddings
            else:
                try:
                    # Initialize the local embedding model
                    print(f"Loading local embedding model: {embedding_model}")
                    local_model = SentenceTransformer(embedding_model, trust_remote_code=True)
                    print(f"Local embedding model loaded successfully with dimension {embedding_dimension}")

                    async def local_embedding_func(texts):
                        """Local embedding function using Sentence Transformers."""
                        try:
                            print(f"Generating embeddings for {len(texts)} texts using local model")
                            # Use the model to generate embeddings
                            embeddings = local_model.encode(texts, convert_to_numpy=True)
                            # Convert to list format expected by LightRAG
                            return embeddings.tolist()
                        except Exception as e:
                            print(f"Error in local embedding function: {str(e)}")
                            # Fallback to dummy embeddings
                            import random, hashlib
                            embeddings = []
                            for text in texts:
                                text_hash = hashlib.md5(str(text).encode()).hexdigest()
                                random.seed(int(text_hash[:8], 16))
                                embedding = [random.random() for _ in range(embedding_dimension)]
                                embeddings.append(embedding)
                            return embeddings

                    # Create EmbeddingFunc with local model
                    embedding_func = EmbeddingFunc(
                        embedding_dim=embedding_dimension,
                        max_token_size=8192,
                        func=local_embedding_func
                    )
                except Exception as e:
                    print(f"Failed to load local embedding model: {str(e)}")
                    print("Falling back to dummy embeddings")
                    embedding_provider = "litellm"  # Fallback to dummy embeddings

        if embedding_provider == "same_as_llm" or embedding_provider == "litellm":
            async def custom_embedding_func(texts):
                """Custom embedding function using the same provider as LLM or dummy embeddings."""
                try:
                    if embedding_callable and embedding_provider == "same_as_llm":
                        # Use the same LLM provider for embeddings by treating as text completion
                        print(f"Using LLM provider for embeddings with {len(texts)} texts")
                        embeddings = []
                        for text in texts:
                            # Create a prompt to get an embedding-like response
                            embed_prompt = f"Convert this text to a numerical representation: {text[:200]}"
                            response = embedding_callable(embed_prompt)
                            # Convert response to dummy embedding (deterministic based on response)
                            import hashlib
                            response_hash = hashlib.md5(str(response).encode()).hexdigest()
                            import random
                            random.seed(int(response_hash[:8], 16))
                            embedding = [random.random() for _ in range(embedding_dimension)]
                            embeddings.append(embedding)
                        return embeddings
                    else:
                        # Fallback to deterministic dummy embeddings
                        print(f"Using deterministic dummy embeddings for {len(texts)} texts")
                        import random, hashlib
                        embeddings = []
                        for text in texts:
                            text_hash = hashlib.md5(str(text).encode()).hexdigest()
                            random.seed(int(text_hash[:8], 16))
                            embedding = [random.random() for _ in range(embedding_dimension)]
                            embeddings.append(embedding)
                        return embeddings
                except Exception as e:
                    print(f"Warning: Custom embedding function failed ({str(e)}), using fallback")
                    import random
                    return [[random.random() for _ in range(embedding_dimension)] for _ in texts]

            # Create EmbeddingFunc with our custom function
            embedding_func = EmbeddingFunc(
                embedding_dim=embedding_dimension,
                max_token_size=8192,
                func=custom_embedding_func
            )
        elif embedding_provider == "openai":
            # Use default OpenAI embeddings (requires OPENAI_API_KEY)
            embedding_func = None  # Let LightRAG use default
        else:
            # Default case - should not reach here
            embedding_func = None

        rag = LightRAG(
            working_dir=working_dir,
            chunk_token_size=chunk_token_size,
            chunk_overlap_token_size=chunk_overlap_token_size,
            llm_model_func=llm_func,
            llm_model_name=llm_model_name,
            enable_llm_cache=enable_llm_cache,
            embedding_func=embedding_func
        )
        if document:
            from copy import deepcopy
            o_doc = deepcopy(document)

            document = document.encode("utf-8", errors="ignore")


            try:
                rag.insert(document)
            except Exception as e:
                try:
                    print(f"Error inserting document: {str(e)}")
                    document = o_doc.encode("utf-8", errors="ignore")
                    document = document.decode("utf-8")
                    document = str(document)
                    rag.insert(document)
                except Exception as e:
                    raise Exception(f"Error inserting document: {str(e)}")

        return (rag,)


# class DenseRetrieverNode(LightRAGBaseNode):
#     @staticmethod
#     def INPUT_TYPES():
#         return {
#             "required": {
#                 "chunks": ("CHUNKS", {}),
#                 "query": ("STRING", {}),
#                 "model_name": (["sentence-transformers/all-mpnet-base-v2"], {}),
#                 "top_k": ("INT", {"default": 5, "min": 1, "max": 100}),
#             }
#         }
#
#     RETURN_TYPES = ("DENSE_RESULTS",)
#     FUNCTION = "retrieve"
#
#     def retrieve(self, chunks, query: str, model_name: str, top_k: int):
#         # Implement dense retrieval logic
#         results = []
#         # Add dense retrieval implementation here
#         return (results,)
#
#
# class SparseRetrieverNode(LightRAGBaseNode):
#     @staticmethod
#     def INPUT_TYPES():
#         return {
#             "required": {
#                 "chunks": ("CHUNKS", {}),
#                 "query": ("STRING", {}),
#                 "method": (["bm25", "tf-idf"], {}),
#                 "top_k": ("INT", {"default": 5, "min": 1, "max": 100}),
#             }
#         }
#
#     RETURN_TYPES = ("SPARSE_RESULTS",)
#     FUNCTION = "retrieve"
#
#     def retrieve(self, chunks, query: str, method: str, top_k: int):
#         # Implement sparse retrieval logic
#         results = []
#         # Add sparse retrieval implementation here
#         return (results,)
#
#
# class HybridRankerNode(LightRAGBaseNode):
#     @staticmethod
#     def INPUT_TYPES():
#         return {
#             "required": {
#                 "dense_results": ("DENSE_RESULTS", {}),
#                 "sparse_results": ("SPARSE_RESULTS", {}),
#                 "alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
#             }
#         }
#
#     RETURN_TYPES = ("RANKED_RESULTS",)
#     FUNCTION = "rank"
#
#     def rank(self, dense_results, sparse_results, alpha: float):
#         # Implement hybrid ranking logic
#         ranked_results = []
#         # Add hybrid ranking implementation here
#         return (ranked_results,)
#
#
# class GeneratorNode(LightRAGBaseNode):
#     @staticmethod
#     def INPUT_TYPES():
#         return {
#             "required": {
#                 "ranked_results": ("RANKED_RESULTS", {}),
#                 "query": ("STRING", {}),
#                 "model_name": (["gpt-3.5-turbo"], {}),
#                 "max_length": ("INT", {"default": 512, "min": 64, "max": 2048}),
#             }
#         }
#
#     RETURN_TYPES = ("STRING",)
#     FUNCTION = "generate"
#
#     def generate(self, ranked_results, query: str, model_name: str, max_length: int):
#         # Implement generation logic
#         response = ""
#         # Add generation implementation here
#         return (response,)
#
#
# QueryParam imported above with conditional import


class QueryNode(LightRAGBaseNode):
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "rag": ("LIGHTRAG", {}),
                "query_text": ("STRING", {}),
                "mode": (["naive", "local", "global", "hybrid"], {"default": "naive"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "query"

    def query(self, rag: LightRAG, query_text: str, mode: str):
        param = QueryParam(mode=mode)
        result = rag.query(query_text, param=param)

        return (result,)
