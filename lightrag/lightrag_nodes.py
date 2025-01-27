# lightrag_nodes.py
from lightrag.llm import gpt_4o_mini_complete

from .lightrag_base import LightRAGBaseNode
import os

from lightrag import LightRAG

from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete


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
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                # a bigger text field
                "enable_llm_cache": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "document": ("STRING", {"default": None, "multiline": True}),
                "working_dir": ("STRING", {"default": "./tmp"}),
                "chunk_token_size": ("INT", {"default": 1200, "min": 100, "max": 5000}),
                "chunk_overlap_token_size": ("INT", {"default": 100, "min": 0, "max": 500}),
                "llm_model_func": (["gpt_4o_mini_complete", "gpt_4o_complete"], {}),
                "llm_model_name": ("STRING", {"default": "meta-llama/Llama-3.2-1B-Instruct"}),
                "override_callable": ("CALLABLE", {"default": None}),
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
        llm_model_func = kwargs.get("llm_model_func", "gpt_4o_mini_complete")
        chunk_token_size = kwargs.get("chunk_token_size", 1200)
        chunk_overlap_token_size = kwargs.get("chunk_overlap_token_size", 100)
        llm_model_name = kwargs.get("llm_model_name", "meta-llama/Llama-3.2-1B-Instruct")
        enable_llm_cache = kwargs.get("enable_llm_cache", True)
        document = kwargs.get("document", "")

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
            llm_func = gpt_4o_mini_complete if llm_model_func == "gpt_4o_mini_complete" else gpt_4o_complete
        else:
            llm_func = override_callable

        rag = LightRAG(
            working_dir=working_dir,
            chunk_token_size=chunk_token_size,
            chunk_overlap_token_size=chunk_overlap_token_size,
            llm_model_func=llm_func,
            llm_model_name=llm_model_name,
            enable_llm_cache=enable_llm_cache
        )
        if document:
            document = document.encode("utf-8", errors="surrogatepass")

            try:
                rag.insert(document)
            except Exception as e:
                try:
                    print(f"Error inserting document: {str(e)}")
                    print(f"trying to insert as string")
                    print(f"Document type: {type(document)}")
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
from lightrag import QueryParam


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
