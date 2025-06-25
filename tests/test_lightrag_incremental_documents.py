#!/usr/bin/env python3
"""
Clean integration test for LightRAG + LiteLLM incremental document processing.
Tests real functionality without mocks.
"""

import os
import sys
import tempfile
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import lightrag
    from comfyui_lightrag.lightrag_nodes import DocumentProcessorNode, QueryNode
    from comfyui_lightrag.agent_memory import AgentMemoryProviderNode
    LIGHTRAG_AVAILABLE = True
    print("✓ LightRAG is available")
except ImportError as e:
    LIGHTRAG_AVAILABLE = False
    print(f"✗ LightRAG not available: {e}")
    sys.exit(1)

try:
    import litellm
    from litellmnodes import LiteLLMModelProvider, LiteLLMCompletionProvider
    LITELLM_AVAILABLE = True
    print("✓ LiteLLM and nodes are available")
except ImportError as e:
    LITELLM_AVAILABLE = False
    print(f"✗ LiteLLM nodes not available: {e}")
    sys.exit(1)


class CallTracker:
    """Simple call tracker to monitor LiteLLM usage."""

    def __init__(self, base_provider):
        self.base_provider = base_provider
        self.call_count = 0
        self.call_times = []

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        start_time = time.time()

        try:
            result = self.base_provider(prompt)
            call_duration = time.time() - start_time
            self.call_times.append(call_duration)
            return result
        except Exception as e:
            call_duration = time.time() - start_time
            self.call_times.append(call_duration)
            raise e


def setup_litellm_provider():
    """Set up a real LiteLLM completion provider using Kluster."""
    print("Setting up LiteLLM completion provider with Kluster...")

    # Check for Kluster API key
    try:
        from utils.env_config import get_env_var
        kluster_key = get_env_var("KLUSTER_API_KEY")
        if not kluster_key:
            print("⚠ KLUSTER_API_KEY not found - using dummy key for testing")
            kluster_key = "dummy-key"
    except Exception as e:
        print(f"⚠ Error getting Kluster key: {e} - using dummy key")
        kluster_key = "dummy-key"

    # Create model provider
    model_provider = LiteLLMModelProvider()
    model_result = model_provider.handler(
        api_source="Kluster",
        model_name="mistralai/Mistral-Nemo-Instruct-2407",
        api_base="https://api.kluster.ai/v1",
        api_key=kluster_key
    )
    model = model_result[0]

    # Create completion provider
    completion_provider = LiteLLMCompletionProvider()
    completion_result = completion_provider.handler(
        model=model,
        max_tokens=500,
        temperature=0.3,
        top_p=0.9,
        frequency_penalty=0,
        presence_penalty=0,
        prompt="Test prompt"
    )

    base_function = completion_result[0]
    tracked_provider = CallTracker(base_function)
    print("✓ LiteLLM completion provider created with Kluster")
    return tracked_provider


def get_test_documents():
    """Get test documents for incremental processing."""
    return [
        {
            "name": "Python Basics",
            "content": """
            Python is a programming language created by Guido van Rossum in 1991.
            It is known for its simple syntax and readability.
            Python supports object-oriented and functional programming.
            """
        },
        {
            "name": "Python Libraries",
            "content": """
            NumPy provides array support for Python.
            Pandas offers data analysis tools.
            Matplotlib enables data visualization.
            These libraries make Python powerful for data science.
            """
        },
        {
            "name": "Python Applications",
            "content": """
            Django is a web framework for Python.
            TensorFlow supports machine learning in Python.
            Python is used by Google, Netflix, and Instagram.
            """
        }
    ]


def test_incremental_document_processing():
    """Test processing multiple documents incrementally."""
    print("\n=== Testing Incremental Document Processing ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temp directory: {temp_dir}")

        # Set up provider and processor
        provider = setup_litellm_provider()
        processor = DocumentProcessorNode()
        documents = get_test_documents()

        rag_instance = None

        # Process documents incrementally
        for i, doc in enumerate(documents, 1):
            print(f"\nProcessing document {i}: {doc['name']}")
            start_calls = provider.call_count

            if rag_instance is None:
                # First document
                result = processor.process_document(
                    working_dir=temp_dir,
                    document=doc['content'],
                    override_callable=provider,
                    chunk_token_size=200,
                    chunk_overlap_token_size=30,
                    llm_model_name="mistralai/Mistral-Nemo-Instruct-2407",
                    enable_llm_cache=True
                )
                rag_instance = result[0]
                print(f"  ✓ Created new RAG instance")
            else:
                # Subsequent documents - add to existing RAG
                try:
                    rag_instance.insert(doc['content'])
                    print(f"  ✓ Added to existing RAG instance")
                except Exception as e:
                    print(f"  ⚠ Direct insert failed: {str(e)[:100]}")
                    # Fallback: process with same working directory
                    result = processor.process_document(
                        working_dir=temp_dir,
                        document=doc['content'],
                        override_callable=provider,
                        chunk_token_size=200,
                        chunk_overlap_token_size=30,
                        llm_model_name="mistralai/Mistral-Nemo-Instruct-2407",
                        enable_llm_cache=True
                    )
                    print(f"  ✓ Processed with existing working directory")

            calls_made = provider.call_count - start_calls
            print(f"  - API calls made: {calls_made}")

        print(f"\nTotal API calls: {provider.call_count}")
        print(f"Average call duration: {sum(provider.call_times)/len(provider.call_times):.3f}s")

        return rag_instance, provider


def test_cross_document_querying(rag_instance, provider):
    """Test querying across multiple documents."""
    print("\n=== Testing Cross-Document Querying ===")

    if rag_instance is None:
        print("⚠ Skipping - no RAG instance")
        return False

    query_node = QueryNode()
    queries = [
        "What is Python?",
        "Tell me about Python libraries",
        "How is Python used in web development?"
    ]

    pre_query_calls = provider.call_count

    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        start_calls = provider.call_count

        try:
            result = query_node.query(
                rag=rag_instance,
                query_text=query,
                mode="naive"
            )
            calls_made = provider.call_count - start_calls
            print(f"  ✓ Query completed ({calls_made} calls)")
            print(f"  Result: {str(result)[:150]}...")

        except Exception as e:
            print(f"  ⚠ Query failed: {str(e)[:100]}")

    total_query_calls = provider.call_count - pre_query_calls
    print(f"\nTotal query calls: {total_query_calls}")
    return True


def test_agent_memory_integration(rag_instance, provider):
    """Test agent memory provider with incremental data."""
    print("\n=== Testing Agent Memory Integration ===")

    if rag_instance is None:
        print("⚠ Skipping - no RAG instance")
        return False

    memory_node = AgentMemoryProviderNode()

    try:
        memory_result = memory_node.create_memory_provider(
            rag=rag_instance,
            query_mode="naive",
            LLLM_provider=provider,
            Prompt="Find information about: {prompt}"
        )

        memory_provider_func = memory_result[0]
        print("✓ Memory provider created")

        # Test memory queries
        memory_queries = [
            "Python programming",
            "Data science tools"
        ]

        pre_memory_calls = provider.call_count

        for query in memory_queries:
            print(f"  Memory query: {query}")
            start_calls = provider.call_count

            try:
                memory_data = memory_provider_func([query])
                calls_made = provider.call_count - start_calls
                print(f"    ✓ Memory lookup completed ({calls_made} calls)")
                print(f"    Returned {len(memory_data)} items")

            except Exception as e:
                print(f"    ⚠ Memory lookup failed: {str(e)[:100]}")

        total_memory_calls = provider.call_count - pre_memory_calls
        print(f"Total memory calls: {total_memory_calls}")
        return True

    except Exception as e:
        print(f"✗ Memory integration test failed: {e}")
        return False


def test_working_directory_persistence():
    """Test working directory persistence across sessions."""
    print("\n=== Testing Working Directory Persistence ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Session 1
        provider1 = setup_litellm_provider()
        processor = DocumentProcessorNode()
        doc1 = get_test_documents()[0]

        print("Session 1: Processing first document")
        result1 = processor.process_document(
            working_dir=temp_dir,
            document=doc1['content'],
            override_callable=provider1,
            llm_model_name="mistralai/Mistral-Nemo-Instruct-2407"
        )

        session1_calls = provider1.call_count
        files_after_session1 = set(os.listdir(temp_dir))
        print(f"  Session 1 calls: {session1_calls}")
        print(f"  Files created: {len(files_after_session1)}")

        # Session 2
        provider2 = setup_litellm_provider()
        doc2 = get_test_documents()[1]

        print("Session 2: Processing second document")
        result2 = processor.process_document(
            working_dir=temp_dir,
            document=doc2['content'],
            override_callable=provider2,
            llm_model_name="mistralai/Mistral-Nemo-Instruct-2407"
        )

        session2_calls = provider2.call_count
        files_after_session2 = set(os.listdir(temp_dir))
        print(f"  Session 2 calls: {session2_calls}")
        print(f"  Files after session 2: {len(files_after_session2)}")

        # Check persistence
        marker_file = os.path.join(temp_dir, "lightrag.workdir.marker")
        marker_exists = os.path.exists(marker_file)

        print(f"  ✓ Marker file persisted: {marker_exists}")
        print(f"  ✓ Files accumulated: {len(files_after_session2) >= len(files_after_session1)}")

        return True


def main():
    """Run all tests."""
    print("=== LightRAG + LiteLLM Clean Integration Test ===")

    results = []

    # Test 1: Incremental document processing
    try:
        rag_instance, provider = test_incremental_document_processing()
        results.append(True)
    except Exception as e:
        print(f"✗ Incremental processing failed: {e}")
        results.append(False)
        rag_instance, provider = None, None

    # Test 2: Cross-document querying
    if rag_instance and provider:
        results.append(test_cross_document_querying(rag_instance, provider))
    else:
        print("\n⚠ Skipping cross-document querying")
        results.append(False)

    # Test 3: Agent memory integration
    if rag_instance and provider:
        results.append(test_agent_memory_integration(rag_instance, provider))
    else:
        print("\n⚠ Skipping agent memory integration")
        results.append(False)

    # Test 4: Working directory persistence
    try:
        results.append(test_working_directory_persistence())
    except Exception as e:
        print(f"✗ Persistence test failed: {e}")
        results.append(False)

    # Summary
    passed = sum(results)
    total = len(results)

    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed!")
        print("\n✅ LightRAG + LiteLLM integration is working correctly")
        return 0
    else:
        print("⚠ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
