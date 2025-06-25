# LightRAG Integration for ComfyUI_LiteLLM

🚀 **Production-Ready Graph-Based RAG Integration**

This module provides ComfyUI nodes that seamlessly integrate [LightRAG](https://github.com/HKUDS/LightRAG) (Graph-based Retrieval-Augmented Generation) with any LiteLLM completion provider. Use advanced graph-based knowledge indexing and retrieval with any LLM supported by LiteLLM.

## 🎯 Overview

LightRAG is a cutting-edge RAG system that uses graph structures for intelligent text indexing and retrieval. Our integration makes it compatible with **any LiteLLM-supported model** through ComfyUI workflows, enabling:

- **Graph-based knowledge extraction** from documents
- **Entity and relationship mapping** across multiple documents  
- **Incremental document processing** with persistent state
- **Cross-document query capabilities** with multiple retrieval strategies
- **Agent memory integration** for contextual LLM interactions

## 📊 Available Nodes

### 🔄 DocumentProcessorNode
- **Display Name**: "LightRAG Document Processor"
- **Purpose**: Processes documents using LightRAG and creates intelligent knowledge graphs
- **Key Features**: 
  - Compatible with built-in OpenAI functions AND any LiteLLM completion provider
  - Automatic async wrapper for LiteLLM synchronous functions
  - Incremental document processing with state persistence
  - Configurable chunking and overlap parameters

### 🔍 QueryNode
- **Display Name**: "LightRAG Query Node" 
- **Purpose**: Queries the LightRAG knowledge graph with advanced retrieval strategies
- **Query Modes**: 
  - **naive**: Simple keyword-based retrieval
  - **local**: Local context-aware retrieval
  - **global**: Global knowledge graph analysis
  - **hybrid**: Combines multiple strategies for optimal results

### ⚡ LinuxMemoryDirectoryNode
- **Display Name**: "Linux Memory Directory"
- **Purpose**: Creates high-performance temporary directory in Linux memory (`/dev/shm`)
- **Benefits**: Significantly faster processing for large knowledge graphs

### 🧠 AgentMemoryProviderNode
- **Display Name**: "Agent Memory Provider"
- **Purpose**: Creates intelligent memory provider functions for LiteLLM agents
- **Features**: Query refinement, context retrieval, and memory-enhanced conversations

## 🔧 LiteLLM Compatibility - How It Works

### The Challenge
LightRAG expects LLM functions with this async signature:
```python
async def llm_func(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs) -> str
```

But LiteLLM completion providers return functions with this synchronous signature:
```python
def completion_function(prompt) -> str
```

### ✅ Our Solution
The `DocumentProcessorNode` includes an **automatic compatibility wrapper** that seamlessly bridges this gap:

1. **🔄 Async Conversion** - Converts synchronous LiteLLM functions to async for LightRAG
2. **📝 Smart Parameter Handling** - Processes system_prompt, history_messages, keyword_extraction
3. **🧩 Intelligent Prompt Combination** - Merges all context into optimally formatted prompts
4. **🛡️ Robust Error Handling** - Comprehensive error recovery with detailed debugging info
5. **⚡ Performance Optimized** - Efficient processing with minimal overhead

### 🔄 Prompt Combination Strategy
The wrapper intelligently combines prompts as follows:
```
System: {system_prompt}

User: {previous_message_1}
Assistant: {previous_response_1}
User: {previous_message_2}

User: {current_prompt}
```

### 🛠️ Example Workflows

**Basic Document Processing:**
```
LiteLLMModelProvider → LiteLLMCompletionProvider → DocumentProcessor
```

**Complete RAG Pipeline:**
```
LiteLLMModelProvider → LiteLLMCompletionProvider → DocumentProcessor → QueryNode
                                                       ↓
                                               Knowledge Graph Created
                                                       ↓
                                               Cross-Document Queries
```

**High-Performance Setup:**
```
LinuxMemoryDirectory → DocumentProcessor → QueryNode
                            ↓
                    (Processing in /dev/shm)
```

**Agent with Memory:**
```
DocumentProcessor → AgentMemoryProvider → [Your LiteLLM Agent Workflow]
                         ↓
                 Contextual Memory Lookup
```

## 📋 Usage Guide

### 🚀 Quick Start

1. **Set up your LLM provider**:
   - Add a `LiteLLMModelProvider` node and select any supported model (GPT, Claude, Gemini, Kluster, etc.)
   - Connect it to a `LiteLLMCompletionProvider` node
   - Configure parameters (temperature, max_tokens, etc.)

2. **Process documents into knowledge graphs**:
   - Connect the completion provider to the `override_callable` input of `DocumentProcessorNode`
   - Input your document text and specify a working directory
   - The node automatically creates an intelligent LightRAG knowledge graph
   - **✨ Magic happens**: Entities, relationships, and context are extracted automatically

3. **Query your knowledge intelligently**:
   - Connect the LightRAG output to a `QueryNode`
   - Enter your query and choose a retrieval strategy:
     - **naive**: Fast keyword-based search
     - **global**: Deep knowledge graph analysis
     - **hybrid**: Best of both worlds
   - Get contextually rich responses that understand relationships between concepts

### 🔄 Advanced: Incremental Document Processing

Add multiple documents progressively while maintaining cross-document relationships:

1. Process first document → Creates initial knowledge graph
2. Process second document → Extends existing graph with new entities and relationships  
3. Query → Finds connections across ALL processed documents

### 🧠 Agent Memory Integration

Enhance your LiteLLM agents with intelligent memory:

1. Process domain-specific documents with `DocumentProcessor`
2. Create memory provider with `AgentMemoryProvider`
3. Connect to your agent workflow for contextual, knowledge-enhanced conversations

## 🔬 Technical Implementation

### 🧩 Advanced Prompt Engineering
The wrapper employs sophisticated prompt combination strategies:

**For Entity Extraction:**
```
System: You are an expert in entity extraction and relationship mapping.

User: Extract entities and relationships from the following text:
{document_chunk}

Format the response as JSON with entities and relationships arrays.
```

**For Contextual Queries:**
```
System: You are a knowledge expert with access to a comprehensive graph database.

User: Previous context: {retrieved_context}
User: Current question: {user_query}

Provide a detailed answer based on the available knowledge.
```

### 🛡️ Robust Error Handling
Comprehensive error recovery with detailed diagnostics:
```python
# API errors with helpful suggestions
LiteLLM API error (check your API key and configuration): {error}

# Wrapper errors with debugging context  
Error in LiteLLM completion wrapper: {detailed_context}

# Graceful fallbacks for missing data
LiteLLM completion returned None - check API configuration
```

### 💾 Smart Memory Management
- **Working Directory Persistence**: Knowledge graphs survive between sessions
- **Incremental Updates**: New documents extend existing graphs efficiently
- **Linux Memory Optimization**: Use `LinuxMemoryDirectoryNode` for `/dev/shm` processing
- **Marker File Protection**: Prevents accidental data loss during restarts
- **Cache Management**: LLM response caching reduces redundant API calls

### ⚡ Performance Optimizations
- **Efficient Chunking**: Configurable token sizes with intelligent overlap
- **Async Processing**: Non-blocking operations for better throughput
- **Memory Efficiency**: Optimized data structures for large knowledge graphs
- **API Call Optimization**: Minimizes redundant calls through intelligent caching

## 📦 Dependencies

- **`lightrag-hku`** - The powerful LightRAG graph-based RAG library
- **`litellm`** - Universal LLM interface (via main ComfyUI_LiteLLM package)
- **`numpy`** - For efficient vector operations
- **`networkx`** - Graph processing and analysis
- **`tenacity`** - Robust retry mechanisms for API calls

## 🧪 Thoroughly Tested

Our integration includes comprehensive test suites:
- ✅ **Real API Integration**: Tested with actual Kluster API calls
- ✅ **Multi-Document Processing**: Incremental document addition scenarios
- ✅ **Cross-Document Queries**: Relationship discovery across multiple documents
- ✅ **State Persistence**: Working directory maintenance across sessions
- ✅ **Error Recovery**: Robust handling of API failures and edge cases
- ✅ **Performance Validation**: Efficient API usage and memory management

## 📝 Important Notes

- **🔄 Backward Compatibility**: Existing workflows using built-in OpenAI functions continue to work unchanged
- **🤖 Universal Model Support**: Any LiteLLM-supported model works automatically (OpenAI, Anthropic, Google, Cohere, Kluster, etc.)
- **📚 Conversation Context**: System prompts and conversation history are intelligently handled
- **🔧 Parameter Flexibility**: All LightRAG parameters (`keyword_extraction`, `chunk_token_size`, etc.) are fully supported
- **⚡ Production Ready**: Tested extensively for real-world usage scenarios

## 🚀 Get Started

Ready to build intelligent, graph-based RAG systems with any LLM? Check out the example workflows above and start processing your documents into powerful knowledge graphs today!