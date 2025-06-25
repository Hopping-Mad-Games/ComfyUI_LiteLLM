# Agent Nodes Usage Guide

This guide explains how to use the agent nodes in ComfyUI_LiteLLM effectively. These nodes provide powerful iterative completion capabilities with memory and processing filters.

## Overview

The agent system consists of three main components:

1. **Iterative Completion Agent** (`AgentNode`) - The main processing node
2. **Completion Enhancement Filter** (`BasicRecursionFilterNode`) - Enhances completions iteratively
3. **Document Chunk Processor** (`DocumentChunkRecursionFilterNode`) - Processes large documents in chunks

## Node Descriptions

### Iterative Completion Agent (AgentNode)

**Purpose**: Performs iterative LLM completions with optional memory integration and recursion filtering.

**Key Features**:
- Multi-iteration processing (configurable max iterations)
- Memory provider integration for RAG-like functionality
- Recursion filters for post-processing completions
- Response caching system
- Support for multiple prompts in a single run

**Inputs**:
- **Required**:
  - `model`: LiteLLM model configuration
  - `prompt` or `List_prompts`: Single prompt or list of prompts to process
  - `max_iterations`: Number of processing iterations (default: 2)
- **Optional**:
  - `memory_provider`: Function to provide context/memories
  - `recursion_filter`: Function to enhance/process completions
  - `use_last_response`: Use cached response if available
  - `messages`: Initial conversation messages

**Outputs**:
- `Model`: LiteLLM model object
- `Messages`: Final conversation messages
- `Completion`: Final completion text
- `List_Completions`: All completions from all iterations
- `List_messages`: All message sets from all iterations
- `Usage`: Usage statistics

### Completion Enhancement Filter (BasicRecursionFilterNode)

**Purpose**: Creates a filter that iteratively enhances completions through multiple rounds of processing.

**How it works**:
1. Takes an initial completion
2. Applies enhancement prompt template
3. Generates improved completion
4. Repeats for specified depth

**Inputs**:
- **Required**:
  - `max_depth`: Number of enhancement iterations
  - `LLLM_provider`: Completion function (usually from LiteLLMCompletionProvider)
- **Optional**:
  - `recursion_prompt`: Template for enhancement (uses {prompt}, {completion}, {date})
  - `inner_recursion_filter`: Optional nested filter

**Output**:
- `Recursion Filter`: Function to connect to AgentNode

### Document Chunk Processor (DocumentChunkRecursionFilterNode)

**Purpose**: Processes large documents by breaking them into chunks and processing sequentially.

**How it works**:
1. Splits document into chunks of specified size
2. Processes one chunk per agent iteration
3. Maintains state across chunk processing

**Inputs**:
- **Required**:
  - `LLLM_provider`: Completion function
  - `document`: Full document text to process
  - `chunk_size`: Size of each chunk in characters
- **Optional**:
  - `recursion_prompt`: Template for chunk processing
  - `inner_recursion_filter`: Optional nested filter

**Output**:
- `Recursion Filter`: Function to connect to AgentNode

## Common Usage Patterns

### Pattern 1: Basic Iterative Completion

```
LiteLLMModelProvider → AgentNode
                       ↓
                    [Results]
```

**Setup**:
1. Connect LiteLLMModelProvider to AgentNode's model input
2. Set prompt and max_iterations
3. Leave recursion_filter and memory_provider empty

**Use case**: Simple multi-iteration refinement of responses

### Pattern 2: Enhanced Completion with Recursion Filter

```
LiteLLMCompletionProvider → BasicRecursionFilterNode → AgentNode
LiteLLMModelProvider ────────────────────────────────→ AgentNode
                                                        ↓
                                                    [Results]
```

**Setup**:
1. Connect LiteLLMCompletionProvider to BasicRecursionFilterNode's LLLM_provider
2. Connect BasicRecursionFilterNode output to AgentNode's recursion_filter
3. Connect LiteLLMModelProvider to AgentNode's model
4. Set prompts and iterations

**Use case**: Completions that get progressively more detailed and thoughtful

### Pattern 3: Document Processing

```
LiteLLMCompletionProvider → DocumentChunkRecursionFilterNode → AgentNode
LiteLLMModelProvider ──────────────────────────────────────→ AgentNode
                                                              ↓
                                                          [Results]
```

**Setup**:
1. Connect LiteLLMCompletionProvider to DocumentChunkRecursionFilterNode
2. Input your document text and set chunk_size
3. Connect filter output to AgentNode's recursion_filter
4. Set max_iterations equal to expected number of chunks

**Use case**: Processing large documents that exceed token limits

### Pattern 4: Complex Pipeline with Memory

```
[Memory System] ──────────────────→ AgentNode
LiteLLMCompletionProvider → Filter → AgentNode
LiteLLMModelProvider ─────────────→ AgentNode
                                     ↓
                                 [Results]
```

**Setup**:
1. Set up recursion filter as in previous patterns
2. Connect memory provider to provide context
3. Configure multiple iterations for complex reasoning

**Use case**: RAG-like systems with iterative reasoning

## Best Practices

### Setting Max Iterations

- **Simple tasks**: 1-2 iterations
- **Complex reasoning**: 3-5 iterations
- **Document processing**: Set to number of expected chunks
- **Note**: More iterations = higher API costs

### Recursion Prompt Templates

The recursion prompts support these placeholders:
- `{prompt}`: Original user prompt
- `{completion}`: Current completion text
- `{date}`: Current date
- `{chunk}`: Document chunk (for document processor)

### Memory Provider Integration

Memory providers should:
- Take a prompt as input
- Return a list of relevant context strings
- Be lightweight (called on every iteration)

### Caching

- Enable `use_last_response` for development/testing
- Cache is based on input hash
- Useful for expensive operations during workflow development

## Troubleshooting

### Common Issues

1. **"CALLABLE" type errors**
   - Ensure LiteLLMCompletionProvider is connected to filter nodes
   - Check that all required connections are made

2. **Memory issues with large documents**
   - Reduce chunk_size in DocumentChunkRecursionFilterNode
   - Increase max_iterations accordingly

3. **High API costs**
   - Reduce max_iterations
   - Use simpler recursion prompts
   - Enable caching during development

4. **Inconsistent results**
   - Check recursion prompt templates
   - Verify model temperature settings
   - Ensure proper message handling

### Performance Tips

- Use smaller chunk sizes for better memory management
- Cache expensive operations during development
- Monitor API usage with complex pipelines
- Test with single iterations before scaling up

## Example Workflows

### Scientific Paper Analysis

1. Use DocumentChunkRecursionFilterNode to process paper sections
2. Set recursion prompt to analyze scientific content
3. Use multiple iterations to build comprehensive analysis
4. Combine with memory provider for cross-referencing

### Creative Writing Enhancement

1. Start with basic story outline
2. Use BasicRecursionFilterNode with creativity-focused prompts
3. Multiple iterations to develop characters, plot, dialogue
4. Each iteration builds on previous work

### Technical Documentation Review

1. Process documentation with DocumentChunkRecursionFilterNode
2. Use recursion prompts focused on clarity and accuracy
3. Memory provider for style guide consistency
4. Multiple iterations for comprehensive review

## Advanced Configuration

### Custom Recursion Prompts

Create specialized prompts for your use case:
- Include specific instructions for your domain
- Use the placeholder system effectively
- Test with different prompt structures
- Consider prompt engineering best practices

### Chaining Multiple Filters

You can chain recursion filters by connecting the output of one filter to the `inner_recursion_filter` input of another:

```
Filter1 → inner_recursion_filter of Filter2 → AgentNode
```

This creates complex processing pipelines for specialized workflows.

### Integration with Other Nodes

These agent nodes work well with:
- LiteLLMMessage nodes for conversation management
- ListToMessages/MessagesToList for data conversion
- ShowMessages for debugging and visualization
- Various completion nodes for preprocessing

## Migration from Legacy Versions

If you're using older versions of these nodes:

1. **AgentNode** was previously just called "Agent Node"
2. **BasicRecursionFilterNode** provides more consistent behavior than older recursion systems
3. **DocumentChunkRecursionFilterNode** replaces manual document splitting workflows

Update your workflows to use the new clearer naming and improved functionality.