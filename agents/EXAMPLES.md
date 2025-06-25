# Agent Nodes Examples

This file contains simple, concrete examples of how to configure and use the agent nodes in ComfyUI_LiteLLM.

## Example 1: Simple Text Enhancement

**Goal**: Take a basic response and make it more detailed and thoughtful.

**Workflow Setup**:
```
1. LiteLLMModelProvider
   - model: "gpt-3.5-turbo"
   
2. LiteLLMCompletionProvider
   - model: [connect from LiteLLMModelProvider]
   
3. BasicRecursionFilterNode (Completion Enhancement Filter)
   - max_depth: 2
   - LLLM_provider: [connect from LiteLLMCompletionProvider]
   - recursion_prompt: (use default or customize)
   
4. AgentNode (Iterative Completion Agent)
   - model: [connect from LiteLLMModelProvider]
   - prompt: "Explain quantum computing to a beginner"
   - max_iterations: 1
   - recursion_filter: [connect from BasicRecursionFilterNode]

Connection Flow: LiteLLMModelProvider → AgentNode
                LiteLLMCompletionProvider → BasicRecursionFilterNode → AgentNode
```

**Expected Behavior**:
- Initial completion gives basic quantum computing explanation
- Recursion filter enhances it 2 times, making it more comprehensive
- Final result is a detailed, well-structured explanation

## Example 2: Large Document Summarization

**Goal**: Summarize a long research paper by processing it in chunks.

**Workflow Setup**:
```
1. LiteLLMModelProvider
   - model: "gpt-4"
   
2. LiteLLMCompletionProvider
   - model: [connect from LiteLLMModelProvider]
   
3. DocumentChunkRecursionFilterNode
   - LLLM_provider: [connect from LiteLLMCompletionProvider]
   - document: [paste your long document text here]
   - chunk_size: 1000
   - recursion_prompt: 
     "Summarize this chunk: {chunk}
     
     Previous summary: {completion}
     
     Combine and create an updated summary."
   
4. AgentNode
   - model: [connect from LiteLLMModelProvider]
   - prompt: "Please summarize this document"
   - max_iterations: [number of chunks, e.g., 10 for 10k character document]
   - recursion_filter: [connect from DocumentChunkRecursionFilterNode]
```

**Expected Behavior**:
- Document gets split into 1000-character chunks
- Each iteration processes one chunk and updates the running summary
- Final result is a comprehensive summary of the entire document

## Example 3: Creative Story Development

**Goal**: Start with a story idea and develop it through multiple iterations.

**Workflow Setup**:
```
1. LiteLLMModelProvider
   - model: "claude-3-sonnet"
   
2. LiteLLMCompletionProvider
   - model: [connect from LiteLLMModelProvider]
   
3. BasicRecursionFilterNode
   - max_depth: 3
   - LLLM_provider: [connect from LiteLLMCompletionProvider]
   - recursion_prompt:
     "Given this story prompt: {prompt}
     
     And this current story: {completion}
     
     Enhance the story by:
     1. Adding more character depth
     2. Improving dialogue
     3. Adding sensory details
     4. Developing the plot further
     
     <enhanced_story>
     [Your enhanced version here]
     </enhanced_story>"
   
4. AgentNode
   - model: [connect from LiteLLMModelProvider]
   - prompt: "Write a short story about a time traveler who gets stuck in the past"
   - max_iterations: 2
   - recursion_filter: [connect from BasicRecursionFilterNode]
```

**Expected Behavior**:
- First iteration creates basic story
- Each recursion level adds more depth, detail, and polish
- Final story goes through 2 iterations × 3 recursion depths = 6 total enhancements

## Example 4: Research Question Analysis

**Goal**: Thoroughly analyze a complex research question from multiple angles.

**Workflow Setup**:
```
1. LiteLLMModelProvider
   - model: "gpt-4"
   
2. LiteLLMCompletionProvider
   - model: [connect from LiteLLMModelProvider]
   
3. BasicRecursionFilterNode
   - max_depth: 2
   - LLLM_provider: [connect from LiteLLMCompletionProvider]
   - recursion_prompt:
     "Research Question: {prompt}
     
     Current Analysis: {completion}
     
     Please expand this analysis by:
     1. Considering alternative perspectives
     2. Identifying potential limitations
     3. Suggesting research methodologies
     4. Adding relevant context
     5. Proposing follow-up questions
     
     Date: {date}
     
     <expanded_analysis>
     [Your expanded analysis here]
     </expanded_analysis>"
   
4. AgentNode
   - model: [connect from LiteLLMModelProvider]
   - prompt: "How might climate change affect urban planning in coastal cities over the next 50 years?"
   - max_iterations: 3
   - recursion_filter: [connect from BasicRecursionFilterNode]
```

**Expected Behavior**:
- First iteration provides initial analysis
- Each subsequent iteration deepens the analysis
- Recursion filter adds multiple perspectives and considerations
- Final result is a comprehensive, multi-faceted research analysis

## Example 5: Code Review and Enhancement

**Goal**: Review code and provide iterative improvements.

**Workflow Setup**:
```
1. LiteLLMModelProvider
   - model: "gpt-4"
   
2. LiteLLMCompletionProvider
   - model: [connect from LiteLLMModelProvider]
   
3. BasicRecursionFilterNode
   - max_depth: 2
   - LLLM_provider: [connect from LiteLLMCompletionProvider]
   - recursion_prompt:
     "Code to review: {prompt}
     
     Previous review: {completion}
     
     Please enhance this code review by:
     1. Checking for additional bugs or issues
     2. Suggesting performance optimizations
     3. Reviewing security considerations
     4. Proposing better practices
     5. Adding implementation suggestions
     
     <enhanced_review>
     [Your enhanced review here]
     </enhanced_review>"
   
4. AgentNode
   - model: [connect from LiteLLMModelProvider]
   - prompt: [paste your code here]
   - max_iterations: 1
   - recursion_filter: [connect from BasicRecursionFilterNode]
```

**Expected Behavior**:
- Initial review identifies basic issues and improvements
- Recursion filter adds deeper analysis and more suggestions
- Final result is a comprehensive code review with multiple improvement suggestions

## Example 6: Multi-Prompt Processing

**Goal**: Process several related prompts together for comparative analysis.

**Workflow Setup**:
```
1. LiteLLMModelProvider
   - model: "claude-3-sonnet"
   
2. CreateListFromStrings (or similar list creation node)
   - Create list with prompts:
     - "What are the benefits of remote work?"
     - "What are the challenges of remote work?"
     - "How can companies optimize remote work?"
   
3. AgentNode
   - model: [connect from LiteLLMModelProvider]
   - List_prompts: [connect from list creation node]
   - max_iterations: 2
   - recursion_filter: [leave empty for this example]
```

**Expected Behavior**:
- Each prompt gets processed in parallel during each iteration
- Second iteration can build on insights from all prompts
- Final result includes comprehensive analysis of all aspects

## Common Recursion Prompt Templates

### Analysis Enhancement Template
```
Original question: {prompt}
Current analysis: {completion}
Date: {date}

Please enhance this analysis by:
1. Adding more depth and detail
2. Considering alternative viewpoints
3. Providing concrete examples
4. Identifying potential limitations
5. Suggesting practical applications

<enhanced_analysis>
[Your enhanced analysis]
</enhanced_analysis>
```

### Creative Writing Template
```
Story prompt: {prompt}
Current story: {completion}

Improve this story by:
1. Developing characters more fully
2. Adding vivid sensory details
3. Improving dialogue and pacing
4. Strengthening the narrative arc
5. Enhancing emotional impact

<improved_story>
[Your improved story]
</improved_story>
```

### Technical Explanation Template
```
Topic: {prompt}
Current explanation: {completion}
Date: {date}

Enhance this explanation by:
1. Adding more technical depth
2. Including practical examples
3. Explaining prerequisites and context
4. Addressing common misconceptions
5. Providing implementation guidance

<enhanced_explanation>
[Your enhanced explanation]
</enhanced_explanation>
```

### Document Processing Template
```
Document chunk: {chunk}
Previous summary: {completion}
Original task: {prompt}

Process this chunk and update the summary:
1. Extract key information from the chunk
2. Integrate it with the previous summary
3. Maintain coherence and flow
4. Highlight important connections

<updated_summary>
[Your updated summary]
</updated_summary>
```

## Tips for Success

1. **Start Simple**: Begin with basic setups before adding complexity
2. **Test Iterations**: Try with max_iterations=1 first, then increase
3. **Monitor Costs**: More iterations and recursion depth = higher API costs
4. **Use Caching**: Enable `use_last_response` during development
5. **Customize Prompts**: Tailor recursion prompts to your specific use case
6. **Check Connections**: Ensure all CALLABLE inputs are properly connected
7. **Gradual Complexity**: Add memory providers and nested filters after mastering basics

## Troubleshooting Common Issues

**Issue**: "CALLABLE type not connected"
**Solution**: Make sure LiteLLMCompletionProvider is connected to filter nodes

**Issue**: Results are too repetitive
**Solution**: Improve recursion prompt templates or reduce max_depth

**Issue**: High API costs
**Solution**: Reduce max_iterations, max_depth, or use smaller models

**Issue**: Inconsistent quality
**Solution**: Use more specific recursion prompts and test with different models