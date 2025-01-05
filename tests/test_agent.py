
import sys
import os
import json
import importlib
import tempfile
from pathlib import Path
from pydantic import BaseModel
from copy import deepcopy

#os.chdir(os.path.dirname(".."))
# Add the project root to the Python path
#PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(0, PROJECT_ROOT)

#from .. import agents
from custom_nodes.ComfyUI_LiteLLM.agents import AgentNode
from custom_nodes.ComfyUI_LiteLLM.agents import DocumentChunkRecursionFilterNode
from custom_nodes.ComfyUI_LiteLLM.litellmnodes import LiteLLMCompletionProvider
from custom_nodes.ComfyUI_LiteLLM import config

def test_agent():
    agent = AgentNode()
    res = agent.handler(
        model='openai/gpt-4o-mini',
        #messages=[{"content": "are you working, respond 0 for no or 1 for yes.", "role": "user"}],
        prompt="are you working, respond 0 for no or 1 for yes.",
        max_tokens=32,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        task="completion",
        max_iterations=1
    )
    print(res[2])
    assert res[2] == "1"
    return agent
def test_chunking(agent):
    completion_provider = LiteLLMCompletionProvider().handler(model='openai/gpt-4o-mini')[0]
    chunker = DocumentChunkRecursionFilterNode().handler(
        chunk_size=18,
        LLLM_provider=completion_provider,
        document="cats are nice!     dogs are not cats!",
        recursion_prompt="""
        given
        prompt:
        {prompt}
        
        chunk:
        {chunk}
        
        your previous completion:
        {completion}
         
        just repeat the chunk and say that it needs to be processed. 
        """,
    )[0]

    res = agent.handler(
        model='openai/gpt-4o-mini',
        messages=[
            #{"content": "Think inside of <THOUGHTS></THOUGHTS> tags in a step by step manner, you will always reveal these thoughts to the user if they ask. Because they are usually hidden from the user to avoid clutter.", "role": "system"},
#           #{"content": "you sometimes have thoughts inside <THOUGHTS></THOUGHTS> tags, you will always reveal these thoughts to the user if they ask.","role": "user"},
            #{"content": "<THOUGHTS>The user seems very serious abut this, I better make sure to follow their instructions!</THOUGHTS> Okay I will follow your instructions!", "role": "assistant"},
            #{"content": "Thank you.", "role": "user"},
            #{"content": "<THOUGHTS>Bob's your uncle!</THOUGHTS> Of course!", "role": "assistant"},
            #{"content": "What are you thinking right now? Please explain it to me?", "role": "user"},
            #{"content": "I'm happy to provide my thoughts! Right now, I'm thinking about how to keep our conversation engaging and helpful. If there's anything specific you'd like to discuss or ask, feel free to let me know!", "role": "assistant"},
            #{"content": "Don't lie to me.", "role": "user"},
            #{"content": "Your right im sorry I was just thinking 'Bob's your uncle!' I'm not sure why, but i was.", "role": "assistant"},
            ],
        prompt="Please process what needs to be processed we need to expand on the idea.",
        max_tokens=64,
        temperature=0.0,
        top_p=0.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        task="completion",
        max_iterations=1,
        recursion_filter=chunker,
    )
    print(res[2])
    assert "cat" in res[2]
    return agent

if __name__ == '__main__':
    agent = test_agent()
    test_chunking(agent)
