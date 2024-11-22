import unittest
import numpy as np
import sys
import os
import copy

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.hash_utils import get_input_hash

class DummyCallable:
    def __call__(self, *args, **kwargs):
        pass

class DummyClass:
    def method(self):
        pass

class TestHashUtils(unittest.TestCase):
    def test_basic_types(self):
        # Test basic Python types
        input1 = {
            "str_val": "test",
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "none_val": None
        }
        input2 = dict(input1)
        
        # Same inputs should produce same hash
        self.assertEqual(get_input_hash(**input1), get_input_hash(**input2))
        
        # Different inputs should produce different hashes
        input2["int_val"] = 43
        self.assertNotEqual(get_input_hash(**input1), get_input_hash(**input2))

    def test_list_handling(self):
        # Test list handling
        input1 = {
            "list_val": [1, 2, "three", True],
            "nested_list": [[1, 2], [3, 4]]
        }
        # Use deep copy to ensure nested structures are copied
        input2 = copy.deepcopy(input1)
        
        self.assertEqual(get_input_hash(**input1), get_input_hash(**input2))
        
        input2["list_val"].append(4)
        self.assertNotEqual(get_input_hash(**input1), get_input_hash(**input2))

    def test_callable_handling(self):
        # Test callable handling
        dummy1 = DummyClass()
        dummy2 = DummyClass()
        
        input1 = {
            "recursion_filter": dummy1.method,
            "memory_provider": DummyCallable()
        }
        input2 = {
            "recursion_filter": dummy2.method,
            "memory_provider": DummyCallable()
        }
        
        # Same class methods should produce same hash
        self.assertEqual(get_input_hash(**input1), get_input_hash(**input2))

    def test_message_handling(self):
        # Test message dictionary handling
        input1 = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
        }
        # Use deep copy to ensure nested dictionaries are copied
        input2 = copy.deepcopy(input1)
        
        self.assertEqual(get_input_hash(**input1), get_input_hash(**input2))
        
        input2["messages"][0]["content"] = "Different"
        self.assertNotEqual(get_input_hash(**input1), get_input_hash(**input2))

    def test_image_handling(self):
        # Test image tensor handling
        image1 = np.ones((64, 64, 3))
        image2 = np.ones((64, 64, 3))
        different_image = np.zeros((64, 64, 3))
        
        input1 = {"image": image1}
        input2 = {"image": image2}
        input3 = {"image": different_image}
        
        # Same image data should produce same hash
        self.assertEqual(get_input_hash(**input1), get_input_hash(**input2))
        
        # Different image data should produce different hash
        self.assertNotEqual(get_input_hash(**input1), get_input_hash(**input3))

    def test_use_last_response_handling(self):
        # Test that use_last_response is properly excluded from hash
        input1 = {
            "prompt": "test",
            "use_last_response": True
        }
        input2 = {
            "prompt": "test",
            "use_last_response": False
        }
        
        # Different use_last_response values should produce same hash
        self.assertEqual(get_input_hash(**input1), get_input_hash(**input2))

    def test_mixed_inputs(self):
        # Test combination of different types
        dummy = DummyClass()
        image = np.ones((32, 32, 3))
        
        input1 = {
            "prompt": "test",
            "temperature": 0.7,
            "recursion_filter": dummy.method,
            "messages": [{"role": "user", "content": "Hello"}],
            "image": image,
            "use_last_response": True,
            "options": ["opt1", "opt2"]
        }
        # Use deep copy for complex nested structure
        input2 = copy.deepcopy(input1)
        
        # Same complex inputs should produce same hash
        self.assertEqual(get_input_hash(**input1), get_input_hash(**input2))
        
        # Modifying any value should produce different hash
        input2["temperature"] = 0.8
        self.assertNotEqual(get_input_hash(**input1), get_input_hash(**input2))

    def test_agent_node_inputs(self):
        # Test exact input structure from AgentNode
        input1 = {
            "max_iterations": 100,
            "List_prompts": ["prompt1", "prompt2"],
            "memory_provider": DummyCallable(),
            "recursion_filter": DummyCallable(),
            "use_last_response": False,
            "model": "anthropic/claude-3-opus",
            "max_tokens": 250,
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "prompt": "Test prompt",
            "task": "completion",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ],
            "image": np.ones((64, 64, 3))
        }
        # Deep copy to test equality
        input2 = copy.deepcopy(input1)
        
        # Same inputs should produce same hash
        self.assertEqual(get_input_hash(**input1), get_input_hash(**input2))
        
        # Modify each important parameter and verify hash changes
        modifications = {
            "max_iterations": 200,
            "List_prompts": ["different", "prompts"],
            "model": "anthropic/claude-3-sonnet",
            "max_tokens": 500,
            "temperature": 0.8,
            "top_p": 0.5,
            "task": "summarization",
            "messages": [{"role": "user", "content": "Different"}]
        }
        
        for key, new_value in modifications.items():
            modified_input = copy.deepcopy(input1)
            modified_input[key] = new_value
            self.assertNotEqual(
                get_input_hash(**input1), 
                get_input_hash(**modified_input),
                f"Hash should change when {key} is modified"
            )

if __name__ == '__main__':
    unittest.main()
