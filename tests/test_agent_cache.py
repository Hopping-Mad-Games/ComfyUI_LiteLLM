import unittest
import sys
import os
import json
import importlib
import tempfile
from pathlib import Path
from pydantic import BaseModel
from copy import deepcopy

# Add the project root to the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from Agents.nodes import AgentNode
import config

class UserModel(BaseModel):
    name: str
    age: int

class TestAgentCache(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for cache
        self.temp_dir = tempfile.mkdtemp()
        # Mock the config settings
        self.original_config = deepcopy(config.config_settings)
        config.config_settings = {'tmp_dir': self.temp_dir}
        
        # Create an instance of AgentNode
        self.agent = AgentNode()
        
    def tearDown(self):
        # Restore original config
        config.config_settings = self.original_config
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_save_and_load_with_pydantic_model(self):
        # Create test data with a Pydantic model
        user_model = UserModel(name="test", age=25)
        response_data = {
            "model": {
                "model": "test-model",
                "type": "kwargs",
                "kwargs": {
                    "response_format": user_model
                }
            },
            "messages": [],
            "completion": "test completion",
            "completion_list": ["test"],
            "messages_results": [],
            "usage": "test usage"
        }
        
        # Save the response
        test_hash = "test_hash"
        self.agent.save_response(test_hash, response_data)
        
        # Load the response
        loaded_data = self.agent.load_response(test_hash)
        
        # Verify the data was saved and loaded correctly
        self.assertIsNotNone(loaded_data)
        self.assertEqual(loaded_data["model"]["model"], "test-model")
        self.assertEqual(loaded_data["completion"], "test completion")
        
        # Verify the Pydantic model was handled correctly
        loaded_model = loaded_data["model"]["kwargs"]["response_format"]
        self.assertIsInstance(loaded_model, UserModel)
        self.assertEqual(loaded_model.name, "test")
        self.assertEqual(loaded_model.age, 25)

    def test_save_and_load_with_pydantic_model_class(self):
        # Define a new Pydantic model class
        class ResponseFormat(BaseModel):
            max_tokens: int
            temperature: float
            stop_sequences: list[str]
        
        # Create test data with the Pydantic model class definition
        response_data = {
            "model": {
                "model": "test-model",
                "type": "kwargs",
                "kwargs": {
                    "model_class": ResponseFormat  # The class itself, not an instance
                }
            },
            "messages": [],
            "completion": "test completion",
            "completion_list": ["test"],
            "messages_results": [],
            "usage": "test usage"
        }
        
        # Save the response
        test_hash = "test_hash_class"
        self.agent.save_response(test_hash, response_data)
        
        # Load the response
        loaded_data = self.agent.load_response(test_hash)
        
        # Verify the data was saved and loaded correctly
        self.assertIsNotNone(loaded_data)
        self.assertEqual(loaded_data["model"]["model"], "test-model")
        self.assertEqual(loaded_data["completion"], "test completion")
        
        # Verify the Pydantic model class was handled correctly
        loaded_model_class = loaded_data["model"]["kwargs"]["model_class"]
        self.assertTrue(isinstance(loaded_model_class, type))
        self.assertTrue(issubclass(loaded_model_class, BaseModel))
        
        # Verify the reconstructed class has the correct fields
        model_fields = loaded_model_class.model_fields
        self.assertIn("max_tokens", model_fields)
        self.assertIn("temperature", model_fields)
        self.assertIn("stop_sequences", model_fields)
        
        # Verify we can create an instance with the loaded class
        test_instance = loaded_model_class(
            max_tokens=100,
            temperature=0.7,
            stop_sequences=[".", "!", "?"]
        )
        self.assertEqual(test_instance.max_tokens, 100)
        self.assertEqual(test_instance.temperature, 0.7)
        self.assertEqual(test_instance.stop_sequences, [".", "!", "?"])

if __name__ == '__main__':
    unittest.main()
