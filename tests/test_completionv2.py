import unittest
import sys
import os
import torch

# Import the class to test - update import approach
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a mock CustomDict module if it doesn't exist
class MockCustomDict:
    def __init__(self):
        self.data = {}

    def update(self, d):
        self.data.update(d)

    def every_value_str(self):
        return str(self.data)

# Add the mock to sys.modules
sys.modules['CustomDict'] = MockCustomDict

# Import after mocking
from litellmnodes import LitellmCompletionV2


class TestLitellmCompletionV2(unittest.TestCase):
    def setUp(self):
        self.litellm_completion = LitellmCompletionV2()

        # Set API key in environment for testing
        os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "46f40053-2e61-43e8-ac7f-5cec7869b031")

    def test_handler(self):
        kwargs = {
            'model': 'openai/google/gemma-3-27b-it',
            'prompt': "are you working, respond 0 for no or 1 for yes.",
            'max_tokens': 1,
            'temperature': 0.5,
            'top_p': 0.5,
            'frequency_penalty': 0.5,
            'presence_penalty': 0.5,
            'task': 'completion'
        }

        model, messages, completion, completions_list, usage = self.litellm_completion.handler(**kwargs)

        self.assertIsInstance(completion, str)
        self.assertIsInstance(messages, list)
        self.assertIsInstance(completions_list, list)

        # Verify the message structure
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "are you working, respond 0 for no or 1 for yes.")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], "1")

        # Verify completions list has the same content as single completion
        self.assertEqual(len(completions_list), 1)
        self.assertEqual(completions_list[0], completion)
        self.assertEqual(completion, "1")

    def test_handler_with_image(self):
        # Create a simple test image tensor (1x10x10x3)
        test_image = torch.ones((1, 32, 32, 3), dtype=torch.float32) * 0.5

        kwargs = {
            'model': 'openai/google/gemma-3-27b-it',
            'prompt': "Describe this image briefly.",
            'max_tokens': 10,
            'temperature': 0.5,
            'top_p': 0.5,
            'frequency_penalty': 0,
            'presence_penalty': 0,
            'task': 'image_captioning',
            'image': test_image
        }

        # This test only verifies the function doesn't crash with an image
        # Since real API calls would be expensive, we'll check type structure only
        try:
            model, messages, completion, completions_list, usage = self.litellm_completion.handler(**kwargs)
            self.assertIsInstance(completion, str)
            self.assertIsInstance(messages, list)
            self.assertIsInstance(completions_list, list)
        except Exception as e:
            # If appropriate credentials aren't set up, we'll skip detailed assertions
            # Just verify the function handles the image parameter appropriately
            if "API key" in str(e) or "authentication" in str(e).lower():
                pass
            else:
                # If it fails for other reasons, propagate the error
                raise


if __name__ == '__main__':
    unittest.main()
