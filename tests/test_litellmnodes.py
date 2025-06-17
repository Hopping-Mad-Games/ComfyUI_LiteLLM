import unittest
from ..litellmnodes import LiteLLMCompletion



class TestLiteLLMCompletion(unittest.TestCase):
    def setUp(self):
        self.litellm_completion = LiteLLMCompletion()

    def test_handler(self):
        kwargs = {
            'model': 'openai/google/gemma-3-27b-it',
            #'messages': [{"content": "are you working, respond 0 for no or 1 for yes.", "role": "user"}],
            'prompt': "are you working, respond 0 for no or 1 for yes.",
            'max_tokens': 1,
            'temperature': 0.5,
            'top_p': 0.5,
            'frequency_penalty': 0.5,
            'presence_penalty': 0.5
        }
        completion, messages,_,_, = self.litellm_completion.handler(**kwargs)
        self.assertIsInstance(completion, str)
        self.assertIsInstance(messages, list)

        # the dict that is returned for messages should be deterministic and it the service is running it should be
        # [{'content': 'are you working, respond yes or no.', 'role': 'user'},
        # {'content': '1', 'role': 'assistant'},]
        self.assertEqual(messages, [
            {"content": "are you working, respond 0 for no or 1 for yes.", "role": "user"},
            {'content': '1', 'role': 'assistant'}
        ])


if __name__ == '__main__':
    unittest.main()
