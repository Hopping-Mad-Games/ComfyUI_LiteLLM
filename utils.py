import re
import os

def mask_sensitive_data(text):
    """
    Mask sensitive data in the given text.
    This includes any data containing 'KEY' (case-insensitive) and other potential credentials.
    """
    # Function to replace matched sensitive data
    def replace_sensitive(match):
        full_match = match.group(0)
        # Keep first 4 and last 4 characters, mask the rest
        if len(full_match) > 8:
            return full_match[:4] + '*' * (len(full_match) - 8) + full_match[-4:]
        else:
            return '*' * len(full_match)

    # Mask environment variables containing 'KEY'
    for var, value in os.environ.items():
        if 'KEY' in var.upper() and value in text:
            text = text.replace(value, replace_sensitive(value))

    # Mask patterns that look like API keys, access tokens, or contain 'KEY'
    sensitive_pattern = r'\b((?=.*KEY)[A-Za-z0-9_-]{20,}|[A-Za-z0-9_-]{20,})\b'
    text = re.sub(sensitive_pattern, replace_sensitive, text, flags=re.IGNORECASE)

    return text

def safe_print(message):
    """
    Safely print a message, masking any sensitive data.
    """
    print(mask_sensitive_data(str(message)))

def safe_error(error):
    """
    Safely format an error message, masking any sensitive data.
    """
    return mask_sensitive_data(str(error))
from pydantic import BaseModel

class CustomDict(dict):
    def every_value_str(self):
        def convert_value(value):
            if isinstance(value, dict):
                # Recursively handle nested dicts
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                # Recursively handle lists and tuples
                return type(value)(convert_value(v) for v in value)
            elif isinstance(value, BaseModel):
                # Convert Pydantic BaseModel to its dict and apply conversion
                return {k: convert_value(v) for k, v in value.dict().items()}
            else:
                # Convert non-dict, non-list/tuple, non-BaseModel values to strings
                return str(value)

        # Apply the conversion to the dictionary's values
        return {key: convert_value(value) for key, value in self.items()}