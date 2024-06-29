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
