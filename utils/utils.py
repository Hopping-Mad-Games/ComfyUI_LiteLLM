import re
import os

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