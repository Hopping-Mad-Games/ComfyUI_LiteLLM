import hashlib
import json

def get_input_hash(**kwargs):
    """
    Creates a stable hash from input parameters, handling various Python types.
    
    Handles:
    - Callable objects (using their class names)
    - Message dictionaries
    - Image data (using shape and mean)
    - Basic types (str, int, float, bool, None)
    - Lists (converting elements to strings)
    - Other types (using string representation)
    
    Args:
        **kwargs: Keyword arguments to hash
        
    Returns:
        str: MD5 hash of the input parameters
    """
    # Create a copy of kwargs without the use_last_response flag
    hash_kwargs = kwargs.copy()
    hash_kwargs.pop('use_last_response', None)
    
    # Create a list to store hashable components
    hash_components = []
    
    # Handle each input type appropriately
    for key, value in sorted(hash_kwargs.items()):
        if key in ('recursion_filter', 'memory_provider'):
            if value is not None:
                try:
                    # Get the class name for callable objects
                    if hasattr(value, '__self__'):
                        # If it's a bound method, get the class name of the instance
                        class_name = value.__self__.__class__.__name__
                    else:
                        # If it's a regular callable, get its class name
                        class_name = value.__class__.__name__
                    hash_components.append(f"{key}_class:{class_name}")
                except:
                    # If we can't get the class name, use a default identifier
                    hash_components.append(f"{key}_class:unknown")
            continue
        elif key == 'messages':
            # Convert messages to a stable string representation using json
            try:
                # Convert each message to a tuple of (role, content) and then to JSON
                messages_data = [(m.get('role', ''), m.get('content', '')) for m in value]
                messages_str = json.dumps(messages_data, sort_keys=True)
                hash_components.append(f"messages:{messages_str}")
            except:
                # If messages can't be processed, skip them
                continue
        elif key == 'image':
            # If there's image data, use its shape and mean value
            try:
                if value is not None:
                    hash_components.append(f"image_shape:{value.shape}")
                    hash_components.append(f"image_mean:{float(value.mean()):.4f}")
            except:
                # If image can't be processed, skip it
                continue
        elif isinstance(value, (str, int, float, bool, type(None))):
            # Handle basic types directly
            hash_components.append(f"{key}:{value}")
        elif isinstance(value, list):
            # Handle lists by converting to JSON for stable string representation
            try:
                list_str = json.dumps(value, sort_keys=True)
                hash_components.append(f"{key}:{list_str}")
            except:
                continue
        else:
            # For any other type, use its string representation
            try:
                hash_components.append(f"{key}:{str(value)}")
            except:
                continue
    
    # Join all components and create hash
    hash_string = "|".join(hash_components)
    return hashlib.md5(hash_string.encode()).hexdigest()
