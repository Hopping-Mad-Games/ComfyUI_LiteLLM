import yaml
import os


def read_yaml(file_path, config_replaclemts):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    for key, value in data.items():
        if isinstance(value, str):
            for key2, value2 in config_replaclemts.items():
                if key2 in value:
                    data[key] = value.replace(key2, value2)
    return data


def get_api_key(key_name):
    """Get API key from environment, checking both original and namespaced versions"""
    # First check if the key exists in environment as-is
    if key_name in os.environ:
        return os.environ[key_name]

    # Then check namespaced version
    namespaced_key = f'COMFYUI_LITELLM_{key_name}'
    if namespaced_key in os.environ:
        return os.environ[namespaced_key]

    # Return None if not found
    return None


def get_config_value(key):
    """Get configuration value safely"""
    return config_settings.get(key)


# Initialize paths
this_file_path = os.path.dirname(os.path.realpath(__file__))
node_addon_dir = this_file_path
comfy_path = os.path.dirname(os.path.dirname(this_file_path))  # path to the comfy folder

config_replacements = {
    'comfy_path': comfy_path,
    'addon_path': node_addon_dir
}

config_file_path = os.path.join(this_file_path, "config.yaml")

# Load the YAML data when the module is imported
config_settings = read_yaml(config_file_path, config_replacements)

# Add the replacements directly to the config_settings
config_settings.update(config_replacements)

# DO NOT set any environment variables to avoid polluting global environment
# Custom node packs should only read environment variables, never set them
# Other packages and the system should manage their own environment variables
