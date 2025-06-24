import yaml


def read_yaml(file_path, config_replaclemts):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    for key, value in data.items():
        if isinstance(value, str):
            for key2, value2 in config_replaclemts.items():
                if key2 in value:
                    data[key] = value.replace(key2, value2)

    return data


import os

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
# add the replacements directly to the config_settings
config_settings.update(config_replacements)
# for every item in the config set an env variable to it
for k,v in config_settings.items():
 os.environ[k]=v
