# for support of being installed by ComfyUI-Manager
import sys
import os.path
import subprocess

custom_nodes_path = os.path.dirname(os.path.abspath(__file__))


def build_pip_install_cmds(args):
    if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
        return [sys.executable, '-s', '-m', 'pip', 'install'] + args
    else:
        return [sys.executable, '-m', 'pip', 'install'] + args


def ensure_package():
    cmds = build_pip_install_cmds(['-r', 'requirements.txt'])
    subprocess.run(cmds, cwd=custom_nodes_path)


# ensure_package() #no longer needed
# here we will remind the user that their keys should be in the env vars and not saved to the graph
print("ComfyUI_LiteLLM: Please ensure that your keys are stored in environment variables and dot not save to the graph.")
print("ComfyUI_LiteLLM: You can at your discretion put them in the config.yaml file, but it is not recommended.")
