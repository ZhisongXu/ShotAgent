import os
import yaml

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "profiles")


def load_profile_config(profile_name: str) -> dict:
    filename = f"{profile_name}.yaml"
    filepath = os.path.join(CONFIG_DIR, filename)
    
    if not os.path.exists(filepath):
        raise ValueError(f"[Config Error] Profile '{profile_name}' not found at {filepath}")
    
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)
    
    return config
