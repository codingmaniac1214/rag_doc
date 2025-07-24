import yaml

def load_config(config_path):
    """
    Load configuration from YAML file.
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        return {}