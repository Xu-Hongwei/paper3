import os
import yaml


def load_config(config_path):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(
            f"Config file is empty: {config_path}"
        )

    return config