"""
Project configuration utilities.
"""

from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path("configs/config.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load project configuration from YAML."""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
