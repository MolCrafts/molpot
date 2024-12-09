from pathlib import Path
from types import ModuleType

import rich
from rich.json import JSON
from importlib import import_module


def load_config(config: Path) -> dict[str, str]:

    # guess the file format
    if config.suffix == ".json":
        import json

        with open(config) as f:
            config = json.load(f)
    elif config.suffix == ".yaml":
        import yaml

        with open(config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise ValueError(f"Unknown file format: {config.suffix}")

    return config


def print_config(config: dict):

    rich.print(JSON(config, indent=2))

def import_from(class_path: str, module: ModuleType|None = None) -> type:
    """
    Obtain a molpot class type from a string

    Args:
        class_path: module path to class, e.g. ``module.submodule.classname``

    Returns:
        class type
    """
    if module is None:
        class_path = class_path.split(".")
        class_name = class_path[-1]
        module_name = ".".join(class_path[:-1])
        cls = getattr(import_module(module_name), class_name)

    else:
        cls = getattr(module, class_path)

    return cls
