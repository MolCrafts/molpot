from pathlib import Path

import rich
from rich.json import JSON
import molpot as mpot


def load_config(config: Path) -> dict:

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


class ConfigProcessor:

    def __init__(self, config: dict, preview: bool = False):
        self.config = config
        assert "task" in config, "Task not defined in configuration"

        if preview:
            print_config(config)

    def process(self):
        # process the configuration
        dataset = self.init_dataset(self.config["data"]["source"])
        dataloaders = self.init_dataloader(dataset, self.config["data"]["loader"])
        return dataset, dataloaders

    def init_dataset(self, source_config: dict):

        match source_config["type"]:
            case "builtin":
                dataset_class = getattr(mpot.dataset, source_config["name"])
            case _:
                raise NotImplementedError("Unknown dataset source")

        dataset: mpot.Dataset = dataset_class(**source_config["kwargs"])
        for proc in source_config["preprocess"]:
            process_class = getattr(mpot.process, proc["name"])
            dataset.add_process(process_class(**proc["kwargs"]))

        return dataset

    def init_dataloader(self, dataset: mpot.Dataset, loader_config: dict):

        if "split" in loader_config:

            split = loader_config["split"]
            loaders = loader_config["loader"]
            assert len(split) == len(
                loaders
            ), "Number of splits and loaders do not match"

            tmp = dataset.split(split.values())
            subsets = {k: tmp[i] for i, k in enumerate(split.keys())}

            subset_dataloaders = {
                l["name"]: mpot.DataLoader(
                    subsets[l["name"]], **loader_config["kwargs"]
                )
                for l in loaders
            }

        else:
            loader = loader_config["loader"]
            subset_dataloaders = {
                loader["name"]: mpot.DataLoader(dataset, **loader_config["kwargs"])
            }

        return subset_dataloaders
