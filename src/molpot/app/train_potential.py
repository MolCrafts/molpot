import molpot as mpot
from .base import ConfigProcessor, MolPotApp
from .utils import import_from

class PotentialTrainingConfigProcessor(ConfigProcessor):

    def check_header(self, config):
        return config

    def process(self):
        
        # now only single source and it's loader is supported
        data_config = self.config["data"]
        data_source_config = data_config["source"]
        data_loader_config = data_config["loader"]

        model_config = self.config["model"]
        trainer_config = self.config["trainer"]

        # should use a functional design?
        dataset = self.init_dataset(data_source_config)
        dataloaders = self.init_dataloader(data_loader_config, dataset)

        model = self.init_model(model_config)

        return dataset, dataloaders

    def init_dataset(self, source_config: dict):

        name = source_config["name"]
        type_ = source_config["type"]
        kwargs = source_config["kwargs"]

        match type_:
            case "builtin":
                dataset_class = import_from(name, mpot.dataset)
            case _:
                raise NotImplementedError("Unknown dataset source")

        dataset: mpot.Dataset = dataset_class(**kwargs)

        preprocess_config = source_config.get("preprocess", [])

        for proc in preprocess_config:
            name = proc["name"]
            kwargs = proc["kwargs"]
            process_class = getattr(mpot.process, name)
            dataset.add_process(process_class(**kwargs)) 

        return dataset

    def init_dataloader(self, loader_config: dict, dataset: mpot.Dataset):

        if "split" in loader_config:

            split_config = loader_config["split"]
            loader_config: list[dict] = loader_config["loader"]
            assert len(split_config) == len(
                loader_config
            ), "Number of splits and loaders do not match"

            tmp = dataset.split(split_config.values())
            subsets = {k: tmp[i] for i, k in enumerate(split_config.keys())}

            subset_dataloaders = {
                l["name"]: mpot.DataLoader(
                    subsets[l["name"]], **loader_config["kwargs"]
                )
                for l in loader_config
            }

        else:
            loader:dict = loader_config["loader"]
            subset_dataloaders = {
                loader["name"]: mpot.DataLoader(dataset, **loader_config["kwargs"])
            }

        return subset_dataloaders
    
    def init_model(self, model_config: dict):

        ...


class PotentialTrainingApp(MolPotApp):
    def __init__(self):
        ...

    