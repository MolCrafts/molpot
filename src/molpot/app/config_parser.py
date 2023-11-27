import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from .parse_json import read_json, write_json
import json
# import json, jsonschema

SCHEMA = json.load(open(Path(__file__).parent / "config_schema.json", "r"))


class ConfigParser:
    def __init__(self, id: str, config: dict):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        self.id = id
        self._config = config
        if id is None:  # use timestamp as default run-id
            id = datetime.now().strftime(r"%m%d_%H%M%S")

        exper_name = config["name"]

        # set save_dir where trained model and log will be saved.
        save_dir = Path(config["trainer"]["save_dir"])

        _save_dir = save_dir / "models" / exper_name / id
        _log_dir = save_dir / "log" / exper_name / id

        # make directory for saving checkpoints and log.
        _save_dir.mkdir(parents=True, exist_ok=True)
        _log_dir.mkdir(parents=True, exist_ok=True)

        # save updated config file to the checkpoint dir
        write_json(config, save_dir / "config.json")

    @classmethod
    def from_json(cls, id: str, config_path: Path | str):
        _config = read_json(config_path)

        # validate configs
        jsonschema.validate(instance=_config, schema=SCHEMA)

        return cls(id, _config)

    @classmethod
    def from_args(cls, args):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        config = args.config
        return cls(config)

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir
