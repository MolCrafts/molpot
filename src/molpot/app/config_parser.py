import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from .parse_json import read_json, write_json
import json, jsonschema

DEFAULT_CONFIG = json.load(open("config.json", "r"))

class ConfigParser:
    def __init__(
        self, config_path: Path | str, format: str = "json", resume=None, run_id=None
    ):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        self.resume = resume

        if format == "json":
            self._config = read_json(config_path)
        else:
            raise NotImplementedError
        
        # validate configs
        jsonschema.validate(instance=self._configs, schema=DEFAULT_CONFIG)

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config["trainer"]["save_dir"])

        exper_name = self.config["name"]
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r"%m%d_%H%M%S")
        self._save_dir = save_dir / "models" / exper_name / run_id
        self._log_dir = save_dir / "log" / exper_name / run_id

        # make directory for saving checkpoints and log.
        self._save_dir.mkdir(parents=True, exist_ok=False)
        self._log_dir.mkdir(parents=True, exist_ok=False)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / "config.json")

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
