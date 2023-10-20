import json


def read_json(config_path):
    configs = json.load(open(config_path, "r"))
    return configs


def write_json(configs, config_path):
    json.dump(configs, open(config_path, "w"), indent=4, sort_keys=False)


CONFIG_EXAMPLE = {
    "name": "MolPot_Config_Example",
    "n_gpu": 0,
    "model": {"type": "PiNet", "args": {}},
    "data_loader": {
        "type": "QM9",
        "args": {
            "data_dir": "data/",
            "pipelines": {
                "batch": {"batch_size": 128},
                "shuffle": {},
                "random_split": {
                        "total_length": 10,
                        "weights": {"train": 0.8, "valid": 0.2},
                        "seed": 0,
                },
            },
            "num_workers": 2,
        },
    },
    "optimizer": {
        "type": "Adam",
        "args": {"lr": 0.001, "weight_decay": 0, "amsgrad": True},
    },
    "loss": "nll_loss",
    "metrics": ["accuracy", "top_k_acc"],
    "lr_scheduler": {"type": "StepLR", "args": {"step_size": 50, "gamma": 0.1}},
    "trainer": {
        "epochs": 100,
        "save_dir": "saved/",
        "save_period": 1,
        "verbosity": 2,
        "monitor": "min val_loss",
        "early_stop": 10,
    },
}
