import os
from ml_collections import ConfigDict
import yaml


def get_train_config():
    config_file = os.path.join(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir),
        ),
        "train_config.yaml",
    )
    with open(
        config_file,
        "r",
        encoding="utf-8",
    ) as f:
        cfg_dict = yaml.safe_load(f.read())

    config = ConfigDict(cfg_dict)

    return config
