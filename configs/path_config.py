from ml_collections import ConfigDict
import os


def get_path_config():
    config = ConfigDict()
    config.proj_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir),
    )
    config.data_path = os.path.join(config.proj_path, "data")
    config.dataset_path = os.path.join(config.data_path, "raw\\PokemonData")
    config.classes_file = os.path.join(config.data_path, "utils\\classes.json")
    config.indices_file = os.path.join(config.data_path, "utils\\indices.json")
    return config