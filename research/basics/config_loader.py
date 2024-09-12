import os

import yaml


def loadConfig():
    config_file = "config.yaml"
    if not os.path.isfile(config_file):
        config_file = "../config.yaml"
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        return config