import os
from matplotlib import pyplot as plt
import yaml

def loadConfig():
    config_file = "config.yaml"
    if not os.path.isfile(config_file):
        config_file = "../config.yaml"
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        return config
    
def savePlotFig(config, alg: str, fig_name: str):
    figs_dir = os.path.join(f"{config['results_dir']}{alg}","figs")
    if not os.path.isdir(figs_dir):
        os.mkdir(figs_dir)
    plt.savefig(os.path.join(figs_dir, fig_name))