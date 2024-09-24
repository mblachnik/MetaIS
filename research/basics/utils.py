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

def getResultsFilePathWithPostfix(config, alg: str, agg: bool, meta: bool, postfix: str):
    return getBaseResultsFilePath(config, alg, f"results_{'Meta' if meta else ''}IS_{'agg_' if agg else ''}{postfix}.csv")

def getResultsFilePaths(config, alg: str, agg: bool, meta: bool):
    files = []
    for postfix in (config['result_postfix'] if meta else config['IS_result_postfix']):
        files.append(getResultsFilePathWithPostfix(config, alg, agg, meta, postfix))
    return files

def getResultsFilePath(config, alg: str, agg: bool, meta: bool):
    return getResultsFilePaths(config, alg, agg, meta)[0]

def getSurfacesResultsFilePath(config, alg: str):
    return getBaseResultsFilePath(config, alg, f"results_surfaces_{config['result_postfix']}.csv")

def getBaseResultsFilePath(config, alg: str, filename: str):
    return os.path.join(config["results_dir"], alg, filename)