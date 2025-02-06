import os
import pickle
import statistics
import pandas as pd
from instance_selection.meta_attributes_enum import MetaAttributesEnum
from research.basics.utils import getBaseResultsFilePath, loadConfig

config = loadConfig()
postfix = config['result_postfix'][0]
result_file_name = f'tree_deep_summary_{postfix}.csv'

n_jobs = config["n_jobs"]
models = []

def loadModelsForDataset(dataset: str):
    for alg in config['models']:
        path = os.path.join(f"{config['models_dir']}{alg}", f"model_{dataset}.dat_meta.pickl")
        with open(path, 'rb') as f:
            model = pickle.load(f)
            return (model, dataset, alg)

for dataset in config['datasets']:
        models.append(loadModelsForDataset(dataset))

results = {}

for model in models:
    if model[2] not in results:
        results[model[2]] = {}
    depths = [estimator.tree_.max_depth for estimator in model[0].estimators_]
    max = max(depths)
    mean = sum(depths) / len(depths)
    print(max, mean)

