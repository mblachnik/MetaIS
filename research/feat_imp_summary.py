import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from instance_selection.meta_attributes_enum import MetaAttributesEnum
from research.basics.utils import getBaseResultsFilePath, loadConfig, savePlotFig

config = loadConfig()

models = []
for dataset in config['datasets']:
    for alg in config['models']:
        with open(f"{config['models_dir']}{alg}\\model_{dataset}.dat_meta.pickl", 'rb') as f:
            model = pickle.load(f)
            models.append((model, dataset, alg))

results = {}

for model in models:
    if model[2] not in results:
        results[model[2]] = {}
    imp = model[0].feature_importances_
    feature_names = MetaAttributesEnum.generateColumns()
    forest_importance = pd.Series(imp, index=feature_names)
    results[model[2]][model[1]] = forest_importance

for x, y_series_dict in results.items():
    df = pd.DataFrame(y_series_dict)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Meta attribute'}, inplace=True)
    filename = f'feat_imp_summary.csv'
    path = getBaseResultsFilePath(config, x, filename)
    df.to_csv(path, index=False)