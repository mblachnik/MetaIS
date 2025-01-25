import os
import pickle
import statistics
import pandas as pd
from instance_selection.meta_attributes_enum import MetaAttributesEnum
from research.basics.utils import getBaseResultsFilePath, loadConfig

config = loadConfig()
result_file_name = 'feat_imp_summary_merged_train_data.csv'

#grouped_k = [[3,5], [9,15,23,33]]
grouped_k = [[3, 5], [9, 15], [23, 33]]
#grouped_k = []


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
    imp = model[0].feature_importances_
    feature_names = MetaAttributesEnum.generateColumns()
    forest_importance = pd.Series(imp, index=feature_names)
    results[model[2]][model[1]] = forest_importance

k_dependent_feats = [
    MetaAttributesEnum.sameClassNeighbors, 
    MetaAttributesEnum.oppositeClassNeighbors,
    MetaAttributesEnum.meanDistanceAnyClass,
    MetaAttributesEnum.meanDistanceSameClass,
    MetaAttributesEnum.meanDistanceOppositeClass,
    ]

for x, y_series_dict in results.items():
    for dataset in config['datasets']:
        for feat in k_dependent_feats:
            for ks in grouped_k:
                meanDistanceAnyClassGroupKey = feat('_'.join(map(str, ks)))
                vals = {}
                vals[meanDistanceAnyClassGroupKey] = []
                for k in ks:
                    meanDistanceAnyClassKey = feat(str(k))
                    vals[meanDistanceAnyClassGroupKey].append(y_series_dict[dataset][meanDistanceAnyClassKey])
                    del y_series_dict[dataset][meanDistanceAnyClassKey]
                mean = statistics.mean(vals[meanDistanceAnyClassGroupKey])
                y_series_dict[dataset][meanDistanceAnyClassGroupKey] = mean

    df = pd.DataFrame(y_series_dict)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Meta attribute'}, inplace=True)
    path = getBaseResultsFilePath(config, x, result_file_name)
    df.to_csv(path, index=False)