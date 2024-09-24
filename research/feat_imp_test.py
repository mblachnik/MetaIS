import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from instance_selection.meta_attributes_enum import MetaAttributesEnum
from research.basics.utils import loadConfig, savePlotFig

config = loadConfig()
SHOW_STD = False

models = []
for dataset in config['datasets']:
    for alg in config['models']:
        with open(f"{config['models_dir']}{alg}\\model_{dataset}.dat_meta.pickl", 'rb') as f:
            model = pickle.load(f)
            models.append((model, dataset, alg))

for model in models:
    imp = model[0].feature_importances_
    feature_names = MetaAttributesEnum.generateColumns()
    forest_importances = pd.Series(imp, index=feature_names)
    fig, ax = plt.subplots()

    if(SHOW_STD):
        std = np.std([tree.feature_importances_ for tree in model[0].estimators_], axis=0)
        forest_importances.plot.bar(yerr=std, ax=ax)
    else:
        forest_importances.plot.bar(ax=ax)

    forest_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    fig_name = f"fig_feat_imp_{model[1]}{'_std' if SHOW_STD else ''}.png"
    savePlotFig(config, model[2], fig_name)