#%%
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from instance_selection.meta_attributes_enum import MetaAttributesEnum

from research.basics.config_loader import loadConfig

config = loadConfig()

#path_data = f"Y:\\Datasets\\MetaIS\\corrected\\Filtered by HMEI\\{dat}\\"


#df = pd.read_csv(path_data + dat + ".dat.csv",sep=";")
#y = df.loc[:,"LABEL"]
#X = df.loc[:, [col for col in df.columns if col not in ("LABEL","id")]]
#tr = ISMetaAttributesTransformer()
#X_meta = tr.fit_transform(X,y)

models = []
for alg in config['models']:
    for dataset in config['datasets']:
        with open(f"{config['models_dir']}{alg}\\model_{dataset}.dat_meta.pickl", 'rb') as f:
            model = pickle.load(f)
            models.append(model)
            

#%%
for model in models:
    #model = RandomForestClassifier()
    imp = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

    feature_names = MetaAttributesEnum.generateColumns()

    forest_importances = pd.Series(imp, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()
