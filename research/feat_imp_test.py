#%%
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from instance_selection.meta_attributes_enum import MetaAttributesEnum
from instance_selection.metais import ISMetaAttributesTransformer
from sklearn.ensemble import RandomForestClassifier
path_model = "Y:\\MetaIS\\models\\HMEI\\"
dat = "ring"

#path_data = f"Y:\\Datasets\\MetaIS\\corrected\\Filtered by HMEI\\{dat}\\"


#df = pd.read_csv(path_data + dat + ".dat.csv",sep=";")
#y = df.loc[:,"LABEL"]
#X = df.loc[:, [col for col in df.columns if col not in ("LABEL","id")]]
#tr = ISMetaAttributesTransformer()
#X_meta = tr.fit_transform(X,y)


files=[f"model_{dat}.dat_meta"]
models = []
for f in files:
    with open(path_model + f + ".pickl", 'rb') as f:
        model = pickle.load(f)
        models.append(model)

#%%
for model in models:
    #model = RandomForestClassifier()
    imp = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

    feature_names = [MetaAttributesEnum.minDistanceSameClass.value, MetaAttributesEnum.minDistanceOppositeClass.value, MetaAttributesEnum.minDistanceAnyClass.value]
    for mat in [3,5,9,15,23,33]:
        strMat = str(mat)
        feature_names.append(MetaAttributesEnum.sameClassNeighbors(strMat))
        feature_names.append(MetaAttributesEnum.oppositeClassNeighbors(strMat))
        feature_names.append(MetaAttributesEnum.meanDistanceAnyClass(strMat))
        feature_names.append(MetaAttributesEnum.meanDistanceSameClass(strMat))
        feature_names.append(MetaAttributesEnum.meanDistanceOppositeClass(strMat))

    forest_importances = pd.Series(imp, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()
