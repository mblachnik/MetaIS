#%%
import pickle
import pandas as pd
from instance_selection.metais import ISMetaAttributesTransformer
from sklearn.ensemble import RandomForestClassifier
path_model = "Y:\\MetaIS\\models\\HMEI\\"
dat = "ring"

path_data = f"Y:\\Datasets\\MetaIS\\corrected\\Filtered by HMEI\\{dat}\\"


df = pd.read_csv(path_data + dat + ".dat.csv",sep=";")
y = df.loc[:,"LABEL"]
X = df.loc[:, [col for col in df.columns if col not in ("LABEL","id")]]
tr = ISMetaAttributesTransformer()
X_meta = tr.fit_transform(X,y)


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
