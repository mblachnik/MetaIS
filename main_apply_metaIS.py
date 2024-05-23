#%%
from instance_selection.metais import MetaIS
import pandas as pd

df = pd.read_csv("data/titanic/titanic.dat.csv", sep=";")
X = df.loc[:, [c for c in df.columns if c not in ["LABEL","id"]]].values
y = df.loc[:, "LABEL"].values
X = X[:,:2]
model = MetaIS()
xp,yp = model.fit_resample(X,y)
print(X.shape)
print(xp.shape)
