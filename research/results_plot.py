import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml

from instance_selection.metais import MetaIS
from research.basics.utils import loadConfig


def plot_cls(df):
    labPos = df["LABEL"] == 1
    plt.figure(1, clear=True)
    plt.scatter(df.loc[labPos, "At1"], df.loc[labPos, "At2"])
    plt.scatter(df.loc[~labPos, "At1"], df.loc[~labPos, "At2"])
    plt.show()

config = loadConfig()

# data_dir
# models_dir
# results_dir

dataset = "banana"
dataDir = f"{config['data_dir']}{dataset}\\{dataset}\\"
#%%
data = pd.read_csv(f"{dataDir}{dataset}.dat.csv",sep=";")
proto_idx = pd.read_csv(f"{dataDir}{dataset}.dat_proto.csv", sep=";").rename(columns={"id": "_id_"})

df = pd.concat((data, proto_idx), axis=1)

proto = df[df.weight==1]
#%%
plot_cls(df)
plot_cls(proto)

#%%
threshold=0.55
model_path = os.path.join(config["models_dir"], f"model_{dataset}.dat_meta.pickl")

model_meta = MetaIS(estimator_src=model_path, threshold=threshold)

X_train = df[["At1","At2"]]
y_train = df[["LABEL"]]

Xp_train,yp_train = model_meta.fit_resample(X_train, y_train)


proto_meta = pd.concat((Xp_train,yp_train),axis=1)
plot_cls(proto_meta)