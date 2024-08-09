
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from experiments.tools import hist_classification
from instance_selection.metais_tools import loadMetaFromDatasets, loadMetaFromDatasets_df

config_file = "config.yaml"
if not os.path.isfile(config_file):
    config_file = "../config.yaml"
    if not os.path.isfile(config_file):
        config_file = "../../config.yaml"
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

# config["datasets"]
#
# config["data_dir"]

files = [(r, f.replace(".csv",""), ".csv")
         for r,ds,fs in os.walk(config["data_dir"])
            for f in fs
                if f.endswith(".csv")
                    and ("_meta" in f)
                    and (not ("tra." in f)) and (not ("tst." in f)) #We train metamodel only for full datasets without split into training and testing
                    and any( True  if c in f else False for c in config['datasets'])
         ]
#%%
#Ładowanie plików i uruchomienie procesu uczenia meta-modelu
dfs, fNames = loadMetaFromDatasets_df(files)

#%%
plt.figure(1, clear=True)
ax=None
plot_names = []
res = []
# i represents dataset id
# j represents column id
for i in [0,1,2,3,4,6,7,8]:
    df = dfs[i].copy(deep=True)
    print(df.describe())
    #Select columns that are distance-based these columns will be normalized
    cols = [c for c in df.columns for n in ["meanDistanceAnyClass",
                                           "meanDistanceOppositeClass",
                                           "meanDistanceSameClass",
                                           "minDistanceSameClass",
                                           "minDistanceOppositeClass",
                                           "minDistanceAnyClass"] if n in c]
    #Normalizing selected columns

    for c in cols:
        X = df.loc[:, c]
        idx = X!=-1
        X = X[idx]
        if idx.isnull().any(): #Error when any value is Null
            print("Jest null")
            exit(-1)
        quantils_val = X.quantile([0.25,0.5,0.75])
        norm_val = quantils_val.values[2]-quantils_val.values[0]
        mean = quantils_val.values[1]
        df.loc[idx, c] = (X-mean)/norm_val
        df.loc[~idx, c] = -100
        res.append({"dataset":fNames[i],
                    "col":c,
                    "q1":quantils_val.values[0],
                    "q3":quantils_val.values[1],
                    "IQR":norm_val,
                    "std":np.std(X)})
    label = "_weight_"
    j=0
    for c in df.columns[j:j+1]:
        print(f"Col name: {c}")
        dft = df.loc[:,[c,label]]
        dft = dft[(dft[c]>-10) & (dft[c]<4)]
        print((dft[c].min(),dft[c].max()))
        # hist_classification(dft,col_x=c, col_y=label, bins=20)
        # plt.show()
        # dft.hist(bins=100)
        ax = dft.plot.hist(ax=ax,column=[c],
                           by=label,
                           figsize=(10, 8),
                           alpha=0.5,
                           label=files[i][1],
                           bins=20
                           )
        plot_names.append(files[i][1])
#plt.title(c)
plt.legend(plot_names)
plt.show()
res_df = pd.DataFrame(res)
print(res_df)
