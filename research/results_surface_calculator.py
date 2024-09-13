import pandas as pd
import numpy as np
from research.basics.utils import getResultsFilePath, loadConfig

config = loadConfig()

dfs = []
for alg in config['models']:
    df = pd.read_csv(getResultsFilePath(config, alg, True, True),header=[0,1])
    df_ref = pd.read_csv(getResultsFilePath(config, alg, True, False),header=[0,1])
    dfs.append((df, df_ref))

for t in dfs:
    datasets = np.unique(df['name'])
    for i,ds in enumerate(datasets):
        df = t[0]
        df_ref = t[1]
        df_tmp = df[(df['name']==ds).values]
        df_ref_tmp = df_ref[(df_ref['name']==ds).values]
        print(df_tmp[('red_rate','mean')])
        print(df_tmp[('acc','mean')])
        print(df_ref_tmp[('red_rate', 'mean')])
        print(df_ref_tmp[('acc', 'mean')])