import pandas as pd
import numpy as np
from research.basics.utils import getResultsFilePath, getSurfacesResultsFilePath, loadConfig

config = loadConfig()

dfs = []
for alg in config['models']:
    df = pd.read_csv(getResultsFilePath(config, alg, True, True),header=[0,1])
    df_ref = pd.read_csv(getResultsFilePath(config, alg, True, False),header=[0,1])
    dfs.append((df, df_ref, alg))



for t in dfs:
    results = []
    datasets = np.unique(df['name'])
    for i,ds in enumerate(datasets):
        df = t[0]
        df_ref = t[1]
        df_tmp = df[(df['name']==ds).values]
        df_tmp_red_rate = df_tmp[('red_rate','mean')].values
        df_tmp_acc = df_tmp[('acc','mean')].values
        df_ref_tmp = df_ref[(df_ref['name']==ds).values]

        powierzchniaWzorca = (df_ref_tmp[('red_rate', 'mean')] * df_ref_tmp[('acc', 'mean')]).sum()
        area = np.trapz(df_tmp_acc, x=df_tmp_red_rate)
        area += df_tmp_red_rate[0] * df_tmp_acc[0] #uwzględnienie w powierzchni pod krzywą prostokąta, którego krawędzie wzynacza oś x i oś y do pierwszego punktu pomiarowego
        results.append([ds, powierzchniaWzorca, area])

    results_df = pd.DataFrame(results, columns=['dataset', 'IS surface', 'MetaIS surface'])
    results_df.to_csv(getSurfacesResultsFilePath(config, t[2]), index=False)