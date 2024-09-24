import os
import pandas as pd
import numpy as np
from research.basics.utils import getResultsFilePath, getResultsFilePathWithPostfix, getResultsFilePaths, loadConfig

config = loadConfig()
LICZBA_POMIAROW = 5

dfs = []
datasets = set()

df_1NN = pd.read_csv(getResultsFilePathWithPostfix(config, '1NN', False, False, "v0"),header=0)
datasets.update(df_1NN['name'].unique())


for alg in config['models']:
    df = pd.read_csv(getResultsFilePath(config, alg, False, True),header=0)
    df_ref = pd.DataFrame()
    ref_file_names = getResultsFilePaths(config, alg, False, False)
    for fileName in ref_file_names:
        temp_df = pd.read_csv(fileName)
        df_ref = pd.concat([df_ref, temp_df], ignore_index=True)
    dfs.append((df, df_ref, alg))


results = []
first_red_rate = 0.0

for i,ds in enumerate(datasets):
    row = []
    for df, df_ref, alg in dfs:
        df_ref_tmp = df_ref[(df_ref['name']==ds).values]
        curves_metaIS = []
        curves_IS = []
        df_tmp1 = df_1NN[(df_1NN['name']==ds).values]
        df_tmp2 = df[(df['name']==ds).values]

        for i in range(LICZBA_POMIAROW):
            acc_values = list()
            first_acc = df_tmp1['acc'].iloc[i]
            acc_values.append(first_acc)
            acc_values.extend(df_tmp2.groupby('threshold')['acc'].nth(i))
            red_rate_values = [first_red_rate]
            red_rate_values.extend(df_tmp2.groupby('threshold')['red_rate'].nth(i))
            auc = np.trapz(acc_values, x=red_rate_values)
            curves_metaIS.append(auc)
            acc_IS = df_ref_tmp['acc'].iloc[i]
            red_rate_IS = df_ref_tmp['red_rate'].iloc[0]
            auc = np.trapz([first_acc, acc_IS], x=[first_red_rate, red_rate_IS])
            curves_IS.append(auc)
        
        metaIS_mean = np.mean(curves_metaIS)
        metaIS_std = np.std(curves_metaIS)
        IS_mean = np.mean(curves_IS)
        IS_std = np.std(curves_IS)
        row.extend([IS_mean, IS_std, metaIS_mean, metaIS_std])
    results.append(row)

columns = pd.MultiIndex.from_product([config['models'], ['mean_IS', 'std_IS', 'mean_metaIS', 'std_metaIS']])
results_df = pd.DataFrame(results, index=list(datasets), columns=columns).reset_index()
results_df.rename(columns={'index': 'dataset'}, inplace=True)
path = os.path.join(config["results_dir"], "summary_of_areas_under_curves.csv")
results_df.to_csv(path, index=False)