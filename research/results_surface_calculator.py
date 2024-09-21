import os
import pandas as pd
import numpy as np
from research.basics.utils import getResultsFilePath, getResultsFilePaths, loadConfig

config = loadConfig()
LICZBA_POMIAROW = 5

dfs = []
datasets = set()
thresholds = set()

for alg in config['models']:
    df = pd.read_csv(getResultsFilePath(config, alg, False, True),header=0)
    datasets.update(df['name'].unique())
    thresholds.update(df['threshold'].unique())
    df_ref = pd.DataFrame()
    ref_file_names = getResultsFilePaths(config, alg, False, False)
    for fileName in ref_file_names:
        temp_df = pd.read_csv(fileName)
        df_ref = pd.concat([df_ref, temp_df], ignore_index=True)
    dfs.append((df, df_ref, alg))


results = []

for i,ds in enumerate(datasets):
    row = []
    for df, df_ref, alg in dfs:
        df_ref_tmp = df_ref[(df_ref['name']==ds).values]
        df_ref_tmp['powierzchnia_wzorca'] = df_ref_tmp['acc'] * df_ref_tmp['red_rate']
        IS_mean = df_ref_tmp['powierzchnia_wzorca'].mean()
        IS_std = df_ref_tmp['powierzchnia_wzorca'].std()

        curves = []
        df_tmp = df[(df['name']==ds).values]
        for i in range(LICZBA_POMIAROW):
            acc_values = df_tmp.groupby('threshold')['acc'].nth(i)
            red_rate_values = df_tmp.groupby('threshold')['red_rate'].nth(i)
            a = acc_values.iloc[0]
            b = red_rate_values.iloc[0]
            auc = np.trapz(acc_values, x=red_rate_values) + a * b
            curves.append(auc)
        
        metaIS_mean = np.mean(curves)
        metaIS_std = np.std(curves)
        row.extend([IS_mean, IS_std, metaIS_mean, metaIS_std])
    results.append(row)

columns = pd.MultiIndex.from_product([config['models'], ['mean_IS', 'std_IS', 'mean_metaIS', 'std_metaIS']])
results_df = pd.DataFrame(results, index=list(datasets), columns=columns).reset_index()
results_df.rename(columns={'index': 'dataset'}, inplace=True)
path = os.path.join(config["results_dir"], "summary_of_areas_under_curves.csv")
results_df.to_csv(path, index=False)