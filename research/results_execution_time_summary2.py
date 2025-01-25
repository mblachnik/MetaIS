"""
The script generates a csv results file containing information on execution time on full dataset
"""
import os
import pandas as pd
from research.basics.utils import getResultsFilePath, loadConfig

config = loadConfig()
results = []
dfs_dict = {}

for dataset in config['datasets']:
    row = []
    for alg in config['models']:
        meta_df = None

        if alg in dfs_dict:
            meta_df = dfs_dict[alg]
        else:
            meta_df = pd.read_csv(getResultsFilePath(config, alg, True, True),header=[0,1])
            dfs_dict[alg] = meta_df

        meta_df_tmp = meta_df[(meta_df['name']==dataset).values]
        meta_value = meta_df_tmp[('process_time','mean')].mean()

        if not meta_value > 0:
            meta_value = -1

        row.extend([float(meta_value)])

    results.append(row)

columns = pd.MultiIndex.from_product([config['models'], ['process_time_meta_CCIS']])
results_df = pd.DataFrame(results, index=config['datasets'], columns=columns).reset_index()
results_df.rename(columns={'index': 'dataset'}, inplace=True)
path = os.path.join(config["results_dir"], f"summary_of_execution_time_{config['result_postfix'][0]}.csv")
results_df.to_csv(path, index=False)
    