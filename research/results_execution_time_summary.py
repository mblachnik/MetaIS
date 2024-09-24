import csv
import os
import pandas as pd
import numpy as np
from research.basics.utils import getResultsFilePath, getResultsFilePaths, loadConfig

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
        
        values = []

        directory = f'{config["data_dir"]}{alg}'

        path = os.path.join(directory, f'execution_time_{dataset}.dat.log')

        if os.path.exists(path):
            with open(path, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter='\t')
                for file_row in reader:
                    if file_row and not file_row[0].startswith('#'):
                        value = float(file_row[0])
                        values.append(value)
                        break   

        # i = 1
        # path = os.path.join(directory, f'execution_time_{dataset}-5-{i}tra.dat.log')

        # while os.path.exists(path):
        #     with open(path, 'r') as csvfile:
        #         reader = csv.reader(csvfile, delimiter='\t')
        #         for file_row in reader:
        #             if file_row and not file_row[0].startswith('#'):
        #                 value = float(file_row[0])
        #                 values.append(value)
        #                 break
        #     i += 1
        #     path = os.path.join(directory, f'execution_time_{dataset}-5-{i}tra.dat.log')

        meta_df_tmp = meta_df[(meta_df['name']==dataset).values]
        meta_value = meta_df_tmp[('process_time','mean')].mean()
        IS_value = np.mean(values)

        if not IS_value > 0:
            IS_value = -1

        if not meta_value > 0:
            meta_value = -1

        row.extend([float(IS_value), float(meta_value)])

    results.append(row)

columns = pd.MultiIndex.from_product([config['models'], ['process_time_IS', 'process_time_meta_IS']])
results_df = pd.DataFrame(results, index=config['datasets'], columns=columns).reset_index()
results_df.rename(columns={'index': 'dataset'}, inplace=True)
path = os.path.join(config["results_dir"], "summary_of_execution_time.csv")
results_df.to_csv(path, index=False)
    