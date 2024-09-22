import csv
import os
import pandas as pd
import numpy as np
from research.basics.utils import getResultsFilePath, getResultsFilePaths, loadConfig

config = loadConfig()

dfs = []
datasets = set()
thresholds = set()

data = []

for alg in config['models']:
    for dataset in config['datasets']:
        values = []
        i = 1
        directory = f'{config["data_dir"]}{alg}'
        path = os.path.join(directory, f'execution_time_{dataset}-5-{i}tra.dat.log')
        while os.path.exists(path):
            with open(path, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter='\t')
                for row in reader:
                    if row and not row[0].startswith('#'):
                        value = float(row[0])
                        values.append(value)
                        break
            i += 1
            path = os.path.join(directory, f'execution_time_{dataset}-5-{i}tra.dat.log')
        data.append((alg, dataset, np.mean(values)))

print(data)
    