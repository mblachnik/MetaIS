import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

results_files = ["results_MetaIS_agg_v2.csv","results_MetaIS_agg.csv"]
dfs = []
for file in results_files:
    df = pd.read_csv(os.path.join(config["results_dir"],file))
    dfs.append(df)
df = pd.merge(dfs,axis=0)



