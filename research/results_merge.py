import shutil
import pandas as pd
import os
from research.basics.utils import getResultsFilePath, loadConfig

config = loadConfig()
for model in config['models']:
    katalog = os.path.join(config["results_dir"], model)
    merged_res_path = getResultsFilePath(config, model, False, True)
    merged_res_path_agg = getResultsFilePath(config, model, True, True)
    katalog_merged = os.path.join(katalog, 'merged')

    list_df = []

    if os.path.exists(merged_res_path):
        list_df.append(pd.read_csv(merged_res_path))

    merged_files = []


    for plik in os.listdir(katalog):
        if plik.endswith('.dat.csv'):
            pelna_sciezka = os.path.join(katalog, plik)
            merged_files.append(plik)
            list_df.append(pd.read_csv(pelna_sciezka))

    if(len(merged_files) > 0):
        df = pd.concat(list_df, ignore_index=True)

        df.to_csv(merged_res_path)
        perf = df.groupby("name").aggregate(["mean","std"])
        perf.to_csv(merged_res_path_agg)

        if not os.path.exists(katalog_merged):
            os.makedirs(katalog_merged)
        for plik in merged_files:
            shutil.move(os.path.join(katalog, plik), os.path.join(katalog_merged, plik))