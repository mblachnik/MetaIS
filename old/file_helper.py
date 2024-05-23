import pandas as pd
import os

from config.appsettings import Appsettings
from type_dictionaries import *


def DatToCsv(path):
    dataset = pd.read_csv(path, sep=';', header=0, decimal='.', encoding='utf-8', index_col=-1)
    return dataset

settings = Appsettings(RunMode.Other, ProcessStep.Other)
#set_folders = ["abalone"]

for set_folder in settings.sets:
    path = os.path.join(settings.original_directory, set_folder)
    if not os.path.exists(path):
        continue
    
    files = [f for f in os.listdir(path) if (not f[0] == '.' and not f.endswith("csv"))]

    for file in files:
        csv = DatToCsv(os.path.join(path, file))

        file_prefix = file.removesuffix(".dat")

        csv_file = f"{file_prefix}.csv"
        csv.to_csv(path_or_buf=os.path.join(path, csv_file), index_label='id')

    for file in files:
        os.remove(os.path.join(path, file))
    
"""   result_folder = os.path.join(directory, '_fullcsv')
    
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    csv.to_csv(os.path.join(result_folder, f'{set_folder}.dat.csv'), index=False, sep=";") """

