import os
import pandas as pd
import sys
from instance_selection.metais_tools import generateMetaForDatasets

files = [(r, f.replace(".csv",""), ".csv")
         for r,ds,fs in os.walk('tests/data/')
            for f in fs
                if f.endswith(".csv")
                    and not any(s in f for s in ["_proto","_meta"])]

generateMetaForDatasets(files)

for dir, file, ext in files:
    df_X = pd.read_csv(dir + os.sep + file + ext,sep=";")
    df_meta = pd.read_csv(dir + os.sep  + file + "_meta"+ext,sep=";")
    ids1 = df_X.loc[:, "id"] #Pobieramy oryginalne ID
    ids2 = df_meta.loc[:, "id"] #Pobieramy ID wygenerowane w meta
    ids1_len = len(ids1)

    if ids1_len != len(ids2):
        print('Porażka: niespójna wielkość danych')
        sys.exit()

    for i in range(ids1_len):
        id1 = ids1[i]
        id2 = ids2[i]
        if ids1[i] != ids2[i]:
            print(f'Porażka: w rekordzie o indeksie {i} oryginalne id to {ids1[i]} a wygenerowane w metadanych to {ids2[i]}')
            sys.exit()

print('Test zakończony pomyślnie')
