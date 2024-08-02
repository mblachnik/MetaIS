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

df_test = pd.read_csv("tests/data/fu.dat_meta_test.csv",sep=";")

test_data_len = len(df_test)

df_meta = pd.read_csv("tests/data/fu.dat_meta.csv",sep=";").head(test_data_len)

cols = df_test.columns

for index in range(test_data_len):
    for col in cols:
        generated_col = df_meta.iloc[index][col]
        test_col = df_test.iloc[index][col]
        if(abs(generated_col - test_col) > 0.00000000000001):
            print("Kolumna: ", col)
            print("Nr wiersza: ", index)
            print("Wartość oczekiwana", test_col)
            print("Wartość uzyskana", generated_col)
            sys.exit()

print("OK")
