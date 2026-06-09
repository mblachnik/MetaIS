import os
import pandas as pd
from sklearn.model_selection import train_test_split
from research.basics.utils import loadConfig

num_splits = 5
config = loadConfig()
for dat in config["datasets"]:
    dataset_dir = os.path.join(config["test_data_dir"], dat)
    path = os.path.join(dataset_dir, f"{dat}.dat.csv")
    df = pd.read_csv(path)

    for i in range(1, num_splits + 1):
        # Podział 80% trening, 20% test
        train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=i)

        fileName = f"{dat}-{num_splits}-{i}tst.dat.csv"
        path = os.path.join(dataset_dir, fileName)
        test_df.to_csv(path, index=False)
        fileName = f"{dat}-{num_splits}-{i}tra.dat.csv"
        path = os.path.join(dataset_dir, fileName)
        # Zapis do plików
        train_df.to_csv(path, index=False)