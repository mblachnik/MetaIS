"""
Script to run MetaIS
note that first other two scripts must be executed that is it must have metamodel but in order to have meta model first we have to run meta attributes generation
"""

from joblib import Parallel, delayed
import pandas as pd
from instance_selection.metais import MetaIS
import sklearn.neighbors as knn
import os
import experiments.tools as tools
from tqdm import tqdm
import time
from research.basics.utils import getResultsFilePath, loadConfig
# %%

def applyFiles(group_files):
    X_train, y_train, X_test, y_test = None, None, None, None
    dir_name, dat_name, dat_ext, dat = None, None, None, None
    for dir_name, dat_name, dat_ext, dat in tqdm(group_files):
        X_train_tmp, y_train_tmp = tools.read_data(os.path.join(dir_name,
                                                        dat_name + dat_ext))
        X_test_tmp, y_test_tmp = tools.read_data(os.path.join(config["test_data_dir"], dat,
                                                    dat_name.replace("tra", "tst")))
        if X_train is not None:
            X_train = X_train_tmp
            y_train = y_train_tmp
            X_test = X_test_tmp
            y_test = y_test_tmp
        else:
            X_train = pd.concat([X_train, X_train_tmp], ignore_index=True)
            y_train = pd.concat([y_train, y_train_tmp], ignore_index=True)
            X_test = pd.concat([X_test, X_test_tmp], ignore_index=True)
            y_test = pd.concat([y_test, y_test_tmp], ignore_index=True)
        model_path = os.path.join(config["models_dir"], multimodel, f"model_{dat}.dat_meta.pickl")
    return applyFile(model_path, X_train, y_train, X_test, y_test, thresholds, multimodel, dir_name, dat_name, dat_ext, dat)

def applyFile(model_path, X_train, y_train, X_test, y_test, thresholds: list[float],
              is_model: str, dir_name: str, dat_name: str, dat_ext: str, dat: str):
    model_meta = MetaIS(estimator_src=model_path, threshold=thresholds[0], keep_proba=True)

    res_all = []
    t1 = time.time()
    t1p = time.process_time()
    Xp_train, yp_train = model_meta.fit_resample(X_train, y_train)
    model_mis = knn.KNeighborsClassifier(n_neighbors=1)
    model_mis.fit(Xp_train, yp_train)
    t2 = time.time()
    t2p = time.process_time()
    dt = t2 - t1
    dtp = t2p - t1p

    for threshold in thresholds:
        model_mis = knn.KNeighborsClassifier(n_neighbors=1)
        model_meta.resample_with_new_threshold(X_train, y_train, threshold)
        Xp_train, yp_train = model_meta.fit_resample(X_train, y_train)
        model_mis.fit(Xp_train, yp_train)
        X_test = X_test[Xp_train.columns]
        yp = model_mis.predict(X_test)
        res = tools.score(yp, y_test)
        res["name"] = dat
        res["threshold"] = threshold
        res = res | tools.scoreIS(X_train, Xp_train)
        res = res | {'time': dt, 'process_time': dtp}
        print(f"       => {dat} Threshold:{threshold}")
        res_all.append(res)

    print(f"Finished: \n"
          f"     Train:{os.path.join(dir_name, dat_name + dat_ext)} \n"
          f"     Test:{os.path.join(config['test_data_dir'], dat, dat_name.replace('tra', 'tst'))} \n"
          )
    # Store intermediate results
    res_all_df = pd.DataFrame(res_all)
    res_all_df.to_csv(getResultsFilePath(config, is_model, False, True, f"_{dat_name}"))
    return res_all

config = loadConfig()
t_start = time.time()

files = {}
for model in config["models"]:
    for r, ds, fs in os.walk(config["data_dir"] + model + "/"):
        for f in fs:
            if (
                f.endswith(".csv") and
                any(c in f for c in config['datasets']) and
                not any(s in f for s in ["_proto", "_meta"]) and
                "-5-" in f and
                "tra." in f
            ):
                # Klucz: nazwa pliku bez .csv
                key = f.replace(".csv", "")
                
                # Wpis (ścieżka, nazwa, rozszerzenie, dataset)
                file_entry = (r, key, ".csv", next(c for c in config['datasets'] if c in f))
                
                # Dodajemy do grupy (jeśli nie istnieje, tworzymy nową listę)
                if key not in files:
                    files[key] = []
                files[key].append(file_entry)

ress = []

thresholds = config["treshholds"]
n_jobs = config["n_jobs"]
multimodel = "_".join(config["models"])
if n_jobs not in {1, 0}:
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(applyFiles)(group_files) for
        group_name, group_files in files.items())
    ress += [item for res in results for item in res]  # Flatten results
else:
    for group_name, group_files in files.items():
        ress += applyFiles(group_files)

res_df = pd.DataFrame(ress)
if res_df.shape[0]>0:
    res_df.to_csv(getResultsFilePath(config, multimodel, False, True))
    perf = res_df.groupby(by=["name", "threshold"]).aggregate(["mean", "std"])
    perf.reset_index(inplace=True)
    perf.to_csv(getResultsFilePath(config, multimodel, True, True))
    print(perf)
print(time.time() - t_start)
