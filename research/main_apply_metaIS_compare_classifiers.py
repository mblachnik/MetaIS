"""
Script to run MetaIS
note that first other two scripts must be executed that is it must have metamodel but in order to have meta model first we have to run meta attributes generation
"""

import time

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from instance_selection.metais import MetaIS
import sklearn.neighbors as knn
import os
import experiments.tools as tools
from tqdm import tqdm
from research.basics.utils import getBaseResultsFilePath, loadConfig
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from functools import partial


# %%

def applyFile(config: dict, dir_name: str, dat_name: str, dat_ext: str, dat: str, thresholds: list[float],
              is_model: str, model_factories: dict):
    print(f"Starting: \n"
          f"     Train:{os.path.join(dir_name, dat_name + dat_ext)} \n"
          f"     Train:{os.path.join(config['test_data_dir'], dat, dat_name.replace('tra', 'tst'))} \n"
          f"     Threshold:{thresholds}"
          )
    X_train, y_train = tools.read_data(os.path.join(dir_name,
                                                    dat_name + dat_ext))
    X_test, y_test = tools.read_data(os.path.join(config["test_data_dir"], dat,
                                                  dat_name.replace("tra", "tst")))
    model_path = os.path.join(config["models_dir"], is_model, f"model_{dat}.dat_meta.pickl")
    model_meta = MetaIS(estimator_src=model_path, threshold=thresholds[0], keep_proba=True)

    model_meta.fit_resample(X_train, y_train)
    res_all = []

    for threshold in thresholds:
        Xp_train, yp_train = model_meta.resample_with_new_threshold(X_train, y_train, threshold)
        Xp_train, yp_train = model_meta.fit_resample(X_train, y_train)

        n_samples_available = len(Xp_train)
        if n_samples_available == 0:
            continue
            
        unique_classes = np.unique(yp_train)
        if len(unique_classes) < 2:
            print(f"       => Ominięto threshold {threshold} dla {dat_name}: pozostała tylko 1 klasa ({unique_classes[0]})")
            continue

        is_metrics = tools.scoreIS(X_train, Xp_train)
        X_test = X_test[Xp_train.columns]

        for clf_name, builder in model_factories.items():
            try:
                model_mis = builder()

                # Pomiar czasu treningu / budowy indeksu klasyfikatora
                t_fit_start = time.perf_counter()
                model_mis.fit(Xp_train, yp_train)
                fit_time = time.perf_counter() - t_fit_start

                # Pomiar czasu predykcji (kluczowy zysk dla kNN)
                t_pred_start = time.perf_counter()
                yp = model_mis.predict(X_test)
                pred_time = time.perf_counter() - t_pred_start

                res = tools.score(yp, y_test)
                res["name"] = dat
                res["file"] = dat_name
                res["classifier"] = clf_name
                res["threshold"] = threshold
                res["fit_time"] = fit_time
                res["pred_time"] = pred_time
                res = res | is_metrics

                res_all.append(res)
            except ValueError as e:
                if "Expected n_neighbors <= n_samples_fit" in str(e):
                    print(f"[POMINIĘTO] Zbiór {dat_name} / {clf_name}: za mało próbek dla kNN ({e})")
                    continue  # lub zwracasz pusty wynik
                raise e  # Inne błędy ValueError przepuszczamy dalej

    print(f"Finished: \n"
          f"     Train:{os.path.join(dir_name, dat_name + dat_ext)} \n"
          f"     Test:{os.path.join(config['test_data_dir'], dat, dat_name.replace('tra', 'tst'))} \n"
          )
    # Store intermediate results
    res_all_df = pd.DataFrame(res_all)
    res_all_df.to_csv(getBaseResultsFilePath(config, is_model, f"_{dat_name}.csv"))
    return res_all


if __name__ == '__main__':
    config = loadConfig()

    model_factories = {
        "1NN": lambda: knn.KNeighborsClassifier(n_neighbors=1),
        "3NN": lambda: knn.KNeighborsClassifier(n_neighbors=3),
        "5NN": lambda: knn.KNeighborsClassifier(n_neighbors=5),
        "7NN": lambda: knn.KNeighborsClassifier(n_neighbors=7),
        "9NN": lambda: knn.KNeighborsClassifier(n_neighbors=9),
        # 2. Support Vector Machine (wymaga skalowania)
        "SVM_RBF": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0)),
        # 3. Modele drzewiaste i zespołowe
        "RandomForest": partial(RandomForestClassifier, n_estimators=100, random_state=42, n_jobs=1),
        "DecisionTree": partial(DecisionTreeClassifier, random_state=42),
        # 4. Model probabilistyczny
        "NaiveBayes": GaussianNB,
        # 5. Model liniowy (wymaga skalowania)
        "LogisticRegression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    }

    for model in config["models"]:
        files = [(r, f.replace(".csv", ""), ".csv", next(c for c in config['datasets'] if c in f))
                for r, ds, fs in os.walk(config["data_dir"] + model + "/")
                for f in fs
                if f.endswith(".csv")
                and any(True if c in f else False for c in config['datasets'])
                and (not any(s in f for s in ["_proto", "_meta"]))
                and ("-5-" in f)
                and ("tra." in f)
                ]

        # %%
        ress = []
        thresholds = config["treshholds"]
        n_jobs = config["n_jobs"]
        if n_jobs not in {1, 0}:
            results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(applyFile)(config, dir_name, dat_name, dat_ext, dat, thresholds, model, model_factories) for
                dir_name, dat_name, dat_ext, dat in files)
            ress += [item for res in results for item in res]  # Flatten results
        else:
            for dir_name, dat_name, dat_ext, dat in tqdm(files):
                print(f"{dir_name}      {dat_name}{dat_ext}")
                ress += applyFile(config, dir_name, dat_name, dat_ext, dat, thresholds, model, model_factories)

