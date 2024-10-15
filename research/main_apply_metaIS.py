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
from multiprocessing import freeze_support

from research.basics.utils import getResultsFilePath, loadConfig


# %%

def applyFile(config: dict, dir_name: str, dat_name: str, dat_ext: str, dat: str, thresholds: list[float],
              is_model: str):
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
        print(f"       => {dat_name}{dat_ext} Threshold:{threshold}")
        res_all.append(res)

    print(f"Finished: \n"
          f"     Train:{os.path.join(dir_name, dat_name + dat_ext)} \n"
          f"     Test:{os.path.join(config['test_data_dir'], dat, dat_name.replace('tra', 'tst'))} \n"
          )
    # Store intermediate results
    res_all_df = pd.DataFrame(res_all)
    res_all_df.to_csv(getResultsFilePath(config, is_model, False, True, f"_{dat_name}"))
    return res_all


if __name__ == '__main__':
    done = { #List of files that should be ommited during calculations, for example because we already have results for them.The list is specific per model - thats why its a dict
        "CCIS":[    "codrnaNorm-5-1tra.dat.csv",
                        "codrnaNorm-5-2tra.dat.csv",
                        "codrnaNorm-5-3tra.dat.csv",
                        "codrnaNorm-5-4tra.dat.csv",
                        "codrnaNorm-5-5tra.dat.csv",
                        "covtype-5-1tra.dat.csv",
                        "covtype-5-2tra.dat.csv",
                        "covtype-5-3tra.dat.csv"
                        "covtype-5-5tra.dat.csv",
                        "php89ntbG-5-1tra.dat.csv",
                        "php89ntbG-5-2tra.dat.csv",
                        "php89ntbG-5-3tra.dat.csv",
                        "php89ntbG-5-4tra.dat.csv",
                        "php89ntbG-5-5tra.dat.csv"
                        ],
            "ENN":["codrnaNorm-5-1tra.dat.csv",
                   "codrnaNorm-5-2tra.dat.csv",
                   "codrnaNorm-5-3tra.dat.csv",
                   "codrnaNorm-5-4tra.dat.csv",
                   "codrnaNorm-5-5tra.dat.csv",
                   "covtype-5-1tra.dat.csv",
                   "covtype-5-2tra.dat.csv",
                   "covtype-5-3tra.dat.csv",
                   "covtype-5-4tra.dat.csv",
                   "covtype-5-5tra.dat.csv",
                   "php89ntbG-5-1tra.dat.csv",
                   "php89ntbG-5-2tra.dat.csv",
                   "php89ntbG-5-3tra.dat.csv",
                   "php89ntbG-5-4tra.dat.csv",
                   "php89ntbG-5-5tra.dat.csv",],
            "Drop3Keel":[],
            "ICFKeel":["codrnaNorm-5-1tra.dat.csv",
                   "codrnaNorm-5-2tra.dat.csv",
                   "codrnaNorm-5-3tra.dat.csv",
                   "codrnaNorm-5-4tra.dat.csv",
                   "codrnaNorm-5-5tra.dat.csv",
                   "covtype-5-1tra.dat.csv",
                   "covtype-5-2tra.dat.csv",
                   "covtype-5-4tra.dat.csv",
                   "covtype-5-5tra.dat.csv",
                   "php89ntbG-5-1tra.dat.csv",
                   "php89ntbG-5-2tra.dat.csv",
                   "php89ntbG-5-3tra.dat.csv",
                   "php89ntbG-5-4tra.dat.csv",
                   "php89ntbG-5-5tra.dat.csv"],
            "HMEI":["codrnaNorm-5-1tra.dat.csv",
                    "codrnaNorm-5-2tra.dat.csv",
                    "codrnaNorm-5-3tra.dat.csv",
                    "codrnaNorm-5-4tra.dat.csv",
                    "codrnaNorm-5-5tra.dat.csv",
                    "covtype-5-1tra.dat.csv",
                    "covtype-5-2tra.dat.csv",
                    "covtype-5-3tra.dat.csv"
                    "covtype-5-4tra.dat.csv",
                    "covtype-5-5tra.dat.csv",
                    "php89ntbG-5-1tra.dat.csv",
                    "php89ntbG-5-2tra.dat.csv",
                    "php89ntbG-5-3tra.dat.csv",
                    "php89ntbG-5-4tra.dat.csv",
                    "php89ntbG-5-5tra.dat.csv"],
            'multimodel':[],
            "1NN":[]
    }
    config = loadConfig()
    t_start = time.time()
    for model in config["models"]:
        files = [(r, f.replace(".csv", ""), ".csv", next(c for c in config['datasets'] if c in f))
                 for r, ds, fs in os.walk(config["data_dir"] + model + "/")
                 for f in fs
                 if f.endswith(".csv")
                 and any(True if c in f else False for c in config['datasets'])
                 and (not any(s in f for s in ["_proto", "_meta"]))
                 and ("-5-" in f)
                 and ("tra." in f)
                 and (f not in done.get(model))
                 ]

        # %%
        ress = []
        thresholds = config["treshholds"]
        n_jobs = config["n_jobs"]
        if n_jobs not in {1, 0}:
            results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(applyFile)(config, dir_name, dat_name, dat_ext, dat, thresholds, model) for
                dir_name, dat_name, dat_ext, dat in files)
            ress += [item for res in results for item in res]  # Flatten results
        else:
            for dir_name, dat_name, dat_ext, dat in tqdm(files):
                print(f"{dir_name}      {dat_name}{dat_ext}")
                ress += applyFile(config, dir_name, dat_name, dat_ext, dat, thresholds, model)

        res_df = pd.DataFrame(ress)
        if res_df.shape[0]>0:
            res_df.to_csv(getResultsFilePath(config, model, False, True))
            perf = res_df.groupby(by=["name", "threshold"]).aggregate(["mean", "std"])
            perf.reset_index(inplace=True)
            perf.to_csv(getResultsFilePath(config, model, True, True))
            print(perf)
    print(time.time() - t_start)
