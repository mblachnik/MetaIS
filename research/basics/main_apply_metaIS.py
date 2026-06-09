from joblib import Parallel, delayed
import pandas as pd
from sklearn.base import ClassifierMixin
from instance_selection.metais import MetaIS
import os
import experiments.tools as tools
from tqdm import tqdm
import time

from research.basics.utils import getResultsFilePath, loadConfig

def applyFile(config: dict, dir_name: str, dat_name: str, dat_ext: str, dat: str, thresholds: list[float],
              is_model: str, classifier: ClassifierMixin):
    print(f"Starting: \n"
          f"     Train:{os.path.join(dir_name, dat_name + dat_ext)} \n"
          f"     Train:{os.path.join(config['test_data_dir'], dat, dat_name.replace('tra', 'tst'))} \n"
          f"     Threshold:{thresholds}"
          )
    X_train, y_train = tools.read_data(os.path.join(dir_name,
                                                    dat_name + dat_ext))
    X_test, y_test = tools.read_data(os.path.join(config["test_data_dir"], dat,
                                                  dat_name.replace("tra", "tst")))
    model_path = os.path.join(config["models_dir"], is_model, f"model_php89ntbG.dat_meta.pickl")
    model_meta = MetaIS(estimator_src=model_path, threshold=thresholds[0], keep_proba=True)

    res_all = []
    t1 = time.time()
    t1p = time.process_time()
    Xp_train, yp_train = model_meta.fit_resample(X_train, y_train)
    columns = Xp_train.columns
    model_mis = classifier
    model_mis.fit(Xp_train, yp_train)
    t2 = time.time()
    t2p = time.process_time()
    dt = t2 - t1
    dtp = t2p - t1p

    for threshold in thresholds:
        model_mis = classifier #klasyfikator należy klonować
        #Xp_train, yp_train = model_meta.fit_resample(X_train, y_train)
        Xp_train, yp_train = model_meta.resample_with_new_threshold(X_train, y_train, threshold)
        try:
            model_mis.fit(Xp_train, yp_train)
            X_test = X_test[columns]
            yp = model_mis.predict(X_test)
            res = tools.score(yp, y_test)
        except ValueError as e:
            res = {"acc": 0,
            "f1": 0,
            'bacc': 0,
            'mcc':0,
           }
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
    res_all_df = pd.DataFrame(res_all)
    res_all_df.to_csv(getResultsFilePath(config, is_model, False, True, f"_{dat_name}"))
    return res_all

def applyMetaIS(classifier: ClassifierMixin):
    config = loadConfig()
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

        ress = []
        thresholds = config["treshholds"]
        n_jobs = config["n_jobs"]
        if n_jobs not in {1, 0}:
            results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(applyFile)(config, dir_name, dat_name, dat_ext, dat, thresholds, model, classifier) for
                dir_name, dat_name, dat_ext, dat in files)
            ress += [item for res in results for item in res]  # Flatten results
        else:
            for dir_name, dat_name, dat_ext, dat in tqdm(files):
                print(f"{dir_name}      {dat_name}{dat_ext}")
                ress += applyFile(config, dir_name, dat_name, dat_ext, dat, thresholds, model, classifier)

        res_df = pd.DataFrame(ress)
        if res_df.shape[0]>0:
            path = getResultsFilePath(config, model, False, True)
            print('zapis do ' + path)
            res_df.to_csv(path)
            perf = res_df.groupby(by=["name", "threshold"]).aggregate(["mean", "std"])
            perf.reset_index(inplace=True)
            path = getResultsFilePath(config, model, True, True)
            print('zapis do ' + path)
            perf.to_csv(path)
            print(perf)
        
        
