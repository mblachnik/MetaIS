"""
Script to run MetaIS
note that first other two scripts must be executed that is it must have metamodel but in order to have meta model first we have to run meta attributes generation
"""

from joblib import Parallel, delayed
import pandas as pd
from  instance_selection.metais import MetaIS
import sklearn.neighbors as knn
import yaml
import os
import experiments.tools as tools
from tqdm import tqdm
#%%

config_file = "config.yaml"
if not os.path.isfile(config_file):
    config_file = "../config.yaml"
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

def applyFile(dir_name: str, dat_name: str, dat_ext: str, dat: str, threshold:float, is_model : str):
    print(f"Starting: \n"
          f"     Train:{os.path.join(dir_name,dat_name+dat_ext)} \n"
          f"     Train:{os.path.join(config['test_data_dir'],dat,dat_name.replace('tra', 'tst'))} \n"
          f"     Threshold:{threshold}"
          )
    X_train, y_train = tools.read_data(os.path.join(dir_name,
                                                        dat_name+dat_ext))
    X_test, y_test = tools.read_data(os.path.join(config["test_data_dir"],dat,
                                                    dat_name.replace("tra", "tst")))
    model_path = os.path.join(config["models_dir"], is_model, f"model_{dat}.dat_meta.pickl")
    model_meta = MetaIS(estimator_src=model_path, threshold=threshold)

    Xp_train,yp_train = model_meta.fit_resample(X_train, y_train)

    model_mis = knn.KNeighborsClassifier(n_neighbors=1)
    model_mis.fit(Xp_train, yp_train)
    X_test = X_test[Xp_train.columns]
    yp = model_mis.predict(X_test)
    res = tools.score(yp,y_test)
    res["name"] = dat
    res["threshold"] = threshold
    res = res | tools.scoreIS(X_train,Xp_train)
    print(f"Finished: \n"
          f"     Train:{os.path.join(dir_name,dat_name+dat_ext)} \n"
          f"     Train:{os.path.join(config['test_data_dir'],dat,dat_name.replace('tra', 'tst'))} \n"
          f"     Threshold:{threshold}"
          )
    return res

for model in config["models"]:
    files = [(r, f.replace(".csv",""), ".csv", next(c  for c in config['datasets'] if c in f))
             for r,ds,fs in os.walk(config["data_dir"] + model + "/")
                for f in fs
                    if f.endswith(".csv")
                        and any( True  if c in f else False for c in config['datasets'])
                        and (not any(s in f for s in ["_proto","_meta"]))
                        and ("-5-" in f)
                        and ("tra." in f)
             ]

#%%
    ress = []
    thresholds = config["treshholds"]
    n_jobs = config["n_jobs"]
    for threshold in thresholds:
        if n_jobs > 1:
            results = Parallel(n_jobs=n_jobs, prefer="threads", backend="loky")(delayed(applyFile)(dir_name, dat_name, dat_ext, dat, threshold, model) for dir_name, dat_name, dat_ext, dat in files)
            ress.extend(results)
        else:
            for dir_name, dat_name, dat_ext, dat in tqdm(files):
                ress.append(applyFile(dir_name, dat_name, dat_ext, dat, threshold, model))

    res_df = pd.DataFrame(ress)
    res_df.to_csv(os.path.join(config["results_dir"], model, "results_MetaIS_%s.csv" % config["result_postfix"]))
    perf = res_df.groupby(by=["name","threshold"]).aggregate(["mean","std"])
    perf.reset_index(inplace=True)
    perf.to_csv(os.path.join(config["results_dir"], model, "results_MetaIS_agg_%s.csv" % config["result_postfix"]))
    print(perf)
