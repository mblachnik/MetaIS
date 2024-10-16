"""
Script to run MetaIS
note that first other two scripts must be executed that is it must have metamodel but in order to have meta model first we have to run meta attributes generation
"""

from joblib import Parallel, delayed
import pandas as pd
from  instance_selection.metais import MetaIS
import sklearn.neighbors as knn
import os
import experiments.tools as tools
from tqdm import tqdm
import time

from research.basics.utils import getResultsFilePath, loadConfig
#%%

config = loadConfig()

def applyFile(dir_name: str, dat_name: str, dat_ext: str, dat: str, threshold:float):
    print(f"Starting: \n"
          f"     Train:{os.path.join(dir_name,dat_name+dat_ext)} \n"
          f"     Train:{os.path.join(config['test_data_dir'],dat,dat_name.replace('tra', 'tst'))} \n"
          f"     Threshold:{threshold}"
          )
    X_train, y_train = tools.read_data(os.path.join(dir_name,
                                                        dat_name+dat_ext))
    X_test, y_test = tools.read_data(os.path.join(config["test_data_dir"],dat,
                                                    dat_name.replace("tra", "tst")))
    model_paths = []
    for model in config["models"]:
        model_paths.append(os.path.join(config["models_dir"], model, f"model_{dat}.dat_meta.pickl"))
    model_meta = MetaIS(estimator_src=modemain_apply_metaIS.pyl_paths, threshold=threshold)

    t1 = time.time()
    t1p = time.process_time()
    Xp_train,yp_train = model_meta.fit_resample(X_train, y_train)
    t2 = time.time()
    t2p = time.process_time()
    dt = t2-t1
    dtp = t2p-t1p
    model_mis = knn.KNeighborsClassifier(n_neighbors=1)
    model_mis.fit(Xp_train, yp_train)
    X_test = X_test[Xp_train.columns]
    yp = model_mis.predict(X_test)
    res = tools.score(yp,y_test)
    res["name"] = dat
    res["threshold"] = threshold
    res = res | tools.scoreIS(X_train,Xp_train)
    res = res | {'time':dt, 'process_time':dtp}
    print(f"Finished: \n"
          f"     Train:{os.path.join(dir_name,dat_name+dat_ext)} \n"
          f"     Train:{os.path.join(config['test_data_dir'],dat,dat_name.replace('tra', 'tst'))} \n"
          f"     Threshold:{threshold}"
          )
    return res

model = config["models"][0]
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
        results = Parallel(n_jobs=n_jobs, prefer="threads", backend="loky")(delayed(applyFile)(dir_name, dat_name, dat_ext, dat, threshold) for dir_name, dat_name, dat_ext, dat in files)
        ress.extend(results)
    else:
        for dir_name, dat_name, dat_ext, dat in tqdm(files):
            ress.append(applyFile(dir_name, dat_name, dat_ext, dat, threshold))

res_df = pd.DataFrame(ress)
res_df.to_csv(getResultsFilePath(config, "multimodel", False, True))
perf = res_df.groupby(by=["name","threshold"]).aggregate(["mean","std"])
perf.reset_index(inplace=True)
perf.to_csv(getResultsFilePath(config, "multimodel", True, True))
print(perf)
