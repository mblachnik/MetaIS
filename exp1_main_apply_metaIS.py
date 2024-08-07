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

def applyFile(dir_name: str, dat_name: str, dat_ext: str, dat: str):
    X_train, y_train = tools.read_data(os.path.join(dir_name,
                                                    dat_name+dat_ext))
    X_test, y_test   = tools.read_data(os.path.join(config["test_data_dir"],dat,
                                                    dat_name.replace("tra", "tst")))

    model_path = os.path.join(config["models_dir"], f"model_{dat}.dat_meta.pickl")
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
    return res

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


files = [(r, f.replace(".csv",""), ".csv", next(c  for c in config['datasets'] if c in f))
         for r,ds,fs in os.walk(config["data_dir"])
            for f in fs
                if f.endswith(".csv")
                    and any( True  if c in f else False for c in config['datasets'])
                    and (not any(s in f for s in ["_proto","_meta"]))
                    and ("-5-" in f)
                    and ("tra." in f)
         ]

#%%
ress = []
thresholds = [0.3,0.5,0.7]
n_jobs = config["n_jobs"]

for threshold in thresholds:
    if n_jobs > 1:
        results = Parallel(n_jobs=n_jobs, prefer="threads", backend="loky")(delayed(applyFile)(dir_name, dat_name, dat_ext, dat) for dir_name, dat_name, dat_ext, dat in tqdm(files))
        ress.extend(results)
    else:
        for dir_name, dat_name, dat_ext, dat in tqdm(files):
            ress.append(applyFile(dir_name, dat_name, dat_ext, dat))
        
res_df = pd.DataFrame(ress)
res_df.to_csv(os.path.join(config["results_dir"],"results_MetaIS_v2.csv"))
perf = res_df.groupby("name").aggregate(["mean","std"])
perf.to_csv(os.path.join(config["results_dir"],"results_MetaIS_agg_v2.csv"))
print(perf)