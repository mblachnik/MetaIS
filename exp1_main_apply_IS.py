import pandas as pd
from  instance_selection.metais import MetaIS
import sklearn.neighbors as knn
import yaml
import os
import experiments.tools as tools
from tqdm import tqdm
#%%

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

for i,(dir_name, dat_name, dat_ext, dat) in  enumerate(files):
    print(f"{i}/{len(files)}")
    Xp_train, yp_train, stats = tools.read_data_and_IS(os.path.join(dir_name,
                                                    dat_name+dat_ext),
                                              os.path.join(dir_name,
                                                           dat_name + "_proto" + dat_ext)
                                              )
    X_test, y_test   = tools.read_data(os.path.join(config["test_data_dir"],dat,
                                                    dat_name.replace("tra", "tst")))

    model_mis = knn.KNeighborsClassifier(n_neighbors=1)
    model_mis.fit(Xp_train, yp_train)
    X_test = X_test[Xp_train.columns]
    yp = model_mis.predict(X_test)
    res = tools.score(yp,y_test)
    res["name"] = dat
    res = res | stats
    ress.append(res)

res_df = pd.DataFrame(ress)
res_df.to_csv(os.path.join(config["results_dir"],"results_IS_2.csv"))
perf = res_df.groupby("name").aggregate(["mean","std"])
perf.to_csv(os.path.join(config["results_dir"],"results_IS_agg_2.csv"))
print(perf)

