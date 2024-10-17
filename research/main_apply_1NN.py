"""
A script to perform simple 1NN prediction without any instance selection.
Results are sotred the same way as with metaIS
"""

import pandas as pd
import sklearn.neighbors as knn
import os
import experiments.tools as tools

from research.basics.utils import loadConfig
#%%

config = loadConfig()

for model in config["models"]:
    print(f"Model: {model}")
    files = [(r, f.replace(".csv",""), ".csv", next(c  for c in config['datasets'] if c in f))
             for r,ds,fs in os.walk(config["data_dir"]+model)
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
        if model not in ["1NN"]:
            Xp_train, yp_train, stats = tools.read_data_and_IS(os.path.join(dir_name,dat_name+dat_ext),
                                                               os.path.join(dir_name,dat_name  + dat_ext))
        else:
            Xp_train, yp_train = tools.read_data(os.path.join(dir_name, dat_name + dat_ext))
            stats = tools.scoreIS(Xp_train, Xp_train)

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
    post_fix="_v6"
    res_df.to_csv(os.path.join(config["results_dir"]+model,f"results_IS{post_fix}.csv"))
    perf = res_df.groupby("name").aggregate(["mean","std"])
    perf.to_csv(os.path.join(config["results_dir"]+model,f"results_IS_agg{post_fix}.csv"))
    print(perf)

