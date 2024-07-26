#%%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from instance_selection.metais_tools import loadMetaFromDatasets, train, store, generateMetaForDatasets
import yaml
import os
doGenerateMeta = False #Jak true to generujemy metaatrybuty, jak False to pomijamy ten krok

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

#Tworzy listę toupli składającą się z katalogu, nazwy pliku i rozszerzenia przechodząc po podkatalogach katalogu data
#Spośród wybranych plików wybierane są te które zawierają  _meta i są ty[u .csv
files = [(r, f.replace(".csv",""), ".csv")
         for r,ds,fs in os.walk(config["data_dir"])
            for f in fs
                if f.endswith(".csv")
                    and ("_meta" in f)
                    and (not ("tra." in f)) and (not ("tst." in f)) #We train metamodel only for full datasets without split into training and testing
                    and any( True  if c in f else False for c in config['datasets'])
         ]
#%%
#Ładowanie plików i uruchomienie procesu uczenia meta-modelu
Xs, ys, fNames = loadMetaFromDatasets(files)
#%%
#The above function gets a list of X and y of metaattributes for given file, this allows us to combine only selected files and train metamodel only on selected files
for i,fN in enumerate(fNames): #We iterate over files so that for each file a separate dataset of metaattributes will be created, but from the full list of Xs and ys we have to droop current file becouse for that file we will perform experiments.
    print(f"Iteration {i+1} out of {len(fNames)} processing {fN}")
    Xx = []
    yy = []
    for Xt,yt,ft in zip(Xs,ys,fNames):
        if ft == fN: continue
        Xx.append(Xt)
        yy.append(yt)
    X = np.vstack(Xx) #Sklejenie części X
    y = np.hstack(yy) #Sklejenie części y
    model = train(X,y,model=RandomForestClassifier(n_jobs=5)) #Nauczenie meta modelu
    models_dir = config["models_dir"]
    store(model, file = f"{models_dir}/model_{fN}.pickl") #Zapisanie meta_modelu
