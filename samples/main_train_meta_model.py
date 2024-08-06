#%%
import pandas as pd
import numpy as np
from instance_selection.metais_tools import loadMetaFromDatasets, train, store, generateMetaForDatasets

import os
doGenerateMeta = False #Jak true to generujemy metaatrybuty, jak False to pomijamy ten krok

#Tworzy listę toupli składającą się z katalogu, nazwy pliku i rozszerzenia przechodząc po podkatalogach katalogu data
#Spośród wybranych plików wybierane są te które zawierają  _meta i są ty[u .csv
files = [(r, f.replace(".csv",""), ".csv")
         for r,ds,fs in os.walk('data/')
            for f in fs
                if f.endswith(".csv")
                    and ("_meta" in f)]

#Ładowanie plików i uruchomienie procesu uczenia meta-modelu
Xs, ys, fNamess = loadMetaFromDatasets(files)

X = np.vstack(Xs) #Sklejenie części X
y = np.hstack(ys) #Sklejenie części y
model = train(X,y) #Nauczenie meta modelu
store(model, file = "models/model.pickl") #Zapisanie meta_modelu
