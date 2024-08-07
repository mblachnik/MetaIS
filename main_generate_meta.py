#%%
from instance_selection.metais_tools import generateMetaForDatasets
import os
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

#Tworzy listę toupli składającą się z katalogu, nazwy pliku i rozszerzenia przechodząc po podkatalogach katalogu data
#Spośród wybranych plików wybierane są te które nie mają w nazwie _proto ani _meta
files = [(r, f.replace(".csv",""), ".csv")
         for r,ds,fs in os.walk('data/')
            for f in fs
                if f.endswith(".csv")
                    and not any(s in f for s in ["_proto","_meta"])]

generateMetaForDatasets(files, config["n_jobs"])    