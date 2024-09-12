"""
A scirpt to generate meta features. It should be run first.
Note that it reads configuration from config.yaml so prepare that file before run
"""
#%%
from instance_selection.metais_tools import generateMetaForDatasets

import os
import yaml

#%%

config_file = "config.yaml"
if not os.path.isfile(config_file):
    config_file = "../config.yaml"
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

#Tworzy listę toupli składającą się z katalogu, nazwy pliku i rozszerzenia przechodząc po podkatalogach katalogu data
#Spośród wybranych plików wybierane są te które nie mają w nazwie _proto ani _meta
for model in config['models']:
    print(f"Processing for {model}")
    files = [(r, f.replace(".csv",""), ".csv")
             for r,ds,fs in os.walk(config["data_dir"]+model+"\\")
                for f in fs
                    if f.endswith(".csv")
                        and any( True  if c in f else False for c in config['datasets'])
                        and (not any(s in f for s in ["_proto","_meta"]))
                        and (
                               not (("-5-" in f) or ("-10-" in f))
             )]
    #%%
    generateMetaForDatasets(files,n_jobs=config["n_jobs"])