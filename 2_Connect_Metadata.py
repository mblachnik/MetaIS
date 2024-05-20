import os
import pandas as pd 
import time
from Helpers.unify import *
from config.appsettings import Appsettings
from type_dictionaries import *

def ImportFileToDataset(path):
    dataset = pd.read_csv( 
            filepath_or_buffer=path, 
            sep=";", 
            encoding="utf-8",
            ##header=None,
            ##index_col=0,
            decimal=".") 
    return dataset

def ConnectData(metadata_set, weight_set):
    metadata_set["LABEL"] = 0
    for id_weight in weight_set['id']:
        metadata_set.at[id_weight-1, 'LABEL'] = 1
    metadata_set = metadata_set.drop(columns=["Unnamed: 0"])

##--------------------------------------------------------
##Main start
##--------------------------------------------------------

# 1) When 2 files, first with metadata, second with id and weights
# 2) When two files, first with meta data, second only with ids which supposed to have value 1
        
start_time = time.time()

settings = Appsettings(RunMode.FullSets, ProcessStep.ConnectMetaWithProto)

for set_folder in settings.sets:

    paths = settings.get_paths(set_folder)

    metadata_directory = paths["source_meta"]
    weight_directory = paths["source_weights"]
    result_directory = paths["destination"]

    #Prepare variables for folder paths
    print("Start folder {}".format(set_folder))

    if not os.path.exists(metadata_directory) or not os.listdir(metadata_directory):
        print(f"Metadata files are not created for this folder: {set_folder} folds: {settings.folds}")
        continue

    if not os.path.exists(weight_directory) or not os.listdir(weight_directory):
        print(f"Weight files are not created for this folder: {set_folder} folds: {settings.folds}")
        continue
    

    weight_files = [f for f in os.listdir(weight_directory) if  not f[0] == '.']

    for weight_file in weight_files:

        print("Start {}".format(weight_file))
        file_prefix = weight_file.removesuffix(".dat_proto.csv")
        metadata_file = file_prefix + settings.meta_suffix
        
        if os.path.exists(os.path.join(result_directory, f"{file_prefix}{settings.meta_weights_suffix}")):
           print("Meta weights already created")
           continue
                          
        if(not os.path.exists(os.path.join(weight_directory, weight_file)) or not os.path.exists(os.path.join(metadata_directory, metadata_file))):
           continue

        weight_set = ImportFileToDataset(os.path.join(weight_directory, weight_file))
        metadata_set = ImportFileToDataset(os.path.join(metadata_directory, metadata_file))

        is_same_length = len(weight_set) == len(metadata_set)
        has_only_one_column = len(weight_set.columns) == 1

        if is_same_length:
            print("Id and weight in weight set!")
            converted_weight_set = unify_weight_set(weight_set, os.path.join(weight_directory, weight_file))
            ConnectData(metadata_set, converted_weight_set)

            metadata_set.to_csv(path_or_buf=os.path.join(result_directory, f"{file_prefix}{settings.meta_weights_suffix}"), index=False, sep=";")    

        elif not is_same_length & has_only_one_column:
            print("Only positive values in weight set!")
            ConnectData(metadata_set, weight_set)
            
            metadata_set.to_csv(path_or_buf=os.path.join(result_directory, f"{file_prefix}{settings.meta_weights_suffix}"), index=False, sep=";")    

        else:
            raise("Datasets have different number of elements!")
        
        print("End {}".format(weight_file))

elapsed_time = time.time() - start_time
print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

s = "test"