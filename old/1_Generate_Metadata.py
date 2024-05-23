import warnings
import pandas as pd 
import os
import numpy as np
import time
from metadata.calculatevalues import *
from metadata.configuration import *
from config.appsettings import Appsettings
from type_dictionaries import *
 
warnings.filterwarnings("ignore")

##--------------------------------------------------------
##Main start
##--------------------------------------------------------
        
start_time = time.time()

settings = Appsettings(RunMode.FoldsOnly, ProcessStep.GenerateMetadata)

for set_folder in ["kr-vs-k"]:

    
    #Prepare variables for folder paths
    print("Start folder {}".format(set_folder))

    paths = settings.get_paths(set_folder)

    files = [f for f in os.listdir(paths["source"]) if not f[0] == '.']

    for file in files:

        #Prepare variables for file paths
        file_without_suffix = file.removesuffix(".csv")
        meta_file_name = f"{file_without_suffix}{settings.meta_suffix}"
        
        if os.path.exists(os.path.join(paths["destination"], meta_file_name)):
            print(f"Metadata already generated: {file_without_suffix}")
            continue

        file_path = os.path.join(paths["source"], file)
        
        print("Start file: {}".format(file))
        # checking if it is a file
        if os.path.isfile(file_path):
            dataset = pd.read_csv( 
                filepath_or_buffer=file_path, 
                sep=",", 
                encoding="utf-8",
                ##header=None,
                index_col=0,
                decimal=".") 

            x_set, y_set = prepareData(dataset) 

            result_dictionary = dict()
            k_values = [3,5,9,15,23,33]


            number_of_elements = x_set.shape[0]

            for k in k_values:
                print("K: {}".format(k))
                if k >= number_of_elements: 
                    break
                neigh = prepareKNN(x_set, y_set,k)
                setSingleColumns(result_dictionary)
                setKColumns(k, result_dictionary)

                for index, row in x_set.iterrows():
                    print("Index: {}".format(index))
                    neighbors_ids, neighbors_distance = getNearestNeighbours(neigh, index, row)
                    neighbors_types = getNeighbourTypesById(y_set, neighbors_ids)
                    row_type = y_set[index]

                    oppositeClassCount(index, neighbors_types, result_dictionary, row_type, k)
                    sameClassCount(index, neighbors_types, result_dictionary, row_type, k)
                    meanDistanceFromAny(index, neighbors_distance, k, result_dictionary)
                    meanDistanceFromSame(neighbors_ids, neighbors_distance, neighbors_types, index, row_type, k, result_dictionary)
                    smallestDistanceSameClass(neighbors_ids, neighbors_distance, neighbors_types, index, row_type, result_dictionary)
                    smallestDistanceOppositeClass(neighbors_ids, neighbors_distance, neighbors_types, index, row_type, result_dictionary)
                    smallestDistanceAnyClass(neighbors_distance, index, result_dictionary)
                

            if  (any(val == -1 for val in result_dictionary['minDistanceOppositeClass'])) | (any(val==-1 for val in result_dictionary['minDistanceSameClass'])):
                
                opposite_class_indexes = [i for i, value in enumerate(result_dictionary['minDistanceOppositeClass']) if value == -1]
                same_class_indexes = [i for i, value in enumerate(result_dictionary['minDistanceSameClass']) if value == -1]
                
                count_entries = x_set.shape[0]-1
                perc10 = int(np.ceil(10*count_entries/100))
                perc25 = int(np.ceil(25*count_entries/100))
                perc50 = int(np.ceil(count_entries/2))

                k_values = [perc10, perc25, perc50, (count_entries-1)]
                for k in k_values:
                    neigh = prepareKNN(x_set, y_set, k)
                    print("K: {}".format(k))

                    for index in opposite_class_indexes:
                        neighbors_ids, neighbors_distance = getNearestNeighbours(neigh, index, x_set.loc[index])
                        neighbors_types = getNeighbourTypesById(y_set, neighbors_ids)
                        row_type = y_set[index]
                        
                        if any(val == -1 for val in result_dictionary['minDistanceOppositeClass']):
                            smallestDistanceOppositeClass(neighbors_ids, neighbors_distance, neighbors_types, index, row_type, result_dictionary)
                            
                    for index in same_class_indexes:
                        
                        neighbors_ids, neighbors_distance = getNearestNeighbours(neigh, index, x_set.loc[index])
                        neighbors_types = getNeighbourTypesById(y_set, neighbors_ids)
                        row_type = y_set[index]

                        v = {key: value for key, value in neighbors_types.items() if value == 27}
                        if any(val == -1 for val in result_dictionary['minDistanceSameClass']):
                            smallestDistanceSameClass(neighbors_ids, neighbors_distance, neighbors_types, index, row_type, result_dictionary)
                            
                    if (not any(val == -1 for val in result_dictionary['minDistanceOppositeClass'])) & (not any(val == -1 for val in result_dictionary['minDistanceSameClass'])):
                        break
                            

            result_dataframe = pd.DataFrame.from_dict(result_dictionary)
            result_dataframe.to_csv(path_or_buf=os.path.join(paths["destination"], meta_file_name), index=True, sep=";")
            
            print(f"End file: {file}")

    print("End folder: {}".format(set_folder))



elapsed_time = time.time() - start_time
print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
##Added comment to make commit
s = "test"