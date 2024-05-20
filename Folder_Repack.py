import os
import re

from config.appsettings import Appsettings
from type_dictionaries import *

#Config
settings = Appsettings(RunMode.Other, ProcessStep.Other)

type = "dat.csv"
run = "clean"
result_folders = [f"{settings.folds}_tra", f"{settings.folds}_tst", "fullset", f"{settings.folds}_tra_proto", "fullset_proto"]

for set_folder in settings.sets:

    # source = os.path.join(settings.original_directory, set_folder)
    # destination = os.path.join(settings.original_directory, set_folder, result_folders[2])

    source = os.path.join(settings.weights_directory, set_folder)
    destination = os.path.join(settings.weights_directory, set_folder, result_folders[4])

    if not os.path.isdir(source):
        continue
    
    print("Start folder {}".format(set_folder))

    #pattern = f'(.+\w+)(-{settings.folds}-)(\d+)({type})'
    pattern = f'.*\{type}$'

    if not os.path.exists(destination) and run == 'repack':
        os.makedirs(destination)

    files = [f for f in os.listdir(source) if (not f[0] == '.' and f.endswith(type))]
   
    for file in files:
        match_name_result = re.match(pattern, file)

        if (match_name_result and run == "repack"):
            os.rename(os.path.join(source, file), os.path.join(destination, file))
        elif(not match_name_result and run == "clean"):
            os.remove(os.path.join(source, file))


        