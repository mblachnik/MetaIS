import numpy as np

def distanceFilteredBySameType(neighbors_ids, neighbors_distance, neighbors_types: dict, row_type):
    
    neighbors_dictionary = dict(zip(neighbors_ids, neighbors_distance))
    filtered_types_dict = {k : v for k, v in neighbors_types.items() if v == row_type}
    return [neighbors_dictionary[i] for i in filtered_types_dict.keys()]

def distanceFilteredByOppositeType(neighbors_ids, neighbors_distance, neighbors_types: dict, row_type):
    
    neighbors_dictionary = dict(zip(neighbors_ids, neighbors_distance))
    filtered_types_dict = {k : v for k, v in neighbors_types.items() if v != row_type}
    return [neighbors_dictionary[i] for i in filtered_types_dict.keys()]

def oppositeClassCount(index, neighbors_types: dict, result_dictionary, row_type, k): 

    counter = 0
    #Get type of neighbors 
    for neighbor_type in neighbors_types.values(): 
        if neighbor_type != row_type: 
            counter = counter + 1 
    result_dictionary[f"oppositeClassNeighbors{k}"].append(int(counter))


def sameClassCount(index, neighbors_types: dict, result_dictionary, row_type, k): 
    
    counter = 0
    #Get type of neighbors 
    for neighbor_type in neighbors_types.values(): 
        if neighbor_type == row_type: 
            counter = counter + 1 
    result_dictionary[f"sameClassNeighbors{k}"].append(int(counter))

     
def meanDistanceFromAny(index, neighbors_distance, k, result_dictionary):

    #Calculate mean distance
    mean = np.mean(neighbors_distance)
    result_dictionary[f"meanDistanceAny{k}"].append(mean)

def meanDistanceFromSame(neighbors_ids, neighbors_distance, neighbors_types: dict, index, row_type, k, result_dictionary):
    mean = np.nan

    if row_type not in neighbors_types.values():
        result_dictionary[f"meanDistanceSame{k}"].append(-1)
        return

    distance_list = distanceFilteredBySameType(neighbors_ids, neighbors_distance, neighbors_types, row_type)

    mean = np.mean(distance_list)
    result_dictionary[f"meanDistanceSame{k}"].append(mean)
    

def smallestDistanceSameClass(neighbors_ids, neighbors_distance, neighbors_types: dict, index, row_type, result_dictionary):

    distance_list = distanceFilteredBySameType(neighbors_ids, neighbors_distance, neighbors_types, row_type)

    if  index >= len(result_dictionary["minDistanceSameClass"]):
        
        if row_type not in neighbors_types.values():
             result_dictionary["minDistanceSameClass"].append(-1)
             return

        result_dictionary["minDistanceSameClass"].append(min(distance_list))
    else:
        if row_type in neighbors_types.values():
            result_dictionary["minDistanceSameClass"][index] = min(distance_list)
        
def smallestDistanceOppositeClass(neighbors_ids, neighbors_distance, neighbors_types: dict, index, row_type, result_dictionary):

    distance_list = distanceFilteredByOppositeType(neighbors_ids, neighbors_distance, neighbors_types, row_type)

    if index >= len(result_dictionary["minDistanceOppositeClass"]):
        
        if all(type_from_list == row_type for type_from_list in neighbors_types.values()):
             result_dictionary["minDistanceOppositeClass"].append(-1)
             return
        
        result_dictionary["minDistanceOppositeClass"].append(min(distance_list))
    else:
        if any(type_from_list != row_type for type_from_list in neighbors_types.values()):
            result_dictionary["minDistanceOppositeClass"][index] = min(distance_list)


def smallestDistanceAnyClass(neighbors_distance, index, result_dictionary):

    if index >= len(result_dictionary["minDistanceAnyClass"]):
        result_dictionary["minDistanceAnyClass"].append(neighbors_distance[0])
