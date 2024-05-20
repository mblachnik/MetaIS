from sklearn.neighbors import NearestNeighbors 
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def prepareData(dataframe): 
    x = dataframe.iloc[:,:-1]
    y = dataframe.iloc[:,-1]

    y = LabelEncoder().fit_transform(y) 

    #x = pd.DataFrame.from_records(x) 
    x = x.copy(deep=True)
    x.index = x.index - 1

    return x,y 


def prepareKNN(x, y, k):
    neigh = NearestNeighbors(n_neighbors=k+1, metric="euclidean") 
    neigh.fit(x,y)
    
    return neigh

def getNearestNeighbours(neigh, index, row):
    neighbors = neigh.kneighbors([row], return_distance=True)
    neighbors_ids = list(neighbors[1][0])
    neighbors_distance = list(neighbors[0][0])
    arguments_count = row.shape[0]

    if index in neighbors_ids:
        neighbors_ids.remove(index)
        neighbors_distance.pop(0)
    else:
        neighbors_ids.pop(-1)
        neighbors_distance.pop(-1)

    ##normalize distances because of numberr of columns
    neighbors_distance = [(d**2)/arguments_count for d in neighbors_distance]

    return neighbors_ids, neighbors_distance


def setKColumns(k, result_dictionary):
    new_columns = ["oppositeClassNeighbors", "sameClassNeighbors", "meanDistanceAny", "meanDistanceSame"]
    for column in new_columns:
        column = column + str(k)
        result_dictionary[column] = []

def setSingleColumns(result_dictionary):
    new_columns = ["minDistanceSameClass", "minDistanceAnyClass", "minDistanceOppositeClass"]
    for column in new_columns:
        result_dictionary[column] = []

def getNeighbourTypesById(y, neighbors_ids):
    result_dict = dict()
    for id in neighbors_ids:
        result_dict[id] = y[id]

    return result_dict

