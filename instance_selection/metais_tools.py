import pickle
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from instance_selection.metais import ISMetaAttributesTransformer


def train(X,y, model=None):
    """
    Simply train given model is model not set then RandomForest is used
    :param X:
    :param y:
    :param model:
    :return:
    """

    if model is None:
        model = RandomForestClassifier(n_jobs=5)
    model.fit(X,y)
    return model

def store(model,file = "/models/model.pickl"):
    """
    Write model to pickle file
    :param model:
    :param file:
    :return:
    """
    with open(file, 'wb') as f:
        pickle.dump(model, f)

def generateMetaForDatasets(files : list, dropColumns: list = ["LABEL","id"], doSave:bool = True, return_meta=False):
    """
    Function takes input list of files and for each file it generates metaattributes. The elements of the list should be
    a tuple (directory_name, file_name, file_extension),
    The results are stored in a new file if doSave is True, if return_meta is True the function returns a list of tuples
    (file, X_metaatributes, y_proto), where file is the file name, X_metaatributes is a dataframe with metaatributes,
    and y_proto is a column containing labels indictating whether given instance selection algorithm indicated given
    instance as selected or not.
    :param files: list of files
    :param dropColumns: list of columns to drop - these are columns with original labels, row id's etc
    :param doSave: bool indicating wheter the results for each element of the list should be stored in a file or not.
    By defult doSave is True
    :param return_meta: bool indicating the output should be returned or not. If True it returnes a list of tuples
    :return: empty list or a list of typles containing file name, metaatributes and labels
    """
    metaTransformer = ISMetaAttributesTransformer()
    out = []
    for dir, file, ext in files:
        dfX = pd.read_csv(dir + os.sep + file + ext,sep=";")
        dfY = pd.read_csv(dir + os.sep  + file + "_proto"+ext,sep=";")
        dfY.rename(columns={"weight":"_weight_"}, inplace=True)
        df = pd.merge(dfX, dfY, on='id', how='outer')
        if df.isnull().any().any():
            raise ValueError(f"In {file} after mergeing with IS weights one of values os NAN but it shouldnt. \n"
                             f"It is likely that {file} and {file}_proto do not match")
        dropColumns.append("_weight_")
        X = df.loc[:, [c for c in dfX.columns if c not in dropColumns]]
        y = df.loc[:, "_weight_"]
        ids = df.loc[:, "id"] #Pobieramy oryginalne ID
        #X_meta = metaTransformer.fit_transform(X,y)
        X_meta = metaTransformer.transform(X,y,ids)
        if doSave:
            df_toSave = pd.concat([X_meta,y],axis=1)
            df_toSave.to_csv(dir + os.sep  + file + "_meta" + ext,index=False, sep=";")
        if return_meta:
            out.append((file,X,y))
    return out


def loadMetaFromDatasets(files : list, dropColumns: list = ["LABEL","id"], doSave:bool = True, return_meta=False):
    """
    Function takes input list of files and for each file it loads metaattributes, each meta_file is then loaded and its
    values are deliver to the output as a topule of lists where the first element of a touple contains list of X part of the meta attributes,
     the second element is a list of Y's of the meta attributes, and the third part is a list of file names.
    ( list[X], list[y], list[file_names])
    :param files: list of files
    :param dropColumns: list of columns to drop -
    :return: a typle of lists containing loaded data ready to be merged
    """
    outX = []
    outY = []
    outFile = []
    for dir, file, ext in files:
        df = pd.read_csv(dir + os.sep  + file + ext, sep=";")
        if df.isnull().any().any():
            raise ValueError(f"In {file} after mergeing with IS weights one of values os NAN but it shouldnt. \n"
                             f"It is likely that {file} and {file}_proto do not match")
        dropColumns.append("_weight_")
        X = df.loc[:, [c for c in df.columns if c not in dropColumns]].values
        y = df.loc[:, "_weight_"].values
        outX.append(X)
        outY.append(y)
        outFile.append(file)
    return outX, outY, outFile

