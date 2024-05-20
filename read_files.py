import pandas as pd

def ImportFileToDataset(path):
    dataset = pd.read_csv( 
            filepath_or_buffer=path, 
            sep=";", 
            encoding="utf-8",
            ##header=None,
            index_col=0,
            decimal=".") 

    return dataset

def ImportFileToDatasetWithCommas(path, index = True):
    if index:
        dataset = pd.read_csv( 
                filepath_or_buffer=path, 
                sep=",", 
                encoding="utf-8",
                ##header=None,
                index_col=0,
                decimal=".")
    else:
        dataset = pd.read_csv( 
        filepath_or_buffer=path, 
        sep=",", 
        encoding="utf-8",
        ##header=None,
        ##index_col=0,
        decimal=".")

    return dataset

def PrepareData(dataframe): 
    x = dataframe.iloc[:,:-1]
    y = dataframe.iloc[:,-1]

    x = x.copy(deep=True)

    return x,y 
