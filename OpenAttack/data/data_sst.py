"""
:type: a tuple of three :py:class:`.Dataset` s, `(train, valid, test)`.
:Size: 1.116MB

SST dataset which is used to train victim models.
`[page] <https://nlp.stanford.edu/sentiment/>`__
"""
import pickle
import os
import datasets

NAME = "Dataset.SST"
DOWNLOAD = "https://cdn.data.thunlp.org/TAADToolbox/dataset/sst.pkl"


def LOAD(path):
    '''
    def mapping(data):
        return Dataset([
            DataInstance(
                x=it[0],
                y=it[1]
            ) for it in data
        ], copy=False)
    
    train, valid, test = pickle.load(open(path, "rb"))
    return mapping(train), mapping(valid), mapping(test)
    '''
    return datasets.load_dataset(os.path.join(os.getcwd(), "data", "sst_model.py"), data_files=path)
    #return datasets.load_dataset('sst')
